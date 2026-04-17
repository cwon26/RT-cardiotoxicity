% =========================================================================
% RT Beam + Heart + LV Dose Overlay Visualization (v2)
% =========================================================================
% - Cylindrical beam tube visualization
% - Correct IEC→DICOM gantry angle conversion
% - OBJ LV registered to patient coordinates for animation
% - Whole heart + LV dose overlay + beating animation
%
% Required Toolboxes:
%   - Computer Vision Toolbox (pcregistercpd, pcregistericp)
%   - Statistics and Machine Learning Toolbox (knnsearch)
%   - Image Processing Toolbox (dicomread, dicominfo)
%
% Author: Chungwon Lee
% Last Updated: 2026-04-15
% =========================================================================

clear; clc; close all;

%% ======================== USER SETTINGS =================================

PATH_RT_PLAN_CT     = 'C:\Users\USER\Desktop\B\CT';
PATH_RT_STRUCT      = 'C:\Users\USER\Desktop\B\RS.48365583.BREAST_Lt.dcm';
PATH_RT_DOSE        = 'C:\Users\USER\Desktop\B\RD.48365583.BRST_Lt(40.5).dcm';
PATH_RT_PLAN        = 'C:\Users\User\Desktop\48365583\RT\RP.48365583.BRST_Lt(40.5).dcm';
PATH_OBJ_FOLDER     = 'C:\Users\User\Desktop\48365583\heart_echo_motion';
PATH_NIFTI_LV_WORLD = 'C:\Users\User\Desktop\48365583\lv_nifti_world.txt';

CYCLES_TO_SHOW = 3;
FPS_ANIM       = 30;
SAVE_VIDEO     = true;
VIDEO_FILENAME = 'heart_beam_dose.mp4';

% Beam display
BEAM_LENGTH_ENTRY = 200;   % mm, isocenter에서 entry 방향 길이
BEAM_LENGTH_EXIT  = 200;   % mm, isocenter에서 exit 방향 길이
SAD = 1000;

%% ======================== STEP 1: RTSTRUCT HEART =========================
fprintf('=== Step 1: Building RTSTRUCT heart mesh ===\n');
[faces_wh, verts_wh, roi_name] = build_heart_mesh_from_rtstruct(PATH_RT_PLAN_CT, PATH_RT_STRUCT);
verts_wh = smooth_mesh_vertices(faces_wh, verts_wh, 0.5, 5);
fprintf('  ROI: "%s" | Vertices: %d\n', roi_name, size(verts_wh, 1));

%% ======================== STEP 2: RT DOSE ================================
fprintf('\n=== Step 2: Loading RT dose ===\n');
info_dose = dicominfo(PATH_RT_DOSE);
dose_vol  = double(squeeze(dicomread(info_dose))) * info_dose.DoseGridScaling;
dPos = info_dose.ImagePositionPatient;
dSpc = info_dose.PixelSpacing;
if isfield(info_dose, 'GridFrameOffsetVector')
    dZ = info_dose.GridFrameOffsetVector + dPos(3);
else
    dZ = dPos(3) + (0:size(dose_vol,3)-1)' * 3.0;
end
dX = dPos(1) + (0:size(dose_vol,2)-1) * dSpc(2);
dY = dPos(2) + (0:size(dose_vol,1)-1) * dSpc(1);
[GridX, GridY, GridZ] = meshgrid(dX, dY, dZ);

dose_wh = interp3(GridX, GridY, GridZ, dose_vol, ...
    verts_wh(:,1), verts_wh(:,2), verts_wh(:,3), 'linear');
dose_wh(isnan(dose_wh)) = 0;
fprintf('  Whole heart max dose: %.2f Gy\n', max(dose_wh));

%% ======================== STEP 3: OBJ LV ================================
fprintf('\n=== Step 3: Loading OBJ motion frames ===\n');
obj_files = dir(fullfile(PATH_OBJ_FOLDER, 'heart_*.obj'));
[~, si] = sort({obj_files.name}); obj_files = obj_files(si);
nFrames = numel(obj_files);

[verts0, faces_obj] = read_obj_mesh(fullfile(PATH_OBJ_FOLDER, obj_files(1).name));
nVerts = size(verts0, 1);
fprintf('  %d frames, %d vertices\n', nFrames, nVerts);

all_verts = zeros(nVerts, 3, nFrames);
all_verts(:,:,1) = verts0;
for i = 2:nFrames
    [v, ~] = read_obj_mesh(fullfile(PATH_OBJ_FOLDER, obj_files(i).name));
    all_verts(:,:,i) = v;
end

vol = zeros(nFrames,1);
for i = 1:nFrames
    try [~,vol(i)] = convhull(all_verts(:,1,i),all_verts(:,2,i),all_verts(:,3,i));
        vol(i)=vol(i)/1000; catch, vol(i)=NaN; end
end
EDV = max(vol); ESV = min(vol); EF = (EDV-ESV)/EDV*100;
fprintf('  EDV: %.1f mL | ESV: %.1f mL | EF: %.1f%%\n', EDV, ESV, EF);

%% ======================== STEP 4: REGISTRATION ===========================
fprintf('\n=== Step 4: OBJ → Patient coords registration ===\n');

lv_world_ras = readmatrix(PATH_NIFTI_LV_WORLD);
lv_lps = lv_world_ras .* [-1, -1, 1];  % RAS → LPS

c_obj = mean(verts0); c_lv = mean(lv_lps);
obj_centered = verts0 - c_obj + c_lv;

fprintf('  CPD...\n');
tformCPD = pcregistercpd(pointCloud(obj_centered), pointCloud(lv_lps), ...
    'Transform', 'Rigid', 'MaxIterations', 200);
verts_cpd = pctransform(pointCloud(obj_centered), tformCPD).Location;

fprintf('  ICP...\n');
[tformICP, movingReg] = pcregistericp(pointCloud(verts_cpd), pointCloud(lv_lps), ...
    'MaxIterations', 200, 'Tolerance', [0.001 0.001]);
verts_registered = movingReg.Location;

% Extract combined rigid transform: raw OBJ → patient coords
% Transform = ICP ∘ CPD ∘ centroid_shift
% For each frame: v_patient = apply_registration(v_raw)
%   Step 1: v_centered = v_raw - c_obj + c_lv
%   Step 2: v_cpd = tformCPD * v_centered
%   Step 3: v_final = tformICP * v_cpd

fprintf('  Transforming all %d frames to patient coords...\n', nFrames);
all_verts_patient = zeros(nVerts, 3, nFrames);
for f = 1:nFrames
    v = all_verts(:,:,f);
    v = v - c_obj + c_lv;                                    % centroid shift
    v = pctransform(pointCloud(v), tformCPD).Location;        % CPD
    v = pctransform(pointCloud(v), tformICP).Location;        % ICP
    all_verts_patient(:,:,f) = v;
end
fprintf('  Done.\n');

[~, nn_dist] = knnsearch(lv_lps, verts_registered);
fprintf('  Registration quality: mean %.2f mm, 95th %.2f mm\n', ...
    mean(nn_dist), prctile(nn_dist, 95));

%% ======================== STEP 5: LV DOSE ================================
fprintf('\n=== Step 5: LV dose mapping ===\n');

dose_nifti_lv = interp3(GridX, GridY, GridZ, dose_vol, ...
    lv_lps(:,1), lv_lps(:,2), lv_lps(:,3), 'linear');
dose_nifti_lv(isnan(dose_nifti_lv)) = 0;

[nn_idx, ~] = knnsearch(lv_lps, verts_registered);
dose_on_obj = dose_nifti_lv(nn_idx);
fprintf('  LV max: %.2f Gy | mean: %.4f Gy\n', max(dose_on_obj), mean(dose_on_obj));

%% ======================== STEP 6: RT PLAN BEAMS ==========================
fprintf('\n=== Step 6: Loading beam info ===\n');
rp = dicominfo(PATH_RT_PLAN);
beam_seq = rp.BeamSequence;
bf = fieldnames(beam_seq);
nBeams = numel(bf);

beam_info = struct();
for i = 1:nBeams
    b = beam_seq.(bf{i});
    cp1 = b.ControlPointSequence.Item_1;
    beam_info(i).name    = b.BeamName;
    beam_info(i).gantry  = cp1.GantryAngle;
    beam_info(i).coll    = cp1.BeamLimitingDeviceAngle;
    beam_info(i).iso     = cp1.IsocenterPosition(:)';
    beam_info(i).energy  = cp1.NominalBeamEnergy;
    
    bld = cp1.BeamLimitingDevicePositionSequence;
    bld_f = fieldnames(bld);
    beam_info(i).jaw_x = [-100, 100];
    beam_info(i).jaw_y = [-110, 110];
    for j = 1:numel(bld_f)
        item = bld.(bld_f{j});
        lp = item.LeafJawPositions;
        switch item.RTBeamLimitingDeviceType
            case 'ASYMX', beam_info(i).jaw_x = [lp(1), lp(2)];
            case 'ASYMY', beam_info(i).jaw_y = [lp(1), lp(2)];
        end
    end
    fprintf('  Beam %d: %s | Gantry %.0f° | %dMV\n', ...
        i, beam_info(i).name, beam_info(i).gantry, beam_info(i).energy);
end

%% ======================== STEP 7: ANIMATION PREP =========================
fprintf('\n=== Step 7: Preparing animation ===\n');

HR = 66; period = 60/HR;
nAnimFrames = max(60, round(CYCLES_TO_SHOW * period * FPS_ANIM));
t_anim = linspace(0, CYCLES_TO_SHOW * period, nAnimFrames);
t_frames = linspace(0, period, nFrames+1); t_frames = t_frames(1:nFrames);
t_in_cycle = mod(t_anim, period);

all_verts_cyc = cat(3, all_verts_patient, all_verts_patient(:,:,1));
t_frames_cyc = [t_frames, period];

fprintf('  Interpolating %d vertices x %d frames...\n', nVerts, nAnimFrames);
anim_verts = zeros(nVerts, 3, nAnimFrames);
for v = 1:nVerts
    for d = 1:3
        anim_verts(v,d,:) = interp1(t_frames_cyc, squeeze(all_verts_cyc(v,d,:)), ...
            t_in_cycle, 'pchip');
    end
end
fprintf('  Done.\n');

%% ======================== STEP 8: VISUALIZATION ==========================
fprintf('\n=== Step 8: Rendering ===\n');

iso = beam_info(1).iso;
beam_colors_rgb = [1 0.85 0; 1 0.85 0];  % 노란색 beam tubes (reference 이미지 스타일)

% --- Figure 1: Static overview ---
fig1 = figure('Position',[50 50 1400 1000], 'Color','k', 'Name','Heart + Beams');
ax1 = axes('Parent',fig1); hold(ax1,'on');

% Whole heart (transparent, gray)
patch(ax1, 'Faces',faces_wh, 'Vertices',verts_wh, ...
    'FaceColor',[0.8 0.8 0.85], 'EdgeColor','none', ...
    'FaceAlpha',0.25, 'FaceLighting','gouraud');

% LV with dose
patch(ax1, 'Faces',faces_obj, 'Vertices',squeeze(all_verts_patient(:,:,1)), ...
    'FaceVertexCData', dose_on_obj, ...
    'FaceColor','interp', 'EdgeColor','none', 'FaceAlpha',0.9, ...
    'FaceLighting','gouraud');
colormap(ax1,'jet');
cb = colorbar(ax1); cb.Label.String = 'Dose (Gy)'; cb.Color='w'; cb.Label.FontSize=12;
caxis(ax1, [0, max(dose_wh)]);

% Isocenter
scatter3(ax1, iso(1),iso(2),iso(3), 150, 'r', 'pentagram', 'filled');

% Draw beam prisms
for b = 1:nBeams
    draw_beam_prism(ax1, beam_info(b), ...
        BEAM_LENGTH_ENTRY, BEAM_LENGTH_EXIT, ...
        beam_colors_rgb(b,:), SAD);
end

axis(ax1,'equal'); view(ax1, [-45 25]); grid(ax1,'on');
ax1.Color = [0.05 0.05 0.08];
ax1.GridColor = [0.2 0.2 0.2];
ax1.XColor='w'; ax1.YColor='w'; ax1.ZColor='w';
xlabel(ax1,'X (mm)','Color','w'); ylabel(ax1,'Y (mm)','Color','w');
zlabel(ax1,'Z (mm)','Color','w');
title(ax1, sprintf('Heart + LV + RT Beams (%.1f Gy)', 40.5), 'Color','w','FontSize',14);
camlight('headlight'); camlight('right');
lighting gouraud; rotate3d on;

% --- Figure 2: Beating animation ---
fig2 = figure('Position',[100 100 1200 900], 'Color','k', 'Name','LV Beating + Beams');
ax2 = axes('Parent',fig2); hold(ax2,'on');

% Whole heart background
patch(ax2, 'Faces',faces_wh, 'Vertices',verts_wh, ...
    'FaceColor',[0.7 0.7 0.75], 'EdgeColor','none', ...
    'FaceAlpha',0.12, 'FaceLighting','gouraud');

% LV (animated)
hp = patch(ax2, 'Faces',faces_obj, 'Vertices',squeeze(anim_verts(:,:,1)), ...
    'FaceVertexCData', dose_on_obj, ...
    'FaceColor','interp', 'EdgeColor','none', 'FaceAlpha',0.9, ...
    'FaceLighting','gouraud');
colormap(ax2,'jet');
cb2 = colorbar(ax2); cb2.Label.String='Dose (Gy)'; cb2.Color='w';
caxis(ax2,[0, max(dose_wh)]);

% Beam prisms
for b = 1:nBeams
    draw_beam_prism(ax2, beam_info(b), ...
        BEAM_LENGTH_ENTRY, BEAM_LENGTH_EXIT, ...
        beam_colors_rgb(b,:), SAD);
end

scatter3(ax2, iso(1),iso(2),iso(3), 100, 'r', 'pentagram', 'filled');

axis(ax2,'equal'); view(ax2,[-45 25]); grid(ax2,'on');
ax2.Color = [0.05 0.05 0.08]; ax2.GridColor=[0.2 0.2 0.2];
ax2.XColor='w'; ax2.YColor='w'; ax2.ZColor='w';
xlabel(ax2,'X','Color','w'); ylabel(ax2,'Y','Color','w'); zlabel(ax2,'Z','Color','w');
title(ax2, sprintf('LV Beating + RT Dose + Beams (HR=%d, EF=%.0f%%)', HR, EF), ...
    'Color','w','FontSize',14);
camlight('headlight'); camlight('right');
lighting gouraud; rotate3d on;

ht = text(ax2, 0.02,0.98, '', 'Units','normalized', 'Color','w', ...
    'FontSize',11, 'VerticalAlignment','top', 'FontWeight','bold');

if SAVE_VIDEO
    vw = VideoWriter(VIDEO_FILENAME,'MPEG-4');
    vw.FrameRate=FPS_ANIM; vw.Quality=95; open(vw);
end

for k = 1:nAnimFrames
    set(hp, 'Vertices', squeeze(anim_verts(:,:,k)));
    set(ht, 'String', sprintf('Cycle %d | t=%.3f s', ...
        floor(t_anim(k)/period)+1, mod(t_anim(k),period)));
    drawnow;
    if SAVE_VIDEO, writeVideo(vw, getframe(fig2)); end
end
if SAVE_VIDEO, close(vw); fprintf('  Video saved: %s\n', VIDEO_FILENAME); end

fprintf('\n=== Complete ===\n');

%% ======================= BEAM DRAWING ====================================

function draw_beam_prism(ax, binfo, len_entry, len_exit, color, SAD)
% DRAW_BEAM_PRISM  Draw a diverging rectangular beam prism based on jaw positions.
%   Uses ASYMX/ASYMY jaw positions to define the rectangular cross-section.
%   The prism diverges from the source according to beam geometry.
%
%   IEC Fixed → DICOM Patient (supine HFS):
%     Gantry 0° = beam from anterior
%     source_dir = [sin(θ), -cos(θ), 0]

    iso = binfo.iso;
    ga = binfo.gantry * pi / 180;
    jx = binfo.jaw_x;  % [x1, x2] mm at isocenter
    jy = binfo.jaw_y;  % [y1, y2] mm at isocenter
    
    % Beam direction vectors
    source_dir = [sin(ga), -cos(ga), 0];
    beam_dir = -source_dir;
    beam_dir = beam_dir / norm(beam_dir);
    
    % Perpendicular axes (X-jaw = lateral, Y-jaw = sup-inf)
    % perp1 = X-jaw direction (lateral, in gantry rotation plane)
    % perp2 = Y-jaw direction (superior-inferior, along couch)
    if abs(beam_dir(3)) < 0.99
        up = [0, 0, 1];
    else
        up = [1, 0, 0];
    end
    perp1 = cross(beam_dir, up); perp1 = perp1 / norm(perp1);  % X-jaw
    perp2 = cross(beam_dir, perp1); perp2 = perp2 / norm(perp2);  % Y-jaw
    
    % Entry and exit points along central axis
    pt_entry = iso - len_entry * beam_dir;
    pt_exit  = iso + len_exit  * beam_dir;
    
    % Divergence scale at entry and exit
    d_entry = norm(pt_entry - (iso + SAD*source_dir));  % distance from source
    d_exit  = norm(pt_exit  - (iso + SAD*source_dir));
    s_entry = d_entry / SAD;
    s_exit  = d_exit  / SAD;
    
    % 4 corners at entry plane
    c_entry = get_rect_corners(pt_entry, perp1, perp2, jx, jy, s_entry);
    % 4 corners at exit plane
    c_exit  = get_rect_corners(pt_exit, perp1, perp2, jx, jy, s_exit);
    
    % Draw 4 side faces
    for i = 1:4
        j = mod(i, 4) + 1;
        vx = [c_entry(i,1) c_entry(j,1) c_exit(j,1) c_exit(i,1)];
        vy = [c_entry(i,2) c_entry(j,2) c_exit(j,2) c_exit(i,2)];
        vz = [c_entry(i,3) c_entry(j,3) c_exit(j,3) c_exit(i,3)];
        patch(ax, 'XData',vx, 'YData',vy, 'ZData',vz, ...
            'FaceColor',color, 'FaceAlpha',0.1, ...
            'EdgeColor',color, 'EdgeAlpha',0.6, 'LineWidth',1.5);
    end
    
    % Draw entry and exit cap faces
    for cc = {c_entry, c_exit}
        corners = cc{1};
        patch(ax, 'XData',corners(:,1), 'YData',corners(:,2), 'ZData',corners(:,3), ...
            'FaceColor',color, 'FaceAlpha',0.08, ...
            'EdgeColor',color, 'EdgeAlpha',0.8, 'LineWidth',2);
    end
    
    % Draw 4 diverging edges (entry corner → exit corner)
    for i = 1:4
        plot3(ax, [c_entry(i,1) c_exit(i,1)], ...
                   [c_entry(i,2) c_exit(i,2)], ...
                   [c_entry(i,3) c_exit(i,3)], ...
            '-', 'Color', [color 0.7], 'LineWidth', 1.5);
    end
    
    % Central axis (dashed)
    plot3(ax, [pt_entry(1) pt_exit(1)], [pt_entry(2) pt_exit(2)], ...
        [pt_entry(3) pt_exit(3)], '--', 'Color', [color 0.4], 'LineWidth', 1);
    
    % Label
    label_pos = pt_entry + 20*perp2;
    text(ax, label_pos(1), label_pos(2), label_pos(3), ...
        sprintf('%s\n%.0f° / %dMV', binfo.name, binfo.gantry, binfo.energy), ...
        'Color', color, 'FontSize', 10, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');
end

function corners = get_rect_corners(center, perp1, perp2, jx, jy, scale)
% GET_RECT_CORNERS  Compute 4 corners of rectangular field at given plane.
    corners = zeros(4, 3);
    corners(1,:) = center + jx(1)*scale*perp1 + jy(1)*scale*perp2;
    corners(2,:) = center + jx(2)*scale*perp1 + jy(1)*scale*perp2;
    corners(3,:) = center + jx(2)*scale*perp1 + jy(2)*scale*perp2;
    corners(4,:) = center + jx(1)*scale*perp1 + jy(2)*scale*perp2;
end

%% ======================= HELPER FUNCTIONS ================================

function [vertices, faces] = read_obj_mesh(filepath)
    fid = fopen(filepath, 'r');
    assert(fid ~= -1, 'Cannot open: %s', filepath);
    raw = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
    fclose(fid);
    lines = raw{1};
    v_lines = lines(startsWith(lines, 'v '));
    f_lines = lines(startsWith(lines, 'f '));
    vertices = zeros(numel(v_lines), 3);
    for i = 1:numel(v_lines)
        vertices(i,:) = sscanf(v_lines{i}, 'v %f %f %f')';
    end
    faces = zeros(numel(f_lines), 3);
    for i = 1:numel(f_lines)
        parts = strsplit(strtrim(f_lines{i}(3:end)));
        for j = 1:min(3, numel(parts))
            faces(i,j) = sscanf(parts{j}, '%d', 1);
        end
    end
end

function [faces, verts_mm, sel_name] = build_heart_mesh_from_rtstruct(path_ct_folder, path_rs)
    files_ct_raw = dir(fullfile(path_ct_folder, '*.dcm'));
    info_ct = {};
    for i = 1:length(files_ct_raw)
        f = fullfile(path_ct_folder, files_ct_raw(i).name);
        try I = dicominfo(f);
            if isfield(I,'ImagePositionPatient'), info_ct{end+1}=I; end
        catch, end
    end
    assert(~isempty(info_ct), 'No valid CT slices.');
    [~,ord] = sort(cellfun(@(s) s.ImagePositionPatient(3), info_ct));
    info_ct = info_ct(ord);
    CT_rows=double(info_ct{1}.Rows); CT_cols=double(info_ct{1}.Columns);
    CT_slices=length(info_ct);
    img_pos0=reshape(double(info_ct{1}.ImagePositionPatient),1,3);
    ps=reshape(double(info_ct{1}.PixelSpacing),1,2); dr=ps(1); dc=ps(2);
    z_positions=cellfun(@(s) double(s.ImagePositionPatient(3)), info_ct);
    dz=median(abs(diff(z_positions)));
    if ~isfinite(dz)||dz==0, dz=1; end

    S=dicominfo(path_rs); roi_list=S.StructureSetROISequence;
    roi_fields=fieldnames(roi_list);
    fprintf('  --- ROI List ---\n');
    for i=1:numel(roi_fields)
        rr=roi_list.(roi_fields{i});
        fprintf('  [%d] %s\n', rr.ROINumber, string(rr.ROIName));
    end
    idx=[]; heart_idx=[];
    for i=1:numel(roi_fields)
        rr=roi_list.(roi_fields{i});
        if contains(string(rr.ROIName),'heart','IgnoreCase',true), heart_idx(end+1)=i; end
    end
    if numel(heart_idx)==1, idx=heart_idx(1);
    elseif numel(heart_idx)>1
        for j=1:numel(heart_idx)
            rname=strtrim(string(roi_list.(roi_fields{heart_idx(j)}).ROIName));
            clean=regexprep(rname,'^\d+[\s_]*','');
            if strcmpi(clean,'Heart'), idx=heart_idx(j); break; end
        end
        if isempty(idx), idx=heart_idx(1); end
    end
    if isempty(idx)
        roi_choice=input('Enter Heart ROI number: ');
        idx=find(cellfun(@(fn) roi_list.(fn).ROINumber==roi_choice, roi_fields),1);
    end
    sel_roi=roi_list.(roi_fields{idx}); sel_name=string(sel_roi.ROIName);
    fprintf('  -> Selected: [%d] %s\n', sel_roi.ROINumber, sel_name);

    contour_seq=[]; rc_fields=fieldnames(S.ROIContourSequence);
    for i=1:numel(rc_fields)
        rc=S.ROIContourSequence.(rc_fields{i});
        if rc.ReferencedROINumber==sel_roi.ROINumber && isfield(rc,'ContourSequence')
            contour_seq=rc.ContourSequence; break;
        end
    end
    assert(~isempty(contour_seq),'No ContourSequence.');

    HeartMask=false(CT_rows,CT_cols,CT_slices);
    cs_fields=fieldnames(contour_seq);
    for i=1:numel(cs_fields)
        citem=contour_seq.(cs_fields{i});
        if ~isfield(citem,'ContourData')||isempty(citem.ContourData), continue; end
        pts=reshape(double(citem.ContourData),3,[]).';
        z_here=pts(1,3); [~,si]=min(abs(z_positions-z_here));
        si=max(1,min(CT_slices,si));
        vy=(pts(:,2)-img_pos0(2))./dr+1; vx=(pts(:,1)-img_pos0(1))./dc+1;
        vx=round(max(1,min(CT_cols,vx))); vy=round(max(1,min(CT_rows,vy)));
        HeartMask(:,:,si)=HeartMask(:,:,si)|poly2mask(vx,vy,CT_rows,CT_cols);
    end
    assert(any(HeartMask(:)),'Heart mask empty.');
    [faces,verts]=isosurface(HeartMask,0.5);
    verts_mm=zeros(size(verts));
    verts_mm(:,1)=(verts(:,1)-1)*dc+img_pos0(1);
    verts_mm(:,2)=(verts(:,2)-1)*dr+img_pos0(2);
    verts_mm(:,3)=(verts(:,3)-1)*dz+z_positions(1);
end

function verts_new = smooth_mesh_vertices(faces, verts, alpha, iter)
    verts_new=verts; TR=triangulation(faces,verts);
    try att=vertexAttachments(TR);
        for k=1:iter, vp=verts_new;
            for i=1:size(verts,1)
                fi=att{i}; if isempty(fi), continue; end
                nbs=unique(faces(fi,:)); nbs(nbs==i)=[];
                if ~isempty(nbs), verts_new(i,:)=(1-alpha)*vp(i,:)+alpha*mean(vp(nbs,:),1); end
            end
        end
    catch, verts_new=verts; end
end