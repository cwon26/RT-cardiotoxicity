% =========================================================================
% Heart 3D Mesh + RT Dose Overlay + Echo-driven Beating Animation
% =========================================================================
% - CT + RTSTRUCT -> Heart mesh (mm coordinates)
% - Echo cine DICOM cycle info for time-varying contraction
% - (Optional) GLS CSV for patient-specific contraction curve
% -------------------------------------------------------------------------
% Requirements: Base MATLAB (VideoWriter for saving animations)
% =========================================================================

clear; clc; close all;

%% ----------- Path Settings ----------------------------------------------
path_rt_plan_ct = 'C:\Users\USER\Desktop\B\CT';
path_rt_struct  = 'C:\Users\USER\Desktop\B\RS.48365583.BREAST_Lt.dcm';
path_echo_dcm   = 'C:\Users\USER\Desktop\B\echo0022.dcm';
path_gls_csv    = '';  % e.g. 'GLS_curve.csv' (columns: time_sec, GLS_percent)
path_rt_dose    = 'C:\Users\USER\Desktop\B\RD.48365583.BRST_Lt(40.5).dcm';

% Animation parameters
GLS_peak_default = 0.18;   % 18% contraction (absolute). Ignored if CSV provided
amp_mm           = 8;      % Max surface displacement (mm). Adjust 4~12 as needed
cycles_to_show   = 2;      % Number of cardiac cycles to animate
fps_anim         = 30;     % Animation frame rate

%% ----------- 1) CT/RTSTRUCT -> Heart mesh --------------------------------
disp('Step 1) Building heart mesh...');
[faces, verts_mm, sel_name] = build_heart_mesh_from_rtstruct(path_rt_plan_ct, path_rt_struct);
TR_raw = triangulation(faces, verts_mm);
verts_smooth = smooth_mesh_vertices(faces, verts_mm, 0.5, 5);
verts_mm = verts_smooth;

TR = triangulation(faces, verts_mm);
VN = vertexNormal(TR);
VN(~isfinite(VN)) = 0;
VN = normalize_rows(VN);

%% ----------- 1.5) RT Dose Mapping ----------------------------------------
disp('Step 1.5) Mapping dose distribution...');
if exist(path_rt_dose, 'file')
    info_dose = dicominfo(path_rt_dose);
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
    dose_on_mesh = interp3(GridX, GridY, GridZ, dose_vol, ...
                           verts_mm(:,1), verts_mm(:,2), verts_mm(:,3), 'linear');
    dose_on_mesh(isnan(dose_on_mesh)) = 0;

    fprintf('   -> Max Dose on Heart: %.2f Gy\n', max(dose_on_mesh(:)));
else
    warning('RTDOSE file not found. Skipping dose mapping.');
    dose_on_mesh = zeros(size(verts_mm,1), 1);
end

%% ----------- 2) Echo cine time axis / heart rate -------------------------
disp('Step 2) Extracting echo cine timing...');
assert(exist(path_echo_dcm,'file')==2, 'Check echo DICOM path.');
EI   = dicominfo(path_echo_dcm);
EIMG = dicomread(EI);
if ndims(EIMG)==4, nEchoFrames = size(EIMG,4); else, nEchoFrames = size(EIMG,3); end

cineRate   = getfield_default(EI, {'CineRate'}, NaN);
frameTimeM = getfield_default(EI, {'FrameTime'}, NaN);
if isfinite(cineRate)
    dt_echo = 1/cineRate;
elseif isfinite(frameTimeM)
    dt_echo = frameTimeM/1000;
else
    dt_echo = 1/50;
end
t_echo = (0:nEchoFrames-1) * dt_echo;

HR = getfield_default(EI, {'HeartRate'}, NaN);
if ~isfinite(HR) || HR < 20 || HR > 220
    HR = 60;
end
period = 60/HR;

%% ----------- 3) Contraction curve s(t) -----------------------------------
if ~isempty(path_gls_csv) && exist(path_gls_csv,'file')==2
    T = readmatrix(path_gls_csv);
    if size(T,2) >= 2
        tt  = T(:,1);
        gls = T(:,2)/100;
    else
        tt  = (0:length(T)-1)' * dt_echo;
        gls = T(:,1)/100;
    end
    gls = fillmissing(gls,'linear');
    GLS_pk = max(0.05, min(0.35, abs(min(gls))));
    s_raw  = max(0, -gls / GLS_pk);
    t_anim = linspace(0, cycles_to_show*period, max(60, round(cycles_to_show*period*fps_anim)));
    s_t    = interp1(tt, s_raw, mod(t_anim, tt(end)), 'pchip', 'extrap');
else
    % Sinusoidal approximation: ED(0) -> ES(peak) -> ED(0)
    t_anim = linspace(0, cycles_to_show*period, max(60, round(cycles_to_show*period*fps_anim)));
    GLS_pk = GLS_peak_default;
    s_t = 0.5*(1 - cos(2*pi * t_anim / period));
end

%% ----------- 4) Animation ------------------------------------------------
disp('Step 3) Rendering animation...');
fig = figure('Name','Beating Heart (Echo-driven)','Position',[100 100 1100 850]);
ax  = axes('Parent', fig); hold(ax,'on');

hp = patch(ax, 'Faces', faces, 'Vertices', verts_mm, ...
    'FaceVertexCData', dose_on_mesh, ...
    'FaceColor', 'interp', ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 0.9);

colormap(ax, 'jet');
c = colorbar(ax);
c.Label.String = 'Dose (Gy)';
if max(dose_on_mesh(:)) > 0
    caxis(ax, [0, max(dose_on_mesh(:))]);
end

axis(ax,'equal'); view(ax,3); grid(ax,'on');
xlabel(ax,'X (mm)'); ylabel(ax,'Y (mm)'); zlabel(ax,'Z (mm)');
title(ax, sprintf('Heart ROI: "%s" - Echo-driven animation', sel_name), 'Interpreter','none');
camlight headlight; lighting gouraud; rotate3d on;

V0 = verts_mm;
VN = normalize_rows(VN);
for k = 1:numel(s_t)
    shrink = amp_mm * s_t(k);
    Vnew   = V0 - VN .* shrink;
    set(hp, 'Vertices', Vnew);
    drawnow;
end

disp('Done. Increase cycles_to_show for longer animation.');

%% ======================= Helper Functions ================================

function [faces, verts_mm, sel_name] = build_heart_mesh_from_rtstruct(path_ct_folder, path_rs)
    % Load CT slice metadata
    files_ct_raw = dir(fullfile(path_ct_folder, '*.dcm'));
    info_ct = {};
    for i = 1:length(files_ct_raw)
        f = fullfile(path_ct_folder, files_ct_raw(i).name);
        try
            I = dicominfo(f);
            if isfield(I,'ImagePositionPatient')
                info_ct{end+1} = I; %#ok<SAGROW>
            end
        catch
        end
    end
    assert(~isempty(info_ct), 'No valid CT slices found.');

    [~, ord] = sort(cellfun(@(s) s.ImagePositionPatient(3), info_ct));
    info_ct = info_ct(ord);

    CT_rows   = double(getfield_default(info_ct{1}, {'Rows','Height'}));
    CT_cols   = double(getfield_default(info_ct{1}, {'Columns','Width'}));
    CT_slices = double(length(info_ct));
    img_pos0  = reshape(double(info_ct{1}.ImagePositionPatient),1,3);
    ps        = reshape(double(info_ct{1}.PixelSpacing),1,2);
    dr = ps(1); dc = ps(2);
    z_positions = cellfun(@(s) double(s.ImagePositionPatient(3)), info_ct);
    dz = 1;
    if numel(z_positions) >= 2
        dz = median(abs(diff(z_positions)));
    end
    if ~isfinite(dz) || dz==0, dz = 1; end

    % Load RTSTRUCT and list ROIs
    S = dicominfo(path_rs);
    roi_list   = S.StructureSetROISequence;
    roi_fields = fieldnames(roi_list);
    fprintf('--- ROI List ---\n');
    for i = 1:numel(roi_fields)
        rr = roi_list.(roi_fields{i});
        fprintf('[%d] %s\n', rr.ROINumber, string(rr.ROIName));
    end

    % Auto-select Heart ROI
    idx = [];
    heart_idx = [];
    for i = 1:numel(roi_fields)
        rr = roi_list.(roi_fields{i});
        if contains(string(rr.ROIName), 'heart', 'IgnoreCase', true)
            heart_idx(end+1) = i; %#ok<SAGROW>
        end
    end

    if numel(heart_idx) == 1
        idx = heart_idx(1);
    elseif numel(heart_idx) > 1
        % Prefer exact 'Heart' match (strip numeric prefix)
        for j = 1:numel(heart_idx)
            rname = strtrim(string(roi_list.(roi_fields{heart_idx(j)}).ROIName));
            clean = regexprep(rname, '^\d+[\s_]*', '');
            if strcmpi(clean, 'Heart')
                idx = heart_idx(j);
                break;
            end
        end
        if isempty(idx)
            idx = heart_idx(1);
        end
    end

    % Manual fallback if auto-selection fails
    if isempty(idx)
        warning('Could not auto-detect Heart ROI. Manual selection required.');
        roi_choice = input('Enter Heart ROI number: ');
        idx = find(cellfun(@(fn) roi_list.(fn).ROINumber==roi_choice, roi_fields), 1);
        assert(~isempty(idx), 'ROINumber %d not found.', roi_choice);
    else
        fprintf('   -> Auto-selected: [%d] %s\n', ...
            roi_list.(roi_fields{idx}).ROINumber, ...
            string(roi_list.(roi_fields{idx}).ROIName));
    end

    sel_roi  = roi_list.(roi_fields{idx});
    sel_name = string(sel_roi.ROIName);

    % Build binary mask from contours
    contour_seq = [];
    rc_fields = fieldnames(S.ROIContourSequence);
    for i = 1:numel(rc_fields)
        rc = S.ROIContourSequence.(rc_fields{i});
        if rc.ReferencedROINumber == sel_roi.ROINumber && isfield(rc,'ContourSequence')
            contour_seq = rc.ContourSequence; break;
        end
    end
    assert(~isempty(contour_seq), 'No ContourSequence found for selected ROI.');

    HeartMask = false(CT_rows, CT_cols, CT_slices);
    cs_fields = fieldnames(contour_seq);
    for i = 1:numel(cs_fields)
        citem = contour_seq.(cs_fields{i});
        if ~isfield(citem,'ContourData') || isempty(citem.ContourData), continue; end
        pts = reshape(double(citem.ContourData),3,[]).';
        z_here = pts(1,3);
        [~, si] = min(abs(z_positions - z_here));
        si = max(1, min(CT_slices, si));
        vy = (pts(:,2) - img_pos0(2))./dr + 1;
        vx = (pts(:,1) - img_pos0(1))./dc + 1;
        vx = round(max(1, min(CT_cols, vx)));
        vy = round(max(1, min(CT_rows, vy)));
        HeartMask(:,:,si) = HeartMask(:,:,si) | poly2mask(vx, vy, CT_rows, CT_cols);
    end
    assert(any(HeartMask(:)), 'Heart mask is empty.');

    % Isosurface extraction and coordinate transform
    [faces, verts] = isosurface(HeartMask, 0.5);
    verts_mm = zeros(size(verts));
    verts_mm(:,1) = (verts(:,1)-1)*dc + img_pos0(1);
    verts_mm(:,2) = (verts(:,2)-1)*dr + img_pos0(2);
    verts_mm(:,3) = (verts(:,3)-1)*dz + z_positions(1);
end

function val = getfield_default(s, names, defaultVal)
    if ischar(names) || isstring(names), names = {char(names)}; end
    for i = 1:numel(names)
        if isfield(s, names{i})
            val = s.(names{i}); return;
        end
    end
    val = defaultVal;
end

function V = normalize_rows(V)
    n = sqrt(sum(V.^2,2));
    n(n==0) = 1;
    V = V ./ n;
end

function verts_new = smooth_mesh_vertices(faces, verts, alpha, iter)
    % Laplacian smoothing
    verts_new = verts;
    TR = triangulation(faces, verts);
    try
        attachedFaces = vertexAttachments(TR);
        for k = 1:iter
            v_prev = verts_new;
            for i = 1:size(verts, 1)
                f_idx = attachedFaces{i};
                if isempty(f_idx), continue; end
                nbs = unique(faces(f_idx, :));
                nbs(nbs == i) = [];
                if ~isempty(nbs)
                    avg_pos = mean(v_prev(nbs, :), 1);
                    verts_new(i, :) = (1 - alpha)*v_prev(i, :) + alpha*avg_pos;
                end
            end
        end
    catch
        warning('Smoothing failed. Using raw mesh.');
        verts_new = verts;
    end
end