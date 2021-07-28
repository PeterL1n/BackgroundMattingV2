#!/usr/bin/octave
arg_list = argv ();
bench_path = arg_list{1};
result_path = arg_list{2};


gt_files = dir(fullfile(bench_path, 'pha', '*.png'));

total_loss_mse = 0;
total_loss_sad = 0;
total_loss_gradient = 0;
total_loss_connectivity = 0;

total_fg_mse = 0;
total_premult_mse = 0;

for i = 1:length(gt_files)
    filename = gt_files(i).name;

    gt_fullname = fullfile(bench_path, 'pha', filename);
    gt_alpha = imread(gt_fullname);
    trimap = imread(fullfile(bench_path, 'trimap', filename));
    crop_edge = idivide(size(gt_alpha), 4) * 4;
    gt_alpha = gt_alpha(1:crop_edge(1), 1:crop_edge(2));
    trimap = trimap(1:crop_edge(1), 1:crop_edge(2));
    
    result_fullname = fullfile(result_path, 'pha', filename);%strrep(filename, '.png', '.jpg'));
    hat_alpha = imread(result_fullname)(1:crop_edge(1), 1:crop_edge(2));
    

    fg_hat_fullname = fullfile(result_path, 'fgr', filename);%strrep(filename, '.png', '.jpg'));
    fg_gt_fullname = fullfile(bench_path, 'fgr', filename);
    hat_fgr = imread(fg_hat_fullname)(1:crop_edge(1), 1:crop_edge(2), :);
    gt_fgr = imread(fg_gt_fullname)(1:crop_edge(1), 1:crop_edge(2), :);
    nonzero_alpha = gt_alpha > 0;


    % fprintf('size(gt_fgr) is %s\n', mat2str(size(gt_fgr)))
    fg_mse = mean(compute_mse_loss(hat_fgr .* nonzero_alpha, gt_fgr .* nonzero_alpha, trimap));
    mse = compute_mse_loss(hat_alpha, gt_alpha, trimap);
    sad = compute_sad_loss(hat_alpha, gt_alpha, trimap);
    grad = compute_gradient_loss(hat_alpha, gt_alpha, trimap);
    conn = compute_connectivity_error(hat_alpha, gt_alpha, trimap, 0.1);


    fprintf(2, strcat(filename, ',%.6f,%.3f,%.0f,%.0f,%.6f\n'), mse, sad, grad, conn, fg_mse);
    fflush(stderr);

    total_loss_mse += mse;
    total_loss_sad += sad;
    total_loss_gradient += grad;
    total_loss_connectivity += conn;
    total_fg_mse += fg_mse;
end

avg_loss_mse = total_loss_mse / length(gt_files);
avg_loss_sad = total_loss_sad / length(gt_files);
avg_loss_gradient = total_loss_gradient / length(gt_files);
avg_loss_connectivity = total_loss_connectivity / length(gt_files);
avg_loss_fg_mse = total_fg_mse / length(gt_files);

fprintf('mse:%.6f,sad:%.3f,grad:%.0f,conn:%.0f,fg_mse:%.6f\n', avg_loss_mse, avg_loss_sad, avg_loss_gradient, avg_loss_connectivity, avg_loss_fg_mse);
