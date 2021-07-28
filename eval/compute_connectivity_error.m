% compute the connectivity error given a prediction, a ground truth and a trimap.
% author Ning Xu
% date 2018-1-1

% pred: the predicted alpha matte
% target: the ground truth alpha matte
% trimap: the given trimap
% step = 0.1

function loss = compute_connectivity_error(pred,target,trimap,step)
pred = single(pred)/255;
target = single(target)/255;

[dimy,dimx] = size(pred);

thresh_steps = 0:step:1;
l_map = ones(size(pred))*(-1);
dist_maps = zeros([dimy,dimx,numel(thresh_steps)]);
for ii = 2:numel(thresh_steps)
    pred_alpha_thresh = pred>=thresh_steps(ii);
    target_alpha_thresh = target>=thresh_steps(ii);
    
    cc = bwconncomp(pred_alpha_thresh & target_alpha_thresh,4);    
    size_vec = cellfun(@numel,cc.PixelIdxList);
    [~,max_id] = max(size_vec);
    
    omega = zeros([dimy,dimx]);
    omega(cc.PixelIdxList{max_id}) = 1;
            
    flag = l_map==-1 & omega==0;    
    l_map(flag==1) = thresh_steps(ii-1);
    
    dist_maps(:,:,ii) = bwdist(omega);  
    dist_maps(:,:,ii) = dist_maps(:,:,ii) / max(max(dist_maps(:,:,ii)));
end
l_map(l_map==-1) = 1;

pred_d = pred - l_map;
target_d = target - l_map;

pred_phi = 1 -  pred_d .* single(pred_d>=0.15);

target_phi = 1 -  target_d .* single(target_d>=0.15);

loss = sum(sum(abs(pred_phi - target_phi).*single(trimap==128)));

