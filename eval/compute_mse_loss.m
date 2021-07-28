% compute the MSE error given a prediction, a ground truth and a trimap.
% author Ning Xu
% date 2018-1-1

% pred: the predicted alpha matte
% target: the ground truth alpha matte
% trimap: the given trimap

function loss = compute_mse_loss(pred,target,trimap)
error_map = (single(pred)-single(target))/255;

% fprintf('size(error_map) is %s\n', mat2str(size(error_map)))
loss = sum(sum(error_map.^2.*single(trimap==128))) / sum(sum(single(trimap==128)));
