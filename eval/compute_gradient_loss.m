% compute the gradient error given a prediction, a ground truth and a trimap.
% author Ning Xu
% date 2018-1-1

% pred: the predicted alpha matte
% target: the ground truth alpha matte
% trimap: the given trimap
% step = 0.1

function loss = compute_gradient_loss(pred,target,trimap)
pred = mat2gray(pred);
target = mat2gray(target);
[pred_x,pred_y] = gaussgradient(pred,1.4);
[target_x,target_y] = gaussgradient(target,1.4);
pred_amp = sqrt(pred_x.^2 + pred_y.^2);
target_amp = sqrt(target_x.^2 + target_y.^2);

error_map = (single(pred_amp) - single(target_amp)).^2;
loss = sum(sum(error_map.*single(trimap==128))) ;
