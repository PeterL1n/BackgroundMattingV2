% compute the SAD error given a prediction, a ground truth and a trimap.
% author Ning Xu
% date 2018-1-1

function loss = compute_sad_loss(pred,target,trimap)
error_map = abs(single(pred)-single(target))/255;
loss = sum(sum(error_map.*single(trimap==128))) ;

% the loss is scaled by 1000 due to the large images used in our experiment.
% Please check the result table in our paper to make sure the result is correct. 
loss = loss / 1000 ;
