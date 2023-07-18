import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

# Assuming you have ground_truth and predicted spectrograms as tensors
# with shape (B, 2, F, T) and (B, 4, F, T) respectively

ground_truth = torch.randn((16, 2, 257, 5))
predicted = torch.randn((16, 4, 257, 5))

# B, _, F, T = ground_truth.shape
# num_sources_gt = 2
# num_sources_pred = 4

# # Reshape the tensors to simplify the computation
# ground_truth_reshape = ground_truth.view(B * num_sources_gt, -1)
# predicted_reshape = predicted.view(B * num_sources_pred, -1)

# # print(ground_truth_reshape[:, None].size())
# # print(predicted_reshape[None, :].size())
# # Compute the pairwise Euclidean distance as the cost matrix
# cost_matrix = torch.sum((ground_truth_reshape[None,:] - predicted_reshape[ :,None]) ** 2, dim=2)

# # Convert the cost matrix to a numpy array
# cost_matrix_np = cost_matrix.cpu().numpy()
# print(cost_matrix_np.shape)
# # Apply the Hungarian algorithm
# indices = linear_sum_assignment(cost_matrix_np)
# print(indices)
# # Retrieve the matched spectrograms based on the indices
# matched_ground_truth = torch.cat([ground_truth_reshape[i, :] for i in indices[0]])


# # Calculate the mean squared error (MSE)
# mse = torch.mean((matched_ground_truth - matched_predicted) ** 2)

# # Calculate the mean squared error (MSE)
# print(mse.size())

def matcher(gt, pred):
    # return index of gt and pred
    batch_size, n_gt, F, T = gt.size()
    _, n_slots, _, _ = pred.size()
    idx = []    
    for i in range(batch_size):
        gt_frame = gt[i].view(n_gt, -1)
        pred_frame = pred[i].view(n_slots, -1)
        
        cost_matrix = torch.sum((gt_frame[None,:] - pred_frame[ :,None]) ** 2, dim=2)
        cost_matrix_np = cost_matrix.cpu().numpy()
        

        row,col = linear_sum_assignment(cost_matrix)
        idx_frame = torch.stack((torch.as_tensor(row,dtype=torch.int64),torch.as_tensor(col,dtype=torch.int64)), axis=1)
        idx.append(idx_frame)
        # stack indices
        

    return idx

idx = matcher(ground_truth, predicted)
print(idx[0])
# matched_ground_truth = torch.cat([ground_truth[i, :, :, :] for i in idx[0][:,0]])
# matched_predicted = torch.cat([predicted[i, :, :, :] for i in idx[0][:,1]])
# mse = torch.mean((matched_ground_truth - matched_predicted) ** 2)
