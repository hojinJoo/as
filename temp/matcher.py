import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
def matcher( gt: torch.Tensor, pred: torch.Tensor):
    # pred: B,2,F,T
    # gt: B,2,F,T
    # example로 잘 되는지 확인
    batch_size, n_gt, F, T = gt.size()
    _, n_slots, _, _ = pred.size()
    idx = []    
    for i in range(batch_size):
        gt_frame = gt[i].view(n_gt, -1)
        pred_frame = pred[i].view(n_slots, -1)

        cost_matrix = torch.sum((gt_frame[None,:] - pred_frame[ :,None]) ** 2, dim=2)
        cost_matrix_np = cost_matrix.detach().cpu().numpy()
        nan = np.isnan(cost_matrix_np).any()
        if nan :
            cost_matrix_np[np.isnan(cost_matrix_np)] = 1e+5
        try :
            row,col = linear_sum_assignment(cost_matrix_np)
            idx_frame = torch.stack((torch.as_tensor(row,dtype=torch.int64),torch.as_tensor(col,dtype=torch.int64)), axis=0)
            idx.append(idx_frame)
        except ValueError:
            print(cost_matrix_np)
            continue
        # stack indices


    return idx

def get_src_permutation_idx( indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
    
def get_tgt_permutation_idx( indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


B =2
F = 2
T = 3

gt = torch.rand(B,2,F,T)
gt[0][0] = torch.rand(F,T)
gt[0][1] = torch.rand(F,T) * 10

gt[1][0] = torch.rand(F,T) * 40
gt[1][1] = torch.rand(F,T) * 80

pred = torch.rand(B,2,F,T)

pred[0][0] = torch.rand(F,T) * 10
pred[0][1] = torch.rand(F,T) 

pred[1][0] = torch.rand(F,T) * 80
pred[1][1] = torch.rand(F,T) * 40



indices = matcher(gt, pred)

pred_idx  = get_src_permutation_idx(indices)
gt_idx = get_tgt_permutation_idx(indices)

print(gt_idx)
print(pred_idx)

print(gt[gt_idx])
print(pred[pred_idx])
        
        
