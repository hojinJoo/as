import numpy as np
import torch
import torch.nn.functional as F
import wandb
import os
import matplotlib.pyplot as plt


def vis_compare(matching_gt, matching_pred,log_dir,rank):
    """
    `gt`: (B,2,F,T)
    `pred`: (B, 4,F,T)
    """
    B, n_src, F, T = matching_gt.size()
    
    fig, axes = plt.subplots(B, 4, figsize=(12, 3*B))
    
    # 배치별로 GT와 Prediction 어텐션 맵 그리기
    for i in range(B):
        for j in range(2):
            # GT 어텐션 맵 그리기
            axes[i,j*2].imshow((20 * np.log10(matching_gt[i,j].detach().cpu().numpy() + 1e-8)), origin="lower", aspect="auto")
            axes[i, j*2].set_title(f'GT {j+1}')

            # Prediction 어텐션 맵 그리기
            axes[i, j*2+1].imshow((20 * np.log10(matching_pred[i,j].detach().cpu().numpy() + 1e-8)), origin="lower", aspect="auto")
            axes[i, j*2+1].set_title(f'Prediction {j+1}')

            # 축 숨기기
            axes[i, j*2].axis('off')
            axes[i, j*2+1].axis('off')
            
            
    # 빈 축 숨기기
    for i in range(B, axes.shape[0]):
        for j in range(4):
            axes[i, j].axis('off')
    
    fig.tight_layout()
    # 그림판 저장
    
    plt.savefig(os.path.join(log_dir,f'gt_with_preds.png'), dpi=300)

def vis_slots(all_preds,log_dir,rank) :
    B, n_slots, F, T = all_preds.size()
    
    pred_fig, pred_axes = plt.subplots(B, 4, figsize=(12, 3*B))

    # 배치별로 GT와 Prediction 어텐션 맵 그리기
    for i in range(B):
        for j in range(n_slots):
            
            
            pred_axes[i,j].imshow((20 * np.log10(all_preds[i,j].detach().cpu().numpy() + 1e-8)), origin="lower", aspect="auto")
            pred_axes[i, j].set_title(f'Slot {j+1}')
            pred_axes[i, j].axis('off')

    # 빈 축 숨기기
    for i in range(B, pred_axes.shape[0]):
        for j in range(4):

            pred_axes[i, j].axis('off')
            
            
    pred_fig.tight_layout()
    # 그림판 저장
    plt.savefig(os.path.join(log_dir,f'slots.png'), dpi=300)
    
