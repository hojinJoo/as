import numpy as np
import torch
import torch.nn.functional as F
import wandb
import os
import matplotlib.pyplot as plt


def vis_compare(matching_gt, matching_pred,log_dir,epoch,gt_idx,pred_idx):
    """
    `gt`: (B,2,F,T)
    `pred`: (B, 4,F,T)
    gt_idx : tuple(0,1)
    """
    B, n_src, F, T = matching_gt.shape
    
    fig, axes = plt.subplots(B, 4, figsize=(12, 3*B))
    matching_pred[matching_pred < 0] =0
    
    for i in range(B):
        for j in range(n_src):
            # GT 어텐션 맵 그리기
            gt_value = 20 * np.log10(matching_gt[i,j] + 1e-8)
            gt_min = gt_value.min()
            gt_max = gt_value.max()
            
            gt = axes[i,j*2].imshow(gt_value, origin="lower", aspect="auto",vmin=gt_min,vmax=gt_max)
            axes[i, j*2].set_title(f'GT {j+1}')
            
            color_bar_gt = fig.colorbar(gt, ax=axes[i,j*2])
            
            # Prediction 어텐션 맵 그리기
            pred = axes[i, j*2+1].imshow((20 * np.log10(matching_pred[i,j] + 1e-8)), origin="lower", aspect="auto",vmin=gt_min,vmax=gt_max)
            axes[i, j*2+1].set_title(f'Slots {pred_idx[1][2*i + j]+1}')

            color_bar_pred = fig.colorbar(pred, ax=axes[i,j*2+1])
            
            # 축 숨기기
            axes[i, j*2].axis('off')
            axes[i, j*2+1].axis('off')
            
            
    # 빈 축 숨기기
    for i in range(B, axes.shape[0]):
        for j in range(n_slots):
            axes[i, j].axis('off')
    
    
    fig.tight_layout()
    # 그림판 저장
    
    plt.savefig(os.path.join(log_dir,epoch,f'gt_with_preds.png'), dpi=300)
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure


def test_vis(matching_gt, matching_pred,log_dir,dir_name,fname) :
    """
    `gt`: (1,2,F,T)
    `pred`: (1, 2,F,T)
    """
    _, F, T = matching_gt.shape
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3*1))
    matching_pred[matching_pred < 0] =0
    
    for j in range(2):
        # GT 어텐션 맵 그리기
        gt_value = 20 * np.log10(matching_gt[j] + 1e-8)
        gt_min = gt_value.min()
        gt_max = gt_value.max()
        
        gt = axes[j*2].imshow(gt_value, origin="lower", aspect="auto",vmin=gt_min,vmax=gt_max)
        axes[ j*2].set_title(f'GT {j+1}')
        
        color_bar_gt = fig.colorbar(gt, ax=axes[j*2])
        
        # Prediction 어텐션 맵 그리기
        pred = axes[ j*2+1].imshow((20 * np.log10(matching_pred[j] + 1e-8)), origin="lower", aspect="auto",vmin=gt_min,vmax=gt_max)
        axes[ j*2+1].set_title(f'Matching pred')

        color_bar_pred = fig.colorbar(pred, ax=axes[j*2+1])
        
        # 축 숨기기
        axes[ j*2].axis('off')
        axes[ j*2+1].axis('off')
            
            
    # 빈 축 숨기기
    for j in range(4):
        axes[ j].axis('off')
    
    
    fig.tight_layout()
    # 그림판 저장
    
    plt.savefig(os.path.join(log_dir,dir_name,f'test_{fname}.png'), dpi=300)
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure


def vis_slots(all_preds,log_dir,epoch) :
    B, n_slots, F, T = all_preds.shape
    
    pred_fig, pred_axes = plt.subplots(B, n_slots, figsize=(12, 3*B))
    all_preds[all_preds < 0] =0
    # 배치별로 GT와 Prediction 어텐션 맵 그리기
    for i in range(B):
        for j in range(n_slots):
            

            
            pred = pred_axes[i,j].imshow((20 * np.log10(all_preds[i,j] + 1e-8)), origin="lower", aspect="auto",vmin=-160,vmax=10)
            pred_axes[i, j].set_title(f'Slot {j+1}')
            color_bar_pred = pred_fig.colorbar(pred, ax=pred_axes[i,j])
            pred_axes[i, j].axis('off')


    # 빈 축 숨기기
    for i in range(B, pred_axes.shape[0]):
        for j in range(n_slots):

            pred_axes[i, j].axis('off')
            
            
    pred_fig.tight_layout()
    # 그림판 저장
    plt.savefig(os.path.join(log_dir,epoch,f'slots.png'), dpi=300)
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure


def vis_attention(attn,log_dir,epoch) :
    """
    attn : (B,N_heads,N_in,N_slots) => 16,4,256,7
    """
    B, N_heads, N_in, N_slots = attn.shape
    attn_mean = attn.mean(axis=1).transpose(0,2,1).reshape(B,N_slots,33,9) # (B,N_in,32,8)
    
    attn_fig, attn_axes = plt.subplots(B, N_slots, figsize=(12, 3*B))

    for i in range(B):
        for j in range(N_slots):
            img = attn_mean[i,j]
            attn_axes[i,j].imshow(img)
            attn_axes[i, j].set_title(f'Slot {j+1}')
            attn_axes[i, j].axis('off')
            
    attn_fig.tight_layout()
    # 그림판 저장
    plt.savefig(os.path.join(log_dir,epoch,f'slots_attn.png'), dpi=300)
    plt.cla()   # clear the current axes
    plt.clf()   # clear the current figure
    plt.close() # closes the current figure     
    
            
    