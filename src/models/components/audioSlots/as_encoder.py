import torch
import torch.nn as nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck
import copy
from typing import Optional, List

import torch.nn.functional as F
from torch import  Tensor




class SlotAttention(nn.Module):
    """Slot Attention module.

    Args:
        num_slots: int - Number of slots in Slot Attention.
        iterations: int - Number of iterations in Slot Attention.
        num_attn_heads: int - Number of multi-head attention in Slot Attention,
    """

    def __init__(
        self,
        num_slots: int = 3,
        num_iterations: int = 6,
        num_attn_heads: int = 2,
        slot_dim: int = 1024,
        hid_dim: int = 512,
        mlp_hid_dim: int = 2048,
        eps: float = 1e-8,
        init_method : str = 'clustering'
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.num_attn_heads = num_attn_heads
        self.slot_dim = slot_dim
        self.hid_dim = hid_dim
        self.mlp_hid_dim = mlp_hid_dim
        self.eps = eps
        self.init_method = init_method
        self.scale = (num_slots // num_attn_heads) ** -0.5

        if self.init_method == 'clustering':
            self.slot_init = SlotInit(num_slots=num_slots,num_cluster=2,slot_dim=slot_dim)
            
        elif self.init_method == 'random':
            self.slots_mu = nn.Parameter(torch.rand(1, 1, self.slot_dim))
            self.slots_sigma = nn.Parameter(torch.rand(1, 1, self.slot_dim))
        
        elif self.init_method == 'learnable':
            self.slots = nn.Parameter(torch.rand(1, 1, self.slot_dim))

        self.norm_input = nn.LayerNorm(self.hid_dim)
        self.norm_slot = nn.LayerNorm(self.slot_dim)
        self.norm_mlp = nn.LayerNorm(self.slot_dim)

        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k = nn.Linear(self.hid_dim, self.slot_dim)
        self.to_v = nn.Linear(self.hid_dim, self.slot_dim)

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, self.mlp_hid_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hid_dim, self.slot_dim),
        )

    def forward(self, inputs,cluster=None, num_slots=None, train=False):
        outputs = dict()
        B, N_in, D_in = inputs.shape
        K = num_slots if num_slots is not None else self.num_slots
        D_slot = self.slot_dim
        N_heads = self.num_attn_heads

        if self.init_method == 'clustering':
            slots = self.slot_init(cluster)
            inputs_slots = self.norm_input(torch.cat([inputs,slots],dim=1)) # B N_in + N_slots D_in
            inputs,slots = inputs_slots[:,:N_in,:],inputs_slots[:,N_in:,:]
            
        elif self.init_method == 'learnable':
            slots = self.slots.expand(B,K,-1)
            inputs = self.norm_input(inputs)
        elif self.init_method == 'random':
            mu = self.slots_mu.expand(B, K, -1)
            sigma = self.slots_sigma.expand(B, K, -1)
            slots = torch.normal(mu, torch.abs(sigma) + self.eps)
            inputs = self.norm_input(inputs)

        k = self.to_k(inputs).reshape(B, N_in, N_heads, -1).transpose(1, 2)
        v = self.to_v(inputs).reshape(B, N_in, N_heads, -1).transpose(1, 2)
        # k, v: (B, N_heads, N_in, D_slot // N_heads).
        if not train:
            attns = list()

        for iter_idx in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slot(slots)

            q = self.to_q(slots).reshape(B, K, N_heads, -1).transpose(1, 2)
            # q: (B, N_heads, K, slot_D // N_heads)
            # K : number of slots

            attn_logits = torch.einsum("bhid, bhjd->bhij", k, q) * self.scale

            attn = attn_logits.softmax(dim=-1) + self.eps  # Normalization over slots
            # attn: (B, N_heads, N_in, K)

            if not train:
                attns.append(attn)

            attn = attn / torch.sum(attn, dim=-2, keepdim=True)  # Weighted mean
            # attn: (B, N_heads, N_in, K)
            updates = torch.einsum("bhij,bhid->bhjd", attn, v)
            # updates: (B, N_heads, K, slot_D // N_heads)
            updates = updates.transpose(1, 2).reshape(B, K, -1)
            # updates: (B, K, slot_D)

            slots = self.gru(updates.reshape(-1, D_slot), slots_prev.reshape(-1, D_slot))
            slots = slots.reshape(B, -1, D_slot)
            slots = slots + self.mlp(self.norm_mlp(slots))
        outputs["slots"] = slots
        outputs["attn"] = attn
        
        if not train:
            outputs["attns"] = torch.stack(attns, dim=1)
            # attns: (B, T, N_heads, N_in, K)
        return outputs
    
class SlotInit(nn.Module) :
    def __init__(self,num_slots,num_cluster,slot_dim) :
        super().__init__()
        self.num_slots = num_slots
        self.num_cluster = num_cluster
        self.slot_dim = slot_dim
        self.mapping_to_slots = nn.Sequential(
            nn.Linear(self.num_cluster, self.num_cluster * slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_cluster * slot_dim, (self.num_cluster + self.num_slots) // 2 * slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear((self.num_cluster + self.num_slots) // 2 * slot_dim, self.num_slots * slot_dim),
        )
    def forward(self,x) :
        # out size B n_slots slot_dim
        B,Fr,T,C,n_cluster = x.size()
        x = x.mean(dim=3) # B Fr T n_cluster
        x = x.flatten(1,2)
        x = x.mean(dim=1) # B n_cluster
        x = x.to(torch.float32)
        x = self.mapping_to_slots(x) 
        x = x.reshape(B,self.num_slots,self.slot_dim)
        
        return x
        
        
        
if __name__ == "__main__" :
    sample_wav = torch.randn(8000,device='cpu')
    after_stft = torch.stft(sample_wav, n_fft=512, win_length=512,
                            hop_length=125, return_complex=True)
    after_stft = torch.abs(after_stft)
    sample = after_stft.unsqueeze(0).repeat((16,  1, 1))
    a = nn.LayerNorm(512)
    model = Backbone(BasicBlock, [3, 4, 6, 3])
    after_model = model(sample)
    sa = SlotAttention()
    result = sa(after_model,train=True)
    print(result['slots'].shape)
    print(result['attn'].shape)
    

    
#output size : [64,4,512]