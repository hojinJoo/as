from typing import Any, List

import torch
import numpy as np
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from scipy.optimize import linear_sum_assignment
import os
import wandb
from pytorch_lightning.utilities import rank_zero_only

from utils.evaluator import SISNREvaluator
from src.utils.audio_vis import vis_compare,vis_slots,vis_attention,test_vis
from src.utils.schedular import CosineAnnealingWarmUpRestarts
from src.utils.write_wav import write_wav
from src.utils.transforms import istft

class AudioSlotModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        name: str = "audioslot",
        istft : Any = None
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        
        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_snr = SISNREvaluator()
        self.val_snr = SISNREvaluator()
        

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        
        # for tracking best so far validation accuracy
        self.train_snr_best = MaxMetric()
        self.val_snr_best = MaxMetric()

    def matcher(self, gt: torch.Tensor, pred: torch.Tensor):
        # pred: B,2,F,T
        # gt: B,2,F,T
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
        
        
    def find_pred_idx_original_gt(self,gt_idx, pred_idx) :
    # permute targets following indices
        new_gt_idx = [gt_idx[0].clone(),gt_idx[1].clone()]
        new_pred_idx = [pred_idx[0].clone(),pred_idx[1].clone()]
        
        for i,val in enumerate(gt_idx[1]) :
            if i % 2 == 0 and val != 0:
                new_pred_idx[1][i] =pred_idx[1][i+1] 
                new_pred_idx[1][i+1] = pred_idx[1][i]
                
                new_gt_idx[1][i] = 0
                new_gt_idx[1][i+1] = 1
        
        return new_gt_idx, new_pred_idx

    def forward(self, x: torch.Tensor,train:bool=False):
        return self.net(x,train=train)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_snr.reset()
        self.val_snr_best.reset()

    def model_step(self, batch: Any,train:bool=False):
        
        source1 = batch["source_1"]
        source2 = batch["source_2"]
        mixture = batch["source_1"] + batch["source_2"]
        gt = torch.stack((source1, source2), dim=1)
        B,n_src,F,T = gt.size()

        pred,attention = self.forward(mixture,train=train) 
        
        indices = self.matcher(gt, pred)
        
        pred_idx = self._get_src_permutation_idx(indices)
        gt_idx = self._get_tgt_permutation_idx(indices)
        
        matching_pred = pred[pred_idx].view(B,n_src,F,T).type(torch.float32)
        matching_gt = gt[gt_idx].view(B,n_src,F,T)

        
        loss = self.criterion( matching_gt,matching_pred)
        outs = {"gt" : matching_gt, "pred" : matching_pred, "attention" : attention, "pred_index" : pred_idx, "gt_index" : gt_idx}
        
        return loss,outs,pred
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx
    
    def training_step(self, batch: Any, batch_idx: int):
        loss, outs,slots = self.model_step(batch,train=True)
        # update and log metrics
        self.train_loss(loss)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        if batch_idx == 0 and self.global_rank == 0 and self.local_rank == 0:
            visualize_num = 8
            gt_vis = outs["gt"][:visualize_num].clone().detach().cpu().numpy()
            matching_pred_vis = outs["pred"][:visualize_num].clone().detach().cpu().numpy()
            slots_vis = slots[:visualize_num].clone().detach().cpu().numpy()
            attention_vis = outs["attention"][:visualize_num].clone().detach().cpu().numpy()
            
            os.makedirs(os.path.join(self.logger.save_dir,str(self.current_epoch+1)),exist_ok=True)
            
            vis_compare(gt_vis,matching_pred_vis,self.logger.save_dir,str(self.current_epoch+1),outs["gt_index"],outs["pred_index"])
            vis_slots(slots_vis,self.logger.save_dir,str(self.current_epoch+1))
            vis_attention(attention_vis,self.logger.save_dir,str(self.current_epoch+1))
                
        return {"loss": loss}
    
    
    @rank_zero_only
    def _log_image(self) :
        
        gt_img = [os.path.join(self.logger.save_dir,f'gt_with_preds.png')]
        slots_img = [os.path.join(self.logger.save_dir,f'slots.png')]
        
        self.logger.log_image(key="GT and preds", images=gt_img,caption=[f"Epoch : {self.current_epoch+1} GT and preds"])
        self.logger.log_image(key="slots", images=slots_img,caption=[f"Epoch : {self.current_epoch+1} slots "])
        
        pass
            
        
    def validation_step(self, batch: Any, batch_idx: int):
        # input : complex64
        # batch : [1,F,T]
        sample = batch
        original_length = sample["original_length"]
        
        _,_,F,T = sample["mixture"].size()
        source1_original = sample["source_1"].squeeze(0)
        source2_original = sample["source_2"].squeeze(0)
        model_input = source1_original + source2_original # [1,F,T]
        
        
        n_src = 2
        segment_step = self.hparams.net.input_ft[1]
        prediction = torch.zeros(1,n_src,F,T)
        loss = 0
        
        #pre-process
        s1 = torch.pow(torch.abs(sample["source_1"].squeeze(0)),0.3) # [1,F,T]
        s2 = torch.pow(torch.abs(sample["source_2"].squeeze(0)),0.3)
        
        for segment in range(0,T,segment_step):
            # pre-process
            source1 = s1[:,:,segment:segment+segment_step]
            source2 = s2[:,:,segment:segment+segment_step]
            mixture = source1 + source2
            mixture_original_size = mixture.size()
            if mixture_original_size[2] != segment_step :
                # print("mixture size",mixture.size())
                # print(f"segment {segment_step}")
                source1 = torch.cat((source1,torch.zeros(source1.size(0),source1.size(1),segment_step-source1.size(2)).to(source1.device) ),dim=2)
                source2 = torch.cat((source2,torch.zeros(source2.size(0),source2.size(1),segment_step-source2.size(2)).to(source2.device)),dim=2)
                mixture = torch.cat((mixture,torch.zeros(mixture.size(0),mixture.size(1),segment_step-mixture.size(2)).to(mixture.device)),dim=2)
            
            loss,outs,slots = self.model_step({"mixture" : mixture, "source_1" : source1, "source_2" : source2},train=False)
            loss += loss
            pred_idx = outs["pred_index"]
            gt_idx = outs["gt_index"]
            
            sorted_gt_idx,sorted_pred_idx = self.find_pred_idx_original_gt(gt_idx, pred_idx)
            sorted_matching_pred = slots[sorted_pred_idx].unsqueeze(0)
            
            if mixture_original_size[2] != segment_step :
                prediction[:,:,:,segment:] = sorted_matching_pred[:,:,:,:mixture_original_size[2]]
            else :
                prediction[:,:,:,segment:segment+segment_step] = sorted_matching_pred

        gt = torch.stack((source1_original,source2_original),dim=1).cpu() # [1,2,F,T]
        gt = istft(gt,fs=self.hparams.istft.sample_rate,window_length=self.hparams.istft.win_length,nfft=self.hparams.istft.n_fft,hop_length=self.hparams.istft.hop_length,original_length=original_length) # [1,2,T]
        
        prediction = torch.pow(prediction,10/3)
        mask = self.ibm_mask(prediction) # [1,2,F,T]
        

        prediction =  model_input.cpu() * mask # dtype : complex64, size : [1,2,F,T]
        
        # dtype : float32 size : [1,2,T]    
        prediction = istft(prediction,fs=self.hparams.istft.sample_rate,window_length=self.hparams.istft.win_length,nfft=self.hparams.istft.n_fft,hop_length=self.hparams.istft.hop_length,original_length=original_length)
        
        
        os.makedirs(os.path.join(self.logger.save_dir,'test'),exist_ok=True)
        if self.global_rank == 0 and self.local_rank == 0 and batch_idx % 100 == 0:
            write_wav(prediction.clone().detach().cpu(),os.path.join(self.logger.save_dir,'test'),batch["mix_id"][0])
            
        
        
        
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        snr = self.val_snr.evaluate(prediction,gt)
        self.log("val/snr", snr, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        
        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        snr = self.val_snr.get_results()
        
        
        self.val_snr_best(snr)
        # self.log("val/snr", snr, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log("val/snr_best", self.val_snr_best.compute(), on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        
        
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

    
    def ibm_mask(self,sources) :
        # ideal binary mask
        # sources : B,2,F,T
        ibm = (sources == torch.max(sources, dim=1, keepdim=True).values).float()
        # ibm = ibm / torch.sum(ibm, dim=1, keepdim=True)
        # ibm[ ibm <= 0.5] = 0
        return ibm
        
    def test_step(self, batch: Any, batch_idx: int):
        # train과 다르게 [B,T,F] 형태로 들어옴
        # for 
        self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: List[Any]):
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            # scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer,T_0=self.hparams.scheduler.scheduler.T_0)
            # def lr_lambda(step):
            #     if step < self.hparams.scheduler.warmup_steps:
            #         warmup_factor = float(step) / float(
            #             max(1.0, self.hparams.scheduler.warmup_steps)
            #         )
            #     else:
            #         warmup_factor = 1.0

            #     decay_factor = self.hparams.scheduler.decay_rate ** (
            #         step / self.hparams.scheduler.decay_steps
            #     )

            #     return warmup_factor * decay_factor
            
            scheduler = self.hparams.scheduler.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None)
