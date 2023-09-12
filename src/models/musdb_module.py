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
from src.utils.transforms import istft
from src.utils.write_wav import write_wav
import museval

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
        name: str = "musdb",
        sources : List[str] = ["vocals", "drums", "bass", "other"],
        cac : bool = False,
        istft : Any = None
        
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.cac = cac
        self.net = net
        self.sources = sources
        # loss function
        self.criterion = torch.nn.MSELoss()
        self.recon_loss = torch.nn.MSELoss()
        
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
        if self.cac :
            B,n_gt,C,F,T,_ = gt.size()
            # print(f"gt size : {gt.size()}")
            _, n_slots, _, _,_,_ = pred.size()
        else : 
            B,n_gt,C,F,T = gt.size()
            # print(f"gt size : {gt.size()}")
            _, n_slots, _, _,_ = pred.size()
        # print(f"pred size : {pred.size()}")
        # print(f"gt size  : {gt.size()}")
        # print(f"pred size : {pred.size()}")
        idx = []    
        for i in range(B):
            
            gt_frame = gt[i].reshape(n_gt, -1)
            pred_frame = pred[i].reshape(n_slots, -1)
            
            
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
        # C : 2
        accompanient = torch.sum(torch.stack([batch[source] for source in self.sources if source != 'vocals'], dim=1),dim=1) # [B,C,F,T]
        vocal = batch['vocals'] # [B,C,F,T]
        gt = torch.stack((vocal,accompanient),dim=1) # [B,2,C,F,T]
        
        if self.cac : 
            
            gt = torch.view_as_real(gt)
            B,n_src,C,F,T,_ = gt.size()
        else :
            gt = torch.abs(gt)
            B,n_src,C,F,T = gt.size()
        
        mixture = accompanient + vocal        
        
        
        
        
        pred,attention = self.forward(mixture,train=train) 
        indices = self.matcher(gt, pred)
        
        pred_idx = self._get_src_permutation_idx(indices)
        gt_idx = self._get_tgt_permutation_idx(indices)
        if self.cac :
            matching_pred = pred[pred_idx].view(B,n_src,C,F,T,2).type(torch.float32)
            matching_gt = gt[gt_idx].view(B,n_src,C,F,T,2)
            loss = self.criterion( matching_gt,matching_pred) +  self.recon_loss(torch.view_as_real(mixture),torch.sum(matching_pred,dim=1))
        else :
            matching_pred = pred[pred_idx].view(B,n_src,C,F,T).type(torch.float32)
            matching_gt = gt[gt_idx].view(B,n_src,C,F,T)
            loss = self.criterion( matching_gt,matching_pred) +  self.recon_loss(torch.abs(mixture),torch.sum(matching_pred,dim=1))
        
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
            gt_vis = outs["gt"][:visualize_num].clone().detach().cpu()
            matching_pred_vis = outs["pred"][:visualize_num].clone().detach().cpu()
            
            
            os.makedirs(os.path.join(self.logger.save_dir,str(self.current_epoch+1)),exist_ok=True)
            
            vis_compare(gt_vis,matching_pred_vis,self.logger.save_dir,str(self.current_epoch+1),outs["gt_index"],outs["pred_index"],self.cac)
            
            # vis_attention(attention_vis,self.logger.save_dir,str(self.current_epoch+1))
                
        return {"loss": loss}
    
    
    @rank_zero_only
    def _log_image(self) :
        
        gt_img = [os.path.join(self.logger.save_dir,f'gt_with_preds.png')]
        slots_img = [os.path.join(self.logger.save_dir,f'slots.png')]
        
        self.logger.log_image(key="GT and preds", images=gt_img,caption=[f"Epoch : {self.current_epoch+1} GT and preds"])
        self.logger.log_image(key="slots", images=slots_img,caption=[f"Epoch : {self.current_epoch+1} slots "])
        
        pass
            
        
    def val_cac(self,batch,batch_idx) :
        B,C,Fr,T = batch["vocals"].size() # 1 C Fr T
        sample = {k : batch[k] for k in self.sources}
        n_src = 2
        segment_step = self.hparams.net.input_ft[1]
        prediction = torch.zeros(1,n_src,C,Fr,T,2)
        loss = 0
        for segment in range(0,T,segment_step):
            model_input = {}
            for key, value in sample.items():
                model_input[key] = value[:,:,:,segment:segment+segment_step]

            inputs_original = model_input['vocals'].size()
            if inputs_original[3] != segment_step :
                # print("mixture size",mixture.size())
                # print(f"segment {segment_step}")
                for key, value in model_input.items():
                    model_input[key] = torch.cat((value,torch.zeros(value.size(0),value.size(1),value.size(2),segment_step-value.size(3)).to(value.device)),dim=-1)
                    
            
            loss,outs,slots = self.model_step(model_input,train=False)
            loss += loss
            pred_idx = outs["pred_index"]
            gt_idx = outs["gt_index"]
            
            sorted_gt_idx,sorted_pred_idx = self.find_pred_idx_original_gt(gt_idx, pred_idx)
            sorted_matching_pred = slots[sorted_pred_idx].unsqueeze(0) # 1,n_src,C,Fr,T,2
            
        
            if inputs_original[3] != segment_step :
                prediction[:,:,:,:,segment:,:] = sorted_matching_pred[:,:,:,:,:inputs_original[3],:]
            else :
                prediction[:,:,:,:,segment:segment+segment_step,:] = sorted_matching_pred

        accompanient = torch.sum(torch.stack([sample[source] for source in self.sources if source != 'vocals'], dim=1),dim=1) # [C,F,T]
        vocal = sample['vocals'] # [B,C,F,T]
        
        mask = self.ibm_mask(prediction).to(vocal.device)
  
        model_input = torch.view_as_real(vocal + accompanient)
        prediction =  model_input * mask
        del model_input
        # update and log metrics
        prediction = istft(torch.view_as_complex(prediction[...,:int(T/2),:].cpu()),nfft=self.hparams.istft.n_fft,hop_length=self.hparams.istft.hop_length,original_length=batch['original_length'])

        gt = torch.stack((vocal,accompanient),dim=1) # [1,2,F,T]
        gt = istft(gt[...,:int(T/2)].cpu(),nfft=self.hparams.istft.n_fft,hop_length=self.hparams.istft.hop_length,original_length=batch['original_length']) # [1,2,T]
        
        del sample
        os.makedirs(os.path.join(self.logger.save_dir,'test'),exist_ok=True)
        
        
        (sdr,isr,sir,sar) = museval.evaluate(gt.permute(0,1,3,2).squeeze(0),prediction.permute(0,1,3,2).squeeze(0))
        print(f"sdr : {np.mean(sdr)}")
        
        if self.global_rank == 0 and self.local_rank == 0 and batch_idx % 100 == 0:
            
            write_wav(prediction,os.path.join(self.logger.save_dir,'test'),batch["name"][0],self.hparams.istft.sample_rate)
            
        

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        snr = self.val_snr.evaluate(prediction,gt)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.log("val/snr", snr, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        return {"loss": loss}
    
    def val_base(self,batch,batch_idx) :
        B,C,Fr,T = batch["vocals"].size() # 1 C Fr T
        sample = {k : batch[k] for k in self.sources}
        n_src = 2
        segment_step = self.hparams.net.input_ft[1]
        prediction = torch.zeros(1,n_src,C,Fr,T)
        loss = 0
        for segment in range(0,T,segment_step):
            model_input = {}
            for key, value in sample.items():
                model_input[key] = value[:,:,:,segment:segment+segment_step]

            inputs_original = model_input['vocals'].size()
            if inputs_original[3] != segment_step :
                # print("mixture size",mixture.size())
                # print(f"segment {segment_step}")
                for key, value in model_input.items():
                    model_input[key] = torch.cat((value,torch.zeros(value.size(0),value.size(1),value.size(2),segment_step-value.size(3)).to(value.device)),dim=-1)
                    
            
            loss,outs,slots = self.model_step(model_input,train=False)
            loss += loss
            pred_idx = outs["pred_index"]
            gt_idx = outs["gt_index"]
            
            sorted_gt_idx,sorted_pred_idx = self.find_pred_idx_original_gt(gt_idx, pred_idx)
            sorted_matching_pred = slots[sorted_pred_idx].unsqueeze(0) # 1,n_src,C,Fr,T,2
            
        
            if inputs_original[3] != segment_step :
                prediction[:,:,:,:,segment:] = sorted_matching_pred[:,:,:,:,:inputs_original[3]]
            else :
                prediction[:,:,:,:,segment:segment+segment_step] = sorted_matching_pred

        accompanient = torch.sum(torch.stack([sample[source] for source in self.sources if source != 'vocals'], dim=1),dim=1) # [C,F,T]
        vocal = sample['vocals'] # [B,C,F,T]
        mask = self.ibm_mask(prediction).to(vocal.device)
        del sample
        
        model_input = (vocal + accompanient)
        prediction =  model_input * mask
        # update and log metrics
        prediction = istft(prediction[...,:int(T/2)].cpu(),fs=self.hparams.istft.sample_rate,window_length=self.hparams.istft.win_length,nfft=self.hparams.istft.n_fft,hop_length=self.hparams.istft.hop_length,original_length=int(batch['original_length']/2))
        gt = torch.stack((vocal,accompanient),dim=1) # [1,2,F,T]
        gt = istft(gt[...,:int(T/2)].cpu(),fs=self.hparams.istft.sample_rate,window_length=self.hparams.istft.win_length,nfft=self.hparams.istft.n_fft,hop_length=self.hparams.istft.hop_length,original_length=int(batch['original_length']/2)) # [1,2,T]

        
        
        os.makedirs(os.path.join(self.logger.save_dir,'test'),exist_ok=True)
            
        

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        snr = self.val_snr.evaluate(prediction,gt)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        if self.global_rank != 0  :
            del prediction
            del gt
            
        self.log("val/snr", snr, on_step=True, on_epoch=True, prog_bar=True,sync_dist=True)
        if self.global_rank == 0 and self.local_rank == 0 and batch_idx % 100 == 0:
            write_wav(prediction,os.path.join(self.logger.save_dir,'test'),batch["name"][0],self.hparams.istft.sample_rate)
        
        return {"loss": loss}
    def validation_step(self, batch: Any, batch_idx: int):
        # batch : [1,F,T]
        # print(batch["mixture"].size())
        # print(f"source {batch['source_1'].size()}")
        return
        # if self.cac :
        #     return self.val_cac(batch,batch_idx)
        # else :
        #     return self.val_base(batch,batch_idx)

    def validation_epoch_end(self, outputs: List[Any]):
        return
        # snr = self.val_snr.get_results()
        # self.val_snr.reset()
        
        # self.val_snr_best(snr)
        # self.log("val/snr", snr, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        # self.log("val/snr_best", self.val_snr_best.compute(), on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        
        # 여기까지
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

    
    def ibm_mask(self,sources) :
        # ideal binary mask
        # sources : B,2,F,T
        ibm = (sources == torch.max(sources, dim=1, keepdim=True).values).float()
        ibm = ibm / torch.sum(ibm, dim=1, keepdim=True)
        ibm[ ibm <= 0.5] = 0
        return ibm
        
    def test_step(self, batch: Any, batch_idx: int):
        # train과 다르게 [B,T,F] 형태로 들어옴
        # for 
        self.validation_step(batch, batch_idx)
        return
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
            scheduler = self.hparams.scheduler.scheduler(optimizer=optimizer)
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
            
            # scheduler = self.hparams.scheduler.scheduler(optimizer=optimizer,gamma=self.hparams.scheduler.gamma)
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
