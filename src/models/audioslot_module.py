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
from src.utils.audio_vis import vis_compare,vis_slots,vis_attention
from src.utils.schedular import CosineAnnealingWarmUpRestarts

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
        name: str = "audioslot"
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
        
        
        
    def forward(self, x: torch.Tensor,train:bool=False):
        return self.net(x,train=train)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_snr.reset()
        self.val_snr_best.reset()

    def model_step(self, batch: Any,train:bool=False):
        mixture = batch["mixture"]
        source1 = batch["source_1"]
        source2 = batch["source_2"]
        gt = torch.stack((source1, source2), dim=1)
        B,n_src,F,T = gt.size()

        pred,attention = self.forward(mixture,train=train) # B,7,F,T
        
        indices = self.matcher(gt, pred)
        
        pred_idx = self._get_src_permutation_idx(indices)
        gt_idx = self._get_tgt_permutation_idx(indices)
        
        matching_pred = pred[pred_idx].view(B,n_src,F,T).type(torch.float32)
        matching_gt = gt[gt_idx].view(B,n_src,F,T)
        # print(f"train : {train} prediction {matching_pred.requires_grad}")
        
        # print(f"preidx size : {pred_idx}")
        # print(f"pred idx : {pred_idx[0]}")
        # print(f"gt idx : {gt_idx[0]}")
        # print(f"prediction max : {matching_pred.max()} min : {matching_pred.min()} mean : {matching_pred.mean()} std : {matching_pred.std()} median : {matching_pred.median()} percent over average : {100*((matching_pred > matching_pred.mean()).sum())/matching_pred.numel()}")
        # print(f"gt max : {matching_gt.max()} min : {matching_gt.min()} mean : {matching_gt.mean()} std : {matching_gt.std()} median : {matching_gt.median()} percent over average : { 100*((matching_gt > matching_gt.mean()).sum())/matching_gt.numel()}")
        
        
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
                
            # gt_img = [wandb.Image(os.path.join(self.logger.save_dir,str(self.current_epoch+1),f'gt_with_preds.png'), caption=f"Epoch : {self.current_epoch+1} GT and preds ")]
            # slots_img = [wandb.Image(os.path.join(self.logger.save_dir,str(self.current_epoch+1),f'slots.png'),caption=f"Epoch : {self.current_epoch+1} slots ")]
            # attention_img = [wandb.Image(os.path.join(self.logger.save_dir,str(self.current_epoch+1),f'slots_attn.png'),caption=f"Epoch : {self.current_epoch+1} attention ")]

            # self.logger.log_image(key="GT and preds", images=gt_img)
            # self.logger.log_image(key="slots", images=slots_img)
            # self.logger.log_image(key="attention", images=attention_img)
            
            
            
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}
    
    
    @rank_zero_only
    def _log_image(self) :
        
        gt_img = [os.path.join(self.logger.save_dir,f'gt_with_preds.png')]
        slots_img = [os.path.join(self.logger.save_dir,f'slots.png')]
        
        self.logger.log_image(key="GT and preds", images=gt_img,caption=[f"Epoch : {self.current_epoch+1} GT and preds"])
        self.logger.log_image(key="slots", images=slots_img,caption=[f"Epoch : {self.current_epoch+1} slots "])
        
        pass
        
    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs
        pass
    # def on_train_epoch_end(self):
    #     # 이거 해보기
    #     # if self.global_rank == 0:
    #     #     gt_img = [wandb.Image(os.path.join(self.logger.save_dir,f'gt_with_preds_{str(self.global_rank)}.png'), caption=f"Epoch : {self.current_epoch+1} GT and preds rank : {self.global_rank}")]
    #     #     slots_img = [wandb.Image(os.path.join(self.logger.save_dir,f'slots_{str(self.global_rank)}.png'),caption=f"Epoch : {self.current_epoch+1} slots rank : {self.global_rank}")]
            
    #     #     try : 
    #     #         self.logger.log_image(key="GT and preds", images=gt_img)
    #     #         self.logger.log_image(key="slots", images=slots_img)
    #     #     except Exception as e:
    #     #         print(e)
    #     #     pass
    #     num_gpus = os.environ["NVIDIA_VISIBLE_DEVICES"]
    #     for i in range(len(num_gpus.split(","))):
            
    #         gt_img = [wandb.Image(os.path.join(self.logger.save_dir,f'gt_with_preds_{str(i)}.png'), caption=f"Epoch : {self.current_epoch+1} GT and preds rank : {i}")]
    #         slots_img = [wandb.Image(os.path.join(self.logger.save_dir,f'slots_{str(i)}.png'),caption=f"Epoch : {self.current_epoch+1} slots rank : {i}")]
            
    #         try : 
    #             self.logger.log_image(key="GT and preds", images=gt_img)
    #             self.logger.log_image(key="slots", images=slots_img)
    #         except Exception as e:
    #             print(e)
    #         pass
    # def on_training_epoch_end(self):
    #     print("TRAINIGN ONE EPOCH END")
    #     print(self.global_rank)            
    #     gt_img = [os.path.join(self.logger.save_dir,f'gt_with_preds.png')]
    #     slots_img = [os.path.join(self.logger.save_dir,f'slots.png')]
    #     self.logger.log_image(key="GT and preds", images=gt_img,caption=[f"Epoch : {self.current_epoch+1} GT and preds"])
    #     self.logger.log_image(key="slots", images=slots_img,caption=[f"Epoch : {self.current_epoch+1} slots "])
                

        
        
    def validation_step(self, batch: Any, batch_idx: int):
        loss, imgs,slots = self.model_step(batch,train=False)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_snr.evaluate(imgs["pred"],imgs['gt'])
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        snr = self.val_snr.get_results()
        self.val_snr.reset()
        
        self.val_snr_best(snr)
        
        self.log_dict(
            {
                "val/snr": snr,
                "val/snr_best": self.val_snr_best.compute(),
            },
            prog_bar=True,
        )
        
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

    def test_step(self, batch: Any, batch_idx: int):
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
            #         return (self.hparams)
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
