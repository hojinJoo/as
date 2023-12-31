import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.special import comb
from torchmetrics import ScaleInvariantSignalNoiseRatio

class SISNREvaluator :
    def __init__(self) :
        self.SI_SNR = []
        self.metric = ScaleInvariantSignalNoiseRatio()
    def evaluate(self,pred,gt) :
        """_summary_

        Arguments:
            pred -- (B,2,F,T)
            gt -- (B,2,F,T)

        Returns:
            (SI_SNR)
        """
        self.metric.to(pred.device)
        snr = self.metric(pred,gt)
        self.SI_SNR.append(snr)
        
        return snr
    def reset(self) :
        self.SI_SNR = []
    
    def get_results(self) :
        self.SI_SNR = [i.cpu() for i in self.SI_SNR]
        return np.mean(self.SI_SNR) if self.SI_SNR != [] else 0
        
class SDREvaluator :
    def __init__(self) :
        self.vocal_SDR = []
        self.accom_SDR = []
        self.total_SDR = []
   
    def get_sdr(self,references, estimates):
        """ 
        Compute the SDR according to the MDX challenge definition.
        Adapted from AIcrowd/music-demixing-challenge-starter-kit (MIT license)
        """
        references = references[None].transpose(1,2).double()
        estimates = estimates[None].transpose(1,2).double()
        assert references.dim() == 4
        assert estimates.dim() == 4
        delta = 1e-7  # avoid numerical errors
        num = torch.sum(torch.square(references), dim=(2, 3))
        den = torch.sum(torch.square(references - estimates), dim=(2, 3))
        num += delta
        den += delta
        scores = 10 * torch.log10(num / den)
        return scores[0]
    
    def evaluate(self,pred,gt) :
        """_summary_

        Arguments:
            pred -- (B,2,F,T)
            gt -- (B,2,F,T)

        Returns:
            (SI_SNR)
        """
        snr = self.get_sdr(gt,pred)
        self.vocal_SDR.append(snr[0])
        self.accom_SDR.append(snr[1])
        self.total_SDR.append(torch.mean(snr))
        
        return snr
    def reset(self) :
        self.vocal_SDR = []
        self.accom_SDR = []
        self.total_SDR = []
    
    def get_results(self) :
        self.vocal_SDR = [i.cpu() for i in self.vocal_SDR]
        self.accom_SDR = [i.cpu() for i in self.accom_SDR]
        self.total_SDR = [i.cpu() for i in self.total_SDR]
        ret = { "vocal" : np.mean(self.vocal_SDR) if self.vocal_SDR != [] else 0,
                "accom" : np.mean(self.accom_SDR) if self.accom_SDR != [] else 0,
                "total" : np.mean(self.total_SDR) if self.total_SDR != [] else 0}
        print(ret)
        return ret
        


class ARIEvaluator:
    """"""

    def __init__(self):
        self.aris = []

    def evaluate(self, pred, label):
        """
        :param data: (image, mask)
            image: (B, 3, H, W)
            pred : (B, N0, H, W)
            label: (B, N1, H, W)
        :return: average ari
        """

        B, K, H, W = pred.size()

        # reduced to (B, K, H, W), with 1-0 values

        # max_index (B, H, W)
        max_index = torch.argmax(pred, dim=1)
        # get binarized masks (B, K, H, W)
        pred = torch.zeros_like(pred)
        pred[
            torch.arange(B)[:, None, None],
            max_index,
            torch.arange(H)[None, :, None],
            torch.arange(W)[None, None, :],
        ] = 1.0

        for b in range(B):
            this_ari = self.compute_mask_ari(label[b].to(pred.device), pred[b].to(pred.device))
            self.aris.append(this_ari)

    def reset(self):
        self.aris = []

    def get_results(self):
        return np.mean(self.aris) if self.aris != [] else 0

    def compute_ari(self, table):
        """Compute ari, given the index table.

        :param table: (r, s)
        :return:
        """

        # # (r,)
        # a = table.sum(axis=1)
        # # (s,)
        # b = table.sum(axis=0)
        # n = a.sum()
        # (r,)
        a = table.sum(dim=1)
        # (s,)
        b = table.sum(dim=0)
        n = a.sum()

        comb_a = comb(a.detach().cpu().numpy(), 2).sum()
        comb_b = comb(b.detach().cpu().numpy(), 2).sum()
        comb_n = comb(n.detach().cpu().numpy(), 2)
        comb_table = comb(table.detach().cpu().numpy(), 2).sum()

        if comb_b == comb_a == comb_n == comb_table:
            # the perfect case
            ari = 1.0
        else:
            ari = (comb_table - comb_a * comb_b / comb_n) / (
                0.5 * (comb_a + comb_b) - (comb_a * comb_b) / comb_n
            )

        return ari

    def compute_mask_ari(self, mask0, mask1):
        """Given two sets of masks, compute ari.

        :param mask0: ground truth mask, (N0, H, W)
        :param mask1: predicted mask, (N1, H, W)
        :return:
        """

        # will first need to compute a table of shape (N0, N1)
        # (N0, 1, H, W)
        mask0 = mask0[:, None].byte()
        # (1, N1, H, W)
        mask1 = mask1[None, :].byte()
        # (N0, N1, H, W)
        agree = mask0 & mask1
        # (N0, N1)
        table = agree.sum(dim=-1).sum(dim=-1)

        return self.compute_ari(table)


class mIoUEvaluator:
    def __init__(self):
        self.mious = []

    def evaluate(self, pred, label):
        """
        :param data: (image, mask)
            image: (B, 3, H, W)
            pred : (B, N0, H, W)
            label: (B, N1, H, W)
        :return: average miou
        """
        from torch import arange as ar

        B, K, H, W = pred.size()

        # reduced to (B, K, H, W), with 1-0 values

        # max_index (B, H, W)
        max_index = torch.argmax(pred, dim=1)
        # get binarized masks (B, K, H, W)
        pred = torch.zeros_like(pred)
        pred[ar(B)[:, None, None], max_index, ar(H)[None, :, None], ar(W)[None, None, :]] = 1.0

        for b in range(B):
            this_miou = self.compute_miou(label[b].to(pred.device), pred[b].to(pred.device))
            self.mious.append(this_miou)

    def reset(self):
        self.mious = []

    def get_results(self):
        return np.mean(self.mious) if self.mious != [] else 0

    def compute_miou(self, mask0, mask1):
        """
        mask0: [N0, H, W]
        mask1: [N1, H, W]
        """

        mask0 = mask0.reshape(mask0.shape[0], -1)[:, None, :]  # [N0, 1, H*W]
        mask1 = mask1.reshape(mask1.shape[0], -1)[None, :, :]  # [1, N1, H*W]

        union = torch.sum(torch.clip(mask0 + mask1, 0, 1), dim=-1)  # [1, N0, N1]
        intersection = torch.sum(mask0 * mask1, dim=-1)  # [1, N0, N1]
        iou = intersection / (union + 1e-8)  # [N0, N1]

        row_ind, col_ind = linear_sum_assignment(iou.cpu().detach().numpy() * -1)
        miou = iou[row_ind, col_ind].mean().cpu().detach().numpy()

        return miou
