from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import musdb
import json
import os
from src.data.components.musdb_dataset import MusDB


class MusDBDataModule(LightningDataModule):
    """LightningDataModule for CLEVR6 dataset.

    A DataModule implements 5 key methods:

        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        name: str = "musdb",
        metadata_path: str = "/media/ssd1/users/hj/musdb18hq/musdb_wav.json",
        segment : int = 5,
        batch_size: int = 64,
        sample_rate : int = 44100,
        sources : [str] = ["vocals", "drums", "bass", "other"],
        num_workers: int = 12,
        pin_memory: bool = False,
        normalize : bool = False,
        n_fft : int = 4096,
        win_length : int = 1024,
        hop_length : int = 1024
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.segment = segment
        self.metadata_path = metadata_path
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.sources = sources
        self.metadata = json.load(open(metadata_path, 'r'))
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.num_workers = num_workers
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        
        
    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        import yaml
        setup_path = os.path.join(musdb.__path__[0], 'configs','mus.yaml')
        setup = yaml.safe_load(open(setup_path, 'r'))
        valid_tracks = setup['validation_tracks']
        
        metadata_train = {name : meta for name, meta in self.metadata.items() if name not in valid_tracks}
        metadata_val = {name : meta for name, meta in self.metadata.items() if name in valid_tracks}
        
        
        self.data_train = MusDB(
            metadata=metadata_train,
            segment=self.segment,
            sources=self.sources,
            sample_rate=self.sample_rate,
            normalize=self.normalize,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            mode="train"
        )
        self.data_val = MusDB(
            metadata=metadata_val,
            segment=self.segment,
            sources=self.sources,
            sample_rate=self.sample_rate,
            normalize=self.normalize,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            mode="val"
        )
        
        

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True

        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            persistent_workers=True
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    test = MusDBDataModule()
    # print(test.data_train[0])