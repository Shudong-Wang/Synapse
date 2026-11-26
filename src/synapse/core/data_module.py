import copy
from functools import cached_property

import lightning as L
import torch

from .config import DataConfig, RunConfig
from .dataset import MapStyleDataset #, HybridDataset

class DataModule(L.LightningDataModule):
    def __init__(
            self,
            data_cfg: DataConfig,
            run_cfg: RunConfig,
            train_file_list: list[str] | None = None,
            val_file_list: list[str] | None = None,
            test_file_list: list[str] | None = None,
    ):
        super().__init__()
        self.val_dataset = None
        self.test_dataset = None
        self.train_file_list = train_file_list
        self.val_file_list = val_file_list
        self.test_file_list = test_file_list
        self.run_cfg = run_cfg
        self.train_data_cfg = copy.deepcopy(data_cfg)
        self.val_data_cfg = copy.deepcopy(data_cfg)
        self.test_data_cfg = copy.deepcopy(data_cfg)

        if run_cfg.cross_validation and run_cfg.cross_validation_var and data_cfg.train_selection:
            self.train_data_cfg.selection = data_cfg.train_selection
            self.val_data_cfg.selection = data_cfg.val_selection
            self.test_data_cfg.selection = data_cfg.test_selection

    @cached_property
    def train_dataset(self):
        # A checkpoint with saved input statistics can score a test file on its own.
        return MapStyleDataset(
            self.train_file_list,
            'train',
            self.train_data_cfg
        )

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.val_dataset = MapStyleDataset(
                self.val_file_list,
                'val',
                self.val_data_cfg
            )
        if stage == "test" or stage is None:
            self.test_dataset = MapStyleDataset(
                self.test_file_list,
                'test',
                self.test_data_cfg
            )

    def _loader(self, dataset, shuffle):
        workers = min(self.run_cfg.num_workers, len(dataset))
        if self.run_cfg.get("array_batches", False):
            sampler = torch.utils.data.RandomSampler(dataset) if shuffle else torch.utils.data.SequentialSampler(dataset)
            batches = torch.utils.data.BatchSampler(sampler, self.run_cfg.batch_size, drop_last=False)
            # Advanced indexing gathers all columns in one call, retaining the sampler's event order.
            return torch.utils.data.DataLoader(dataset, batch_size=None, sampler=batches, num_workers=workers)
        return torch.utils.data.DataLoader(dataset, batch_size=self.run_cfg.batch_size, num_workers=workers,
                                           shuffle=shuffle, pin_memory=False, persistent_workers=False)

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, True)

    def test_dataloader(self):
        return self._loader(self.test_dataset, False)
