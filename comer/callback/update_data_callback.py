from pytorch_lightning.callbacks import Callback
from comer.datamodule.dataset import CROHMEDataset


class CurriculumUpdateData(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.type = self.config.curriculum.learning.type
        assert self.type == 'Vanilla'
        self.step = 0
        self.pacing_epoch = self.config.curriculum.learning.pacing_epoch
        
        
    def on_validation_start(self, trainer, pl_module, *args, **kwargs):
        if trainer.current_epoch == 0 or self.config.trainer.resume_from_checkpoint is not None:
            self.original_dataset = trainer.datamodule.original_train_dataset
            if self.config.trainer.resume_from_checkpoint is not None:
                self.step = self._update_step(trainer)
            self._update_data(trainer, pl_module)
            
    def on_epoch_start(self, trainer, pl_module, *args, **kwargs):
        prev_step = self.step
        self.step = self._update_step(trainer)
        if prev_step != self.step:
            self._update_data(trainer, pl_module)
        trainer.logger.log_metrics({"Curriculum_step": self.step}, step=trainer.global_step)
        
    def _update_step(self, trainer):
        step = 0
        if trainer.current_epoch >= self.pacing_epoch and trainer.current_epoch < 3*self.pacing_epoch:
            step = 1
        elif trainer.current_epoch >= 3*self.pacing_epoch:
            step = 2
        return step
    
    def _update_data(self,trainer, pl_module):
        if self.step == 0:
            trainer.datamodule.train_dataset = CROHMEDataset(
                        self.original_dataset[self.step],
                        True,
                        self.config.data.scale_aug,
                    )
        elif self.step == 1:
            data = self.original_dataset[self.step] + self.original_dataset[self.step - 1]
            trainer.datamodule.train_dataset = CROHMEDataset(
                        data,
                        True,
                        self.config.data.scale_aug,
                    )

        elif self.step == 2:
            data = self.original_dataset[self.step] + self.original_dataset[self.step - 1] + self.original_dataset[self.step - 2]
            trainer.datamodule.train_dataset = CROHMEDataset(
                        data,
                        True,
                        self.config.data.scale_aug,
                    )
            
        trainer.reset_train_dataloader(model = pl_module)
        