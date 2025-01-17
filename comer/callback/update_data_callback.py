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
        self.original_dataset = trainer.datamodule.original_train_dataset
        if self.config.trainer.resume_from_checkpoint is not None:
            self.step = self._update_step(trainer)
        self._update_data(trainer)
            
    def on_epoch_end(self, trainer, pl_module, *args, **kwargs):
        prev_step = self.step
        self.step = self._update_step(trainer)
        if prev_step != self.step:
            self._update_data(trainer)
        
    def _update_step(self, trainer):
        if trainer.current_epoch > self.pacing_epoch and trainer.current_epoch < 3*self.pacing_epoch:
            step = 1
        elif trainer.current_epoch >= 3*self.pacing_epoch:
            step = 2
        return step
    
    def _update_data(self,trainer):
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
            trainer.datamodule.train_dataset = CROHMEDataset(
                        self.original_dataset,
                        True,
                        self.config.data.scale_aug,
                    )
        
    # def _update_data_percent(self, trainer, pl_module, data_percent):
    #     assert self.data_percent <= 1
    #     self.data_percent = data_percent
    #     trainer.datamodule.train_dataset = CROHMEDataset(
    #                 data_iterator(
    #                     data = trainer.datamodule.original_train_dataset[:int(len(trainer.datamodule.original_train_dataset)*self.data_percent)],
    #                     batch_size= self.config.data.train_batch_size
    #                 ),
    #                 True,
    #                 self.config.data.scale_aug,
    #             )
    #     trainer.reset_train_dataloader(model = pl_module)
    #     print(len(trainer.datamodule.train_dataset)) # debug
    #     print(self.data_percent)
    #     trainer.logger.log_metrics({"Data_percent": self.data_percent}, step=trainer.global_step)

    # def on_validation_start(self, trainer, pl_module, *args, **kwargs):
    #     if self.start_training == True:
    #         print('Sorted all data by lenght')
    #         self._Sort(trainer)
    #         print('Done sorted')
    #         if self.config.trainer.resume_from_checkpoint is not None:
    #             self.current_step = trainer.current_epoch // self.pacing_epoch
    #             self.data_percent = min((self.start_percent + self.step*self.current_step),1.0)
    #         else:
    #             self.data_percent = self.start_percent

    #         self._update_data_percent(trainer = trainer, data_percent = self.data_percent, pl_module = pl_module)
    #         self.start_training = False
    
    # def on_epoch_start(self, trainer, pl_module, *args, **kwargs):
    #     if trainer.current_epoch != 0 and trainer.current_epoch % self.pacing_epoch == 0:
    #         self.data_percent = min(self.data_percent + self.step,1.0)
    #         print("Update data percent: ", self.data_percent) # debug
    #         self._update_data_percent(trainer = trainer, data_percent = self.data_percent, pl_module = pl_module)
        
        
        
