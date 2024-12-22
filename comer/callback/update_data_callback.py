from pytorch_lightning.callbacks import Callback
from comer.curriculum.CL_datamodule import CL_CROHMEDatamodule, data_iterator
from comer.datamodule.dataset import CROHMEDataset

Type = ['Vanilla', 'Self-Paced','Self-Paced-CL']

class CurriculumUpdateData(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.type = self.config.curriculum.learning.type
        if self.type not in Type:
            raise ValueError('Type available: ',Type)
        self.step = self.config.curriculum.learning.step
        self.start_percent = self.config.curriculum.learning.start_percent
        self.pacing_epoch = self.config.curriculum.learning.pacing_epoch
        
    
    def _Sort(self, trainer):
        if self.type == 'Vanilla':
            trainer.datamodule.original_train_dataset = self._Vanilla_sort(trainer)
            

    def _Vanilla_sort(self,trainer):
        assert trainer.datamodule.original_train_dataset != None
        return sorted(trainer.datamodule.original_train_dataset, key=lambda x: len(x[2]))
    # def _Self_paced_sort(self,trainer):
    #     if self.config.trainer.resume_from_checkpoint != None:
    #         pass
    #     else:
    #         assert trainer.data_module.original_train_dataset != None
    #         pass
    
    def _update_data_percent(self, trainer, data_percent):
        self.data_percent = data_percent
        trainer.datamodule.train_dataset = CROHMEDataset(
                    data_iterator(
                        data = trainer.datamodule.original_train_dataset[:int(len(trainer.datamodule.original_train_dataset)*self.data_percent)],
                        batch_size= self.config.data.train_batch_size
                    ),
                    True,
                    self.config.data.scale_aug,
                )
        print(len(trainer.datamodule.train_dataset)) # debug
        trainer.logger.log_metrics({"Data_percent": self.data_percent}, step=trainer.global_step)

    def on_validation_start(self, trainer, pl_module, *args, **kwargs):
        if trainer.current_epoch == 0:
            print('Sorted all data by lenght')
            self._Sort(trainer)
            print('Done sorted')
            if self.config.trainer.resume_from_checkpoint != None:
                self.current_step = trainer.current_epoch // self.pacing_epoch
                self.data_percent = (self.start_percent + self.step*self.current_step)
            else:
                self.data_percent = self.start_percent

            self._update_data_percent(trainer = trainer, data_percent = self.data_percent)
    
    def on_epoch_start(self, trainer, pl_module, *args, **kwargs):
        if trainer.current_epoch != 0 and trainer.current_epoch % self.pacing_epoch == 0:
            self.data_percent = self.data_percent + self.step
            self._update_data_percent(trainer = trainer, data_percent = self.data_percent)
        
        
        
