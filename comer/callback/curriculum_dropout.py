from pytorch_lightning.callbacks import Callback
import torch
import math

# # dropout_current = 1 - (1 - dropout_final)*exp(-10*step/total_step) + dropout_final + self.start_dropout
     

class CurriculumDropout(Callback):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.start_dropout = self.config.curriculum.dropout.start_dropout
            self.end_dropout = self.config.curriculum.dropout.end_dropout
            self.current_dropout = self.start_dropout
            self.current_step = 0
            self.total_step = 0
            self.max_epochs = config.trainer.max_epochs
        
        def _update_dropout(self, trainer, pl_module):
            for module in pl_module.comer_model.decoder.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = self.current_dropout

        def on_train_start(self, trainer, pl_module, *args, **kwargs):
            self.total_step = len(trainer.datamodule.train_dataloader())*self.max_epochs
            print("total step: ", self.total_step)
            if self.config.trainer.resume_from_checkpoint is None:
                 self._update_dropout(trainer, pl_module)
            else:
                 self.current_dropout = pl_module.comer_model.decoder.dropout.p
        
        def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
            self.current_dropout = 1 - (( 1 - self.end_dropout)*math.exp(-10*self.current_step/self.total_step) + self.end_dropout + self.start_dropout)
            self._update_dropout(trainer, pl_module)
            self.current_step += 1
        
        def on_epoch_end(self, trainer, pl_module, *args, **kwargs):
            print("current dropout: ", self.current_dropout)
            trainer.logger.log_metrics(
                {"current_dropout": self.current_dropout}, 
                step=trainer.global_step
            )
            
            
            
