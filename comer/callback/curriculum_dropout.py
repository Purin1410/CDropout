from pytorch_lightning.callbacks import Callback
import torch
import math

# # dropout_current = (1 - dropout_final)*exp(-10*step/total_step) + dropout_final
     

class CurriculumDropout(Callback):
        def __init__(self, config):
            super().__init__()
            self.start_dropout = config.curriculum_dropout.start_dropout
            self.end_dropout = config.curriculum_dropout.end_dropout
            self.current_dropout = self.start_dropout
            self.current_step = 0
            self.total_step = 0
        
        def _update_dropout(self, trainer, pl_module):
            for module in pl_module.comer_model.decoder.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = self.current_dropout
            trainer.logger.log_metrics(
                {"current_dropout": self.current_dropout}, 
                step=trainer.global_step
            )
        def on_train_start(self, trainer, pl_module):
            self.total_step = trainer.estimated_stepping_batches
            self._update_dropout(trainer, pl_module)
        
        def on_train_batch_start(self, trainer, pl_module):
            self.current_dropout = ( 1 - self.end_dropout)*math.exp(-10*self.current_step/self.total_step) + self.end_dropout
            self._update_dropout(trainer, pl_module)
            self.current_step += 1
            print("current dropout: ", self.current_dropout)
            
            
            
