from pytorch_lightning.callbacks import Callback
import torch
import math
from comer.curriculum.CL_datamodule import data_iterator
# # dropout_current = 1 - [dropout_end*exp(-10*step/total_step) + (1 - dropout_end)]
     

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
        self.slope = config.curriculum.dropout.slope
        self.total_batch = 0
        self.pacing_epoch = config.curriculum.learning.pacing_epoch
        self.check_resume_checkpoint = bool(config.trainer.resume_from_checkpoint)
        self.dropout_modules = []
        self.debug = []
    
    def _update_dropout(self, trainer, pl_module):
        for module in self.dropout_modules:
            module.p = self.current_dropout
    
    def _calculate_train_step(self, trainer, pl_module):
        # TODO: REMOVE THIS LINE LATER
        ##############################
        self.debug = [m for m in pl_module.comer_model.decoder.model.layers.modules()]
        for module in self.debug:  #debug
            print(module)
            print()
        ##############################
        self.dropout_modules = [m for m in pl_module.comer_model.decoder.model.layers.modules() if isinstance(m, torch.nn.Dropout)]
        print("Self.dropout_modules:", self.dropout_modules) # debug
        if self.config.curriculum.learning.type == "Vanilla":
            cl_start = int(self.config.curriculum.learning.start_percent*10)
            # calculate total batch model will train in CL mode
            origin_dataset = trainer.datamodule.original_train_dataset
            cl_total_batch = 0
            for i in range(cl_start, 11):
                batch = len(data_iterator(
                    data = origin_dataset[:int(len(origin_dataset)*i/10)],
                    batch_size= self.config.data.train_batch_size
                ))
                cl_total_batch += batch
                print("total batch: ", cl_total_batch) # debug
                print("batch: ", batch) # debug
                print()
            cl_total_step = cl_total_batch*self.pacing_epoch
            
            # TODO: need to fix this dirty code
            self.cl_total_step = cl_total_step #THIS IS DIRTY CODE
            self.rest_epoch = - (11-cl_start)*self.pacing_epoch #THIS IS DIRTY CODE
            self.batch = batch #THIS IS DIRTY CODE
            
            # calculate the rest of step in the rest of epoch
            rest_epoch = self.max_epochs - (11-cl_start)*self.pacing_epoch
            rest_step =  rest_epoch*batch
            total_step = cl_total_step + rest_step
        else:
            origin_dataset = trainer.datamodule.train_dataset
            total_step = len(origin_dataset)*self.max_epochs
            print(total_step)
        return total_step

    def _dropout(self):
        return (1 - (( self.end_dropout)*math.exp(-self.slope*self.current_step/self.total_step) + (1 - self.end_dropout)))

    def on_train_start(self, trainer, pl_module, *args, **kwargs):
        if self.check_resume_checkpoint:
            self.total_step = self._calculate_train_step(trainer,pl_module)
            print("total step: ", self.total_step)
            print("Start from epoch: ", trainer.current_epoch)
            # debug + dirty code
            #TODO: need to fix this dirty code
            self.current_step = self.cl_total_step + self.batch*(trainer.current_epoch + self.rest_epoch)
            print("current step: ", self.current_step)
            
            self.current_dropout = self._dropout()
            self._update_dropout(trainer, pl_module)
            print("current dropout: ", self.current_dropout)
            print("current step: ", self.current_step)
            self.check_resume_checkpoint = False
        else:
            self.total_step = self._calculate_train_step(trainer,pl_module)
            self._update_dropout(trainer, pl_module)
        print(self.debug)
                
    
    def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
        self.current_dropout = self._dropout()
        self._update_dropout(trainer, pl_module)
        self.current_step += 1
    
    def on_epoch_end(self, trainer, pl_module, *args, **kwargs):
        # print(self.debug)
        if not self.check_resume_checkpoint:
            print("current dropout: ", self.current_dropout)
            trainer.logger.log_metrics(
                {"current_dropout": self.current_dropout}, 
                step=trainer.global_step
            )
            
            
            
