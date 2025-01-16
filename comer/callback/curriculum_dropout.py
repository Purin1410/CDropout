from pytorch_lightning.callbacks import Callback
import torch
import math
from comer.curriculum.CL_datamodule import data_iterator

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
        self.debug = []
    
    def _update_dropout(self, trainer, pl_module):
        if self.config.curriculum.dropout.mha:
            for layer in pl_module.comer_model.decoder.model.decoder_layer:
                for attr in ['self_attn', 'multihead_attn']:
                    attn_layer = getattr(layer, attr, None)
                    if hasattr(attn_layer, 'dropout') and isinstance(attn_layer.dropout, torch.nn.Dropout):
                        attn_layer.dropout.p = self.current_dropout
                        print("attn_layer.dropout.p: ", attn_layer.dropout.p) # debug TODO: REMOVE LATER
        
        if self.config.curriculum.dropout.densenet:
            for layer in pl_module.comer_model.encoder.model.modules():
                if isinstance(layer, torch.nn.Dropout):
                    layer.p = self.current_dropout
        
        if self.config.curriculum.dropout.ffn:
            for layer in pl_module.comer_model.decoder.model.decoder_layer:
                for attr in ['dropout', 'dropout1', 'dropout2', 'dropout3']:
                    dropout_layer = getattr(layer, attr, None)
                    if hasattr(dropout_layer, 'dropout') and isinstance(dropout_layer, torch.nn.Dropout):
                        dropout_layer.p = self.current_dropout
                        print("dropout_layer.dropout.p: ", dropout_layer.p) # debug TODO: REMOVE LATER

    
    def _calculate_train_step(self, trainer, pl_module):
        print(layer.p for layer in pl_module.comer_model.encoder.model.modules())
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
        self.total_step = self._calculate_train_step(trainer,pl_module)
        if self.check_resume_checkpoint:
            print("total step: ", self.total_step)
            print("Start from epoch: ", trainer.current_epoch)
            if self.config.curriculum.learning.type == "Vanilla":
                self.current_step = self.cl_total_step + self.batch*(trainer.current_epoch + self.rest_epoch)
                self.current_dropout = self._dropout()
            else:
                self.total_step = self._calculate_train_step(trainer,pl_module)
                self.current_step = trainer.global_step
                self.current_dropout = self._dropout()
            self.check_resume_checkpoint = False           
        self._update_dropout(trainer, pl_module)
                
    
    def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
        self.current_dropout = self._dropout()
        self._update_dropout(trainer, pl_module)
        self.current_step += 1
    
    def on_epoch_end(self, trainer, pl_module, *args, **kwargs):
        if not self.check_resume_checkpoint:
            print("current dropout: ", self.current_dropout)
            trainer.logger.log_metrics(
                {"current_dropout": self.current_dropout}, 
                step=trainer.global_step
            )
            
            
            
