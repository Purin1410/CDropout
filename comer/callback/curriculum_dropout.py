from pytorch_lightning.callbacks import Callback
import torch
import math
from comer.curriculum.CL_datamodule import data_iterator

class CurriculumDropout(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Dropout: MHA, DenseNet, FFN
        # MHA
        self.mha_start_dropout = self.config.curriculum.dropout.mha_start_dropout
        self.mha_end_dropout = self.config.curriculum.dropout.mha_end_dropout
        # FFN
        self.ffn_start_dropout = self.config.curriculum.dropout.ffn_start_dropout
        self.ffn_end_dropout = self.config.curriculum.dropout.ffn_end_dropout
        # DenseNet
        self.densenet_start_dropout = self.config.curriculum.dropout.densenet_start_dropout
        self.densenet_end_dropout = self.config.curriculum.dropout.densenet_end_dropout
        # Current dropout
        self.mha_current_dropout = self.mha_start_dropout
        self.ffn_current_dropout = self.ffn_start_dropout
        self.densenet_current_dropout = self.densenet_start_dropout
        # List of dropout layer
        self.mha_dropout_layer = []
        self.ffn_dropout_layer = []
        self.densenet_dropout_layer = []
        # Other
        self.total_step = 0
        self.slope = config.curriculum.dropout.slope
        self.total_batch = 0
        self.pacing_epoch = config.curriculum.learning.pacing_epoch
        self.check_resume_checkpoint = bool(config.trainer.resume_from_checkpoint)

    def on_train_start(self, trainer, pl_module, *args, **kwargs):
        # Create a list of dropout layer
        # Create decoder dropout list
        if not self.mha_dropout_layer or not self.ffn_dropout_layer or not self.densenet_dropout_layer:
            if self.config.curriculum.dropout.mha or self.config.curriculum.dropout.ffn or self.config.curriculum.dropout.densenet:
                self._initialize_dropout_layers(pl_module)
        
        # Calculate total step                                   
        self.total_step = self._calculate_train_step(trainer,pl_module)      
        self._update_dropout(trainer)
    
    def on_train_batch_start(self, trainer, pl_module, *args, **kwargs):
        self._update_dropout(trainer)    
           
        
    def _update_dropout(self, trainer):
        dropout_map = {
            "MHA": (self.mha_dropout_layer, self.mha_end_dropout),
            "DenseNet": (self.densenet_dropout_layer, self.densenet_end_dropout),
            "FFN": (self.ffn_dropout_layer, self.ffn_end_dropout),
        }
        
        metrics = {}
        for key, (dropout_layer_list, end_dropout) in dropout_map.items():
            if getattr(self.config.curriculum.dropout, key.lower(), False):  # Check if enabled
                current_dropout = self._dropout(trainer, end_dropout)
                self._update_dropout_layer(current_dropout, dropout_layer_list)
                metrics[f"{key}_dropout"] = current_dropout

        trainer.logger.log_metrics(metrics, step=trainer.global_step)

        
    def _dropout(self, trainer, end_dropout):
        return (1 - (( end_dropout)*math.exp(-self.slope*trainer.global_step/self.total_step) + (1 - end_dropout)))
    
    def _update_dropout_layer(self,current_dropout, dropout_layer_list):
        for layer in dropout_layer_list:
            layer.p = current_dropout
    
    def _initialize_dropout_layers(self,pl_module):
        for layer in pl_module.comer_model.decoder.model.layers:
            if self.config.curriculum.dropout.mha:
                for attr in ['self_attn', 'multihead_attn']:
                    attn_layer = getattr(layer, attr, None)
                    if hasattr(attn_layer, 'dropout') and isinstance(attn_layer.dropout, torch.nn.Dropout):
                        self.mha_dropout_layer.append(attn_layer.dropout)
                        
            if self.config.curriculum.dropout.ffn:
                for attr in ['dropout', 'dropout1', 'dropout2', 'dropout3']:
                    dropout_layer = getattr(layer, attr, None)
                    if isinstance(dropout_layer, torch.nn.Dropout):
                        self.ffn_dropout_layer.append(dropout_layer)
        
        if self.config.curriculum.dropout.densenet:
            for layer in pl_module.comer_model.encoder.model.modules():
                if isinstance(layer, torch.nn.Dropout):
                    self.densenet_dropout_layer.append(layer)
    
    def _calculate_train_step(self, trainer, pl_module):
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
            cl_total_step = cl_total_batch*self.pacing_epoch
            
            # calculate the rest of step in the rest of epoch
            rest_epoch = self.config.trainer.max_epochs - (11-cl_start)*self.pacing_epoch
            rest_step =  rest_epoch*batch
            total_step = cl_total_step + rest_step
        else:
            origin_dataset = trainer.datamodule.train_dataset
            total_step = len(origin_dataset)*self.config.trainer.max_epochs
        return total_step
            
            
