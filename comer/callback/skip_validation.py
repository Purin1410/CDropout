from pytorch_lightning.callbacks import Callback

class SkipValidation(Callback):
    def __init__(self, skip_val_epoch: int):
        super().__init__()
        self.skip_val_epoch = skip_val_epoch

    def on_validation_start(self, trainer, pl_module):
        if trainer.current_epoch >= 328:
            trainer.check_val_every_n_epoch = 1
        elif trainer.current_epoch >= 298:
            trainer.check_val_every_n_epoch = 10
        else:
            trainer.check_val_every_n_epoch = 50
    
    def on_train_start(self, trainer, pl_module):
        if trainer.current_epoch >= 328:
            trainer.check_val_every_n_epoch = 1
        elif trainer.current_epoch >= 298:
            trainer.check_val_every_n_epoch = 10
        else:
            trainer.check_val_every_n_epoch = 50
    