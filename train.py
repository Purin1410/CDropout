import pytorch_lightning as pl
from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    Callback,
)
from pytorch_lightning.loggers import WandbLogger as Logger
import argparse
from sconf import Config
from comer.callback.grad_norm_callback import GradNormCallback
from comer.callback.rclone_callback import RcloneUploadCallback
from comer.callback.curriculum_dropout import CurriculumDropout
from comer.curriculum.CL_datamodule import CL_CROHMEDatamodule
import subprocess
from comer.callback.update_data_callback import CurriculumUpdateData
from comer.callback.skip_validation import SkipValidation
from comer.callback.gaussian_blur import CurriculumInputBlur

Type = ['Vanilla', 'Self-Paced','Self-Paced-CL']

def train(config):
    # Seed
    pl.seed_everything(config.seed_everything, workers=True)

    # Model
    model_module = LitCoMER(**config.model)

   # Logger
    logger = Logger("", project="CoMER_KAGGLE", config=dict(config), log_model='all')
    logger.watch(model_module.comer_model, log="all", log_freq=100)

    # Data
    if config.curriculum.learning.type not in Type:
        data_module = CROHMEDatamodule(**config.data)
    else:
        data_module = CL_CROHMEDatamodule(config = config)
        data_module.setup(stage = "fit", model = model_module)

   # Callback
    lr_callback = LearningRateMonitor(logging_interval=config.trainer.callbacks[0].init_args.logging_interval)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.trainer.callbacks[1].init_args.save_top_k, 
                                                    monitor=None, #config.trainer.callbacks[1].init_args.monitor,
                                                    mode=config.trainer.callbacks[1].init_args.mode,
                                                    filename=config.trainer.callbacks[1].init_args.filename)

    grad_norm_callback = GradNormCallback()

    local_dir = "/kaggle/working/CoMER_checkpoints"
    remote_dir =  "one_drive:Projects/HMER\ Project/Checkpoints/CoMER_KAGGLE"
    r_clone_callback = RcloneUploadCallback(
        local_dir = local_dir,
        remote_dir = remote_dir)
    
    skip_validation = SkipValidation(skip_val_epoch= 200)

    # Curriculum module
    curriculum_dropout = CurriculumDropout(config = config)
    gaussian_blur = CurriculumInputBlur(config = config)
    update_data = CurriculumUpdateData(config = config)
    
    trainer = pl.Trainer(
        gpus=config.trainer.gpus,
        accelerator=config.trainer.accelerator,
        val_check_interval=1.0,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        max_epochs=config.trainer.max_epochs,
        logger=logger,
        deterministic=config.trainer.deterministic,
        callbacks= [lr_callback,
                    grad_norm_callback,
                    checkpoint_callback,
                    r_clone_callback,
                    skip_validation,
                    # Curriculum module
                    curriculum_dropout,
                    gaussian_blur,
                    update_data],
        default_root_dir=local_dir,
        resume_from_checkpoint=config.trainer.resume_from_checkpoint,
    )

    try:
        trainer.fit(model_module,data_module)
    except Exception as e:
        print(f"Training crashed due to: {e}")
    finally:
        print("Ensuring final upload to OneDrive before exit...")
        subprocess.run(f"rclone copy {local_dir} {remote_dir} --verbose", shell=True, check=True)
        print("Final upload completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    print(config.dumps())
    train(config)