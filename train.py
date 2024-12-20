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

Type = ['Vanilla', 'Self-Paced','Self-Paced-CL']

def train(config):
    # Seed
    pl.seed_everything(config.seed_everything, workers=True)

    # Model
    if config.trainer.resume_from_checkpoint is not None:
        print("Resuming from checkpoint: ", config.trainer.resume_from_checkpoint)
        model_module = LitCoMER.load_from_checkpoint(config.trainer.resume_from_checkpoint)
    else:
        print("Training from new weights")
        model_module = LitCoMER(**config.model)

   # Logger
    logger = Logger("CoMer_10_50_cosine", project="CDropout_traditional", config=dict(config), log_model='all')
    logger.watch(model_module.comer_model, log="all", log_freq=100)

    # Data
    # data_module = CL_CROHMEDatamodule(config = config)
    # data_module.setup(stage = "fit", model = model_module)
    if config.curriculum.learning.type not in Type:
        data_module = CROHMEDatamodule(**config.data)
    else:
        data_module = CL_CROHMEDatamodule(config = config)
        data_module.setup(stage = "fit", model = model_module)

   # Callback
    lr_callback = LearningRateMonitor(logging_interval=config.trainer.callbacks[0].init_args.logging_interval)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.trainer.callbacks[1].init_args.save_top_k, 
                                                    monitor=None, 
                                                    mode=config.trainer.callbacks[1].init_args.mode,
                                                    filename=config.trainer.callbacks[1].init_args.filename)

    grad_norm_callback = GradNormCallback()

    curriculum_dropout = CurriculumDropout(config = config)

    local_dir = "/kaggle/working/CoMER_checkpoints"
    remote_dir =  "one_drive:Projects/HMER\ Project/Checkpoints/CoMER_CDropout"
    r_clone_callback = RcloneUploadCallback(
        local_dir = local_dir,
        remote_dir = remote_dir)
    
    update_data = CurriculumUpdateData(config = config)
    
    trainer_config = {k: v for k, v in config.trainer.items() if k not in ["callbacks","resume_from_checkpoint"]}
    trainer = pl.Trainer(
        devices=config.trainer.gpus,
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
                    curriculum_dropout,
                    update_data],
        default_root_dir=local_dir,
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