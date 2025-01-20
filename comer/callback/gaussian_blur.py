import torch
from pytorch_lightning.callbacks import Callback
import torchvision.transforms.functional as F
from torchvision.transforms import GaussianBlur

class CurriculumInputBlur(Callback):
    def __init__(self, config):
        """
        Callback for progressively reducing Gaussian blur on input images during training.

        Args:
            sigma_init (float): Initial standard deviation for Gaussian blur.
            max_steps (int): Total steps over which to reduce the blur.
        """
        super().__init__()
        self.config = config
        self.sigma_init = self.config.curriculum.blur.sigma
        self.max_steps = 0
        self.kernel_size = int(6*self.sigma_init + 1)
        self.resume_training  = bool(config.trainer.resume_from_checkpoint)
        
    def on_validation_start(self, trainer, pl_module):
        if trainer.current_epoch == 0 or self.resume_training:
            if self.config.curriculum.learning.type != "Vanilla":
                origin_dataset = trainer.datamodule.train_dataset
                self.max_steps = len(origin_dataset)*trainer.max_epochs
                print("self.max_steps: ", self.max_steps)
            else:
                origin_dataset = trainer.datamodule.original_train_dataset
                curriculum_step = 1
                step = 0
                batch = 0
                for i in range(len(origin_dataset)):
                    batch += len(origin_dataset[i])
                    step = batch*self.config.curriculum.learning.pacing_epoch*curriculum_step
                    self.max_steps += step
                    curriculum_step *= 2
                step = 0
                batch = 0
                print("self.max_steps: ", self.max_steps)
            self.resume_training = False
            
            
    
    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        """
        Apply progressive Gaussian blur to input images at the start of each batch during training.
        """
        current_step = 2*trainer.global_step
        
        if current_step > self.max_steps:
            return 
        
        current_sigma = max(self.sigma_init*(1 - current_step / self.max_steps), 0)
        
        trainer.logger.log_metrics({"sigma": current_sigma}, step=trainer.global_step)

        if current_sigma <= 0:
            return
        
        if not isinstance(batch.imgs, torch.Tensor):
            raise ValueError("batch.imgs must be a tensor of shape [batch_size, C, H, W].")
        
        if batch.imgs.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor [batch_size, C, H, W], got {batch.imgs.dim()}D tensor."
            )

        blurred_imgs = [
            F.gaussian_blur(img, kernel_size=self.kernel_size, sigma=current_sigma) for img in batch.imgs
        ]
        batch.imgs = torch.stack(blurred_imgs, dim=0)

        
        
    


