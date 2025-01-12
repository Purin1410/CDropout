import torch
from pytorch_lightning.callbacks import Callback
import torchvision.transforms.functional as F
from torchvision.transforms import GaussianBlur


class CurriculumInputBlur(Callback):
    def __init__(self, sigma_init: float):
        """
        Callback for progressively reducing Gaussian blur on input images during training.

        Args:
            sigma_init (float): Initial standard deviation for Gaussian blur.
            max_steps (int): Total steps over which to reduce the blur.
        """
        super().__init__()
        self.sigma_init = sigma_init
        self.max_steps = 0
    def on_validation_start(self, trainer, pl_module):
        origin_dataset = trainer.datamodule.train_dataset
        self.max_steps = len(origin_dataset)*trainer.max_epochs
        print("self.max_steps: ", self.max_steps)
    
    def on_train_batch_start(self, trainer, pl_module, batch, *args, **kwargs):
        """
        Apply progressive Gaussian blur to input images at the start of each batch during training.
        """
        current_step = trainer.global_step
        if current_step > self.max_steps:
            return 
        
        current_sigma = self.sigma_init*(1 - current_step / self.max_steps)
        
        if not isinstance(batch.imgs, torch.Tensor):
            raise ValueError("batch.imgs must be a tensor of shape [batch_size, C, H, W].")
        
        if batch.imgs.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor [batch_size, C, H, W], got {batch.imgs.dim()}D tensor."
            )

        blurred_imgs = [
            F.gaussian_blur(img, kernel_size=3, sigma=current_sigma) for img in batch.imgs
        ]
        
        trainer.logger.log_metrics({"sigma": current_sigma}, step=trainer.global_step)

        batch.imgs = torch.stack(blurred_imgs, dim=0)
        
        
    


