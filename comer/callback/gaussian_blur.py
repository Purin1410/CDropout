import torch
from pytorch_lightning.callbacks import Callback
import torchvision.transforms.functional as F
from torchvision.transforms import GaussianBlur

import matplotlib.pyplot as plt

def visualize_blur(img_tensor, blurred_tensor):
    """
    Visualize the original image and the blurred image side by side.

    Args:
        img_tensor (torch.Tensor): Original image tensor of shape [C, H, W].
        blurred_tensor (torch.Tensor): Blurred image tensor of shape [C, H, W].
    """
    # Convert tensors to numpy arrays
    img = img_tensor.squeeze().cpu().numpy()
    blurred_img = blurred_tensor.squeeze().cpu().numpy()

    # Plot images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(blurred_img, cmap="gray")
    axes[1].set_title("Blurred Image")
    axes[1].axis("off")

    plt.show()

def debug_pixel_values(img_tensor, blurred_tensor):
    """
    Print a few pixel values before and after applying Gaussian blur.

    Args:
        img_tensor (torch.Tensor): Original image tensor of shape [C, H, W].
        blurred_tensor (torch.Tensor): Blurred image tensor of shape [C, H, W].
    """
    print("Original Image Tensor:")
    print(img_tensor.squeeze().cpu().numpy()[:5, :5])  # Print top-left 5x5 patch
    print("\nBlurred Image Tensor:")
    print(blurred_tensor.squeeze().cpu().numpy()[:5, :5])



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
        self.debug_steps = 5
        

    def on_validation_start(self, trainer, pl_module):
        origin_dataset = trainer.datamodule.train_dataset
        self.max_steps = len(origin_dataset)*trainer.max_epochs
        print(self.max_steps)
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, *args, **kwargs):
        """
        Apply progressive Gaussian blur to input images at the start of each batch during training.
        """
        current_step = trainer.global_step
        print(current_step)
        if current_step > self.max_steps:
            return 
        
        current_sigma = self.sigma_init * (1 - min(current_step / self.max_steps, 1.0))
        
        if not isinstance(batch.imgs, torch.Tensor):
            raise ValueError("batch.imgs must be a tensor of shape [batch_size, C, H, W].")
        
        if batch.imgs.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor [batch_size, C, H, W], got {batch.imgs.dim()}D tensor."
            )

        blurred_imgs = [
            F.gaussian_blur(img, kernel_size=3, sigma=current_sigma) for img in batch.imgs
        ]
        
        # TODO: debug, delete later
        ###########################################################
        if current_step < self.debug_steps:
            print(f"Step {current_step}: sigma = {current_sigma:.4f}")
            visualize_blur(batch.imgs[0], blurred_imgs[0])
            debug_pixel_values(batch.imgs[0], blurred_imgs[0])
        ############################################################

        batch.imgs = torch.stack(blurred_imgs, dim=0)
        
        
    


