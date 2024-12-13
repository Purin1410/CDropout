from pytorch_lightning.callbacks import Callback

class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """
    @staticmethod
    def gradient_norm(model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def on_after_backward(self, trainer, model):
        model.log("train/grad_norm", self.gradient_norm(model))