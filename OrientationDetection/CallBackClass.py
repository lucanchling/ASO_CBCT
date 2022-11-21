from pytorch_lightning.callbacks import Callback
import torch
import matplotlib.pyplot as plt
import numpy as np
import io
import PIL.Image
from torchvision.transforms import ToTensor
from icecream import ic

def gen_plot(direction, direction_hat, scan_path):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.quiver(0, 0, 0, direction[0], direction[1], direction[2], color='r', label='y')
    ax.quiver(0, 0, 0, direction_hat[0], direction_hat[1], direction_hat[2], color='b', label='y_hat')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.legend()
    plt.title(scan_path)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf


class DirectionLogger(Callback):
    def __init__(self, train_scan, val_scan, log_steps=10):
        self.log_steps = log_steps
        self.train_scan = train_scan
        self.val_scan = val_scan

        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):        

        if batch_idx % self.log_steps == 0:

                scan, directionVector, scan_path = batch
                batch_size = scan.shape[0]
                scan_path = [scan_path[i] for i in range(batch_size)]

                try:
                    idx = scan_path.index(self.train_scan)
                    scan = scan.to(pl_module.device, non_blocking=True)
                    directionVector = directionVector.to(pl_module.device, non_blocking=True)
                    
                    with torch.no_grad():
                        
                        
                        directionVector_hat = pl_module(scan)
                        
                        directionVector_hat = directionVector_hat.cpu().numpy()
                        directionVector = directionVector.cpu().numpy()

                        directionVector_hat = directionVector_hat[idx]
                        directionVector = directionVector[idx]

                        buf = gen_plot(directionVector, directionVector_hat, scan_path[idx])
                        image = PIL.Image.open(buf)
                        image = ToTensor()(image)
                        trainer.logger.experiment.add_image('train_direction', image, trainer.global_step)

                except:
                    pass