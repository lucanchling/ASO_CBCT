import numpy as np
import pandas as pd
import os
from icecream import ic
import argparse

import torch

from Net import DenseNet
from DataModule import DataModuleClass, RandomRotation3D

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import SimpleITK as sitk


import matplotlib.pyplot as plt


def gen_plot(direction, direction_hat,scan_path):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.quiver(0, 0, 0, direction[0], direction[1], direction[2], color='r', label='y')
    ax.quiver(0, 0, 0, direction_hat[0], direction_hat[1], direction_hat[2], color='b', label='y_hat')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    ax.legend()
    plt.title(scan_path)
    # plt.title('DIRECTION:  {:.4f}  |  {:.4f}  |  {:.4f}\nDIRECTION_HAT :  {:.4f}  |  {:.4f}  |  {:.4f}'.format(direction[0],direction[1],direction[2],direction_hat[0],direction_hat[1],direction_hat[2]))
    plt.show()


def main(args):
    data_dir = args.data_dir
    
    csv_path = os.path.join(data_dir, 'CSV')

    df_train = pd.read_csv(os.path.join(csv_path, 'train.csv'))
    df_val = pd.read_csv(os.path.join(csv_path, 'val.csv'))
    df_test = pd.read_csv(os.path.join(csv_path, 'test.csv'))
    
    test_transform = None
    if args.test_rot:
        test_transform = RandomRotation3D(x_angle=args.angle, y_angle=args.angle, z_angle=args.angle)
    db = DataModuleClass(df_train, df_val, df_test, batch_size=1, train_transform=None, val_transform=None, test_transform=test_transform)
    db.setup('test')
    
    model = DenseNet(args.lr)

    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])

    model.to('cuda')
    
    model.eval()
    ds_test = db.test_dataloader()
    with torch.no_grad():
        for i, batch in enumerate(ds_test):
            scan, directionVector, scan_path = batch
            directionVector_pred = model(scan.to('cuda'))
            directionVector_pred = directionVector_pred.cpu().numpy()
            directionVector = directionVector.numpy()
            # print(directionVector_pred, directionVector)
            # direction_pred = direction_pred[0]# / np.linalg.norm(direction_pred[0])
            gen_plot(directionVector[0], directionVector_pred[0],scan_path)
            # print(scale.item(),scale_pred.item())
            # break
        # for i in range(5):
        #     print(model(torch.rand(1,1,128,128,128).to('cuda')))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/Oriented/LARGEFOV_RESAMPLED/')
    parser.add_argument('--test_rot', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint', type=str, default='/home/luciacev/Desktop/Luc_Anchling/Training_OR/NEW_LFOV/Models/lr1e-04_bs30_angle1.57.ckpt')
    parser.add_argument('--angle', type=float, default=np.pi/2)

    args = parser.parse_args()
    
    main(args)    