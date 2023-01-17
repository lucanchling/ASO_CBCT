import numpy as np
import pandas as pd
import os
from icecream import ic
import argparse
import glob
import torch

from Net import DenseNet
from DataModule import DataModuleClass, RandomRotation3D

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

from torch.nn import CosineSimilarity
import torch
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

def loss_fn(directionVector, directionVector_pred):
    return (1 - CosineSimilarity()(torch.Tensor(directionVector_pred), torch.Tensor(directionVector))).item()

def main(args):
    data_dir = args.data_dir
    
    csv_path = os.path.join(data_dir, 'CSV')

    df_train = pd.read_csv(os.path.join(csv_path, 'train.csv'))
    df_val = pd.read_csv(os.path.join(csv_path, 'val.csv'))
    df_test = pd.read_csv(os.path.join(csv_path, 'test.csv'))
    

    normpath = os.path.normpath("/".join([args.model_dir, '**', '*']))
    list_ckpt = [i for i in glob.glob(normpath,recursive=True) if i.endswith(".ckpt")==True]
    
    df = pd.DataFrame([ckpt.split('/')[-1] for ckpt in list_ckpt], columns=['checkpoint_path'])
    
    for angle in [3.14159265359, 1.57079632679, 1.0471975512, 0.78539816339, 0.52359877559, 0.39269908169, 0.26179938779, 0.0]:
        print("Generating for angle {:.3}".format(angle))
        
        test_transform = RandomRotation3D(x_angle=angle, y_angle=angle, z_angle=angle)
        db = DataModuleClass(df_train, df_val, df_test, batch_size=1, train_transform=None, val_transform=None, test_transform=test_transform)
        
        db.setup('test')
        ds_test = db.test_dataloader()
        SCAN, DIRECTIONVECTOR, SCANPATH = [], [], []
        for i, batch in enumerate(ds_test):
            scan, directionVector, scan_path = batch
            SCAN.append(scan)
            DIRECTIONVECTOR.append(directionVector)
            SCANPATH.append(scan_path)
        
        
        ALL_LOSSES = []
        for cp,checkpoint_path in enumerate(list_ckpt):
            LOSS = []
            # print(checkpoint_path.split('/')[-1])

            lr = float(checkpoint_path.split('_bs')[0].split('/')[-1].split('lr')[1])

            model = DenseNet(lr)

            model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

            model.to('cuda')
            
            model.eval()
            with torch.no_grad():
                for i in range(len(SCAN)):
                    scan = SCAN[i].to('cuda')
                    directionVector = DIRECTIONVECTOR[i].to('cuda')
                    scan_path = SCANPATH[i]
                    directionVector_pred = model(scan.to('cuda'))
                    directionVector_pred = directionVector_pred.cpu().numpy()
                    directionVector = directionVector.cpu().numpy()
                    loss = loss_fn(directionVector, directionVector_pred)
                    # ic(loss)
                    # ic(scan_path)
                    LOSS.append(loss)
                    # ic(directionVector, directionVector_pred, scan_path, loss(directionVector, directionVector_pred).item())
                    # gen_plot(directionVector[0], directionVector_pred[0],scan_path)
                    # break
            # print('MEAN LOSS:', np.mean(LOSS))
            ALL_LOSSES.append(np.mean(LOSS))

            # if cp == 1:
            #     break

        df['mean_loss (angle='+str(round(angle,2))+')'] = ALL_LOSSES
    df.to_csv(os.path.join(args.model_dir, 'ALL_LOSSES.csv'))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/Oriented/LargeFOV_RESAMPLED/')
    parser.add_argument('--model_dir', type=str, default='/home/luciacev/Desktop/Luc_Anchling/Training_OR/NEW_LFOV/Models/')

    args = parser.parse_args()

    main(args)    