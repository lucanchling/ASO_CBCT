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

from CallBackClass import DirectionLogger

def main(args):
    
    mount_point = args.mount_point

    csv_path = os.path.join(mount_point, 'CSV/')

    df_train = pd.read_csv(os.path.join(csv_path, args.csv_train))
    df_val = pd.read_csv(os.path.join(csv_path, args.csv_valid))
    df_test = pd.read_csv(os.path.join(csv_path, args.csv_test))


    checkpoint_callback = ModelCheckpoint(
        dirpath= os.path.join(args.out,'checkpoints'),
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    db = DataModuleClass(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, 
                        train_transform=RandomRotation3D(x_angle=args.angle, y_angle=args.angle, z_angle=args.angle), 
                        val_transform=None,#RandomRotation3D(x_angle=np.pi/10, y_angle=np.pi/10, z_angle=np.pi/10), 
                        test_transform=None, mount_point=mount_point)
    
    model = DenseNet(lr=args.lr)
    
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=os.path.join(args.out,args.tb_dir), name=None)
    
    direction_logger = DirectionLogger(train_scan=df_train['scan_path'][0], val_scan=df_val['scan_path'][0])
    # ic(df_train['scan_path'][0])
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[early_stop_callback, checkpoint_callback],#, direction_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=db)#, ckpt_path=args.model)
    
    trainer.test(datamodule=db)

    # print the path of the best model
    ic(trainer.checkpoint_callback.best_model_path)

    if not os.path.exists(os.path.join(args.out, 'Models')):
        os.mkdir(os.path.join(args.out, 'Models'))
    
    # save the best model to Models folder
    os.rename(trainer.checkpoint_callback.best_model_path, os.path.join(args.out, 'Models', "lr"+"{:.0e}".format(args.lr)+"_bs"+str(args.batch_size)+"_angle"+str(round(args.angle,2))+".ckpt"))
    
    # rename tb dir Version_0 to lr=args.lr ; bs=args.batch_size
    os.rename(os.path.join(args.out, args.tb_dir,'version_0'), os.path.join(args.out,args.tb_dir, "lr="+"{:.0e}".format(args.lr)+" ; bs="+str(args.batch_size)+" ; angle="+str(round(args.angle,2))))


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='ALI CBCT Training')
    parser.add_argument('--csv_train', help='CSV with Scan and Landmarks files', type=str, default='train.csv')    
    parser.add_argument('--csv_valid', help='CSV with Scan and Landmarks files', type=str, default='val.csv')
    parser.add_argument('--csv_test', help='CSV with Scan and Landmarks files', type=str, default='test.csv')      
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)    
    parser.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    parser.add_argument('--model', help='Model to continue training', type=str, default= None)
    parser.add_argument('--out', help='Output', type=str, default="")
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="")
    parser.add_argument('--num_workers', help='Number of workers for loading', type=int, default=20)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=30)
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=30)
    parser.add_argument('--angle', help='Angle for random rotation', type=float, default=np.pi/2)

    parser.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default='tb_logs/')

    args = parser.parse_args()

    main(args)