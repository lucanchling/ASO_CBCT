import numpy as np
from icecream import ic
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import SimpleITK as sitk
import os
import pandas as pd


class DataModuleClass(pl.LightningDataModule):
    # It is used to store information regarding batch size, transforms, etc. 
    def __init__(self, df_train, df_val, df_test, mount_point='./', batch_size=1, num_workers=4, train_transform=None, val_transform=None, test_transform=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def prepare_data(self):
    # On only ONE GPU. 
    # It’s usually used to handle the task of downloading the data. 
        pass

    def setup(self, stage=None):
    # On ALL the available GPU. 
    # It’s usually used to handle the task of loading the data. (like splitting data, applying transform etc.)
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = DatasetClass(df=self.df_train, mount_point=self.mount_point, transform=self.train_transform)
            self.val_dataset = DatasetClass(df=self.df_val, mount_point=self.mount_point, transform=self.val_transform)
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = DatasetClass(df=self.df_test, mount_point=self.mount_point, transform=self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class DatasetClass(Dataset):
    def __init__(self, df, mount_point='', transform=None):
        super().__init__()
        self.df = df
        self.transform = transform
        self.mount_point = mount_point

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        scan_path = self.df['scan_path'][idx]

        scan = sitk.ReadImage(scan_path)

        origin = np.array(scan.GetOrigin())
        spacing = np.array(scan.GetSpacing())
        
        directionVector = torch.Tensor([0,0,1])
        
        if self.transform is not None:
            scan, directionVector = self.transform(scan, directionVector)

        scan = torch.Tensor(sitk.GetArrayFromImage(scan)).unsqueeze(0)  # Conversion for Monai transforms
                
        return scan, directionVector, scan_path.split('/')[-1]

class RandomRotation3D(pl.LightningDataModule):
    def __init__(self, x_angle=np.pi/2, y_angle=np.pi/2, z_angle=np.pi/2):
        super().__init__()
        self.x_angle = x_angle
        self.y_angle = y_angle
        self.z_angle = z_angle

    def __call__(self, scan, directionVector):
        randanglex = np.random.uniform(-self.x_angle, self.x_angle)
        randangley = np.random.uniform(-self.y_angle, self.y_angle)
        randanglez = np.random.uniform(-self.z_angle, self.z_angle)
        
        R = sitk.Euler3DTransform()
        R.SetRotation(randanglex, randangley, randanglez)
        
        scan = sitk.Resample(image1=scan, transform=R, interpolator=sitk.sitkAffine)

        rotmatrix = np.array(R.GetMatrix()).reshape(3,3)

        directionVector = np.matmul(directionVector, rotmatrix)
            
        
        return scan, directionVector   

import matplotlib.pyplot as plt

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
    plt.show()

if __name__ == "__main__":

    data_dir="/home/luciacev/Desktop/Luc_Anchling/DATA/ALI_CBCT/RESAMPLED/"
    
    csv_path = os.path.join(data_dir, 'CSV/ALL')

    df_train = pd.read_csv(os.path.join(csv_path, 'train.csv'))
    df_val = pd.read_csv(os.path.join(csv_path, 'val.csv'))
    df_test = pd.read_csv(os.path.join(csv_path, 'test.csv'))

    dm = DataModuleClass(df_train=df_train, df_val=df_val, df_test=df_test, batch_size=25, num_workers=4, train_transform=RandomRotation3D(), val_transform=None, test_transform=None)

    dm.setup('fit')

    train_loader = dm.val_dataloader()

    for batch in train_loader:
        scan, directionVector, scan_name = batch
        # scan_name = [scan_name for scan_name in scan_name]
        # # find index of a specfic scan in the batch
        # try:
        #     idx = scan_name.index('40_T0_scan.nii.gz')
        # except:
        #     continue
        # # write the image to a file
        # img = sitk.GetImageFromArray(scan.squeeze(0).squeeze(0))
        
        # img.SetOrigin(origin.tolist()[0])
        # img.SetSpacing(spacing.tolist()[0])
        # if not os.path.exists(os.path.join(data_dir, 'TEST')):
        #     os.makedirs(os.path.join(data_dir, 'TEST'))
        # sitk.WriteImage(img, os.path.join(data_dir, 'TEST', scan_name[0]))
        
        
        # gen_plot(np.array([0,0,1]), directionVector[0].numpy(), scan_name[0])

        # break
