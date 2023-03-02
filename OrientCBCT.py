import numpy as np
from icecream import ic
import argparse
import torch
from OrientationDetection.Net import DenseNet
import SimpleITK as sitk
import matplotlib.pyplot as plt
import glob 
import os

from utils import RotationMatrix, AngleAndAxisVectors
from ResampleFunction import PreASOResample

cross = lambda x,y:np.cross(x,y)

def gen_plot(a,b,c):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.quiver(0, 0, 0, a[0], a[1], a[2], color='b', label='Detected')
    ax.quiver(0, 0, 0, b[0], b[1], b[2], color='r', label='Goal')
    ax.quiver(0, 0, 0, c[0], c[1], c[2], color='g', label='Corrected')  
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    ax.legend()
    # plt.title('DIRECTION:  {:.4f}  |  {:.4f}  |  {:.4f}\nDIRECTION_HAT :  {:.4f}  |  {:.4f}  |  {:.4f}'.format(direction[0],direction[1],direction[2],direction_hat[0],direction_hat[1],direction_hat[2]))
    plt.show()

def ResampleImage(image, transform):
    '''
    Resample image using SimpleITK
    
    Parameters
    ----------
    image : SimpleITK.Image
        Image to be resampled
    target : SimpleITK.Image
        Target image
    transform : SimpleITK transform
        Transform to be applied to the image.
        
    Returns
    -------
    SimpleITK image
        Resampled image.
    '''
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(image)
    resample.SetTransform(transform)
    resample.SetInterpolator(sitk.sitkLinear)
    orig_size = np.array(image.GetSize(), dtype=int)
    ratio = 1
    new_size = orig_size * ratio
    new_size = np.ceil(new_size).astype(int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    resample.SetDefaultPixelValue(0)

    # Set New Origin
    orig_origin = np.array(image.GetOrigin())
    # apply transform to the origin
    orig_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    # new_center = np.array(target.TransformContinuousIndexToPhysicalPoint(np.array(target.GetSize())/2.0))
    new_origin = orig_origin - orig_center #- np.array((10,10,10))
    resample.SetOutputOrigin(new_origin)

    return resample.Execute(image)

def main(args):

    # temp_folder = '/home/lucia/Documents/TEMP/'
    # PreASOResample(args.scan_folder,temp_folder)
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)       
    
    CosSim = torch.nn.CosineSimilarity() # /!\ if loss < 0.1 dont apply rotation /!\
    Loss = lambda x,y: 1 - CosSim(torch.Tensor(x),torch.Tensor(y))
    device = 'cuda'
    
    lr = float(args.checkpoint.split('_bs')[0].split('/')[-1].split('lr')[1])   
    DN_type = args.checkpoint.split('_lr')[0].split('/')[-1].split('DN_')[1]

    model = DenseNet(lr,DN_type)    
    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    
    # model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    model.to(device)
    model.eval()
    liste = []
    
    normpath = os.path.normpath("/".join([args.scan_folder,'**','']))
    for file in (sorted(glob.iglob(normpath,recursive=True))):
        if file.endswith('.nii.gz'):

            img = sitk.ReadImage(os.path.join(args.scan_folder,os.path.basename(file)))

            # Translation
            T = - np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
            translation = sitk.TranslationTransform(3)
            translation.SetOffset(T.tolist())
            # ic(os.path.basename(file),T)
            # img_trans = ResampleImage(img,translation.GetInverse())
            # sitk.WriteImage(img_trans,'/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/NotOriented/TEST/Out/Centered.nii.gz')
            
            goal = np.array((0.0,0.0,1.0))

            img_temp = sitk.ReadImage(file)
            array = sitk.GetArrayFromImage(img_temp)
            scan = torch.Tensor(array).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                directionVector_pred = model(scan.to(device))

            directionVector_pred = directionVector_pred.cpu().numpy()
            # ic(directionVector_pred)
            # if Loss(directionVector_pred,goal) > 0.1:# and np.min(array) >= -1200 :
            angle, axis = AngleAndAxisVectors(goal,directionVector_pred[0])
            angle = angle# * 2.5
            ic(os.path.basename(file),angle*180/np.pi,Loss(directionVector_pred,goal))#,np.min(array),np.max(array))
            Rotmatrix = RotationMatrix(axis,angle)
            afterRot = np.matmul(directionVector_pred[0],Rotmatrix)
            # gen_plot(directionVector_pred[0], goal, afterRot)
            
            # rotation = sitk.Euler3DTransform()
            # rotation = sitk.VersorRigid3DTransform()
            # Rotmatrix = np.linalg.inv(Rotmatrix)
            # rotation.SetMatrix(Rotmatrix.flatten().tolist())
            
            # TransformList = [translation,rotation]
            
            # # Compute the final transform (inverse all the transforms)
            # TransformSITK = sitk.CompositeTransform(3)
            # for i in range(len(TransformList)-1,-1,-1):
            #     TransformSITK.AddTransform(TransformList[i])
            # TransformSITK = TransformSITK.GetInverse()
            
            # img_out = ResampleImage(img_temp,TransformSITK)

            # outpath = os.path.join(args.output_folder,os.path.basename(file))
            # if not os.path.exists(outpath):
            #     sitk.WriteImage(img_out,outpath)
    
    '''
    '''
    #print(liste)
    # print(np.mean(liste,axis=0))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--scan_folder',default='/tmp/Slicer-lucia/__SlicerTemp__2023-03-01_17+55+11.960/')#'/home/lucia/Desktop/Luc/DATA/ASO/Test/')
    parser.add_argument('--output_folder',default='/home/lucia/Desktop/Luc/DATA/ASO/ACCURACY/Head/ASOTESTOUTPUT/')
    parser.add_argument('--checkpoint',default='/home/lucia/Desktop/Luc/Models/ASO_BIS/Auto_Or_Models/DN_169_lr1e-04_bs30_angle3.14.ckpt')
    args = parser.parse_args()

    main(args)