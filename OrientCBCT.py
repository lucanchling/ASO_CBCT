import numpy as np
from icecream import ic
import argparse
import torch
from OrientationDetection.Net import DenseNet
import SimpleITK as sitk
import shutil
import matplotlib.pyplot as plt

from utils import RotationMatrix, AngleAndAxisVectors

cross = lambda x,y:np.cross(x,y)

def gen_plot(a,b,c):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.quiver(0, 0, 0, a[0], a[1], a[2], color='r', label='a')
    ax.quiver(0, 0, 0, b[0], b[1], b[2], color='b', label='b')
    ax.quiver(0, 0, 0, c[0], c[1], c[2], color='g', label='c')  
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
    CosSim = torch.nn.CosineSimilarity() # /!\ if loss < 0.1 dont apply rotation /!\
    Loss = lambda x,y: 1 - CosSim(torch.Tensor(x),torch.Tensor(y))
    
    model = DenseNet.load_from_checkpoint(args.checkpoint)
    # model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    model.to('cuda')   
    model.eval()
    
    num = args.num
    if num<10:
        num = '000'+str(num)
    elif num<100:
        num = '00'+str(num)
    elif num<1000:
        num = '0'+str(num)
    if args.tilted:
        scan_path = '/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/NotOriented/TEST/Out/tilted.nii.gz'
    else:
        scan_path = '/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/NotOriented/TEST/IC_'+num+'.nii.gz'

    # copy scan to output folder
    shutil.copy(scan_path,'/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/NotOriented/TEST/Out/scan.nii.gz')
    img = sitk.ReadImage(scan_path)
    scan = torch.Tensor(sitk.GetArrayFromImage(img)).unsqueeze(0).unsqueeze(0)

    # Translation
    T = - np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    translation = sitk.TranslationTransform(3)
    translation.SetOffset(T.tolist())
    
    img_trans = ResampleImage(img,translation.GetInverse())
    sitk.WriteImage(img_trans,'/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/NotOriented/TEST/Out/Centered.nii.gz')
    


    goal = np.array((0.0,0.0,1.0))

    with torch.no_grad():
        directionVector_pred = model(scan.to('cuda'))

    directionVector_pred = directionVector_pred.cpu().numpy()
    # ic(directionVector_pred)
    # if Loss(directionVector_pred,goal) > 0.2:
    ic(num,Loss(directionVector_pred,goal))
    angle, axis = AngleAndAxisVectors(goal,directionVector_pred[0])
    # ic(num,angle)
    Rotmatrix = RotationMatrix(axis,angle)
    # afterRot = np.matmul(Rotmatrix,directionVector_pred[0])
    # gen_plot(directionVector_pred[0], goal, afterRot)
    
    # rotation = sitk.Euler3DTransform()
    rotation = sitk.VersorRigid3DTransform()
    Rotmatrix = np.linalg.inv(Rotmatrix)
    rotation.SetMatrix(Rotmatrix.flatten().tolist())
    
    TransformList = [translation,rotation]
    
    # Compute the final transform (inverse all the transforms)
    TransformSITK = sitk.CompositeTransform(3)
    for i in range(len(TransformList)-1,-1,-1):
        TransformSITK.AddTransform(TransformList[i])
    TransformSITK = TransformSITK.GetInverse()
    
    img_out = ResampleImage(img,TransformSITK)
    
    sitk.WriteImage(img_out,'/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/NotOriented/TEST/Out/output.nii.gz')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint', type=str, default='/home/luciacev/Desktop/Luc_Anchling/Training_OR/NEW_LFOV/Models/lr1e-04_bs30_angle1.57.ckpt')
    parser.add_argument('--angle', type=float, default=np.pi/2)
    parser.add_argument('--num', type=int)
    parser.add_argument('--tilted', type=bool,default=False)
    parser.add_argument('--all',type=bool,default=False)

    args = parser.parse_args()
    
    # for i in range(1,146):
    #     args.num = i
    #     if i!=6:
    main(args)