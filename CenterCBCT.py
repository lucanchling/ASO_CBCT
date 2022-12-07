import SimpleITK as sitk
import numpy as np
from OrientCBCT import ResampleImage
import os 
import glob
from tqdm import tqdm
import argparse

def CenterImage(scan_path, out_path):
    img = sitk.ReadImage(scan_path)
    
    # Translation
    T = - np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    translation = sitk.TranslationTransform(3)
    translation.SetOffset(T.tolist())
    print("resample")
    img_trans = ResampleImage(img,translation.GetInverse())
    print("Write")
    sitk.WriteImage(img_trans, out_path)
        

def main(args):

    files = []
    normpath = os.path.normpath("/".join([args.data_dir, '**', '']))
    for file in sorted(glob.iglob(normpath, recursive=True)):
        if os.path.isfile(file) and True in [ext in file for ext in [".nrrd", ".nii", ".nii.gz", 'gipl.gz']]:
            files.append(file)
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for file in tqdm(files,total=len(files)):
        outpath = os.path.join(args.out_dir,os.path.basename(file))
        CenterImage(file,outpath)
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',help='directory where json files to merge are',type=str,required=True)
    parser.add_argument('--out_dir',help='directory where json files to merge are',type=str,default=os.path.join(parser.parse_args().data_dir,'Output'))
    args = parser.parse_args()
    main(args)
    
    