import SimpleITK as sitk
import numpy as np
from OrientCBCT import ResampleImage
import os 
import glob
import argparse
import multiprocessing as mp
import numpy as np
from time import sleep
from utils import CheckSharedList

def CenterImage(scan_path, out_path):
    img = sitk.ReadImage(scan_path)
    
    # Translation
    T = - np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    translation = sitk.TranslationTransform(3)
    translation.SetOffset(T.tolist())
    img_trans = ResampleImage(img,translation.GetInverse())
    sitk.WriteImage(img_trans, out_path)

def ListFiles(input_path):
    files = []
    normpath = os.path.normpath("/".join([input_path, '**', '']))
    for file in sorted(glob.iglob(normpath, recursive=True)):
        if os.path.isfile(file) and True in [ext in file for ext in [".nrrd", ".nii", ".nii.gz", 'gipl.gz']]:
            files.append(file)
    return files

def CenterBatch(out_dir,files,shared_list,num_worker):
    for file in files:
        outpath = os.path.join(out_dir,os.path.basename(file))
        if not os.path.exists(outpath):
            CenterImage(file,outpath)
            shared_list[num_worker] += 1


def main(args):
    out_dir = args.out_dir
    nb_worker = args.nb_proc

    manager = mp.Manager()

    nb_scan_done = manager.list([0 for i in range(nb_worker)])
    
    files = ListFiles(args.data_dir)
    check = mp.Process(target=CheckSharedList,args=(nb_scan_done,len(files)))
    check.start()

    
    if out_dir == '':
        out_dir = os.path.join(args.data_dir,'Output')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    splits = np.array_split(files,nb_worker)

    processes = [mp.Process(target=CenterBatch,args=(out_dir,splits[i],nb_scan_done,i)) for i in range(nb_worker)]

    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()
    
    check.join()

    print(nb_scan_done) 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',help='directory where json files to merge are',type=str,default='/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/Oriented/LargeFOV_RESAMPLED/MARILIA')#'/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/NotOriented/Anonymized')#required=True)
    parser.add_argument('--out_dir',help='directory where json files to merge are',type=str,default='/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/NotOriented/TEST')
    parser.add_argument('--nb_proc',help='number of processes to use for computation',type=int,default=5)
    args = parser.parse_args()
    main(args)