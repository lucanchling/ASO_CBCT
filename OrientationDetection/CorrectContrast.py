import SimpleITK as sitk
import numpy as np
import argparse
import glob
import os

def CorrectHisto(filepath,outpath,min_porcent=0.01,max_porcent = 0.95,i_min=-1000, i_max=4000):

    print("Correcting scan contrast :", filepath)
    input_img = sitk.ReadImage(filepath) 
    input_img = sitk.Cast(input_img, sitk.sitkFloat32)
    img = sitk.GetArrayFromImage(input_img)


    img_min = np.min(img)
    img_max = np.max(img)
    img_range = img_max - img_min
    # print(img_min,img_max,img_range)

    definition = 1000
    histo = np.histogram(img,definition)
    cum = np.cumsum(histo[0])
    cum = cum - np.min(cum)
    cum = cum / np.max(cum)

    res_high = list(map(lambda i: i> max_porcent, cum)).index(True)
    res_max = (res_high * img_range)/definition + img_min

    res_low = list(map(lambda i: i> min_porcent, cum)).index(True)
    res_min = (res_low * img_range)/definition + img_min

    res_min = max(res_min,i_min)
    res_max = min(res_max,i_max)


    # print(res_min,res_min)

    img = np.where(img > res_max, res_max,img)
    img = np.where(img < res_min, res_min,img)

    output = sitk.GetImageFromArray(img)
    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)
    return output

def main(args):

    normpath = os.path.normpath("/".join([args.input_dir, '**', '']))
    for file in sorted(glob.iglob(normpath, recursive=True)):
        #  print(img_fn)
        basename = os.path.basename(file)
        if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
            if not os.path.exists((file.replace(args.input_dir,args.output_dir)).split(os.path.basename(file))[0]):
                os.makedirs((file.replace(args.input_dir,args.output_dir)).split(os.path.basename(file))[0])
            CorrectHisto(file,file.replace(args.input_dir,args.output_dir))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir',default='')
    parser.add_argument('--output_dir',default='')
    args = parser.parse_args()

    main(args)