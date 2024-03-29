# https://kitware.github.io/vtk-examples/site/Python/Filtering/IterativeClosestPoints/
# https://kitware.github.io/vtk-examples/site/Cxx/Filtering/IterativeClosestPointsTransform/

# NOT VERY PROPER WAY TO DO IT BUT IT WORKS

import numpy as np
from utils import *
import time
import os
import glob
from icecream import ic
import argparse
from tqdm import tqdm
import shutil
import multiprocessing as mp

'''
8888888  .d8888b.  8888888b.      
  888   d88P  Y88b 888   Y88b         
  888   888    888 888    888
  888   888        888   d88P
  888   888        8888888P"          
  888   888    888 888 
  888   Y88b  d88P 888 
8888888  "Y8888P"  888
'''
def ICP_Transform(source, target):

    # ============ create source points ==============
    source = ConvertToVTKPoints(source)

    # ============ create target points ==============
    target = ConvertToVTKPoints(target)

    # ============ render source and target points ==============
    # VTKRender(source, target)

    # ============ create ICP transform ==============
    icp = vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(1000)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    # print("Number of iterations: {}".format(icp.GetNumberOfIterations()))

    # ============ apply ICP transform ==============
    transformFilter = vtkTransformPolyDataFilter()
    transformFilter.SetInputData(source)
    transformFilter.SetTransform(icp)
    transformFilter.Update()

    return source,target,icp

def first_ICP(source,target,render=False):
    source,target,icp = ICP_Transform(source,target)
    # print("ICP error:{:.2f}%".format(ComputeErrorInPercent(source, target, icp)))
    # PrintMatrix(icp.GetMatrix())
    if render:
        VTKRender(source, target, transform=icp)
    return VTKMatrixToNumpy(icp.GetMatrix())

'''
8888888  .d8888b.  8888888b.      8888888 888b    888 8888888 88888888888 
  888   d88P  Y88b 888   Y88b       888   8888b   888   888       888     
  888   888    888 888    888       888   88888b  888   888       888     
  888   888        888   d88P       888   888Y88b 888   888       888     
  888   888        8888888P"        888   888 Y88b888   888       888     
  888   888    888 888              888   888  Y88888   888       888     
  888   Y88b  d88P 888              888   888   Y8888   888       888     
8888888  "Y8888P"  888            8888888 888    Y888 8888888     888
'''

def InitICP(source,target,render=False, Print=False, BestLMList=None, search=False):
    TransformList = []
    # TransformMatrix = np.eye(4)
    TranslationTransformMatrix = np.eye(4)
    RotationTransformMatrix = np.eye(4)

    labels = list(source.keys())
    if render:
        Actors = []
    if BestLMList is not None:
        firstpick, secondpick, thirdpick = BestLMList[0], BestLMList[1], BestLMList[2]
        if Print:
            print("Best Landmarks are: {},{},{}".format(firstpick, secondpick, thirdpick))
    # print("Mean Distance:{:.2f}".format(ComputeMeanDistance(source, target)))

    # ============ Pick a Random Landmark ==============
    if BestLMList is None:
        firstpick = labels[np.random.randint(0, len(labels))]
        # firstpick = 'LOr'
    if Print:
        print("First pick: {}".format(firstpick))

    if render:
        Actors.extend(list(CreateActorLabel(source,color='white',convert_to_vtk=True)))  # Original source landmarks
        Actors.extend(list(CreateActorLabel(target,color='green',convert_to_vtk=True))) # Original target landmarks
    # ============ Compute Translation Transform ==============
    T = target[firstpick] - source[firstpick]
    TranslationTransformMatrix[:3, 3] = T
    Translationsitk = sitk.TranslationTransform(3)
    Translationsitk.SetOffset(T.tolist())
    TransformList.append(Translationsitk)
    # ============ Apply Translation Transform ==============
    source = ApplyTranslation(source,T)
    # source = ApplyTransform(source, TranslationTransformMatrix)

    if render:
        # Actors.extend(list(CreateActorLabel(source,color='red',convert_to_vtk=True))) # Translated source landmarks
        pass
    # print("Mean Distance:{:.2f}".format(ComputeMeanDistance(source, target)))

    # ============ Pick Another Random Landmark ==============
    if BestLMList is None:
        while True:
            secondpick = labels[np.random.randint(0, len(labels))]
            # secondpick = 'ROr'
            if secondpick != firstpick:
                break
    if Print:
        print("Second pick: {}".format(secondpick))

    # ============ Compute Rotation Angle and Axis ==============
    v1 = abs(source[secondpick] - source[firstpick])
    v2 = abs(target[secondpick] - target[firstpick])
    angle,axis = AngleAndAxisVectors(v2, v1)

    # print("Angle: {:.4f}".format(angle))
    # print("Angle: {:.2f}°".format(angle*180/np.pi))

    # ============ Compute Rotation Transform ==============
    R = RotationMatrix(axis,angle)
    # TransformMatrix[:3, :3] = R
    RotationTransformMatrix[:3, :3] = R
    Rotationsitk = sitk.VersorRigid3DTransform()
    Rotationsitk.SetMatrix(R.flatten().tolist())
    TransformList.append(Rotationsitk)
    # ============ Apply Rotation Transform ==============
    # source = ApplyRotation(source,R)
    source = ApplyTransform(source, RotationTransformMatrix)
    if render:
        # Actors.extend(list(CreateActorLabel(source,color='yellow',convert_to_vtk=True))) # Rotated source landmarks
        pass
    # print("Mean Distance:{:.2f}".format(ComputeMeanDistance(source, target)))
    # print("Rotation:\n{}".format(R))
    
    # ============ Compute Transform Matrix (Rotation + Translation) ==============
    TransformMatrix = RotationTransformMatrix @ TranslationTransformMatrix

    # ============ Pick another Random Landmark ==============
    if BestLMList is None:
        while True:
            thirdpick = labels[np.random.randint(0, len(labels))]
            # thirdpick = 'Ba'
            if thirdpick != firstpick and thirdpick != secondpick:
                break
    if Print:
        print("Third pick: {}".format(thirdpick))
    
    # ============ Compute Rotation Angle and Axis ==============
    v1 = abs(source[thirdpick] - source[firstpick])
    v2 = abs(target[thirdpick] - target[firstpick])
    angle,axis = AngleAndAxisVectors(v2, v1)
    # print("Angle: {:.4f}".format(angle))

    # ============ Compute Rotation Transform ==============
    RotationTransformMatrix = np.eye(4)
    R = RotationMatrix(abs(source[secondpick] - source[firstpick]),angle)
    RotationTransformMatrix[:3, :3] = R
    Rotationsitk = sitk.VersorRigid3DTransform()
    Rotationsitk.SetMatrix(R.flatten().tolist())
    TransformList.append(Rotationsitk)
    # ============ Apply Rotation Transform ==============
    # source = ApplyRotation(source,R)
    source = ApplyTransform(source, RotationTransformMatrix)

    # ============ Compute Transform Matrix (Init ICP) ==============
    TransformMatrix = RotationTransformMatrix @ TransformMatrix

    if render:
        Actors.extend(list(CreateActorLabel(source,color='orange',convert_to_vtk=True))) # Rotated source landmarks
    if Print:
        print("Mean Distance:{:.2f}".format(ComputeMeanDistance(source, target)))
    
    if render:
        RenderWindow(Actors)

    # return source
    if search:
        return firstpick,secondpick,thirdpick, ComputeMeanDistance(source, target)

    return source, TransformMatrix, TransformList

'''
888b     d888        d8888 8888888 888b    888 
8888b   d8888       d88888   888   8888b   888 
88888b.d88888      d88P888   888   88888b  888 
888Y88888P888     d88P 888   888   888Y88b 888 
888 Y888P 888    d88P  888   888   888 Y88b888 
888  Y8P  888   d88P   888   888   888  Y88888 
888   "   888  d8888888888   888   888   Y8888 
888       888 d88P     888 8888888 888    Y888
'''

def ICP(input_file,input_json_file,gold_file,gold_json_file,list_landmark):
    
    # Read input files
    input_image = sitk.ReadImage(input_file)
    # print('input spacing:',input_image.GetSpacing())
    gold_image = sitk.ReadImage(gold_file)
    # print('gold spacing:',gold_image.GetSpacing())
    source = LoadJsonLandmarks(input_image, input_json_file,list_landmark)
    target = LoadJsonLandmarks(gold_image, gold_json_file,list_landmark, gold=True)

    nb_lmrk = len(source.keys())

    # Make sure the landmarks are in the same order
    source = SortDict(source)
    source_orig = source.copy()
    target = SortDict(target)

    # save the source and target landmarks arrays
    # np.save('cache/source.npy', source)
    # np.save('cache/target.npy', target)

    # load the source and target landmarks arrays
    # source = np.load('cache/source.npy', allow_pickle=True).item()
    # target = np.load('cache/target.npy', allow_pickle=True).item()
    # Actors = list(CreateActorLabel(source, color='white', convert_to_vtk=True)) # Original source landmarks
    Actors = []
    Actors.extend(list(CreateActorLabel(target, color='green', convert_to_vtk=True)))  # Original target landmarks
    
    # Apply Init ICP with only the best landmarks
    source_transformed, TransformMatrix, TransformList = InitICP(source,target,render=False, Print=False, BestLMList=FindOptimalLandmarks(source,target,nb_lmrk))
    # Actors.extend(list(CreateActorLabel(source, color='pink', convert_to_vtk=True)))    # Init ICP Transformed source landmarks
    
    # Apply ICP
    TransformMatrixBis = first_ICP(source_transformed,target,render=False) 

    # Split the transform matrix into translation and rotation simpleitk transform
    TransformMatrixsitk = sitk.VersorRigid3DTransform()
    TransformMatrixsitk.SetTranslation(TransformMatrixBis[:3, 3].tolist())
    try:
        TransformMatrixsitk.SetMatrix(TransformMatrixBis[:3, :3].flatten().tolist())
    except RuntimeError:
        print('Error: The rotation matrix is not orthogonal')
        mat = TransformMatrixBis[:3, :3]
        print(mat)
        print('det:', np.linalg.det(mat))
        print('AxA^T:', mat @ mat.T)
    TransformList.append(TransformMatrixsitk)



    # Compute the final transform (inverse all the transforms)
    TransformSITK = sitk.CompositeTransform(3)
    for i in range(len(TransformList)-1,-1,-1):
        TransformSITK.AddTransform(TransformList[i])

    TransformSITK = TransformSITK.GetInverse()
    # Write the transform to a file
    # sitk.WriteTransform(TransformSITK, 'data/output/transform.tfm')

    TransformMatrixFinal = TransformMatrixBis @ TransformMatrix
    # print(TransformMatrixFinal)
    
    # Apply the final transform matrix
    source_transformed = ApplyTransform(source_transformed,TransformMatrixBis)
    # Actors.extend(list(CreateActorLabel(source, color='red', convert_to_vtk=True)))

    source = ApplyTransform(source_orig,TransformMatrixFinal)
    Actors.extend(list(CreateActorLabel(source, color='yellow', convert_to_vtk=True)))

    # Invert the transform matrix
    # TransformMatrixFinal = np.linalg.inv(TransformMatrixFinal)
    # print(TransformMatrixFinal)
    # test(np.load('cache/source.npy', allow_pickle=True).item(),target,TransformMatrixFinalInv)

    # Resample the source image with the final transform 
    # print("Resampling...")
    # tic = time.time()
    output = ResampleImage(input_image, gold_image, transform=TransformSITK)
    return output,source_transformed
    RenderWindow(Actors)

def FindOptimalLandmarks(source,target,nb_lmrk):
    '''
    Find the optimal landmarks to use for the Init ICP
    
    Parameters
    ----------
    source : dict
        source landmarks
    target : dict
        target landmarks
    
    Returns
    -------
    list
        list of the optimal landmarks
    '''
    dist, LMlist,ii = [],[],0
    while len(dist) < (nb_lmrk*(nb_lmrk-1)*(nb_lmrk-2)) and ii < 2500:
        ii+=1
        firstpick,secondpick,thirdpick, d = InitICP(source,target,render=False, Print=False, search=True)
        if [firstpick,secondpick,thirdpick] not in LMlist:
            dist.append(d)
            LMlist.append([firstpick,secondpick,thirdpick])
    WriteTXT("Min Dist: {:.2f} | for LM: {} | len = {}".format(min(dist),LMlist[dist.index(min(dist))],len(dist)),'sumup.txt')
    return LMlist[dist.index(min(dist))]

def WriteJsonLandmarks(landmarks, input_json_file ,output_file):
    '''
    Write the landmarks to a json file
    
    Parameters
    ----------
    landmarks : dict
        landmarks to write
    output_file : str
        output file name
    '''
    with open(input_json_file, 'r') as outfile:
        tempData = json.load(outfile)
    for i in range(len(landmarks)):
        pos = landmarks[tempData['markups'][0]['controlPoints'][i]['label']]
        # pos = (pos + abs(inorigin)) * inspacing
        tempData['markups'][0]['controlPoints'][i]['position'] = [pos[0],pos[1],pos[2]]
    shutil.copy(input_json_file,output_file)
    with open(output_file, 'w') as outfile:
        json.dump(tempData, outfile, indent=4)

def RunBatch(input_dir, out_dir, list_landmark, gold_json_file, gold_file, input_files, input_json_files, shared_list, num_worker):
    for i in range(len(input_files)):
        input_file,input_json_file = input_files[i],input_json_files[i]

        
        # print("Working on scan {} with lm {}".format(os.path.basename(input_file),os.path.basename(input_json_file)))
        WriteTXT("Working on scan {} with lm {}".format(os.path.basename(input_file),os.path.basename(input_json_file)),'sumup.txt')
        tic = time.time()
        output,source_transformed = ICP(input_file,input_json_file,gold_file,gold_json_file,list_landmark)
        
        # Write JSON
        dir_json = os.path.dirname(input_json_file.replace(input_dir,out_dir))
        if not os.path.exists(dir_json):
            os.makedirs(dir_json)
        
        json_path = os.path.join(dir_json,os.path.basename(input_json_file).split('.mrk.json')[0]+'_Or.mrk.json')

        WriteJsonLandmarks(source_transformed, input_json_file, output_file=json_path)

        # Write Scan
        dir_scan = os.path.dirname(input_file.replace(input_dir,out_dir))
        if not os.path.exists(dir_scan):
            os.makedirs(dir_scan)
        
        file_outpath = os.path.join(dir_scan,os.path.basename(input_file).split('.')[0]+'_Or.nii.gz')
        sitk.WriteImage(output, file_outpath)
        shared_list[num_worker]+=1
        #file_size = os.path.getsize(file_outpath)
        WriteTXT("Done in {:.2f} seconds".format(time.time()-tic),'sumup.txt')
        WriteTXT("="*70,'sumup.txt')

def main(args):
    input_dir, gold_dir, out_dir, list_landmark, nb_worker = args.data_dir,args.gold_dir,args.out_dir,args.list_landmark, args.nb_proc
    #ic(input_dir, gold_dir, out_dir,nb_lmrk)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    normpath = os.path.normpath("/".join([gold_dir, '**', '']))
    for file in glob.iglob(normpath, recursive=True):
        if os.path.isfile(file) and True in [ext in file for ext in ["json"]]:
            gold_json_file = file
        if os.path.isfile(file) and True in [ext in file for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", ".jpg", ".png", 'gipl.gz']]:
            gold_file = file
    
    input_files = []
    input_json_files = []
    normpath = os.path.normpath("/".join([input_dir, '**', '']))
    for file in sorted(glob.iglob(normpath, recursive=True)):
        if os.path.isfile(file) and True in [ext in file for ext in ["json"]]:
            input_json_files.append(file)
        if os.path.isfile(file) and True in [ext in file for ext in [".nrrd", ".nii", ".nii.gz", ".mhd", ".dcm", ".DCM", ".jpg", ".png", 'gipl.gz']]:
            input_files.append(file)

    manager = mp.Manager()

    nb_scan_done = manager.list([0 for i in range(nb_worker)])

    check = mp.Process(target=CheckSharedList,args=(nb_scan_done,len(input_files)))
    check.start()

    scan_splits = np.array_split(input_files,nb_worker)
    json_splits = np.array_split(input_json_files,nb_worker)

    processes = [mp.Process(target=RunBatch,args=(input_dir, out_dir, list_landmark, gold_json_file, gold_file, scan_splits[i], json_splits[i],nb_scan_done,i)) for i in range(nb_worker)]

    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()
    
    check.join()
    
        
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',help='directory where json files to merge are',type=str,default='/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/NotOriented/SaraCentered/')
    parser.add_argument('--gold_dir',help='directory where json files to merge are',type=str,default='/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/GOLD/Sara/')
    parser.add_argument('--list_landmark',help='List of landmarks used for the ICP',type=list,default=['IF','ANS','UR6O','UL6O','UR1O','PNS'])
    parser.add_argument('--out_dir',help='directory where json files to merge are',type=str,default = '/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/NotOriented/TEST/')#os.path.join(parser.parse_args().data_dir,'Output'))
    parser.add_argument('--nb_proc',help='Number of processes to use for computation',type=int,default=5)
    args = parser.parse_args()
    main(args)