# https://kitware.github.io/vtk-examples/site/Python/Filtering/IterativeClosestPoints/
# https://kitware.github.io/vtk-examples/site/Cxx/Filtering/IterativeClosestPointsTransform/

# NOT VERY PROPER WAY TO DO IT BUT IT WORKS

import numpy as np
import vtk
import platform
from utils import *
import time
import os
import shutil
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

def main(input_file, input_json_file, gold_json_file, gold_file,num):
    
    # Read input files
    input_image = sitk.ReadImage(input_file)
    # print('input spacing:',input_image.GetSpacing())
    gold_image = sitk.ReadImage(gold_file)
    # print('gold spacing:',gold_image.GetSpacing())
    source = LoadJsonLandmarks(input_image, input_json_file)
    target = LoadJsonLandmarks(gold_image, gold_json_file, gold=True)

    # Make sure the landmarks are in the same order
    source = SortDict(source)
    source_orig = source.copy()
    target = SortDict(target)

    # save the source and target landmarks arrays
    np.save('cache/source.npy', source)
    np.save('cache/target.npy', target)

    # load the source and target landmarks arrays
    # source = np.load('cache/source.npy', allow_pickle=True).item()
    # target = np.load('cache/target.npy', allow_pickle=True).item()
    # Actors = list(CreateActorLabel(source, color='white', convert_to_vtk=True)) # Original source landmarks
    Actors = []
    Actors.extend(list(CreateActorLabel(target, color='green', convert_to_vtk=True)))  # Original target landmarks
    
    # Apply Init ICP with only the best landmarks
    source_transformed, TransformMatrix, TransformList = InitICP(source,target,render=False, Print=False, BestLMList=FindOptimalLandmarks(source,target))
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
    # WriteJsonLandmarks(source_transformed, input_file,'data/output/output.json')
    # Actors.extend(list(CreateActorLabel(source, color='red', convert_to_vtk=True)))

    source = ApplyTransform(source_orig,TransformMatrixFinal)
    Actors.extend(list(CreateActorLabel(source, color='yellow', convert_to_vtk=True)))

    # Invert the transform matrix
    # TransformMatrixFinal = np.linalg.inv(TransformMatrixFinal)
    # print(TransformMatrixFinal)
    # test(np.load('cache/source.npy', allow_pickle=True).item(),target,TransformMatrixFinalInv)

    # Resample the source image with the final transform 
    print("Resampling...")
    tic = time.time()
    output = ResampleImage(input_image, gold_image, transform=TransformSITK)
    outpath = 'data/output/test/'+str(num)+'_output.nii.gz'
    sitk.WriteImage(output, outpath)
    file_size = os.path.getsize(outpath)
    print('Output Resampled and Saved in {:.2f} seconds (file_size: {:.2f}MB)'.format(time.time()-tic,file_size/1e6))
    return 0
    RenderWindow(Actors)


def FindOptimalLandmarks(source,target):
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
    while len(dist) < 210 and ii < 2500:
        ii+=1
        source = np.load('cache/source.npy', allow_pickle=True).item()
        firstpick,secondpick,thirdpick, d = InitICP(source,target,render=False, Print=False, search=True)
        if [firstpick,secondpick,thirdpick] not in LMlist:
            dist.append(d)
            LMlist.append([firstpick,secondpick,thirdpick])
    print("Min Dist: {:.2f} | for LM: {} | len = {}".format(min(dist),LMlist[dist.index(min(dist))],len(dist)))
    return LMlist[dist.index(min(dist))]

def WriteJsonLandmarks(landmarks, input_file ,output_file):
    '''
    Write the landmarks to a json file
    
    Parameters
    ----------
    landmarks : dict
        landmarks to write
    output_file : str
        output file name
    '''
    # # Load the input image
    # spacing, origin = LoadImage(input_file)
    inspacing, inorigin = np.load('cache/sourcedata.npy', allow_pickle=True)[0], np.load('cache/sourcedata.npy', allow_pickle=True)[1]
    goldspacing, goldorigin = np.load('cache/targetdata.npy', allow_pickle=True)[0], np.load('cache/targetdata.npy', allow_pickle=True)[1]
    
    with open('/home/luciacev/Desktop/Luc_Anchling/Projects/ASO_CBCT/TEMP.mrk.json', 'r') as outfile:
        tempData = json.load(outfile)
    for i in range(len(landmarks)):
        pos = landmarks[tempData['markups'][0]['controlPoints'][i]['label']]
        # pos = (pos + abs(inorigin)) * inspacing
        tempData['markups'][0]['controlPoints'][i]['position'] = [pos[0],pos[1],pos[2]]
    with open(output_file, 'w') as outfile:
        json.dump(tempData, outfile, indent=4)

if __name__ == '__main__':
    for num in range(7, 20):
        num
        if num < 10:
            num = "000" + str(num)
        elif num < 100:
            num = "00" + str(num)
        elif num < 1000:
            num = "0" + str(num)

        if platform.system() == "Darwin":
            input_file = '/Users/luciacev-admin/Desktop/Luc_Anchling/Projects/ASO_CBCT/data/IC_0086.nii.gz'
            input_json_file = '/Users/luciacev-admin/Desktop/Luc_Anchling/Projects/ASO_CBCT/data/IC_0086.mrk.json'
            gold_file = '/Users/luciacev-admin/Desktop/Luc_Anchling/Projects/ASO_CBCT/data/MAMP_0002_Or_T1.nii.gz'
            gold_json_file = '/Users/luciacev-admin/Desktop/Luc_Anchling/Projects/ASO_CBCT/data/MAMP_02_T1.mrk.json'
        elif platform.system() == "Linux":
            input_file = '/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/Anonymized/IC_'+num+'.nii.gz'
            shutil.copy(input_file, 'data/output/test/'+str(num)+'_input.nii.gz')
            input_json_file = '/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/Anonymized/Landmarks/IC_'+num+'.mrk.json'
            gold_json_file = '/home/luciacev/Desktop/Luc_Anchling/Projects/ASO_CBCT/data/Gold_Standard/GOLD_MAMP_02_T1.mrk.json'
            gold_file = '/home/luciacev/Desktop/Luc_Anchling/Projects/ASO_CBCT/data/Gold_Standard/GOLD_MAMP_0002_Or_T1.nii.gz'

        print('IC_'+num+'.nii.gz')
        main(input_file, input_json_file, gold_json_file, gold_file,num)
        print('='*70)
        # np.save('cache/sourcedata.npy', np.array(LoadImage(input_file)))
        # np.save('cache/targetdata.npy', np.array(LoadImage(gold_file)))