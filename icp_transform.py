# https://kitware.github.io/vtk-examples/site/Python/Filtering/IterativeClosestPoints/
# https://kitware.github.io/vtk-examples/site/Cxx/Filtering/IterativeClosestPointsTransform/

# NOT VERY PROPER WAY TO DO IT BUT IT WORKS
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import vtk
import platform
from utils import *

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkIterativeClosestPointTransform,
    vtkPolyData
)
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter

cross = lambda x,y:np.cross(x,y) # to avoid unreachable code error on np.cross function

def ICP_Transform(source, target):

    # ============ create source points ==============
    source = ConvertToVTKPoints(source)

    # ============ create target points ==============
    target = ConvertToVTKPoints(target)

    # ============ render source and target points ==============
    # vtk_render(source, target)

    # ============ create ICP transform ==============
    icp = vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(100)
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
        vtk_render(source, target, transform=icp)

def ApplyTranslation(source,transform):
    sourcee = source.copy()
    for key in sourcee.keys():
        sourcee[key] = sourcee[key] + transform
    return sourcee


def AngleAndAxisVectors(v1, v2):
    # Compute angle between two vectors
    v1_u = v1 / np.amax(v1)
    v2_u = v2 / np.amax(v2)
    angle = np.arccos(np.dot(v1_u, v2_u) / (np.linalg.norm(v1_u) * np.linalg.norm(v2_u)))
    axis = cross(v1_u, v2_u)
    #axis = axis / np.linalg.norm(axis)
    return angle,axis

def RotationMatrix(angle,axis):
    # Compute Rotation matrix
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(angle) + axis[0] ** 2 * (1 - np.cos(angle))
    R[0, 1] = axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle)
    R[0, 2] = axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)
    R[1, 0] = axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle)
    R[1, 1] = np.cos(angle) + axis[1] ** 2 * (1 - np.cos(angle))
    R[1, 2] = axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)
    R[2, 0] = axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle)
    R[2, 1] = axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle)
    R[2, 2] = np.cos(angle) + axis[2] ** 2 * (1 - np.cos(angle))
    return R

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    import math
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def ApplyRotation(source,R):
    sourcee = source.copy()
    for key in sourcee.keys():
        sourcee[key] = np.dot(R,sourcee[key])
    return sourcee

def InitICP(source,target):
    labels = list(source.keys())
    Actors = []

    # Pick a random landmark
    firstpick = labels[np.random.randint(0, len(labels))]
    print("First pick: {}".format(firstpick))
    SPt1 = source[firstpick]
    TPt1 = target[firstpick]

    Actors.extend(list(CreateActorLabel(source,color='white',convert_to_vtk=True)))  # Original source landmarks
    Actors.extend(list(CreateActorLabel(target,color='green',convert_to_vtk=True))) # Original target landmarks

    # Compute Translation transform
    T = TPt1 - SPt1
    
    # Apply Translation transform
    source = ApplyTranslation(source,T)
    # Actors.extend(list(CreateActorLabel(source,color='red',convert_to_vtk=True))) # Translated source landmarks
    SPt1 = source[firstpick]

    # Pick another random landmark
    while True:
        secondpick = labels[np.random.randint(0, len(labels))]
        SPt2 = source[secondpick]
        TPt2 = target[secondpick]

        # Compute Rotation angle and vector
        v1 = abs(SPt2 - SPt1)
        v2 = abs(TPt2 - TPt1)
        angle,axis = AngleAndAxisVectors(v2, v1)
        if secondpick != firstpick and angle != 0:
            break
    print("Second pick: {}".format(secondpick))
    
    print("Angle: {:.4f}".format(angle))
    # print("Angle: {:.2f}°".format(angle*180/np.pi))

    # Compute Rotation matrix
    R = rotation_matrix(axis,angle)
    # Apply Rotation transform
    source = ApplyRotation(source,R)
    Actors.extend(list(CreateActorLabel(source,color='yellow',convert_to_vtk=True))) # Rotated source landmarks
    
    # print("Rotation:\n{}".format(R))
    
    # Pick another random landmark
    while True:
        thirdpick = labels[np.random.randint(0, len(labels))]
        SPt3 = source[thirdpick]
        TPt3 = target[thirdpick]
        if thirdpick != firstpick and thirdpick != secondpick:
            break
    print("Third pick: {}".format(thirdpick))

    # Compute Rotation angle
    v1 = abs(SPt3 - source[firstpick])
    v2 = abs(TPt3 - TPt1)
    angle,axis = AngleAndAxisVectors(v2, v1)
    print("Angle: {:.4f}".format(angle))

    # Compute Rotation matrix
    R = rotation_matrix(abs(source[secondpick] - source[firstpick]),angle)

    # Apply Rotation transform
    source = ApplyRotation(source,R)
    Actors.extend(list(CreateActorLabel(source,color='orange',convert_to_vtk=True))) # Rotated source landmarks

    # RenderWindow(Actors)

    return source

def main(input_file, input_json_file, gold_json_file, gold_file):
    
    # source = LoadJsonLandmarks(input_file, input_json_file)
    # target = LoadJsonLandmarks(gold_file, gold_json_file, gold=True)

    # # Make sure the landmarks are in the same order
    # source = SortDict(source)
    # target = SortDict(target)

    # # save the source and target landmarks arrays
    # np.save('cache/source.npy', source)
    # np.save('cache/target.npy', target)

    # load the source and target landmarks arrays
    source = np.load('cache/source.npy', allow_pickle=True).item()
    target = np.load('cache/target.npy', allow_pickle=True).item()


    source = InitICP(source,target)

    first_ICP(source,target,render=True)

if __name__ == '__main__':
    for num in [3]:#,96,34]:
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
            input_json_file = '/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/Anonymized/Landmarks/IC_'+num+'.mrk.json'
            gold_json_file = '/home/luciacev/Desktop/Luc_Anchling/Projects/ASO_CBCT/data/Gold_Standard/GOLD_MAMP_02_T1.mrk.json'
            gold_file = '/home/luciacev/Desktop/Luc_Anchling/Projects/ASO_CBCT/data/Gold_Standard/GOLD_MAMP_0002_Or_T1.nii.gz'

        # print('IC_'+num+'.nii.gz')
        main(input_file, input_json_file, gold_json_file, gold_file)


    