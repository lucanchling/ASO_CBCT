# https://kitware.github.io/vtk-examples/site/Python/Filtering/IterativeClosestPoints/
# https://kitware.github.io/vtk-examples/site/Cxx/Filtering/IterativeClosestPointsTransform/

import json
import numpy as np
import vtk
import copy
import SimpleITK as sitk

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkIterativeClosestPointTransform,
    vtkPolyData
)
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter

COLOR = {
    'red': (1, 0, 0),
    'green': (0, 1, 0),
    'blue': (0, 0, 1),
    'yellow': (1, 1, 0),
    'cyan': (0, 1, 1),
    'magenta': (1, 0, 1),
    'white': (1, 1, 1),
    'black': (0, 0, 0),
}


def LoadJsonLandmarks(img_path, ldmk_path, gold=False):
    # print("Loading landmarks for {}...".format(img_path.split('/')[-1]))
    spacing, origin = LoadImage(img_path)
    # print("Spacing: {}".format(spacing))
    # print("Origin: {}".format(origin))
    with open(ldmk_path) as f:
        data = json.load(f)
    
    markups = data["markups"][0]["controlPoints"]
    
    landmarks = {}
    for markup in markups:
        lm_ph_coord = np.array([markup["position"][2],markup["position"][1],markup["position"][0]])
        lm_coord = ((lm_ph_coord + abs(origin)) / spacing).astype(np.float16)

        landmarks[markup["label"]] = lm_coord
    return landmarks

def LoadImage(image_path):
    img = sitk.ReadImage(image_path)
    spacing = np.array(img.GetSpacing())
    origin = img.GetOrigin()
    origin = np.array([origin[2],origin[1],origin[0]])

    return spacing, origin

def SortDict(input_dict):
    output_dict = {}
    for key in sorted(input_dict.keys()):
        output_dict[key] = input_dict[key]
    return output_dict

def save_transform_txt(transform, file_path):
    with open(file_path, 'a') as f:
        f.write(str(transform)+'\n')


def PrintMatrix(transform):
    for i in range(4):
        for j in range(4):
            print(transform.GetElement(i,j), end=' ')
        print()

def vtk_render(source, target, transform=None):
    # Create a mapper and actor
    sourceMapper = vtk.vtkPolyDataMapper()
    sourceMapper.SetInputData(source)
    sourceActor = vtk.vtkActor()
    sourceActor.SetMapper(sourceMapper)
    sourceActor.GetProperty().SetColor(COLOR['white'])
    sourceActor.GetProperty().SetPointSize(5)

    targetMapper = vtk.vtkPolyDataMapper()
    targetMapper.SetInputData(target)
    targetActor = vtk.vtkActor()
    targetActor.SetMapper(targetMapper)
    targetActor.GetProperty().SetColor(COLOR['green'])
    targetActor.GetProperty().SetPointSize(5)

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    # Add the actors to the scene
    renderer.AddActor(sourceActor)
    renderer.AddActor(targetActor)
    renderer.SetBackground(0.1, 0.2, 0.4) # Background color dark blue

    # Apply transform
    if transform is not None:
        transformFilter = vtkTransformPolyDataFilter()
        transformFilter.SetInputData(source)
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        # Create a mapper and actor
        transformMapper = vtk.vtkPolyDataMapper()
        transformMapper.SetInputData(transformFilter.GetOutput())
        transformActor = vtk.vtkActor()
        transformActor.SetMapper(transformMapper)
        transformActor.GetProperty().SetColor(COLOR['red'])
        transformActor.GetProperty().SetPointSize(5)

        renderer.AddActor(transformActor)
    
    # # Render and interact
    renderWindow.Render()
    renderWindowInteractor.Start()

def vtkmatrix_to_numpy(matrix):
    """
    Copies the elements of a vtkMatrix4x4 into a numpy array.

    :param matrix: The matrix to be copied into an array.
    :type matrix: vtk.vtkMatrix4x4
    :rtype: numpy.ndarray
    """
    m = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            m[i, j] = matrix.GetElement(i, j)
    return m

def ComputeErrorInPercent(source, target, transform):
    """
    Computes the error of the transform.

    :param source: The source points.
    :type source: vtk.vtkPoints
    :param target: The target points.
    :type target: vtk.vtkPoints
    :param transform: The transform to be evaluated.
    :type transform: vtk.vtkTransform
    :rtype: float
    """
    # Create a transform filter
    transformFilter = vtkTransformPolyDataFilter()
    transformFilter.SetInputData(source)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    # Compute the error
    sourcePoints = source.GetPoints()
    targetPoints = target.GetPoints()
    transformPoints = transformFilter.GetOutput().GetPoints()
    error = 0.0
    for i in range(sourcePoints.GetNumberOfPoints()):
        error += np.linalg.norm(np.array(transformPoints.GetPoint(i)) - np.array(targetPoints.GetPoint(i)))
    error /= sourcePoints.GetNumberOfPoints()
    return error / np.linalg.norm(np.array(source.GetCenter()) - np.array(target.GetCenter())) * 100




def ICP_Transform(source, target):
    # ICP Transform
    # ============ create source points ==============
    # print("Creating source points...")   
    sourcePoints = vtkPoints()
    sourceVertices = vtkCellArray()
    
    for i,landmark in enumerate(source.keys()):
        sp_id = sourcePoints.InsertNextPoint(source[landmark])
        sourceVertices.InsertNextCell(1)
        sourceVertices.InsertCellPoint(sp_id)
        
    source = vtkPolyData()
    source.SetPoints(sourcePoints)
    source.SetVerts(sourceVertices)

    # ============ create target points ==============
    # print("Creating target points...")
    targetPoints = vtkPoints()
    targetVertices = vtkCellArray()

    for i,landmark in enumerate(target.keys()):
        tp_id = targetPoints.InsertNextPoint(target[landmark])
        targetVertices.InsertNextCell(1)
        targetVertices.InsertCellPoint(tp_id)
    
    target = vtkPolyData()
    target.SetPoints(targetPoints)
    target.SetVerts(targetVertices)

    # ============ render source and target points ==============
    # vtk_render(source, target)

    # ============ create ICP transform ==============
    # print("Creating ICP transform...")
    icp = vtkIterativeClosestPointTransform()
    icp.SetSource(source)
    icp.SetTarget(target)
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.SetMaximumNumberOfIterations(100)
    icp.StartByMatchingCentroidsOn()
    icp.Modified()
    icp.Update()

    # ============ apply ICP transform ==============
    # print("Applying ICP transform...")
    transformFilter = vtkTransformPolyDataFilter()
    transformFilter.SetInputData(source)
    transformFilter.SetTransform(icp)
    transformFilter.Update()

    
    


    return source,target,icp

def first_ICP(source,target,render=False):
    source,target,icp = ICP_Transform(source,target)
    print("ICP error:{:.2f}%".format(ComputeErrorInPercent(source, target, icp)))
    PrintMatrix(icp.GetMatrix())
    if render:
        vtk_render(source, target, icp)


def main(input_file, input_json_file, gold_json_file, gold_file):
    
    source = LoadJsonLandmarks(input_file, input_json_file)
    target = LoadJsonLandmarks(gold_file, gold_json_file, gold=True)

    # Make sure the landmarks are in the same order
    source = SortDict(source)
    target = SortDict(target)

    first_ICP(source,target,render=False)

    # print()
    


if __name__ == '__main__':
    for num in [86]:#,96,34]:
        num
        if num < 10:
            num = "000" + str(num)
        elif num < 100:
            num = "00" + str(num)
        elif num < 1000:
            num = "0" + str(num)

        input_file = '/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/Anonymized/IC_'+num+'.nii.gz'
        input_json_file = '/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/Anonymized/Landmarks/IC_'+num+'.mrk.json'
        gold_json_file = '/home/luciacev/Desktop/Luc_Anchling/Projects/ASO_CBCT/data/Gold_Standard/GOLD_MAMP_02_T1.mrk.json'
        gold_file = '/home/luciacev/Desktop/Luc_Anchling/Projects/ASO_CBCT/data/Gold_Standard/GOLD_MAMP_0002_Or_T1.nii.gz'

        print('IC_'+num+'.nii.gz')
        main(input_file, input_json_file, gold_json_file, gold_file)


    