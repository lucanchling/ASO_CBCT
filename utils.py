
import json
import numpy as np
import vtk
import copy
import SimpleITK as sitk
import platform

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
    'dark blue': (0.1, 0.2, 0.4),
    'pink' : (1, 0.75, 0.8),
    'orange' : (1, 0.5, 0.5),
}

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


def LoadJsonLandmarks(img_path, ldmk_path, gold=False):
    print("Loading landmarks for {}...".format(img_path.split('/')[-1]))
    spacing, origin = LoadImage(img_path)
    print("Spacing: {}".format(spacing))
    print("Origin: {}".format(origin))
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

def PrintMatrix(transform):
    for i in range(4):
        for j in range(4):
            print(transform.GetElement(i,j), end=' ')
        print()


"""
 __      __  _______   _  __    _____    ______   _   _   _____    ______   _____  
 \ \    / / |__   __| | |/ /   |  __ \  |  ____| | \ | | |  __ \  |  ____| |  __ \ 
  \ \  / /     | |    | ' /    | |__) | | |__    |  \| | | |  | | | |__    | |__) |
   \ \/ /      | |    |  <     |  _  /  |  __|   | . ` | | |  | | |  __|   |  _  / 
    \  /       | |    | . \    | | \ \  | |____  | |\  | | |__| | | |____  | | \ \ 
     \/        |_|    |_|\_\   |_|  \_\ |______| |_| \_| |_____/  |______| |_|  \_\
                                                                                   
"""                                                                                   

def ConvertToVTKPoints(dict_landmarks):
    """
    Convert dictionary of landmarks to vtkPoints
    
    Parameters
    ----------
    dict_landmarks : dict
        Dictionary of landmarks with key as landmark name and value as landmark coordinates\
        Example: {'L1': [0, 0, 0], 'L2': [1, 1, 1], 'L3': [2, 2, 2]}

    Returns
    -------
    vtkPoints
        VTK points object
    """
    Points = vtkPoints()
    Vertices = vtkCellArray()
    labels = vtk.vtkStringArray()
    labels.SetNumberOfValues(len(dict_landmarks.keys()))
    labels.SetName("labels")

    for i,landmark in enumerate(dict_landmarks.keys()):
        sp_id = Points.InsertNextPoint(dict_landmarks[landmark])
        Vertices.InsertNextCell(1)
        Vertices.InsertCellPoint(sp_id)
        labels.SetValue(i, landmark)
        
    output = vtkPolyData()
    output.SetPoints(Points)
    output.SetVerts(Vertices)
    output.GetPointData().AddArray(labels)

    return output

def CreateActorLabel(source,color='white',convert_to_vtk=False):
    """
    Create a mapper and actor with labels
    
    Parameters
    ----------
    source : vtkPolyData
        source points
        color : str
        color of the points
        
    Returns
    -------
    actor : vtkActor
    labelActor : vtkActor2D
    """
    if convert_to_vtk:
        source = ConvertToVTKPoints(source)

    Mapper = vtk.vtkPolyDataMapper()
    Mapper.SetInputData(source)
    Actor = vtk.vtkActor()
    Actor.SetMapper(Mapper)
    Actor.GetProperty().SetColor(COLOR[color])
    Actor.GetProperty().SetPointSize(5)

    LabelMapper = vtk.vtkLabeledDataMapper()
    LabelMapper.SetInputData(source)
    LabelMapper.SetLabelModeToLabelFieldData()
    LabelMapper.SetFieldDataName("labels")
    LabelActor = vtk.vtkActor2D()
    LabelActor.SetMapper(LabelMapper)
    LabelActor.GetProperty().SetColor(COLOR[color])
    
    return Actor,LabelActor


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

def RenderWindow(Actors,backgroundColor='dark blue'):

    # Create a renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    for Actor in Actors:
        renderer.AddActor(Actor)
    renderer.SetBackground(COLOR[backgroundColor])

    # Render and interact
    renderWindow.Render()
    renderWindowInteractor.Start()

def vtk_render(source, target=None, transform=None, convert_to_vtk=False):
    """
    Render the source and target points for ICP Transform
    
    Parameters
    ----------
    source : vtkPoints
        Source points
    target : vtkPoints
        Target points
    transform : vtkIterativeClosestPointTransform
        ICP transform
    convert_to_vtk : bool
        Convert source and target to vtkPoints if True
    
    Returns
    -------
    None
    """
    if convert_to_vtk:
        source = ConvertToVTKPoints(source)
        if target is not None:
            target = ConvertToVTKPoints(target)
    
    # Create a mapper and actor

    sourceActor, sourceLabelActor = CreateActorLabel(source, 'white')
    Actors = [sourceActor, sourceLabelActor]
    if target is not None:
        targetActor, targetLabelActor = CreateActorLabel(target, 'green')
        Actors.extend([targetActor, targetLabelActor])


    # Apply transform
    if transform is not None:
        transformFilter = vtkTransformPolyDataFilter()
        transformFilter.SetInputData(source)
        transformFilter.SetTransform(transform)
        transformFilter.Update()

        # Create a mapper and actor
        transformActor, transformLabelActor = CreateActorLabel(transformFilter.GetOutput(), 'red')

        Actors.append(transformActor)
        Actors.append(transformLabelActor)
    
    RenderWindow(Actors)