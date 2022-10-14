'''
8888888 888b     d888 8888888b.   .d88888b.  8888888b.  88888888888  .d8888b.  
  888   8888b   d8888 888   Y88b d88P" "Y88b 888   Y88b     888     d88P  Y88b 
  888   88888b.d88888 888    888 888     888 888    888     888     Y88b.      
  888   888Y88888P888 888   d88P 888     888 888   d88P     888      "Y888b.   
  888   888 Y888P 888 8888888P"  888     888 8888888P"      888         "Y88b. 
  888   888  Y8P  888 888        888     888 888 T88b       888           "888 
  888   888   "   888 888        Y88b. .d88P 888  T88b      888     Y88b  d88P 
8888888 888       888 888         "Y88888P"  888   T88b     888      "Y8888P"  
'''

import json
import numpy as np
import vtk
import SimpleITK as sitk

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkIterativeClosestPointTransform,
    vtkPolyData
)
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter


cross = lambda x,y:np.cross(x,y) # to avoid unreachable code error on np.cross function

'''
888     888        d8888 8888888b.  8888888        d8888 888888b.   888      8888888888  .d8888b.  
888     888       d88888 888   Y88b   888         d88888 888  "88b  888      888        d88P  Y88b 
888     888      d88P888 888    888   888        d88P888 888  .88P  888      888        Y88b.      
Y88b   d88P     d88P 888 888   d88P   888       d88P 888 8888888K.  888      8888888     "Y888b.   
 Y88b d88P     d88P  888 8888888P"    888      d88P  888 888  "Y88b 888      888            "Y88b. 
  Y88o88P     d88P   888 888 T88b     888     d88P   888 888    888 888      888              "888 
   Y888P     d8888888888 888  T88b    888    d8888888888 888   d88P 888      888        Y88b  d88P 
    Y8P     d88P     888 888   T88b 8888888 d88P     888 8888888P"  88888888 8888888888  "Y8888P"
'''

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

'''
888       .d88888b.         d8888 8888888b.  8888888888 8888888b.   .d8888b.  
888      d88P" "Y88b       d88888 888  "Y88b 888        888   Y88b d88P  Y88b 
888      888     888      d88P888 888    888 888        888    888 Y88b.      
888      888     888     d88P 888 888    888 8888888    888   d88P  "Y888b.   
888      888     888    d88P  888 888    888 888        8888888P"      "Y88b. 
888      888     888   d88P   888 888    888 888        888 T88b         "888 
888      Y88b. .d88P  d8888888888 888  .d88P 888        888  T88b  Y88b  d88P 
88888888  "Y88888P"  d88P     888 8888888P"  8888888888 888   T88b  "Y8888P"  
'''

def LoadJsonLandmarks(img_path, ldmk_path, gold=False):
    """
    Load landmarks from json file
    
    Parameters
    ----------
    img_path : str
        Path to the image
    ldmk_path : str
        Path to the json file
    gold : bool, optional
        If True, load gold standard landmarks, by default False
    
    Returns
    -------
    dict
        Dictionary of landmarks
    
    Raises
    ------
    ValueError
        If the json file is not valid
    """
    
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
    """
    Load image from path
    
    Parameters
    ----------
    image_path : str
        Path to the image
    
    Returns
    -------
    tuple
        Spacing and origin of the image
    """
    img = sitk.ReadImage(image_path)
    spacing = np.array(img.GetSpacing())
    origin = img.GetOrigin()
    origin = np.array([origin[2],origin[1],origin[0]])

    return spacing, origin

'''
888b     d888 8888888888 88888888888 8888888b.  8888888  .d8888b.   .d8888b.  
8888b   d8888 888            888     888   Y88b   888   d88P  Y88b d88P  Y88b 
88888b.d88888 888            888     888    888   888   888    888 Y88b.      
888Y88888P888 8888888        888     888   d88P   888   888         "Y888b.   
888 Y888P 888 888            888     8888888P"    888   888            "Y88b. 
888  Y8P  888 888            888     888 T88b     888   888    888       "888 
888   "   888 888            888     888  T88b    888   Y88b  d88P Y88b  d88P 
888       888 8888888888     888     888   T88b 8888888  "Y8888P"   "Y8888P" 
'''

def ComputeErrorInPercent(source, target, transform):
    """
    Computes the error of the transform.
    
    Parameters
    ----------
    source : dict
        Dictionary of landmarks in the source image
    target : dict
        Dictionary of landmarks in the target image
    transform : SimpleITK transform
        Transform to be evaluated

    Returns
    -------
    float
        Error in percent
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

def ComputeMeanDistance(source, target):
    """
    Computes the mean distance between two point sets.
    
    Parameters
    ----------
    source : dict
        Source landmarks
    target : dict
        Target landmarks
    
    Returns
    -------
    float
        Mean distance
    """
    distance = 0
    for key in source.keys():
        distance += np.linalg.norm(source[key] - target[key])
    distance /= len(source.keys())
    return distance

'''
888     888 88888888888 8888888 888       .d8888b.  
888     888     888       888   888      d88P  Y88b 
888     888     888       888   888      Y88b.      
888     888     888       888   888       "Y888b.   
888     888     888       888   888          "Y88b. 
888     888     888       888   888            "888 
Y88b. .d88P     888       888   888      Y88b  d88P 
 "Y88888P"      888     8888888 88888888  "Y8888P"                                                   
'''

def SortDict(input_dict):
    """
    Sorts a dictionary by key
    
    Parameters
    ----------
    input_dict : dict
        Dictionary to be sorted
    
    Returns
    -------
    dict
        Sorted dictionary
    """
    return {k: input_dict[k] for k in sorted(input_dict)}

def PrintMatrix(transform):
    """
    Prints a matrix
    
    Parameters
    ----------
    transform : vtk.vtkMatrix4x4
        Matrix to be printed
    """
    for i in range(4):
        print(transform.GetElement(i,0), transform.GetElement(i,1), transform.GetElement(i,2), transform.GetElement(i,3))
    print()
   


'''
888     888 88888888888 888    d8P       .d8888b.  88888888888 888     888 8888888888 8888888888 
888     888     888     888   d8P       d88P  Y88b     888     888     888 888        888        
888     888     888     888  d8P        Y88b.          888     888     888 888        888        
Y88b   d88P     888     888d88K          "Y888b.       888     888     888 8888888    8888888    
 Y88b d88P      888     8888888b            "Y88b.     888     888     888 888        888        
  Y88o88P       888     888  Y88b             "888     888     888     888 888        888        
   Y888P        888     888   Y88b      Y88b  d88P     888     Y88b. .d88P 888        888        
    Y8P         888     888    Y88b      "Y8888P"      888      "Y88888P"  888        888  
'''                                                                                   

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


def VTKMatrixToNumpy(matrix):
    """
    Copies the elements of a vtkMatrix4x4 into a numpy array.
    
    Parameters
    ----------
    matrix : vtkMatrix4x4
        Matrix to be copied
    
    Returns
    -------
    numpy array
        Numpy array with the elements of the vtkMatrix4x4
    """
    m = np.ones((4, 4))
    for i in range(4):
        for j in range(4):
            m[i, j] = matrix.GetElement(i, j)
    return m

def RenderWindow(Actors,backgroundColor='dark blue'):
    """
    Create a render window with actors
    
    Parameters
    ----------
    Actors : list
        List of actors to be rendered
    backgroundColor : str
        Background color of the render window
        
    Returns
    -------
    None
    """

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

def VTKRender(source, target=None, transform=None, convert_to_vtk=False):
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
    VTK render window
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


'''
8888888  .d8888b.  8888888b.       .d8888b.  88888888888 888     888 8888888888 8888888888 
  888   d88P  Y88b 888   Y88b     d88P  Y88b     888     888     888 888        888        
  888   888    888 888    888     Y88b.          888     888     888 888        888        
  888   888        888   d88P      "Y888b.       888     888     888 8888888    8888888    
  888   888        8888888P"          "Y88b.     888     888     888 888        888        
  888   888    888 888                  "888     888     888     888 888        888        
  888   Y88b  d88P 888            Y88b  d88P     888     Y88b. .d88P 888        888        
8888888  "Y8888P"  888             "Y8888P"      888      "Y88888P"  888        888 
'''

'''
 .d8888b.  8888888 88888888888 888    d8P       .d8888b.  88888888888 888     888 8888888888 8888888888 
d88P  Y88b   888       888     888   d8P       d88P  Y88b     888     888     888 888        888        
Y88b.        888       888     888  d8P        Y88b.          888     888     888 888        888        
 "Y888b.     888       888     888d88K          "Y888b.       888     888     888 8888888    8888888    
    "Y88b.   888       888     8888888b            "Y88b.     888     888     888 888        888        
      "888   888       888     888  Y88b             "888     888     888     888 888        888        
Y88b  d88P   888       888     888   Y88b      Y88b  d88P     888     Y88b. .d88P 888        888        
 "Y8888P"  8888888     888     888    Y88b      "Y8888P"      888      "Y88888P"  888        888 
'''

def ResampleImage(image_path,transform):
    '''
    Resample image using SimpleITK
    
    Parameters
    ----------
    image_path : String
        Path of the image to be resampled.
    transform : SimpleITK transform
        Transform to be applied to the image.
        
    Returns
    -------
    SimpleITK image
        Resampled image.
    '''
    image = sitk.ReadImage(image_path)
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(image)
    resample.SetTransform(transform)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputSpacing(image.GetSpacing())
    resample.SetDefaultPixelValue(0)
    return resample.Execute(image)

def ConvertTransformMatrixToSimpleITK(transformMatrix):
    '''
    Convert transform matrix to SimpleITK transform
    
    Parameters
    ----------
    transformMatrix : vtkMatrix4x4
        Transform matrix to be converted.
    
    Returns
    -------
    SimpleITK transform
        SimpleITK transform.
    '''
    transform = sitk.VersorRigid3DTransform()
    transform.SetMatrix(transformMatrix[:3,:3].flatten())
    # transform.SetTranslation(transformMatrix[:3,3]*0.1)
    return transform

'''
88888888888 8888888b.         d8888 888b    888  .d8888b.  8888888888 .d88888b.  8888888b.  888b     d888  .d8888b.  
    888     888   Y88b       d88888 8888b   888 d88P  Y88b 888       d88P" "Y88b 888   Y88b 8888b   d8888 d88P  Y88b 
    888     888    888      d88P888 88888b  888 Y88b.      888       888     888 888    888 88888b.d88888 Y88b.      
    888     888   d88P     d88P 888 888Y88b 888  "Y888b.   8888888   888     888 888   d88P 888Y88888P888  "Y888b.   
    888     8888888P"     d88P  888 888 Y88b888     "Y88b. 888       888     888 8888888P"  888 Y888P 888     "Y88b. 
    888     888 T88b     d88P   888 888  Y88888       "888 888       888     888 888 T88b   888  Y8P  888       "888 
    888     888  T88b   d8888888888 888   Y8888 Y88b  d88P 888       Y88b. .d88P 888  T88b  888   "   888 Y88b  d88P 
    888     888   T88b d88P     888 888    Y888  "Y8888P"  888        "Y88888P"  888   T88b 888       888  "Y8888P" 
'''

def ApplyTranslation(source,transform):
    '''
    Apply translation to source dictionary of landmarks

    Parameters
    ----------
    source : Dictionary
        Dictionary containing the source landmarks.
    transform : numpy array
        Translation to be applied to the source.
    
    Returns
    -------
    Dictionary
        Dictionary containing the translated source landmarks.
    '''
    sourcee = source.copy()
    for key in sourcee.keys():
        sourcee[key] = sourcee[key] + transform
    return sourcee

def ApplyTransform(source,transform):
    '''
    Apply a transform matrix to a set of landmarks
    
    Parameters
    ----------
    source : dict
        Dictionary of landmarks
    transform : np.array
        Transform matrix
    
    Returns
    -------
    source : dict
        Dictionary of transformed landmarks
    '''
    Translation = transform[:3,3]
    Rotation = transform[:3,:3]
    for key in source.keys():
        source[key] = Rotation @ source[key] + Translation
    return source

def RotationMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.

    Parameters
    ----------
    axis : np.array
        Axis of rotation
    theta : float
        Angle of rotation in radians
    
    Returns
    -------
    np.array
        Rotation matrix
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
    '''
    Apply a rotation matrix to a set of landmarks
    
    Parameters
    ----------
    source : dict
        Dictionary of landmarks
    R : np.array
        Rotation matrix
    
    Returns
    -------
    source : dict
        Dictionary of transformed landmarks
    '''
    sourcee = source.copy()
    for key in sourcee.keys():
        sourcee[key] = np.dot(R,sourcee[key])
    return sourcee

def AngleAndAxisVectors(v1, v2):
    '''
    Return the angle and the axis of rotation between two vectors
    
    Parameters
    ----------
    v1 : numpy array
        First vector
    v2 : numpy array
        Second vector
    
    Returns
    -------
    angle : float
        Angle between the two vectors
    axis : numpy array
        Axis of rotation between the two vectors
    '''
    # Compute angle between two vectors
    v1_u = v1 / np.amax(v1)
    v2_u = v2 / np.amax(v2)
    angle = np.arccos(np.dot(v1_u, v2_u) / (np.linalg.norm(v1_u) * np.linalg.norm(v2_u)))
    axis = cross(v1_u, v2_u)
    #axis = axis / np.linalg.norm(axis)
    return angle,axis