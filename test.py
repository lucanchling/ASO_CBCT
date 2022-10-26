# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/21_Transforms_and_Resampling.html

import SimpleITK as sitk
import numpy as np
import sys
from utils import ConvertTransformMatrixToSimpleITK

# Read image
img = sitk.ReadImage('/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/Anonymized/IC_0005.nii.gz')
fixed = sitk.ReadImage('/home/luciacev/Desktop/Luc_Anchling/Projects/ASO_CBCT/data/Gold_Standard/GOLD_MAMP_0002_Or_T1.nii.gz')
 
dim = img.GetDimension()
spacing = np.array(img.GetSpacing())
origin = np.array(img.GetOrigin())
direction = img.GetDirection()
direction = np.array(direction).reshape(dim, dim)
size = np.array(img.GetSize())

def PhysicalPointToIndex(physical_point, origin, spacing, direction):
    '''
    Convert a physical point to an index with the direction taken into account.
    '''
    return np.round((np.dot(np.linalg.inv(direction), physical_point - origin) / spacing)).astype(int)

TransformMatrix = np.load('cache/TransformMatrixFinal.npy', allow_pickle=True)
transform = ConvertTransformMatrixToSimpleITK(TransformMatrix)

resample = sitk.ResampleImageFilter()
resample.SetReferenceImage(fixed)
resample.SetInterpolator(sitk.sitkLinear)
transform = sitk.TranslationTransform(dim)
transform.SetOffset([-266.0, -280.0, 13.0])
resample.SetTransform(transform)
out = resample.Execute(img)

print("Writing output...")
sitk.WriteImage(out, '/home/luciacev/Desktop/Luc_Anchling/Projects/ASO_CBCT/data/output/outputtest.nii.gz')
print("Done")