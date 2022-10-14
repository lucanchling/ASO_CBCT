# http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/21_Transforms_and_Resampling.html

import SimpleITK as sitk
import numpy as np
import itk
# Read image
img = sitk.ReadImage('/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/Anonymized/IC_0005.nii.gz')
npimg = sitk.GetArrayFromImage(img)

dim = img.GetDimension()
spacing = img.GetSpacing()
origin = img.GetOrigin()
direction = img.GetDirection()
size = img.GetSize()

def transform_point(transform, point):
    print('Point: {} has been transformed to: {}'.format(point, transform.TransformPoint(point)))

# trans = [-15.0, 50.0, -25.0]

# npimg = npimg + trans

translation = sitk.TranslationTransform(dim)
offset = [2] * dim
translation.SetOffset(offset)
translation.SetParameters((0, 150, 0))

# transform_point(translation, [1,0,2])


def resample(image, transform):
    refImage = image
    interpolator = sitk.sitkLinear
    defaultPixelValue = 0.0
    return sitk.Resample(image, refImage, transform, interpolator, defaultPixelValue)

def itk_resample(image, transform):
    refImage = image
    interpolator = itk.LinearInterpolateImageFunction.New(image)
    defaultPixelValue = 0.0
    return itk.Resample(image, refImage, transform, interpolator, defaultPixelValue)

resampled = itk_resample(img, translation)
# resampled = sitk.GetImageFromArray(npimg)
sitk.WriteImage(resampled, 'data/output/resampled.nii.gz')
print('Done')