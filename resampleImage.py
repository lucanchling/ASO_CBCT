import numpy as np
import SimpleITK as sitk


def resample_image(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False , transform = sitk.Transform()):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    out_spacing = itk_image.GetSpacing()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    transfParam = transform.GetParameters()

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    modifiedOrigin = np.array(itk_image.GetOrigin()) + np.array(transfParam)
    resample.SetOutputOrigin(modifiedOrigin)
    resample.SetTransform(transform)
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image) 

if __name__ == '__main__':
    img = sitk.ReadImage('/home/luciacev/Desktop/Luc_Anchling/DATA/ASO_CBCT/Anonymized/IC_0005.nii.gz')

    translation = sitk.TranslationTransform(3)
    # translation.SetOffset((0, 150, 0))
    translation.SetParameters((0, 0, 0))

    resampled = resample_image(img, [1.0, 1.0, 1.0], False, translation)
    sitk.WriteImage(resampled, 'data/output/resampledimi.nii.gz')
    print('Done')