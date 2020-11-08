import SimpleITK as sitk
import glob

file_names = glob.glob('train/*.png')
reader = sitk.ImageSeriesReader()
reader.SetFileNames(file_names)
vol = reader.Execute()
sitk.WriteImage(vol, 'volume.nii.gz')
