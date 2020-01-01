# -*- coding=utf-8 -*-
import matplotlib.pyplot as plt
import pydicom
import pydicom.uid
import sys
import PIL.Image as Image
from PyQt5 import QtGui
# from PyQt5 import QtCore, QtGui, QtWidgets
import os
import cv2
have_numpy = True

try:
    import numpy as np
except ImportError:
    have_numpy = False
    raise

sys_is_little_endian = (sys.byteorder == 'little')

NumpySupportedTransferSyntaxes = [
    pydicom.uid.ExplicitVRLittleEndian,
    pydicom.uid.ImplicitVRLittleEndian,
    pydicom.uid.DeflatedExplicitVRLittleEndian,
    pydicom.uid.ExplicitVRBigEndian,
]

def get_new_pixel_array(pixel_array):
    # print(pixel_array)
    # rows, cols = pixel_array.shape[:2]
    # new_pixel_array = np.zeros((rows, cols), dtype=np.int16)
    # for r in range(rows):
    #     for c in range(cols):
    #         if pixel_array[r,c]> 100 and pixel_array[r,c]<2048:
    #             new_pixel_array[r,c] = pixel_array[r,c]
    #         else:
    #             new_pixel_array[r,c] = -2000
    # return new_pixel_array
    len_of_pixel_array = len(pixel_array)
    print(type(pixel_array))
    new_pixel_array = np.zeros(pixel_array.shape, pixel_array.dtype)

    for i in range(len_of_pixel_array):
        if pixel_array[i] >= 200 and pixel_array[i] <= 2861:
            new_pixel_array[i] = pixel_array[i]
        else:
            new_pixel_array[i] = -2000
    print(new_pixel_array==pixel_array)
    return new_pixel_array

# 支持的传输语法
def supports_transfer_syntax(dicom_dataset):
    """
    Returns
    -------
    bool
        True if this pixel data handler might support this transfer syntax.
        False to prevent any attempt to try to use this handler
        to decode the given transfer syntax
    """
    return (dicom_dataset.file_meta.TransferSyntaxUID in
            NumpySupportedTransferSyntaxes)


def needs_to_convert_to_RGB(dicom_dataset):
    return False


def should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
    return False


# 加载Dicom图像数据
def get_pixeldata(dicom_dataset):
    """If NumPy is available, return an ndarray of the Pixel Data.
    Raises
    ------
    TypeError
        If there is no Pixel Data or not a supported data type.
    ImportError
        If NumPy isn't found
    NotImplementedError
        if the transfer syntax is not supported
    AttributeError
        if the decoded amount of data does not match the expected amount
    Returns
    -------
    numpy.ndarray
       The contents of the Pixel Data element (7FE0,0010) as an ndarray.
    """
    if (dicom_dataset.file_meta.TransferSyntaxUID not in
            NumpySupportedTransferSyntaxes):
        raise NotImplementedError("Pixel Data is compressed in a "
                                  "format pydicom does not yet handle. "
                                  "Cannot return array. Pydicom might "
                                  "be able to convert the pixel data "
                                  "using GDCM if it is installed.")

    # 设置窗宽窗位
    #dicom_dataset.

    if not have_numpy:
        msg = ("The Numpy package is required to use pixel_array, and "
               "numpy could not be imported.")
        raise ImportError(msg)
    if 'PixelData' not in dicom_dataset:
        raise TypeError("No pixel data found in this dataset.")

    # Make NumPy format code, e.g. "uint16", "int32" etc
    # from two pieces of info:
    # dicom_dataset.PixelRepresentation -- 0 for unsigned, 1 for signed;
    # dicom_dataset.BitsAllocated -- 8, 16, or 32
    if dicom_dataset.BitsAllocated == 1:
        # single bits are used for representation of binary data
        format_str = 'uint8'
    elif dicom_dataset.PixelRepresentation == 0:
        format_str = 'uint{}'.format(dicom_dataset.BitsAllocated)
    elif dicom_dataset.PixelRepresentation == 1:
        format_str = 'int{}'.format(dicom_dataset.BitsAllocated)
    else:
        format_str = 'bad_pixel_representation'
    try:
        numpy_dtype = np.dtype(format_str)
    except TypeError:
        msg = ("Data type not understood by NumPy: "
               "format='{}', PixelRepresentation={}, "
               "BitsAllocated={}".format(
                   format_str,
                   dicom_dataset.PixelRepresentation,
                   dicom_dataset.BitsAllocated))
        raise TypeError(msg)

    if dicom_dataset.is_little_endian != sys_is_little_endian:
        numpy_dtype = numpy_dtype.newbyteorder('S')

    pixel_bytearray = dicom_dataset.PixelData
    # print(pixel_bytearray)

    if dicom_dataset.BitsAllocated == 1:
        # if single bits are used for binary representation, a uint8 array
        # has to be converted to a binary-valued array (that is 8 times bigger)
        try:
            pixel_array = np.unpackbits(
                np.frombuffer(pixel_bytearray, dtype='uint8'))
        except NotImplementedError:
            # PyPy2 does not implement numpy.unpackbits
            raise NotImplementedError(
                'Cannot handle BitsAllocated == 1 on this platform')
    else:
        pixel_array = np.frombuffer(pixel_bytearray, dtype=numpy_dtype)
    # pixel_array = get_new_pixel_array(pixel_array)
    length_of_pixel_array = pixel_array.nbytes
    expected_length = dicom_dataset.Rows * dicom_dataset.Columns
    if ('NumberOfFrames' in dicom_dataset and
            dicom_dataset.NumberOfFrames > 1):
        expected_length *= dicom_dataset.NumberOfFrames
    if ('SamplesPerPixel' in dicom_dataset and
            dicom_dataset.SamplesPerPixel > 1):
        expected_length *= dicom_dataset.SamplesPerPixel
    if dicom_dataset.BitsAllocated > 8:
        expected_length *= (dicom_dataset.BitsAllocated // 8)
    padded_length = expected_length
    if expected_length & 1:
        padded_length += 1
    if length_of_pixel_array != padded_length:
        raise AttributeError(
            "Amount of pixel data %d does not "
            "match the expected data %d" %
            (length_of_pixel_array, padded_length))
    if expected_length != padded_length:
        pixel_array = pixel_array[:expected_length]
    if should_change_PhotometricInterpretation_to_RGB(dicom_dataset):
        dicom_dataset.PhotometricInterpretation = "RGB"
    if dicom_dataset.Modality.lower().find('ct') >= 0:  # CT图像需要得到其CT值图像
        pixel_array = pixel_array * dicom_dataset.RescaleSlope + dicom_dataset.RescaleIntercept  # 获得图像的CT值
    pixel_array = pixel_array.reshape(dicom_dataset.Rows, dicom_dataset.Columns*dicom_dataset.SamplesPerPixel)
    return pixel_array, dicom_dataset.Rows, dicom_dataset.Columns

# 调整CT图像的窗宽窗位
def setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols):
    img_temp = img_data
    img_temp.flags.writeable = True
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    for i in np.arange(rows):
        for j in np.arange(cols):
            img_temp[i, j] = int((img_temp[i, j]-min)*dFactor)

    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255

    return img_temp


# 加载Dicom图片中的Tag信息
def loadFileInformation(filename):
    information = {}
    ds = pydicom.read_file(filename)
    information['PatientID'] = ds.PatientID
    information['PatientName'] = ds.PatientName
    information['PatientBirthDate'] = ds.PatientBirthDate
    information['PatientSex'] = ds.PatientSex
    information['StudyID'] = ds.StudyID
    information['StudyDate'] = ds.StudyDate
    information['StudyTime'] = ds.StudyTime
    information['InstitutionName'] = ds.InstitutionName
    information['Manufacturer'] = ds.Manufacturer
    print(dir(ds))
    print(type(information))
    return information
filename = r"D:\medicalProject\dicomsource\sunxiulan\sunxiulan1\14.dcm"
# filename = r"D:\medicalProject\dicomsource\xuhongli\xuhongli1\6.徐红丽1.dcm"

dcm = pydicom.dcmread(filename)  # 加载Dicom数据63

pixel_bytearray = dcm.pixel_array
pixel_array = np.frombuffer(pixel_bytearray, dtype=np.int16)

img_data, row, col = get_pixeldata(dcm)
new_img_data = np.zeros(img_data.shape, img_data.dtype)
s1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
# s2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
background = np.zeros(img_data.shape, np.uint8)

for r in range(512):
    for c in range(512):
        if img_data[r,c] < -200.0:
            background[r,c] = 255
        else:
            background[r,c] = 0

close_background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, s1, iterations=1)

for r in range(512):
    for c in range(512):
        if img_data[r,c] >= 80 and img_data[r,c]<=2048:
            new_img_data[r,c] = 255
        else:
            new_img_data[r,c] = 0

new_img_data_1 = np.zeros(img_data.shape, img_data.dtype)
for r in range(512):
    for c in range(512):
        if img_data[r,c] >= -200 and img_data[r,c]<=-20:
            new_img_data_1[r,c] = 255
        else:
            new_img_data_1[r,c] = 0
img_data_1 = new_img_data+new_img_data_1
print(img_data)

combine_gu_and_kongqi = cv2.morphologyEx(img_data_1, cv2.MORPH_CLOSE, s1, iterations=1)


# xifenge_base_img = setDicomWinWidthWinCenter(xifenge_base, 80, 40, 512, 512)
img_temp = setDicomWinWidthWinCenter(img_data, 80, 52, 512, 512)
result = np.zeros(img_data.shape, img_data.dtype)
# print(img_temp)
combine_gu_and_kongqi.astype(img_data.dtype)


for r in range(512):
    for c in range(512):
        if combine_gu_and_kongqi[r,c] == 0.0:
            result[r,c] = img_temp[r,c]
        else:
            continue
result.astype(img_data.dtype)

naojiye = np.zeros(img_data.shape, img_data.dtype)
for r in range(512):
    for c in range(512):
        if (img_data[r,c] > 2 and img_data[r,c]<20) and close_background[r,c] == 0:
            naojiye[r,c] =  255
        else:
            naojiye[r,c] = 0


naozuzhi = np.zeros(img_data.shape, img_data.dtype)
for r in range(512):
    for c in range(512):
        if img_data[r,c]>20 and img_data[r,c]<80 and close_background[r,c] == 0:
            naozuzhi[r,c] = 255
        else:
            naozuzhi[r,c] = 0





dcm_img = Image.fromarray(new_img_data)  # 将Numpy转换为PIL.Image
dcm_img_1 = Image.fromarray(new_img_data_1)  # 将Numpy转换为PIL.Image
dcm_img_2 = Image.fromarray(combine_gu_and_kongqi)
dcm_img_3 = Image.fromarray(result)
dcm_img_4 = Image.fromarray(img_temp)
dcm_img_5 = Image.fromarray(naojiye)
dcm_img_6 = Image.fromarray(close_background)
# dcm_img_7= Image.fromarray(naozuzhi)

dcm_img_1 = dcm_img_1.convert('L')
dcm_img_2 = dcm_img_2.convert('L')
dcm_img_3 = dcm_img_3.convert('L')
dcm_img_4 = dcm_img_4.convert('L')
dcm_img_5 = dcm_img_5.convert('L')
dcm_img_6 = dcm_img_6.convert('L')
# dcm_img_7 = dcm_img_7.convert('L')
# print(dcm_img_2)

# 保存为jpg文件，用作后面的生成label用
dcm_img_3.save('temp.jpg')
# 显示图像
dcm_img.show()
dcm_img_1.show()
dcm_img_2.show()
dcm_img_3.show()
dcm_img_4.show()
dcm_img_5.show()
dcm_img_6.show()
# dcm_img_7.show()
