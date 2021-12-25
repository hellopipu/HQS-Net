# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

'''
File function:
1. Generate emulated single-coil data from OCMR dataset using the method in
https://arxiv.org/pdf/1811.08026.pdf. This procedure takes several hours.
2. Split train, val, and test sets to 1874, 544, and 1104, respectively
'''

import os
import pandas
from tqdm import tqdm
from utils import read_ocmr
import numpy as np
from numpy.fft import fftshift, ifftshift, ifft2
import math
from scipy.optimize import minimize
from utils import pad_crop


def get_data(path_dir, csv_file, scn):
    '''

    :param path_dir: data folder path
    :param csv_file:
    :param scn:
    :return:
    '''
    print('------- Emulating single-coil for scn = {} -------'.format(scn))
    array_list = []
    ## read csv files
    df = pandas.read_csv(csv_file)
    ## Cleanup empty rows and columns
    df.dropna(how='all', axis=0, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    ## filter files
    selected_df = df.query('`file name`.str.contains("fs_") and fov=="noa" and  scn==' + scn, engine='python')
    list_name = selected_df['file name']

    for filename in tqdm(list_name):
        data_path = os.path.join(path_dir, filename)
        kData, param = read_ocmr(data_path)
        dim_kData = kData.shape
        CH = dim_kData[3]

        ## Average the k-sapce along phase(time) dimension
        kData_tmp = np.mean(kData, axis=8);  # average the k-space if average > 1

        ## Coil images are combined using SOS (sum of square.)
        im_coil = fftshift(ifft2(ifftshift(kData_tmp, (0, 1)), axes=(0, 1), norm='ortho'), (0, 1))  # IFFT (2D image)
        im_sos = np.sqrt(np.sum(np.abs(im_coil) ** 2, 3))  # Sum of Square

        ## Remove ReadOut oversampling
        RO = im_sos.shape[0]
        image = im_sos[math.floor(RO / 4):math.floor(RO / 4 * 3), :, :]  # Remove RO oversampling
        im_coil_ = im_coil[math.floor(RO / 4):math.floor(RO / 4 * 3), :, :]
        image = image.reshape((image.shape[0], image.shape[1], -1))
        im_coil_ = im_coil_.reshape((image.shape[0], image.shape[1], CH, -1))

        ## pad or crop to fixed size
        image = pad_crop(image, (192, 160, image.shape[2]))
        im_coil_ = pad_crop(im_coil_, (192, 160, im_coil_.shape[2], im_coil_.shape[3]))

        ## emulate single-coil img from multi-coil using LBFGS
        def error_func(x):
            kk = np.matmul(im_coil_.transpose(0, 1, 3, 2), x)
            error = np.sum((np.abs(kk) ** 0.5 - np.abs(image) ** 0.5) ** 2) ** 0.5
            print('emulating error: ', error, end="\r")
            return error

        x0 = np.ones((CH, 1)) / CH
        res = minimize(error_func, x0, method='BFGS')
        esc = np.matmul(im_coil_.transpose(0, 1, 3, 2), res.x)

        ### save
        esc_comp = np.stack([np.real(esc), np.imag(esc)], axis=-1).astype(np.float32)
        array_list.append(esc_comp)

    return array_list


if __name__ == '__main__':
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    ### 1. simulating the single-coil MRI from multi-coil data using the method in https://arxiv.org/pdf/1811.08026.pdf
    ocmr_data_attributes_location = 'data/ocmr_data_attributes.csv'
    ocmr_data_location = 'data/OCMR_data'

    im_avan = get_data(ocmr_data_location, ocmr_data_attributes_location, '"15avan"')
    im_30pris = get_data(ocmr_data_location, ocmr_data_attributes_location, '"30pris"')
    im_15sola = get_data(ocmr_data_location, ocmr_data_attributes_location, '"15sola"')

    ## 2. split dataset
    trainset = im_avan[0:6] + im_30pris[0:22] + im_15sola[0:13]
    valset = im_avan[6:8] + im_30pris[22:29] + im_15sola[13:17]
    testset = im_avan[8::] + im_30pris[29::] + im_15sola[17::]

    trainset = np.concatenate(trainset, axis=2).transpose(2, 3, 0, 1)
    valset = np.concatenate(valset, axis=2).transpose(2, 3, 0, 1)
    testset = np.concatenate(testset, axis=2).transpose(2, 3, 0, 1)

    ## 3. save to numpy
    np.save("data/fs_train.npy", trainset)
    np.save("data/fs_val.npy", valset)
    np.save("data/fs_test.npy", testset)

    print('slices of trainset: ', len(trainset))
    print('slices of valset: ', len(valset))
    print('slices of testset: ', len(testset))
