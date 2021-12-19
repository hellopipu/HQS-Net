# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers
import pandas
import math
import os
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn,fft2,ifft2
# import ReadWrapper
from tqdm import tqdm
### https://arxiv.org/pdf/1811.08026.pdf
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import glob
import h5py as h5
from utils import read_ocmr
import numpy as np
from numpy.fft import fftshift, ifftshift, fft2, ifft2
import math
import operator
from skimage.metrics import structural_similarity as cal_ssim
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from skimage.metrics import normalized_root_mse as cal_nrmse
def is_num(a):
    return isinstance(a, int) or isinstance(a, float)


def delta(x1, x2):
    delta_ = x2 - x1
    return delta_ // 2, delta_ - delta_ // 2


def get_padding_width(o_shape, d_shape):
    if is_num(o_shape):
        o_shape, d_shape = [o_shape], [d_shape]
    assert len(o_shape) == len(d_shape), 'Length mismatched!'
    borders = []
    for o, d in zip(o_shape, d_shape):
        borders.extend(delta(o, d))
    return borders


def get_crop_width(o_shape, d_shape):
    return get_padding_width(d_shape, o_shape)


def get_padding_shape_with_stride(o_shape, stride):
    assert isinstance(o_shape, list) or isinstance(o_shape, tuple) or isinstance(o_shape, np.ndarray)
    o_shape = np.array(o_shape)
    d_shape = np.ceil(o_shape / stride) * stride
    return d_shape.astype(np.int32)


def pad(arr, d_shape, mode='constant', value=0, strict=True):
    """
    pad numpy array, tested!
    :param arr: numpy array
    :param d_shape: array shape after padding or minimum shape
    :param mode: padding mode,
    :param value: padding value
    :param strict: if True, d_shape must be greater than arr shape and output shape is d_shape. if False, d_shape is minimum shape and output shape is np.maximum(arr.shape, d_shape)
    :return: padded arr with expected shape
    """
    assert arr.ndim == len(d_shape), 'Dimension mismatched!'
    if not strict:
        d_shape = np.maximum(arr.shape, d_shape)
    else:
        assert np.all(np.array(d_shape) >= np.array(arr.shape)), 'Padding shape must be greater than arr shape'
    borders = np.array(get_padding_width(arr.shape, d_shape))
    before = borders[list(range(0, len(borders), 2))]
    after = borders[list(range(1, len(borders), 2))]
    padding_borders = tuple(zip([int(x) for x in before], [int(x) for x in after]))
    # print(padding_borders)
    if mode == 'constant':
        return np.pad(arr, padding_borders, mode=mode, constant_values=value)
    else:
        return np.pad(arr, padding_borders, mode=mode)


def crop(arr, d_shape, strict=True):
    """
    central  crop numpy array, tested!
    :param arr: numpy array
    :param d_shape: expected shape
    :return: cropped array with expected array
    """
    assert arr.ndim == len(d_shape), 'Dimension mismatched!'
    if not strict:
        d_shape = np.minimum(arr.shape, d_shape)
    else:
        assert np.all(np.array(d_shape) <= np.array(arr.shape)), 'Crop shape must be smaller than arr shape'
    borders = np.array(get_crop_width(arr.shape, d_shape))
    start = borders[list(range(0, len(borders), 2))]
    # end = - borders[list(range(1, len(borders), 2))]
    end = map(operator.add, start, d_shape)
    slices = tuple(map(slice, start, end))
    return arr[slices]


def pad_crop(arr, d_shape, mode='constant', value=0):
    """
    pad or crop numpy array to expected shape, tested!
    :param arr: numpy array
    :param d_shape: expected shape
    :param mode: padding mode,
    :param value: padding value
    :return: padded and cropped array
    """
    assert arr.ndim == len(d_shape), 'Dimension mismatched!'
    arr = pad(arr, d_shape, mode, value, strict=False)
    return crop(arr, d_shape)
def get_mag_for_2ch_complex(img):
    '''

    :param img: 2 X W X H, torch tensor
    :return:
    '''
    return (img[0] ** 2 + img[1] ** 2) ** 0.5
def center_pad_kspace(k,size):
    '''

    :param img: W X H X ...
    :param size: constant number
    :return: padded img
    '''
    w, h = k.shape[:2]
    pad_w = (size[0] - w)//2
    pad_h = (size[1] - h) // 2
    k_padded = np.pad(k, ((pad_w,size[0]-pad_w-w),(pad_h,size[1]-pad_h-h),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)))
    return k_padded

def center_pad_img(img,size):
    '''

    :param img: W X H X FRAMES
    :param size: constant number
    :return: padded img
    '''
    w, h, p = img.shape
    pad_w = (size-w)//2
    pad_h = (size - h) // 2
    img_padded = np.pad(img, ((pad_w,size-pad_w-w),(pad_h,size-pad_h-h),(0,0)))
    return img_padded
def get_data(path_dir,csv_file, scn):
    '''

    :param path_dir:
    :param csv_file:
    :param istrain:
    :return:  SLICES X W X H
    '''

    K_SIZE = (394,160)
    array_list = []
    #### 1. read csv files
    df = pandas.read_csv(csv_file)
    # Cleanup empty rows and columns
    df.dropna(how='all', axis=0, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    ## filter files for training
    selected_df = df.query('`file name`.str.contains("fs_") and fov=="noa" and  scn=='+ scn, engine='python') #scn=="15avan" and viw=="sax" and sli=="stk"

    list_name = selected_df['file name']
    print('len: ',len(list_name))
    print(list_name)
    for filename in tqdm(list_name):
        data_path = os.path.join(path_dir, filename)
        print(data_path)
        kData, param = read_ocmr(data_path)
        print('data shape: ',kData.shape)
        print(param)
        # print(filename, 'Dimension of kData: ', kData.shape)


        # Image reconstruction (SoS)
        dim_kData = kData.shape;
        CH = dim_kData[3];
        PHA = dim_kData[4];
        SLC = dim_kData[6];
        kData_tmp = np.mean(kData, axis=8);  # average the k-space if average > 1

        ## ifft
        im_coil = fftshift(ifft2(ifftshift(kData_tmp,(0,1)),axes=(0,1),norm='ortho'),(0,1))
        im_sos = np.sqrt(np.sum(np.abs(im_coil) ** 2, 3));  # Sum of Square

        RO = im_sos.shape[0];
        image = im_sos[math.floor(RO / 4):math.floor(RO / 4 * 3), :, :];  # Remove RO oversampling
        im_coil_ = im_coil[math.floor(RO / 4):math.floor(RO / 4 * 3), :, :]
        # print('Dimension of Image (without ReadOout ovesampling): ', image.shape)
        image = image.reshape((image.shape[0], image.shape[1],-1))
        im_coil_ = im_coil_.reshape((image.shape[0], image.shape[1],CH, -1))
        ##pad or crop to fixed size
        image = pad_crop(image,(192,160,image.shape[2]))
        im_coil_ = pad_crop(im_coil_, (192, 160, im_coil_.shape[2], im_coil_.shape[3]))

        def myff(x):
            kk = np.matmul(im_coil_.transpose(0, 1, 3,2), x)
            error = np.sum((np.abs(kk)**0.5-np.abs(image)**0.5)**2)**0.5
            # error = np.mean(np.abs((kk - image)) ** 2)
            print('error: ',error)
            return error

        from scipy.optimize import minimize, rosen, rosen_der
        x0 = np.ones((CH, 1)) / CH
        res = minimize(myff, x0, method='BFGS') #1e-8  #, options={'gtol': 1e-8, 'disp': True}

        esc = np.matmul(im_coil_.transpose(0, 1, 3,2), res.x)

        import matplotlib.pyplot as plt
        plt.subplot(121)
        plt.imshow(image[:, :, image.shape[2]//2])
        plt.subplot(122)
        plt.imshow(np.abs(esc[:, :, esc.shape[2]//2]))
        plt.savefig('debug.png')

        print(cal_psnr(image[:, :, image.shape[2]//2], np.abs(esc[:, :, esc.shape[2]//2])))
        print(cal_ssim(image[:, :, image.shape[2]//2], np.abs(esc[:, :, esc.shape[2]//2])))

        ### save
        esc_comp = np.stack([np.real(esc), np.imag(esc)], axis=-1).astype(np.float32)
        print('final shape: ', esc_comp.shape)
        array_list.append(esc_comp)

    return array_list



if __name__ == '__main__':

    ### simulating the single-coil MRI from multi-coil data using the method in https://arxiv.org/pdf/1811.08026.pdf

    ocmr_data_attributes_location = '/research/cbim/vast/bx64/project/kspace/ocmr_data_attributes.csv'
    ocmr_data_location = '/research/cbim/vast/bx64/project/public_dataset/OCMR_data'
    
    im_avan = get_data(ocmr_data_location, ocmr_data_attributes_location,'"15avan"')
    im_30pris = get_data(ocmr_data_location, ocmr_data_attributes_location,'"30pris"')
    im_15sola = get_data(ocmr_data_location, ocmr_data_attributes_location,'"15sola"')
    
    ## split dataset
    trainset = im_avan[0:6] + im_30pris[0:22] + im_15sola[0:13]
    valset = im_avan[6:8] + im_30pris[22:29] + im_15sola[13:17]
    testset = im_avan[8::] + im_30pris[29::] + im_15sola[17::]
    
    trainset = np.concatenate(trainset,axis=2).transpose(2,3,0,1)
    valset = np.concatenate(valset,axis=2).transpose(2,3,0,1)
    testset = np.concatenate(testset,axis=2).transpose(2,3,0,1)

    ## save to numpy
    np.save("/research/cbim/vast/bx64/project/public_dataset/my_ocmr/fs_train.npy",trainset)
    np.save("/research/cbim/vast/bx64/project/public_dataset/my_ocmr/fs_val.npy",valset)
    np.save("/research/cbim/vast/bx64/project/public_dataset/my_ocmr/fs_test.npy",testset)
    
    print('slices of trainset: ', len(trainset))
    print('slices of valset: ', len(valset))
    print('slices of testset: ', len(testset))


