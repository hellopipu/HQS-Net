# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

from torch.fft import fft2, ifft2
import torch
import math
import numpy as np
import time
from numpy.fft import ifftshift
from numpy.lib.stride_tricks import as_strided

def RF(x_rec, mask, norm='ortho'):
    '''
    RF means R*F(input), F is fft, R is applying mask;
    return the masked k-space of x_rec,
    '''
    x_rec = x_rec.permute(0, 2, 3, 1)
    mask = mask.permute(0, 2, 3, 1)
    k_rec = torch.fft.fft2(torch.view_as_complex(x_rec.contiguous()), norm=norm)
    k_rec = torch.view_as_real(k_rec)
    k_rec *= mask
    k_rec = k_rec.permute(0, 3, 1, 2)
    return k_rec

def undersample(image, mask, norm='ortho'):
    assert image.shape == mask.shape

    k = fft2(image, norm=norm)
    k_und = mask * k
    # import matplotlib.pyplot as plt
    # print(k_und.shape)
    # plt.imshow(np.log10(np.abs(k_und)+1e-30))
    # plt.savefig('debug.png')
    x_und = ifft2(k_und, norm=norm)

    return x_und, k_und, k

def revert_scale(im_tensor, a=2., b=-1.):
    '''
    param: im_tensor : [B, 2, W, H]
    '''
    b = b * torch.ones_like(im_tensor)
    im = (im_tensor - b) / a

    return im
def gray2rgb(x):
    x = x.unsqueeze(1)
    return torch.cat([x,x,x],1)

def output2complex(im_tensor, revert = False):
    '''
    param: im_tensor : [B, 2, W, H]
    return : [B,W,H] magnitude of complex value
    '''
    ############## revert each channel to [0,1.] range
    if revert:
        im_tensor = revert_scale(im_tensor)
    # 2 channel to complex
    im_tensor = torch.view_as_complex(im_tensor.permute(0, 2, 3, 1).contiguous()).abs()

    return im_tensor

def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2) ** 2)

def cartesian_mask(shape: object, acc: object, centred: object = False,
                   sample_random=True) -> object:
    """
    Sampling density estimated from implementation of kt FOCUSS

    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    sample_n: num of lines in low frequency to be sampled

    """
    shape = shape[:-2] + (shape[-1],shape[-2])
    if acc==5:
        center_fraction = 0.08
    elif acc==10:
        center_fraction = 0.04

    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    sample_n = int(round(Nx * center_fraction))
    pdf_x = normal_pdf(Nx, 0.5 / (Nx / 10.) ** 2)
    lmda = Nx / (2. * acc)
    n_lines = int(Nx / acc)

    # add uniform distribution
    pdf_x += lmda * 1. / Nx

    if sample_n:
        pdf_x[Nx // 2 - sample_n // 2:Nx // 2 + sample_n - sample_n // 2] = 0  # sample_n -
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n

    mask = np.zeros((N, Nx))
    ##############################################
    # set fixed seed
    if not sample_random:
        np.random.seed(233)
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1
    ## cancel seed when finish
    if not sample_random:
        t = 1000 * time.time()  # current time in milliseconds
        np.random.seed(int(t) % 2 ** 32)
    #####################################3

    if sample_n:
        mask[:, Nx // 2 - sample_n // 2:Nx // 2 + sample_n - sample_n // 2] = 1

    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))

    mask = mask.reshape(shape)

    if not centred:
        mask = ifftshift(mask, axes=(-1, -2))

    return mask.transpose((-1,-2))

'''
borrowed from  https://github.com/MRIOSU/OCMR/blob/master/Python/read_ocmr.py
'''

import os
import ismrmrd
import ismrmrd.xsd
import numpy as np


def read_ocmr(filename):
    # Before running the code, install ismrmrd-python and ismrmrd-python-tools:
    #  https://github.com/ismrmrd/ismrmrd-python
    #  https://github.com/ismrmrd/ismrmrd-python-tools
    # Last modified: 06-12-2020 by Chong Chen (Chong.Chen@osumc.edu)
    #
    # Input:  *.h5 file name
    # Output: all_data    k-space data, orgnazide as {'kx'  'ky'  'kz'  'coil'  'phase'  'set'  'slice'  'rep'  'avg'}
    #         param  some parameters of the scan
    #

    # This is a function to read K-space from ISMRMD *.h5 data
    # Modifid by Chong Chen (Chong.Chen@osumc.edu) based on the python script
    # from https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/recon_ismrmrd_dataset.py

    if not os.path.isfile(filename):
        print("%s is not a valid file" % filename)
        raise SystemExit
    dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = header.encoding[0]

    # Matrix size
    eNx = enc.encodedSpace.matrixSize.x
    # eNy = enc.encodedSpace.matrixSize.y
    eNz = enc.encodedSpace.matrixSize.z
    eNy = (enc.encodingLimits.kspace_encoding_step_1.maximum + 1);  # no zero padding along Ny direction

    # Field of View
    eFOVx = enc.encodedSpace.fieldOfView_mm.x
    eFOVy = enc.encodedSpace.fieldOfView_mm.y
    eFOVz = enc.encodedSpace.fieldOfView_mm.z

    # Save the parameters
    param = dict();
    param['TRes'] = str(header.sequenceParameters.TR)
    param['FOV'] = [eFOVx, eFOVy, eFOVz]
    param['TE'] = str(header.sequenceParameters.TE)
    param['TI'] = str(header.sequenceParameters.TI)
    param['echo_spacing'] = str(header.sequenceParameters.echo_spacing)
    param['flipAngle_deg'] = str(header.sequenceParameters.flipAngle_deg)
    param['sequence_type'] = header.sequenceParameters.sequence_type

    # Read number of Slices, Reps, Contrasts, etc.
    nCoils = header.acquisitionSystemInformation.receiverChannels
    try:
        nSlices = enc.encodingLimits.slice.maximum + 1
    except:
        nSlices = 1

    try:
        nReps = enc.encodingLimits.repetition.maximum + 1
    except:
        nReps = 1

    try:
        nPhases = enc.encodingLimits.phase.maximum + 1
    except:
        nPhases = 1;

    try:
        nSets = enc.encodingLimits.set.maximum + 1;
    except:
        nSets = 1;

    try:
        nAverage = enc.encodingLimits.average.maximum + 1;
    except:
        nAverage = 1;

        # TODO loop through the acquisitions looking for noise scans
    firstacq = 0
    for acqnum in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)

        # TODO: Currently ignoring noise scans
        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            # print("Found noise scan at acq ", acqnum)
            continue
        else:
            firstacq = acqnum
            print("Imaging acquisition starts acq ", acqnum)
            break

    # assymetry echo
    kx_prezp = 0;
    acq_first = dset.read_acquisition(firstacq)
    if acq_first.center_sample * 2 < eNx:
        kx_prezp = eNx - acq_first.number_of_samples

    # Initialiaze a storage array
    param['kspace_dim'] = {'kx ky kz coil phase set slice rep avg'};
    all_data = np.zeros((eNx, eNy, eNz, nCoils, nPhases, nSets, nSlices, nReps, nAverage), dtype=np.complex64)

    # Loop through the rest of the acquisitions and stuff
    for acqnum in range(firstacq, dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)

        # Stuff into the buffer
        y = acq.idx.kspace_encode_step_1
        z = acq.idx.kspace_encode_step_2
        phase = acq.idx.phase;
        set = acq.idx.set;
        slice = acq.idx.slice;
        rep = acq.idx.repetition;
        avg = acq.idx.average;
        all_data[kx_prezp:, y, z, :, phase, set, slice, rep, avg] = np.transpose(acq.data)

    return all_data, param

if __name__ == '__main__':
    a = cartesian_mask((192,160), acc=5, centred = False, sample_random=False)
    import matplotlib.pyplot as plt
    plt.imshow(a)
    plt.show()
    print(a.shape,a.dtype)