import torch
import numpy as np
from mne.io import read_raw_gdf
from scipy.signal import cheby2, filtfilt
import scipy.io as sio
import scipy.io
from scipy.io import loadmat

def getData(subject_index):
    subject_index = 6  # 1-9

    # T data
    session_type = 'T'  # T and E
    dir_1 = f'./BCICIV_2a_gdf/A0{subject_index}{session_type}.gdf'  # set your path of the downloaded data
    raw = read_raw_gdf(dir_1)
    # s, HDR: (n_channels, n_times_samps)
    s, HDR = raw.get_data()

    # Label
    labeldir_1 = f'D:\MI\true_labels\A0{subject_index}{session_type}.mat'
    mat = scipy.io.loadmat(labeldir_1)
    label_1 = mat['classlabel']

    # construct sample - data Section 1000*22*288
    Pos = raw.annotations.onset * raw.info['sfreq']
    Typ = raw.annotations.description.astype(int)

    k = 0
    data_1 = torch.zeros(1000, 22, 288)
    for j in range(len(Typ)):
        if Typ[j] == 768:
            k += 1
            data_1[:,:,k-1] = torch.from_numpy(s[Pos[j]+500:Pos[j]+1500,:])

    

    # wipe off NaN
    data_1[np.isnan(data_1)] = 0
    

    # E data
    session_type = 'E'
    # dir_2 = f'D:\Lab\MI\BCICIV_2a_gdf\A0{subject_index}{session_type}.gdf'
    dir_2 = f'./BCICIV_2a_gdf/A0{subject_index}{session_type}.gdf'
    # dir = 'D:\Lab\MI\BCICIV_2a_gdf\A01E.gdf';
    raw = read_raw_gdf(dir_2)
    # s, HDR: (n_channels, n_times_samps)
    s, HDR = raw.get_data()
    

    # Label
    # label = HDR.Classlabel;
    labeldir_2 = f'D:\Lab\MI\true_labels\A0{subject_index}{session_type}.mat'
    label_2 = loadmat(labeldir_2)['classlabel'].ravel()

    # construct sample - data Section 1000*22*288
    Pos = HDR.annotations.onset * HDR.info['sfreq']
    # Dur = HDR.EVENT.DUR;
    Typ = HDR.annotations.description 

    k = 0
    data_2 = np.zeros((1000, 22, 288))

    for j in range(len(Typ)):
        if Typ[j] == 768:
            k += 1
            data_2[:,:,k-1] = s[Pos[j]+500:Pos[j]+1500, 0:22]

    # wipe off NaN
    data_2[np.isnan(data_2)] = 0

    ## preprocessing
    # option - band-pass filter
    fc = 250 # sampling rate
    Wl = 4; Wh = 40 # pass band
    Wn = [Wl*2/float(fc), Wh*2/float(fc)]
    b, a = cheby2(6, 60, Wn)

    # a better filter for 4-40 Hz band-pass
    # fc = 250
    # Wl = 4; Wh = 40
    # Wn = [Wl*2/float(fc), Wh*2/float(fc)]
    # b, a = cheby2(8, 20, Wn)

    data_1 = np.zeros((1000, 22, 288))
    for j in range(288):
        data_1[:,:,j] = filtfilt(b, a, data_1[:,:,j])
        data_2[:,:,j] = filtfilt(b, a, data_2[:,:,j])

    # option - a simple standardization
    #{
    # eeg_mean = np.mean(data, axis=2)
    # eeg_std = np.std(data, axis=2, ddof=1)
    # fb_data = (data-eeg_mean[:,:,np.newaxis])/eeg_std[:,:,np.newaxis]
    #}

    ## Save the data to a mat file
    data = data_1
    label = label_1
    # label = t_label + 1
    saveDir = 'D:/MI/standard_2a_data/A0' + str(subject_index) + 'T.mat'
    sio.savemat(saveDir, {'data': data, 'label': label})

    data = data_2
    label = label_2
    saveDir = 'D:/MI/standard_2a_data/A0' + str(subject_index) + 'E.mat'
    sio.savemat(saveDir, {'data': data, 'label': label})

if __name__ == "__main__":
    getData(6)


