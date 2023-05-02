# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
Neuromod cleaning utilities
"""
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal


def neuromod_bio_clean(tsv=None, data=None, h5=None, sampling_rate=1000):
    """
    Filter biosignals.

    NOTE : add downsampling option

    Prepare biosignals for extraction of characteristics of physiological
    activity with a set of filters and smoothing functions

    Parameters
    ----------
    tsv :
        directory of BIDSified biosignal recording
    h5 :
        directory of h5 file
    data (optional) :
        pandas DataFrame object
    """

    # check input and sanitize
    # if data and tsv is None:
    # raise ValueError("You have to give at least one of the two \n"
    # "parameters: tsv or df")

    if tsv is not None:
        data = pd.read_csv(tsv, sep="t", compression="gz")

    if h5 is not None:
        data = pd.read_hdf(h5, key="bio_df")
        sampling_rate = pd.read_hdf(h5, key="sampling_rate")

    # sanitize by columns
    if "RSP" in data.keys():
        rsp = data["RSP"]
    else:
        rsp = None
    if "EDA" in data.keys():
        eda = data["EDA"]
    else:
        eda = None
    if "ECG" in data.keys():
        ecg = data["ECG"]
    elif "EKG" in data.keys():
        ecg = data["EKG"]
    else:
        ecg = None
    if "PPG" in data.keys():
        ppg = data["PPG"]
    else:
        ppg = None

    # Keep unkown columns in data
    cols = ["ECG", "EKG", "PPG", "RSP", "EDA"]
    keep_keys = [key for key in data.keys() if key not in cols]
    if len(keep_keys) != 0:
        keep = data[keep_keys]
    else:
        keep = None

    # initialize output
    bio_df = pd.DataFrame()

    # sanitize input signals
    # PPG_
    if ppg is not None:
        ppg = nk.as_vector(ppg)
        ppg_clean = neuromod_ppg_clean(ppg, sampling_rate=sampling_rate)

        bio_df = pd.concat([bio_df, ppg_clean], axis=1)
    # ECG
    if ecg is not None:
        ecg = nk.as_vector(ecg)
        ecg_clean = neuromod_ecg_clean(ecg, sampling_rate=sampling_rate)
        bio_df = pd.concat([bio_df, ecg_clean], axis=1)

    # RSP
    if rsp is not None:
        rsp = nk.as_vector(rsp)
        rsp_clean = nk.rsp_clean(rsp, sampling_rate=sampling_rate)
        bio_df = pd.concat([bio_df, rsp_clean], axis=1)

    # EDA
    if eda is not None:
        eda = nk.as_vector(eda)
        eda_clean = nk.eda_clean(eda, sampling_rate=sampling_rate)
        eda_clean = _eda_clean_bottenhorn(eda_clean, sampling_rate=sampling_rate)
        bio_df = pd.concat([bio_df, eda_clean], axis=1)

    return bio_df


# ======================================================================
# Photoplethysmograph (PPG)
# =======================================================================


def neuromod_ppg_clean(ppg_signal, sampling_rate=10000, method="nabian2018"):
    """
    Clean a PPG signal.

    Prepare raw PPG signal for peak detection with specified method.

    Parameters
    ----------
    ppg_signal : list, array or Series
        The raw PPG channel.
    sampling_rate : int
        The sampling frequency of `ppg_signal` (in Hz, i.e., samples/second).
        Defaults to 10000.
    method : str
        The processing pipeline to apply. Defaults to 'nabian2018'.
    Returns
    -------
    ppg_cleaned : array
        Vector containing the cleaned PPG signal.
    """
    ppg_cleaned = nk.ppg_clean(ppg_signal, sampling_rate=sampling_rate, method=method)
    return ppg_cleaned


# ======================================================================
# Electrocardiogram (ECG)
# =======================================================================


def neuromod_ecg_clean(
    ecg_signal, trigger_pulse, sampling_rate=10000.0, method="biopac", me=False
):
    """
    Clean an ECG signal.

    Prepare a raw ECG signal for R-peak detection with the specified method.

    Parameters
    ----------
    ecg_signal : list, array or Series
        The raw ECG channel.
    trigger_pulse : list, array or Series
        Trigger channel.
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
        Defaults to 10000.
    method : str
        The processing pipeline to apply. Defaults to 'biopac'.
    me : bool
        Specify if the MRI sequence used was the multi-echo (True) or the single-echo (False). Defaults to False.
    Returns
    -------
    clean : array
        Vector containing the cleaned ECG signal.
    """
    # find a way to pass these variables to the function from the fetcher
    if me:
        tr = 2.65
        mb = 2
        slices = 70
    else:
        tr = 1.49
        mb = 4
        slices = 60
    # choose method
    method = method.lower()  # remove capitalised letters
    if method in ["schmidt", "schmidt2018"]:
        clean = _ecg_clean_schmidt(ecg_signal, sampling_rate)
    # arrange format
    else:
        timeseries = pd.DataFrame(columns=["ECG", "Trigger"])
        timeseries["ECG"] = ecg_signal
        timeseries["Trigger"] = trigger_pulse
        if method in ["biopac"]:
            clean = _ecg_clean_biopac(timeseries, sampling_rate)
        if method in ["bottenhorn", "bottenhorn2022"]:
            # Remove respiration-related noise using a 2Hz highpass filter
            print("---Cleaning respiration-related noise---")
            ecg_signal_hp = _butter_highpass_filter(ecg_signal, 2.0, sampling_rate)
            # Apply comb band pass filter with Bottenhorn correction
            print("---Applying the corrected comb band pass filter---")
            clean = _ecg_clean_bottenhorn(
                ecg_signal_hp, sampling_rate=sampling_rate, tr=tr, mb=mb, slices=slices
            )

    return clean


# =============================================================================
# ECG internal : Schmidt et al. 2016
# =============================================================================
def _ecg_clean_schmidt(ecg_signal, sampling_rate=16000):
    """
    from Schmidt, M., Krug, J. W., & Rose, G. (2016).
    Reducing of gradient induced artifacts on the ECG signal during MRI
    examinations using Wilcoxon filter.
    Current Directions in Biomedical Engineering.
    https://doi.org/10.1515/cdbme-2016-0040
    """
    # enveloppe at least 100 ms in samples
    env = int(0.01 * sampling_rate)
    # initialize empty array
    ecg_clean = np.empty(len(ecg_signal), dtype="float64")

    # iterate through each sample
    for current_sample in range(env, (len(ecg_signal) - 1)):
        # window to convolve median smoothing operation
        past_samples = current_sample - env
        window = np.array(ecg_signal[past_samples:current_sample])

        # compute the walsh averages with Wilcoxon method
        # initialize empty array half the size of window
        walsh_arr = np.empty(int(len(window) / 2))

        # iterate from index 1 to half of window, and -1 to -(half of window)
        # in order to perform reduced computation walsh averages
        for i in range(0, (len(walsh_arr))):
            # -1 is last element and 0 is first
            r_i = -(i + 1)
            # the last element is index of middle of window
            if i != len(walsh_arr) - 1:
                # mirror elements in window are conjugated until middle index
                walsh_arr[i] = (window[i] + window[r_i]) / 2

            # middle value in window is last value of walsh array and stays as
            else:
                walsh_arr[i] = window[i]
        # compute the median of walsh averages array
        ecg_clean[current_sample] = np.median(walsh_arr)
    # Bandpass filtering
    ecg_clean = nk.signal_filter(
        ecg_clean, lowcut=0.05, highcut=45, method="bessel", order=5
    )

    return ecg_clean


# =============================================================================
# ECG internal : biopac recommendations
# =============================================================================
def _ecg_clean_biopac(timeseries, sampling_rate=10000.0, tr=1.49, slices=60, Q=100):
    """
    Single-band sequence gradient noise reduction.

    This function is a reverse-engineered appropriation of BIOPAC's application note 242.
    It only applies to signals polluted by single-band (f)MRI sequence

    Parameters
    -----------
    timeseries : pd.DataFrame
        - ECG signal
        - TTL
    sampling_rate: float
        sampling frequency
    tr : int
        Time Repetitiion of scanner
    slices :
        nb of volumes acquired in the tr period
    Q : int
        filter quality factor
    Returns
    ---------
    filtered : array
        the filtered sign
    References
    -----------
    Biopac Systems, Inc. Application Notes: application note 242
    ECG Signal Processing During fMRI
    https://www.biopac.com/wp-content/uploads/app242x.pdf
    """
    # Setting scanner sequence parameters
    nyquist = np.float64(sampling_rate / 2)
    notches = {"slices": slices / tr, "tr": 1 / tr}
    # find trigger timing
    triggers = timeseries[timeseries["Trigger"] > 4].index.values
    print(triggers)
    # remove baseline wandering
    filtered = nk.signal_filter(
        timeseries["ECG"][triggers[1] : triggers[-1]],
        sampling_rate=int(sampling_rate),
        lowcut=2,
    )
    # Filtering at specific harmonics, with trigger timing info
    filtered = _comb_band_stop(notches, nyquist, filtered, Q, sampling_rate)
    # bandpass filtering
    filtered = nk.signal_filter(
        filtered,
        sampling_rate=sampling_rate,
        lowcut=2,
        highcut=20,
        method="butter",
        order=5,
    )

    return filtered


def _ecg_clean_bottenhorn(
    ecg_signal, sampling_rate=10000.0, tr=1.49, mb=4, slices=60, Q=100
):
    """
    https://github.com/62442katieb/mbme-physio-denoising/blob/main/notebooks/denoising_eda.ipynb
    """
    # Setting scanner sequence parameters
    nyquist = np.float64(sampling_rate / 2)
    notches = {"slices": slices / mb / tr, "tr": 1 / tr}
    # remove baseline wandering
    filtered = nk.signal_filter(ecg_signal, sampling_rate=int(sampling_rate), lowcut=2)
    # Filtering at specific harmonics, with trigger timing info
    filtered = _comb_band_stop(notches, nyquist, filtered, Q)
    # bandpass filtering
    filtered = nk.signal_filter(
        filtered,
        sampling_rate=sampling_rate,
        lowcut=2,
        highcut=20,
        method="butter",
        order=5,
    )

    return filtered


# =============================================================================
# EDA
# =============================================================================
def _eda_clean_bottenhorn(
    eda_signal, sampling_rate=10000.0, Q=100, mb=4, tr=1.49, slices=60
):

    notches = {"slices": slices / mb / tr, "tr": 1 / tr}

    # hp_eda = butter_highpass_filter(scan1['EDA'], 1, fs, order=5)
    bottenhorn_filtered = eda_signal
    for notch in notches:
        bottenhorn_filtered = _comb_band_stop(
            notches[notch],
            np.float64(sampling_rate / 2),
            bottenhorn_filtered,
            Q,
            sampling_rate,
        )


# =============================================================================
# General functions
# =============================================================================


def _comb_band_stop(notches, nyquist, filtered, Q):
    """
    A serie of notch filters aligned with the scanner gradient's harmonics

    Biopac Systems, Inc. Application Notes: application note 242
    ECG Signal Processing During fMRI
    https://www.biopac.com/wp-content/uploads/app242x.pdf
    """
    # band stoping each frequency specified with notches dict
    for notch in notches:
        for i in np.arange(1, (nyquist / notches[notch])):
            f0 = notches[notch] * i
            w0 = f0 / nyquist
            b, a = signal.iirnotch(w0, Q)
            filtered = signal.filtfilt(b, a, filtered)
    return filtered


def _butter_highpass(cutoff, fs, order=5):
    """
    reference: https://github.com/62442katieb/mbme-physio-denoising/blob/main/notebooks/denoising_eda.ipynb
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def _butter_highpass_filter(data, cutoff, fs, order=5):
    """
    reference: https://github.com/62442katieb/mbme-physio-denoising/blob/main/notebooks/denoising_eda.ipynb
    """
    b, a = _butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
