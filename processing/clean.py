# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
Neuromod cleaning utilities
"""
import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal

# ======================================================================
# Photoplethysmograph (PPG)
# =======================================================================

def neuromod_ppg_clean(ppg_signal, sampling_rate=10000.):
    """
    Clean a PPG signal.

    Prepare raw PPG signal for peak detection with specified method.

    Parameters
    ----------
    ppg_signal : list, array or Series
        The raw PPG channel.
    sampling_rate : float
        The sampling frequency of `ppg_signal` (in Hz, i.e., samples/second).
        Defaults to 10000.

    Returns
    -------
    ppg_cleaned : array
        Vector containing the cleaned PPG signal.
    """
    # Apply band pass filter
    ppg_cleaned = nk.signal_filter(
        ppg_signal, sampling_rate=sampling_rate, lowcut=0.5, highcut=8, order=3
    )

    return ppg_cleaned


# ======================================================================
# Electrocardiogram (ECG)
# =======================================================================

def neuromod_ecg_clean(ecg_signal, sampling_rate=10000., method="biopac", me=False):
    """
    Clean an ECG signal.

    Prepare a raw ECG signal for R-peak detection with the specified method.

    Parameters
    ----------
    ecg_signal : list, array or Series
        The raw ECG channel.
    sampling_rate : float
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
        Default to 10000.
    method : str
        The processing pipeline to apply between 'biopac' and 'bottenhorn'.
        Default to 'biopac'.
    me : bool
        Specify if the MRI sequence used was the multi-echo (True) 
        or the single-echo (False). 
        Default to False.

    Returns
    -------
    clean : array
        Vector containing the cleaned ECG signal.
    """
    if me:
        tr = 2.65
        mb = 2
        slices = 70
    else:
        tr = 1.49
        mb = 4
        slices = 60
        
    if method in ["biopac"]:
        clean = _ecg_clean_biopac(ecg_signal, sampling_rate)
    if method in ["bottenhorn", "bottenhorn2022"]:
        # Apply comb band pass filter with Bottenhorn correction
        print("... Applying the corrected comb band pass filter.")
        clean = _ecg_clean_bottenhorn(
            ecg_signal, sampling_rate=sampling_rate, tr=tr, mb=mb, slices=slices
        )

    return clean


# =============================================================================
# ECG internal : biopac recommendations
# =============================================================================
def _ecg_clean_biopac(ecg_signal, sampling_rate=10000., tr=1.49, slices=60, Q=10):
    """
    Single-band sequence gradient noise reduction.

    This function is a reverse-engineered appropriation of BIOPAC's application note 242.
    It only applies to signals polluted by single-band (f)MRI sequence.

    Parameters
    ----------
    ecg_signal : array
        The ECG channel.
    sampling_rate: float
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
        Default to 10000.
    tr : int
        The time Repetition of the MRI scanner.
    slices :
        The number of volumes acquired in the tr period.
    Q : int
        The filter quality factor.

    Returns
    -------
    ecg_clean : array
        The cleaned ECG signal.

    References
    ----------
    Biopac Systems, Inc. Application Notes: application note 242
        ECG Signal Processing During fMRI
        https://www.biopac.com/wp-content/uploads/app242x.pdf
    """
    # Setting scanner sequence parameters
    nyquist = np.float64(sampling_rate / 2)
    notches = {"slices": slices / tr, "tr": 1 / tr}
    # remove baseline wandering
    ecg_clean = nk.signal_filter(
        ecg_signal,
        sampling_rate=int(sampling_rate),
        lowcut=2,
    )
    # Filtering at specific harmonics
    ecg_clean = _comb_band_stop(notches, nyquist, ecg_clean, Q)
    # bandpass filtering
    ecg_clean = nk.signal_filter(
        ecg_clean,
        sampling_rate=sampling_rate,
        lowcut=2,
        highcut=20,
        method="butter",
        order=5,
    )

    return ecg_clean


def _ecg_clean_bottenhorn(ecg_signal, sampling_rate=10000., tr=1.49, mb=4, slices=60, Q=10):
    """
    Multiband sequence gradient noise reduction.

    Parameters
    ----------
    ecg_signal : array
        The ECG channel.
    sampling_rate : float
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
        Default to 10000.
    tr : float
        The time Repetition of the MRI scanner.
    mb : 4
        The multiband acceleration factor.
    slices : int
        The number of volumes acquired in the tr period.
    Q : int
        The filter quality factor.

    Returns
    -------
    ecg_clean : array
        The cleaned ECG signal.

    References
    ----------
    Bottenhorn, K. L., Salo, T., Riedel, M. C., Sutherland, M. T., Robinson, J. L.,
        Musser, E. D., & Laird, A. R. (2021). Denoising physiological data collected 
        during multi-band, multi-echo EPI sequences. bioRxiv, 2021-04.
        https://doi.org/10.1101/2021.04.01.437293

    See also
    --------
    https://neuropsychology.github.io/NeuroKit/_modules/neurokit2/signal/signal_filter.html#signal_filter
    """
    # Setting scanner sequence parameters
    nyquist = np.float64(sampling_rate / 2)
    notches = {"slices": slices / mb / tr, "tr": 1 / tr}

    # Remove low frequency artefacts: respiration & baseline wander using high pass butterworth filter (order=2)
    print("... Applying high pass filter.")
    ecg_clean = nk.signal_filter(
        ecg_signal, 
        sampling_rate=sampling_rate, 
        lowcut=2, 
        method="butter"
    )
    # Filtering at fundamental and specific harmonics per Biopac application note #265
    print("... Applying notch filter.")
    ecg_clean = _comb_band_stop(notches, nyquist, ecg_clean, Q)
    # Low pass filtering at 40Hz per Biopac application note #242
    print("... Applying low pass filtering.")
    ecg_clean = nk.signal_filter(ecg_signal,
        sampling_rate=sampling_rate, 
        highcut=40
    )
    # bandpass filtering
    ecg_clean = nk.signal_filter(
        ecg_clean,
        sampling_rate=sampling_rate,
        lowcut=2,
        highcut=20,
        method="butter",
        order=5,
    )

    return ecg_clean


# =============================================================================
# EDA
# =============================================================================
def neuromod_eda_clean(eda_signal, sampling_rate=10000., me=True, Q=10): 
    """
    Multiband sequence gradient noise reduction.

    Parameters
    ----------
    eda_signal : array
        The EDA channel.
    sampling_rate : float
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
        Default to 10000.
    tr : float
        The time Repetition of the MRI scanner.
    mb : int
        The multiband acceleration factor.
    slices : int
        The number of volumes acquired in the tr period.
    Q : int
        The filter quality factor.

    Returns
    -------
    eda_clean : array
        The cleaned EDA signal.

    References
    ----------
    Bottenhorn, K. L., Salo, T., Riedel, M. C., Sutherland, M. T., Robinson, J. L.,
        Musser, E. D., & Laird, A. R. (2021). Denoising physiological data collected 
        during multi-band, multi-echo EPI sequences. bioRxiv, 2021-04.
        https://doi.org/10.1101/2021.04.01.437293

    See also
    --------
    https://neuropsychology.github.io/NeuroKit/functions/eda.html#preprocessing
    """
    if me:
        tr = 2.65
        mb = 2
        slices = 70
    else:
        tr = 1.49
        mb = 4
        slices = 60

    # Setting scanner sequence parameters
    nyquist = np.float64(sampling_rate / 2)
    notches = {"slices": slices / mb / tr, "tr": 1 / tr}

    # Low pass filtering at 3Hz, order=4
    eda_clean = nk.eda_clean(eda_signal, sampling_rate=sampling_rate)
    # Filtering at fundamental and specific harmonics 
    print("... Applying notch filter.")
    eda_clean = _comb_band_stop(notches, nyquist, eda_clean, Q)
    
    return eda_clean


# =============================================================================
# General functions
# =============================================================================

def _comb_band_stop(notches, nyquist, filtered, Q):
    """
    A serie of notch filters aligned with the scanner gradient's harmonics.
    
    Parameters
    ----------
    notches : dict
        Frequencies to use in the IIR notch filter.
    nyquist : float
        The Nyquist frequency.
    filtered : array
        Data to be filtered.
    Q : int
        The filter quality factor.

    Returns
    -------
    filtered : array
        The filtered signal.

    References
    ----------
    Biopac Systems, Inc. Application Notes: application note 242
        ECG Signal Processing During fMRI
        https://www.biopac.com/wp-content/uploads/app242x.pdf

    See also
    --------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
    """
    # band stoping each frequency specified with notches dict
    for notch in notches:
        for i in np.arange(1, int(nyquist / notches[notch])):
            f0 = notches[notch] * i
            w0 = f0 / nyquist
            b, a = signal.iirnotch(w0, Q)
            filtered = signal.filtfilt(b, a, filtered)
    return filtered
