# -*- coding: utf-8 -*-
# !/usr/bin/env python -W ignore::DeprecationWarning
"""Physiological data quality assessment"""

import os 
import json
from vital_sqi import sqi

def sqi_cardiac(signal, type='ecg', sampling_rate=10000):
    """
    Extract SQI for ECG/PPG processed signal

    Parameters
    ----------
    signal : DataFrame
        Output from the process.py script
    type : str  
        Type of the signal. Valid options include 'ecg' and 'ppg'.
        Default to 'ecg'.
    sampling_rate : int
        The sampling frequency of `signal_raw` (in Hz, i.e., samples/second).
        Default to 10000.

    Returns
    -------
    summary : dict
        Dictionnary containing sqi values.

    Examples
    --------

    Reference
    ---------
    Le, V. K. D., Ho, H. B., Karolcik, S., Hernandez, B., Greeff, H., 
        Nguyen, V. H., ... & Clifton, D. (2022). vital_sqi: A Python 
        package for physiological signal quality control. 
        Frontiers in Physiology, 2248.
    """
    summary = {}
    
    return summary

def sqi_eda(signal, sampling_rate=10000):
    """
    Extract SQI for EDA processed signal

    Parameters
    ----------
    signal : DataFrame
        Output from the process.py script
    sampling_rate : int
        The sampling frequency of `signal_raw` (in Hz, i.e., samples/second).
        Default to 10000.

    Returns
    -------
    summary : dict
        Dictionnary containing sqi values.

    Examples
    --------
    """

def sqi_rsp(signal, sampling_rate=10000):
    """
    Extract SQI for respiratory processed signal

    Parameters
    ----------
    signal : DataFrame
        Output from the process.py script
    sampling_rate : int
        The sampling frequency of `signal_raw` (in Hz, i.e., samples/second).
        Default to 10000.

    Returns
    -------
    summary : dict
        Dictionnary containing sqi values.

    Examples
    --------
    """

def generate_report(sqi_info, save):
    """
    Generate quality assessment report in html format

    Parameters
    ----------
    sqi_info : dict or list of dict
        Dictionnary contaning sqi values for a specified signal.
        List of dictationaries can be passed to include multiple 
        signals to the report.
    save : str
        Directory to save the generated report.

    Examples
    --------
    """