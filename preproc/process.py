# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
Neuromod processing utilities
"""
# dependencies
import os
import json
import click
import numpy as np
import pandas as pd

# high-level processing utils
from heartpy import process
from systole.correction import correct_rr, correct_peaks
from neurokit2 import eda_process, rsp_process, ecg_peaks, ppg_findpeaks, ecg_process

# signal utils
from neurokit2.misc import as_vector
from systole.utils import input_conversion
from neurokit2 import signal_rate, signal_filter
from neurokit2.signal.signal_formatpeaks import _signal_from_indices

# home brewed cleaning utils
from clean import neuromod_ecg_clean


def neuromod_bio_process(tsv=None, h5=None, df=None, sampling_rate=10000.):
    """
    Process biosignals.

    Parameters
    ----------
    tsv : str
        The directory of BIDSified biosignal recording.
    h5 : str
        Default to None
    df : DataFrame
        pandas DataFrame object
        Default to None.
    sampling_rate : float
        The sampling frequency of the data (in Hz, i.e., samples/second).
        Default to 10000.

    Returns
    -------
    bio_df : DataFrame
    bio_info : dict
    """
    if df and tsv and h5 is None:
        raise ValueError(
            "You have to give at least one of the two \n" "parameters: tsv or df"
        )

    if tsv is not None:
        df = pd.read_csv(tsv, sep="\t", compression="gzip")
        print("Reading TSV file")

    if h5 is not None:
        df = pd.read_hdf(h5, key="bio_df")
        sampling_rate = pd.read_hdf(h5, key="sampling_rate")
    # initialize returned objects
    if df is not None:
        print("Reading pandas DataFrame")

        bio_info = {}
        bio_df = pd.DataFrame()

        # initialize each signals
        ppg_raw = df["PPG"]
        rsp_raw = df["RSP"]
        eda_raw = df["EDA"]
        ecg_raw = df["ECG"]

        # ppg
        ppg, ppg_info = neuromod_ppg_process(ppg_raw, sampling_rate=sampling_rate)
        bio_info.update(ppg_info)
        bio_df = pd.concat([bio_df, ppg], axis=1)

        # ecg
        ecg, ecg_info = neuromod_ecg_process(ecg_raw, sampling_rate=sampling_rate)
        bio_info.update(ecg_info)
        bio_df = pd.concat([bio_df, ecg], axis=1)

        #  rsp
        rsp, rsp_info = rsp_process(
            rsp_raw, sampling_rate=sampling_rate, method="khodadad2018"
        )
        bio_info.update(rsp_info)
        bio_df = pd.concat([bio_df, rsp], axis=1)
        print("Respiration workflow: done")

        #  eda
        eda, eda_info = eda_process(eda_raw, sampling_rate, method="neurokit")
        bio_info.update(eda_info)
        bio_df = pd.concat([bio_df, eda], axis=1)
        print("Electrodermal activity workflow: done")

        # return a dataframe
        bio_df["TTL"] = df["TTL"]
        bio_df["time"] = df["time"]

    return (bio_df, bio_info)


def neuromod_process_cardiac(
    signal_raw, signal_cleaned, sampling_rate=10000, data_type="PPG"
):
    """
    Process cardiac signal.

    Extract features of interest for neuromod project.

    Parameters
    ----------
    signal_raw : array
        Raw PPG/ECG signal.
    signal_cleaned : array
        Cleaned PPG/ECG signal.
    sampling_rate : int
        The sampling frequency of `signal_raw` (in Hz, i.e., samples/second).
        Defaults to 10000.
    data_type : str
        Precise the type of signal to be processed. The function currently 
        support PPG and ECG signal processing.
        Default to 'PPG'.
    Returns
    -------
    signals : DataFrame
        A DataFrame containing the cleaned PPG/ECG signals.
        - the raw signal.
        - the cleaned signal.
        - the heart rate as measured based on PPG/ECG peaks.
        - the PPG/ECG peaks marked as "1" in a list of zeros.
    info : dict
        Dictionary containing list of intervals between peaks.
    """
    # heartpy
    print("HeartPy processing started")
    wd, m = process(
        signal_cleaned,
        sampling_rate,
        reject_segmentwise=True,
        interp_clipping=True,
        report_time=True,
    )
    cumsum = 0
    rejected_segments = []
    for i in wd["rejected_segments"]:
        cumsum += int(np.diff(i) / sampling_rate)
        rejected_segments.append((int(i[0]), int(i[1])))
    print("Heartpy found peaks")

    # Find peaks
    print("Neurokit processing started")
    if data_type in ["ecg", "ECG"]:
        _, info = ecg_peaks(
            ecg_cleaned=signal_cleaned,
            sampling_rate=sampling_rate,
            method="nabian2018",
            correct_artifacts=True,
        )
        info[f"{data_type.upper()}_Peaks"] = info[f"{data_type.upper()}_R_Peaks"]
    elif data_type in ["ppg", "PPG"]:
        info = ppg_findpeaks(signal_cleaned, sampling_rate=sampling_rate)
    else:
        print("Please use a valid data type: 'ECG' or 'PPG'")

    info[f"{data_type.upper()}_Peaks"] = info[f"{data_type.upper()}_Peaks"].tolist()
    peak_list_nk = _signal_from_indices(
        info[f"{data_type.upper()}_Peaks"], desired_length=len(signal_cleaned)
    )
    print("Neurokit found peaks")

    # peak to intervals
    rr = input_conversion(
        info[f"{data_type.upper()}_Peaks"],
        input_type="peaks_idx",
        output_type="rr_ms",
        sfreq=sampling_rate,
    )

    # correct beat detection
    corrected, (nMissed, nExtra, nEctopic, nShort, nLong) = correct_rr(rr)
    corrected_peaks = correct_peaks(peak_list_nk, n_iterations=4)
    print("systole corrected RR series")
    # Compute rate based on peaks
    rate = signal_rate(
        info[f"{data_type.upper()}_Peaks"],
        sampling_rate=sampling_rate,
        desired_length=len(signal_raw),
    )

    # sanitize info dict
    info.update(
        {
            f"{data_type.upper()}_ectopic": nEctopic,
            f"{data_type.upper()}_short": nShort,
            f"{data_type.upper()}_long": nLong,
            f"{data_type.upper()}_extra": nExtra,
            f"{data_type.upper()}_missed": nMissed,
            f"{data_type.upper()}_clean_rr_systole": corrected.tolist(),
            f"{data_type.upper()}_clean_rr_hp": [float(v) for v in wd["RR_list_cor"]],
            f"{data_type.upper()}_rejected_segments": rejected_segments,
            f"{data_type.upper()}_cumulseconds_rejected": int(cumsum),
            f"{data_type.upper()}_%_rejected_segments": float(
                cumsum / (len(signal_raw) / sampling_rate)
            ),
        }
    )
    if data_type in ["ecg", "ECG"]:
        info[f"{data_type.upper()}_R_Peaks"] = info[f"{data_type.upper()}_R_Peaks"].tolist()
        del info[f"{data_type.upper()}_Peaks"]

    # Prepare output
    signals = pd.DataFrame(
        {
            f"{data_type.upper()}_Raw": signal_raw,
            f"{data_type.upper()}_Clean": signal_cleaned,
            f"{data_type.upper()}_Peaks_NK": peak_list_nk,
            f"{data_type.upper()}_Peaks_Systole": corrected_peaks["clean_peaks"],
            f"{data_type.upper()}_Rate": rate,
        }
    )

    return signals, info


def ppg_process(ppg_raw, sampling_rate=10000):
    """
    Process PPG signal.

    Custom processing function for neuromod PPG acquisition.

    Parameters
    ----------
    ppg_raw : vector
        The raw PPG channel.
    sampling_rate : int
        The sampling frequency of `ppg_signal` (in Hz, i.e., samples/second).
        Default to 10000.

    Returns
    -------
    signals : DataFrame
        A DataFrame containing the cleaned ppg signals.
        - *"PPG_Raw"*: the raw signal.
        - *"PPG_Clean"*: the cleaned signal.
        - *"PPG_Rate"*: the heart rate as measured based on PPG peaks.
        - *"PPG_Peaks"*: the PPG peaks marked as "1" in a list of zeros.
    info : dict
        Dictionary containing list of intervals between peaks.
    """
    ppg_signal = as_vector(ppg_raw)

    # Prepare signal for processing
    ppg_cleaned = signal_filter(
        ppg_signal, sampling_rate=sampling_rate, lowcut=0.5, highcut=8, order=3
    )
    print("PPG Cleaned")
    # Process clean signal
    signals, info = neuromod_process_cardiac(
        ppg_signal, ppg_cleaned, sampling_rate=10000, data_type="PPG"
    )

    return signals, info


def ecg_process(ecg_raw, trigger_pulse, sampling_rate=10000, method="bottenhorn"):
    """
    Process ECG signal.

    Custom processing for neuromod ECG acquisition.

    Parameters
    -----------
    ecg_raw : vector
        The raw ECG channel.
    sampling_rate : int
        The sampling frequency of `ecg_signal` (in Hz, i.e., samples/second).
        Default to 10000.
    method : str
        The processing pipeline to apply.
        Default to 'bottenhorn'.

    Returns
    -------
    signals : DataFrame
        A DataFrame containing the cleaned ppg signals.
        - *"ECG_Raw"*: the raw signal.
        - *"ECG_Clean"*: the cleaned signal.
        - *"ECG_Rate"*: the heart rate as measured based on PPG peaks.
    info : dict
        Dictionary containing list of peaks.
    """
    ecg_signal = as_vector(ecg_raw)

    # Prepare signal for processing
    ecg_cleaned = neuromod_ecg_clean(
        ecg_signal, trigger_pulse, sampling_rate=sampling_rate, method=method, me=True
    )
    print("ECG Cleaned")
    # Process clean signal
    signals, info = neuromod_process_cardiac(
        ecg_signal, ecg_cleaned, sampling_rate=sampling_rate, data_type="ECG"
    )

    return signals, info


def eda_process():
    # TO DO
    return


def load_json(filename):
    """
    Parameters
    ----------
    filename : str
        File path of the .json to load.

    Returns
    -------
    data : dict
        Dictionary with the content of the .json passed in argument.
    """
    tmp = open(filename)
    data = json.load(tmp)
    tmp.close()

    return data


def load_segmented_runs(source, sub, ses):
    """
    Parameters
    ----------
    source : str
        The main directory contaning the segmented runs.
    sub : str
        The id of the subject.
    ses : str
        The id of the session.

    Returns
    -------
    data_tsv : list
        List containing dataframes with the raw signal for each run.
    filenames : list
        List contaning the filename for each segmented run without the extension.
    """
    data_tsv, filenames = [], []
    files_tsv = [f for f in os.listdir(os.path.join(source, sub, ses)) if "tsv.gz" in f]
    # Remove files ending with _01
    files_tsv = [f for f in files_tsv if "_01." not in f]
    files_tsv.sort()

    for tsv in files_tsv:
        filename = tsv.split(".")[0]
        filenames.append(filename)
        print(f"---Reading data for {sub} {ses}: run {filename[-2:]}---")
        json = filename + ".json"
        print("Reading json file")
        data_json = load_json(os.path.join(source, sub, ses, json))
        print("Reading tsv file")
        data_tsv.append(
            pd.read_csv(
                os.path.join(source, sub, ses, tsv),
                sep="\t",
                compression="gzip",
                names=data_json["Columns"],
            )
        )

    return data_tsv, filenames

@click.command()
@click.argument("source", type=str)
@click.argument("sub", type=str)
@click.argument("ses", type=str)
@click.argument("outdir", type=str)
@click.argument("save", type=bool)
@click.argument("data_type", default=None)
def process_physio_data(source, sub, ses, outdir, save, data_type):
    """
    Run processing pipeline on specified biosignals.

    Parameters
    ----------
    source : str
        The main directory contaning the segmented runs.
    sub : str
        The id of the subject.
    ses : str
        The id of the session.
    outdir : str
        The directory to save the outputs.
    save : bool
        Indicate if the outputs should be saved or not.
    data_type : list
        The list of the channels to process.

    Examples
    --------
    In terminal
    >>> python process.py /home/user/dataset/converted/ sub-01 ses-001 /home/user/dataset/derivatives/ True ["PPG", "ECG", "RSP", "EDA"]
    """
    data_type = [d.upper() for d in json.loads(data_type)]
    # PPG processing pipeline
    if "PPG" in data_type:
        process_ppg_data(source, sub, ses, outdir, save)
    # ECG processing pipeline
    if "ECG" in data_type:
        process_ecg_data(source, sub, ses, outdir, save)
    # RSP processing pipeline
    if "RSP" in data_type:
        process_rsp_data(source, sub, ses, outdir, save)
    # EDA processing pipeline
    if "EDA" in data_type:
        process_eda_data(source, sub, ses, outdir, save)

def process_ppg_data(source, sub, ses, outdir, save=True):
    """
    Parameters
    ----------
    source : str
        The main directory contaning the segmented runs.
    sub : str
        The id of the subject.
    ses : str
        The id of the session.
    outdir : str
        The directory to save the outputs.
    save : bool
        Indicate if the outputs should be saved or not.
        Default to True.

    Returns
    -------
    signals : DataFrame
        Dataframe containing the signal cleaned and processed.
    info : dict
        Dictionnary containing the info of peaks and sampling rate.
    """
    data_tsv, filenames_tsv = load_segmented_runs(source, sub, ses)
    for idx, d in enumerate(data_tsv):
        print(
            f"---Processing PPG signal for {sub} {ses}: run {filenames_tsv[idx][-2:]}---"
        )
        signals, info = ppg_process(d["PPG"], sampling_rate=10000)
        if save:
            print("Saving processed data")
            signals.to_csv(
                os.path.join(
                    outdir, sub, ses, f"{filenames_tsv[idx]}" + "_ppg_signals" + ".tsv"
                ),
                sep="\t",
            )
            with open(
                os.path.join(
                    outdir, sub, ses, f"{filenames_tsv[idx]}" + "_ppg_info" + ".json"
                ),
                "w",
            ) as fp:
                json.dump(info, fp)

    return signals, info

def process_ecg_data(source, sub, ses, outdir, save=True):
    """
    Parameters
    ----------
    source : str
        main directory contaning the segmented runs
    sub : str
        id of the subject
    ses : str
        id of the session
    outdir : str
        directory to save the outputs
    save : bool
        indicate if the outputs should be saved or not 
        Default to True

    Returns
    -------
    signals :  DataFrame
        Dataframe containing the signal cleaned and processed
    info : dict
        dictionnary containing the info of peaks and sampling rate
    """
    data_tsv, filenames_tsv = load_segmented_runs(source, sub, ses)
    for idx, d in enumerate(data_tsv):
        print(
            f"---Processing ECG signal for {sub} {ses}: run {filenames_tsv[idx][-2:]}---"
        )
        print("--Cleaning the signal---")
        signals, info = ecg_process(
            d["ECG"], d["TTL"], sampling_rate=10000, method="bottenhorn"
        )
        
        if save:
            print("Saving processed data")
            signals.to_csv(
                os.path.join(
                    outdir, sub, ses, f"{filenames_tsv[idx]}_ecg_signals.tsv"
                ),
                sep="\t",
            )
            with open(
                os.path.join(
                    outdir, sub, ses, f"{filenames_tsv[idx]}_ecg_info.json"
                ),
                "w",
            ) as fp:
                json.dump(info, fp)

    return signals, info

def process_rsp_data(source, sub, ses, outdir, save=True):
    """
    Parameters
    ----------
    source : str
        The main directory contaning the segmented runs.
    sub : str
        The id of the subject.
    ses : str
        The id of the session.
    outdir : str
        The directory to save the outputs.
    save : bool
        Indicate if the outputs should be saved or not.
        Default to True.

    Returns
    -------
    signals : DataFrame
        Dataframe containing the signal cleaned and processed.
    """
    data_tsv, filenames_tsv = load_segmented_runs(source, sub, ses)
    for idx, d in enumerate(data_tsv):
        print(
            f"---Processing RSP signal for {sub} {ses}: run {filenames_tsv[idx][-2:]}---"
        )
        signals, _ = rsp_process(d["RSP"], sampling_rate=10000, method="khodadad2018")
        if save:
            print("Saving processed data")
            signals.to_csv(
                os.path.join(
                    outdir, sub, ses, f"{filenames_tsv[idx]}_rsp_signals.tsv"
                ),
                sep="\t",
            )

    return signals

def process_eda_data(source, sub, ses, outdir, save=True):
    """
    Parameters
    ----------
    source : str
        The main directory contaning the segmented runs.
    sub : str
        The id of the subject.
    ses : str
        The id of the session.
    outdir : str
        The directory to save the outputs.
    save : bool
        Indicate if the outputs should be saved or not.
        Default to True.

    Returns
    -------
    """
    data_tsv, filenames_tsv = load_segmented_runs(source, sub, ses)
    for idx, d in enumerate(data_tsv):
        print(
            f"---Processing EDA signal for {sub} {ses}: run {filenames_tsv[idx][-2:]}---"
        )
        signals, info = eda_process(d["EDA"], sampling_rate=10000)
        print("--Cleaning the signal---")
    return

if __name__ == "__main__":
    # Physio processing pipeline
    process_physio_data()
