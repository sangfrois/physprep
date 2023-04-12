# -*- coding: utf-8 -*-
# !/usr/bin/env python -W ignore::DeprecationWarning
"""Physiological data conversion to BIDS"""

import gc
import os
import json
import click
import logging
import pandas as pd
from phys2bids.phys2bids import phys2bids

import sys
sys.path.append("../utils")

@click.command()
@click.argument("root", type=click.Path(exists=True))
@click.argument("save", type=click.Path())
@click.argument("sub", type=str)
@click.option("--ses", type=str, default=None, required=False)
@click.option("--tr", type=float, default=1.49, required=False)
@click.option("--ch_name", default=None, required=False)
def call_convert(root, save, sub, ses=None, tr=1.49, ch_name=None):
    """
    Call `convert` function only if `convert.py` is called as CLI

    For parameters description, please refer to the documentation of the `convert` function
    """
    if ch_name is not None:
        ch_name = json.loads(ch_name)
    convert(root, save, sub, ses=None, tr=1.49, ch_name=None)

def convert(root, save, sub, ses=None, tr=1.49, ch_name=None):
    """
    Phys2Bids conversion for one subject data

    Parameters
    ----------
    root : path
        main directory containing the biopac data (e.g. /to/dataset/info)
    save : path
        directory to save data and to retrieve acquisition info (`.json file`)
    subject : string
        name of path for a specific subject (e.g.'sub-03')
    sessions : list
        specific session numbers can be listed (e.g. ['ses-001', 'ses-002']
    tr : float
        tr value used for mri acquisition
    ch_name : list
        specify the name of the channels in the acqknowledge file

    Examples
    --------
    In script
    >>> convert(root="/home/user/dataset/info", save="/home/user/dataset/convert/", sub="sub-01", ses="ses-001", tr=1.49, ch_name=["EDA", "ECG", "TTL"])
    In terminal
    >>> python convert.py /home/user/dataset/info /home/user/dataset/convert/ sub-01 --ses ses-001 --tr 1.49 --ch_name '["EDA", "ECG", "TTL"]'
    NOTE: if you want to specify the `ch_name` using the CLI, specify your list inside single quote ('') just like the example above.
    """
    logger = logging.getLogger(__name__)
    # fetch info
    fetcher = f"{sub}_volumes_all-ses-runs.json"
    logger.info(f"Reading fetcher:\n{os.path.join(root, sub, fetcher)}")
    info = pd.read_json(os.path.join(root, sub, f"{sub}_volumes_all-ses-runs.json"))
    # define sessions
    if ses is None:
        ses = info.columns
    elif isinstance(ses, list) is False:
        ses = [ses]
    # Define ch_name
    if info[ses]['ch_name'] is None:
        logger.info("Warning: you did not specify a value for ch_name, the values that will be use are the following: ")
        ch_name = ["EDA", "PPG", "ECG", "TTL", "RSP"]
        info[ses]['ch_name'] = ch_name
        logger.info("Please make sure, those values are the right ones :\n{ch_name}")
        chtrig=4
    else:
        # Define chtrig ; should find a way to find it from a list of possible values
        chtrig = info[ses]['ch_name'].index('TTL')

    # iterate through info
    for col in ses:
        # skip empty sessions
        if info[col] is None:
            continue
        logger.info(col)

        # Iterate through files in each session and run phys2bids
        filename = info[col]["in_file"]
        if filename is list:
            for i in range(len(filename) - 1):
                phys2bids(
                    filename[i],
                    info=False,
                    indir=os.path.join(root, sub, col),
                    outdir=os.path.join(save, sub, col),
                    heur_file=None,
                    sub=sub[-2:],
                    ses=col[-3:],
                    chtrig=chtrig,
                    chsel=None,
                    num_timepoints_expected=info[col]["recorded_triggers"][
                        f"run-0{i+1}"
                    ],
                    tr=info[col]["tr"],
                    thr=4,
                    pad=9,
                    ch_name=info[col]["ch_name"],
                    yml="",
                    debug=False,
                    quiet=False,
                )
        else:
            try:
                phys2bids(
                    filename,
                    info=False,
                    indir=os.path.join(root, "physio", sub, col),
                    outdir=os.path.join(save, sub, col),
                    heur_file=None,
                    sub=sub[-2:],
                    ses=col[-3:],
                    chtrig=chtrig,
                    chsel=None,
                    num_timepoints_expected=info[col]["recorded_triggers"]["run-01"],
                    tr=info[col]["tr"],
                    thr=4,
                    pad=9,
                    ch_name=info[col]["ch_name"],
                    yml="",
                    debug=False,
                    quiet=False,
                )
            except AttributeError:
                filename.sort()
                for i in range(len(filename)):
                    print(i)
                    phys2bids(
                        filename[i],
                        info=False,
                        indir=os.path.join(root, sub, col),
                        outdir=os.path.join(save, sub, col),
                        heur_file=None,
                        sub=sub[-2:],
                        ses=col[-3:],
                        chtrig=chtrig,
                        chsel=None,
                        num_timepoints_expected=info[col]["recorded_triggers"][
                            f"run-0{i+1}"
                        ],
                        tr=info[col]["tr"],
                        thr=4,
                        pad=9,
                        ch_name=info[col]["ch_name"],
                        yml="",
                        debug=False,
                        quiet=False,
                    )

            except TypeError:
                print(f"No input file for {col}")
                continue
        gc.collect()
        print("~" * 30)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    call_convert()
