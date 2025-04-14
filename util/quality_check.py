#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 13:58:46 2024

@author: nbnapu
"""
import mne
import os
from collections import defaultdict
import numpy as np
from mne.preprocessing.bads import _find_outliers
import matplotlib.pyplot as plt
import statistics
from math import sqrt
from datetime import timedelta


def compute_weekly_mean(W):
    recording_counts = list(W[0].values())
    mean_ec = statistics.mean(recording_counts)
    
    recording_counts = list(W[1].values())
    mean_eo = statistics.mean(recording_counts)
    
    recording_counts = list(W[2].values())
    mean_pc = statistics.mean(recording_counts)
    
    return mean_ec, mean_eo, mean_pc
    # std_num_recordings = statistics.stdev(recording_counts) 

def get_week_start_date(date):
     return date - timedelta(days=date.weekday())


def get_researcher_id_from_fname(fname):

    split_strings = fname.split('_')
    subject_id = split_strings[0]
    researcher_id = split_strings[1]
    device_id = split_strings[2]
    task = split_strings[3]
    device_name = split_strings[4]
    return subject_id, researcher_id, device_id, task, device_name


def compute_throughput_by_cond(cond, grouped_fnames):
    fnames = grouped_fnames[cond]
    fnames_ok = list()
    eeg_start_dt = list()
    researcher_id_list = list()
   # bad_by_nan_flat=list()
    for i in range(len(fnames)):
        file_object = open('prob_edf_files_%s_tanzania.txt'%cond, 'a')
        try:
            fname =  fnames[i] #'/Users/nbnapu/Documents/Projects/SL-EEG-feasibility/EEG_Files_Device1/LABE351_LABE_01_EO_EPOCFLEX_203786_2023.12.06T11.47.00+03.00.edf'
            raw = mne.io.read_raw_edf(fname) #instance of the mne.io.Raw class
            fnames_ok.append(fname)
            eeg_start_dt.append(raw.info['meas_date'])
            _,researcher_id,_,_,_ = get_researcher_id_from_fname(os.path.basename(fname)) 
            researcher_id_list.append(researcher_id)
        except:
            file_object.write(os.path.basename(fname))
            file_object.write("\n")
            continue
        
    indexed_dt = list(enumerate(eeg_start_dt))
    sorted_indexed_start_dt = sorted(indexed_dt, key=lambda x: x[1])
    sorted_start_dt = [value for index, value in sorted_indexed_start_dt]
    sorted_index = [index for index, value in sorted_indexed_start_dt]
    sorted_researcher_id = [researcher_id_list[i] for i in sorted_index]
    recording_dates = [sorted_start_dt[i].date() for i in range(len(sorted_start_dt))]
   # eeg_start_dt.sort()
   # recording_dates = [ eeg_start_dt[i].date() for i in range(len( eeg_start_dt))]

   
    # Dictionary to store the count of recordings per week
    weekly_recordings = defaultdict(int)
    weekly_researchers = defaultdict(list)
    # Count recordings per week
    for date in recording_dates:
        week_start = get_week_start_date(date)
        weekly_recordings[week_start] += 1
        
    for j in range(len(sorted_researcher_id)):
        week_start = get_week_start_date(recording_dates[j])
        weekly_researchers[week_start].append(sorted_researcher_id[j]) 
        
        
    # recording_counts = list(weekly_recordings.values())
    # mean_num_recordings = statistics.mean(recording_counts)
    # std_num_recordings = statistics.stdev(recording_counts)    
    return weekly_recordings, weekly_researchers

def compute_throughput_by_researcher(researcher_id, cond, grouped_fnames):
    fnames = grouped_fnames[researcher_id][cond]
    fnames_ok = list()
    eeg_start_dt = list()
   # bad_by_nan_flat=list()
    for i in range(len(fnames)):
        file_object = open('prob_edf_files_%s_tanzania.txt'%cond, 'a')
        try:
            fname =  fnames[i] #'/Users/nbnapu/Documents/Projects/SL-EEG-feasibility/EEG_Files_Device1/LABE351_LABE_01_EO_EPOCFLEX_203786_2023.12.06T11.47.00+03.00.edf'
            raw = mne.io.read_raw_edf(fname, preload=(True)) #instance of the mne.io.Raw class
            fnames_ok.append(fname)
            eeg_start_dt.append(raw.info['meas_date'])
        except:
            file_object.write(os.path.basename(fname))
            file_object.write("\n")
            continue
        
    eeg_start_dt.sort()
    recording_dates = [ eeg_start_dt[i].date() for i in range(len( eeg_start_dt))]

   
    # Dictionary to store the count of recordings per week
    weekly_recordings = defaultdict(int)

    # Count recordings per week
    for date in recording_dates:
        week_start = get_week_start_date(date)
        weekly_recordings[week_start] += 1
        
        
    #recording_counts = list(weekly_recordings.values())
    #mean_num_recordings = statistics.mean(recording_counts)
    #std_num_recordings = statistics.stdev(recording_counts)    
    return weekly_recordings
  
def _freqs_power(data, sfreq, freqs):
    fs, ps = _efficient_welch(data, sfreq)
    try:
        return np.sum([ps[..., np.searchsorted(fs, f)] for f in freqs], axis=0)
    except IndexError:
        raise ValueError(
            ("Insufficient sample rate to  estimate power at {} Hz for line "
             "noise detection. Use the 'metrics' parameter to disable the "
             "'line_noise' metric.").format(freqs))

def _efficient_welch(data, sfreq):
    """Calls scipy.signal.welch with parameters optimized for greatest speed
    at the expense of precision. The window is set to ~10 seconds and windows
    are non-overlapping.
    Parameters
    ----------
    data : array, shape (..., n_samples)
        The timeseries to estimate signal power for. The last dimension
        is assumed to be time.
    sfreq : float
        The sample rate of the timeseries.
    Returns
    -------
    fs : array of float
        The frequencies for which the power spectra was calculated.
    ps : array, shape (..., frequencies)
        The power spectra for each timeseries.
    """
    from scipy.signal import welch
    nperseg = min(data.shape[-1],
                  2 ** int(np.log2(10 * sfreq) + 1))  # next power of 2

    return welch(data, sfreq, nperseg=nperseg, noverlap=0, axis=-1)


def list_edf_files(dir):
    edf_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.edf'):
                edf_files.append(os.path.join(root, file))
    return edf_files





def group_files_by_researcher_and_conds(edf_files, researcher_ids, identifiers):
    grouped_files = defaultdict(lambda: defaultdict(list))
    for file in edf_files:
        fname = os.path.basename(file)
        tmp_ = fname.split('_')
        for researcher_id in researcher_ids:
            if researcher_id == tmp_[1]:
                for identifier in identifiers:
                    if identifier in os.path.basename(file):
                        grouped_files[researcher_id][identifier].append(file)
                        break  # Assuming each file matches only one identifier
                break  # Assuming each file matches only one researcher_id
    return grouped_files




def make_epochs(x, win_len_sec, win_step_sec, samp_rate, axis=-1):
    from numpy.lib.stride_tricks import as_strided
    step = win_step_sec*samp_rate
    window = win_len_sec*samp_rate
    step = int(step)
    window = int(window)
    shape = list(x.shape)
    shape[axis] = np.floor(x.shape[axis] / step - window / step + 1).astype(int)
    shape.append(window)
    strides = list(x.strides)
    strides[axis] *= step
    strides.append(x.strides[axis])
    strided = as_strided(x, shape=shape, strides=strides)
    if strided.ndim > 2:
          strided = np.rollaxis(strided, -2, 0)
    #epochs = mne.EpochsArray(strided, info, baseline=(None, None))
    return strided

def find_bad_chans_by_line_noise(data, ch_names, samp_rate,ln_freq):
    thres=5
    chan_ln = _freqs_power(data, samp_rate, ln_freq)
    bad_chans_by_line_noise = _find_outliers(chan_ln, thres)
    return ch_names[bad_chans_by_line_noise].tolist()
    
def find_bad_chans_by_correlation(data,ch_names,sample_rate, n_samples, usable_idx, correlation_secs=2.0, correlation_threshold=0.3, frac_bad=0.01):
        """Detect channels that sometimes don't correlate with any other channels.

        Channel correlations are calculated by splitting the recording into
        non-overlapping windows of time (default: 1 second), getting the absolute
        correlations of each usable channel with every other usable channel for
        each window, and then finding the highest correlation each channel has
        with another channel for each window (by taking the 98th percentile of
        the absolute correlations).

        A correlation window is considered "bad" for a channel if its maximum
        correlation with another channel is below the provided correlation
        threshold (default: ``0.4``). A channel is considered "bad-by-correlation"
        if its fraction of bad correlation windows is above the bad fraction
        threshold (default: ``0.01``).

        This method also detects channels with intermittent dropouts (i.e.,
        regions of flat signal). A channel is considered "bad-by-dropout" if
        its fraction of correlation windows with a completely flat signal is
        above the bad fraction threshold (default: ``0.01``).

        Parameters
        ----------
        correlation_secs : float, optional
            The length (in seconds) of each correlation window. Defaults to ``1.0``.
        correlation_threshold : float, optional
            The lowest maximum inter-channel correlation for a channel to be
            considered "bad" within a given window. Defaults to ``0.4``.
        frac_bad : float, optional
            The minimum proportion of bad windows for a channel to be considered
            "bad-by-correlation" or "bad-by-dropout". Defaults to ``0.01`` (1% of
            all windows).

        """
        IQR_TO_SD = 0.7413  # Scales units of IQR to units of SD, assuming normality
        # Reference: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/IQR.html

        n_chans = len(ch_names)
        # Determine the number and size (in frames) of correlation windows
        win_size = int(correlation_secs * sample_rate)
        win_offsets = np.arange(1, (n_samples - win_size), win_size)
        win_count = len(win_offsets)

        # Initialize per-window arrays for each type of noise info calculated below
        max_correlations = np.ones((win_count, n_chans))
        dropout = np.zeros((win_count, n_chans), dtype=bool)
        noiselevels = np.zeros((win_count, n_chans))
        channel_amplitudes = np.zeros((win_count, n_chans))

        for w in range(win_count):
            # Get both filtered and unfiltered data for the current window
            start, end = (w * win_size, (w + 1) * win_size)
            win_data = data[:, start:end]
            

            # Get channel amplitude info for the window
            usable = usable_idx.copy()
          
            # Check for any channel dropouts (flat signal) within the window
            eeg_amplitude = compute_mad(win_data, axis=1)
            dropout[w, usable] = eeg_amplitude == 0

            # Exclude any dropout chans from further calculations (avoids div-by-zero)
            usable[usable] = eeg_amplitude > 0
            win_data = win_data[eeg_amplitude > 0, :]
            eeg_amplitude = eeg_amplitude[eeg_amplitude > 0]

            
            # Get inter-channel correlations for the window
            win_correlations = np.corrcoef(win_data)
            abs_corr = np.abs(win_correlations - np.diag(np.diag(win_correlations)))
            max_correlations[w, usable] = _mat_quantile(abs_corr, 0.98, axis=0)
            max_correlations[w, dropout[w, :]] = 0  # Set dropout correlations to 0

        # Flag channels with above-threshold fractions of bad correlation windows
        thresholded_correlations = max_correlations < correlation_threshold
        fraction_bad_corr_windows = np.mean(thresholded_correlations, axis=0)
        bad_correlation_mask = fraction_bad_corr_windows > frac_bad
        bad_correlation_channels = ch_names[bad_correlation_mask]

        # Flag channels with above-threshold fractions of drop-out windows
        fraction_dropout_windows = np.mean(dropout, axis=0)
        dropout_mask = fraction_dropout_windows > frac_bad
        dropout_channels = ch_names[dropout_mask]

        # Update names of low-correlation/dropout channels & save additional info
        bad_by_correlation = bad_correlation_channels.tolist()
        bad_by_dropout = dropout_channels.tolist()
        return bad_by_correlation, bad_by_dropout

def compute_mad(data, axis=1):
    med_ = np.median(data, axis=axis)
    med_ = med_.reshape(-1,1)
    
    #detect flat channels
    mad_ =np.median(np.abs(data-med_), axis=axis)
    return mad_
        

def find_bad_chans_by_nan_flat_channels(data, ch_names, flat_threshold=1e-15, axis=1):
    
    
    bad_by_nan = list()
    # Detect channels containing any NaN values
    nan_channel_mask = np.isnan(np.sum(data, axis=1))
    nan_channels = ch_names[nan_channel_mask]

   
    # Update names of bad channels by NaN or flat signal
    if nan_channels.size > 0 :
        bad_by_nan = nan_channels.tolist()
    
    
    
    bad_by_flat = list()
    
    #compute median absolute deviation
    med_ = np.median(data, axis=axis)
    med_ = med_.reshape(-1,1)
    
    #detect flat channels
    mad_ =np.median(np.abs(data-med_), axis=axis)
    flat_by_mad = mad_ < flat_threshold
    flat_by_stdev = np.std(data, axis=1) < flat_threshold
    flat_channel_mask = flat_by_mad | flat_by_stdev
    flat_channels = ch_names[flat_channel_mask]
    if flat_channels.size > 0:
        bad_by_flat = flat_channels.tolist()
    
    bad_by_nan_flat = bad_by_flat + bad_by_nan
    usable_idx = np.isin(ch_names, bad_by_nan_flat, invert=True)
    return bad_by_nan_flat, usable_idx



def _mat_quantile(arr, q, axis=None):
    """Calculate the numeric value at quantile (`q`) for a given distribution.

    Parameters
    ----------
    arr : np.ndarray
        Input array containing samples from the distribution to summarize. Must
        be either a 1-D or 2-D array.
    q : float
        The quantile to calculate for the input data. Must be between 0 and 1,
        inclusive.
    axis : {int, tuple of int, None}, optional
        Axis along which quantile values should be calculated. Defaults to
        calculating the value at the given quantile for the entire array.

    Returns
    -------
    quantile : scalar or np.ndarray
        If no axis is specified, returns the value at quantile (q) for the full
        input array as a single numeric value. Otherwise, returns an
        ``np.ndarray`` containing the values at quantile (q) for each row along
        the specified axis.

    Notes
    -----
    MATLAB calculates quantiles using different logic than Numpy: Numpy treats
    the provided values as a whole population, whereas MATLAB treats them as a
    sample from a population of unknown size and adjusts quantiles accordingly.
    This function mimics MATLAB's logic to produce identical results.

    """
    # Sort the array in ascending order along the given axis (any NaNs go to the end)
    # Return NaN if array is empty.
    if len(arr) == 0:
        return np.NaN
    arr_sorted = np.sort(arr, axis=axis)

    # Ensure array is either 1D or 2D
    if arr_sorted.ndim > 2:
        e = "Only 1D and 2D arrays are supported (input has {0} dimensions)"
        raise ValueError(e.format(arr_sorted.ndim))

    # Reshape data into a 2D array with the shape (num_axes, data_per_axis)
    if axis is None:
        arr_sorted = arr_sorted.reshape(-1, 1)
    else:
        arr_sorted = np.moveaxis(arr_sorted, axis, 0)

    # Initialize quantile array with values for non-usable (n < 2) axes.
    # Sets quantile to only non-NaN value if n == 1, or NaN if n == 0
    quantiles = arr_sorted[0, :]

    # Get counts of non-NaN values for each axis and determine which have n > 1
    n = np.sum(np.isfinite(arr_sorted), axis=0)
    n_usable = n[n > 1]

    if np.any(n > 1):
        # Calculate MATLAB-style sample-adjusted quantile values
        q = np.asarray(q, dtype=np.float64)
        q_adj = ((q - 0.5) * n_usable / (n_usable - 1)) + 0.5

        # Get the exact (float) index position of the quantile for each usable axis, as
        # well as the indices of the values below and above it (if not a whole number)
        exact_idx = (n_usable - 1) * np.clip(q_adj, 0, 1)
        pre_idx = np.floor(exact_idx).astype(np.int32)
        post_idx = np.ceil(exact_idx).astype(np.int32)

        # Interpolate exact quantile values for each usable axis
        axis_idx = np.arange(len(n))[n > 1]
        pre = arr_sorted[pre_idx, axis_idx]
        post = arr_sorted[post_idx, axis_idx]
        quantiles[n > 1] = pre + (post - pre) * (exact_idx - pre_idx)

    return quantiles[0] if quantiles.size == 1 else quantiles

def _mat_iqr(arr, axis=None):
    """Calculate the inter-quartile range (IQR) for a given distribution.

    Parameters
    ----------
    arr : np.ndarray
        Input array containing samples from the distribution to summarize.
    axis : {int, tuple of int, None}, optional
        Axis along which IQRs should be calculated. Defaults to calculating the
        IQR for the entire array.

    Returns
    -------
    iqr : scalar or np.ndarray
        If no axis is specified, returns the IQR for the full input array as a
        single numeric value. Otherwise, returns an ``np.ndarray`` containing
        the IQRs for each row along the specified axis.

    Notes
    -----
    See notes for :func:`utils._mat_quantile`.

    """
    return _mat_quantile(arr, 0.75, axis) - _mat_quantile(arr, 0.25, axis)

def find_bad_chans_by_deviation(data, ch_names, usable_idx, deviation_threshold=5.0):
        """Detect channels with abnormally high or low overall amplitudes.

        A channel is considered "bad-by-deviation" if its amplitude deviates
        considerably from the median channel amplitude, as calculated using a
        robust Z-scoring method and the given deviation threshold.

        Amplitude Z-scores are calculated using the formula
        ``(channel_amplitude - median_amplitude) / amplitude_sd``, where
        channel amplitudes are calculated using a robust outlier-resistant estimate
        of the signals' standard deviations (IQR scaled to units of SD), and the
        amplitude SD is the IQR-based SD of those amplitudes.

        Parameters
        ----------
        deviation_threshold : float, optional
            The minimum absolute z-score of a channel for it to be considered
            bad-by-deviation. Defaults to ``5.0``.

        """
        IQR_TO_SD = 0.7413  # Scales units of IQR to units of SD, assuming normality
        # Reference: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/IQR.html

        # Get channel amplitudes and the median / robust SD of those amplitudes
        chan_amplitudes = _mat_iqr(data, axis=1) * IQR_TO_SD
        amp_sd = _mat_iqr(chan_amplitudes) * IQR_TO_SD
        amp_median = np.nanmedian(chan_amplitudes)

        # Calculate robust Z-scores for the channel amplitudes
        amplitude_zscore = np.zeros(len(ch_names))
        amplitude_zscore[usable_idx] = (chan_amplitudes - amp_median) / amp_sd

        # Flag channels with amplitudes that deviate excessively from the median
        abnormal_amplitude = np.abs(amplitude_zscore) > deviation_threshold
        deviation_channel_mask = np.isnan(amplitude_zscore) | abnormal_amplitude

        # Update names of bad channels by excessive deviation & save additional info
        deviation_channels = ch_names[deviation_channel_mask]
        bad_by_deviation = deviation_channels.tolist()
    
        return bad_by_deviation

def find_bad_epochs_by_dev(data, win_len, win_overlap, samp_rate, thres=5):
    IQR_TO_SD = 0.7413
    win_data = make_epochs(data, win_len, win_overlap, samp_rate)
    num_epochs=win_data.shape[0]
    chan_epoch_rzscore = np.zeros((win_data.shape[1], win_data.shape[0]))
    for n_chan in range(win_data.shape[1]):
        epoch_amps = _mat_iqr(win_data[:,n_chan,:].squeeze(), axis=1)*IQR_TO_SD
        amp_sd = _mat_iqr(epoch_amps)*IQR_TO_SD
        amp_median = np.nanmedian(epoch_amps)
        chan_epoch_rzscore[n_chan,:] = (epoch_amps - amp_median) / amp_sd
    max_rzscore=np.max(np.abs(chan_epoch_rzscore), axis=0)
    abnormal_epochs = max_rzscore > 5
    return sum(abnormal_epochs), num_epochs



def plot_confidence_interval(x, values, z=1.96, color='#2187bb', horizontal_line_width=0.25):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color='#f44336')

    return mean, confidence_interval


def get_bad_chans_epochs(grouped_fnames, ln_freq, researcher_id,cond):

    fnames = grouped_fnames[researcher_id][cond]
    fnames = sorted(fnames)
    fnames_ok = list()
    perc_bad_chans=list()
    perc_bad_epochs_per_id = list()
    bad_chan_names = list()
    unusable_file_short_rec = list()
    unusable_file_all_flat = list()
    num_channels_by_id = list()
   # bad_by_nan_flat=list()
    for i in range(len(fnames)):
        file_object = open('prob_edf_files_%s_tanzania.txt'%cond, 'a')
        try:
            fname =  fnames[i] #'/Users/nbnapu/Documents/Projects/SL-EEG-feasibility/EEG_Files_Device1/LABE351_LABE_01_EO_EPOCFLEX_203786_2023.12.06T11.47.00+03.00.edf'
            raw = mne.io.read_raw_edf(fname, preload=(True)) #instance of the mne.io.Raw class
            fnames_ok.append(fname)
        except:
            file_object.write(os.path.basename(fname))
            file_object.write("\n")
            continue
        print(raw.info) # print info about the data e.g. channel names , sampling freq etc
        ch_names = raw.info['ch_names']
        print(ch_names)
        print(fname)

#PICK ONLY EEG CHANNELS
        pick_idx_1 = mne.pick_channels_regexp(ch_names,'INTERPOLATED')
        pick_idx_2 = mne.pick_channels_regexp(ch_names,'HighBitFlex')
        if len(pick_idx_2) == 0:
            pick_idx_2 = mne.pick_channels_regexp(ch_names,'RAW_CQ')
        eeg_chs_list = list(range(pick_idx_1[0]+1, pick_idx_2[0]))
        all_chans = range(0,len(ch_names))
        unwanted_chans = [x for x in all_chans if x not in eeg_chs_list]
        raw.drop_channels(ch_names=[ch_names[i] for i in unwanted_chans])
        raw.filter(l_freq=0.5, h_freq=None)
        ch_names = np.asarray(raw.info['ch_names'])
        samp_rate=raw.info['sfreq']
        #raw.info.set_montage('standard_1020')
        data = raw.get_data()
        n_samples=data.shape[1]
        if n_samples < 30*samp_rate:
            unusable_file_short_rec.append(fname)
        else:
            bad_chans_by_nan_flat, usable_idx = find_bad_chans_by_nan_flat_channels(data, ch_names, 1e-15, 1)
            if sum(usable_idx) > 0:
                bad_epochs_by_dev, num_epochs = find_bad_epochs_by_dev(data, 2, 2, samp_rate, thres=5)
                perc_bad_epochs_per_id.append(100*bad_epochs_by_dev / num_epochs)
                bad_chans_by_line_noise = find_bad_chans_by_line_noise(data, ch_names, samp_rate, ln_freq)
                bad_chans_by_dev = find_bad_chans_by_deviation(data, ch_names, usable_idx, deviation_threshold=5.0)
                bad_chans_by_correlation, bad_chans_by_dropout = find_bad_chans_by_correlation(data,ch_names,samp_rate, n_samples, usable_idx, correlation_secs=1.0, correlation_threshold=0.4, frac_bad=0.01)
                bad_chans_by_all = list(set(bad_chans_by_line_noise+bad_chans_by_nan_flat + bad_chans_by_dev + bad_chans_by_correlation + bad_chans_by_dropout))
                perc_bad_chans.append(100 * (len(bad_chans_by_all)/len(ch_names)))
                bad_chan_names.append(bad_chans_by_all)
                num_channels_by_id.append(len(ch_names))
            else:
                unusable_file_all_flat.append(fname)
        #bad_by_nan_flat.append(bads_)
       # total_t = data.shape[1]/raw.info['sfreq']
       # t,epochs = make_epochs(data, 2, 2, raw.info)
    perc_of_16ch = 100*num_channels_by_id.count(16)/ len(num_channels_by_id)   
    return perc_bad_epochs_per_id,perc_bad_chans, bad_chan_names, unusable_file_all_flat, unusable_file_short_rec, perc_of_16ch, fnames


def get_bad_chans_epochs_benchmark(eeg_files, ln_freq):

    fnames_ok = list()
    perc_bad_chans=list()
    perc_bad_epochs_per_id = list()
    bad_chan_names = list()
    unusable_file_short_rec = list()
    unusable_file_all_flat = list()
    num_channels_by_id = list()
   # bad_by_nan_flat=list()
    for i in range(len(eeg_files)):
        file_object = open('prob_benchmark_files_%d.txt'%i, 'a')
        try:
            fname =  eeg_files[i] #'/Users/nbnapu/Documents/Projects/SL-EEG-feasibility/EEG_Files_Device1/LABE351_LABE_01_EO_EPOCFLEX_203786_2023.12.06T11.47.00+03.00.edf'
            raw = mne.io.read_raw_eeglab(fname, preload=(True)) #instance of the mne.io.Raw class
            fnames_ok.append(fname)
        except:
            file_object.write(os.path.basename(fname))
            file_object.write("\n")
            continue
        print(raw.info) # print info about the data e.g. channel names , sampling freq etc
        ch_names = raw.info['ch_names']
        print(ch_names)
        print(fname)

#PICK ONLY EEG CHANNELS
        raw.filter(l_freq=0.5, h_freq=None)
        ch_names = np.asarray(raw.info['ch_names'])
        samp_rate=raw.info['sfreq']
        #raw.info.set_montage('standard_1020')
        data = raw.get_data()
        n_samples=data.shape[1]
        if n_samples < 30*samp_rate:
            unusable_file_short_rec.append(fname)
        else:
            bad_chans_by_nan_flat, usable_idx = find_bad_chans_by_nan_flat_channels(data, ch_names, 1e-15, 1)
            if sum(usable_idx) > 0:
                bad_epochs_by_dev, num_epochs = find_bad_epochs_by_dev(data, 2, 2, samp_rate, thres=5)
                perc_bad_epochs_per_id.append(100*bad_epochs_by_dev / num_epochs)
                bad_chans_by_line_noise = find_bad_chans_by_line_noise(data, ch_names, samp_rate, ln_freq)
                bad_chans_by_dev = find_bad_chans_by_deviation(data, ch_names, usable_idx, deviation_threshold=5.0)
                bad_chans_by_correlation, bad_chans_by_dropout = find_bad_chans_by_correlation(data,ch_names,samp_rate, n_samples, usable_idx, correlation_secs=1.0, correlation_threshold=0.4, frac_bad=0.01)
                bad_chans_by_all = list(set(bad_chans_by_line_noise+bad_chans_by_nan_flat + bad_chans_by_dev + bad_chans_by_correlation + bad_chans_by_dropout))
                perc_bad_chans.append(100 * (len(bad_chans_by_all)/len(ch_names)))
                bad_chan_names.append(bad_chans_by_all)
                num_channels_by_id.append(len(ch_names))
            else:
                unusable_file_all_flat.append(fname)  
    return perc_bad_epochs_per_id,perc_bad_chans, bad_chan_names, unusable_file_all_flat, unusable_file_short_rec

