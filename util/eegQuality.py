#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:09:44 2024

@author: nbnapu
"""

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
from datetime import timedelta
from mne.utils import logger
from scipy.stats import kurtosis
import scipy.signal


def list_edf_files(dir):
    edf_files = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.edf'):
                edf_files.append(os.path.join(root, file))
                
                
    return edf_files



def get_info_from_fname(fname):

    split_strings = fname.split('_')
    subject_id = split_strings[0]
    researcher_id = split_strings[1]
    device_id = split_strings[2]
    task = split_strings[3]
    device_name = split_strings[4]
    return subject_id, researcher_id, device_id, task, device_name


def group_files_by_researchers_and_conds(edf_files, researcher_ids, identifiers):
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
    win_data = as_strided(x, shape=shape, strides=strides)
    if win_data.ndim > 2:
          win_data = np.rollaxis(win_data, -2, 0)
          
          
    return win_data


def get_week_start_date(date):
    
    
     return date - timedelta(days=date.weekday())


def compute_weekly_throughput(cond, grouped_fnames):
    fnames = grouped_fnames[cond]
    fnames_ok = list()
    eeg_start_dt = list()
    researcher_id_list = list()
    for i in range(len(fnames)):
        file_object = open('prob_edf_files_%s_tanzania.txt'%cond, 'a')
        try:
            fname =  fnames[i] 
            raw = mne.io.read_raw_edf(fname)
            fnames_ok.append(fname)
            eeg_start_dt.append(raw.info['meas_date'])
            _,researcher_id,_,_,_ = get_info_from_fname(os.path.basename(fname)) 
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
        
        
    return weekly_recordings, weekly_researchers

def pick_eeg_data(raw):
    ch_names = raw.info['ch_names']
    pick_idx_1 = mne.pick_channels_regexp(ch_names,'INTERPOLATED')
    pick_idx_2 = mne.pick_channels_regexp(ch_names,'HighBitFlex')
    if len(pick_idx_2) == 0:
        pick_idx_2 = mne.pick_channels_regexp(ch_names,'RAW_CQ')
    eeg_chs_list = list(range(pick_idx_1[0]+1, pick_idx_2[0]))
    all_chans = range(0,len(ch_names))
    unwanted_chans = [x for x in all_chans if x not in eeg_chs_list]
    raw.drop_channels(ch_names=[ch_names[i] for i in unwanted_chans])

    return raw


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


def _freqs_power(data, sfreq, freqs):
    fs, ps = _efficient_welch(data, sfreq)
    try:
        return np.sum([ps[..., np.searchsorted(fs, f)] for f in freqs], axis=0)
    except IndexError:
        raise ValueError(
            ("Insufficient sample rate to  estimate power at {} Hz for line "
             "noise detection. Use the 'criteria' parameter to disable the "
             "'power_line_noise' metric.").format(freqs))

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

def _corr_faster(data):
    C = np.corrcoef(data)
    corr = (C.sum(1)-1)/(C.shape[1]-1) #exclude diagnols
    
    return corr

def hurst(x):
    """Estimate Hurst exponent on a timeseries.

    The estimation is based on the second order discrete derivative.

    Parameters
    ----------
    x : 1D numpy array
        The timeseries to estimate the Hurst exponent for.

    Returns
    -------
    h : float
        The estimation of the Hurst exponent for the given timeseries.
    """
    y = np.cumsum(np.diff(x, axis=1), axis=1)

    b1 = [1, -2, 1]
    b2 = [1, 0, -2, 0, 1]

    # second order derivative
    y1 = scipy.signal.lfilter(b1, 1, y, axis=1)
    y1 = y1[:, len(b1) - 1 : -1]  # first values contain filter artifacts

    # wider second order derivative
    y2 = scipy.signal.lfilter(b2, 1, y, axis=1)
    y2 = y2[:, len(b2) - 1 : -1]  # first values contain filter artifacts

    s1 = np.mean(y1**2, axis=1)
    s2 = np.mean(y2**2, axis=1)

    return 0.5 * np.log2(s2 / s1)


def compute_mad(data, axis=1):
    med_ = np.median(data, axis=axis)
    med_ = med_.reshape(-1,1)
    
    #detect flat channels
    mad_ =np.median(np.abs(data-med_), axis=axis)
    return mad_

def _nan(data, ch_names, axis=1):
    nan_channel_mask = np.isnan(np.sum(data, axis=axis))
    nan_channels = [ch for ch, mask in zip(ch_names, nan_channel_mask) if mask]
    
    return nan_channels
    
def _flat(data, ch_names, flat_threshold=1e-15, axis=1):
    _median = np.median(data, axis=axis)
    _median = _median.reshape(-1,1)
    
    
    MAD =np.median(np.abs(data-_median), axis=axis)
    flat_by_MAD = MAD < flat_threshold
    flat_by_SD = np.std(data, axis=axis) < flat_threshold
    flat_channel_mask = flat_by_MAD | flat_by_SD
    flat_channels = [ch for ch, mask in zip(ch_names, flat_channel_mask) if mask]
    
    return flat_channels

def _deviation(data, usable_idx, ch_names, dev_threshold, axis=1):
    IQR_TO_SD = 0.7413  # Scales units of IQR to units of SD, assuming normality
    # Reference: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/IQR.html

    # Get channel amplitudes and the median / robust SD of those amplitudes
    chan_amplitudes = _mat_iqr(data, axis=axis) * IQR_TO_SD
    amp_sd = _mat_iqr(chan_amplitudes) * IQR_TO_SD
    amp_median = np.nanmedian(chan_amplitudes)

    # Calculate robust Z-scores for the channel amplitudes
    amplitude_zscore = np.zeros(len(ch_names))
    amplitude_zscore[usable_idx] = (chan_amplitudes - amp_median) / amp_sd

    # Flag channels with amplitudes that deviate excessively from the median
    abnormal_amplitude = np.abs(amplitude_zscore) > dev_threshold
    dev_channel_mask = np.isnan(amplitude_zscore) | abnormal_amplitude

    # Update names of bad channels by excessive deviation & save additional info
    dev_channels = [ch for ch, mask in zip(ch_names, dev_channel_mask) if mask]
    
    return dev_channels



def _corr_dropout_prep(data,ch_names,sample_rate, n_samples, usable_idx, correlation_secs=2.0, correlation_threshold=0.3, frac_bad=0.01):
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
        bad_corr_channels = [ch for ch, mask in zip(ch_names, bad_correlation_mask) if mask]

        # Flag channels with above-threshold fractions of drop-out windows
        fraction_dropout_windows = np.mean(dropout, axis=0)
        dropout_mask = fraction_dropout_windows > frac_bad
        dropout_chans = [ch for ch, mask in zip(ch_names, dropout_mask) if mask]

        return list(set(bad_corr_channels + dropout_chans))


def _bad_epochs_by_dev_prep(epochs, dev_threshold):
    IQR_TO_SD = 0.7413
   
    chan_epoch_rzscore = np.zeros((epochs.shape[1], epochs.shape[0]))
    for n_chan in range(epochs.shape[1]):
        epoch_amps = _mat_iqr(epochs[:,n_chan,:].squeeze(), axis=1)*IQR_TO_SD
        amp_sd = _mat_iqr(epoch_amps)*IQR_TO_SD
        amp_median = np.nanmedian(epoch_amps)
        chan_epoch_rzscore[n_chan,:] = (epoch_amps - amp_median) / amp_sd
    max_rzscore=np.max(np.abs(chan_epoch_rzscore), axis=0)
    bad_epochs = max_rzscore > dev_threshold
    return bad_epochs

def FASTER_find_bad_chans(data, samp_freq, ch_names, thres=3, max_iter=1, use_criteria=None):
    criteria = {
    "variance": lambda x: np.var(x, axis=1),
    "kurtosis": lambda x: kurtosis(x, axis=1),
    "correlation": lambda x: _corr_faster(data),
    "power_line_noise" : lambda x: _freqs_power(x, samp_freq, [50, 60]),
    "Hurst" : lambda x: hurst(x),
        }
    if use_criteria is None:
        use_criteria = criteria.keys()
    
    bads = defaultdict(list)
    for criterion in use_criteria:
        scores = criteria[criterion](data)
        bad_channels = [
                ch_names[i] for i in _find_outliers(scores, thres, max_iter)
                ]
        logger.info("\tBad by %s: %s" % (criterion, bad_channels))
        bads[criterion].append(bad_channels)
    bads = dict((k, np.concatenate(v).tolist()) for k, v in bads.items())
    
    return bads

def FASTER_find_bad_epochs(data, thres=3, max_iter=1, use_criteria=None):
    ch_mean = np.mean(np.mean(data,axis=2), axis=0)
    
    criteria = {
    "amp_range": lambda x: np.mean(np.max(data, axis=2) - np.min(data, axis=2), axis=1),
    "variance": lambda x: np.mean(np.var(data, axis=2), axis=1),
    "chan_dev": lambda x: np.mean(np.mean(data, axis=2) - ch_mean, axis=1),
           }
    if use_criteria is None:
        use_criteria = criteria.keys()
    
    bads = defaultdict(list)
    for criterion in use_criteria:
        scores = criteria[criterion](data)
        bad_epochs = _find_outliers(scores, thres, max_iter)
        logger.info("\tBad by %s: %s" % (criterion, bad_epochs))
        bads[criterion].append(bad_epochs)
    bads = dict((k, np.concatenate(v).tolist()) for k, v in bads.items())
    
    return bads


def PREP_find_bad_epochs(epochs, dev_threshold, use_criteria=None):
    criteria = {
    "deviation": lambda x: _bad_epochs_by_dev_prep(epochs, dev_threshold),
           }
    if use_criteria is None:
        use_criteria = criteria.keys()
    
    bads = defaultdict(list)
    for criterion in use_criteria:
        bad_epochs = criteria[criterion](epochs)
        logger.info("\tBad by %s: %s" % (criterion, bad_epochs))
        bads[criterion].append(np.where(bad_epochs==True)[0].tolist())
    bads = dict((k, np.concatenate(v).tolist()) for k, v in bads.items())
    
    return bads

def PREP_find_flat_nan_chans(data, ch_names, use_criteria=None):
    bads = defaultdict(list)
    criteria = {
    "nan": lambda x: _nan(data, ch_names, axis=1),
    "flat": lambda x: _flat(data, ch_names, flat_threshold=1e-15, axis=1),
        }
    use_criteria = criteria.keys()
    for criterion in use_criteria:
        bad_channels = criteria[criterion](data)
        bads[criterion].append(bad_channels)
        logger.info("\tBad by %s: %s" % (criterion, bad_channels))
    bads = dict((k, np.concatenate(v).tolist()) for k, v in bads.items())
    
    return bads

def PREP_find_bad_chans(data, samp_freq, usable_idx, ch_names, dev_threshold, use_criteria=None):
    n_samples = data.shape[1]
    bads = defaultdict(list)
    criteria = {
    "deviation": lambda x: _deviation(data, usable_idx, ch_names, dev_threshold, axis=1),
    "correlation_and_dropout": lambda x: _corr_dropout_prep(data,ch_names,samp_freq, n_samples, usable_idx, correlation_secs=1.0, correlation_threshold=0.4, frac_bad=0.05),
   # "dropout": lambda x: _dropout(data, ch_names, flat_threshold=1e-15, axis=1)
        }
    use_criteria = criteria.keys()
    for criterion in use_criteria:
        bad_channels = criteria[criterion](data)
        bads[criterion].append(bad_channels)
        logger.info("\tBad by %s: %s" % (criterion, bad_channels))
    bads = dict((k, np.concatenate(v).tolist()) for k, v in bads.items())
    
    return bads


def detect_bad_chans(data, samp_freq, ch_names, method=None):
    flag=1
    if method=='FASTER':
        bad_chans = FASTER_find_bad_chans(data, samp_freq, ch_names, thres=3, max_iter=1, use_criteria=None)
        combine_bad_chans_by_criteria = list(set(v for val in bad_chans.values() if len(val) > 0 for v in val))
    if method=='PREP':
        dev_threshold = 5
        print(dev_threshold)
        flat_nan_chans = PREP_find_flat_nan_chans(data, ch_names, use_criteria=None)
        combine_flat_chans_by_criteria = list(set(v for val in flat_nan_chans.values() if len(val) > 0 for v in val))
        usable_idx = np.isin(ch_names, combine_flat_chans_by_criteria, invert=True)
        if sum(usable_idx) == 0:
            flag = 0
            bad_chans = []
            combine_bad_chans_by_criteria = []
        else:
            bad_chans = PREP_find_bad_chans(data, samp_freq, usable_idx, ch_names, dev_threshold, use_criteria=None)
            combine_bad_chans_by_criteria = list(set(v for val in bad_chans.values() if len(val) > 0 for v in val))
            bad_chans = {**bad_chans, **flat_nan_chans}
            combine_bad_chans_by_criteria = set(combine_bad_chans_by_criteria + combine_flat_chans_by_criteria)
    return bad_chans, combine_bad_chans_by_criteria, flag

def detect_bad_epochs(data, samp_freq, ch_names, method=None):
    if method=='FASTER':
        bad_epochs = FASTER_find_bad_epochs(data, thres=3, max_iter=1, use_criteria=None)
        combine_bad_epochs_by_criteria = list(set(v for val in bad_epochs.values() if len(val) > 0 for v in val))
    if method=='PREP':        
        dev_threshold = 5
        bad_epochs = PREP_find_bad_epochs(data, dev_threshold, use_criteria=None)
        combine_bad_epochs_by_criteria = list(set(v for val in bad_epochs.values() if len(val) > 0 for v in val))
    return bad_epochs, combine_bad_epochs_by_criteria        
    
        
