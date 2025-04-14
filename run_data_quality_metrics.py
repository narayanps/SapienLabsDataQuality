#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:00:25 2024

@author: nbnapu
"""
from util import eegQuality as qc
import os
import mne

from collections import defaultdict


# Example usage
method='PREP'
country='TANZANIA'
dir_path = f'/Projects/SL-EEG-feasibility/codes/eneuro_codes/{country}/'  
save_path = f'/Projects/SL-EEG-feasibility/codes/eneuro_codes/' 
edf_files = qc.list_edf_files(dir_path)
researcher_ids = list()
for j in range(len(edf_files)):
    _,researcher_id,_,_,_ = qc.get_info_from_fname(os.path.basename(edf_files[j]))
    researcher_ids.append(researcher_id)
researcher_ids = list(set(researcher_ids))
#researcher_ids.remove('ARPA')
#researcher_ids.remove('ARPA01')
#researcher_ids.remove('IJU')

conds = ['EC', 'EO','PC']
grouped_files = qc.group_files_by_researchers_and_conds(edf_files, researcher_ids, conds)
win_len_sec=2
win_step_sec =2

rows_bad_chans=[]
rows_bad_epochs=[]
min_length=30
for r_id in range(len(researcher_ids)):
    for cond in range(len(conds)):
        file_names = grouped_files[researcher_ids[r_id]][conds[cond]]
        file_names = sorted(file_names)
        bad_chans_all_sub = defaultdict(list)
        PREP_bad_chans_all_sub = []
        bad_epochs_all_sub = defaultdict(list)
        perc_of_bad_chans = defaultdict(list)
        PREP_perc_of_bad_chans = []
        perc_of_bad_epochs = defaultdict(list)
        num_chans_sub=list()
        ch_names_sub=list()
        fname_processed=list()
        for ii in range(len(file_names)):
            file_object = open('prob_files_%s.txt'%conds[1], 'a')
            try:
                fname =  file_names[ii]
                raw = mne.io.read_raw_edf(fname, preload=(True)) #instance of the mne.io.Raw class
            except:
                file_object.write(os.path.basename(fname))
                file_object.write("\n")
                continue
            print(raw.info) # print info about the data e.g. channel names , sampling freq etc
    
            fname_processed.append(os.path.basename(fname))
            raw = qc.pick_eeg_data(raw)
            raw.filter(l_freq=0.5, h_freq=None)
            ch_names = raw.info['ch_names']
            samp_freq = raw.info['sfreq']
            data = raw.get_data()
            if data.shape[1] > int(min_length*samp_freq):
                num_chans = data.shape[0]
                bad_chans, combine_bad_chans_by_criteria, use_flag = qc.detect_bad_chans(data, samp_freq, ch_names, method=method)
                if use_flag == 1:
                    epochs = qc.make_epochs(data, win_len_sec, win_step_sec, samp_freq)
                    num_epochs  =epochs.shape[0]
                    bad_epochs, combine_bad_epochs_by_criteria = qc.detect_bad_epochs(epochs, samp_freq, ch_names, method=method)
                    perc_of_bad_chans[method].append(100*len(combine_bad_chans_by_criteria)/num_chans)
                    perc_of_bad_epochs[method].append(100*len(combine_bad_epochs_by_criteria)/num_epochs)
                    bad_chans_all_sub[method].append(bad_chans)
                    bad_epochs_all_sub[method].append(bad_epochs)
                    num_chans_sub.append(num_chans)
                    ch_names_sub.append(ch_names)
        data_list_bad_chans = perc_of_bad_chans[method]
        for i in range(len(data_list_bad_chans)):
            bad_chan_details = list(bad_chans_all_sub[method][i].values()) 
            rows_bad_chans.append([researcher_ids[r_id], fname_processed[i], conds[cond], data_list_bad_chans[i], num_chans_sub[i], ch_names_sub[i], *bad_chan_details])
            
        data_list_bad_epochs = perc_of_bad_epochs[method]
        for i in range(len(data_list_bad_epochs)):
            bad_epoch_details = list(bad_epochs_all_sub[method][i].values()) 
            rows_bad_epochs.append([researcher_ids[r_id], fname_processed[i], conds[cond], data_list_bad_epochs[i], *bad_epoch_details])



#save to csv files

headers = ['Researcher ID', 'File name', 'Task', '% of bad chans', 'No. of channels', 'Channel List']
additional_headers = list(bad_chans_all_sub[method][0].keys())     
headers.extend([f"{item} ({method})" for item in additional_headers])      
csv_file = save_path+f'{country}_{method}_bad_chans.csv'
import csv
# Write the rows to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write the header row
    writer.writerows(rows_bad_chans)    # Write the data rows


headers = ['Researcher ID', 'File name', 'Task', '% of bad epochs']
additional_headers = list(bad_epochs_all_sub[method][0].keys())     
headers.extend([f"{item} ({method})" for item in additional_headers])    
csv_file = save_path+f'{country}_{method}_bad_epochs.csv'
import csv
# Write the rows to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write the header row
    writer.writerows(rows_bad_epochs)    # Write the data rows



print(f'Data written to {csv_file}')
    