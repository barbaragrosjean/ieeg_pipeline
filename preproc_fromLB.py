import os.path as op
import os
import pickle
import json
import mne
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from src.config import PROJECT_PATH, OUT_PATH, FREQ_BAND_DICT, FREQS, EVENT_ID, BWIDTH, TASK
    
def get_bads(SBJ, bads_path, elec_path, segments_path=None):

    # Load bads channels
    bads_df = pd.read_csv(bads_path)
    bad_channels = list(bads_df['name'])

    # Add bad segments
    bad_segments = {'onsets': [], 'duration': [], 'description': []}
    if segments_path:
        segs_df = pd.read_csv(segments_path)
        bad_segments['onsets'] = list(segs_df['onset'])
        bad_segments['duration'] = list(segs_df['duration'])
        bad_segments['description'] = ['bad']*len(segs_df['onset']) #[so for so in str(list(bads_info['bad_segment_description'].iloc[subix])[0]).split(';') if so != 'nan']

    #Exclude out of brain, lesion and fetch white matter
    elecs = pd.read_csv(elec_path)
    bad_channels += [elecs.loc[e]['electrode'] for e in range(len(elecs['electrode'])) if elecs.loc[e]['label'] in ['OOB','LSN']]
    white_channels = [elecs.loc[e]['electrode'] for e in range(len(elecs['electrode'])) if elecs.loc[e]['label'] in ['WM','VENT']]
    white_channels = [wch for wch in white_channels if wch not in bad_channels]
    
    return bad_channels, white_channels, bad_segments

def LB_event_fun(events, event_id, sfreq):
    wanted_labels = ['learning_bach_10','learning_bach_20'] #,'learning_bach_48','learning_bach_49','learning_bach_50']
    wanted_codes = np.array([event_id[cc] for cc in wanted_labels])
    events2 = events[[e in wanted_codes for e in events[:,2]],:]
    cond_mappings =  {'learning_bach_10': ['old',1], 
                      'learning_bach_20': ['new',2]}
    for cix,ce in enumerate(events2[:,2]):
        wix = np.where(wanted_codes == ce)[0][0]
        events2[cix,2] = cond_mappings[wanted_labels[wix]][1]
    
    ### Classify correct vs incorrect vs null
    # First get key presses
    wanted_keys = ['learning_bach_48','learning_bach_49','learning_bach_50']
    key_codes = np.array([event_id[c] for c in wanted_keys if c in event_id.keys()])
    key_events = events[[e in key_codes for e in events[:,2]],:]

    # Get correct, incorrect or null
    for eix2 in range(len(events2)):
        # Select first key press between 0.1 and 4 seconds after each onset
        rix = np.where([x and y for x,y in zip((np.array(key_events[:,0])/sfreq - np.array(events2[eix2,0])/sfreq) > 0.1,
                                               (np.array(key_events[:,0])/sfreq - np.array(events2[eix2,0])/sfreq) < 5)])[0]
        
        # Calssify key press
        if len(rix) > 0:
            rix = rix[0]        
            if (key_events[rix,2] == event_id['learning_bach_49']) and (events2[eix2,2] == 2):
                events2[eix2,2] = events2[eix2,2] + 100
            elif (key_events[rix,2] == event_id['learning_bach_50']) and (events2[eix2,2] == 1):
                events2[eix2,2] = events2[eix2,2] + 100
            elif ('learning_bach_48' in event_id.keys()) and (key_events[rix,2] == event_id['learning_bach_48']):
                events2[eix2,2] = events2[eix2,2] + 200
                    
    # Update event ids
    event_id2 = {}
    acc_map = {'correct': 0, 'incorrect': 100, 'null': 200}
    for am in acc_map:
        for ei in cond_mappings:
            cm = cond_mappings[ei]
            event_id2[cm[0] + '/' + am] = cm[1] + acc_map[am]
            
    return events2, event_id2

def smooth_data(data, tstart, tstep, twin, Fs, taxis=2):
    
    # get data shape
    old_dims = data.shape
    
    # Arrange dimensions in standard form
    new_dimord = np.array([taxis] + [d for d in range(len(old_dims)) if d != taxis])
    old_dimord = np.argsort(new_dimord)
    data = np.transpose(data,new_dimord)
    new_dims = data.shape
    
    # Calculate old and new time vectors
    tend = tstart + new_dims[0]/Fs
    ctime = np.arange(tstart, tend + 1/Fs, 1/Fs)
    ntime = np.arange(tstart + twin/2, tend-twin/2 + 1/Fs, tstep)
    
    # Initialize output data
    new_data = np.ones((tuple([len(ntime)]) + new_dims[1:])) * np.nan
    
    # Loop over timesteps and smooth
    for ntix, nt in enumerate(ntime):
        lims = np.array([nt - twin / 2, nt + twin / 2]) # Current interval to average
        cix = [np.argmin(np.abs(l - ctime)) for l in lims] # Limit indices
        new_data[ntix] = np.mean(data[cix[0]:(cix[1]+1)],0) # Average interval and store
    
    # Reorder dimesions and return
    new_data = np.transpose(new_data, old_dimord)
    return new_data, ntime

def smooth(indata, tstep, twin):
    
    # Copy data to avoid rewriting
    sdata = indata.copy()
    
    # Identify time axis
    taxis = np.where(np.array(sdata.data.shape) == len(sdata.times))[0][0]
    
    # Define starting time
    tstart = sdata.times[0]
    
    # Get sampling frequency
    Fs = sdata.info['sfreq']
    
    # Smooth data
    cdata, times = smooth_data(sdata.data, tstart=tstart,tstep=tstep,twin=twin,Fs = Fs,taxis=taxis)
    
    # Update instance
    with sdata.info._unlock():
            sdata.info['sfreq'] = 1/tstep

    sdata.data = cdata
    sdata._set_times(np.array(times, dtype=float))
    sdata._raw_times = np.array(times, dtype=float)

    return sdata

def normalize_epochs(epochs):
    data_mean = np.mean(epochs.get_data(), axis=(0,2), keepdims=True)
    data_std = np.std(epochs.get_data(), axis=(0,2), keepdims=True)
    norm_data = (epochs.get_data() - data_mean)/data_std
    norm_data /= 1000
    epochs = mne.EpochsArray(norm_data, epochs.info, events=epochs.events, baseline=epochs.baseline,
                              event_id = epochs.event_id, tmin = epochs.tmin, on_missing='warn')
    return epochs        

def set_reference(raw0, bads=None, white=None, rename=True, summary=False):
    
    if bads is None:
        bads = []
    if white is None:
        white = []
    
    pattern = r'[0-9]'
    shafts = [re.sub(pattern, '', s) for s in raw0.ch_names]

    # Get pairs of adjacent contacts on same shaft
    pairs = [(ch1, ch2) for ch1, ch2, s1, s2 in zip(
        raw0.ch_names[:-1], raw0.ch_names[1:], shafts[:-1], shafts[1:]) if s1 == s2]

    anode = [p[0] for p in pairs]
    cathode = [p[1] for p in pairs]

    # Compute new coordinates as midpoint of each pair
    coords = {ch['ch_name']: ch['loc'][:3] for ch in raw0.info['chs']}
    new_coords = [(coords[a] + coords[c]) / 2 for a, c in pairs]

    # Bipolar reference
    raw_bip = mne.set_bipolar_reference(raw0, anode, cathode, drop_refs=True)

    # Rename channels meaningfully
    if rename:
        new_names = {f"{a}-{c}": f"{a}_{c}" for a, c in pairs}  # optional underscore
        raw_bip.rename_channels(new_names)

    # Build montage dictionary (channel_name → coordinates)
    montage_coords = {raw_bip.ch_names[i]: new_coords[i] for i in range(len(raw_bip.ch_names))}
    montage = mne.channels.make_dig_montage(ch_pos=montage_coords, coord_frame='mni_tal')
    raw_bip.set_montage(montage, on_missing='warn')

    # Restore annotations
    raw_bip.set_annotations(raw0.annotations.copy())

    # Mark bad/white channels
    for idx, name in enumerate(raw_bip.ch_names):
        a, c = pairs[idx]
        if a in bads or c in bads:
            if summary: print(f"Marking bad: {name}")
            raw_bip.info['bads'].append(name)
        elif a in white or c in white:
            if summary: print(f"Marking white: {name}")
            raw_bip.info['bads'].append(name)

    return raw_bip

def MM_compute_TFR(epochs, freqs, n_cycles, baseline, zscore=True, trial_baseline = True, picks='all',n_jobs=2, summary=False):
    if summary : print('###### Call tfr morlet')
    TFR = mne.time_frequency.tfr_morlet(epochs,freqs,n_cycles,return_itc=False,average=False,n_jobs = n_jobs,picks=picks)
    
    if summary : print('##### Log transforming')
    for r in np.arange(TFR.data.shape[0]):
        if summary : print('trial ', r+1)
        TFR.data[r] = np.log(TFR.data[r])

    if zscore:
        if summary : print('##### z-scoring to baseline')
        bix = [a and b  for a, b in zip(TFR.times >= baseline[0], TFR.times <= baseline[1])]
        bmean = np.nanmean(TFR.data[:,:,:,bix],axis=(0,3),keepdims=True) 
        bstd = np.nanstd(TFR.data[:,:,:,bix],axis=(0,3),keepdims=True)

        TFR.data -= bmean 
        TFR.data /= bstd
        
    if trial_baseline:
        if summary : print('#####subtracting baseline per trial')
        bix = [a and b  for a, b in zip(TFR.times >= baseline[0], TFR.times <= baseline[1])]
        bmean = np.nanmean(TFR.data[:,:,:,bix],axis=(3),keepdims=True)
        bstd = np.nanstd(TFR.data[:,:,:,bix],axis=(3),keepdims=True)
        TFR.data -= bmean 
    return TFR

def TFR_mean(TFR,freq_bands=FREQ_BAND_DICT, freqs=FREQS, event_id= EVENT_ID, trials=True, bwitdh=BWIDTH): 
    band_indices = {band: [freqs.index(f) for f in band_freqs if f in freqs] for band, band_freqs in freq_bands.items()}
    TFR_mean_band = np.zeros((TFR.data.shape[0], TFR.data.shape[1], len(freq_bands), TFR.data.shape[3]))
    for i, inde in enumerate(band_indices.values()):
        #TFR_mean_band[:, :, i, :] = np.mean(TFR.data[:, :, inde, :], axis=2) 
        weights = bwitdh[inde] #MODIF
        weights = weights / weights.sum()
        TFR_mean_band[:, :, i, :] = np.average(TFR.data[:, :, inde, :],axis=2,weights=weights)

    unique_ev = np.unique(TFR.events[:, 2])
    id_ev ={}

    for val in unique_ev :
        idx = np.where(TFR.events[:, 2] == val)[0]
        id_ev[val] = idx


    if trials : 
        return TFR_mean_band
    else :
        TFR_mean_band_trials = np.zeros((len(event_id.values()), TFR_mean_band.shape[1], TFR_mean_band.shape[2], TFR_mean_band.shape[3]))
        for ev_id, (_, idx) in enumerate(id_ev.items()):
            TFR_mean_band_trials[ev_id, :, :, :] = np.mean(TFR_mean_band[idx, :, :, :], axis=0)

        return TFR_mean_band, TFR_mean_band_trials
    
def preproc(subj, sfreq = 600,new_sfreq = 200, freqs = FREQS, bwidth = BWIDTH, event_id = EVENT_ID, project_path = PROJECT_PATH, out_path = OUT_PATH + '/Data', trials = True, save_epoch=True, compute_TFR=True) :  
    info = {}
    if not os.path.exists(out_path) : 
        os.makedirs(out_path)

    bids_root = op.join(project_path,'data/BIDS/')
    bids_path = BIDSPath(subject=subj,task=TASK,root=bids_root)
    elec_path = op.join(project_path, 'misc/electrodes/', subj + '_electrodes.csv')

    bads_path = op.join(project_path, 'misc/bad_channels/', subj + '_bad_channels.csv')
    bad_channels, white_channels, _ = get_bads(subj, bads_path, elec_path)

    raw = read_raw_bids(bids_path)  
    raw.pick_types(seeg=True, ecog = True)

    onsets = []
    decription = []

    for ano in raw.annotations : 
        if 'bach' in ano['description'] : 
            onsets.append(ano['onset'])
            decription.append(ano['description'])
    if min(onsets) -2 <0 : 
        tmin_  = min(onsets)
    else : 
        tmin_ = min(onsets) -2
    raw.crop(tmin = tmin_ , tmax = max(onsets) + 50)
    raw.load_data() 
    raw = set_reference(raw, bads = bad_channels, white = white_channels)
    raw = raw.drop_channels(raw.info['bads'],on_missing='warn')

    ch_coord_df = pd.DataFrame({ch['ch_name']: ch['loc'][:3] for ch in raw.info['chs']}).T
    ch_coord_df = ch_coord_df.rename(columns = {0 : 'x', 1 : 'y', 2:'z'})
    
    for c in ['x', 'y', 'z'] :
        ch_coord_df[c] = ch_coord_df[c].apply(lambda x : x * 1000) # change the coordinate unit

    ch_coord_df['subj'] = subj

    raw.resample(sfreq=sfreq)
    l_freq=0.1
    h_freq=None
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    nfreqs = np.arange(1,np.ceil(raw.info['lowpass']/raw.info['line_freq']))*raw.info['line_freq']


    if nfreqs.size>0:            
        raw.notch_filter(freqs=nfreqs)

    try : 
        events, event_id = mne.events_from_annotations(raw)      
        events, event_id = LB_event_fun(events, event_id, sfreq=sfreq)
    except Exception as e : 
        print(subj) 
        print(e)
        return subj
    
    epoching_kwargs = {'tmin':-1.5, 'tmax': 5, 'baseline': None, 'resample': 600, 'l_freq': None, 'h_freq': None,
                             'event_fun': LB_event_fun, 'event_fun_kwargs': {'sfreq': 600}}
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=epoching_kwargs['tmin'], tmax=epoching_kwargs['tmax'],
                            preload=True, baseline=epoching_kwargs['baseline'], on_missing='warn',
                            reject=None)
    epochs = normalize_epochs(epochs)

    del raw

    if compute_TFR :
        n_cycles = np.array(freqs) * 2 / np.array(bwidth) 
        TFR = MM_compute_TFR(epochs,np.array(freqs), n_cycles, baseline = (-1.5,4), zscore=True, trial_baseline = False, picks='all',n_jobs=1, summary = False)
        TFR = TFR.crop(-1,4)

        smooth_kwargs = {'tstep': 0.025, 'twin': .1}
        TFR = smooth(TFR, **smooth_kwargs)

        if trials :
            TFRtrials = TFR_mean(TFR=TFR, trials=trials)
        else : 
            TFRtrials, TFRm = TFR_mean(TFR=TFR) 
            with open(out_path + f'/{subj}_TFRm.p', "wb") as f:
                pickle.dump(TFRm, f)
            del TFRm

        with open(out_path + f'/{subj}_TFRtrials.p', "wb") as f:
            pickle.dump(TFRtrials, f)

        info['time_tfr'] = list(TFR.times)
        del TFRtrials
        del TFR

    epochs = epochs.crop(-0.5,4) 

    if new_sfreq != 'none' :
        epochs = epochs.resample(new_sfreq)

    # extracted and save info 
    info['chnames'] = epochs.info['ch_names']
    info['time_epoch'] = list(epochs.times)
    info['event_count'] = [[list(event_id.keys())[list(event_id.values()).index(event)], str(count)] for event, count in zip(np.unique(epochs.events[: , 2], return_counts=True)[0], np.unique(epochs.events[: , 2], return_counts=True)[1])]
    info['event_id'] = [str(e) for e in epochs.events[:, 2]]
    
    with open(out_path + f'/{subj}_info.json', "w") as f:
        json.dump(info, f)

    ch_coord_df.to_csv(out_path + f'/{subj}_coords.csv' )

    if save_epoch : 
        with open(out_path + f'/{subj}_epochs.p', "wb") as f:
            pickle.dump(epochs.get_data(), f)
    else :
        return epochs.get_data()

def TFRmEvents(subj, event_ids = [1, 2], test_id=False, freq_id=False, events_index=[],  baseline_corr=False, data_path = OUT_PATH + '/Data') : 
    
    info_file = f'{data_path}/{subj}_info.json'
    with open(info_file) as f:
        info = json.load(f)
        time = info['time_tfr']
        if len(events_index) == 0 : 
            events_index = np.array([int(i) for i in info['event_id']])

    with open(f'{data_path}/{subj}_TFRtrials.p', "rb") as f:
        TFRtrials = pickle.load(f)

    if freq_id == False :
        TFRm = np.zeros((len(event_ids), TFRtrials.shape[1], TFRtrials.shape[2], TFRtrials.shape[3]))
    else :
        TFRm = np.zeros((len(event_ids), TFRtrials.shape[1], TFRtrials.shape[3]))

    for i, ev_id in enumerate(event_ids) :
        index_condi = list(np.where(events_index == ev_id)[0])

        if baseline_corr == 'mean' : 
            baseline_end = int(time.index([t for t in time if t>=-0.5][0]))
            TFRbmean = TFRtrials[np.array(index_condi), :, :, :baseline_end].mean(axis=-1, keepdims=True)
            TFRbstd = TFRtrials[np.array(index_condi), :, :, :baseline_end].std(axis=-1, keepdims=True)
            tfr = (TFRtrials[np.array(index_condi), :, :, :] - TFRbmean) 
        
        elif baseline_corr == 'z_score' :
            baseline_end = int(time.index([t for t in time if t>=-0.5][0]))
            TFRbmean = TFRtrials[np.array(index_condi), :, :, :baseline_end].mean(axis=-1, keepdims=True)
            TFRbstd = TFRtrials[np.array(index_condi), :, :, :baseline_end].std(axis=-1, keepdims=True)
            tfr = (TFRtrials[np.array(index_condi), :, :, :] - TFRbmean)/TFRbstd

        else :
            tfr = TFRtrials[np.array(index_condi), :, :, :]

        if test_id != False : # remove the excluded trials that we keep for testing
            index_condi.remove(test_id[i])
        
        if freq_id == False:
            TFRm[i, :,:, :] = tfr.mean(0) 

        else : 
            TFRm[i, :, :] = tfr[:, :,freq_id,:].mean(0)

    return TFRm
    
def BbEvents(subj, event_ids = [1, 2], test_id=False, events_index=[], data_path=OUT_PATH + '/Data') : 
    
    if len(events_index) == 0 : 
        info_file = f'{data_path}/{subj}_info.json'
        with open(info_file) as f:
            info = json.load(f)
            events_index = np.array([int(i) for i in info['event_id']])

    with open(f'{data_path}/{subj}_epochs.p', "rb") as f:
        epochs = pickle.load(f)

    epochsm = np.zeros((len(event_ids), epochs.shape[1], epochs.shape[2]))
    
    for i, ev_id in enumerate(event_ids) :
        index_condi1 = list(np.where(events_index == ev_id)[0])
        if test_id != False : # remove the excluded trials that we keep for testing
            index_condi1.remove(test_id[i])

        epochsm[i, :, :] = epochs[np.array(index_condi1), :,:].mean(0)
    return epochsm
