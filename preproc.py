"""Preprocessing functions and an explicitly configured pipeline runner."""
from dataclasses import replace
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import csv
import re

import numpy as np
from setting import load_data, save_data, output_directory, attach_coordinates

def bipolar_reference(session, config):
    """Return a new bipolar Session plus its source-contact mapping.

    V_(A1-A2)(t) = V_A2(t) - V_A1(t), matching MATLAB diff(data, 1, 1).
    Pair only numeric neighbors within an identical letter prefix. For example,
    A1/A2 is valid; A2/A4 and A8/B1 are not. Input order is irrelevant.
    Configured non-SEEG/ECoG auxiliary channels are not paired.
    A bipolar pair is flagged bad if either original contact is flagged bad.
    """
    electrodes = {}
    for index, name in enumerate(session.channel_names):
        match = re.fullmatch(r'([A-Za-z]+)(\d+)', name)
        if (match is None or config.CHANNEL_TYPE_OVERRIDES.get(name, session.channel_types[index])
                not in ('seeg', 'ecog')):
            continue
        prefix, number = match.group(1), int(match.group(2))
        contacts = electrodes.setdefault(prefix, {})
        if number in contacts:
            raise ValueError(f'Ambiguous contact number for {name}; check duplicate labels')
        contacts[number] = index
    pairs = []
    for prefix in sorted(electrodes):
        contacts = electrodes[prefix]
        for number in sorted(contacts):
            if number + 1 in contacts:
                left, right = contacts[number], contacts[number + 1]
                pairs.append((left, right))
    if not pairs:
        raise ValueError('No consecutive intracranial contacts found for bipolar referencing')
    names = [f'{session.channel_names[left]}-{session.channel_names[right]}'
             for left, right in pairs]
    data = []
    for segment in session.data:
        bipolar = np.empty((len(pairs), segment.shape[1]), dtype=np.float64)
        for row, (left, right) in enumerate(pairs):
            np.subtract(segment[right], segment[left], out=bipolar[row])
        data.append(bipolar)
    bads = [name for name, (left, right) in zip(names, pairs)
            if session.channel_names[left] in session.bad_channels
            or session.channel_names[right] in session.bad_channels]
    mapping = [{'channel': name, 'negative_contact': session.channel_names[left],
                'positive_contact': session.channel_names[right]}
               for name, (left, right) in zip(names, pairs)]
    used = {i for pair in pairs for i in pair}
    unpaired = [name for i, name in enumerate(session.channel_names) if i not in used]
    positions = {name: ((np.asarray(session.channel_positions[session.channel_names[left]])
                         + np.asarray(session.channel_positions[session.channel_names[right]])) / 2).tolist()
                 for name, (left, right) in zip(names, pairs)
                 if session.channel_names[left] in session.channel_positions
                 and session.channel_names[right] in session.channel_positions}
    metadata = deepcopy(session.metadata)
    metadata.update(bipolar_mapping=mapping, unpaired_channels=unpaired,
                    reference='positive_contact minus negative_contact (A2 - A1)',
                    coordinate_method='midpoint of original contacts')
    return replace(session, channel_names=names, data=data, bad_channels=bads, metadata=metadata,
                   channel_types=[session.channel_types[left] for left, _ in pairs],
                   channel_positions=positions, decision_csv=None), mapping, unpaired

def variance_outliers(data, threshold):
    """Return variance, z scores, and proposal mask across continuous segments.

    v_i = sample variance of channel i over all acquired samples;
    d_i = |v_i - median(v)|; z_i = (d_i - mean(d)) / std(d, ddof=1).
    Propose |z_i| > threshold. This is MATLAB's sample-SD convention and is
    not a robust MAD score. Proposals do not establish an artifact etiology.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Variance threshold must be positive and finite')
    total = sum(x.shape[1] for x in data)
    if total < 2 or not data or data[0].shape[0] == 0:
        raise ValueError('At least one channel and two samples are required')
    mean = sum(x.mean(axis=1) * x.shape[1] for x in data) / total
    variance = sum(x.var(axis=1) * x.shape[1]
                   + x.shape[1] * (x.mean(axis=1) - mean)**2 for x in data) / (total - 1)
    distances = np.abs(variance - np.median(variance))
    sd = distances.std(ddof=1) if len(distances) > 1 else 0.0
    z = (distances - distances.mean()) / sd if sd > 0 else np.zeros_like(distances)
    return variance, z, np.abs(z) > threshold

def preprocess(session, config):
    """GMNUM segment-wise detrend, demean, Butterworth filtering, then resample."""
    from scipy import signal
    fs = float(session.sfreq)
    order, high = config.FILTER_ORDER, config.HIGH_PASS_HZ
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError('FILTER_ORDER must be a positive integer')
    if high is None or not np.isfinite(high) or not 0 < high < fs / 2:
        raise ValueError('HIGH_PASS_HZ must be positive and below input Nyquist')
    centers = config.NOTCH_HZ
    centers = [] if centers is None else ([centers] if np.isscalar(centers) else list(centers))
    width = config.NOTCH_WIDTH_HZ
    if centers and (width is None or not np.isfinite(width) or width <= 0):
        raise ValueError('Specify a positive NOTCH_WIDTH_HZ')
    filters = [signal.butter(order, high, btype='highpass', fs=fs, output='sos')]
    for center in centers:
        if not np.isfinite(center) or not 0 < center - width / 2 < center + width / 2 < fs / 2:
            raise ValueError(f'Notch band around {center} Hz is outside input Nyquist')
        filters.append(signal.butter(order, [center - width / 2, center + width / 2],
                                     btype='bandstop', fs=fs, output='sos'))
    target = fs if config.FS_out is None else float(config.FS_out)
    if not np.isfinite(target) or target <= 2 * high:
        raise ValueError('FS_out must be finite and above twice HIGH_PASS_HZ')
    ratio = (Fraction(str(target)) / Fraction(str(fs))).limit_denominator(10000)
    actual_rate = fs * ratio.numerator / ratio.denominator
    if max(ratio.numerator, ratio.denominator) > 10000 or not np.isclose(actual_rate, target, rtol=1e-10, atol=0):
        raise ValueError('Requested sampling rates need an impractically large resampling ratio')
    for index, data in enumerate(session.data):
        if data.shape[1] < 2 or not np.isfinite(data).all():
            raise ValueError(f'Invalid data in segment {index}')
        try:
            for sos in filters:
                signal.sosfiltfilt(sos, np.zeros(data.shape[1]))
        except ValueError as exc:
            raise ValueError(f'Segment {index} is too short for configured filters') from exc
    processed = []
    for data in session.data:
        x = signal.detrend(data, axis=-1, type='linear')
        x -= x.mean(axis=-1, keepdims=True)
        for sos in filters:
            x = signal.sosfiltfilt(sos, x, axis=-1)
        if ratio.numerator != ratio.denominator:
            x = signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=-1,
                                     window=('kaiser', 5.0), padtype='line')
        processed.append(np.asarray(x, dtype=np.float64))
    result = replace(session, data=processed, sfreq=actual_rate,
                     events=deepcopy(session.events), metadata=deepcopy(session.metadata))
    result.map_events()
    result.metadata['filter_parameters'] = {
        'detrend': 'linear', 'demean': True, 'design': 'Butterworth', 'design_order': order,
        'phase': 'forward-backward', 'padding': 'scipy default odd',
        'high_pass_hz': high, 'notch_hz': centers, 'notch_width_hz': width,
        'input_sfreq': fs, 'output_sfreq': actual_rate,
        'resampling': None if ratio.numerator == ratio.denominator else {
            'method': 'resample_poly', 'up': ratio.numerator, 'down': ratio.denominator,
            'window': ['kaiser', 5.0], 'padtype': 'line'},
    }
    return result


def read_exclusions(path, criteria=None):
    """Read name,criterion CSVs; also accept LB name-only or electrode,label CSVs."""
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        name_key = 'name' if 'name' in fields else 'electrode'
        if name_key not in fields:
            raise ValueError(f'{path}: expected name or electrode column')
        criterion_key = 'criterion' if 'criterion' in fields else ('label' if 'label' in fields else None)
        allowed = criteria
        if criterion_key == 'label' and allowed is None:
            allowed = ['OOB', 'LSN', 'WM', 'VENT']
        records = []
        for row in reader:
            name = row[name_key].strip()
            reason = row[criterion_key].strip() if criterion_key else 'listed_bad_channel'
            if not name or not reason:
                raise ValueError(f'{path}: channel name and criterion cannot be blank')
            if allowed is None or reason in allowed:
                records.append({'name': name, 'criterion': reason})
    return records


def _write_exclusions(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', 'criterion'])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _review_channel(session, index, criterion):
    """Plot one proposed channel; show at most 10,000 display samples per segment."""
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots()
    for onset, data in zip(session.segment_onsets, session.data):
        stride = max(1, int(np.ceil(data.shape[1] / 10000)))
        samples = np.arange(0, data.shape[1], stride)
        axis.plot(onset + samples / session.sfreq, data[index, samples], linewidth=.5)
    axis.set(xlabel='Time (s)', ylabel='Signal (µV)',
             title=f'{session.channel_names[index]}: {criterion} (display may be decimated)')
    plt.show(block=False)
    plt.pause(.1)
    try:
        while True:
            decision = input(f'Exclude {session.channel_names[index]} ({criterion})? [y/n]: ').strip().lower()
            if decision in ('y', 'n'):
                return decision == 'y'
    finally:
        plt.close(figure)


def gross_channel_removal(session, config):
    """Export confirmed GMNUM variance/exact-zero exclusions; never alter signals."""
    path = output_directory(session, config) / 'gross_exclusions.csv'
    if path.exists():
        raise FileExistsError(f'{path} exists; reuse it via BAD_CHANNEL_CSVS instead of repeating review')
    threshold = config.VARIANCE_Z_THRESHOLD
    if threshold is None:
        raise ValueError('Set VARIANCE_Z_THRESHOLD before gross-channel review')
    variance, z, mask = variance_outliers(session.data, threshold)
    flat = np.logical_and.reduce([np.all(data == 0, axis=1) for data in session.data])
    mode = getattr(config, 'GROSS_REVIEW_MODE', 'interactive')
    if mode not in ('interactive', 'automatic'):
        raise ValueError('GROSS_REVIEW_MODE must be interactive or automatic')
    rows, decisions = [], []
    for index in np.flatnonzero(mask | flat):
        reason = 'exact_zero' if flat[index] else 'variance_outlier'
        excluded = mode == 'automatic' or _review_channel(session, index, reason)
        decisions.append({'name': session.channel_names[index], 'criterion': reason,
                          'excluded': excluded, 'variance_uv2': float(variance[index]), 'z': float(z[index])})
        if excluded:
            rows.append({'name': session.channel_names[index], 'criterion': reason})
    _write_exclusions(path, rows)
    return {'csv': path, 'decisions': decisions, 'mode': mode, 'threshold': threshold}


def epilepsy_removal(session, config):
    """Export supplied epilepsy-channel decisions; this is not an IED detector."""
    source = getattr(config, 'EPILEPSY_CHANNEL_CSV', None)
    if source is None:
        raise ValueError('Set EPILEPSY_CHANNEL_CSV to reviewed channel decisions, or omit epilepsy_removal')
    rows = read_exclusions(Path(config.PROJECT_PATH) / source)
    known = set(session.channel_names)
    for pair in session.metadata.get('bipolar_mapping', []):
        known.update((pair['negative_contact'], pair['positive_contact']))
    unknown = {row['name'] for row in rows} - known
    if unknown:
        raise ValueError(f'Epilepsy CSV contains unknown channels: {sorted(unknown)}')
    path = _write_exclusions(output_directory(session, config) / 'epilepsy_exclusions.csv', rows)
    return {'csv': path, 'source': str(source), 'decisions': rows, 'method': 'provided_channel_decisions'}


def remove_bad(session, config, path_to_csv=None):
    """Apply CSV exclusions, including source contacts of an existing bipolar pair."""
    paths = ([path_to_csv] if path_to_csv is not None else
             list(getattr(config, 'BAD_CHANNEL_CSVS', []) or [])
             + list(session.metadata.get('exclusion_csvs', [])))
    rows = []
    for path in dict.fromkeys(map(str, paths)):
        resolved = Path(config.PROJECT_PATH) / path
        for row in read_exclusions(resolved, getattr(config, 'EXCLUSION_CRITERIA', None)):
            rows.append({**row, 'source': str(resolved)})
    mapping = {pair['channel']: (pair['negative_contact'], pair['positive_contact'])
               for pair in session.metadata.get('bipolar_mapping', [])}
    known = set(session.channel_names) | {name for pair in mapping.values() for name in pair}
    previously_removed = {row['name'] for row in session.metadata.get('excluded_channels', [])}
    unknown = {row['name'] for row in rows} - known - previously_removed
    if unknown:
        raise ValueError(f'Exclusion CSV contains unknown channels: {sorted(unknown)}')
    remove = {row['name'] for row in rows}
    indices = [i for i, name in enumerate(session.channel_names)
               if name not in remove and not (set(mapping.get(name, ())) & remove)]
    if not indices:
        raise ValueError('Exclusion CSVs would remove every channel')
    names = [session.channel_names[i] for i in indices]
    metadata = deepcopy(session.metadata)
    exclusions = metadata.setdefault('excluded_channels', [])
    for name in session.channel_names:
        if name not in names:
            reasons = [row for row in rows if row['name'] == name or row['name'] in mapping.get(name, ())]
            exclusions.append({'name': name, 'reasons': reasons})
    return replace(session, data=[data[indices].copy() for data in session.data], channel_names=names,
                   channel_types=[session.channel_types[i] for i in indices],
                   channel_positions={name: xyz for name, xyz in session.channel_positions.items() if name in names},
                   bad_channels=[name for name in session.bad_channels if name in names], metadata=metadata)


# Configuration alias retained for the existing project list.
bad_removal = remove_bad


def broadband(session, config):
    pass


def power(session, config):
    pass


def run_preprocess(config, session=None, *, save=True):
    """Execute the configured steps literally, then save one neural/info pair."""
    handlers = {'preprocess': preprocess, 'gross_channel_removal': gross_channel_removal,
                'bipolar_reference': bipolar_reference, 'epilepsy_removal': epilepsy_removal,
                'bad_removal': remove_bad, 'remove_bad': remove_bad}
    steps = list(config.preproc_pipeline)
    for step in steps:
        if step not in handlers:
            raise ValueError(f'Unknown or unfinished step {step!r}; broadband and power are not implemented')
    if session is None:
        session, config = load_data(config)
    session = attach_coordinates(session, config)
    session = replace(session, metadata=deepcopy(session.metadata))
    if save:
        destination = output_directory(session, config)
        if any((destination / name).exists() for name in ('neural.npz', 'info.json')):
            raise FileExistsError(f'Output already exists in {destination}')
    outputs = []
    for step in steps:
        input_rate = session.sfreq
        result = handlers[step](session, config)
        if step == 'bipolar_reference':
            session, mapping, unpaired = result
        elif step in ('gross_channel_removal', 'epilepsy_removal'):
            session.metadata.setdefault('exclusion_csvs', []).append(str(result['csv']))
            session.metadata.setdefault('review_decisions', {})[step] = result
        else:
            session = result
        session.metadata.setdefault('preprocessing_steps', []).append(step)
        session.metadata.setdefault('processing_history', []).append({
            'step': step, 'input_FS': input_rate, 'output_FS': session.sfreq,
            'parameters': deepcopy(session.metadata.get('filter_parameters')) if step == 'preprocess' else None})
        outputs.append((step, result))
    if save:
        save_data(session, config)
    return session, outputs
