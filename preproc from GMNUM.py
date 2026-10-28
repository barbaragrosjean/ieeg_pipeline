"""Macro preprocessing, interactive gross-channel review, and output saving.

Run: python -m src.preproc num_new/SESSION_FOLDER
Inspired by MainPipeline.m through its first 'Save output' section.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import signal

try:
    from . import config
    from .setting import Session, load_session
    from .visualization import plot_channels as _plot_review, load_review, save_review
except ImportError:
    import config
    from setting import Session, load_session
    from visualization import plot_channels as _plot_review, load_review, save_review


def preprocess(session):
    """Detrend, demean, and zero-phase filter each continuous segment in place.

    Fourth-order Butterworth designs match the MATLAB settings. Second-order
    sections improve numerical stability at the very low high-pass cutoff.
    Forward/backward filtering doubles the effective order. Edge behavior is
    SciPy's default padding, not an exact reproduction of MATLAB filtfilt.
    """
    nyquist = session.sfreq / 2
    lower = config.NOTCH_HZ - config.NOTCH_WIDTH_HZ / 2
    upper = config.NOTCH_HZ + config.NOTCH_WIDTH_HZ / 2
    if not 0 < config.HIGH_PASS_HZ < nyquist or not 0 < lower < upper < nyquist:
        raise ValueError('Filter frequencies must lie strictly below Nyquist')
    highpass = signal.butter(config.FILTER_ORDER, config.HIGH_PASS_HZ,
                             btype='highpass', fs=session.sfreq, output='sos')
    notch = signal.butter(config.FILTER_ORDER, [lower, upper],
                          btype='bandstop', fs=session.sfreq, output='sos')
    # Validate all segments before modifying any data.
    for index, data in enumerate(session.data):
        if data.shape[1] < 2 or not np.isfinite(data).all():
            raise ValueError(f'Segment {index}: insufficient or nonfinite data')
        try:
            signal.sosfiltfilt(notch, np.zeros(data.shape[1]))
            signal.sosfiltfilt(highpass, np.zeros(data.shape[1]))
        except ValueError as exc:
            raise ValueError(f'Segment {index} is too short for zero-phase filtering') from exc
    for data in session.data:
        for channel in range(len(session.channel_names)):
            x = signal.detrend(data[channel], type='linear')
            x -= x.mean()
            x = signal.sosfiltfilt(highpass, x)
            data[channel] = signal.sosfiltfilt(notch, x)
    return session


def variance_outliers(data, threshold=config.VARIANCE_Z_THRESHOLD):
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


def gross_channel_removal(session):
    """Review variance proposals individually; mark bad channels without deletion.

    The historical function name is retained, but all signal rows are kept.
    Confirmed bad-channel names are saved as metadata for later analysis.
    """
    review_path = config.OUTPUT_ROOT / session.path.name / 'reviews' / 'gross.json'
    session.bad_channels = load_review(session, review_path)['bad_channels']
    kept = np.arange(len(session.channel_names))
    variance, z, mask = variance_outliers(session.data)
    proposed = kept[mask]
    print('Proposed channels:', ', '.join(session.channel_names[i] for i in proposed) or '(none)')
    decisions = []
    history = []
    if len(proposed):
        reviewed, decisions = _plot_review(session, kept, proposed, 0, 0,
                                          confirm_proposals=True)
        session.bad_channels = list(reviewed.info['bads'])
        history.append({'stage': 'variance', 'channels': list(session.channel_names),
                        'variance_uv2': variance.tolist(), 'z': z.tolist(),
                        'decisions': decisions, 'browser_bad_channels': list(session.bad_channels),
                        'removed': []})
    if not len(proposed):
        history.append({'stage': 'variance', 'channels': list(session.channel_names),
                        'decisions': [], 'removed': []})
    history.append(check_flat_channels(session, review_path))
    return kept, history


def check_flat_channels(session, review_path):
    """Finally review channels exactly zero at every sample in every segment.

    This criterion applies to the processed arrays, with no tolerance or
    variance threshold. Confirmed exclusions are bad flags, not deleted rows.
    """
    zero = np.ones(len(session.channel_names), dtype=bool)
    for data in session.data:
        if data.shape[1] == 0:
            raise ValueError('Cannot check flatness on an empty segment')
        for channel in np.flatnonzero(zero):
            zero[channel] = np.all(data[channel] == 0)
    proposed = np.flatnonzero(zero)
    names = [session.channel_names[i] for i in proposed]
    print('Final exact-zero channel check:', ', '.join(names) or 'none found.', flush=True)
    review = load_review(session, review_path)
    review['bad_channels'] = list(session.bad_channels)
    check = {'checked': True, 'completed': not bool(len(proposed)),
             'criterion': 'Every processed sample equals zero across all continuous segments',
             'channels_checked': list(session.channel_names),
             'segment_samples': [x.shape[1] for x in session.data],
             'proposed_channels': names, 'decisions': []}
    review['flat_channel_check'] = check
    save_review(review, review_path)
    if len(proposed):
        reviewed, decisions = _plot_review(
            session, np.arange(len(session.channel_names)), proposed, 0, 0,
            title='Exactly zero channels: exclusion review',
            review_path=review_path, confirm_proposals=True)
        session.bad_channels = list(reviewed.info['bads'])
        # Reload to preserve per-channel decisions and interval annotations saved by the viewer.
        review = load_review(session, review_path)
        check['decisions'] = decisions
        check['completed'] = True
        review['flat_channel_check'] = check
        review['bad_channels'] = list(session.bad_channels)
        save_review(review, review_path)
    print('Flat-channel check complete. Gross review finished.', flush=True)
    return {'stage': 'exact_zero', **check,
            'browser_bad_channels': list(session.bad_channels), 'removed': []}


def write_channel_decisions(session, history, path):
    """One channel per row; an exclusion in any check remains an exclusion."""
    columns = ['variance', 'exact_zero', 'manual']
    rows = {name: dict.fromkeys(columns, 'not_checked') for name in session.channel_names}
    for entry in history:
        stage = entry.get('stage')
        if stage not in ('variance', 'exact_zero'):
            continue
        checked = entry.get('channels', entry.get('channels_checked', []))
        for name in checked:
            rows[name][stage] = 'not_flagged'
        for decision in entry.get('decisions', []):
            rows[decision['channel']][stage] = 'exclude' if decision['marked_bad'] else 'keep'
    for name, row in rows.items():
        # Preserve additional manual flags without inventing a test attribution.
        row['manual'] = ('exclude' if name in session.bad_channels
                         and 'exclude' not in row.values() else 'not_flagged')
    session.bad_channels = [name for name, row in rows.items() if 'exclude' in row.values()]
    with Path(path).open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['channel', *columns])
        writer.writeheader()
        writer.writerows({'channel': name, **row} for name, row in rows.items())


def save_output(session, kept, history, output_dir):
    """Save each segment, excluded signals, events, and a readable audit record.

    .npy arrays can be reopened with np.load(..., mmap_mode='r'). A separate
    file per segment avoids padding gaps and copying all segments at once.
    Existing output directories are never overwritten.
    """
    if not np.array_equal(kept, np.arange(len(session.channel_names))):
        raise ValueError('Preprocessing output must retain every channel in original order')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_channel_decisions(session, history, output_dir / 'channel_decisions.csv')
    removed = np.setdiff1d(np.arange(len(session.channel_names)), kept)
    for i, data in enumerate(session.data):
        np.save(output_dir / f'segment_{i:03d}.npy', data[kept], allow_pickle=False)
        np.save(output_dir / f'removed_{i:03d}.npy', data[removed], allow_pickle=False)
    np.save(output_dir / 'events.npy', session.events, allow_pickle=False)
    metadata = {
        'session': str(session.path), 'units': 'uV', 'sfreq': session.sfreq,
        'native_sfreq': session.native_sfreq, 'sample_index_base': 0,
        'channel_names': [session.channel_names[i] for i in kept],
        'removed_channel_names': [session.channel_names[i] for i in removed],
        'bad_channel_names': list(session.bad_channels),
        'segment_start_us': session.segment_start_us.tolist(),
        'segment_samples': [x.shape[1] for x in session.data],
        'source_files': session.source_files, 'source_headers': session.headers,
        'channel_names_text': session.channel_names_text,
        'parameters': {name: value for name, value in vars(config).items()
                       if name.isupper() and isinstance(value, (int, float, str))},
        'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        'pipeline_version': 1, 'review_completed': True, 'review_history': history,
        'event_mapping': 'Nearest zero-based sample within segment; -1 outside acquisition. '
                         'Original timestamps and TTL values retained; task codes not decoded.',
    }
    # Written last: its presence marks a completed save.
    (output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    return output_dir


def run_session(session_path, output_dir=None):
    session_path = Path(session_path).expanduser().resolve()
    destination = Path(output_dir) if output_dir else config.OUTPUT_ROOT / session_path.name / 'macro_preprocessed'
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    print(f'Loading macro recordings: {session_path.name}', flush=True)
    session = load_session(session_path)
    print(f'{len(session.channel_names)} channels, {len(session.data)} continuous segments; '
          f'{session.native_sfreq:g} → {session.sfreq:g} Hz. Filtering.', flush=True)
    preprocess(session)
    kept, history = gross_channel_removal(session)
    return save_output(session, kept, history, destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path, help='One session directory containing macro/')
    parser.add_argument('--output-dir', type=Path, help='New output directory (must not exist)')
    args = parser.parse_args()
    print(f'Saved: {run_session(args.session, args.output_dir)}')
