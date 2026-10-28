"""Data-management entry points; each receives an explicit project configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields, field, MISSING
from pathlib import Path
import runpy
import warnings
import numpy as np
from typing import Any

config_path = Path(__file__).with_name('config.py')

# Voltage channels supported by the shared microvolt representation. Digital
# stimulus and other non-voltage channels require a separate representation.
VOLTAGE_TYPES = {'seeg', 'ecog', 'eeg', 'eog', 'ecg', 'emg', 'dbs', 'bio'}
HEADER_BYTES = 16384
NEV_DTYPE = np.dtype([('reserved', '<i2'), ('system_id', '<i2'), ('data_size', '<i2'),
                    ('timestamp', '<u8'), ('event_id', '<i2'), ('ttl', '<u2'),
                    ('crc', '<i2'), ('dummy1', '<i2'), ('dummy2', '<i2'),
                    ('extra', '<i4', (8,)), ('description', 'S128'),])


@dataclass
class ProjectConfig:
    PROJECT_PATH: Path | None
    OUT_PATH: Path | None
    data_format: str | None
    EVENT_ID: dict | None
    TASK: str | None
    events: list | None
    color_event: dict | None
    preproc_pipeline: list | None
    col_pc: dict | None
    cmap_pc0: Any
    cmap_pc1: Any
    cmap_pc2: Any
    cmap_pcs: list | None
    FS_in: float | None
    FS_out: float | None
    FREQS: Any | None
    BWIDTH: Any | None
    FREQ_BAND_DICT: dict | None
    FREQ_BAND: list | None
    REGION: dict | None
    COL_REG: dict | None
    CHANNEL_TYPE_OVERRIDES: dict | None
    HIGH_PASS_HZ: float | None
    NOTCH_HZ: list[float] | float | None
    NOTCH_WIDTH_HZ: float | None
    FILTER_ORDER: int | None
    VARIANCE_Z_THRESHOLD: float | None
    DATA_PATH: str | Path | None = None
    BIDS_ENTITIES: dict = field(default_factory=dict)
    NCS_FILES: list = field(default_factory=list)
    NCS_EVENT_FILES: list = field(default_factory=list)
    NCS_CHANNEL_MAP: dict = field(default_factory=dict)
    NCS_DEFAULT_CHANNEL_TYPE: str = 'seeg'
    PARTICIPANT_ID: str | None = None
    SESSION_ID: str | None = None
    BAD_CHANNEL_CSVS: list = field(default_factory=list)
    EXCLUSION_CRITERIA: list | None = None
    ELECTRODES_CSV: str | Path | None = None
    COORDINATE_SPACE: str | None = None
    COORDINATE_UNITS: str = 'm'
    GROSS_REVIEW_MODE: str = 'interactive'
    EPILEPSY_CHANNEL_CSV: str | Path | None = None

def load_config(path: str | Path = config_path) -> ProjectConfig:
    source = Path(path).expanduser().resolve(strict=True)
    namespace = runpy.run_path(str(source))
    required = [f.name for f in fields(ProjectConfig)
                if f.default is MISSING and f.default_factory is MISSING]
    missing = [name for name in required if name not in namespace]
    if missing:
        raise ValueError(f'Missing configuration settings: {missing}')
    values = {f.name: namespace[f.name] for f in fields(ProjectConfig) if f.name in namespace}
    for name, base in [('PROJECT_PATH', source.parent), ('OUT_PATH', None)]:
        value = values[name]
        value = Path(value).expanduser()
        base = base if base is not None else values['PROJECT_PATH']
        values[name] = (base / value).resolve()
    return ProjectConfig(**values)


@dataclass
class Event:
    """Onset/duration are seconds relative to the first acquired sample.

    segment/sample are zero-based, or -1 outside acquisition (including gaps).
    code is a task code only when supplied by metadata or EVENT_ID. Neuralynx
    event_id and TTL are retained separately and are not interpreted as tasks.
    """
    onset: float
    duration: float
    description: str
    code: int | None = None
    segment: int = -1
    sample: int = -1
    timestamp_us: int | None = None
    ttl: int | None = None
    acquisition_event_id: int | None = None
    source: str | None = None


@dataclass
class Session:
    path: Path
    data: list[np.ndarray]  # float64, channels x samples, microvolts
    sfreq: float
    channel_names: list[str]
    channel_types: list[str]
    segment_start_us: np.ndarray
    time_origin: str  # 'neuralynx_clock' or 'recording_start'
    segment_duration_s: list[float] | None = None
    native_sfreq: float | None = None
    units: str = 'uV'
    events: list[Event] = field(default_factory=list)
    bad_channels: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    headers: list[dict] = field(default_factory=list)
    decision_csv: Path | None = None
    channel_positions: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.path = Path(self.path)
        self.segment_start_us = np.asarray(self.segment_start_us, dtype=np.int64)
        if self.native_sfreq is None:
            self.native_sfreq = self.sfreq
        if self.segment_duration_s is None:
            self.segment_duration_s = [x.shape[1] / self.sfreq for x in self.data]

    @property
    def segment_onsets(self):
        """Seconds relative to the first acquired sample, preserving gaps."""
        return (self.segment_start_us - self.segment_start_us[0]) / 1e6

    @property
    def channel_names_text(self):
        return '\n'.join(self.channel_names)

    def map_events(self, event_id=None):
        """Map event onsets to nearest acquired sample without bridging gaps."""
        self.events.sort(key=lambda event: event.onset)
        event_id = event_id or {}
        for event in self.events:
            if event.description in event_id:
                event.code = event_id[event.description]
            event.segment = event.sample = -1
            for index, (start, data) in enumerate(zip(self.segment_onsets, self.data)):
                offset = event.onset - start
                if 0 <= offset < self.segment_duration_s[index]:
                    sample = int(np.floor(offset * self.sfreq + 0.5))
                    event.segment = index
                    event.sample = min(sample, data.shape[1] - 1)
                    break


def _data_path(config):
    path = getattr(config, 'DATA_PATH', None)
    path = (Path(config.PROJECT_PATH) / Path(path).expanduser()).resolve(strict=True)
    return path

def load_data(config):
    """Dispatch one recording based on config.data_format."""
    loaders = {'bids': load_bids, 'ncs': load_ncs}
    fmt = str(config.data_format).lower()
    if fmt not in loaders:
        raise ValueError('data_format must be "bids" or "ncs"')
    return loaders[fmt](config)

def _bids_segments(raw):
    """Split at explicit acquisition skips/boundaries; retain other bad data."""
    cuts = {0, raw.n_times}
    gaps = []
    for annotation in raw.annotations:
        description = annotation['description'].lower()
        onset = float(annotation['onset']) - raw.first_time
        left = int(np.clip(round(onset * raw.info['sfreq']), 0, raw.n_times))
        if description == 'bad_acq_skip':
            right = int(np.clip(round((onset + annotation['duration']) * raw.info['sfreq']),
                                0, raw.n_times))
            cuts.update((left, right))
            if right > left:
                gaps.append((left, right))
        elif description in ('bad boundary', 'edge boundary'):
            cuts.add(left)
    boundaries = sorted(cuts)
    return [(left, right) for left, right in zip(boundaries[:-1], boundaries[1:])
            if right > left and not any(a <= left and right <= b for a, b in gaps)]

def load_bids(config):
    """Read exactly one BIDS iEEG recording using MNE-BIDS.

    BIDS_ENTITIES selects subject/session/task/run/acquisition/etc. Ambiguous
    selections are rejected. Event codes and bads come from BIDS metadata.
    """
    from mne_bids import BIDSPath, read_raw_bids

    root = _data_path(config)
    entities = dict(getattr(config, 'BIDS_ENTITIES', {}) or {})
    if not entities.get('subject'):
        raise ValueError('BIDS_ENTITIES must specify a subject label (without sub-)')
    forbidden = {'root', 'datatype', 'suffix', 'check'} & entities.keys()
    if forbidden:
        raise ValueError(f'BIDS_ENTITIES cannot override {sorted(forbidden)}')
    if getattr(config, 'TASK', None) is not None:
        entities.setdefault('task', config.TASK)
    query = BIDSPath(root=root, datatype='ieeg', suffix='ieeg', **entities)
    matches = [p for p in query.match() if p.extension in
               ('.vhdr', '.edf', '.set', '.nwb', '.mefd', '.fif')]
    if len(matches) != 1:
        raise ValueError(f'BIDS_ENTITIES must select one recording; found {len(matches)}: '
                         + ', '.join(str(p.fpath) for p in matches))
    bids_path = matches[0]
    # MNE-BIDS 0.20 cannot return an event dictionary when events.tsv is absent.
    events_path = bids_path.find_matching_sidecar(suffix='events', extension='.tsv', on_error='ignore')
    if events_path is None:
        raw = read_raw_bids(bids_path, verbose=False)
        event_ids = {}
    else:
        raw, event_ids = read_raw_bids(bids_path, return_event_dict=True, verbose=False)
    try:
        overrides = getattr(config, 'CHANNEL_TYPE_OVERRIDES', {}) or {}
        if overrides:
            raw.set_channel_types(overrides)
        types = raw.get_channel_types()
        picks = [i for i, kind in enumerate(types) if kind in VOLTAGE_TYPES]
        if not picks:
            raise ValueError('No voltage channels found in the BIDS recording')
        names = [raw.ch_names[i] for i in picks]
        excluded = [raw.ch_names[i] for i in range(len(types)) if i not in picks]
        if excluded:
            warnings.warn(f'Non-voltage channels excluded from Session.data: {excluded}', stacklevel=2)
        ranges = _bids_segments(raw)
        sfreq = float(raw.info['sfreq'])
        events = [Event(onset=float(a['onset']) - raw.first_time,
                        duration=float(a['duration']), description=str(a['description']),
                        code=event_ids.get(str(a['description'])), source=str(bids_path.fpath))
                  for a in raw.annotations]
        montage = raw.get_montage()
        positions = montage.get_positions() if montage is not None else None
        coordinates = {} if positions is None else {
            name: np.asarray(position).tolist() for name, position in positions['ch_pos'].items()
            if name in names}
        session = Session(
            path=Path(bids_path.fpath),
            data=[raw.get_data(picks=picks, start=a, stop=b).astype(np.float64) * 1e6
                  for a, b in ranges],
            sfreq=sfreq, channel_names=names, channel_types=[types[i] for i in picks],
            segment_start_us=np.array([round(a / sfreq * 1e6) for a, _ in ranges]),
            time_origin='recording_start', events=events,
            bad_channels=[name for name in raw.info['bads'] if name in names],
            source_files=[str(path) for path in raw.filenames if path is not None],
            channel_positions=coordinates,
            metadata={'format': 'bids', 'bids_path': str(bids_path.fpath),
                      'subject': bids_path.subject, 'session': bids_path.session,
                      'task': bids_path.task, 'run': bids_path.run,
                      'measurement_date': str(raw.info['meas_date']),
                      'coordinate_frame': None if positions is None else positions['coord_frame'],
                      'coordinate_units': 'm', 'excluded_non_voltage_channels': excluded,
                      'original_sample_ranges': ranges})
        # Account for leading acquisition skips in the common onset convention.
        shift = session.segment_start_us[0] / 1e6
        for event in session.events:
            event.onset -= shift
        session.map_events(getattr(config, 'EVENT_ID', None))
        return session, config
    finally:
        raw.close()

def _selected_files(root, explicit, suffix):
    if explicit is not None and not isinstance(explicit, (list, tuple)):
        raise ValueError(f'The {suffix} file selection must be a list of paths')
    if explicit:
        files = [(root / Path(name).expanduser()).resolve(strict=True) for name in explicit]
    else:
        files = sorted(path for path in root.iterdir() if path.suffix.lower() == suffix)
    if len(set(files)) != len(files):
        raise ValueError(f'Duplicate {suffix} files selected')
    for path in files:
        if not path.is_file() or path.suffix.lower() != suffix:
            raise ValueError(f'Expected a {suffix} file: {path}')
    return files

def _read_nev(path, origin_us):
    """Read standard Neuralynx NEV records, including events outside acquisition.

    Binary layout: NeuralynxDataFileFormats.pdf, revision 1.1, event records.
    TTL is exposed as the unsigned 16-bit bitmask; no task decoding is assumed.
    """
    size = path.stat().st_size - HEADER_BYTES
    if size < 0 or size % NEV_DTYPE.itemsize:
        raise ValueError(f'Truncated or invalid NEV file: {path}')
    with path.open('rb') as stream:
        header = stream.read(HEADER_BYTES)
        if not header.startswith(b'########'):
            raise ValueError(f'Invalid Neuralynx header: {path}')
        records = np.fromfile(stream, dtype=NEV_DTYPE)
    return [Event(onset=(int(row['timestamp']) - origin_us) / 1e6, duration=0.,
                  description=bytes(row['description']).decode('utf-8', errors='replace'),
                  timestamp_us=int(row['timestamp']), ttl=int(row['ttl']),
                  acquisition_event_id=int(row['event_id']), source=str(path))
            for row in records]

def load_ncs(config):
    """Read standard Neuralynx NCS channels using Neo, retaining acquisition gaps.

    Each channel is parsed separately so incompatible rates or segment layouts
    fail explicitly instead of being padded, truncated, or silently aligned.
    Neo applies calibration and InputInverted polarity from each file header.
    All selected channels must share their sample grid. No resampling occurs.
    """
    from neo.rawio import NeuralynxRawIO

    root = _data_path(config)
    files = _selected_files(root, getattr(config, 'NCS_FILES', None), '.ncs')
    if not files:
        raise FileNotFoundError(f'No NCS files in {root}')
    name_map = getattr(config, 'NCS_CHANNEL_MAP', {}) or {}
    unknown = set(name_map) - {path.name for path in files}
    if unknown:
        raise ValueError(f'NCS_CHANNEL_MAP keys must be selected filenames: {sorted(unknown)}')
    data, names, headers = [], [], []
    starts = rate = sizes = None
    for channel_index, path in enumerate(files):
        size = path.stat().st_size - HEADER_BYTES
        if size <= 0 or size % 1044:
            raise ValueError(f'Empty, truncated, or invalid NCS file: {path}')
        reader = NeuralynxRawIO(dirname=str(path.parent), include_filenames=[path.name],
                                keep_original_times=True, gap_tolerance_ms=0.0)
        reader.parse_header()
        channels = reader.header['signal_channels']
        if len(channels) != 1:
            raise ValueError(f'Expected one continuous channel in {path}')
        channel = channels[0]
        unit = str(channel['units'])
        factor = {'uV': 1., 'µV': 1., 'mV': 1e3, 'V': 1e6}.get(unit)
        if factor is None:
            raise ValueError(f'Unsupported signal unit {unit!r} in {path}')
        this_rate = float(reader.get_signal_sampling_rate(stream_index=0))
        this_starts, segments = [], []
        for index in range(reader.segment_count(0)):
            this_starts.append(round(reader.get_signal_t_start(0, index, stream_index=0) * 1e6))
            chunk = reader.get_analogsignal_chunk(0, index, stream_index=0)
            scaled = reader.rescale_signal_raw_to_float(chunk, dtype='float64', stream_index=0)
            segments.append(scaled[:, 0] * factor)
        this_sizes = [len(segment) for segment in segments]
        if starts is None:
            starts, rate, sizes = np.array(this_starts, dtype=np.int64), this_rate, this_sizes
            data = [np.empty((len(files), size), dtype=np.float64) for size in sizes]
        elif (not np.isclose(this_rate, rate, rtol=1e-9, atol=0)
              or this_sizes != sizes or not np.array_equal(this_starts, starts)):
            raise ValueError(f'NCS channel is not aligned with the other channels: {path.name}. '
                             'Select a synchronized channel group; no implicit resampling is performed.')
        names.append(name_map.get(path.name, str(channel['name'])))
        for target, segment in zip(data, segments):
            target[channel_index] = segment
        headers.append(dict(next(iter(reader.file_headers.values()))))
    overrides = getattr(config, 'CHANNEL_TYPE_OVERRIDES', {}) or {}
    if set(overrides) - set(names):
        raise ValueError('CHANNEL_TYPE_OVERRIDES contains unknown NCS channel names')
    default_type = getattr(config, 'NCS_DEFAULT_CHANNEL_TYPE', 'seeg')
    types = [overrides.get(name, default_type) for name in names]
    if any(kind not in VOLTAGE_TYPES for kind in types):
        raise ValueError('Selected NCS channels must have voltage channel types')
    event_files = _selected_files(root, getattr(config, 'NCS_EVENT_FILES', None), '.nev')
    events = [event for path in event_files for event in _read_nev(path, int(starts[0]))]
    if not event_files:
        warnings.warn('No NEV files selected; NCS session has no acquisition events', stacklevel=2)
    session = Session(
        path=root, data=data,
        sfreq=rate, channel_names=names, channel_types=types, segment_start_us=starts,
        time_origin='neuralynx_clock', events=events,
        source_files=[str(path) for path in files + event_files], headers=headers,
        metadata={'format': 'ncs', 'channel_source_files': dict(zip(names, map(str, files))),
                  'polarity': 'Neo header-calibrated; InputInverted corrected',
                  'coordinate_units': 'm', 'coordinate_frame': None})
    session.map_events(getattr(config, 'EVENT_ID', None))
    return session, config

def participant_session(session, config):
    """Explicit IDs for NCS; BIDS labels may be inherited from the source."""
    import re
    entities = getattr(config, 'BIDS_ENTITIES', {}) or {}
    subject = getattr(config, 'PARTICIPANT_ID', None) or session.metadata.get('subject') or entities.get('subject')
    visit = getattr(config, 'SESSION_ID', None) or session.metadata.get('session') or entities.get('session')
    if visit is None and session.metadata.get('format') == 'bids':
        visit = 'nosession'
    for name, value in [('PARTICIPANT_ID', subject), ('SESSION_ID', visit)]:
        if value is None or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', str(value)):
            raise ValueError(f'Set {name} to a safe participant/session label')
    return str(subject), str(visit)


def output_directory(session, config):
    subject, visit = participant_session(session, config)
    return Path(config.OUT_PATH) / f'sub-{subject}' / f'ses-{visit}'


def attach_coordinates(session, config):
    """Optional CSV: name (or electrode), x, y, z; explicit coordinate space."""
    import csv
    filename = getattr(config, 'ELECTRODES_CSV', None)
    if filename is None:
        return session
    from copy import deepcopy
    from dataclasses import replace
    space = getattr(config, 'COORDINATE_SPACE', None)
    units = getattr(config, 'COORDINATE_UNITS', 'm')
    if not space or units not in ('m', 'cm', 'mm'):
        raise ValueError('Coordinate CSV requires COORDINATE_SPACE and units m, cm, or mm')
    positions = {}
    with (Path(config.PROJECT_PATH) / filename).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        names = reader.fieldnames or []
        label = 'name' if 'name' in names else 'electrode'
        if not {label, 'x', 'y', 'z'} <= set(names):
            raise ValueError('Electrode CSV requires name (or electrode), x, y, z columns')
        for row in reader:
            name = row[label].strip()
            if name in positions:
                raise ValueError(f'Duplicate electrode coordinate: {name}')
            if name in session.channel_names:
                xyz = np.array([float(row[axis]) for axis in ('x', 'y', 'z')])
                if not np.isfinite(xyz).all():
                    raise ValueError(f'Nonfinite coordinate for {name}')
                positions[name] = (xyz * {'m': 1., 'cm': .01, 'mm': .001}[units]).tolist()
    metadata = deepcopy(session.metadata)
    metadata.update(coordinate_frame=space, coordinate_units='m', coordinate_source=str(filename))
    return replace(session, channel_positions=positions, metadata=metadata)


def _json_value(value):
    """Use JSON null for unavailable/nonfinite metadata, never nonstandard NaN."""
    import math
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return value


def get_info(session, config):
    """Build metadata for the saved continuous signal; no files are written."""
    from dataclasses import asdict
    import hashlib
    from importlib.metadata import version, PackageNotFoundError
    import platform
    from datetime import datetime, timezone
    subject, visit = participant_session(session, config)
    versions = {'python': platform.python_version()}
    for package in ('numpy', 'scipy', 'mne', 'mne-bids', 'neo'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            pass
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ('preproc.py', 'setting.py')}
    segments = []
    for i, (onset, data, duration) in enumerate(zip(session.segment_onsets, session.data, session.segment_duration_s)):
        segments.append({'array': f'segment_{i:03d}', 'shape': list(data.shape),
                         'start_seconds': float(onset), 'acquisition_duration_seconds': duration,
                         'source_start_us': int(session.segment_start_us[i]),
                         'run': session.metadata.get('run'),
                         'time': (onset + np.arange(data.shape[1]) / session.sfreq).tolist()})
    info = {
        'schema_version': 1, 'participant': subject, 'session': visit,
        'FS': session.sfreq, 'native_FS': session.native_sfreq, 'units': session.units,
        'array_axes': ['channel', 'sample'], 'time_units': 'seconds',
        'time_reference': 'first acquired sample', 'source_time_origin': session.time_origin,
        'segments': segments, 'electrode_names': session.channel_names,
        'channel_types': session.channel_types,
        'electrode_coordinates': [session.channel_positions.get(name) for name in session.channel_names],
        'space': session.metadata.get('coordinate_frame'),
        'coordinate_units': session.metadata.get('coordinate_units', 'm'),
        'preprocessing_steps': session.metadata.get('preprocessing_steps', []),
        'filter_parameters': session.metadata.get('filter_parameters'),
        'excluded_channels': session.metadata.get('excluded_channels', []),
        'flagged_channels': session.bad_channels,
        'review_decisions': session.metadata.get('review_decisions', {}),
        'bipolar_mapping': session.metadata.get('bipolar_mapping', []),
        'reference': session.metadata.get('reference', 'source reference; see source headers'),
        'events': [asdict(event) for event in session.events],
        'source_files': session.source_files,
        'source_headers': session.headers,
        'run_boundaries': [{'run': session.metadata.get('run'), 'segments': list(range(len(segments)))}],
        'software_versions': versions, 'pipeline_source_sha256': source_hashes,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_metadata': {key: value for key, value in session.metadata.items()
                            if key not in ('review_decisions', 'preprocessing_steps', 'filter_parameters')},
    }
    return _json_value(info)


def save_data(session, config):
    """Save only neural.npz and info.json; refuse to overwrite either file."""
    import json
    import os
    import tempfile
    destination = output_directory(session, config)
    neural, info = destination / 'neural.npz', destination / 'info.json'
    if neural.exists() or info.exists():
        raise FileExistsError(f'Output already exists in {destination}')
    metadata = get_info(session, config)
    destination.mkdir(parents=True, exist_ok=True)
    # Stage both files; exclusive hard links prevent accidental overwrites.
    with tempfile.TemporaryDirectory(prefix='.saving-', dir=destination) as tmp:
        tmp = Path(tmp)
        np.savez(tmp / 'neural.npz', **{f'segment_{i:03d}': x for i, x in enumerate(session.data)})
        with (tmp / 'info.json').open('w', encoding='utf-8') as stream:
            json.dump(metadata, stream, allow_nan=False, indent=2)
        os.link(tmp / 'neural.npz', neural)
        try:
            os.link(tmp / 'info.json', info)
        except Exception:
            neural.unlink()
            raise
    return neural, info
