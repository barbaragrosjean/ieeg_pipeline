# ieeg_pipeline

## Configuration and loading

Keep a `config.py` in each project. Shared code receives the explicitly loaded
configuration. Use Python 3.10+ and install the dependencies:

```sh
python -m pip install -r requirements.txt
```

```python
from setting import load_config, load_data

config = load_config('/absolute/path/to/project/config.py')
session, config = load_data(config)
```

`load_bids(config)` and `load_ncs(config)` are also available directly from `setting`. `PROJECT_PATH` is relative to the configuration file; `OUT_PATH` and
`DATA_PATH` are relative to `PROJECT_PATH`. Absolute paths are supported. Loading
Python configuration executes code, so use trusted configuration files.

### BIDS

```python
data_format = 'bids'
DATA_PATH = 'data/BIDS'
TASK = 'learning'
BIDS_ENTITIES = {'subject': '01', 'session': '01', 'run': '01'}
```

Select **one recording**, using entity labels without prefixes such as `sub-`.
Additional BIDS entities (e.g. `acquisition`, `extension`) can disambiguate the
selection. Multiple matches produce an error. Reading uses MNE-BIDS and its
supported iEEG readers. Events, bad-channel labels, channel types, and available
electrode coordinates are imported. Coordinates remain in meters and their
coordinate frame is recorded in `session.metadata`.

Explicit `BAD_ACQ_SKIP` intervals are excluded from signal arrays, preserving
their elapsed time. `BAD boundary`/`EDGE boundary` annotations split segments.
Other bad intervals remain in the signal and event/annotation list. Unknown,
unmarked acquisition gaps cannot be reconstructed. Non-voltage channels such
as digital stimulus channels are excluded with a warning; event metadata comes
from annotations/BIDS sidecars rather than automatic trigger decoding.

### NCS

```python
data_format = 'ncs'
DATA_PATH = 'recordings/session01/macro'
NCS_FILES = []  # All .ncs files directly in DATA_PATH; or an explicit list
NCS_EVENT_FILES = ['../Events.nev']  # Relative to DATA_PATH
NCS_CHANNEL_MAP = {'CSC1.ncs': 'A1', 'CSC2.ncs': 'A2'}  # Optional
NCS_DEFAULT_CHANNEL_TYPE = 'seeg'
CHANNEL_TYPE_OVERRIDES = {}  # Optional final channel name -> type, e.g. ECG -> ecg
```

Without a map, channel names come from NCS headers. Explicit maps use exact
filenames, including `.ncs`. If `NCS_EVENT_FILES` is empty, only NEV files directly
in `DATA_PATH` are discovered; parent folders are not searched automatically.
Missing event files produce an empty event list and a warning.

Neo reads and calibrates each selected NCS channel, including its `InputInverted`
polarity correction. Segments remain separate; invalid tail samples in partial
records are not treated as acquired data. Selected channels must have matching
sampling rates, segment starts, and segment lengths. Mismatches fail rather than
silently cropping, padding, or resampling channels. Select a synchronized group
using `NCS_FILES` when a folder contains incompatible streams. Data are loaded
into memory; this is not a streaming loader.

Standard NEV files are read independently so events in gaps or outside the
recording are retained. The original microsecond timestamp, unsigned 16-bit TTL
bitmask, acquisition event ID, and description are preserved. TTL patterns are
not automatically interpreted as task conditions. NCS files do not provide
anatomical coordinates or reviewed bad-channel decisions; these remain empty
until supplied by a later metadata/review stage.

### Common output

Both loaders return `(session, config)`. The configuration object is returned unchanged.
The shared `Session` dataclass contains:

| Field | Meaning |
| --- | --- |
| `data` | List of float64 arrays, channels × samples, in microvolts |
| `sfreq`, `native_sfreq` | Sampling frequency in Hz; unchanged during loading |
| `channel_names`, `channel_types` | Aligned channel labels and types |
| `segment_start_us` | Segment starts in microseconds on the source time axis |
| `segment_onsets` | Segment starts in seconds relative to the first acquired sample |
| `time_origin` | `recording_start` for BIDS; `neuralynx_clock` for NCS |
| `events` | List of `Event` objects using the same fields for both formats |
| `bad_channels` | Channel labels marked bad in available metadata |
| `channel_positions` | Available coordinates in meters, keyed by channel name |
| `source_files`, `headers`, `metadata` | Provenance and source-specific information |

An `Event` has an onset/duration in seconds relative to the first acquired sample,
a description, optional task code, and zero-based `segment`/`sample` indices.
Events outside acquisition have both indices set to `-1`. Within acquisition,
onsets map to the nearest sample (half-sample ties go forward, with the final
sample as the upper bound). Durations are retained even if an event spans a gap.
Original NCS timestamps and TTLs occupy separate optional fields; BIDS events
leave them unset. Source clocks are not assumed to be synchronized across datasets.

`EVENT_ID` may map descriptions to explicit task codes, overriding source codes
for those descriptions. Unmapped events remain present; NCS task codes remain
`None` unless supplied. The legacy `config.events` field is not used by these
loaders. The loaders use the native sampling rate from the recording; `FS_in`
and `FS_out` do not override it or trigger resampling. No filtering, rereferencing, epoching, or task-specific event processing
occurs in either loader.

## Preprocessing integration

```python
from preproc import run_preprocess

# Loading is automatic when no session is supplied.
session, step_outputs = run_preprocess(config)
# Or reuse a loaded session:
session, step_outputs = run_preprocess(config, session=session)
```

`preproc_pipeline` lists processing function names; loading is separate and should
not be included as a step. Preprocessing/review stages are still under development;
the loaders can be used independently of unfinished stages.

## Verification

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

Tests create real synthetic BrainVision/BIDS and NCS/NEV files. They check units,
polarity, partial records, acquisition gaps, events, channel metadata, ambiguous
BIDS selections, channel alignment, and equivalent outputs across both formats.
Real recordings have not yet been validated.

References for the reader interfaces and binary event format:
- [MNE-BIDS read_raw_bids](https://mne.tools/mne-bids/stable/generated/mne_bids.read_raw_bids.html)
- [Neo RawIO](https://neo.readthedocs.io/en/latest/rawio.html)
- [Neuralynx binary file formats](https://neuralynx.fh-co.com/wp-content/uploads/2023/05/NeuralynxDataFileFormats.pdf)
