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

## Preprocessing

```python
from setting import load_config
from preproc import run_preprocess

config = load_config('/path/to/project/config.py')
session, step_outputs = run_preprocess(config)
```

The runner loads once and executes `preproc_pipeline` in its exact listed order.
It saves `neural.npz` and `info.json` under
`OUT_PATH/sub-<PARTICIPANT_ID>/ses-<SESSION_ID>/`. BIDS IDs can be inherited;
NCS requires explicit IDs. A BIDS recording without a session label uses
`ses-nosession`. One selected recording is processed at a time; multiple runs
are not automatically merged. Existing final files are never overwritten.

The neural archive contains `segment_000`, `segment_001`, etc., each a
channels-by-samples array. All continuous segments remain in one archive.
No epoching, task cropping, normalization, or power calculation occurs.
`info.json` contains full per-segment time arrays, FS/native FS, ordered steps,
filter/resampling parameters, coordinate space and units, channel names/types,
events and updated indices, exclusions/reasons, review decisions, bipolar
mappings, run/segment boundaries, source identifiers/headers, software versions,
and source-code hashes. Missing coordinates and metadata are JSON null.

`preprocess()` follows GMNUM: linear detrend, demean, zero-phase Butterworth
high-pass, then sequential Butterworth bandstop filters. `FILTER_ORDER=4` is the
prototype design order; forward/backward filtering increases the effective
order. Configure `HIGH_PASS_HZ`, `NOTCH_HZ` (e.g. `[50, 100, 150]`), and
`NOTCH_WIDTH_HZ`. An empty notch list disables notches. These example frequencies
are not study-specific recommendations. Requested bands outside input Nyquist
are rejected. `FS_out=None` retains the rate; otherwise polyphase resampling
with anti-alias filtering follows the GMNUM filtering sequence. Acquisition
start times and durations stay unchanged. Each segment is processed separately;
filters never span gaps. Very short segments fail explicitly.

### CSV exclusions

All exclusion outputs have only these columns:

```csv
name,criterion
A2,artifact
A4,reviewed_epilepsy
```

- `gross_channel_removal`: proposes GMNUM variance outliers and exact-zero
  channels, writing only `gross_exclusions.csv`. `GROSS_REVIEW_MODE='interactive'`
  plots each proposed channel and asks for y/n confirmation in the terminal.
  `automatic` accepts proposals without confirmation. This heuristic does not
  determine the physiological cause of a channel's abnormality.
- `epilepsy_removal`: reads `EPILEPSY_CHANNEL_CSV` containing reviewed decisions
  and writes only `epilepsy_exclusions.csv`. It does not detect epileptiform
  activity automatically. Omit this step when no such review is available.
- `bad_removal` (also named `remove_bad`): reads `BAD_CHANNEL_CSVS` and CSVs
  produced by preceding review stages, then removes matching channels in memory.
  `EXCLUSION_CRITERIA=None` applies every listed criterion; a list selects only
  matching criteria. LB `name`-only lists are accepted. LB `electrode,label`
  anatomy tables default to OOB/LSN/WM/VENT exclusions, avoiding removal of every
  listed electrode. Relative CSV paths resolve against PROJECT_PATH.

Review stages never save neural data or modify signal arrays. Their CSVs are the
only extra files beyond the final neural/info pair. Existing review CSVs are not
overwritten; reuse them through `BAD_CHANNEL_CSVS` and omit the review step.

To remove contacts before referencing, place `bad_removal` before
`bipolar_reference` in your list. Pairs require consecutive contact numbers:
a missing A2 never produces an A1–A3 pair. Coordinates are contact midpoints.
The sign convention remains **A2 minus A1**, recorded in metadata. If exclusions
are applied after referencing, an excluded source contact removes every bipolar
pair containing it. Unpaired channels are recorded in metadata.

Optional `ELECTRODES_CSV` provides `name` (or `electrode`), `x`, `y`, `z`.
Set `COORDINATE_SPACE` and `COORDINATE_UNITS` (`m`, `cm`, or `mm`); positions are
converted to meters. The CSV is attached before preprocessing/referencing.
Without it, available BIDS coordinates are retained and missing NCS coordinates
remain unknown.

`broadband()` and `power()` intentionally remain empty. They are excluded from
the default list, and requesting them raises an explicit error before loading.
`run_preprocess(..., save=False)` allows in-memory processing without the final
neural/info write; configured review stages still write their CSVs.

## Verification

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

Tests use synthetic BrainVision/BIDS and NCS/NEV files and known signals. They
cover loader equivalence, calibration, gaps, partial records, event remapping,
notch attenuation, CSV exclusions, bipolar gaps/midpoints, and output round trips.
Real recordings and interactive plotting still require practical validation.

References:
- [MNE-BIDS reader](https://mne.tools/mne-bids/stable/generated/mne_bids.read_raw_bids.html)
- [Neuralynx binary file formats](https://neuralynx.fh-co.com/wp-content/uploads/2023/05/NeuralynxDataFileFormats.pdf)
- [SciPy sosfiltfilt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html)
- [SciPy resample_poly](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
