"""Project settings. Copy this file into each project and edit its values.

Load explicitly with setting.load_config(path), then pass the returned
object to shared functions. No plotting style is changed when loading.
"""
from matplotlib.colors import LinearSegmentedColormap

# Paths
PROJECT_PATH = "."  # Relative to this configuration file
OUT_PATH = "outs"  # Relative to PROJECT_PATH
data_format = None  # "bids" or "ncs"
DATA_PATH = None  # BIDS root or NCS directory, relative to PROJECT_PATH
BIDS_ENTITIES = {}  # e.g. {"subject": "01", "session": "01", "run": "01"}
NCS_FILES = []  # Empty: all .ncs files directly in DATA_PATH
NCS_EVENT_FILES = []  # Empty: all .nev files there; otherwise relative paths allowed
NCS_CHANNEL_MAP = {}  # Optional filename -> electrode label, e.g. {"CSC1.ncs": "A1"}
NCS_DEFAULT_CHANNEL_TYPE = "seeg"  # Override auxiliary channels below

# Task
EVENT_ID = {}  # Event codes, or None for resting state
TASK = None
events = []
color_event = {}

# Sampling and analysis parameters
FS_in = None
FS_out = None
FREQS = []
BWIDTH = []

# Preproc pipline
preproc_pipeline = [
        'preprocess',
        'gross_channel_removal',
        'bipolar_reference',
        'epilepsy_removal', 
        'bad_removal']  # broadband/power remain unimplemented

# Component colormaps
cmap_pc0 = LinearSegmentedColormap.from_list("pc0_blue", ["#FFFFFF", "#77ABC8"])
cmap_pc1 = LinearSegmentedColormap.from_list("pc1_orange", ["#FFFFFF", "#E1AB82"])
cmap_pc2 = LinearSegmentedColormap.from_list("pc2_green", ["#FFFFFF", "#74C4AF"])
cmap_pcs = [cmap_pc0, cmap_pc1, cmap_pc2]

col_pc = {0:"#0072B2", 1:"#D55E00", 2:"#009E73", 4:"#009E73"}

FREQ_BAND_DICT = {
    'Delta'      : [0.5, 1, 2, 3],
    'Theta'      : [4, 5, 6, 7],
    'Alpha'      : [8, 9, 10, 11, 12],
    'Low_Beta'   : [13, 15, 17, 19],
    'High_Beta'  : [21, 24, 27, 30],
    'Low_Gamma'  : [35, 40, 45, 50, 55, 60],
    'High_Gamma' : [70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
}

REGION = {'parietal': ['IPS','IP','SP','SPL','AG','SMG','TPJ'],
        'premotor' : ['FEF','SMA'],
        'DLPFC': ['MFG','SFS','IFS', 'SFG'],
        'M1': ['preCG','M1','PreCG'],
        'S1': ['postCG','PostCG'],
        'INS': ['INS'],
        'VLPFC': ['IFG','FOP','IFS'],
        'EC': ['EC','MEC','LEC'],
        'PRH' : ['PRH'],
        'A1': ['A1'],
        'MTG': ['MTG'],
        'AMY': ['AMY'],
        'PCC': ['PCC'],
        'ACC': ['ACC','MCC'],
        'HPC': ['HPC'],
        'PHC': ['PHC','PHG'],
        'STG': ['STG'],
        'STS': ['STS'],
        'TP': ['TP'],
        'OFC': ['OFC'],
        'VS': ['LG','FUG','ITG','ITS'],
        'THAL': ['THAL']}

COL_REG = {
    # --- Memory system (distinct blues)
    'HPC':  '#0b2e8a',   # deep navy
    'PHC':  '#2457c5',   # strong blue
    'EC':  '#6fa3ff',   # light blue
    'PCC':  '#3b4f9f',   # blue–indigo
    'PRH' : '#528EA4',

    # --- Limbic / affective / reward (distinct purples)
    'AMY':  '#5b1a8e',   # deep violet
    'VS':   '#8e44ad',   # saturated purple
    'OFC':  '#c39bd3',   # light lavender

    # --- Prefrontal / executive control (clearly separated reds/oranges)
    'DLPFC':"#7c1f16",   # dark red
    'ACC':  "#aa3124",   # red
    'VLPFC':"#b46433",   # orange

    # --- Sensory–motor (clearly separated greens)
    'M1':       '#145a32',   # dark green
    'premotor': '#229954',   # medium green
    'S1':       '#7dcea0',   # light green

    # --- Auditory / temporal association (strongly separated pink–reds)
    'A1':  "#e00101",   # dark rose
    'STG': "#ae1a3d",   # crimson pink
    'STS': '#e04b78',   # saturated pink
    'MTG': "#fc92b2",   # light pink
    'TP':  "#fbb1b1",   # very pale rose

    # --- Insula (salience bridge: blue–cyan, isolated)
    'INS': '#f1c40f',

    # --- Parietal association (yellow–green, isolated)
    'parietal': '#a3cb38',

    # --- Thalamic relay (olive–brown, isolated)
    'THAL': '#7f6d1f',

    # --- Non-brain / baseline
    'N': '#b3b3b3'
}

FREQ_BAND = list(FREQ_BAND_DICT)

# Set filter parameters explicitly before calling preprocess().
CHANNEL_TYPE_OVERRIDES = {}
HIGH_PASS_HZ = None
NOTCH_HZ = []  # e.g. [50, 100, 150]; no notch filtering when empty
NOTCH_WIDTH_HZ = None
FILTER_ORDER = 4

VARIANCE_Z_THRESHOLD = None  # Set before gross-channel review

# Output identity: NCS requires explicit labels; BIDS labels can be inferred.
PARTICIPANT_ID = None
SESSION_ID = None
# Exclusion files use name,criterion. Relative paths resolve from PROJECT_PATH.
BAD_CHANNEL_CSVS = []
EXCLUSION_CRITERIA = None  # None applies all listed criteria
GROSS_REVIEW_MODE = 'interactive'  # or 'automatic'
EPILEPSY_CHANNEL_CSV = None  # Reviewed decisions; no epilepsy detector is assumed
# Optional coordinates, especially for NCS: CSV with name (or electrode), x,y,z.
ELECTRODES_CSV = None
COORDINATE_SPACE = None  # Must be supplied when ELECTRODES_CSV is used
COORDINATE_UNITS = 'm'  # 'm', 'cm', or 'mm'
