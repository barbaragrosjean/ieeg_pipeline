"""Synthetic on-disk recordings exercise the actual MNE-BIDS and Neo readers."""
import struct
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
from mne_bids import BIDSPath, write_raw_bids

from setting import load_bids, load_data, load_ncs
from setting import Session, save_data
from preproc import bipolar_reference


# Independent fixture writers use the documented Neuralynx binary layout.
def header(name='A1', rate=1000, gain=1e-6, inverted=False):
    text = f'''######## Neuralynx Data File Header
-FileType NCS
-FileVersion 3.4
-RecordSize 1044
-ApplicationName Cheetah "6.4.2"
-AcquisitionSystem AcqSystem1 DigitalLynxSX
-TimeCreated 2020/01/01 12:00:00
-TimeClosed 2020/01/01 12:05:00
-SamplingFrequency {rate}
-ADBitVolts {gain}
-AcqEntName {name}
-NumADChannels 1
-ADChannel 1
-InputRange 1000
-InputInverted {str(inverted)}
'''
    return text.encode().ljust(16384, b'\x00')


def write_ncs(path, name='A1', offset=0, rate=1000, gain=1e-6, inverted=False):
    # Two continuous blocks (second partly valid), then a gap, then a third.
    records = [(1_000_000 + offset, np.arange(512)),
               (1_512_000 + offset, np.arange(100) + 512),
               (3_000_000 + offset, np.arange(512) + 1000)]
    with path.open('wb') as stream:
        stream.write(header(name, rate, gain, inverted))
        for timestamp, values in records:
            stream.write(struct.pack('<QIII', timestamp, 1, rate, len(values)))
            samples = np.full(512, -30000, dtype='<i2')
            samples[:len(values)] = values
            stream.write(samples.tobytes())
    return [np.arange(612), np.arange(512) + 1000]


def write_nev(path):
    with path.open('wb') as stream:
        stream.write(b'######## Neuralynx Data File Header\n'.ljust(16384, b'\x00'))
        for timestamp, ttl, text in [(1_010_000, 7, 'target'),
                                     (2_000_000, 0, 'gap event'),
                                     (3_010_000, 32769, 'target'),
                                     (4_000_000, 0, 'outside')]:
            stream.write(struct.pack('<hhhQhhhhh8i128s', 0, 0, 2, timestamp, 19,
                                     ttl if ttl < 32768 else ttl - 65536,
                                     0, 0, 0, *([0] * 8), text.encode()))


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def config(self, **kwargs):
        values = dict(PROJECT_PATH=self.root, DATA_PATH='.', data_format='ncs',
                      BIDS_ENTITIES={'subject': '01', 'task': 'test'},
                      TASK=None, FS_in=None, EVENT_ID={}, CHANNEL_TYPE_OVERRIDES={},
                      NCS_FILES=[], NCS_EVENT_FILES=[], NCS_CHANNEL_MAP={},
                      NCS_DEFAULT_CHANNEL_TYPE='seeg')
        values.update(kwargs)
        return SimpleNamespace(**values)

    def ncs(self):
        expected = write_ncs(self.root / 'CSC1.ncs', name='A1', gain=2e-6)
        write_ncs(self.root / 'CSC2.ncs', name='A2', inverted=True)
        write_nev(self.root / 'Events.nev')
        return expected

    def bids(self, gaps=False, run='01'):
        values = np.vstack((np.arange(2000) % 101, -(np.arange(2000) % 71))).astype(float)
        raw = mne.io.RawArray(values * 1e-6,
                             mne.create_info(['A1', 'A2'], 1000, ['seeg', 'seeg']),
                             verbose=False)
        raw.set_montage(mne.channels.make_dig_montage(
            ch_pos={'A1': [0.01, 0.02, 0.03], 'A2': [0.02, 0.02, 0.03]}, coord_frame='ras'))
        raw.info['line_freq'] = 50
        raw.info['bads'] = ['A2']
        onset, duration, descriptions = [0.1, 1.5], [0., 0.], ['target', 'target']
        if gaps:
            onset += [0.4, 0.5]
            duration += [0.2, 0.]
            descriptions += ['BAD_ACQ_SKIP', 'gap event']
        raw.set_annotations(mne.Annotations(onset, duration, descriptions))
        path = BIDSPath(root=self.root, subject='01', task='test', run=run, datatype='ieeg')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            write_raw_bids(raw, path, format='BrainVision', allow_preload=True,
                           event_id={'target': 7, 'BAD_ACQ_SKIP': 99, 'gap event': 8},
                           overwrite=True, acpc_aligned=True, verbose=False)
        return values

    def test_ncs_calibration_gaps_partial_blocks_and_events(self):
        expected = self.ncs()
        session, returned_config = load_data(self.config(EVENT_ID={'target': 42}))
        self.assertIsInstance(session, Session)
        self.assertEqual(session.units, 'uV')
        self.assertEqual(session.sfreq, 1000)
        np.testing.assert_array_equal(session.segment_start_us, [1_000_000, 3_000_000])
        for segment, original in zip(session.data, expected):
            np.testing.assert_allclose(segment[0], original * 2)
            np.testing.assert_allclose(segment[1], -original)
        self.assertEqual([(e.segment, e.sample) for e in session.events],
                         [(0, 10), (-1, -1), (1, 10), (-1, -1)])
        self.assertEqual(session.events[2].ttl, 32769)
        self.assertEqual(session.events[0].code, 42)
        self.assertEqual(session.events[0].acquisition_event_id, 19)
        self.assertEqual(session.events[2].timestamp_us, 3_010_000)

    def test_ncs_rejects_misalignment_and_truncation(self):
        self.ncs()
        write_ncs(self.root / 'CSC2.ncs', name='A2', offset=1000)
        with self.assertRaisesRegex(ValueError, 'not aligned'):
            load_ncs(self.config())
        (self.root / 'CSC2.ncs').write_bytes(b'invalid')
        with self.assertRaisesRegex(ValueError, 'invalid NCS'):
            load_ncs(self.config())

    def test_ncs_channel_mapping(self):
        self.ncs()
        config = self.config(NCS_CHANNEL_MAP={'CSC1.ncs': 'B1', 'CSC2.ncs': 'B2'},
                             CHANNEL_TYPE_OVERRIDES={'B2': 'ecg'})
        session, returned_config = load_ncs(config)
        self.assertEqual(session.channel_names, ['B1', 'B2'])
        self.assertEqual(session.channel_types, ['seeg', 'ecg'])
        self.assertIs(returned_config, config)

    def test_bids_signal_units_bads_and_events(self):
        expected = self.bids()
        session, returned_config = load_data(self.config(data_format='bids'))
        self.assertIsInstance(session, Session)
        self.assertEqual(session.units, 'uV')
        np.testing.assert_allclose(session.data[0], expected, atol=1e-5)
        self.assertEqual(session.bad_channels, ['A2'])
        self.assertEqual(session.channel_types, ['seeg', 'seeg'])
        self.assertEqual([(e.segment, e.sample, e.code) for e in session.events],
                         [(0, 100, 7), (0, 1500, 7)])
        self.assertEqual(session.time_origin, 'recording_start')
        np.testing.assert_allclose(session.channel_positions['A1'], [0.01, 0.02, 0.03])

    def test_bids_gap_is_not_filtered_as_acquired_data(self):
        expected = self.bids(gaps=True)
        session, returned_config = load_bids(self.config())
        self.assertEqual([part.shape[1] for part in session.data], [400, 1400])
        np.testing.assert_array_equal(session.segment_start_us, [0, 600000])
        np.testing.assert_allclose(session.data[1], expected[:, 600:], atol=1e-5)
        event = next(e for e in session.events if e.description == 'gap event')
        self.assertEqual((event.segment, event.sample), (-1, -1))
        event = next(e for e in session.events if e.onset == 1.5)
        self.assertEqual((event.segment, event.sample), (1, 900))

    def test_bids_ambiguous_selection_rejected(self):
        self.bids(run='01')
        self.bids(run='02')
        with self.assertRaisesRegex(ValueError, 'found 2'):
            load_bids(self.config())
        session, returned_config = load_bids(self.config(BIDS_ENTITIES={'subject': '01', 'task': 'test', 'run': '02'}))
        self.assertIn('run-02', str(session.path))

    def test_resting_bids_without_events(self):
        self.bids()
        # A missing events sidecar is valid for resting-state recordings.
        for path in self.root.rglob('*_events.*'):
            path.unlink()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            session, returned_config = load_bids(self.config())
        self.assertIsInstance(session, Session)
        # Embedded BrainVision markers may remain; they have no invented task codes.
        self.assertTrue(all(event.code is None for event in session.events))

    def test_equivalent_signals_produce_the_same_session_data(self):
        self.ncs()
        ncs, _ = load_ncs(self.config(EVENT_ID={'target': 7}))
        values = np.zeros((2, 2512))
        values[:, :612] = ncs.data[0]
        values[:, 2000:] = ncs.data[1]
        raw = mne.io.RawArray(values * 1e-6,
                             mne.create_info(['A1', 'A2'], 1000, ['seeg', 'seeg']),
                             verbose=False)
        raw.info['line_freq'] = 50
        raw.set_annotations(mne.Annotations([.01, .612, 1., 2.01],
                                           [0., 1.388, 0., 0.],
                                           ['target', 'BAD_ACQ_SKIP', 'gap event', 'target']))
        path = BIDSPath(root=self.root / 'bids', subject='01', task='test', datatype='ieeg')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            write_raw_bids(raw, path, format='BrainVision', allow_preload=True,
                           event_id={'target': 7, 'BAD_ACQ_SKIP': 99, 'gap event': 8},
                           overwrite=True, verbose=False)
            bids, _ = load_bids(self.config(DATA_PATH='bids'))
        self.assertEqual(ncs.channel_names, bids.channel_names)
        self.assertEqual(ncs.channel_types, bids.channel_types)
        self.assertEqual(ncs.units, bids.units)
        self.assertEqual(ncs.sfreq, bids.sfreq)
        np.testing.assert_allclose(ncs.segment_onsets, bids.segment_onsets)
        for expected, actual in zip(ncs.data, bids.data):
            np.testing.assert_allclose(actual, expected, atol=1e-4)
        self.assertEqual([(e.segment, e.sample, e.code) for e in ncs.events if e.description == 'target'],
                         [(e.segment, e.sample, e.code) for e in bids.events if e.description == 'target'])

    def test_save_loaded_ncs_metadata(self):
        self.ncs()
        config = self.config()
        config.PARTICIPANT_ID = '01'
        config.SESSION_ID = '01'
        config.OUT_PATH = self.root / 'outs'
        session, _ = load_ncs(config)
        neural, info = save_data(session, config)
        self.assertTrue(neural.exists())
        import json
        metadata = json.loads(info.read_text())
        self.assertEqual(len(metadata['source_headers']), 2)
        self.assertEqual(metadata['native_FS'], 1000)

    def test_shared_bipolar_interface(self):
        self.ncs()
        config = self.config()
        session, returned_config = load_ncs(config)
        result, mapping, unpaired = bipolar_reference(session, config)
        self.assertEqual(result.channel_names, ['A1-A2'])
        self.assertEqual(result.channel_types, ['seeg'])
        np.testing.assert_allclose(result.data[0][0], session.data[0][1] - session.data[0][0])
        self.assertEqual(len(result.events), 4)
        self.assertEqual(len(mapping), 1)
        self.assertEqual(unpaired, [])


if __name__ == '__main__':
    unittest.main()
