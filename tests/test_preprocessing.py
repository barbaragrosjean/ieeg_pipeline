import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from setting import Session, Event, save_data, attach_coordinates
from preproc import (preprocess, run_preprocess, gross_channel_removal,
                     epilepsy_removal, remove_bad, bipolar_reference)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = SimpleNamespace(
            PROJECT_PATH=self.root, OUT_PATH=self.root / 'outs', PARTICIPANT_ID='01', SESSION_ID='02',
            HIGH_PASS_HZ=.5, NOTCH_HZ=[50, 100], NOTCH_WIDTH_HZ=4, FILTER_ORDER=4, FS_out=None,
            CHANNEL_TYPE_OVERRIDES={}, BAD_CHANNEL_CSVS=[], EXCLUSION_CRITERIA=None,
            GROSS_REVIEW_MODE='automatic', VARIANCE_Z_THRESHOLD=3,
            EPILEPSY_CHANNEL_CSV=None, ELECTRODES_CSV=None, preproc_pipeline=[])

    def session(self, n=6000):
        t = np.arange(n) / 1000
        x = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 100 * t)
        data = np.stack([x, x * .5, x * 2, x * 3])
        return Session(path=self.root / 'recording', data=[data, data.copy()], sfreq=1000.,
                       channel_names=['A1', 'A2', 'A3', 'A4'], channel_types=['seeg'] * 4,
                       segment_start_us=np.array([0, (n + 1000) * 1000]), time_origin='recording_start',
                       events=[Event(.1, 0, 'target'), Event(n / 1000 + .5, 0, 'gap')],
                       channel_positions={f'A{i}': [i * .001, 0, 0] for i in range(1, 5)},
                       metadata={'coordinate_frame': 'mni_tal', 'coordinate_units': 'm'})

    def csv(self, name, rows):
        path = self.root / name
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['name', 'criterion'])
            writer.writerows(rows)
        return path

    def test_filter_notches_and_no_input_mutation(self):
        original = self.session()
        before = original.data[0].copy()
        result = preprocess(original, self.config)
        np.testing.assert_array_equal(original.data[0], before)
        t = np.arange(1000, 5000) / 1000
        for frequency in (50, 100):
            amplitude = abs(np.mean(result.data[0][0, 1000:5000] * np.exp(-2j * np.pi * frequency * t))) * 2
            self.assertLess(amplitude, .1)
        amplitude = abs(np.mean(result.data[0][0, 1000:5000] * np.exp(-2j * np.pi * 10 * t))) * 2
        self.assertGreater(amplitude, .9)
        np.testing.assert_allclose(result.data[0], result.data[1])

    def test_resampling_retains_gaps_and_remaps_events(self):
        self.config.FS_out = 250
        session = self.session(n=4001)
        session.events.append(Event(4.0015, 0, 'outside'))
        result = preprocess(session, self.config)
        self.assertEqual(result.sfreq, 250)
        self.assertEqual(result.native_sfreq, 1000)
        self.assertEqual(result.data[0].shape, (4, 1001))
        np.testing.assert_array_equal(result.segment_start_us, session.segment_start_us)
        self.assertEqual(result.segment_duration_s, session.segment_duration_s)
        self.assertEqual(result.events[0].sample, 25)
        self.assertTrue(all(event.segment == -1 for event in result.events[1:]))
        self.assertEqual(session.events[0].sample, -1)

    def test_filter_validation(self):
        self.config.NOTCH_HZ = [500]
        with self.assertRaisesRegex(ValueError, 'Nyquist'):
            preprocess(self.session(), self.config)
        self.config.NOTCH_HZ = []
        with self.assertRaisesRegex(ValueError, 'too short'):
            preprocess(self.session(4), self.config)

    def test_csv_review_does_not_mutate_signal(self):
        session = self.session()
        for data in session.data:
            data[1] = 0
        before = session.data[0].copy()
        self.config.GROSS_REVIEW_MODE = 'interactive'
        with patch('preproc._review_channel', return_value=True) as review:
            output = gross_channel_removal(session, self.config)
        review.assert_called_once()
        np.testing.assert_array_equal(session.data[0], before)
        self.assertEqual(session.channel_names, ['A1', 'A2', 'A3', 'A4'])
        self.assertEqual(list(self.config.OUT_PATH.rglob('*.*')), [output['csv']])
        self.config.BAD_CHANNEL_CSVS = [output['csv']]
        cleaned = remove_bad(session, self.config)
        referenced, mapping, unused = bipolar_reference(cleaned, self.config)
        self.assertEqual(referenced.channel_names, ['A3-A4'])
        np.testing.assert_allclose(referenced.channel_positions['A3-A4'], [.0035, 0, 0])
        self.assertEqual(unused, ['A1'])

    def test_contact_exclusion_after_bipolar(self):
        self.config.BAD_CHANNEL_CSVS = [self.csv('bad.csv', [('A2', 'artifact')])]
        referenced, _, _ = bipolar_reference(self.session(), self.config)
        result = remove_bad(referenced, self.config)
        self.assertEqual(result.channel_names, ['A3-A4'])
        self.assertEqual(len(result.metadata['excluded_channels']), 2)

    def test_epilepsy_exports_only_csv(self):
        self.config.EPILEPSY_CHANNEL_CSV = self.csv('reviewed.csv', [('A2', 'reviewed_epilepsy')])
        session = self.session()
        result = epilepsy_removal(session, self.config)
        self.assertEqual(list(self.config.OUT_PATH.rglob('*.*')), [result['csv']])
        self.assertEqual(session.channel_names, ['A1', 'A2', 'A3', 'A4'])

    def test_save_exact_outputs_and_metadata(self):
        session = self.session()
        self.config.preproc_pipeline = ['preprocess', 'bipolar_reference']
        self.config.FS_out = 250
        result, outputs = run_preprocess(self.config, session)
        destination = self.config.OUT_PATH / 'sub-01' / 'ses-02'
        self.assertEqual({path.name for path in destination.iterdir()}, {'neural.npz', 'info.json'})
        with np.load(destination / 'neural.npz', allow_pickle=False) as archive:
            self.assertEqual(archive.files, ['segment_000', 'segment_001'])
            np.testing.assert_array_equal(archive['segment_000'], result.data[0])
        info = json.loads((destination / 'info.json').read_text())
        self.assertEqual(info['FS'], 250)
        self.assertEqual(info['preprocessing_steps'], self.config.preproc_pipeline)
        self.assertEqual(len(info['segments'][0]['time']), result.data[0].shape[1])
        self.assertEqual(info['space'], 'mni_tal')
        self.assertEqual(info['electrode_names'], result.channel_names)
        self.assertEqual(info['events'][0]['sample'], 25)
        self.assertEqual(info['events'][1]['sample'], -1)
        with self.assertRaises(FileExistsError):
            save_data(result, self.config)

    def test_coordinate_csv(self):
        path = self.root / 'coords.csv'
        path.write_text('name,x,y,z\nA1,1,2,3\nA2,3,4,5\n')
        self.config.ELECTRODES_CSV = path
        self.config.COORDINATE_SPACE = 'ACPC'
        self.config.COORDINATE_UNITS = 'mm'
        session = attach_coordinates(self.session(), self.config)
        np.testing.assert_allclose(session.channel_positions['A1'], [.001, .002, .003])
        self.assertEqual(session.metadata['coordinate_frame'], 'ACPC')


if __name__ == '__main__':
    unittest.main()
