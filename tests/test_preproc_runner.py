import unittest
from types import SimpleNamespace
from unittest.mock import patch

import preproc
from setting import Session
from pathlib import Path
import numpy as np


class RunnerTests(unittest.TestCase):
    def config(self, steps, fmt='bids'):
        return SimpleNamespace(preproc_pipeline=steps, data_format=fmt, ELECTRODES_CSV=None)

    def test_order_and_session_handoff_for_both_formats(self):
        for fmt in ('bids', 'ncs'):
            with self.subTest(fmt=fmt):
                calls = []
                original, filtered, bipolar = [Session(Path('.'), [np.ones((1, 20))], 1000, ['A1'], ['seeg'], np.array([0]), 'recording_start') for _ in range(3)]
                cfg = self.config(['preprocess', 'bipolar_reference', 'gross_channel_removal'], fmt)
                def load(config):
                    self.assertIs(config, cfg)
                    calls.append('load')
                    return original, config
                def process(session, config):
                    np.testing.assert_array_equal(session.data[0], original.data[0])
                    calls.append('preprocess')
                    return filtered
                def reference(session, config):
                    self.assertIs(session, filtered)
                    calls.append('bipolar_reference')
                    return bipolar, ['mapping'], ['unpaired']
                def review(session, config):
                    self.assertIs(session, bipolar)
                    calls.append('gross_channel_removal')
                    return {'csv': Path('review.csv'), 'decisions': []}
                with patch.multiple(preproc, load_data=load, preprocess=process,
                                    bipolar_reference=reference, gross_channel_removal=review):
                    session, outputs = preproc.run_preprocess(cfg, save=False)
                self.assertIs(session, bipolar)
                self.assertEqual(calls, ['load'] + cfg.preproc_pipeline)
                self.assertEqual(outputs[1][1][1:], (['mapping'], ['unpaired']))
                self.assertEqual(outputs[2][1]['csv'], Path('review.csv'))

    def test_supplied_session_and_repeated_steps(self):
        session = Session(Path('.'), [np.ones((1, 20))], 1000, ['A1'], ['seeg'], np.array([0]), 'recording_start')
        with patch.object(preproc, 'preprocess', return_value=session) as process:
            result, outputs = preproc.run_preprocess(self.config(['preprocess', 'preprocess']), session, save=False)
        self.assertIs(result, session)
        self.assertEqual(process.call_count, 2)
        self.assertEqual([name for name, _ in outputs], ['preprocess', 'preprocess'])
        with patch.object(preproc, 'load_data') as load:
            result, outputs = preproc.run_preprocess(self.config([]), session, save=False)
            load.assert_not_called()
            np.testing.assert_array_equal(result.data[0], session.data[0])


if __name__ == '__main__':
    unittest.main()
