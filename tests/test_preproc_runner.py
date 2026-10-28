import unittest
from types import SimpleNamespace
from unittest.mock import patch

import preproc


class RunnerTests(unittest.TestCase):
    def config(self, steps, fmt='bids'):
        return SimpleNamespace(preproc_pipeline=steps, data_format=fmt)

    def test_order_and_session_handoff_for_both_formats(self):
        for fmt in ('bids', 'ncs'):
            with self.subTest(fmt=fmt):
                calls = []
                original, filtered, bipolar = object(), object(), object()
                cfg = self.config(['preprocess', 'bipolar_reference', 'gross_channel_removal'], fmt)
                def load(config):
                    self.assertIs(config, cfg)
                    calls.append('load')
                    return original, config
                def process(session, config):
                    self.assertIs(session, original)
                    calls.append('preprocess')
                    return filtered
                def reference(session, config):
                    self.assertIs(session, filtered)
                    calls.append('bipolar_reference')
                    return bipolar, ['mapping'], ['unpaired']
                def review(session, config):
                    self.assertIs(session, bipolar)
                    calls.append('gross_channel_removal')
                    return ['kept'], ['history']
                with patch.multiple(preproc, load_data=load, preprocess=process,
                                    bipolar_reference=reference, gross_channel_removal=review):
                    session, outputs = preproc.run_preprocess(cfg)
                self.assertIs(session, bipolar)
                self.assertEqual(calls, ['load'] + cfg.preproc_pipeline)
                self.assertEqual(outputs[1][1][1:], (['mapping'], ['unpaired']))
                self.assertEqual(outputs[2][1], (['kept'], ['history']))

    def test_supplied_session_and_repeated_steps(self):
        session = object()
        with patch.object(preproc, 'preprocess', return_value=session) as process:
            result, outputs = preproc.run_preprocess(self.config(['preprocess', 'preprocess']), session)
        self.assertIs(result, session)
        self.assertEqual(process.call_count, 2)
        self.assertEqual([name for name, _ in outputs], ['preprocess', 'preprocess'])
        with patch.object(preproc, 'load_data') as load:
            result, outputs = preproc.run_preprocess(self.config([]), session)
            load.assert_not_called()
            self.assertIs(result, session)


if __name__ == '__main__':
    unittest.main()
