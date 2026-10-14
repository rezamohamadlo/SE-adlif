from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import queue_shd_dropout as queue
import search_shd


class DropoutQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_only_dropout_changes_from_control(self):
        source = search_shd.source_config()
        for item in queue.build_plan(self.root):
            cfg = item['manifest']['config']
            self.assertIn(cfg['dropout'], (.25, .35))
            self.assertEqual(cfg['lr'], .01)
            self.assertEqual(cfg['batch_size'], 512)
            self.assertFalse(cfg['run_test'])
            self.assertIsNone(cfg['init_ckpt_path'])
            self.assertIsNone(cfg['ckpt_path'])
            for key in ('l1', 'l2', 'l_out', 'early_stopping_patience', 'patience'):
                self.assertEqual(cfg[key], source[key])

    def test_queue_waits_for_all_nine_and_checks_margin(self):
        for candidate in search_shd.candidates()[:-1]:
            path = self.root/candidate['id']/'seed_42/completed.txt'
            path.parent.mkdir(parents=True)
            path.touch()
        self.assertFalse(queue.source_finished(self.root))
        path = self.root/'r09/seed_42/completed.txt'
        path.parent.mkdir(parents=True)
        path.touch()
        self.assertTrue(queue.source_finished(self.root))
        with patch.object(queue.time, 'time', return_value=1000):
            self.assertFalse(queue.fits_budget(2000, 300, 25))
            self.assertTrue(queue.fits_budget(11000, 300, 25))

    def test_selecting_d25_does_not_schedule_d35(self):
        plan = queue.build_plan(self.root, ['d25'])
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]['manifest']['candidate']['id'], 'd25')
