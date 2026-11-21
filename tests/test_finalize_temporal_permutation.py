"""USER-RUN ONLY: synthetic finalization contracts, no neural execution."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import finalize_temporal_permutation as f
from scripts.adaptation_temporal_permutation.run import summary


def score(sid, loss, correct=True, permuted=False):
    audit = dict(original_sha256=f'input-{sid}', transformed_sha256=f'permuted-{sid}' if permuted else f'input-{sid}',
                 event_count=7, channel_counts=[3, 4], width=5, status='effective' if permuted else 'clean')
    return dict(sample_id=sid, loss=loss, predictions=[1], valid_steps=1,
                correct_steps=int(correct), accuracy=float(correct), transformation=audit)


def toy_inputs(seed):
    cp = [score(sid, 1. + sid / 10) for sid in range(4)]
    cv = [score(sid, .8 + sid / 10, sid != 1) for sid in range(4)]
    pp = [[score(sid, 1. + sid / 10 + seed * .0001 + rep * .1, permuted=True)
           for sid in range(4)] for rep in range(3)]
    vp = [[score(sid, .9 + sid / 10 + rep * .02, sid != 2, permuted=True)
           for sid in range(4)] for rep in range(3)]
    return cp, cv, pp, vp


class StreamingTests(unittest.TestCase):
    def test_serialization_consumes_generator_without_materializing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'rows.jsonl'
            def produce():
                for i in range(1000):
                    yield {'sample_id': i, 'text': 'فارسی', 'value': -i / 7}
            # Whole-object json.dumps was the actual production failure.
            with patch.object(f.json, 'dumps', side_effect=AssertionError('whole-object dumps forbidden')):
                self.assertEqual(f.write_rows(p, produce()), 1000)
            self.assertEqual(sum(1 for _ in f.read_rows(p)), 1000)
            self.assertEqual(next(f.read_rows(p))['text'], 'فارسی')

    def test_failed_write_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'rows.jsonl'
            p.write_text('old evidence\n', encoding='utf-8')
            def broken():
                yield {'ok': 1}
                raise MemoryError('synthetic interruption')
            with self.assertRaises(MemoryError):
                f.write_rows(p, broken())
            self.assertEqual(p.read_text(), 'old evidence\n')
            self.assertTrue(list(p.parent.glob('*.tmp')))

    def test_resume_reuses_authenticated_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'rows.jsonl'
            identity = dict(manifest='source', batch=0)
            first = f.ensure_shard(p, identity, lambda: iter([{'n': 1}]))
            second = f.ensure_shard(p, identity, lambda: self.fail('completed shard recomputed'))
            self.assertEqual(first, second)
            p.write_text('tampered\n')
            with self.assertRaises(ValueError):
                f.ensure_shard(p, identity, lambda: self.fail('tampering bypassed'))

    def test_existing_sidecar_missing_array_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'rows.jsonl'
            f.ensure_shard(p, {'batch': 0}, lambda: iter([{'n': 1}]))
            p.unlink()
            with self.assertRaises(ValueError):
                f.ensure_shard(p, {'batch': 0}, lambda: self.fail('missing evidence replaced'))

    def test_unpublished_shard_can_be_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'rows.jsonl'
            p.write_text('interrupted unpublished output')
            r = f.ensure_shard(p, {'batch': 0}, lambda: iter([{'n': 1}, {'n': 2}]))
            self.assertEqual(r['rows'], 2)
            self.assertEqual(list(f.read_rows(p)), [{'n': 1}, {'n': 2}])

    def test_summary_exactly_matches_original_statistical_order(self):
        all_rows = []
        for seed in (42, 123, 456):
            rows = list(f.pair_rows('SHD', seed, 'amt_mt', 5, *toy_inputs(seed)))
            all_rows.extend(rows)
            observed = list(f.sample_metrics(iter(rows)))
            self.assertEqual(len(observed), 4)
            for i, sample in enumerate(observed):
                group = rows[i * 3:(i + 1) * 3]
                self.assertEqual(sample['delta_benefit'], f.statistics.mean(r['delta_benefit'] for r in group))
        legacy = summary(all_rows, dict(training_seeds=[42, 123, 456], permutation_replicates=3))[0]
        for seed in (42, 123, 456):
            metrics = list(f.sample_metrics(r for r in all_rows if r['seed'] == seed))
            self.assertEqual(f.statistics.mean(r['delta_benefit'] for r in metrics), legacy['per_seed'][str(seed)])
            for name in f.FIELDS[1:]:
                self.assertEqual(f.statistics.mean(r[name] for r in metrics), legacy['metrics'][name]['per_seed'][str(seed)])

    def test_missing_replicate_or_wrong_pair_is_rejected(self):
        values = toy_inputs(42)
        rows = list(f.pair_rows('SHD', 42, 'amt_mt', 5, *values))
        with self.assertRaises(ValueError):
            list(f.sample_metrics(rows[:-1]))
        broken = copy.deepcopy(values)
        broken[3][0][0]['transformation']['original_sha256'] = 'other input'
        with self.assertRaises(ValueError):
            list(f.pair_rows('SHD', 42, 'amt_mt', 5, *broken))

    def test_cache_guard_checks_manifest_checkpoint_ids_and_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            m = dict(dataset_metadata={'SHD': {'validation_indices': [3]}}, cohorts={'SHD': [3]})
            run = dict(dataset='SHD', model='MT', seed=42, checkpoint_sha256='ckpt', config={'dataset': {'batch_size': 256}})
            cond = ['clean', 0, 0]
            path = f.score_path(source, run, cond, 0)
            path.parent.mkdir(parents=True)
            value = [score(3, .1)]
            identity = dict(manifest_sha256=f.digest(m), checkpoint='ckpt', original_batch_ids=[3], selected_ids=[3], condition=cond)
            record = dict(identity=identity, value=value, value_sha256=f.digest(value))
            path.write_text(json.dumps(record))
            self.assertEqual(f.cache_rows(source, m, run, cond, 0), value)
            for field, bad in [('checkpoint', 'wrong'), ('original_batch_ids', [9]), ('manifest_sha256', 'wrong')]:
                changed = copy.deepcopy(record); changed['identity'][field] = bad
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    f.cache_rows(source, m, run, cond, 0)
            changed = copy.deepcopy(record); changed['value'][0]['loss'] = 7.
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                f.cache_rows(source, m, run, cond, 0)

    def test_check_only_cannot_aggregate_write_or_run_tests(self):
        with patch.object(f, 'inspect_source', return_value=(None, None, None, None)), \
             patch.object(f, 'finalize', side_effect=AssertionError('scientific execution forbidden')), \
             patch.object(f, 'atomic_write', side_effect=AssertionError('write forbidden')), \
             patch.object(f.subprocess, 'run', side_effect=AssertionError('tests forbidden')):
            self.assertEqual(f.main(['--check-only']), 0)

    def test_wrong_resume_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'rows.jsonl'
            f.ensure_shard(p, {'batch': 0}, lambda: iter([{'n': 1}]))
            with self.assertRaises(ValueError):
                f.ensure_shard(p, {'batch': 1}, lambda: self.fail('wrong identity reused'))

    def test_end_to_end_synthetic_shards_match_legacy_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source', Path(directory) / 'output'
            source.mkdir(); output.mkdir()
            spec = dict(training_seeds=[42, 123, 456], window_widths=[5, 10, 20], permutation_replicates=3,
                        pairs=[dict(id='amt_mt', parent='P', variant='V')])
            runs = [dict(dataset='SHD', model=model, seed=seed, checkpoint_sha256=model + str(seed),
                         config={'dataset': {'batch_size': 2}}) for seed in spec['training_seeds'] for model in ['P', 'V']]
            m = dict(specification=spec, runs=runs, blocked={}, cohorts={'SHD': [0, 1, 2, 3]},
                     dataset_metadata={'SHD': {'validation_indices': [0, 1, 2, 3]}})
            contract = dict(source_files_sha256={})
            anchors, legacy_rows = [], []
            for run in runs:
                cp, cv, pp, vp = toy_inputs(run['seed'])
                clean = cp if run['model'] == 'P' else cv
                anchors.append(dict(dataset='SHD', model=run['model'], seed=run['seed'],
                    loss=f.statistics.mean(r['loss'] for r in clean), accuracy=sum(r['correct_steps'] for r in clean) / 4))
                for condition in f.conditions(spec):
                    values = copy.deepcopy(clean if condition[0] == 'clean' else
                                           (pp if run['model'] == 'P' else vp)[condition[2]])
                    if condition[0] != 'clean':
                        for row in values:
                            row['transformation']['width'] = condition[1]
                    for number in range(2):
                        path = f.score_path(source, run, condition, number)
                        ids, wanted = f.batch_info(m, run, number)
                        rows = values[number * 2:number * 2 + 2]
                        record = dict(identity=dict(manifest_sha256=f.digest(m), checkpoint=run['checkpoint_sha256'],
                            original_batch_ids=ids, selected_ids=wanted, condition=condition),
                            value=rows, value_sha256=f.digest(rows))
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(record), encoding='utf-8')
                        contract['source_files_sha256'][f.portable_path(path)] = f.sha(path)
            (source / 'ANCHORS.json').write_text(json.dumps(anchors), encoding='utf-8')
            for seed in spec['training_seeds']:
                for width in spec['window_widths']:
                    inputs = copy.deepcopy(toy_inputs(seed))
                    for reps in inputs[2:]:
                        for rows in reps:
                            for row in rows:
                                row['transformation']['width'] = width
                    legacy_rows.extend(f.pair_rows('SHD', seed, 'amt_mt', width, *inputs))
            f.finalize(source, output, m, contract)
            self.assertEqual(f.read(output / 'SUMMARY.json')['results'], summary(legacy_rows, spec))
            index = f.read(output / 'PAIRED_RESULTS_INDEX.json')
            self.assertEqual(index['rows'], 108)
            self.assertEqual(sum(1 for s in index['shards'] for _ in f.read_rows(f.local_path(s['path']))), 108)


if __name__ == '__main__':
    unittest.main()
