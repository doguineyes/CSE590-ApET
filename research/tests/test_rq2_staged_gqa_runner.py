"""Synthetic GQA orchestration/packaging and shard checks; no model/Torch locally."""
import contextlib
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
import run_rq2_staged_gqa800 as runner
import summarize_rq2_staged_gqa800 as aggregate
from rq2_staged_protocol import CONDITIONS, IDENTITY_CONDITION, specification
from rq2_staged_gqa_protocol import load_profile, selected_ids, summarize
from rq2_protocol import score_response
from research.tests.test_rq2_staged_textvqa_runner import bank_fixture
from research.tests.test_rq2_staged_gqa_protocol import primary_rows

def data_fixture():
    p = load_profile()
    cases = []
    for cid in p['case_ids']:
        image = p['case_image_ids'][cid]
        i = p['image_ids'].index(image)
        cases.append(dict(case_id=cid, image_id=image, image_sha256=hashlib.sha256(image.encode()).hexdigest(), question_id=cid.split(':')[1], question='Synthetic question?', generation_question='Synthetic question?\nInstruction', canonical_target='yes', gold_answers=['yes'], compression_seed=1000 + i, types=dict(structural='verify', semantic='obj', detailed='exist'), known_repeatability_case=cid == 'gqa:2058428'))
    return dict(cases=cases, images={})

class GQAStagedRunnerTests(unittest.TestCase):

    def test_full800_shares906_input_captures_and_keeps7200_primary_rows(self):
        import numpy as np
        from PIL import Image
        data = data_fixture()
        p = load_profile()
        banks = {}
        members = {}

        def capture(model, pixels, seed, condition, key):
            if key not in banks:
                banks[key] = bank_fixture(key)
            return (copy.deepcopy(banks[key][condition]), dict(obs=banks[key][condition]))

        def generate(tokenizer, model, processor, image, case, pixels, condition, snapshot, seed, *, decoder_replay=None, probe=False):
            members.setdefault((case['image_id'], condition), set()).add(id(snapshot))
            tokens = 4 if probe else 2
            r = dict(case_id=case['case_id'], image_sha256=case['image_sha256'], condition=condition, **specification(condition), seed=seed, response='no', scoring=score_response(case, 'no'), input_encoding=snapshot['obs'], generation=dict(prompt=case['generation_question'], generated_tokens=tokens), generation_tokens=dict(generated_token_ids=list(range(tokens))), control_only=bool(probe or decoder_replay is not None or condition == IDENTITY_CONDITION))
            return (r, None if condition == 'baseline576' else ('synthetic-layer', condition))

        def raw(tokenizer, model, processor, image, case, condition, seed):
            return dict(case_id=case['case_id'], condition=condition, response='no', scoring=score_response(case, 'no'), generation_tokens=dict(generated_token_ids=[0, 1]))

        def save(args, report, key, *rest):
            name = 'features/' + key + '-synthetic.pt'
            path = args.output / name
            path.write_text('synthetic fixture only')
            report['feature_sha256'][name] = runner.sha256(path)
            return dict(file=name, sha256=runner.sha256(path))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'run'
            root.mkdir()
            args = SimpleNamespace(output=root, data_manifest=root / 'input-manifest.json', image_shards=[0, 1, 2, 3])
            report = dict(status='passed', version=runner.VERSION)
            model = SimpleNamespace(model=SimpleNamespace(layers=[None] * 32))
            with patch.object(runner, 'save_image', return_value=Image.new('RGB', (2, 2))), patch.object(runner, 'prepare_pixels', return_value=(object(), np.zeros((2, 2, 3), dtype=np.uint8))), patch('rq2_staged_native.capture_input', side_effect=capture) as cap, patch('rq2_staged_native.generate_condition', side_effect=generate) as gen, patch('rq2_staged_native.generate_raw_native', side_effect=raw) as fresh, patch.object(runner, 'save_input_bank', side_effect=save) as save_input, patch.object(runner, 'save_decoder', side_effect=lambda a, r, c, condition, s: save(a, r, c + '-' + condition)) as save_layer, contextlib.redirect_stdout(io.StringIO()):
                runner.run_panel(args, report, p, data, (None, model, None))
                runner.finish_run(root, report, time.perf_counter())
            self.assertEqual(cap.call_count, 906)
            self.assertEqual(gen.call_count, 7228)
            self.assertEqual(fresh.call_count, 4)
            self.assertEqual(save_input.call_count, 2)
            self.assertEqual(save_layer.call_count, 16)
            self.assertTrue(all((len(s) == 1 for s in members.values())))
            self.assertEqual(report['progress']['completed_images'], 100)
            self.assertEqual(report['progress']['completed_questions'], 800)
            self.assertEqual(report['progress']['completed_primary_predictions'], 7200)
            self.assertTrue(report['summaries']['full_development800'])
            self.assertEqual(report['summaries']['baseline_correct'], 0)
            self.assertEqual(len(report['encoding_sha256']), 100)
            self.assertEqual(len(report['feature_sha256']), 18)
            with zipfile.ZipFile(root.with_suffix('.zip')) as z:
                self.assertIsNone(z.testzip())
                hashes = json.loads(z.read('artifact-sha256.json'))
                self.assertTrue(all((hashlib.sha256(z.read(n)).hexdigest() == v for n, v in hashes.items())))
                self.assertEqual(len(z.read('predictions.jsonl').splitlines()), 7200)
                self.assertEqual(len(list(csv.DictReader(io.StringIO(z.read('answers.csv').decode())))), 7200)

    def test_interruption_excludes_an_incomplete_image_but_preserves_raw_questions(self):
        p = load_profile()
        cases = {c['case_id']: c for c in data_fixture()['cases']}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'partial'
            (root / 'questions').mkdir(parents=True)
            for i, cid in enumerate(p['case_ids'][:16]):
                rows = primary_rows([cid]) if i < 15 else primary_rows([cid])[:2]
                arms = {r['condition']: {**r, **specification(r['condition']), 'seed': 1} for r in rows}
                q = dict(case_id=cid, case=cases[cid], status='passed' if i < 15 else 'failed', arms=arms)
                (root / 'questions' / (cid.replace(':', '-') + '.json')).write_text(json.dumps(q))
            report = dict(status='interrupted', execution_passed=False)
            with contextlib.redirect_stdout(io.StringIO()):
                runner.finish_run(root, report, time.perf_counter())
            self.assertEqual(report['summaries']['questions'], 8)
            self.assertEqual(report['summary_coverage']['passed_questions'], 15)
            with (root / 'answers.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 137)

    def test_runtime_failure_archived_without_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / 'failed'
            with patch.object(runner, 'provenance', return_value={}), patch.object(runner, 'validate_runtime', side_effect=RuntimeError('synthetic failure')), patch.object(runner, 'load_model') as load, patch.object(runner.signal, 'signal'), patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
                code = runner.main(['--data-manifest', str(root / 'data/manifest.json'), '--output', str(out), '--run-gqa', '--setup-log', str(root / 'setup'), '--runtime-report', str(root / 'runtime'), '--prerequisite-report', str(root / 'proof')])
            load.assert_not_called()
            self.assertEqual(code, 1)
            with zipfile.ZipFile(out.with_suffix('.zip')) as z:
                self.assertEqual(json.loads(z.read('report.json'))['status'], 'failed')

    def test_prerequisites_bound_to_sources_runtime_and_zero_skips(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = SimpleNamespace(setup_log=root / 'setup', runtime_report=root / 'runtime', prerequisite_report=root / 'proof')
            args.setup_log.write_text('tests')
            args.runtime_report.write_text('runtime')
            proof = dict(version=runner.VERSION, status='passed', tests_run=40, skipped=0, test_modules=list(runner.TESTS), source_sha256={'x': 'same'}, setup_log_sha256=runner.sha256(args.setup_log), runtime_report_sha256=runner.sha256(args.runtime_report))
            args.prerequisite_report.write_text(json.dumps(proof))
            with patch.object(runner, 'prerequisite_signature', return_value={'x': 'same'}):
                self.assertEqual(runner.verify_prerequisites(args)['status'], 'passed')
                args.runtime_report.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'changed since prerequisites'):
                    runner.verify_prerequisites(args)
                proof['skipped'] = 1
                args.prerequisite_report.write_text(json.dumps(proof))
                with self.assertRaisesRegex(ValueError, 'zero skips'):
                    runner.verify_prerequisites(args)

    def test_notebook_idle_and_helper_refuses_pretrained_without_prerequisites(self):
        n = json.loads((REPO / 'research/notebooks/kaggle_rq2_staged_gqa800.ipynb').read_text())
        code = [''.join(c['source']) for c in n['cells'] if c['cell_type'] == 'code']
        for i, c in enumerate(code):
            compile(c, f'cell-{i}', 'exec')
        self.assertIn('GQA_SHARDS = [0, 1, 2, 3]', code[0])
        with patch('subprocess.run') as run:
            with self.assertRaisesRegex(RuntimeError, 'RUN_STAGED_GQA'):
                exec(code[0], {})
        run.assert_not_called()
        helper = next((c for c in code if c.startswith('def verify_rq2_staged_gqa_prerequisites(')))
        namespace = {'rq2_staged_gqa_prerequisites_passed': True}
        exec(helper, namespace)
        self.assertFalse(namespace['rq2_staged_gqa_prerequisites_passed'])
        with self.assertRaisesRegex(RuntimeError, 'prerequisites'):
            namespace['run_rq2_staged_gqa800']()

    def test_disjoint_shards_combine_without_tripling_samples_and_overlap_rejected(self):
        p = load_profile()
        identity = {'code': 'same'}

        def read(path):
            shard = int(str(path))
            rows = primary_rows(selected_ids(p, [shard]))
            rows = [{**r, **specification(r['condition']), 'seed': 1} for r in rows]
            return (rows, identity, dict(file=str(path), shards=[shard]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(aggregate, 'read_archive', side_effect=read):
                out = aggregate.combine([0, 1, 2, 3], root / 'all')
                self.assertEqual(out['status'], 'complete')
                self.assertEqual(out['summaries']['questions'], 800)
                self.assertEqual(out['summaries']['paired_comparisons']['pure-random-96-to-32-seed-mean minus native-96-to-32']['images'], 100)
                partial = aggregate.combine([0], root / 'one')
                self.assertEqual(partial['status'], 'partial-coverage')
                with self.assertRaisesRegex(ValueError, 'Overlapping'):
                    aggregate.combine([0, 0], root / 'bad')

    def test_archive_reader_checks_real_inventory_sources_and_rescores_synthetic_answers(self):
        p = load_profile()
        data = data_fixture()
        cases = {c['case_id']: c for c in data['cases']}
        raw_images = {}
        image_bytes = {}
        for c in cases.values():
            payload = ('synthetic image ' + c['image_id']).encode()
            digest = hashlib.sha256(payload).hexdigest()
            c['image_sha256'] = digest
            image_bytes[c['image_id']] = payload
            raw_images[c['image_id']] = dict(sha256=digest, compression_seed=c['compression_seed'], image_size=[2, 2], file='unused.jpg')
        manifest = dict(images=raw_images, questions=[dict(id=c['question_id'], imageId=c['image_id'], question=c['question'], answer=c['canonical_target'], types=c['types']) for c in cases.values()])
        encode = lambda v: json.dumps(v).encode()
        files = {'evidence/inputs/development/manifest.json': encode(manifest), 'evidence/inputs/study.json': b'synthetic study fixture', 'evidence/notebook/setup.log': b'synthetic test-only setup', 'evidence/notebook/runtime.json': b'{}'}
        signature = runner.prerequisite_signature()
        measurement = runner.measurement_hashes()
        for name in signature:
            files['evidence/sources/' + name] = (REPO / name).read_bytes()
        proof = dict(status='passed', skipped=0, tests_run=39, source_sha256=signature, setup_log_sha256=hashlib.sha256(files['evidence/notebook/setup.log']).hexdigest(), runtime_report_sha256=hashlib.sha256(files['evidence/notebook/runtime.json']).hexdigest())
        files['evidence/notebook/prerequisites.json'] = encode(proof)
        ids = selected_ids(p, [0])
        rows = []
        for cid in ids:
            c = cases[cid]
            for condition in CONDITIONS:
                rows.append(dict(case_id=cid, condition=condition, **specification(condition), image_id=c['image_id'], image_sha256=c['image_sha256'], seed=c['compression_seed'], types=c['types'], response='yes', scoring=score_response(c, 'yes'), control_only=False, generated_tokens=2))
            files['questions/' + cid.replace(':', '-') + '.json'] = encode(dict(case=c, status='passed'))
            files['images/' + c['image_sha256'] + '.jpg'] = image_bytes[c['image_id']]
        files['predictions.jsonl'] = b''.join((encode(r) + b'\n' for r in rows))
        digest = lambda b: hashlib.sha256(b).hexdigest()
        native = json.loads((REPO / 'research/provenance/original_llava_reference.json').read_text())
        report = dict(version=runner.VERSION, status='passed', execution_passed=True, profile=p, profile_sha256=runner.sha256(runner.PROFILE), conditions=CONDITIONS, image_shards=[0], expected_case_ids=ids, progress=dict(completed_primary_predictions=1800), questions=[dict(case_id=c, status='passed') for c in ids], feature_sha256={}, encoding_sha256={}, question_sha256={n: digest(b) for n, b in files.items() if n.startswith('questions/')}, evidence_sha256={n[len('evidence/'):]: digest(b) for n, b in files.items() if n.startswith('evidence/')}, measurement_sources_sha256=measurement, prerequisite=proof, implementation=dict(native_matches_reference=True), source_sha256=native['native_source_sha256'], lock_sha256=measurement['environments/original/uv.lock'], checkpoints={}, packages={'fixture': 'no model inference'}, commit='synthetic')
        files['report.json'] = encode(report)

        def write(path, parts):
            parts = {**parts, 'artifact-sha256.json': encode({n: digest(b) for n, b in parts.items()})}
            with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED) as z:
                for n, b in parts.items():
                    z.writestr(n, b)
        with tempfile.TemporaryDirectory() as folder, patch.object(aggregate, 'MANIFEST_SHA256', digest(files['evidence/inputs/development/manifest.json'])), patch.object(aggregate, 'STUDY_SHA256', digest(files['evidence/inputs/study.json'])):
            root = Path(folder)
            write(root / 'valid.zip', files)
            admitted, _, _ = aggregate.read_archive(root / 'valid.zip')
            self.assertEqual(len(admitted), 1800)
            bad = copy.deepcopy(rows)
            bad[0]['scoring']['correct'] = False
            changed = {**files, 'predictions.jsonl': b''.join((encode(r) + b'\n' for r in bad))}
            write(root / 'bad-score.zip', changed)
            with self.assertRaisesRegex(ValueError, 'Prediction identity/scoring'):
                aggregate.read_archive(root / 'bad-score.zip')
if __name__ == '__main__':
    unittest.main()
