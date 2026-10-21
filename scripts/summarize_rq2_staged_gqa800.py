"""Combine disjoint passed GQA image shards without Torch or tensor loading."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq1_protocol import sha256, write_json
from rq2_protocol import require, score_response
from rq2_staged_protocol import CONDITIONS
from rq2_staged_gqa_protocol import VERSION, PROFILE, MANIFEST_SHA256, STUDY_SHA256, load_profile, selected_ids, adapt_manifest, summarize, load_calibration_reference

def stream_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def read_archive(path):
    """Check every member's CRC/hash, then retain only small answer/statistic rows."""
    profile = load_profile()
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)), 'Duplicate ZIP member')
        inventory = json.loads(z.read('artifact-sha256.json'))
        digests = {}
        require(set(names) == set(inventory) | {'artifact-sha256.json'}, 'Incomplete artifact inventory')
        for name in names:
            rel = Path(name)
            require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe ZIP member')
            h = hashlib.sha256()
            with z.open(name) as stream:
                for b in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    h.update(b)
            digests[name] = h.hexdigest()
        require(all((digests[n] == v for n, v in inventory.items())), 'Archive artifact hash mismatch')
        r = json.loads(z.read('report.json'))
        require(r.get('version') == VERSION and r.get('status') == 'passed' and r.get('execution_passed') and (r.get('profile_sha256') == sha256(PROFILE)) and (r.get('profile') == profile) and (r.get('conditions') == CONDITIONS), 'Only passed matching GQA development shards can be combined')
        require(digests['evidence/inputs/development/manifest.json'] == MANIFEST_SHA256 and digests['evidence/inputs/study.json'] == STUDY_SHA256, 'Input study/manifest changed')
        for field, prefix in (('evidence_sha256', 'evidence/'), ('question_sha256', ''), ('feature_sha256', ''), ('encoding_sha256', '')):
            require(all((digests.get(prefix + n) == v for n, v in r[field].items())), 'Recorded evidence hash changed: ' + field)
        require(all((digests.get('evidence/sources/' + n) == v for n, v in r['measurement_sources_sha256'].items())), 'Archived measurement sources changed')
        calibrated = load_calibration_reference()['measurement_sources_sha256']
        require(all((r['measurement_sources_sha256'].get(n) == v for n, v in calibrated.items())), 'Calibrated operator sources changed')
        native = json.loads(z.read('evidence/sources/research/provenance/original_llava_reference.json'))
        require(r['implementation']['native_matches_reference'] and all((r['source_sha256'].get(n) == v for n, v in native['native_source_sha256'].items())) and (r['lock_sha256'] == calibrated['environments/original/uv.lock']), 'Native source/lock changed')
        proof = json.loads(z.read('evidence/notebook/prerequisites.json'))
        require(proof == r['prerequisite'] and proof.get('status') == 'passed' and (proof.get('skipped') == 0) and (proof.get('tests_run', 0) > 26) and (proof['setup_log_sha256'] == digests['evidence/notebook/setup.log']) and (proof['runtime_report_sha256'] == digests['evidence/notebook/runtime.json']) and all((digests.get('evidence/sources/' + n) == v for n, v in proof['source_sha256'].items())), 'Prerequisite evidence changed')
        cases = {c['case_id']: c for c in adapt_manifest(json.loads(z.read('evidence/inputs/development/manifest.json')))['cases']}
        ids = selected_ids(profile, r['image_shards'])
        require(r['expected_case_ids'] == ids and r['progress']['completed_primary_predictions'] == 9 * len(ids) and ([q['case_id'] for q in r['questions']] == ids), 'Incomplete requested image shards')
        fields = ('case_id', 'condition', 'image_id', 'response', 'scoring', 'types', 'seed', 'random_seed', 'input_tokens', 'decoder_tokens', 'layer', 'control_only', 'generated_tokens')
        rows = []
        seen = set()
        with z.open('predictions.jsonl') as stream:
            for line in io.TextIOWrapper(stream, encoding='utf-8'):
                row = json.loads(line)
                cid, condition = (row['case_id'], row['condition'])
                require(cid in ids and condition in CONDITIONS and ((cid, condition) not in seen) and (row['control_only'] is False), 'Duplicate, unknown or control prediction')
                case = cases[cid]
                require(row['image_id'] == case['image_id'] and row['image_sha256'] == case['image_sha256'] and (row['types'] == case['types']) and (row['seed'] == case['compression_seed']) and (row['scoring'] == score_response(case, row['response'])) and all((row[k] == v for k, v in CONDITIONS[condition].items())), 'Prediction identity/scoring differs')
                require(digests['images/' + case['image_sha256'] + '.jpg'] == case['image_sha256'], 'Source image bytes changed')
                seen.add((cid, condition))
                rows.append({k: row[k] for k in fields})
        require(len(rows) == 9 * len(ids), 'Missing primary rows')
        checkpoints = {k: {name: v for name, v in plan.items() if name != 'missing_bytes'} for k, plan in r['checkpoints'].items()}
        identity = dict(measurement_sources_sha256=r['measurement_sources_sha256'], lock_sha256=r['lock_sha256'], checkpoints=checkpoints, packages=r['packages'])
        return (rows, identity, dict(file=str(Path(path).resolve()), sha256=stream_hash(path), shards=r['image_shards'], questions=len(ids), members_verified=len(names), commit=r['commit']))

def combine(paths, output):
    output = Path(output)
    require(not output.exists(), 'Preserve existing analysis; choose a new output directory')
    rows = []
    sources = []
    identity = None
    seen = set()
    for path in paths:
        current, contract, source = read_archive(path)
        require(identity is None or identity == contract, 'Shards used different measurement code/runtime/checkpoints')
        identity = contract
        ids = {r['case_id'] for r in current}
        require(not seen & ids, 'Overlapping questions/shards: supply one passed export per image shard')
        seen.update(ids)
        rows.extend(current)
        sources.append(source)
    require(bool(rows), 'Supply at least one passed export')
    p = load_profile()
    rank = {c: i for i, c in enumerate(p['case_ids'])}
    arms = {c: i for i, c in enumerate(CONDITIONS)}
    rows.sort(key=lambda r: (rank[r['case_id']], arms[r['condition']]))
    result = summarize(rows, intervals=True)
    out = dict(version=VERSION + '-saved-summary', status='complete' if result['full_development800'] else 'partial-coverage', source_archives=sources, profile_sha256=sha256(PROFILE), identity=identity, summaries=result, note='Derived CPU-only analysis of disjoint passed shards. Raw ZIPs unchanged; tensor files only stream-hashed. Partial coverage is not full800.')
    output.mkdir(parents=True)
    write_json(output / 'summary.json', out)
    with (output / 'predictions-compact.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (output / 'answers.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ('case_id', 'image_id', 'condition', 'response', 'correct', 'seed', 'random_seed', 'input_tokens', 'decoder_tokens')
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({**{k: r[k] for k in fields if k != 'correct'}, 'correct': r['scoring']['correct']})
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    out = combine(args.archives, args.output)
    print(out['status'], out['summaries']['questions'], 'questions; summary:', args.output / 'summary.json')
if __name__ == '__main__':
    main()
