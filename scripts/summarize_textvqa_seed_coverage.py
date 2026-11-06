"""CPU-only integrity and descriptive review; does not infer OCR coverage."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import zipfile


def analyze(archive):
    with zipfile.ZipFile(archive) as z:
        hashes=json.loads(z.read('artifact-sha256.json'))
        for name, expected in hashes.items():
            with z.open(name) as f:
                digest=hashlib.sha256()
                for block in iter(lambda:f.read(1024*1024),b''): digest.update(block)
            if digest.hexdigest()!=expected: raise ValueError('Checksum mismatch: '+name)
        report=json.loads(z.read('report.json'))
        questions=[json.loads(z.read(n)) for n in sorted(z.namelist()) if n.startswith('questions/') and n.endswith('.json')]
        if report['status']!='passed' or len(questions)!=100: raise ValueError('Incomplete full run')
        names=[f'pure-random-96-to-32-s{s}' for s in report['seeds']]
        expected={'baseline576','native-96-to-32',*names}
        records=[]; controls={}; drift={}
        for q in questions:
            if q['status']!='passed' or set(q['arms'])!=expected: raise ValueError('Incomplete question')
            for name,control in q['controls'].items():
                controls[name]=controls.get(name,0)+1
                if name!='new_seed_cache_probe' and control['status']!='passed': raise ValueError('Failed control')
                if name=='new_seed_cache_probe' and control['generation']['generated_tokens']!=4: raise ValueError('Cache probe incomplete')
            if q['feature_dump']['file'] not in z.namelist(): raise ValueError('Missing features')
            a=q['arms']; baseline=a['baseline576']['scoring']; native=a['native-96-to-32']['scoring']
            rows=[a[n] for n in names]
            for row in rows:
                plan=row['decoder']['random_plan']; first=plan['input_original_positions']; compact=plan['decoder_compact_positions']
                if len(set(first))!=96 or len(set(compact))!=32 or plan['decoder_original_positions']!=[first[i] for i in compact]: raise ValueError('Mask geometry')
            scores=[r['scoring']['consensus_score'] for r in rows]
            correct=sum(r['scoring']['correct'] for r in rows)
            records.append(dict(case_id=q['case']['case_id'],question=q['case']['question'],
                baseline_score=baseline['consensus_score'],baseline_full=baseline['correct'],
                native_score=native['consensus_score'],native_full=native['correct'],
                random_mean=statistics.mean(scores),random_min=min(scores),random_max=max(scores),
                random_full_count=correct,random_distinct_answers=len({r['scoring']['prediction'] for r in rows}),
                native_minus_random=native['consensus_score']-statistics.mean(scores)))
            for name,d in q['historical_comparison'].items():
                if not d['generated_ids_equal']: drift.setdefault(q['case']['case_id'],{})[name]=d
        deltas=[r['native_minus_random'] for r in records]
        rng=random.Random(5901105)
        boot=sorted(statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(5000))
        seed_scores=[statistics.mean(q['arms'][n]['scoring']['consensus_score'] for q in questions) for n in names]
        native_correct=[r for r in records if r['native_full']]
        native_wrong=[r for r in records if not r['native_full']]
        fullbase={r['case_id'] for r in records if r['baseline_full']}
        fullnative={r['case_id'] for r in records if r['native_full']}
        summary=dict(status=report['status'],checksum_files_verified=len(hashes),questions=len(records),
            primary_predictions=report['completed_predictions'],wall_minutes=report['wall_seconds']/60,
            controls=controls,feature_files=len(records),
            baseline_score=statistics.mean(r['baseline_score'] for r in records),
            native_score=statistics.mean(r['native_score'] for r in records),
            random_seed_mean=statistics.mean(seed_scores),random_seed_sd=statistics.stdev(seed_scores),
            random_seed_min=min(seed_scores),random_seed_max=max(seed_scores),
            native_minus_random=statistics.mean(deltas),paired_image_bootstrap95=[boot[124],boot[4874]],
            interval_scope='5000 paired image resamples; conditional on these20 seeds and development cohort',
            random_full_credit_always=sum(r['random_full_count']==20 for r in records),
            random_full_credit_never=sum(r['random_full_count']==0 for r in records),
            random_full_credit_varies=sum(0<r['random_full_count']<20 for r in records),
            random_consensus_varies=sum(r['random_min']!=r['random_max'] for r in records),
            native_full_credit=len(native_correct),
            native_full_random_never=sum(r['random_full_count']==0 for r in native_correct),
            native_not_full_random_sometimes=sum(r['random_full_count']>0 for r in native_wrong),
            native_losses=sorted(fullbase-fullnative),native_gains=sorted(fullnative-fullbase),
            native_APR=len(fullbase&fullnative)/len(fullbase),historical_drift=drift,
            limits='No OCR/evidence coverage computed. Random seed rows are repeated questions. Whole native versus pure random; cannot attribute gap to E alone.')
        return summary,records

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    summary,records=analyze(args.archive);args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (args.output/'questions.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(json.dumps(summary,indent=2))
