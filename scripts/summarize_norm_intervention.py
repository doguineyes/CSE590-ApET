"""Paired image-bootstrap contrasts from a completed norm-intervention ZIP."""
import argparse,json,zipfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from norm_intervention_protocol import VERSION,SEEDS

def analyze(path):
    with zipfile.ZipFile(path) as z:
        report=json.loads(z.read('report.json'))
        if report['status']!='passed' or report['version']!=VERSION:raise ValueError('Require passed norm run')
        qs=[json.loads(z.read(n)) for n in sorted(z.namelist()) if n.startswith('questions/') and n.endswith('.json')]
        groups=defaultdict(list)
        for q in qs:groups[q['case']['image_sha256']].append(q)
        rng=np.random.default_rng(590);result={}
        contrasts=[(f'loc-norm-{f}',[f'loc-norm-{f}c-s{s}' for s in SEEDS]) for f in ('a1','a2','b1','c1')]
        contrasts += [('loc-norm-a-fp32',['loc-native-drop']),('loc-norm-b-fp32',['loc-fps-fps']),('loc-norm-c2',['loc-norm-c1'])]
        def score(q,n):return q['arms'][n]['scoring']['consensus_score']
        for arm,controls in contrasts:
            values=np.array([np.mean([score(q,arm)-np.mean([score(q,c) for c in controls]) for q in group]) for group in groups.values()])
            boot=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)*100
            result[arm]=dict(controls=controls,image_mean_difference_pp=float(values.mean()*100),CI95_pp=np.quantile(boot,[.025,.975]).tolist())
        strata={}
        for flag in (False,True):
            chosen=[]
            for q in qs:
                a=q['arms']['loc-native-drop'];norms=a['state_dump']['norms'];top=int(np.argmax(norms))
                patch=a['input_encoding']['direct_source_original_positions'][top]
                props=json.loads(z.read('properties/'+q['case']['image_sha256']+'.json'))
                if bool(props['padding'][patch]==1)==flag:chosen.append(q)
            strata[str(flag)]={'questions':len(chosen),'means':{n:float(np.mean([score(q,n) for q in chosen])) if chosen else None for n in report['conditions']}}
        return dict(source=str(path),status='passed',images=len(groups),questions=len(qs),paired_image_contrasts=result,top_source_fully_padding=strata,
                    limits='Exploratory intervals; no multiplicity adjustment. C2 versus C1 is not count-matched. Padding denotes source position, not context-free content.',historical_changed_ids=report.get('historical_changed_ids'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('zip',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps(analyze(a.zip),indent=2)+'\n')
