"""CPU paired TextVQA contrasts and numerical audit of a completed localization ZIP."""
import argparse,collections,hashlib,json,zipfile
from pathlib import Path
import numpy as np

def run(path,out):
    with zipfile.ZipFile(path) as z:
        report=json.loads(z.read('report.json'));assert report['status']=='passed' and report['completed_predictions']==1600
        qs=[json.loads(z.read(n)) for n in sorted(z.namelist()) if n.startswith('questions/') and n.endswith('.json')]
        assert len(qs)==100 and len({q['case']['image_id'] for q in qs})==100
        rng=np.random.default_rng(590);draws=rng.integers(100,size=(10000,100));contrasts={}
        definitions={
          'basis_stage2_given_E_input':{'loc-error-native':1,'grid-error-drop-s0':-1},
          'basis_stage1_given_E_decoder':{'loc-native-error':1,'grid-error-drop-s0':-1},
          'basis_both_vs_neither':{'loc-native-drop':1,'grid-error-drop-s0':-1},
          'basis_stage2_given_native_input':{'loc-native-drop':1,'loc-native-error':-1},
          'basis_interaction':{'loc-native-drop':1,'loc-native-error':-1,'loc-error-native':-1,'grid-error-drop-s0':1},
          'native_merge_minus_native_drop':{'native-96-to-32':1,'loc-native-drop':-1},
          'fps_only_minus_native':{'loc-fps-fps':1,'native-96-to-32':-1},
          'E_to_fps_minus_E_to_E':{'loc-error-fps':1,'grid-error-drop-s0':-1},
          'E_to_detail_minus_E_to_E':{'loc-error-detail':1,'grid-error-drop-s0':-1},
          'detail_to_E_minus_detail_to_detail':{'loc-detail-error':1,'grid-detail-drop-s0':-1},
          'E_to_random_mean_minus_E_to_E':{**{f'loc-error-random-s{s}':.2 for s in [101,202,303,404,505]},'grid-error-drop-s0':-1}}
        for name,weights in definitions.items():
            delta=np.array([sum(w*q['arms'][arm]['scoring']['consensus_score'] for arm,w in weights.items()) for q in qs])
            contrasts[name]=dict(difference_pp=float(100*delta.mean()),image_bootstrap95_pp=(100*np.quantile(delta[draws].mean(1),[.025,.975])).tolist())
        audit={};source_checks=[]
        for q in qs:
            assert q['status']=='passed'
            source_checks.extend(q['shared_stage1_controls'].values())
            for arm,a in q['arms'].items():
                for stage,key in [('input','input_encoding'),('decoder','decoder')]:
                    d=(a.get(key) or {}).get('fps_numerics')
                    if not d:continue
                    row=audit.setdefault(arm+'/'+stage,dict(cases=0,overflow_cases=0,nan_evaluations=0,sequence_mismatches=0,max_ties=0,max_remaining_cap=0))
                    row['cases']+=1;row['overflow_cases']+=d['any_inf'];row['nan_evaluations']+=d['nan_distance_evaluations'];row['sequence_mismatches']+=not d['selected_sequence_reproduced']
                    row['max_ties']=max(row['max_ties'],d['max_argmax_tie_count']);row['max_remaining_cap']=max(row['max_remaining_cap'],d['max_remaining_1e10_count'])
        assert len(source_checks)==1000 and all(x['status']=='passed' for x in source_checks)
        result=dict(source=str(path),summary=report['summary'],contrasts=contrasts,fps_audit=audit,historical_drift=report['historical_changed_ids'],limits='100 fixed TextVQA images;10000 image bootstrap draws seed590; tested random seeds averaged per image. Exploratory intervals not multiplicity-adjusted. FP16-overflow FPS arms cannot be interpreted as ideal FP32 diversity selection; no precision intervention performed.')
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(contrasts,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.archive,a.output)
