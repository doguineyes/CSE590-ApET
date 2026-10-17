"""Freeze the bounded RQ1 validity panel from archived input-stage results; CPU only."""

import argparse
import json
from pathlib import Path

from render_rq1_gallery import patch_regions, photo_bounds
from rq1_protocol import SOURCES, TEXTVQA_SELECTION, average_ranks, sha256, token_order, write_json
from rq1_validation import (PROFILE, VERSION, CONDITIONS, METHODS, METHOD_DEFINITIONS,
                            SHORTCUT_CASE_IDS, PERTURBATION_CASE_IDS)

REPO=Path(__file__).resolve().parents[1]
RUNS={'gqa':'rq1-gqa-input-pilot-v2-20261016T143422920388Z',
      'mmstar':'rq1-mmstar-twelvecases-v2-20261016T215926569196Z',
      'textvqa':'rq1-textvqa-input-pilot-v1-20261016T204945039325Z'}
CASES=PERTURBATION_CASE_IDS


def prepare(results, output):
    if output.exists():raise ValueError('Profile exists; choose a new version rather than overwrite')
    synthesis=json.loads((REPO/'research/protocols/rq1_saved_analysis_v1.json').read_text())
    sources={};manifests={}
    for dataset,stem in RUNS.items():
        source=results/stem
        digest=sha256(source.with_suffix('.zip'))
        if digest != synthesis['sources'][dataset]['archive_sha256']:raise ValueError('Archive differs')
        manifests[dataset]=json.loads((source/'evidence/manifest.json').read_text())
        sources[dataset]={'run':stem,'archive_sha256':digest,'report_sha256':sha256(source/'report.json'),
            'manifest_sha256':SOURCES[dataset][1]}
    profile={'version':VERSION,'created_local':'2026-10-17',
        'purpose':'Targeted RQ1 case-grounding and intervention-proxy validation; not a new compression method or independent test set',
        'selection_status':'Post-hoc representative questions and diagnostic patches fixed before new validation outcomes',
        'inputs':sources,'seed':590,'control_tolerance':.001,'methods':list(METHODS),'method_definitions':METHOD_DEFINITIONS,
        'textvqa_selection_sha256':sha256(TEXTVQA_SELECTION),
        'shortcuts':{'case_ids':SHORTCUT_CASE_IDS,
            'conditions':list(CONDITIONS),'shifts':[0,1,2,3],'predictions':72,'reference_repeats':6,
            'blank':'336x336 RGB filled with int(CLIP image_mean*255); normal vision encoder, not zero embeddings',
            'question_only':'Remove image placeholder and image tensor; preserve question/options; sequence length differs',
            'permutation':'Rotate only original option content; identity is byte-for-byte original question; map predictions back to original semantic choices'},
        'patch_selection':'16 seeded random indices first; then 8 highest positive archived I among non-basis photo/mixed patches not already selected; then 8 highest signed I within the low-E non-basis third, photo/mixed, excluding earlier picks (this last stratum can include negative effects). Diagnostic strata are post-hoc and summarized separately.',
        'perturbations':[]}
    for cid in CASES:
        dataset,qid=cid.split(':');source=results/RUNS[dataset]
        path=source/'questions'/(qid+'.json');case=json.loads(path.read_text())
        feature=json.loads((source/'features'/(case['image_sha256']+'.json')).read_text())
        basis=set(feature['diagnostics']['input']['basis_indices'])
        regions=patch_regions(photo_bounds(manifests[dataset]['images'][case['image_id']]['size']))
        lookup={t['token_index']:t for t in case['tokens']}
        nb=[t for t in case['tokens'] if t['token_index'] not in basis]
        ranks=dict(zip([t['token_index'] for t in nb],average_ranks([t['reconstruction_error'] for t in nb])))
        chosen=[(i,'random') for i in token_order(case['image_sha256'])[:16]]
        for stratum in ['high_I','low_E_high_I']:
            used={i for i,_ in chosen}
            pool=[t for t in nb if t['token_index'] not in used and regions[t['token_index']]['region']!='padding'
                  and ((stratum=='high_I' and t['intervention_score']>0)
                       or (stratum=='low_E_high_I' and ranks[t['token_index']]<=len(nb)/3))]
            pool.sort(key=lambda t:(-t['intervention_score'],t['token_index']))
            if len(pool)<8:raise ValueError('Insufficient diagnostic patches: '+cid+' '+stratum)
            chosen += [(t['token_index'],stratum) for t in pool[:8]]
        profile['perturbations'].append({
            'case_id':cid,'image_id':case['image_id'],'image_sha256':case['image_sha256'],
            'canonical_target':case['canonical_target'],'measurement_target':case['measurement_target'],
            'target_token_ids':case['target_token_ids'],'seed':case['seed'],
            'source_question_sha256':sha256(path),'archived_reference':case['controls']['reference'],
            'archived_clip_sha256':case['clip_sha256'],'archived_projected_sha256':case['projected_sha256'],
            'archived_role':case['outcomes']['input-only'],
            'patches':[{'token_index':i,'stratum':stratum,'region':regions[i]['region'],
                'photo_fraction':regions[i]['photo_fraction'],'fps_basis_source':i in basis,
                'archived_E':lookup[i]['reconstruction_error'],'archived_I':lookup[i]['intervention_score']} for i,stratum in chosen]})
    write_json(output,profile)
    print('Frozen six questions × 32 patches, three interventions; 72 option/input trials + 6 repeats:',output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=REPO/'research/results')
    p.add_argument('--output',type=Path,default=PROFILE);a=p.parse_args();prepare(a.results,a.output)
