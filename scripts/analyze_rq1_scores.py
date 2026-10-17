"""CPU reanalysis of archived answer-token scores; no inference or tensor loading.

Use the existing editor environment for the native TextVQA scorer's tqdm import.
Reads ZIPs directly, verifies provenance/arithmetic, writes a NEW derived folder.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import html
from itertools import combinations
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import zipfile

from analyze_rq1_saved import contrast
from analyze_rq1_validity import assert_close, verify_question, verify_score
from render_rq1_gallery import patch_regions, photo_bounds
from rq1_protocol import SOURCES, measurement_target, sha256, spearman, write_json
from rq1_validation import METHODS
from run_rq1_input import generation_score
from summarize_rq1_pilot import write_csv

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO/'research/protocols/rq1_score_reanalysis_v1.json'
METRICS = ('whole_log', 'first_log', 'first_probability')


def effects(reference, perturbed):
    verify_score(reference); verify_score(perturbed)
    a,b = reference['token_logprobs'], perturbed['token_logprobs']
    if len(a) != len(b): raise ValueError('Reference/perturbation answer length differs')
    if max([*a,*b]) > 1e-6: raise ValueError('A log probability is positive')
    return {'whole_log':reference['mean_logprob']-perturbed['mean_logprob'],
            'first_log':a[0]-b[0], 'first_probability':math.exp(a[0])-math.exp(b[0])}


def score_statistics(tokens, basis):
    result = {}
    nonbasis = [t for t in tokens if not t['fps_basis_source']]
    photo = [t for t in nonbasis if t['region']=='photo']
    for metric in METRICS:
        selected = [{**t,'intervention_score':t[metric]} for t in tokens]
        groups, fps, delta, median_delta = contrast(selected,basis)
        result[metric] = {'rho_all':spearman([t['E'] for t in tokens],[t[metric] for t in tokens]),
            'rho_nonbasis':spearman([t['E'] for t in nonbasis],[t[metric] for t in nonbasis]),
            'rho_photo_nonbasis':spearman([t['E'] for t in photo],[t[metric] for t in photo]),
            'high_minus_low_mean':delta,'high_minus_low_median':median_delta,
            'groups':groups,'basis':fps,'nonbasis_positions':len(nonbasis),'photo_nonbasis_positions':len(photo)}
    return result


def load_sources(results, plan):
    """Read canonical archives, never trust an edited extracted question file."""
    records, source_audits = [], []
    for dataset, info in plan['sources'].items():
        archive = results/(info['run']+'.zip')
        if sha256(archive) != info['archive_sha256']: raise ValueError('Canonical archive hash differs: '+dataset)
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None or len(z.namelist())!=len(set(z.namelist())): raise ValueError('Invalid ZIP')
            report = json.loads(z.read('report.json'))
            if report['status']!='passed' or report['version']!='rq1-input-zero-v2' or report['dataset']['name']!=dataset:
                raise ValueError('Unexpected original run status/version/dataset')
            for name,digest in report['evidence_sha256'].items():
                if hashlib.sha256(z.read('evidence/'+name)).hexdigest()!=digest: raise ValueError('Evidence hash mismatch')
            raw_manifest = z.read('evidence/manifest.json')
            if hashlib.sha256(raw_manifest).hexdigest()!=SOURCES[dataset][1]: raise ValueError('Accepted manifest differs')
            if hashlib.sha256(z.read('evidence/sources/environments/original/uv.lock')).hexdigest()!=report['lock_sha256']:
                raise ValueError('Lock hash mismatch')
            if not report['implementation']['native_matches_reference']: raise ValueError('Native implementation mismatch')
            for name,digest in report['implementation']['native_source_sha256'].items():
                if hashlib.sha256(z.read('evidence/sources/'+name)).hexdigest()!=digest: raise ValueError('Native file differs')
            manifest = json.loads(raw_manifest)
            lookup = {c['case_id']:c for c in manifest['cases']}
            paths = sorted(n for n in z.namelist() if n.startswith('questions/') and n.endswith('.json'))
            if len(paths)!=info['questions']: raise ValueError('Canonical question count differs')
            seen_images = set(); seen_cases = set(); drift = []
            for path in paths:
                case = json.loads(z.read(path)); original = lookup[case['case_id']]
                if case['case_id'] in seen_cases: raise ValueError('Duplicate canonical question')
                seen_cases.add(case['case_id'])
                for field in ['case_id','question_id','image_id','image_sha256','question','canonical_target']:
                    if case[field]!=original[field]: raise ValueError('Accepted case identity mismatch: '+field)
                if case['status']!='passed' or not case['controls']['passed']: raise ValueError('Question controls failed')
                if case['measurement_target']!=measurement_target(original)['text']: raise ValueError('Fixed target rendering mismatch')
                if case['prompt']!=case['free_generation']['prompt']: raise ValueError('Generation/scoring prompts differ')
                target_ids = case['target_token_ids']; n=len(target_ids)
                if (not n or case['target_expanded_start']+n!=case['expanded_sequence_length']
                        or case['image_expanded_start']+576>case['target_expanded_start']):
                    raise ValueError('Target/visual position mapping differs')
                if (not case['target_alignment']['passed'] or not case['target_alignment']['target_matches_generated_prefix']
                        or case['generation_tokens']['generated_token_ids'][:n]!=target_ids):
                    raise ValueError('Fixed target is not the fresh generated prefix')
                assert_close(case['free_generation']['scoring'],generation_score(dataset,original,case['free_generation']['response']),'Free baseline scoring')
                if not case['free_generation']['scoring']['correct']: raise ValueError('Baseline is no longer scored correct')
                feature = json.loads(z.read('features/'+case['image_sha256']+'.json'))
                for field in ['image_id','image_sha256','seed','clip_sha256','projected_sha256']:
                    if case[field]!=feature[field]: raise ValueError('Question/feature identity mismatch: '+field)
                image_meta=manifest['images'][case['image_id']]
                if case['image_sha256'] not in seen_images:
                    if hashlib.sha256(z.read('features/'+feature['tensor_file'])).hexdigest()!=feature['tensor_file_sha256']:
                        raise ValueError('Feature tensor file byte hash mismatch')
                    image_files=[p for p in z.namelist() if p.startswith('images/'+case['image_sha256']+'.')]
                    if len(image_files)!=1 or hashlib.sha256(z.read(image_files[0])).hexdigest()!=case['image_sha256']:
                        raise ValueError('Encoded original image hash mismatch')
                    seen_images.add(case['image_sha256'])
                native=feature['diagnostics']; basis=set(native['input']['basis_indices'])
                if len(basis)!=10 or len(native['input']['errors'])!=576: raise ValueError('Native source/basis geometry differs')
                reference=case['controls']['reference']; verify_score(reference)
                if len(reference['token_logprobs'])!=n: raise ValueError('Reference token count mismatch')
                max_drift=0.0
                for key in ['reference_repeat','noop','native_multimodal','reference_after']:
                    control=case['controls'][key]; verify_score(control)
                    if len(control['token_logprobs'])!=n: raise ValueError('Control answer length differs')
                    max_drift=max(max_drift,*(abs(a-b) for a,b in zip(reference['token_logprobs'],control['token_logprobs'])))
                if max_drift>.001: raise ValueError('Full-forward control drift')
                drift.append(max_drift)
                indices=[t['token_index'] for t in case['tokens']]
                if len(indices)!=576 or set(indices)!=set(range(576)) or indices!=case['token_execution_order']:
                    raise ValueError('Incomplete, duplicate or reordered patch sweep')
                regions=patch_regions(photo_bounds(image_meta['size']))
                tokens=[]
                for t in sorted(case['tokens'],key=lambda t:t['token_index']):
                    i=t['token_index']
                    if (t['patch_row'],t['patch_col'])!=divmod(i,24): raise ValueError('Patch grid coordinates differ')
                    assert_close(t['reconstruction_error'],native['input']['errors'][i],'Native E')
                    scores=effects(reference,t['perturbed_score'])
                    assert_close(t['intervention_score'],scores['whole_log'],'Archived whole-answer I')
                    if n==1 and scores['whole_log']!=scores['first_log']: raise ValueError('One-token score identity failed')
                    tokens.append({'token_index':i,'E':t['reconstruction_error'],
                        'reconstruction_error':t['reconstruction_error'],'fps_basis_source':i in basis,
                        'direct_original_retained':i in native['input_merge']['target_indices'],
                        'basis_replacement_slot':i in native['input_merge']['basis_slots'],
                        **regions[i],**scores,
                        'per_answer_token_log_deltas':[a-b for a,b in zip(reference['token_logprobs'],t['perturbed_score']['token_logprobs'])]})
                records.append({'dataset':dataset,'case':case,'tokens':tokens,'native':native,'basis':basis,
                    'question_record_sha256':hashlib.sha256(z.read(path)).hexdigest(),
                    'feature_metadata_sha256':hashlib.sha256(z.read('features/'+case['image_sha256']+'.json')).hexdigest(),
                    'max_control_drift':max_drift,'statistics':score_statistics(tokens,basis)})
            if seen_cases!=set(report['case_ids']): raise ValueError('Report/question IDs differ')
            source_audits.append({'dataset':dataset,'archive':str(archive),'archive_sha256':info['archive_sha256'],
                'questions':len(paths),'images':len(seen_images),'verified_evidence_files':len(report['evidence_sha256']),
                'verified_feature_file_bytes':len(seen_images),'max_control_drift':max(drift),
                'commit':report['commit'],'lock_sha256':report['lock_sha256']})
    if len({r['case']['case_id'] for r in records})!=len(records): raise ValueError('Canonical samples duplicated')
    if len({s['lock_sha256'] for s in source_audits})!=1: raise ValueError('Original dependency locks differ')
    return records, source_audits


def paired_scores(a,b):
    for field in ['image_id','image_sha256','seed','clip_sha256','projected_sha256']:
        if a['case'][field]!=b['case'][field]: raise ValueError('Same-image pair mismatch: '+field)
    if a['native']!=b['native']: raise ValueError('Same-image native reconstruction differs')
    x={t['token_index']:t for t in a['tokens']}; y={t['token_index']:t for t in b['tokens']}
    if x.keys()!=y.keys() or any(x[i]['E']!=y[i]['E'] for i in x): raise ValueError('Same-image grid/E differs')
    ids=sorted(i for i in x if i not in a['basis']); k=math.ceil(len(ids)*.1)
    rows=[]
    for metric in METRICS:
        first=sorted(ids,key=lambda i:(-x[i][metric],i))[:k]
        second=sorted(ids,key=lambda i:(-y[i][metric],i))[:k]
        rows.append({'image_id':a['case']['image_id'],'question_a':a['case']['case_id'],'question_b':b['case']['case_id'],
            'metric':metric,'rho_nonbasis':spearman([x[i][metric] for i in ids],[y[i][metric] for i in ids]),
            'top_set_size':k,'top_overlap':len(set(first)&set(second)),
            'first_target_token_same':a['case']['target_token_ids'][0]==b['case']['target_token_ids'][0],
            'target_tokens_a':len(a['case']['target_token_ids']),'target_tokens_b':len(b['case']['target_token_ids'])})
    return rows


def median_defined(values):
    xs=[v for v in values if v is not None]
    return statistics.median(xs) if xs else None


def cohort_summary(records):
    result={'questions':len(records),'images':len({r['case']['image_sha256'] for r in records}),
        'one_token_answers':sum(len(r['case']['target_token_ids'])==1 for r in records),
        'multi_token_answers':sum(len(r['case']['target_token_ids'])>1 for r in records)}
    for metric in METRICS:
        result[metric]={'median_rho_all':median_defined([r['statistics'][metric]['rho_all'] for r in records]),
            'median_rho_nonbasis':median_defined([r['statistics'][metric]['rho_nonbasis'] for r in records]),
            'median_rho_photo_nonbasis':median_defined([r['statistics'][metric]['rho_photo_nonbasis'] for r in records]),
            'high_mean_exceeds_low_questions':sum(r['statistics'][metric]['high_minus_low_mean']>0 for r in records),
            'median_high_minus_low_mean':median_defined([r['statistics'][metric]['high_minus_low_mean'] for r in records])}
    return result


def validity_norm_audit(results,info):
    """Supplemental selected-panel association checks; no full-grid norm claim."""
    archive=results/(info['run']+'.zip')
    if sha256(archive)!=info['archive_sha256']: raise ValueError('Validity source hash mismatch')
    rows=[]
    with zipfile.ZipFile(archive) as z:
        report=json.loads(z.read('report.json')); profile=json.loads(z.read('evidence/validation-profile.json'))
        for name,digest in report['evidence_sha256'].items():
            if hashlib.sha256(z.read('evidence/'+name)).hexdigest()!=digest: raise ValueError('Validity evidence differs')
        lookup={}
        for dataset in ['gqa','mmstar','textvqa']:
            lookup.update({c['case_id']:c for c in json.loads(z.read('evidence/'+dataset+'-manifest.json'))['cases']})
        for entry in profile['perturbations']:
            q=json.loads(z.read('questions/'+entry['case_id'].replace(':','-')+'.json'))
            verify_question(q,entry,lookup[q['case_id']])
            for stratum in ['random','high_I','low_E_high_I','all_selected']:
                items=[r for r in q['alternatives'] if stratum=='all_selected' or r['stratum']==stratum]
                norms=[r['effects']['zero']['original_l2'] for r in items]
                zero=[r['effects']['zero']['I'] for r in items]
                for method in METHODS:
                    whole=[r['effects'][method]['I'] for r in items]
                    first=[effects(q['controls']['reference'],r['effects'][method]['score'])['first_log'] for r in items]
                    rows.append({'case_id':q['case_id'],'stratum':stratum,'method':method,'positions':len(items),
                        'rho_whole_first':spearman(whole,first),
                        'rho_E_first':spearman([r['current_E'] for r in items],first),
                        'rho_zero_I_feature_norm':spearman(zero,norms),
                        'rho_abs_zero_I_feature_norm':spearman([abs(v) for v in zero],norms),
                        'rho_E_feature_norm':spearman([r['current_E'] for r in items],norms),
                        'median_absolute_I':statistics.median(abs(v) for v in whole),
                        'median_feature_norm':statistics.median(norms),
                        'median_delta_norm':statistics.median(r['effects'][method]['delta_l2'] for r in items)})
    return rows


def analyze(results,output,plan_path=PLAN):
    results,output,plan_path=Path(results).resolve(),Path(output).resolve(),Path(plan_path).resolve()
    if output.exists(): raise ValueError('Choose a new derived output directory; prior evidence is preserved')
    plan=json.loads(plan_path.read_text())
    if plan['schema']!='rq1-saved-score-reanalysis-v1': raise ValueError('Unexpected analysis plan')
    records,audits=load_sources(results,plan)
    norms=validity_norm_audit(results,plan['validity_source'])
    questions,tokens,groups,pieces=[],[],[],[]
    for record in records:
        q=record['case']; ts=record['tokens']; nb=[t for t in ts if not t['fps_basis_source']]
        strings=q['generation_tokens']['generated_token_strings'][:len(q['target_token_ids'])]
        row={'dataset':record['dataset'],'case_id':q['case_id'],'image_id':q['image_id'],'question':q['question'],
            'target':q['measurement_target'],'target_tokens':len(q['target_token_ids']),
            'target_token_pieces':' | '.join(strings),'first_target_piece':strings[0],
            'reference_first_logprob':q['controls']['reference']['token_logprobs'][0],
            'reference_first_probability':math.exp(q['controls']['reference']['token_logprobs'][0]),
            'reference_mean_logprob':q['controls']['reference']['mean_logprob'],
            'archived_role':q['outcomes']['input-only'],'max_control_drift':record['max_control_drift'],
            'rho_whole_first_nonbasis':spearman([t['whole_log'] for t in nb],[t['first_log'] for t in nb]),
            'rho_first_log_probability_nonbasis':spearman([t['first_log'] for t in nb],[t['first_probability'] for t in nb]),
            'sign_flip_beyond_1e_minus4':sum(t['whole_log']*t['first_log']<0 and min(abs(t['whole_log']),abs(t['first_log']))>1e-4 for t in nb),
            'first_probability_extra_ties_nonbasis':len({t['first_log'] for t in nb})-len({t['first_probability'] for t in nb}),
            'new_mmstar_grounding_flag':q['case_id'] in plan['mmstar_grounding_flags'],
            'prior_mmstar_quality_flag':q['case_id'] in plan['mmstar_prior_quality_flags'],
            'question_record_sha256':record['question_record_sha256']}
        for metric,stats in record['statistics'].items():
            for field in ['rho_all','rho_nonbasis','rho_photo_nonbasis','high_minus_low_mean']:
                row[metric+'_'+field]=stats[field]
            for name,dist in zip(['low','middle','high','fps_basis'],[*stats['groups'],stats['basis']]):
                groups.append({'case_id':q['case_id'],'dataset':record['dataset'],'metric':metric,'group':name,**dist})
        row['first_minus_whole_rho_nonbasis']=row['first_log_rho_nonbasis']-row['whole_log_rho_nonbasis']
        questions.append(row)
        for index,piece in enumerate(strings):
            deltas=[t['per_answer_token_log_deltas'][index] for t in nb]
            pieces.append({'dataset':record['dataset'],'case_id':q['case_id'],'target':q['measurement_target'],
                'answer_token_index':index,'answer_token_id':q['target_token_ids'][index],'answer_token_piece':piece,
                'reference_piece_logprob':q['controls']['reference']['token_logprobs'][index],
                'rho_E_piece_delta_nonbasis':spearman([t['E'] for t in nb],deltas),
                'mean_piece_delta_nonbasis':statistics.mean(deltas),
                'median_abs_piece_delta_nonbasis':statistics.median(abs(v) for v in deltas)})
        for t in ts:
            tokens.append({'dataset':record['dataset'],'case_id':q['case_id'],**{k:v for k,v in t.items() if k!='reconstruction_error'},
                           'per_answer_token_log_deltas':json.dumps(t['per_answer_token_log_deltas'])})
    paired=[]; by_image=defaultdict(list)
    for record in records:
        if record['dataset']=='gqa': by_image[record['case']['image_sha256']].append(record)
    for same in by_image.values():
        for a,b in combinations(same,2): paired.extend(paired_scores(a,b))
    mm=[r for r in records if r['dataset']=='mmstar']
    strata=[]
    for label,flags in [('all_canonical',[]),('without_new_grounding_flags',plan['mmstar_grounding_flags']),
                        ('without_prior_quality_flags',plan['mmstar_prior_quality_flags']),
                        ('without_either_flag',plan['mmstar_grounding_flags']+plan['mmstar_prior_quality_flags'])]:
        strata.append({'label':label,'excluded_for_sensitivity_only':flags,
                       **cohort_summary([r for r in mm if r['case']['case_id'] not in flags])})
    summary={'schema':plan['schema'],'plan_sha256':sha256(plan_path),'analysis_script_sha256':sha256(Path(__file__)),
        'sources':audits,'question_count':len(questions),'image_count':sum(a['images'] for a in audits),
        'intervention_count':len(tokens),'cohorts':{d:cohort_summary([r for r in records if r['dataset']==d]) for d in plan['sources']},
        'gqa_pairs':paired,'mmstar_sensitivity_subsets':strata,
        'one_token_questions':sum(r['target_tokens']==1 for r in questions),
        'multi_token_questions':sum(r['target_tokens']>1 for r in questions),
        'max_abs_first_vs_whole_E_rho_change_nonbasis':max(abs(r['first_minus_whole_rho_nonbasis']) for r in questions),
        'max_abs_first_log_probability_E_rho_difference_nonbasis':max(abs(r['first_log_rho_nonbasis']-r['first_probability_rho_nonbasis']) for r in questions),
        'interpretation':plan['status'],'scope':plan['scope'],
        'tensor_contents_loaded':False,'no_inference':True}
    output.mkdir(parents=True)
    analysis_sources=['scripts/analyze_rq1_scores.py','scripts/analyze_rq1_saved.py',
        'scripts/analyze_rq1_validity.py','scripts/rq1_protocol.py','scripts/rq1_validation.py',
        'scripts/render_rq1_gallery.py','scripts/run_rq1_input.py','scripts/summarize_rq1_pilot.py',
        'scripts/mmstar_scoring.py','scripts/gqa_scoring.py','llava/eval/m4c_evaluator.py',
        'research/protocols/rq1_score_reanalysis_v1.json','research/protocols/rq1_saved_analysis_v1.json']
    for relative in analysis_sources:
        target=output/'evidence'/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(REPO/relative,target)
    summary['analysis_evidence_sha256']={str(p.relative_to(output/'evidence')):sha256(p)
        for p in (output/'evidence').rglob('*') if p.is_file()}
    summary['analysis_commit']=subprocess.run(['git','rev-parse','HEAD'],cwd=REPO,
        check=True,capture_output=True,text=True,timeout=10).stdout.strip()
    summary['analysis_git_status']=subprocess.run(['git','status','--short'],cwd=REPO,
        check=True,capture_output=True,text=True,timeout=10).stdout.strip()
    for name,rows in [('questions',questions),('tokens',tokens),('question-groups',groups),('gqa-pairs',paired),
                     ('answer-piece-summary',pieces),('validity-norm-and-score-audit',norms)]:
        write_csv(output/(name+'.csv'),rows)
    write_json(output/'plan.json',plan); write_json(output/'summary.json',summary)
    render(output,summary,questions)
    return summary


def render(output,summary,questions):
    def number(v): return 'undefined' if v is None else f'{v:+.5f}' if isinstance(v,float) else str(v)
    def table(headers,rows):
        return '<table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(number(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    parts=['<!doctype html><meta charset="utf-8"><title>RQ1 saved-score reanalysis</title>',
        '<style>body{font:16px system-ui;max-width:1500px;margin:2em auto;padding:1em}table{border-collapse:collapse;margin:1em 0}td,th{border:1px solid #ccc;padding:.5em;text-align:left}td{font-variant-numeric:tabular-nums}h2{margin-top:2em}</style>',
        '<h1>RQ1: first-token versus whole-answer sensitivity</h1>',
        '<p>CPU-only derived analysis. Original ZIPs, predictions, cohorts and scores are unchanged. 30 questions / 24 images / 17,280 singleton interventions; repeats are not additional samples. All measures here use the SAME leave-one-out zeroing counterfactual. None reproduces Information Horizon.</p>',
        '<p><a href="summary.json">Summary / verification</a> · <a href="questions.csv">Question comparisons</a> · <a href="tokens.csv">Every token / answer-token effect</a> · <a href="answer-piece-summary.csv">Individual answer pieces</a> · <a href="gqa-pairs.csv">Same-image pairs</a> · <a href="validity-norm-and-score-audit.csv">Selected-panel norm / scoring audit</a></p>',
        '<h2>Canonical cohorts</h2>',table(['Dataset','Questions','Images','Multi-token answers','Whole E–I rho, median nonbasis','First E–I rho, median nonbasis','Whole, photo nonbasis','First, photo nonbasis'],[
            [d,c['questions'],c['images'],c['multi_token_answers'],c['whole_log']['median_rho_nonbasis'],c['first_log']['median_rho_nonbasis'],c['whole_log']['median_rho_photo_nonbasis'],c['first_log']['median_rho_photo_nonbasis']] for d,c in summary['cohorts'].items()]),
        '<p>These are descriptive per-question medians in curated, outcome-enriched cases. No pooled-token p-values or causal dataset/task effects. Photo-only is a labeled sensitivity analysis; padding features can carry contextual information.</p>',
        '<h2>Per-question scoring changes</h2>',table(['Case','Target','Saved target pieces','Tokens','Whole E–I rho, nonbasis','First E–I rho, nonbasis','Whole/first I rank agreement','Meaningful sign flips'],[
            [q['case_id'],q['target'],q['target_token_pieces'],q['target_tokens'],q['whole_log_rho_nonbasis'],q['first_log_rho_nonbasis'],q['rho_whole_first_nonbasis'],q['sign_flip_beyond_1e_minus4']] for q in questions]),
        '<p>First-token means the first TOKENIZER PIECE, which may be only B in Bottom or L in Laughter. It is not necessarily the first full word or semantic answer. Whole-answer scoring also conditions later answer pieces on the correct preceding pieces. Neither metric automatically has privileged validity.</p>',
        '<h2>Why first-log and first-probability ranks agree</h2><p>For a fixed question, the full reference score is constant. Both first-token measures decrease monotonically with the perturbed first-token probability. Therefore their within-question ranks should agree, except possible finite-precision ties. This is a mathematical check, not independent confirmation. Raw magnitudes and group means need not agree under the nonlinear transformation.</p>',
        '<h2>MMStar flagged-subset sensitivity</h2>',table(['Post-hoc subset','Questions','Images','Whole E–I rho, median nonbasis','First E–I rho, median nonbasis'],[
            [s['label'],s['questions'],s['images'],s['whole_log']['median_rho_nonbasis'],s['first_log']['median_rho_nonbasis']] for s in summary['mmstar_sensitivity_subsets']]),
        '<p>All 12 remain in the canonical analysis. Without-flags subsets are not a new accepted cohort and are NOT known to be clean or image-grounded: most remaining cases have not received the same grounding checks.</p>',
        '<h2>Other interpretation limits</h2><p>Baseline confidence bounds possible negative effects: an already near-certain reference has little room for probability improvement. This affects comparisons of signed effect magnitude between questions, but subtracting a fixed reference does not itself change within-question ranks. Answer length, subword semantics, feature norms, FPS roles, contextualized padding and the choice of intervention remain distinct issues.</p>',
        '<p>Norm associations use only the separate six-question, 32-position validity panel, with its random and post-hoc targeted strata kept separate. They cannot establish a cause or describe feature norms across all 576 positions. The full-forward controls and provenance checks address reproducibility; they do not certify semantic token importance or the validity of every intervention.</p>']
    (output/'index.html').write_text('\n'.join(parts),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=REPO/'research/results')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); result=analyze(args.results,args.output)
    print(json.dumps(result['cohorts'],indent=2))
    print('Review:',args.output/'index.html')
