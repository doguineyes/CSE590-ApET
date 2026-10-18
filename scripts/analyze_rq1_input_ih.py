"""Verify archived input IH calibration/full scores and estimate remaining cost.

CPU only. Read the original ZIP/log without changing them; extract into a NEW
derived folder, verify raw evidence and pairing, and use no model/tensor loading.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import zipfile

from analyze_rq1_validity import assert_close
from rq1_input_ih import (PROFILE,VERSION,comparison,exact_score,file_at,require,
                          source_hashes,validate_calibration,verify_record)
from rq1_protocol import SOURCES,sha256,spearman,write_json
from run_rq1_input import generation_score
from summarize_rq1_pilot import write_csv


def analyze(archive,output,log=None,expected_sha256=None):
    archive,output=Path(archive).resolve(),Path(output).resolve()
    require(not output.exists(),'Choose a new derived folder; originals are preserved')
    digest=sha256(archive)
    require(expected_sha256 is None or digest == expected_sha256,'Archive SHA256 differs')
    profile=json.loads(PROFILE.read_text());records=[];summaries=[];flat=[];drifts=[];timings=[]
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        require(len(names)==len(set(names)) and z.testzip() is None,'ZIP CRC/duplicate-member failure')
        for name in names:file_at(output,name)
        r=json.loads(z.read('report.json'));count=r['arguments']['tokens']
        require(r['status']=='passed' and r['version']==VERSION and count in (32,576),'Expected a passed input IH run')
        require(r['implementation']['native_matches_reference'],'Native reference differs')
        require(r['measurement_sources_sha256']==source_hashes(),'Current measurement source differs')
        require(r['profile_sha256']==sha256(PROFILE)
                ==hashlib.sha256(z.read('evidence/input-ih-profile.json')).hexdigest(),'Profile differs')
        assert_close(r['profile'],profile,'Frozen profile')
        require(r['reference_index_sha256']==profile['reference_index_sha256']
                ==hashlib.sha256(z.read('evidence/references/index.json')).hexdigest(),'Reference index differs')
        for name,h in r['evidence_sha256'].items():
            require(hashlib.sha256(z.read('evidence/'+name)).hexdigest()==h,'Evidence hash mismatch: '+name)
        for name,h in r['measurement_sources_sha256'].items():
            require(hashlib.sha256(z.read('evidence/sources/'+name)).hexdigest()==h,'Measurement source mismatch')
        for name,h in r['implementation']['native_source_sha256'].items():
            require(hashlib.sha256(z.read('evidence/sources/'+name)).hexdigest()==h,'Native source mismatch')
        require(hashlib.sha256(z.read('evidence/sources/environments/original/uv.lock')).hexdigest()
                ==r['lock_sha256'],'Lock differs')
        for name,h in r['fresh_feature_sha256'].items():
            require(hashlib.sha256(z.read(name)).hexdigest()==h,'Fresh feature metadata differs')
        index=json.loads(z.read('evidence/references/index.json'));lookup={e['case_id']:e for e in index['cases']}
        for name,h in index['files_sha256'].items():
            require(hashlib.sha256(z.read('evidence/'+name)).hexdigest()==h,'Archived reference file differs')
        manifests={}
        for dataset,(_,h) in SOURCES.items():
            raw=z.read('evidence/inputs/'+dataset+'/manifest.json')
            require(hashlib.sha256(raw).hexdigest()==h,'Accepted manifest differs')
            manifests[dataset]=json.loads(raw)
        expected=profile['calibration_case_ids'] if count==32 else profile['case_ids']
        require(r['expected_case_ids']==expected==[q['case_id'] for q in r['questions']],'Question panel differs')
        for entry in r['questions']:
            cid=entry['case_id'];reference_entry=lookup[cid];name='questions/'+cid.replace(':','-')+'.json'
            raw=z.read(name)
            require(hashlib.sha256(raw).hexdigest()==r['question_sha256'][name],'Question hash differs')
            q=json.loads(raw);old=json.loads(z.read('evidence/'+reference_entry['question_file']))
            feature=json.loads(z.read('evidence/'+reference_entry['feature_file']))
            for key in ['case_id','image_id','image_sha256','question','measurement_target']:
                require(q[key]==old[key],'Question identity differs: '+key)
            dataset=cid.split(':')[0]
            case=next(c for c in manifests[dataset]['cases'] if c['case_id']==cid)
            assert_close(q['free_generation']['scoring'],generation_score(dataset,case,q['free_generation']['response']),
                         'Fresh generated-answer scoring')
            require(q['archived_reference_file']==reference_entry['question_file']
                    and q['archived_reference_sha256']==index['files_sha256'][reference_entry['question_file']],'Reference association differs')
            fresh=json.loads(z.read(q['fresh_feature_file']))
            require(q['fresh_feature_file'] in r['fresh_feature_sha256'],'Unverified fresh feature metadata')
            for key in ['image_sha256','seed','clip_sha256','projected_sha256']:
                require(fresh[key]==feature[key],'Fresh feature identity differs')
            require(fresh['diagnostics']['input']['errors']==feature['diagnostics']['input']['errors']
                    and fresh['diagnostics']['input']['basis_indices']==feature['diagnostics']['input']['basis_indices'],
                    'Fresh E/basis differs')
            drifts.append(verify_record(q,old,feature))
            rows,summary=comparison(old,q['ih'],feature['diagnostics']['input']['basis_indices'])
            assert_close(entry,{'case_id':cid,'status':'passed','positions':count,'archived_reuse_exact':True,
                               'seconds':q['ih']['seconds'],**summary['all']},'Report question summary')
            values=[t['ih_first_probability'] for t in rows];loop=sum(t['seconds'] for t in q['ih']['tokens'])
            require(all(t['seconds']>0 for t in q['ih']['tokens']),'Invalid retention timing')
            summaries.append({'case_id':cid,'target':q['measurement_target'],'first_piece':q['ih']['first_reference_token_string'],
                **summary['all'],**{'nonbasis_'+k:v for k,v in summary['nonbasis'].items()},
                'rho_E_absolute_IH_auxiliary':spearman([t['E'] for t in rows],[abs(v) for v in values]),
                'full_probability':rows[0]['full_probability'],'remove_all_probability':rows[0]['remove_all_probability'],
                'best_measured_retained_probability':max(t['retained_probability'] for t in rows),
                'median_absolute_IH':statistics.median(map(abs,values)),
                'min_signed_IH':min(values),'max_signed_IH':max(values),
                'positive':sum(v>0 for v in values),'negative':sum(v<0 for v in values),
                'unique_IH_scores':len(set(values))})
            flat.extend({'case_id':cid,**row} for row in rows)
            timings.append({'case_id':cid,'dataset':cid.split(':')[0],'loop_seconds':loop,
                'mean_retention_seconds':loop/count,'question_seconds':q['ih']['seconds'],
                'control_seconds':q['ih']['seconds']-loop,'shortened_length':q['ih']['tokens'][0]['shortened_length']})
            originals=[n for n in names if n.startswith('images/'+q['image_sha256']+'.')]
            require(len(originals)==1 and hashlib.sha256(z.read(originals[0])).hexdigest()==q['image_sha256'],
                    'Original image bytes differ')
            records.append(q)
        require(r['progress']['completed_questions']==len(expected)
                and r['progress']['completed_retention_scores']==r['new_IH_retention_scores']==len(expected)*count
                and r['fresh_zero_audit_scores']==len(expected)*3
                and r['archived_zero_scores_joined']==len(expected)*count,'Completed counts differ')
        output.mkdir(parents=True);z.extractall(output)
    if count==32:
        eligibility,_=validate_calibration(output/'report.json',r['arguments']['model'],r['arguments']['vision_model'])
    else:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shutil.copytree(output/'evidence/references',root/'evidence/references')
            shutil.copytree(output/'evidence/prerequisite-calibration/questions',root/'questions')
            shutil.copy2(output/'evidence/prerequisite-input-ih-calibration.json',root/'report.json')
            eligibility,prior=validate_calibration(root/'report.json',r['arguments']['model'],r['arguments']['vision_model'])
            require(eligibility==r['calibration_prerequisite'],'Embedded calibration gate differs')
            for q in records:
                if q['case_id'] not in prior:continue
                original=prior[q['case_id']]['ih'];by_index={t['token_index']:t for t in q['ih']['tokens']}
                exact_score(q['ih']['controls']['remove_all_before'],original['controls']['remove_all_before'],'Calibration null repeat')
                for t in original['tokens']:exact_score(by_index[t['token_index']]['retained_score'],t['retained_score'],'Calibration retained repeat')
    estimates=[]
    for dataset in SOURCES:
        measured=[t for t in timings if t['dataset']==dataset]
        n=sum(e['dataset']==dataset for e in index['cases'])
        estimates.append({'dataset':dataset,'full_questions':n,
            'observed_mean_seconds_per_retention':statistics.mean(t['mean_retention_seconds'] for t in measured),
            'linear_full_minutes_with_measured_question_controls':n*statistics.mean(
                t['mean_retention_seconds']*576+t['control_seconds'] for t in measured)/60})
    pair=records[:2]
    same_image={'case_a':pair[0]['case_id'],'case_b':pair[1]['case_id'],
        'rho_IH_first':spearman([t['ih_first_probability'] for t in pair[0]['ih']['tokens']],
                               [t['ih_first_probability'] for t in pair[1]['ih']['tokens']])}
    verification={'version':'rq1-input-ih-saved-review-v1','archive':str(archive),'archive_sha256':digest,
        'log':str(log) if log else None,'log_sha256':sha256(log) if log else None,'source_commit':r['commit'],
        'verified_evidence_files':len(r['evidence_sha256']),'fresh_feature_files':len(r['fresh_feature_sha256']),
        'questions':len(records),'images':len({q['image_sha256'] for q in records}),'positions_per_question':count,
        'new_retention_scores':len(flat),'exact_zero_audit_repeats':r['fresh_zero_audit_scores'],
        'reused_zero_scores':r['archived_zero_scores_joined'],'max_retention_control_drift':max(drifts),
        'archived_reuse_exact':True,'source_gate':eligibility,'wall_seconds':r['wall_seconds'],'same_image':same_image,
        'full_timing_linear_minutes':sum(t['linear_full_minutes_with_measured_question_controls'] for t in estimates),
        'timing_caveat':'Extrapolation includes measured per-question controls, excludes fresh generation/encoding, model load, serialization and archiving. Other question lengths and shared GPU load vary.',
        'inference_performed':False,'tensor_contents_loaded':False,
        'scope':'Fixed 30-case input cohort; calibration is only the six-case 32-position prefix. Source-informed 1x1 physical retention. First target tokenizer piece, not answer accuracy or intrinsic importance.'}
    source=Path(__file__).resolve();target=output/'analysis-evidence'/source.name
    target.parent.mkdir();shutil.copy2(source,target);verification['analysis_script_sha256']=sha256(target)
    write_json(output/'verification.json',verification);write_json(output/'effect-summary.json',summaries)
    write_csv(output/'effect-summary.csv',summaries);write_csv(output/'verified-comparison.csv',flat)
    write_json(output/'timing.json',{'estimates':estimates,'questions':timings})
    banner='<p><strong>CPU-verified derived review.</strong> All archived reuse checks are exact; retention controls have zero drift. Original ZIP/log preserved. <a href="verification.json">Verification</a> · <a href="effect-summary.csv">Effect sizes and correlations</a> · <a href="timing.json">Timing estimate</a>. Calibration maxima cover sampled positions only; strong unsampled states may exist. Small probability differences need numerical caution even with stable reference controls.</p>'
    path=output/'index.html';path.write_text(path.read_text(encoding='utf-8')+banner,encoding='utf-8')
    return verification


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--log',type=Path);p.add_argument('--expected-sha256')
    a=p.parse_args();result=analyze(a.archive,a.output,a.log,a.expected_sha256)
    print('Verified',result['new_retention_scores'],'retention scores. Source gate:',result['source_gate']['passed'])
