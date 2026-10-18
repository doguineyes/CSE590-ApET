"""CPU synthesis of the verified 30-case input IH sweep and six-case layer panel.

Read original archives, preserve them, and write only to a new derived folder.
No inference, tensor loading, selection changes or population significance tests.
"""

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import shutil
import statistics
import zipfile

from analyze_rq1_input_ih import analyze
from analyze_rq1_layer_ih_full import e_percentile
from render_rq1_gallery import patch_regions,photo_bounds
from rq1_input_ih import comparison,require
from rq1_layer_ih import comparison_rows
from rq1_protocol import REPO,sha256,spearman,write_json
from summarize_rq1_pilot import write_csv

FULL='rq1-input-ih-thirtycases-v1-20261018T140840768523Z'
FULL_SHA='724a0e13ad5db7ab025e0f11cb0f96ecee377bdbb38719652cc273f53e68f0ef'
LAYER='rq1-layer16-ih-sixcases-v1-20261018T014830278963Z.zip'
LAYER_SHA='e396b7a1c0f0c554e2642ac4e0ac9a83d4f9c4b2a4ecdfbfee6e86bb5a845b37'


def cohort_summary(name,rows):
    result={'group':name,'questions':len(rows)}
    for metric in ['rho_E_zero_first','rho_E_ih_first','rho_zero_ih_first','nonbasis_rho_E_ih_first',
                   'photo_nonbasis_rho_E_ih_first','rho_E_absolute_IH_auxiliary']:
        values=[r[metric] for r in rows]
        result.update({metric+'_median':statistics.median(values),metric+'_min':min(values),metric+'_max':max(values),
                       metric+'_positive':sum(v>0 for v in values),metric+'_negative':sum(v<0 for v in values)})
    return result


def synthesize(results,output):
    results,output=Path(results).resolve(),Path(output).resolve()
    require(not output.exists(),'Choose a new synthesis folder')
    archive=results/(FULL+'.zip');layer_path=results/LAYER
    require(sha256(archive)==FULL_SHA and sha256(layer_path)==LAYER_SHA,'Pinned source archive differs')
    verification=analyze(archive,output,archive.with_suffix('.log'),FULL_SHA)
    summaries=json.loads((output/'effect-summary.json').read_text());summary_by={r['case_id']:r for r in summaries}
    questions={};tokens={};best=[];gallery=[];pairs=[];depth=[];calibration=[]
    with zipfile.ZipFile(archive) as z:
        report=json.loads(z.read('report.json'));index=json.loads(z.read('evidence/references/index.json'))
        lookup={e['case_id']:e for e in index['cases']}
        images={}
        for dataset in ['gqa','mmstar','textvqa']:
            images.update(json.loads(z.read('evidence/inputs/'+dataset+'/manifest.json'))['images'])
        flags=json.loads(z.read('evidence/sources/research/protocols/rq1_score_reanalysis_v1.json'))
        for e in report['questions']:
            cid=e['case_id'];q=json.loads(z.read('questions/'+cid.replace(':','-')+'.json'));entry=lookup[cid]
            old=json.loads(z.read('evidence/'+entry['question_file']));f=json.loads(z.read('evidence/'+entry['feature_file']))
            rows,_=comparison(old,q['ih'],f['diagnostics']['input']['basis_indices'])
            rows=sorted(rows,key=lambda t:t['token_index']);bounds=photo_bounds(images[q['image_id']]['size']);regions=patch_regions(bounds)
            nb=[t for t in rows if not t['fps_basis_source']];photo=[t for t in nb if regions[t['token_index']]['region']=='photo']
            s=summary_by[cid]
            s.update(dataset=cid.split(':')[0],image_id=q['image_id'],
                photo_nonbasis_rho_E_ih_first=spearman([t['E'] for t in photo],[t['ih_first_probability'] for t in photo]),
                mmstar_grounding_flag=cid in flags['mmstar_grounding_flags'],
                mmstar_prior_quality_flag=cid in flags['mmstar_prior_quality_flags'])
            for rank,row in enumerate(sorted(rows,key=lambda t:(-t['retained_probability'],t['token_index']))[:5],1):
                best.append({'case_id':cid,'target':q['measurement_target'],'rank':rank,**row,
                    'E_percentile_nonbasis':None if row['fps_basis_source'] else e_percentile(row['E'],[t['E'] for t in nb]),
                    **regions[row['token_index']]})
            if cid in report['calibration_prerequisite']['case_ids']:
                prior=json.loads(z.read('evidence/prerequisite-calibration/questions/'+cid.replace(':','-')+'.json'))
                calibration.append({'case_id':cid,'best_retained_32':max(t['retained_probability'] for t in prior['ih']['tokens']),
                    'best_retained_576':s['best_measured_retained_probability'],
                    'rho_E_IH_32':prior['comparison']['all']['rho_E_ih_first'],'rho_E_IH_576':s['rho_E_ih_first']})
            questions[cid]={'current':q,'archived':old,'feature':f};tokens[cid]=rows
            gallery.append({'case_id':cid,'question':q['question'],'target':q['measurement_target'],
                'image':'images/'+q['image_sha256']+'-canvas.png','bounds':bounds,'summary':s,
                'tokens':[[t['E'],t['zero_first_probability'],t['ih_first_probability'],t['retained_probability'],
                           t['fps_basis_source'],regions[t['token_index']]['region']] for t in rows]})
        grouped=defaultdict(list)
        for cid,q in questions.items():
            if cid.startswith('gqa:'):grouped[q['current']['image_sha256']].append(cid)
        for ids in grouped.values():
            require(len(ids)==2,'Expected six GQA same-image pairs')
            a,b=ids;qa,qb=questions[a],questions[b]
            require(qa['archived']['projected_sha256']==qb['archived']['projected_sha256']
                    and [t['E'] for t in tokens[a]]==[t['E'] for t in tokens[b]],'Same-image features/E differ')
            pairs.append({'case_a':a,'case_b':b,'target_a':qa['current']['measurement_target'],'target_b':qb['current']['measurement_target'],
                'input_features_E_exact':True,
                'rho_IH_between_questions':spearman([t['ih_first_probability'] for t in tokens[a]],
                                                   [t['ih_first_probability'] for t in tokens[b]]),
                'rho_zero_first_between_questions':spearman([t['zero_first_probability'] for t in tokens[a]],
                                                           [t['zero_first_probability'] for t in tokens[b]])})
        with zipfile.ZipFile(layer_path) as lz:
            require(lz.testzip() is None,'Layer ZIP CRC failed')
            lr=json.loads(lz.read('report.json'))
            for e in lr['ih_questions']:
                cid=e['case_id'];q=json.loads(lz.read('questions/'+cid.replace(':','-')+'.json'))
                old=questions[cid]['archived'];ir=tokens[cid]
                require(q['input']['target_token_ids']==old['target_token_ids']
                        and q['input']['controls']['reference']==old['controls']['reference']
                        and q['clip_sha256']==old['clip_sha256'] and q['projected_sha256']==old['projected_sha256'],
                        'Layer/input case or reference identity differs')
                prior_zero={t['token_index']:t for t in q['input']['tokens']}
                for t in old['tokens']:
                    require(t['perturbed_score']==prior_zero[t['token_index']]['perturbed_score']
                            and t['reconstruction_error']==prior_zero[t['token_index']]['reconstruction_error'],
                            'Archived layer-panel input repeat differs')
                layer_rows=sorted(comparison_rows(q['decoder'],q['ih']),key=lambda t:t['token_index'])
                depth.append({'case_id':cid,'target':q['measurement_target'],
                    'input_E_zero_first':summary_by[cid]['rho_E_zero_first'],
                    'layer16_E_zero_first':q['ih_comparison']['all']['rho_E_zero_first'],
                    'input_E_IH_first':summary_by[cid]['rho_E_ih_first'],
                    'layer16_E_IH_first':q['ih_comparison']['all']['rho_E_ih_first'],
                    'cross_stage_IH_rank_rho':spearman([t['ih_first_probability'] for t in ir],[t['ih_first_probability'] for t in layer_rows]),
                    'input_null_probability':ir[0]['remove_all_probability'],
                    'layer16_null_probability':layer_rows[0]['remove_all_probability'],
                    'input_best_retained_probability':max(t['retained_probability'] for t in ir),
                    'layer16_best_retained_probability':max(t['retained_probability'] for t in layer_rows)})
    dataset_rows=[cohort_summary(d,[r for r in summaries if r['dataset']==d]) for d in ['gqa','mmstar','textvqa']]
    exclusions={'all12':[], 'without_grounding_flags':flags['mmstar_grounding_flags'],
        'without_prior_quality_flags':flags['mmstar_prior_quality_flags'],
        'without_flag_union':flags['mmstar_grounding_flags']+flags['mmstar_prior_quality_flags']}
    robustness=[cohort_summary(name,[r for r in summaries if r['dataset']=='mmstar' and r['case_id'] not in excluded])
                for name,excluded in exclusions.items()]
    write_json(output/'cohort-summary.json',{'datasets':dataset_rows,'mmstar_flag_sensitivity':robustness,
        'scope':'Per-question descriptive summaries, not pooled-token or population inference. Primary cohort remains all 30 questions.'})
    write_csv(output/'dataset-summary.csv',dataset_rows);write_csv(output/'mmstar-flag-sensitivity.csv',robustness)
    write_csv(output/'question-synthesis.csv',summaries);write_csv(output/'top-retained-tokens.csv',best)
    write_csv(output/'same-image-pairs.csv',pairs);write_csv(output/'depth-comparison.csv',depth)
    write_csv(output/'sampled-full-comparison.csv',calibration)
    write_json(output/'synthesis-verification.json',{'input_archive_sha256':FULL_SHA,'layer_archive_sha256':LAYER_SHA,
        'input_verification':verification,'layer_input_zero_scores_repeated_exactly':3456,
        'same_image_pairs':len(pairs),'top_tokens_per_case':5,'inference_performed':False,'tensor_contents_loaded':False,
        'posthoc_scope':'Primary dataset/cohort unchanged. Top-five and gallery markers are illustrative rankings; flags are sensitivity analyses, not a clean denominator.'})
    shutil.copy2(Path(__file__),output/'analysis-evidence'/Path(__file__).name)
    render(output,gallery,dataset_rows,depth)
    return {'questions':len(summaries),'pairs':len(pairs),'depth_questions':len(depth),'output':str(output)}


def render(output,cases,datasets,depth):
    """Lazy HTML maps: only one case's cells are built at a time on the Intel Mac."""
    payload=json.dumps(cases,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    esc=lambda v:html.escape(str(v))
    table=lambda rows,cols:'<table><tr>'+''.join('<th>'+esc(c)+'</th>' for c in cols)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+esc(f'{r[c]:.3f}' if isinstance(r[c],float) else r[c])+'</td>' for c in cols)+'</tr>' for r in rows)+'</table>'
    intro=table(datasets,['group','questions','rho_E_zero_first_median','rho_E_ih_first_median','rho_zero_ih_first_median'])
    page='''<!doctype html><meta charset="utf-8"><title>RQ1 full E / zero / IH synthesis</title>
<style>body{font:16px system-ui;max-width:1500px;margin:2em auto;padding:1em}table{border-collapse:collapse;font-size:14px}td,th{padding:.5em;border:1px solid #ccc}.scroll{overflow:auto}.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}.photo,.map{position:relative;aspect-ratio:1}.photo img{width:100%}.map{display:grid;grid-template-columns:repeat(24,1fr)}.footprint{position:absolute;border:2px solid #00b9cc;box-sizing:border-box;pointer-events:none}.mark{position:absolute;border:2px solid #ffec00;box-sizing:border-box;color:white;background:#0007;font:bold 11px system-ui;text-align:center}.basis{outline:1px dotted gray;outline-offset:-1px}h3{font-size:16px}select{font:inherit;max-width:100%;margin:.5em}#question{white-space:pre-wrap}.panel p{font-size:13px}</style>
<h1>RQ1 input: reconstruction E, zero-one and retain-one</h1>
<p>All 30 questions / 24 images / 17,280 input IH effects verified; old E/zero scores reused after exact controls. Original archives unchanged. These are first-target-token counterfactual probability changes, not whole-answer accuracy or intrinsic importance. Input nulls preserve position gaps; CLIP features are contextualized.</p>
<p><a href="verification.json">Raw verification</a> · <a href="question-synthesis.csv">All questions</a> · <a href="same-image-pairs.csv">Same-image pairs</a> · <a href="depth-comparison.csv">Six-case depth comparison</a> · <a href="top-retained-tokens.csv">Top-five states</a> · <a href="mmstar-flag-sensitivity.csv">MMStar flag sensitivity</a></p>
<div class="scroll">DATASET_TABLE</div>
<p>Dataset rows are medians of per-question correlations. They describe this curated cohort; no pooled-token p-values or population generalization. All 12 MMStar cases remain primary, including existing quality/grounding flags.</p>
<label>Case <select id="case"></select></label><label>Marker ranking <select id="ranking"><option value="2">IH signed sensitivity</option><option value="1">Zero-one signed sensitivity</option><option value="0">Reconstruction E</option></select></label>
<h2 id="heading"></h2><p id="question"></p><p id="stats"></p>
<p>24×24 grid, rows top-to-bottom and columns left-to-right, both 0–23. Index = 24×row + column. Cyan shows the photo footprint inside the actual padded 336×336 input; padding remains included. Markers 1–3 stay inside the selected cells and show the largest values for the chosen ranking. Dotted map cells are FPS basis sources. Hover for exact values. Each map has its own linear scale: compare printed ranges before comparing colors.</p>
<div class="panels"><div class="panel"><h3>Actual padded CLIP input</h3><div id="photo" class="photo"></div></div><div class="panel"><h3>Native input E</h3><div id="E" class="map"></div><p id="Escale"></p></div><div class="panel"><h3>Zero-one first-token sensitivity</h3><div id="zero" class="map"></div><p id="zeroscale"></p></div><div class="panel"><h3>Retain-one minus null sensitivity</h3><div id="IH" class="map"></div><p id="IHscale"></p></div></div>
<div id="markers" class="scroll"></div>
<h2>Matched input / layer-16 panel</h2><p>Same six questions and targets. E uses different feature spaces, and the retained context differs with depth; late text has already seen the image. These descriptive comparisons do not isolate a causal depth effect.</p><div class="scroll">DEPTH_TABLE</div>
<script type="application/json" id="data">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent),select=document.getElementById('case'),ranking=document.getElementById('ranking');
const format=v=>v===null?'basis / N/A':Number(v).toPrecision(6);
data.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.case_id+' — '+c.target;select.append(o)});
function footprint(box,bounds){const e=document.createElement('div');e.className='footprint';Object.assign(e.style,{left:bounds[0]/3.36+'%',top:bounds[1]/3.36+'%',width:(bounds[2]-bounds[0])/3.36+'%',height:(bounds[3]-bounds[1])/3.36+'%'});box.append(e)}
function render(){const c=data[Number(select.value)],rows=c.tokens,metric=Number(ranking.value),top=rows.map((r,i)=>({r,i})).sort((a,b)=>b.r[metric]-a.r[metric]||a.i-b.i).slice(0,3),s=c.summary;
document.getElementById('heading').textContent=c.case_id+' — '+c.target;document.getElementById('question').textContent=c.question;
document.getElementById('stats').textContent='E–zero rho '+format(s.rho_E_zero_first)+'; E–IH rho '+format(s.rho_E_ih_first)+'; zero–IH rho '+format(s.rho_zero_ih_first)+'. Full / null / best retained probability: '+[s.full_probability,s.remove_all_probability,s.best_measured_retained_probability].map(format).join(' / ')+'. Existing MMStar flags: grounding='+s.mmstar_grounding_flag+', quality='+s.mmstar_prior_quality_flag+'.';
const photo=document.getElementById('photo');photo.replaceChildren();const img=document.createElement('img');img.src=c.image;img.alt='Saved actual padded CLIP canvas';photo.append(img);footprint(photo,c.bounds);
top.forEach(({r,i},j)=>{const mark=document.createElement('span');mark.className='mark';mark.textContent=j+1;mark.title='index '+i+', '+r[5]+', FPS basis '+r[4];Object.assign(mark.style,{left:(i%24)*100/24+'%',top:Math.floor(i/24)*100/24+'%',width:100/24+'%',height:100/24+'%'});photo.append(mark)});
['E','zero','IH'].forEach((id,k)=>{const box=document.getElementById(id);box.replaceChildren();const maximum=Math.max(...rows.map(r=>Math.abs(r[k]))),limit=maximum||1;rows.forEach((r,i)=>{const cell=document.createElement('span'),v=r[k]/limit,end=k===0?[245,204,65]:(v>=0?[181,24,45]:[33,102,172]),start=k===0?[30,65,125]:[248,248,248],rgb=end.map((x,j)=>Math.round(start[j]+(x-start[j])*Math.abs(v)));cell.style.background='rgb('+rgb.join(',')+')';if(r[4])cell.className='basis';cell.title='index '+i+'; row '+Math.floor(i/24)+', column '+i%24+'; value '+format(r[k])+'; '+r[5]+'; FPS basis '+r[4];box.append(cell)});footprint(box,c.bounds);document.getElementById(id+'scale').textContent='Linear range '+(k===0?'0 to '+format(maximum):'-'+format(maximum)+' to +'+format(maximum))});
const panel=document.getElementById('markers');panel.replaceChildren();const table=document.createElement('table'),header=document.createElement('tr');['Rank','Index (row, col)','E','Zero I','IH I','Retained probability','E percentile, nonbasis','Location / FPS'].forEach(t=>{const th=document.createElement('th');th.textContent=t;header.append(th)});table.append(header);const errors=rows.filter(r=>!r[4]).map(r=>r[0]);top.forEach(({r,i},j)=>{const tr=document.createElement('tr'),pct=r[4]?null:100*(errors.filter(e=>e<r[0]).length+(errors.filter(e=>e===r[0]).length-1)/2)/(errors.length-1);[j+1,i+' ('+Math.floor(i/24)+', '+i%24+')',...r.slice(0,4).map(format),format(pct),r[5]+' / '+r[4]].forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.append(td)});table.append(tr)});panel.append(table)}
select.onchange=render;ranking.onchange=render;render();
</script>'''
    page=page.replace('DATASET_TABLE',intro).replace('DEPTH_TABLE',table(depth,['case_id','input_E_IH_first','layer16_E_IH_first','cross_stage_IH_rank_rho'])).replace('PAYLOAD',payload)
    (output/'synthesis.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=REPO/'research/results');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(synthesize(a.results,a.output),indent=2))
