"""Render a separate, padding-aware review gallery from saved RQ1 measurements.

No inference. Matplotlib/Pillow/NumPy are plotting-only dependencies; the original
Kaggle environment already provides them. Geometry/statistical helpers are stdlib.
Original exports and the numerical analysis remain unchanged.
"""

import argparse
from collections import defaultdict
import html
import json
import os
from pathlib import Path
import statistics
import textwrap

from rq1_protocol import sha256, write_json
from summarize_rq1_pilot import write_csv

REPO = Path(__file__).resolve().parents[1]
COLORS = {"I": "#d52c35", "E": "#e7ab00", "boundary": "#00a8bb"}


def photo_bounds(size, canvas=336):
    """Geometric photo footprint after native integer-centered square padding."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    side = max(size)
    left, top = (side-width)//2, (side-height)//2
    scale = canvas / side
    return [left*scale, top*scale, (left+width)*scale, (top+height)*scale]


def patch_regions(bounds, grid=24, canvas=336):
    """Physical patch/photo intersection, independent of E, I or pixel colors."""
    x0, y0, x1, y1 = bounds
    patch = canvas/grid
    result = {}
    for i in range(grid*grid):
        x, y = (i % grid)*patch, (i//grid)*patch
        area = max(0, min(x+patch,x1)-max(x,x0)) * max(0,min(y+patch,y1)-max(y,y0))
        fraction = min(1.0, max(0.0, area/(patch*patch)))
        region = "padding" if fraction < 1e-10 else "photo" if fraction > 1-1e-10 else "mixed"
        result[i] = {"region": region, "photo_fraction": fraction}
    return result


def marker_choices(tokens, regions, basis):
    """Two photo +I, two photo high-E non-basis; one +I in each other region."""
    selected = defaultdict(set)
    for region, count in [("photo",2), ("mixed",1), ("padding",1)]:
        positive = [t for t in tokens if regions[t["token_index"]]["region"] == region and t["intervention_score"] > 0]
        for t in sorted(positive, key=lambda t:(-t["intervention_score"],t["token_index"]))[:count]:
            selected[t["token_index"]].add("I")
    candidates = [t for t in tokens if regions[t["token_index"]]["region"] == "photo" and t["token_index"] not in basis]
    for t in sorted(candidates,key=lambda t:(-t["reconstruction_error"],t["token_index"]))[:2]:
        selected[t["token_index"]].add("E")
    return {i:sorted(roles) for i,roles in sorted(selected.items())}


def region_stats(case, regions, basis):
    rows=[]
    for region in ["photo","mixed","padding"]:
        items=[t for t in case["tokens"] if regions[t["token_index"]]["region"]==region]
        nb=[t for t in items if t["token_index"] not in basis]
        rows.append({"case_id":case["case_id"],"region":region,"patches":len(items),
            "basis_sources":len(items)-len(nb),"nonbasis_patches":len(nb),
            "mean_signed_I_all":statistics.mean(t["intervention_score"] for t in items) if items else None,
            "mean_signed_I_nonbasis":statistics.mean(t["intervention_score"] for t in nb) if nb else None})
    return rows


def decorations(ax, bounds, markers, regions, basis):
    from matplotlib.patches import Rectangle
    from matplotlib import patheffects
    # All axes use patch-center coordinates, including the original CLIP canvas.
    x0,y0,x1,y1=[v/14-.5 for v in bounds]
    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=COLORS["boundary"],linewidth=2.0,zorder=4))
    for x,y,w,h in [(-.5,-.5,24,y0+.5),(-.5,y1,24,23.5-y1),(-.5,y0,x0+.5,y1-y0),(x1,y0,23.5-x1,y1-y0)]:
        if w>1e-8 and h>1e-8:
            ax.add_patch(Rectangle((x,y),w,h,fill=False,hatch="///",edgecolor=(.35,.35,.35,.25),linewidth=0,zorder=2))
            if min(w,h)>.8:
                ax.text(x+w/2,y+h/2,"padding",ha="center",va="center",fontsize=8,color="#444",zorder=3,
                    bbox={"facecolor":"white","alpha":.75,"edgecolor":"none","pad":1})
    for i,roles in markers.items():
        region=regions[i]["region"]
        style="--" if region=="padding" else "-." if region=="mixed" else "-"
        for j,role in enumerate(roles):
            inset=.06+j*.12
            line=Rectangle((i%24-.5+inset,i//24-.5+inset),1-2*inset,1-2*inset,
                fill=False,edgecolor=COLORS[role],linewidth=2,linestyle=style,zorder=5)
            line.set_path_effects([patheffects.Stroke(linewidth=3.4,foreground="white"),patheffects.Normal()])
            ax.add_patch(line)
        suffix="P" if region=="padding" else "M" if region=="mixed" else ""
        # Labels stay inside their patch and the square input, including row zero.
        ax.text(i%24,i//24,str(i)+suffix,ha="center",va="center",color="white",fontsize=6.5,
            bbox={"facecolor":"#111","alpha":.72,"edgecolor":"none","pad":.6},zorder=6)
    ax.set_xlim(-.5,23.5);ax.set_ylim(23.5,-.5)
    ax.set_xticks([0,5,10,15,20,23]);ax.set_yticks([0,5,10,15,20,23])
    ax.set_xlabel("Patch column (0–23)");ax.set_ylabel("Patch row (0–23)")


def plot(record, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    from PIL import Image

    case=record["case"]
    canvas=np.array(Image.open(record["canvas"]).convert("RGB"))
    if canvas.shape != (336,336,3):
        raise ValueError("Saved canvas does not match the native 336×336 input")
    # Check our dimension-derived footprint against the saved actual input.
    original=Image.open(record["encoded"]).convert("RGB")
    side=max(original.size)
    mean=record["processor"]["image_mean"]
    reconstructed=Image.new("RGB",(side,side),tuple(int(v*255) for v in mean))
    reconstructed.paste(original,((side-original.width)//2,(side-original.height)//2))
    reconstructed=reconstructed.resize((336,336),resample=Image.Resampling.BICUBIC)
    difference=np.abs(np.array(reconstructed,dtype=float)-canvas)
    if float(difference.mean())>3:
        raise ValueError("Dimension-derived square padding does not match saved input")
    record["geometry_mean_pixel_difference"]=float(difference.mean())
    errors=np.array([t["reconstruction_error"] for t in sorted(case["tokens"],key=lambda t:t["token_index"])])
    effects=np.array([t["intervention_score"] for t in sorted(case["tokens"],key=lambda t:t["token_index"])])
    fig,axes=plt.subplots(1,3,figsize=(16,6.4))
    fig.subplots_adjust(left=.04,right=.93,bottom=.23,top=.80,wspace=.4)
    axes[0].imshow(canvas,extent=(-.5,23.5,23.5,-.5))
    axes[0].set_title("Actual padded CLIP input\n336×336 pixels → 24×24 patches",fontsize=11)
    heat=axes[1].imshow(errors.reshape(24,24),cmap="viridis",interpolation="nearest")
    axes[1].set_title("Reconstruction error E\nColor = native feature residual",fontsize=11)
    fig.colorbar(heat,cax=axes[1].inset_axes([1.025,.1,.045,.8]),label="E (L2 residual)")
    heat=axes[2].imshow(effects.reshape(24,24),cmap="coolwarm",vmin=-record["I_scale"],vmax=record["I_scale"],interpolation="nearest")
    axes[2].set_title("Signed answer sensitivity I\nRed: supports reference; blue: opposite",fontsize=11)
    fig.colorbar(heat,cax=axes[2].inset_axes([1.025,.1,.045,.8]),label="I (nats / answer token)")
    for ax in axes:
        decorations(ax,record["bounds"],record["markers"],record["regions"],record["basis"])
        ax.tick_params(labelsize=8)
    question=case["question"]
    if "Question:" in question:question=question.split("Question:",1)[1]
    for delimiter in ["Options:","Choices:","\n(A)"]:
        question=question.split(delimiter,1)[0]
    heading=f'{case["case_id"]}: {question.strip()}'
    fig.suptitle(textwrap.fill(heading,115)+f'\nFixed reference: {case["measurement_target"]} | Archived input-only: {case["outcomes"]["input-only"]}',fontsize=12,y=.97)
    handles=[Line2D([0],[0],color=COLORS["I"],lw=2,label="High positive I"),
             Line2D([0],[0],color=COLORS["E"],lw=2,label="High E (photo, non-basis)"),
             Line2D([0],[0],color=COLORS["boundary"],lw=2,label="Photograph boundary"),
             Line2D([0],[0],color="#555",lw=2,ls="--",label="P: padding"),
             Line2D([0],[0],color="#555",lw=2,ls="-.",label="M: mixed boundary")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.09),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.045,"Patch IDs are row-major: ID = 24 × row + column. Red/gold indicate the marker selection rule, not retention by ApET.",ha="center",fontsize=9)
    fig.text(.5,.012,"Padding measurements are preserved. Features contain context; these are not purely local pixel-importance maps. "+record["scale_note"],ha="center",fontsize=8)
    fig.savefig(destination.with_suffix(".png"),dpi=170,facecolor="white")
    fig.savefig(destination.with_suffix(".pdf"),facecolor="white")
    plt.close(fig)


def table(headers,rows):
    return '<table><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</table>'


def render(analysis,output,results):
    from PIL import Image
    analysis,output,results=Path(analysis).resolve(),Path(output).resolve(),Path(results).resolve()
    if output.exists():raise ValueError("Choose a new gallery output; existing figures are preserved")
    summary=json.loads((analysis/'summary.json').read_text())
    for source in summary['sources']:
        run=results/source['run']
        if sha256(run.with_suffix('.zip'))!=source['archive_sha256'] or sha256(run/'report.json')!=source['report_sha256']:
            raise ValueError("Original source hash differs")
        if run==output or run in output.parents:raise ValueError("Output must be outside original runs")
    output.mkdir(parents=True);(output/'figures').mkdir()
    records=[]
    for source in summary['sources']:
        run=results/source['run']
        manifest=json.loads((run/'evidence/manifest.json').read_text())
        for qpath in sorted((run/'questions').glob('*.json')):
            case=json.loads(qpath.read_text())
            if len(case['tokens'])!=576 or {t['token_index'] for t in case['tokens']}!=set(range(576)):
                raise ValueError("Gallery expects complete singleton measurements")
            meta=json.loads((run/'features'/(case['image_sha256']+'.json')).read_text())
            image_meta=manifest['images'][case['image_id']]
            encoded=run/'images'/Path(image_meta['file']).name
            if sha256(encoded)!=case['image_sha256']:raise ValueError("Encoded image hash differs")
            size=Image.open(encoded).size
            if list(size)!=image_meta['size']:raise ValueError("Manifest image dimensions differ")
            processor=meta['processor']
            if processor['crop_size']!={'height':336,'width':336} or processor['size']!={'shortest_edge':336}:
                raise ValueError("Unsupported saved image preprocessing geometry")
            bounds=photo_bounds(size);regions=patch_regions(bounds)
            basis=set(meta['diagnostics']['input']['basis_indices'])
            records.append({'case':case,'source':run,'dataset':source['dataset'],'bounds':bounds,'regions':regions,
                'basis':basis,'retained':set(meta['diagnostics']['input_merge']['target_indices']),
                'replacement_slots':set(meta['diagnostics']['input_merge']['basis_slots']),
                'markers':marker_choices(case['tokens'],regions,basis),'encoded':encoded,
                'canvas':run/'images'/(case['image_sha256']+'-canvas.png'),'processor':processor})
    image_scales=defaultdict(float)
    for r in records:
        key=(r['dataset'],r['case']['image_id'])
        image_scales[key]=max(image_scales[key],*(abs(t['intervention_score']) for t in r['case']['tokens']))
    parts=['<!doctype html><meta charset="utf-8"><title>RQ1 padding-aware review gallery</title>',
        '<style>body{font:17px system-ui;max-width:1500px;margin:2em auto;padding:0 1em;color:#18232d}a{color:#075e9d}img{width:100%;height:auto}section{border-top:1px solid #bac9d0;margin-top:2.5em;padding-top:1em}table{border-collapse:collapse;margin:1em 0}th,td{padding:.5em;border:1px solid #ccd5da;text-align:left}th{background:#eef4f6}.guide{background:#eef7fa;border-left:4px solid #00a8bb;padding:1em}code{background:#eef1f3;padding:.1em .3em}</style>',
        '<h1>RQ1: clearer patch locations and marker meanings</h1><p>New review figures from the same 30 saved questions. Original exports and numerical group-analysis results are unchanged. All 576 patches remain visible.</p>',
        '<div class="guide"><b>Reading the figures</b><ul><li><b style="color:#008999">Cyan boundary</b>: actual photograph inside the padded square input. Hatching labels the padding.</li><li><b style="color:#d52c35">Red outline</b>: high positive signed sensitivity I; <b style="color:#a87500">gold outline</b>: high reconstruction E. Two outlines mean both selection rules chose that patch.</li><li><b>P / dashed</b>: patch entirely in padding. <b>M / dash-dot</b>: patch crosses the photo boundary. Unmarked patches still have measurements.</li><li>Patch IDs are coordinates: ID = 24 × row + column, from 0 to 575. Marker colors do not indicate native ApET retention.</li><li>Default markers show the top two positive-I photo patches and top two high-E non-basis photo patches, plus the highest positive-I padding patch and mixed patch when present. This displays padding effects separately without suppressing them.</li><li>GQA same-image pairs share sensitivity color limits. Other images use individual scales; read the color bars. Red/blue heatmap values are signed answer-support changes, not correctness or pixel-level causal relevance.</li></ul></div>',
        '<p>Start with <a href="#gqa-202218829">Window versus paper towels</a>, <a href="#mmstar-136">soccer and padding</a>, or <a href="#textvqa-35962">Cookery school</a>. Compare <a href="#gqa-201636988">the two cake/stove questions</a> using their shared color scale.</p>',
        '<p><a href="regions.csv">All patch locations</a> · <a href="region-summary.csv">Photo/mixed/padding summaries</a> · <a href="markers.csv">Marker values and native roles</a> · <a href="manifest.json">Rendering provenance</a></p>',
        '<p><b>Region statistics are auxiliary.</b> They describe physical patch footprints and separate singleton effects, not independent samples or jointly removed groups. Basis sources remain separately counted. The previous full-grid low/middle/high-E group results remain authoritative and are not recomputed or replaced here.</p>']
    marker_rows=[];region_rows=[];stat_rows=[];figures=[]
    for record in records:
        case=record['case'];cid=case['case_id'];slug=cid.replace(':','-')
        record['I_scale']=max(image_scales[(record['dataset'],case['image_id'])],1e-8)
        record['scale_note']='I scale shared by same-image questions.' if record['dataset']=='gqa' else 'I scale is specific to this image.'
        plot(record,output/'figures'/slug)
        stats=region_stats(case,record['regions'],record['basis']);stat_rows.extend(stats)
        lookup={t['token_index']:t for t in case['tokens']}
        for i,region in record['regions'].items():
            region_rows.append({'case_id':cid,'token_index':i,**region,'fps_basis_source':i in record['basis'],
                'E':lookup[i]['reconstruction_error'],'I':lookup[i]['intervention_score']})
        rows=[]
        for i,roles in record['markers'].items():
            region=record['regions'][i]
            row={'case_id':cid,'token_index':i,'marker_reason':'+'.join(roles),**region,
                'E':lookup[i]['reconstruction_error'],'I':lookup[i]['intervention_score'],
                'fps_basis_source':i in record['basis'],'direct_original_retained':i in record['retained'],
                'basis_replacement_slot':i in record['replacement_slots']}
            marker_rows.append(row)
            rows.append([i,' + '.join(roles),region['region'],f'{100*region["photo_fraction"]:.2f}%',f'{row["E"]:.3f}',f'{row["I"]:+.5f}',row['fps_basis_source'],row['direct_original_retained']])
        parts.append(f'<section id="{slug}"><h2>{html.escape(cid)}</h2><p>{html.escape(case["question"])}</p><p>Fixed reference: <b>{html.escape(case["measurement_target"])}</b>. Archived input-only role: {case["outcomes"]["input-only"]}. {record["scale_note"]}</p>'
            f'<a href="figures/{slug}.png"><img loading="lazy" src="figures/{slug}.png" alt="Photo boundary and region-aware E/I markers for {cid}"></a>'
            f'<p><a href="figures/{slug}.png">Open full-resolution PNG</a> · <a href="figures/{slug}.pdf">Export PDF</a></p>'
            +table(['Patch ID','Selected by','Location','Photo coverage','E','Signed I','FPS source','Direct original retained'],rows)
            +table(['Location','All patches','FPS sources','Non-basis patches','Mean signed I (non-basis)'],[
                [s['region'],s['patches'],s['basis_sources'],s['nonbasis_patches'],f'{s["mean_signed_I_nonbasis"]:+.6f}' if s['mean_signed_I_nonbasis'] is not None else '—'] for s in stats])+'</section>')
        figures.append({'case_id':cid,'source_question_sha256':sha256(record['source']/'questions'/(case['question_id']+'.json')),
            'source_canvas_sha256':sha256(record['canvas']),'original_image_sha256':case['image_sha256'],
            'photo_bounds_canvas_pixels':record['bounds'],'I_scale':record['I_scale'],
            'geometry_mean_pixel_difference':record['geometry_mean_pixel_difference'],
            'png_sha256':sha256(output/'figures'/(slug+'.png')),'pdf_sha256':sha256(output/'figures'/(slug+'.pdf'))})
        print('Rendered',cid,flush=True)
    write_csv(output/'markers.csv',marker_rows);write_csv(output/'regions.csv',region_rows)
    write_csv(output/'region-summary.csv',stat_rows)
    write_json(output/'manifest.json',{'schema':'rq1-padding-aware-gallery-v1',
        'source_analysis_summary_sha256':sha256(analysis/'summary.json'),'renderer_sha256':sha256(Path(__file__)),
        'source_runs':summary['sources'],'region_rule':'Geometric photo footprint under native centered square padding and resize; no classification by E/I or pixel colors',
        'figures':figures,'limitations':'Physical patch location is not a claim about the locality of contextualized CLIP information. Rendering only: original source results and full-grid numerical analysis unchanged.'})
    (output/'index.html').write_text('\n'.join(parts))
    return figures


def link_existing_galleries(analysis,output,results):
    """Add a reversible pointer only to derived review pages, never raw exports."""
    analysis,output,results=Path(analysis).resolve(),Path(output).resolve(),Path(results).resolve()
    summary=json.loads((analysis/'summary.json').read_text())
    folders=[analysis,results/'rq1-input-stage-summary-2026-10-16']
    folders += [results/(s['run']+'-review') for s in summary['sources']]
    for folder in folders:
        page=folder/'index.html'
        if not page.exists():continue
        content=page.read_text()
        if '<!-- rq1-padding-gallery-link -->' in content:continue
        backup=folder/'index-before-padding-aware-gallery.html'
        if backup.exists():raise ValueError('Existing review-page backup: '+str(backup))
        backup.write_text(content)
        link=html.escape(os.path.relpath(output/'index.html',folder),quote=True)
        banner=('<!-- rq1-padding-gallery-link --><div style="background:#eef7fa;border:2px solid #00a8bb;padding:14px;margin:16px 0;font:17px system-ui">'
            f'<b>Clearer figures are available:</b> <a href="{link}">Open the padding-aware gallery</a>. '
            'It separates photo, mixed-edge and padding patches, explains E/I markers and shares GQA pair color scales. '
            'The original figures below remain preserved.</div>')
        page.write_text(content.replace('<h1>',banner+'<h1>',1))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',type=Path,default=REPO/'research/results/rq1-saved-analysis-v1-2026-10-17')
    p.add_argument('--results',type=Path,default=REPO/'research/results')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--link-existing',action='store_true',help='Add pointers and backups to existing derived review pages')
    a=p.parse_args();render(a.analysis,a.output,a.results)
    if a.link_existing:link_existing_galleries(a.analysis,a.output,a.results)
