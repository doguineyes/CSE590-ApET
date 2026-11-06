"""Geometry and descriptive statistics for the fixed 336/14 CLIP canvas."""
import numpy as np


def canvas_box(box,width,height):
    side=max(width,height); left=(side-width)//2; top=(side-height)//2
    x,y,w,h=[float(box[k]) for k in ('top_left_x','top_left_y','width','height')]
    if not all(np.isfinite([x,y,w,h])) or w<=0 or h<=0 or x<-.02 or y<-.02 or x+w>1.02 or y+h>1.02:
        raise ValueError('OCR box outside supported bounds')
    return [336*(left+max(0,x)*width)/side,336*(top+max(0,y)*height)/side,
            336*(left+min(1,x+w)*width)/side,336*(top+min(1,y+h)*height)/side]


def patch_overlap(box):
    x0,y0,x1,y1=box
    x=np.arange(24)*14;y=np.arange(24)*14
    return (np.maximum(0,np.minimum(y+14,y1)-np.maximum(y,y0))[:,None]*
            np.maximum(0,np.minimum(x+14,x1)-np.maximum(x,x0))[None,:]).reshape(-1)/196


def padding_fraction(width,height):
    return 1-patch_overlap(canvas_box(dict(top_left_x=0,top_left_y=0,width=1,height=1),width,height))


def detail_energy(pixels):
    if pixels.shape!=(1,3,336,336):raise ValueError('Wrong pixel shape')
    # Differences only inside each patch; exclude artificial borders between patches.
    p=pixels.astype(np.float64).reshape(3,24,14,24,14).transpose(1,3,0,2,4)
    return (np.abs(np.diff(p,axis=3)).mean(axis=(2,3,4))+
            np.abs(np.diff(p,axis=4)).mean(axis=(2,3,4))).reshape(-1)


def ranks(a):
    a=np.asarray(a);order=np.argsort(a,kind='stable');out=np.empty(len(a),float)
    i=0
    while i<len(a):
        j=i+1
        while j<len(a) and a[order[j]]==a[order[i]]:j+=1
        out[order[i:j]]=(i+j-1)/2;i=j
    return out


def spearman(a,b):
    a,b=ranks(a),ranks(b)
    if np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def overlap_coverage(indices,area):
    relevant=set(np.flatnonzero(area>0).tolist())
    if not relevant:return None
    return len(set(indices)&relevant)/len(relevant)


def source_paths(selected,basis_map,graph,upstream):
    """Trace direct carrier and merge paths, not information or semantic preservation."""
    basis={int(slot):int(source) for slot,source in basis_map}
    direct={i:set(upstream[basis.get(i,i)]) for i in selected}
    combined={i:set(v) for i,v in direct.items()}
    for src,target in zip(graph['dropped_indices'],graph['assigned_target_indices']):
        combined[target].update(upstream[src])
    return direct,combined


def detail_without_padding(pixels,width,height):
    """Within-patch gradients with both pixel footprints wholly inside real image.

    Geometric exclusion also avoids pad/image transition edges. Each direction is
    averaged over its valid differences; patches with none receive zero.
    """
    if pixels.shape!=(1,3,336,336):raise ValueError('Wrong pixel shape')
    rect=canvas_box(dict(top_left_x=0,top_left_y=0,width=1,height=1),width,height)
    x=np.arange(336); y=np.arange(336)
    valid=((y>=rect[1])&(y+1<=rect[3]))[:,None]&((x>=rect[0])&(x+1<=rect[2]))[None,:]
    v=valid.reshape(24,14,24,14).transpose(0,2,1,3)
    p=pixels.astype(np.float64).reshape(3,24,14,24,14).transpose(1,3,0,2,4)
    out=np.zeros((24,24))
    for axis in (2,3):
        if axis==2:mask=v[:,:,:-1,:]&v[:,:,1:,:]
        else:mask=v[:,:,:,:-1]&v[:,:,:,1:]
        diff=np.abs(np.diff(p,axis=axis+1)).mean(axis=2)
        total=(diff*mask).sum(axis=(2,3)); count=mask.sum(axis=(2,3))
        out+=np.divide(total,count,out=np.zeros_like(total),where=count>0)
    return out.reshape(-1)
