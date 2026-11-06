"""Fixed carrier-slot replacements; no answer-dependent selection."""
import hashlib
import numpy as np
from basis_localization_protocol import specification as parent_spec
from rq2_protocol import require
VERSION='norm-intervention-v1'
SEEDS=(101,202,303,404,505)
PARENTS={'a':'loc-native-drop','b':'loc-fps-fps','c':'loc-native-error'}
def conditions(dataset='textvqa'):
    require(dataset=='textvqa','TextVQA only')
    return ['baseline576',*PARENTS.values()]+['loc-norm-'+x for x in
        ['a1','a2','b1','c1','c2','a-fp32','b-fp32']+
        [f'{family}-s{s}' for family in ('a1c','a2c','b1c','c1c') for s in SEEDS]]
def parent(name):
    return PARENTS[name.split('-')[2][0]] if name.startswith('loc-norm-') else name
def specification(name):
    require(name in conditions(),'Unknown norm arm')
    return parent_spec(parent(name))
def swap_plan(name,norms,sources,carriers,basis_slots,errors,key,fps_replacement=None):
    """Return output-slot / old-source / new-source triples. Slots never move."""
    kind=name.split('-')[2];norms=np.asarray(norms,dtype=float)
    require(len(sources)==len(carriers)==32 and len(set(carriers))==32,'Invalid parent slots')
    require(np.isfinite(norms).all() and len(norms)==96,'Invalid pool norms')
    high=set(np.flatnonzero(norms>10*np.median(norms)).tolist());present=set(sources)
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256((key+'|'+name).encode()).digest()[:8],'big'))
    count=2 if kind in ('a2','a2c') else 1
    if kind.startswith('c'):
        eligible=[i for i in range(96) if i not in present and i not in high]
        if kind=='c1c':adds=rng.choice(eligible,1,replace=False).tolist()
        elif kind=='c1':adds=[int(np.argmax(norms))] if int(np.argmax(norms)) not in present else []
        else:adds=sorted(high-present,key=lambda i:(-norms[i],i))
        slots=sorted(range(32),key=lambda j:(errors[sources[j]],sources[j],j))[:len(adds)]
    else:
        if kind.endswith('c'):
            eligible=[j for j,s in enumerate(sources) if s not in high and
                      (not kind.startswith('a') or carriers[j] in basis_slots)]
            require(len(eligible)>=count,'Insufficient matched low-norm control slots')
            slots=sorted(rng.choice(eligible,count,replace=False).tolist())
        else:slots=sorted(range(32),key=lambda j:(-norms[sources[j]],j))[:count]
        if kind.startswith('b'):
            require(fps_replacement is not None,'Missing FPS replacement');adds=[fps_replacement(slots)]
        else:
            adds=[i for i in sorted(range(96),key=lambda i:(-errors[i],i)) if i not in present][:count]
    require(len(slots)==len(adds) and len(set(adds))==len(adds) and not (set(adds)&present),'Invalid replacement')
    return [(j,int(sources[j]),int(src)) for j,src in zip(slots,adds)]
