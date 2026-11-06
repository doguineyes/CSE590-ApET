"""Explicit no-basis selector ablations. No OCR/answer-box dependencies."""
import hashlib
import random
import numpy as np
from rq2_protocol import require
from rq2_staged_protocol import VERSION as MASK_VERSION,specification as old_spec
VERSION='selector-grid-v1'
SEEDS=tuple(101*i for i in range(1,11))


def conditions(dataset):
    result=['baseline576','native-96-to-32']
    for selector in ('random','nonpad','detail','error'):
        for compensation in ('drop','merge'):
            if dataset=='gqa' and compensation=='merge':continue
            seeds=SEEDS[:10 if selector=='nonpad' and compensation=='drop' and dataset=='textvqa' else 5] if selector in ('random','nonpad') else (0,)
            result += [f'grid-{selector}-{compensation}-s{s}' for s in seeds]
    return result


def specification(condition):
    if not condition.startswith('grid-'):return old_spec(condition)
    _,selector,compensation,seed=condition.split('-')
    require(selector in ('random','nonpad','detail','error') and compensation in ('drop','merge'),'Unknown arm')
    seed=int(seed[1:]);require(seed in (SEEDS if selector in ('random','nonpad') else (0,)),'Invalid seed')
    return dict(method='selector-ablation',selector=selector,compensation=compensation,
        input_tokens=96,decoder_tokens=32,layer=16,random_seed=seed or None)


def permutation(image_sha,seed,stage,size):
    require(seed in SEEDS and stage in ('input','decoder'),'Bad seed/stage')
    boundary='input' if stage=='input' else 'after-block16-pool96'
    digest=hashlib.sha256(f'{MASK_VERSION}|{image_sha}|{seed}|{boundary}'.encode()).hexdigest()
    order=list(range(size));random.Random(int(digest,16)).shuffle(order)
    return order


def choose(selector,budget,padding,detail,errors,image_sha,seed,stage):
    n=len(padding); require(0<budget<=n,'Invalid budget')
    if selector in ('random','nonpad'):
        order=permutation(image_sha,seed,stage,n)
        if selector=='nonpad':
            # Stable sort preserves random tie order; fill least-padded if needed.
            order=sorted(order,key=lambda i:0.0 if padding[i]<=1e-9 else float(padding[i]))
    else:
        score=np.asarray(detail if selector=='detail' else errors,dtype=float)
        require(score.shape==(n,) and np.isfinite(score).all(),'Invalid selector scores')
        order=sorted(range(n),key=lambda i:(-score[i],i))
    selected=sorted(order[:budget])
    require(len(set(selected))==budget,'Duplicate selections')
    if selector=='nonpad':require(sum(padding[i]<=1e-9 for i in selected)==min(budget,sum(np.asarray(padding)<=1e-9)), 'Nonpad eligibility failure')
    return selected
