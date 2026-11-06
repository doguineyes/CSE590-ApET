"""New seed experiment policy; archived staged policy remains byte-for-byte intact."""
from contextlib import contextmanager
import hashlib
import json
import random
from rq2_protocol import require
from rq2_staged_protocol import VERSION as MASK_VERSION, specification as old_specification

VERSION = 'textvqa-seed-coverage-v1'
SEEDS = tuple(101 * i for i in range(1, 21))
CALIBRATION_SEEDS = SEEDS[:4]


def specification(condition):
    if condition.startswith('pure-random-96-to-32-s'):
        seed = int(condition.rsplit('s', 1)[1])
        require(seed in SEEDS, 'Unregistered seed')
        return dict(method='pure-random', input_tokens=96, decoder_tokens=32,
                    layer=16, random_seed=seed)
    return old_specification(condition)


def random_plan(image_sha256, replicate, input_tokens, decoder_tokens):
    require(len(image_sha256) == 64 and all(c in '0123456789abcdef' for c in image_sha256), 'Invalid image hash')
    require(type(replicate) is int and replicate in SEEDS, 'Unregistered seed')
    require((input_tokens, decoder_tokens) in ((96, 32), (576, 576)), 'Unsupported schedule')
    def permutation(size, boundary):
        digest = hashlib.sha256(f'{MASK_VERSION}|{image_sha256}|{replicate}|{boundary}'.encode()).hexdigest()
        order = list(range(size))
        random.Random(int(digest, 16)).shuffle(order)
        return order, digest
    first, input_seed = permutation(576, 'input')
    selected = sorted(first[:input_tokens])
    second, decoder_seed = permutation(input_tokens, f'after-block16-pool{input_tokens}')
    compact = sorted(second[:decoder_tokens])
    return dict(version=MASK_VERSION, replicate=replicate, input_original_positions=selected,
        decoder_compact_positions=compact, decoder_original_positions=[selected[i] for i in compact],
        input_seed_sha256=input_seed, decoder_seed_sha256=decoder_seed,
        input_permutation_sha256=hashlib.sha256(json.dumps(first).encode()).hexdigest(),
        decoder_permutation_sha256=hashlib.sha256(json.dumps(second).encode()).hexdigest(),
        policy='Uniform without replacement at each stage; independent stage RNG streams; spatial order restored. Layer16 states are contextualized, not isolated original patches.')


@contextmanager
def adapter_policy():
    # Local, single-process extension. Never mutate the old protocol's registry or files.
    import rq2_staged_native as native
    from rq1_native import replaced
    with replaced(native, 'specification', specification), replaced(native, 'random_plan', random_plan):
        yield
