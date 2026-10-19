"""Project-owned selection intervention; original native compression stays pinned.

Only native encode_images' top-k index result is replaced for the random control.
Native FPS, reconstruction, basis replacement, merging and projection still run.
The already validated v2 capture/clone-replay implementation is reused unchanged.
"""
from contextlib import contextmanager
import copy
import sys

from rq1_native import replaced
from rq2_native import tensor_fingerprint, snapshot_encoded
from rq2_frozen_native import (capture_encoding, generate_from_snapshot, validate_snapshot,
    _snapshot_fingerprints, _runtime_scope)
from rq2_crossed_protocol import (CONDITIONS, ORACLE_TARGET, random_indices,
                                  assert_selector_identity, require)


@contextmanager
def selection_override(reference, ranked):
    """Scoped top-k shim restricted to one call from the pinned native function."""
    import torch
    from llava.model.llava_arch import LlavaMetaForCausalLM
    original = torch.topk
    code = LlavaMetaForCausalLM.encode_images.__code__
    calls = []
    require(len(ranked) == len(set(ranked)) == 288 and all(0 <= i < 576 for i in ranked),
            'Random selector needs 288 unique valid coordinates')

    def override(values, *args, **kwargs):
        if sys._getframe(1).f_code is not code:
            return original(values, *args, **kwargs)
        k = kwargs.get('k', args[0] if args else None)
        dim = kwargs.get('dim', args[1] if len(args) > 1 else None)
        require(k == 288 and dim == 1 and tuple(values.shape) == (1, 576),
                'Unexpected native selector invocation')
        require(not calls, 'Native random selector invoked more than once')
        native = original(values, *args, **kwargs)
        require(values[0].detach().cpu().tolist() == reference['input']['errors']
                and native[1][0].detach().cpu().tolist() == reference['input']['selected_indices'],
                'Native E/top-k changed before random selector intervention')
        indices = torch.tensor([ranked], device=values.device, dtype=native[1].dtype)
        calls.append({'native_E_ranked': native[1][0].detach().cpu().tolist(), 'used_ranked': ranked})
        # encode_images reads [1]; keep [0] consistent with the new index result.
        return values.gather(1, indices), indices

    with replaced(torch, 'topk', override):
        yield calls
    require(len(calls) == 1, 'Native random selector was not executed exactly once')


def capture_condition(model, pixels, archived_seed, condition, reference=None, image_sha256=None):
    specification = CONDITIONS[condition]
    arm = specification['replay_arm']
    selected, selection = None, None
    if specification['selection'] == 'random':
        require(reference is not None and image_sha256 is not None, 'Random selector needs the native E reference')
        selected, selection = random_indices(reference, image_sha256, archived_seed, specification['random_seed'])
        with selection_override(reference, selected) as calls:
            observation, snapshot = capture_encoding(model, pixels, archived_seed, arm)
        selection['native_selector_calls'] = len(calls)
    else:
        observation, snapshot = capture_encoding(model, pixels, archived_seed, arm)
    controls = None
    if reference is not None and arm != 'baseline576':
        controls = assert_selector_identity(reference, observation, selected)
    metadata = {'condition': condition, **specification, 'random_selection': selection,
                'cross_selector_identity': controls}
    # Keep native observation unchanged. Experimental ownership lives separately.
    snapshot['experiment'] = metadata
    return observation, snapshot


def generate_condition(tokenizer, model, processor, image, case, pixels, condition, snapshot, seed, target):
    specification = CONDITIONS[condition]
    row = generate_from_snapshot(tokenizer, model, processor, image, case, pixels,
                                specification['replay_arm'], snapshot, seed, target)
    row.update(condition=condition, native_replay_arm=row['arm'], arm=condition,
               selection=specification['selection'], selection_seed=specification['random_seed'],
               merge_enabled=specification['merge'], selection_intervention=copy.deepcopy(snapshot['experiment']))
    return row


def oracle_snapshot(model, pixels, native_snapshot, seed, target=None):
    """Restore one native pre-merge target, splice only its projected state.

    This post-hoc intervention never changes the selection/mask. Unchanged output
    coordinates remain byte exact by construction and are checked explicitly.
    """
    import torch
    validate_snapshot(native_snapshot, 'native-input288', pixels)
    observation = native_snapshot['capture']['encoding']
    merge = observation['input_merge']
    if target is not None:
        require(target in merge['target_indices'], 'Oracle target must be an ordinary retained merge destination')
    result = {k: tuple(t.clone() for t in value) if k == 'encoded'
              else value.clone() if k in ('pre_merge', 'post_merge') else copy.deepcopy(value)
              for k, value in native_snapshot.items()}
    with _runtime_scope(model, 'native-input288', seed), torch.inference_mode():
        noop_projected = model.get_model().mm_projector(native_snapshot['post_merge'].clone())
        require(torch.equal(noop_projected, native_snapshot['encoded'][0]),
                'Oracle native projector no-op changed frozen features')
        if target is not None:
            restored = native_snapshot['post_merge'].clone()
            restored[:, target] = native_snapshot['pre_merge'][:, target]
            projected = model.get_model().mm_projector(restored)
    if target is not None:
        require(projected.shape == result['encoded'][0].shape, 'Oracle projector output shape changed')
        with torch.inference_mode():
            result['encoded'][0][:, target] = projected[:, target]
            result['post_merge'][:, target] = native_snapshot['pre_merge'][:, target]
    untouched = [i for i in range(576) if i != target]
    require(torch.equal(result['encoded'][0][:, untouched], native_snapshot['encoded'][0][:, untouched])
            and torch.equal(result['encoded'][1], native_snapshot['encoded'][1]),
            'Oracle changed a non-target projected state or mask')
    changed = (result['encoded'][0] != native_snapshot['encoded'][0]).any(dim=-1)[0].nonzero().flatten().tolist()
    require(set(changed) <= ({target} if target is not None else set()), 'Unexpected oracle changed positions')
    # Fingerprints describe the intervention output, not an unmodified native run.
    result['capture']['encoding'] = copy.deepcopy(observation)
    result['capture']['encoding']['encoded'] = snapshot_encoded(result['encoded'])
    result['capture']['encoding']['post_merge'] = tensor_fingerprint(result['post_merge'])
    if target is not None:
        result['capture']['encoding']['project_owned_oracle'] = {'restored_target': target}
    result['capture']['fingerprints'] = _snapshot_fingerprints(result)
    result['capture']['oracle'] = {'target': target, 'changed_projected_positions': changed,
        'untouched_coordinates_exact': True, 'mask_exact': True, 'projector_noop_exact': True,
        'kind': 'post-hoc oracle; not unchanged native ApET or a deployable policy'}
    validate_snapshot(result, 'native-input288', pixels)
    return result


def matched_oracle_control(native_observation):
    merge = native_observation['input_merge']
    lookup = dict(zip(merge['target_indices'], merge['merge_counts']))
    require(ORACLE_TARGET in lookup, 'Prespecified Brolio target is not a native merge target')
    return min((i for i in lookup if i != ORACLE_TARGET),
               key=lambda i: (abs(lookup[i]-lookup[ORACLE_TARGET]), i))
