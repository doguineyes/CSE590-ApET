"""Project routing interventions inside the pinned native merging call.

Native selection, basis replacement, FP16 scatter/normalization and projection
execute unchanged. Only the cosine result used to choose destinations is
overridden in one scoped native call. Observation and frozen replay are reused.
"""
from contextlib import contextmanager
import copy
import sys

from rq1_native import replaced
from rq2_native import tensor_fingerprint, assert_selection_equal
from rq2_frozen_native import capture_encoding, generate_from_snapshot
from rq2_routing_protocol import (CONDITIONS, routing_plan, assert_routing_identity, require)


@contextmanager
def routing_override(reference, plan):
    import torch
    import torch.nn.functional as functional
    from llava.model.llava_arch import LlavaMetaForCausalLM
    original = functional.cosine_similarity
    code = LlavaMetaForCausalLM.token_merging.__code__; calls = []
    targets = plan['target_indices']; target_position = {t:i for i,t in enumerate(targets)}

    def override(*args, **kwargs):
        if sys._getframe(1).f_code is not code:
            return original(*args, **kwargs)
        require(not calls, 'Routing override ran more than once')
        similarity = original(*args, **kwargs)
        require(tuple(similarity.shape) == (1, 288, 278) and bool(torch.isfinite(similarity).all()),
                'Unexpected native routing cosine geometry/values')
        native = similarity.argmax(dim=2)[0].detach().cpu().tolist()
        require([targets[i] for i in native] == plan['native_assigned_target_indices'],
                'Native nearest graph changed before routing intervention')
        changed = similarity.clone()
        if plan['routing'] == 'shuffle':
            positions = [target_position[t] for t in plan['assigned_target_indices']]
            changed.fill_(-1)
            changed.scatter_(2, torch.tensor([positions], device=similarity.device).unsqueeze(-1), 1)
        else:
            positions = [target_position[t] for t in plan['protected_targets']]
            changed[:, :, positions] = -torch.inf
        actual = changed.argmax(dim=2)[0].detach().cpu().tolist()
        chosen = similarity[0, torch.arange(288, device=similarity.device),
                            torch.tensor(actual, device=similarity.device)]
        calls.append({'native_similarity': tensor_fingerprint(similarity),
            'native_assigned_target_indices': [targets[i] for i in native],
            'actual_assigned_target_indices': [targets[i] for i in actual],
            'mean_original_cosine_at_new_destinations': float(chosen.float().mean().item()),
            'mean_original_cosine_at_native_destinations': float(similarity.max(dim=2)[0].float().mean().item()),
            'nearest_eligible_policy': plan['routing'] != 'shuffle'})
        return changed

    with replaced(functional, 'cosine_similarity', override):
        yield calls
    require(len(calls) == 1, 'Routing override did not execute exactly once')


def capture_condition(model, pixels, archived_seed, condition, reference=None, image_sha256=None):
    import torch
    specification = CONDITIONS[condition]; arm = specification['replay_arm']
    plan, controls, call = None, None, None
    if specification['routing'] in ('shuffle', 'protect-E', 'protect-random'):
        require(reference is not None and image_sha256 is not None, 'Routing needs native reference/image identity')
        plan = routing_plan(reference, image_sha256, archived_seed, condition)
        with routing_override(reference, plan) as calls:
            observation, snapshot = capture_encoding(model, pixels, archived_seed, arm)
        call = calls[0]
        require(call['actual_assigned_target_indices'] == observation['input_merge']['assigned_target_indices'],
                'Native merge did not consume the overridden routing')
        protected = plan['protected_targets']
        exact = torch.equal(snapshot['pre_merge'][:, protected], snapshot['post_merge'][:, protected])
        # Native merging may change only ordinary retained destinations. Basis
        # slots and discarded coordinates must retain their original values.
        untouched = sorted(set(range(576))-set(plan['target_indices']))
        require(torch.equal(snapshot['pre_merge'][:, untouched], snapshot['post_merge'][:, untouched]),
                'Routing changed a basis or discarded feature')
        controls = assert_routing_identity(reference, observation, plan, exact)
        controls.update(override_calls=1, non_target_features_exact=True)
    else:
        observation, snapshot = capture_encoding(model, pixels, archived_seed, arm)
        if condition == 'no-merge':
            require(reference is not None, 'No-merge needs the same native selection reference')
            controls = assert_selection_equal(reference, observation)
    snapshot['experiment'] = {'condition': condition, **specification,
        'plan': plan, 'routing_override': call, 'routing_identity': controls,
        'ownership': 'Project routing variant; original native sources unchanged'}
    return observation, snapshot


def generate_condition(tokenizer, model, processor, image, case, pixels, condition, snapshot, seed, target):
    specification = CONDITIONS[condition]
    row = generate_from_snapshot(tokenizer, model, processor, image, case, pixels,
                                specification['replay_arm'], snapshot, seed, target)
    row.update(condition=condition, native_replay_arm=row['arm'], arm=condition,
        routing=specification['routing'], routing_seed=specification['seed'],
        routing_intervention=copy.deepcopy(snapshot['experiment']))
    return row
