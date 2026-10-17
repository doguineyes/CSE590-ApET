"""Project-only decoder hooks; original ApET reconstruction/forward stay intact.

E is observed by executing native layer_prune on protected copies and discarding
its shortened output. I changes one block-output vector during a fresh full
forward. No cached suffix, early exit, new attention path, or native source edit.
"""

from contextlib import contextmanager
import time
from types import SimpleNamespace

from original_diagnostics import observe_native_values
from rq1_native import (max_score_delta, prepare_question, score_embeddings,
                        synchronize, tensor_sha256, zero_patch)


@contextmanager
def boundary_hook(model, prepared, layer=16, *, observer=None, token_index=None, noop=False):
    """After block `layer`, inspect or replace output[0]; restore even on failure."""
    import torch
    if not 1 <= layer <= len(model.model.layers):
        raise ValueError('Decoder boundary is outside this model')
    if model.training or model.model.layer_list or model.visual_token_num != 576:
        raise ValueError('Layer measurement requires eval mode and uncompressed baseline')
    expected = tuple(prepared['embeddings'].shape)
    seen = {'calls': 0}

    def hook(module, args, output):
        seen['calls'] += 1
        if seen['calls'] != 1 or not isinstance(output, tuple) or not output:
            raise RuntimeError('Expected one full-forward tuple at the decoder boundary')
        hidden = output[0]
        if tuple(hidden.shape) != expected or not torch.isfinite(hidden).all():
            raise RuntimeError('Unexpected or non-finite decoder boundary states')
        if observer is not None:
            observer(hidden)
        if token_index is not None or noop:
            changed = zero_patch(hidden, prepared['image_start'], token_index)
            if noop and not torch.equal(changed, hidden):
                raise RuntimeError('Layer clone-only no-op changed hidden states')
            return (changed, *output[1:])
        return None  # Observation returns the exact native output object.

    handle = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        yield seen
        if seen['calls'] != 1:
            raise RuntimeError('Decoder boundary was not visited exactly once')
    finally:
        handle.remove()


def observe_layer_errors(model, prepared, hidden, seed, layer=16, keep=288, basis=10):
    """Run the ORIGINAL method on an isolated receiver/tensors, preserving RNG."""
    import torch
    from llava.model.modeling_llama_x import LlamaModel
    if tuple(hidden.shape) != tuple(prepared['embeddings'].shape):
        raise ValueError('Layer observation requires the original complete sequence')
    # Native layer_prune reads these fields only. Avoid changing the live model's
    # image metadata/budgets. In particular, never add layer 16 to layer_list.
    proxy = SimpleNamespace(config=model.model.config, image_tokens=[576],
                            image_token_posi=[prepared['image_start']],
                            image_token_list=[576, keep], basis_token_num=basis)
    before = tensor_sha256(hidden)
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with observe_native_values() as observed, torch.inference_mode():
            result = LlamaModel.layer_prune(
                proxy, cur_num=0, rank_layer=layer, features=hidden.detach().clone(),
                position_ids=prepared['position_ids'].to(hidden.device).clone(),
                attention_mask=prepared['attention_mask'].to(hidden.device).clone(),
                labels=prepared['labels'].to(hidden.device).clone(), selected_indices=[],
                sum_visual_attention=[], indices_attention=[])
    if tensor_sha256(hidden) != before:
        raise RuntimeError('Native E observation mutated the live decoder states')
    if len(observed['decoder']) != 1 or len(observed['decoder_merge']) != 1:
        raise RuntimeError('Native decoder reconstruction/merge was not observed once')
    stage, merge = observed['decoder'][0], observed['decoder_merge'][0]
    expected_shape = (1, hidden.shape[1] - 576 + keep, hidden.shape[2])
    if (stage['layer'] != layer or stage['source_tokens'] != 576
            or len(stage['basis_indices']) != basis or len(set(stage['basis_indices'])) != basis
            or len(stage['selected_indices']) != keep or tuple(result[2].shape) != expected_shape):
        raise RuntimeError('Native layer observation geometry/budget mismatch')
    stage['basis_slot_to_source'] = list(zip(merge['basis_assignment_slots'], stage['basis_indices']))
    return {'stage': stage, 'merge': merge, 'seed': seed, 'keep_tokens_observation_only': keep,
            'compressed_output_discarded': True, 'live_hidden_sha256_before_after': before,
            'rng_restored': True, 'native_method': 'llava.model.modeling_llama_x.LlamaModel.layer_prune'}


def measure_layer(model, tokenizer, case, encoded, input_record, entry, order, tolerance, progress,
                  save_boundary, layer=16):
    """Fresh layer forward controls, then one intervention per fixed patch index."""
    import torch
    prepared = prepare_question(tokenizer, model, case, encoded)
    if prepared['target_token_ids'] != entry['target_token_ids']:
        raise RuntimeError('Layer reference target differs; retargeting is forbidden')
    if input_record['status'] != 'passed' or not input_record['controls']['passed']:
        raise RuntimeError('Fresh input/reference controls must pass first')
    reference = input_record['controls']['reference']
    started = time.perf_counter()
    record = {'case_id': case['case_id'], 'status': 'running', 'stage': 'decoder', 'layer': layer,
              'module_index': layer - 1, 'boundary': f'after_block_{layer}_before_block_{layer+1}',
              'target_token_ids': prepared['target_token_ids'], 'measurement_target': input_record['measurement_target'],
              'image_expanded_start': prepared['image_start'], 'target_expanded_start': prepared['target_start'],
              'expanded_sequence_length': prepared['embeddings'].shape[1], 'token_execution_order': order,
              'score_definition': 'I = full uncompressed mean reference-token log p - single block-output zeroing score; no EOS',
              'controls': {'reference': reference, 'tolerance_per_answer_token': tolerance}, 'tokens': []}
    controls = record['controls']
    progress(record)
    capture = {}

    def observe(hidden):
        capture['hidden'] = hidden.detach().clone()
        capture['diagnostics'] = observe_layer_errors(model, prepared, hidden, encoded['seed'], layer)

    with boundary_hook(model, prepared, layer, observer=observe):
        controls['observation_only'] = score_embeddings(model, prepared)
    hidden = capture.pop('hidden')
    record['diagnostics'] = capture['diagnostics']
    record['visual_hidden_sha256'] = tensor_sha256(hidden[:, prepared['image_start']:prepared['image_start']+576])
    record['boundary_hidden_sha256'] = tensor_sha256(hidden)
    record['boundary_shape'] = list(hidden.shape)
    record['boundary_feature_file'] = save_boundary(hidden, record)
    controls['reference_after_E_observation'] = score_embeddings(model, prepared)
    controls['reference_repeat'] = score_embeddings(model, prepared)
    with boundary_hook(model, prepared, layer, noop=True):
        controls['noop_at_boundary'] = score_embeddings(model, prepared)

    def check_repeat(value):
        if not torch.equal(value, hidden):
            raise RuntimeError('Repeated decoder boundary states changed before interventions')

    with boundary_hook(model, prepared, layer, observer=check_repeat):
        controls['boundary_repeat'] = score_embeddings(model, prepared)
    controls['boundary_repeat_exact'] = True
    controls['max_pre_token_logprob_delta'] = max(max_score_delta(reference, controls[k]) for k in
        ['observation_only', 'reference_after_E_observation', 'reference_repeat', 'noop_at_boundary', 'boundary_repeat'])
    controls['passed'] = controls['max_pre_token_logprob_delta'] <= tolerance
    progress(record)
    if not controls['passed']:
        raise RuntimeError('Layer observation/reference/no-op parity failed; token loop not run')
    visual = hidden[:, prepared['image_start']:prepared['image_start']+576].clone()
    del hidden
    errors = record['diagnostics']['stage']['errors']
    for index in order:
        synchronize()
        tick = time.perf_counter()
        with boundary_hook(model, prepared, layer, token_index=index):
            score = score_embeddings(model, prepared)
        synchronize()
        record['tokens'].append({'token_index': index, 'patch_row': index//24, 'patch_col': index%24,
                                 'reconstruction_error': errors[index],
                                 'intervention_score': reference['mean_logprob']-score['mean_logprob'],
                                 'perturbed_score': score, 'perturbation_l2': float(visual[:, index].float().norm().item()),
                                 'seconds': time.perf_counter()-tick})
        progress(record)
    final = {}

    def observe_final(value):
        final['boundary_sha256'] = tensor_sha256(value)

    with boundary_hook(model, prepared, layer, observer=observe_final):
        controls['reference_after'] = score_embeddings(model, prepared)
    controls['final_boundary_exact'] = final['boundary_sha256'] == record['boundary_hidden_sha256']
    controls['post_token_logprob_delta'] = max_score_delta(reference, controls['reference_after'])
    controls['passed'] = (controls['passed'] and controls['final_boundary_exact']
                          and controls['post_token_logprob_delta'] <= tolerance)
    record['seconds'] = time.perf_counter()-started
    record['status'] = 'passed' if controls['passed'] else 'unstable_reference'
    progress(record)
    return record
