"""IH released-style physical retention, adapted to unchanged ApET native SDPA.

Temporary pre-hooks shorten the sequence before block 17 and adapt masks/positions
for all remaining native blocks. The first 16 blocks run normally every time.
No native source changes or independent decoder/suffix implementation.
"""

from contextlib import contextmanager
import time

from rq1_layer_ih import kept_positions, probability
from rq1_layer_native import boundary_hook
from rq1_native import (max_score_delta, prepare_question, score_embeddings, score_from_logits,
                        synchronize, tensor_sha256)


@contextmanager
def retained_visual_hook(model, prepared, retained, *, layer=16, expected_boundary=None):
    import torch
    retained = list(retained)
    if (model.training or model.model.layer_list or model.visual_token_num != 576
            or not 1 <= layer < len(model.model.layers)
            or not model.model._use_sdpa or getattr(model.config, 'rope_scaling', None) is not None
            or not bool(prepared['attention_mask'].all())):
        raise ValueError('IH requires uncompressed eval, native SDPA, unscaled RoPE and an unpadded reference')
    shape = tuple(prepared['embeddings'].shape)
    length = shape[1]
    keep = kept_positions(length, prepared['image_start'], retained)
    n = len(keep)
    if not torch.equal(prepared['position_ids'][0], torch.arange(length, device=prepared['position_ids'].device)):
        raise ValueError('IH expects the existing contiguous original position mapping')
    seen = {'blocks': [], 'rotary': [], 'kept_positions': keep,
            'retained_visual': sorted(retained), 'shortened_length': n}
    handles = []

    def before_block(index):
        def hook(module, args, kwargs):
            if not args or kwargs.get('use_cache') or kwargs.get('past_key_value') is not None:
                raise RuntimeError('IH expects fresh noncached native decoder forwards')
            hidden = args[0]
            if index == layer:
                if tuple(hidden.shape) != shape:
                    raise RuntimeError('Wrong pre-retention boundary shape')
                if expected_boundary is not None and not torch.equal(hidden, expected_boundary.to(hidden.device)):
                    raise RuntimeError('Pre-retention states differ from the full reference')
                hidden = hidden.index_select(1, torch.tensor(keep, dtype=torch.long, device=hidden.device))
            elif hidden.shape[1] != n:
                raise RuntimeError('Downstream block did not receive the shortened sequence')
            changed = dict(kwargs)
            changed['position_ids'] = torch.tensor([keep], dtype=torch.long, device=hidden.device)
            mask = changed.get('attention_mask')
            if mask is not None:
                if mask.ndim != 4 or mask.shape[-2:] != (length, length):
                    raise RuntimeError('Expected native unpadded full causal mask or None')
                # Exactly the released causal-prefix crop; this is valid only for
                # this unpadded reference. RoPE position gaps remain unchanged.
                changed['attention_mask'] = mask[:, :, :n, :n].to(hidden.device)
            seen['blocks'].append(index)
            return (hidden, *args[1:]), changed
        return hook

    def before_rotary(index):
        def hook(module, args, kwargs):
            # Old ApET slices RoPE tables to kv length, which would be too short
            # for original gapped positions after deletion. Extend lookup extent
            # without changing the native RoPE values, weights or position IDs.
            if kwargs.get('seq_len') != n:
                raise RuntimeError('Unexpected native RoPE/cache length')
            changed = dict(kwargs); changed['seq_len'] = length
            seen['rotary'].append(index)
            return args, changed
        return hook

    try:
        for index in range(layer, len(model.model.layers)):
            block = model.model.layers[index]
            handles.append(block.register_forward_pre_hook(before_block(index), with_kwargs=True))
            handles.append(block.self_attn.rotary_emb.register_forward_pre_hook(before_rotary(index), with_kwargs=True))
        yield seen
        expected = list(range(layer, len(model.model.layers)))
        if seen['blocks'] != expected or seen['rotary'] != expected:
            raise RuntimeError('IH did not traverse each downstream block and RoPE lookup exactly once')
    finally:
        for handle in reversed(handles): handle.remove()


def score_retained(model, prepared, retained, *, layer=16, expected_boundary=None):
    import torch
    with retained_visual_hook(model, prepared, retained, layer=layer, expected_boundary=expected_boundary) as trace:
        with torch.inference_mode():
            output = model(inputs_embeds=prepared['embeddings'], position_ids=prepared['position_ids'],
                attention_mask=prepared['attention_mask'], use_cache=False, return_dict=True)
            labels = prepared['labels'].index_select(1, torch.tensor(
                trace['kept_positions'], dtype=torch.long, device=prepared['labels'].device))
            if output.logits.shape[1] != trace['shortened_length']:
                raise RuntimeError('Physical deletion did not shorten final logits')
            score = score_from_logits(output.logits, labels)
    return score, trace


def measure_ih(model, tokenizer, case, encoded, layer_record, order, tolerance, progress, layer=16):
    import torch
    prepared = prepare_question(tokenizer, model, case, encoded)
    if prepared['target_token_ids'] != layer_record['target_token_ids'] or order != layer_record['token_execution_order']:
        raise ValueError('IH reference targets/positions differ from singleton zeroing')
    reference = layer_record['controls']['reference']
    controls = {'reference': reference, 'tolerance_per_answer_token': tolerance, 'passed': False}
    record = {'case_id': case['case_id'], 'status': 'running', 'stage': 'ih', 'layer': layer,
        'operator': 'physical-retain-one-original-positions', 'window_size': 1,
        'target_token_ids': prepared['target_token_ids'],
        'first_reference_token_string': tokenizer.convert_ids_to_tokens(prepared['target_token_ids'][0]),
        'token_execution_order': order, 'controls': controls, 'tokens': [],
        'score_definition': 'I_IH = p(first fixed gold token | retain one at boundary) - p(first fixed gold token | remove all visual at boundary)',
        'position_policy': 'Original kept position IDs; shortened causal mask; native RoPE lookup extends to original length',
        'upstream_context': 'Full native context before boundary; all nonvisual states retained unchanged at boundary'}
    started = time.perf_counter(); capture = {}
    with boundary_hook(model, prepared, layer=layer, observer=lambda h: capture.update(hidden=h.detach().clone())):
        controls['observation_reference'] = score_embeddings(model, prepared)
    boundary = capture['hidden']
    record['pre_retention_boundary_sha256'] = tensor_sha256(boundary)
    if record['pre_retention_boundary_sha256'] != layer_record['boundary_hidden_sha256']:
        raise RuntimeError('IH full upstream states differ from the saved zeroing boundary')
    controls['retain_all_noop'], noop_trace = score_retained(model, prepared, list(range(576)),
        layer=layer, expected_boundary=boundary)
    controls['remove_all_before'], empty_trace = score_retained(model, prepared, [], layer=layer, expected_boundary=boundary)
    controls['remove_all_repeat'], _ = score_retained(model, prepared, [], layer=layer, expected_boundary=boundary)
    controls['reference_before'] = score_embeddings(model, prepared)
    controls['max_reference_delta'] = max(max_score_delta(reference, controls[k]) for k in
        ['observation_reference', 'retain_all_noop', 'reference_before'])
    controls['remove_all_repeat_delta'] = max_score_delta(controls['remove_all_before'], controls['remove_all_repeat'])
    controls['passed'] = max(controls['max_reference_delta'], controls['remove_all_repeat_delta']) <= tolerance
    record['remove_all_probability'] = probability(controls['remove_all_before'])
    record['remove_all_shortened_length'] = empty_trace['shortened_length']
    record['remove_all_kept_positions'] = empty_trace['kept_positions']
    progress(record)
    if not controls['passed']: raise RuntimeError('IH keep-all/reference/remove-all controls failed')
    for index in order:
        synchronize(); tick = time.perf_counter()
        score, trace = score_retained(model, prepared, [index], layer=layer, expected_boundary=boundary)
        synchronize()
        record['tokens'].append({'token_index': index, 'patch_row': index//24, 'patch_col': index%24,
            'reconstruction_error': layer_record['diagnostics']['stage']['errors'][index],
            'retained_score': score, 'retained_probability': probability(score),
            'ih_first_probability': probability(score)-record['remove_all_probability'],
            'shortened_length': trace['shortened_length'], 'kept_positions': trace['kept_positions'],
            'pre_retention_boundary_exact': True, 'seconds': time.perf_counter()-tick})
        progress(record)
    controls['first_retained_repeat'], _ = score_retained(model, prepared, [order[0]], layer=layer, expected_boundary=boundary)
    controls['first_retained_repeat_delta'] = max_score_delta(record['tokens'][0]['retained_score'], controls['first_retained_repeat'])
    controls['remove_all_after'], _ = score_retained(model, prepared, [], layer=layer, expected_boundary=boundary)
    controls['remove_all_after_delta'] = max_score_delta(controls['remove_all_before'], controls['remove_all_after'])
    final = {}
    with boundary_hook(model, prepared, layer=layer, observer=lambda h: final.update(hash=tensor_sha256(h))):
        controls['reference_after'] = score_embeddings(model, prepared)
    controls['final_reference_delta'] = max_score_delta(reference, controls['reference_after'])
    controls['final_boundary_exact'] = final['hash'] == record['pre_retention_boundary_sha256']
    controls['passed'] = controls['passed'] and controls['final_boundary_exact'] and max(
        controls['first_retained_repeat_delta'], controls['remove_all_after_delta'], controls['final_reference_delta']) <= tolerance
    record['seconds'] = time.perf_counter()-started
    record['status'] = 'passed' if controls['passed'] else 'unstable_reference'
    progress(record)
    return record
