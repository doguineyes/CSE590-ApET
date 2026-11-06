"""Versioned TextVQA basis and stage-localization experiment, fixed96→32."""
from selector_grid_protocol import specification as grid_spec
from rq2_protocol import require
VERSION='basis-localization-v1'
SEEDS=(101,202,303,404,505)

def conditions(dataset='textvqa'):
    require(dataset=='textvqa','Localization is TextVQA only')
    return ['baseline576','native-96-to-32','grid-error-drop-s0','grid-detail-drop-s0',
            'loc-native-drop','loc-error-native','loc-native-error','loc-fps-fps','loc-error-fps','loc-error-detail','loc-detail-error']+[
            f'loc-error-random-s{s}' for s in SEEDS]

def specification(condition):
    if not condition.startswith('loc-'):return grid_spec(condition)
    require(condition in conditions(),'Unknown localization arm')
    parts=condition.split('-');first,last=parts[1:3]
    native=first=='native'
    return dict(method='basis-localization',selector='native' if native else first,
                decoder_selector='native' if last in ('drop','native') else last,compensation='drop',
                input_tokens=96,decoder_tokens=32,layer=16,
                random_seed=int(parts[3][1:]) if len(parts)==4 else None)
