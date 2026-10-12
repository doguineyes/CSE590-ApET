Today I focused on reading script codes and understanding the overall evaluation architecture rather than modifying code.

## eval-mmstar.py:

MMStar dataset
    ↓
eval_mmstar.py
    ↓
load_llava()
    ↓
HF LLaVA model + processor
    ↓
optional HFInputCompressionAdapter
    ↓
generate_answer()
    ↓
parse A/B/C/D
    ↓
save JSONL predictions + summary report

## llava_runtime.py:

image + question
    ↓
build LLaVA prompt
    ↓
processor(...)
    ↓
input_ids + attention_mask + pixel_values
    ↓
if ApET:
    adapter.prepare_inputs(...)
    ↓
model.generate(...)
    ↓
generated token IDs
    ↓
processor.decode(...)
    ↓
text answer
    ↓
record timing / token counts / GPU memory

## adapters and core
