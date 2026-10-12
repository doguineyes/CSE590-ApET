Today I focused on reading scripts/eval_mmstar.py and understanding the overall evaluation architecture rather than modifying code.

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