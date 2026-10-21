"""Download the pinned MMStar parquet and restore the unchanged full1500 inputs."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq1_protocol import sha256
from rq2_protocol import require
from rq2_staged_mmstar_protocol import (DATASET_ID, DATASET_REVISION, PARQUET_SHA256,
    load_profile, prepare_rows, validate_inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-mmstar-dataset-cache"))
    args = parser.parse_args()
    load_profile()
    if args.output.exists():
        validate_inputs(args.output / "manifest.json")
        print("Reused verified full1500 inputs:", args.output / "manifest.json", flush=True)
        return
    if args.parquet:
        parquet = args.parquet
    else:
        from huggingface_hub import hf_hub_download
        parquet = Path(hf_hub_download(DATASET_ID, "mmstar.parquet", repo_type="dataset",
            revision=DATASET_REVISION, cache_dir=str(args.cache_dir)))
    require(sha256(parquet) == PARQUET_SHA256, "Pinned MMStar parquet hash differs")
    import pyarrow.parquet as pq
    path = prepare_rows(pq.read_table(parquet).to_pylist(), args.output)
    validate_inputs(path)
    print("Prepared verified full1500 inputs:", path, flush=True)


if __name__ == "__main__":
    main()
