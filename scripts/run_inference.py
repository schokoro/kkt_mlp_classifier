"""Инференс без ноутбука — те же вызовы, что в notebooks/inference.ipynb.

Пример:
    python scripts/run_inference.py --override '{"artifact_dir": "outputs/train_ods/artifact",
        "input_path": "fixtures/infer_unknown_labels.parquet"}'
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf import InferenceConfig, run_inference, setup_logging  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--override", default="{}", help="JSON с параметрами InferenceConfig")
    a = p.parse_args()
    setup_logging()
    res = run_inference(InferenceConfig.from_dict(json.loads(a.override)))
    print("output:", res.output_path, "rows:", len(res.predictions))
    if res.evaluation is not None:
        print(res.evaluation["summary"].to_string(index=False))
