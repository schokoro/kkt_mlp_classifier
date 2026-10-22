"""Обучающий прогон без ноутбука — те же вызовы, что в notebooks/train.ipynb.

Пример:
    python scripts/run_train.py --preset ods_light \
        --override '{"data": {"path": "fixtures/ods.parquet"}, "output_dir": "outputs/train_ods"}'
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf import build_config, run_training, setup_logging, verify_roundtrip  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="ods_light", help="default | ods_light")
    p.add_argument("--override", default="{}", help="JSON с переопределениями конфига")
    a = p.parse_args()
    setup_logging()
    cfg = build_config(a.preset, json.loads(a.override))
    result = run_training(cfg)
    print(result.summary.to_string(index=False))
    print(f"coverage(test) = {result.evaluation['coverage']:.4f}, tau = {result.tau_selection['tau']:.2f}")
    verify_roundtrip(result)
    print("round-trip OK; artifact:", result.artifact_dir)
