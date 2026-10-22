"""Генерация всех фикстур из data/ods_labels.csv в fixtures/ (детерминированно, seed=42)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf.fixtures import DEFAULT_SRC, build_all  # noqa: E402
from kkt_clf.pipeline import setup_logging  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--out", default="fixtures")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    setup_logging()
    for name, path in build_all(a.out, a.src, a.seed).items():
        print(f"{name:24s} {path}")
