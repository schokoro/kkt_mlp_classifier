"""S05, этап B (D03): заморозка воспроизводимого group-aware сплита ODS.

Читает размеченный parquet, применяет объявленную политику ``min_count=10``
без молчаливого удаления (исключённые row ids фиксируются в manifest) и
строит детерминированный стратифицированный group-aware сплит 80/10/10
(outer train / comparison validation / sealed test) плюс train-internal
monitor (train_fit / train_monitor) внутри outer train. Группировка — по
реальному preprocessing обеих TF-IDF-веток (kkt_clf.features): одинаковые
после ``strip_whitespace(preprocess(name))`` строки всегда попадают в одну
часть, поэтому пересечений нормализованных текстов между частями нет.

Метрики качества не вычисляются; sealed test используется только для
count/hash/support-проверок целостности. При конфликтных метках, пустых
значениях, нарушении таксономии, недостижимой стратификации или недопустимом
отклонении долей частей — отказ без записи файлов. Существующие outputs не
перезаписываются.

Пример:
    .venv/bin/python scripts/freeze_split.py \\
        --input fixtures/ods.parquet --min-count 10 \\
        --out-dir outputs/S05/splits --audit-dir outputs/S05/audit
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf.data import REQUIRED_COLUMNS, read_parquet  # noqa: E402
from kkt_clf.freeze_split import (  # noqa: E402
    FreezeSplitError,
    freeze_split,
    save_frozen_split,
)

DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="fixtures/ods.parquet", help="размеченный parquet (name, L1, L2, L3)")
    p.add_argument("--min-count", type=int, default=10)
    p.add_argument("--fractions", type=float, nargs=3, default=list(DEFAULT_FRACTIONS),
                   help="доли outer train/val/test")
    p.add_argument("--monitor-fraction", type=float, default=0.1,
                   help="доля train_monitor внутри outer train")
    p.add_argument("--seed", type=int, default=42, help="seed разбиения (train seed — отдельно)")
    p.add_argument("--share-tolerance", type=float, default=0.05,
                   help="допуск отклонения фактических долей частей от целевых")
    p.add_argument("--out-dir", default="outputs/S05/splits")
    p.add_argument("--audit-dir", default="outputs/S05/audit")
    a = p.parse_args()

    try:
        df = read_parquet(a.input)
        df = df[list(REQUIRED_COLUMNS)]
        frozen = freeze_split(
            df,
            min_count=a.min_count,
            fractions=tuple(a.fractions),
            monitor_fraction=a.monitor_fraction,
            seed=a.seed,
            share_tolerance=a.share_tolerance,
        )
        summary = save_frozen_split(
            frozen,
            input_path=a.input,
            out_dir=a.out_dir,
            audit_dir=a.audit_dir,
            min_count=a.min_count,
            fractions=tuple(a.fractions),
            monitor_fraction=a.monitor_fraction,
            seed=a.seed,
            share_tolerance=a.share_tolerance,
        )
    except FreezeSplitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
