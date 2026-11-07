"""S06, этап A3 (D03): заморозка воспроизводимого group-aware сплита ODS с
жёсткими per-class минимумами.

Читает размеченный parquet, применяет политику S06 ``support >= 12`` без
молчаливого удаления (исключённые row ids фиксируются в manifest) и строит
детерминированный стратифицированный group-aware сплит (~80/10/10 →
outer_train / comparison_val / sealed_test плюс train_fit / train_monitor
внутри outer_train) с минимумами: каждая сохранённая L3 имеет >= min-per-class
строк в comparison_val, sealed_test и train_monitor и >= min-train-fit в
train_fit. Группы — по реальному preprocessing обеих TF-IDF-веток
(kkt_clf.features): пересечений нормализованных текстов между частями нет.

При несовместимости минимумов с неделимыми группами — остановка без записи
сплита; доказательная диагностика сохраняется в ``<audit-dir>/split_blocked.json``.
Метрики качества не вычисляются; sealed test используется только для
count/hash/support-проверок целостности. Существующие outputs не перезаписываются.

Пример:
    .venv/bin/python scripts/freeze_split_s06.py \\
        --input fixtures/ods.parquet --min-count 12 \\
        --out-dir outputs/S06/splits --audit-dir outputs/S06/audit
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf.data import REQUIRED_COLUMNS, file_fingerprint, read_parquet  # noqa: E402
from kkt_clf.freeze_split import FreezeSplitError  # noqa: E402
from kkt_clf.s06_split import S06SplitError, s06_split, save_s06_split  # noqa: E402

DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)


def _write_blocked_diagnostics(audit_dir: str, args: argparse.Namespace, err: Exception) -> str:
    """Сохраняет доказательную диагностику блокировки; существующий файл не трогает."""
    os.makedirs(audit_dir, exist_ok=True)
    path = os.path.join(audit_dir, "split_blocked.json")
    if os.path.exists(path):
        raise SystemExit(f"Отказ: файл диагностики уже существует и не будет перезаписан: {path}")
    payload = {
        "status": "blocked",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "error_type": type(err).__name__,
        "error": str(err),
        "infeasible_classes": getattr(err, "infeasible", None),
        "min_violations": getattr(err, "min_violations", None),
        "policy": {
            "min_count": args.min_count,
            "fractions": list(args.fractions),
            "monitor_fraction": args.monitor_fraction,
            "split_seed": args.seed,
            "min_rows_per_class": args.min_per_class,
            "min_train_fit": args.min_train_fit,
        },
        "input": file_fingerprint(args.input),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="fixtures/ods.parquet", help="размеченный parquet (name, L1, L2, L3)")
    p.add_argument("--min-count", type=int, default=12)
    p.add_argument("--fractions", type=float, nargs=3, default=list(DEFAULT_FRACTIONS),
                   help="целевые доли outer train/val/test (минимумы имеют приоритет)")
    p.add_argument("--monitor-fraction", type=float, default=0.1,
                   help="целевая доля train_monitor внутри outer train")
    p.add_argument("--seed", type=int, default=42, help="seed разбиения (train seed — отдельно)")
    p.add_argument("--min-per-class", type=int, default=2,
                   help="минимум строк каждой сохранённой L3 в comparison_val/sealed_test/train_monitor")
    p.add_argument("--min-train-fit", type=int, default=2,
                   help="минимум строк каждой сохранённой L3 в train_fit")
    p.add_argument("--out-dir", default="outputs/S06/splits")
    p.add_argument("--audit-dir", default="outputs/S06/audit")
    a = p.parse_args()

    try:
        df = read_parquet(a.input)
        df = df[list(REQUIRED_COLUMNS)]
        frozen = s06_split(
            df,
            min_count=a.min_count,
            fractions=tuple(a.fractions),
            monitor_fraction=a.monitor_fraction,
            seed=a.seed,
            min_rows_per_class=a.min_per_class,
            min_train_fit=a.min_train_fit,
        )
        summary = save_s06_split(
            frozen,
            input_path=a.input,
            out_dir=a.out_dir,
            audit_dir=a.audit_dir,
            min_count=a.min_count,
            fractions=tuple(a.fractions),
            monitor_fraction=a.monitor_fraction,
            seed=a.seed,
            min_rows_per_class=a.min_per_class,
            min_train_fit=a.min_train_fit,
        )
    except S06SplitError as e:
        diag = _write_blocked_diagnostics(a.audit_dir, a, e)
        print(f"blocked: {e}", file=sys.stderr)
        print(f"диагностика сохранена: {diag}", file=sys.stderr)
        return 2
    except FreezeSplitError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
