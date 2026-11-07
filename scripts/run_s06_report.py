"""S06 Gate 4 (D09–D11): агрегированный отчёт сравнительного эксперимента (CLI).

Без обучения, без fit TF-IDF и без повторного inference собирает восемь
обязательных файлов ``report/`` из сохранённых машинных результатов Gate 0–3
(см. ``kkt_clf.s06_report``): report.html, runs.csv, per_class.parquet,
stages.csv, manifest.json, confusion_pairs.csv, predictions_val.parquet,
support_bins.csv. Planned runs без result-файла честно получают статус
pending; failed/skipped не скрываются. Существующие файлы не перезаписываются.

Пример:

    python scripts/run_s06_report.py --output outputs/S06
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf import setup_logging  # noqa: E402
from kkt_clf.s06_report import S06ReportError, build_s06_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", default="outputs/S06",
                    help="корень результатов S06 (default: outputs/S06)")
    ap.add_argument("--report-dir", default=None,
                    help="куда писать отчёт (default: <output>/report); "
                         "существующие файлы не перезаписываются")
    args = ap.parse_args()
    setup_logging()
    try:
        result = build_s06_report(output_root=args.output, report_dir=args.report_dir)
    except S06ReportError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
