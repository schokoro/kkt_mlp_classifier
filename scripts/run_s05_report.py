"""S05 (D09): агрегированный отчёт из машинных результатов обучения (CLI).

Без обучения и без пересчёта метрик строит автономный ``report/report.html``
и обязательные ``report/runs.csv``, ``report/per_class.parquet``,
``report/stages.csv``, ``report/history.csv``, ``report/manifest.json`` из
файлов, записанных обучающим контуром (``scripts/run_s05_baseline.py``) и
аудитом в ``--output`` (по умолчанию ``outputs/S05``).

Контракт D09: planned failed/skipped видны; отсутствующие числа — ``null``,
не нули; test-метрик нет; кривые train/monitor, кривые ``selection:*``
(train-only selection), train–monitor gap (включая phase=selection), разброс
seed, majority-сравнение, per-class/support и ресурсы — из сохранённых
таблиц; максимальный train↔monitor gap по loss считается по всем сопоставленным
эпохам настоящего train-only selection (train_fit ↔ train_monitor, с указанием
run_id/эпохи), финальный refit (monitor внутри outer train) в максимум не входит
и даёт только значение последней сопоставленной эпохи; сохранённые предсказания
моделей отсутствуют — таблица L3-путаниц (confusion) не строится, обучение/
inference не запускается, ограничение указано в HTML явно; для сплит-аудита
используется ``audit/split_audit_corrected.csv``
(ошибка исходного ``split_audit.csv`` в ``n_row_ids_unique`` указана в
отчёте явно), для поддержки классов по сплиту —
``audit/class_support_split_corrected.csv`` при наличии (ошибка исходного
``class_support_split.csv`` в ``support_all`` редких классов указана явно).
Gate S06 требует фактического подтверждения файлами (provenance checkpoint,
selection, split, полные кривые train/monitor, идентификаторы TF-IDF) и
корректно трактует ``restored_best=False`` как единственно допустимое
значение для финального refit. Существующие файлы не перезаписываются — при
пересечении отказ до любых записей.

Пример:

    python scripts/run_s05_report.py --output outputs/S05
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf import setup_logging  # noqa: E402
from kkt_clf.s05_report import S05ReportError, build_s05_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", default="outputs/S05",
                    help="каталог машинных результатов обучающего контура S05")
    ap.add_argument("--report-dir", default=None,
                    help="куда писать отчёт (по умолчанию <output>/report); "
                         "существующие файлы не перезаписываются")
    args = ap.parse_args()
    setup_logging()
    try:
        result = build_s05_report(output_dir=args.output, report_dir=args.report_dir)
    except S05ReportError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
