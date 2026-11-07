"""S06 Gate 2–3 (D06–D08): сравнительный контур feature selection (CLI).

Подкоманды:

- ``plan`` — фиксация плановых 30 runs (10 конфигураций × 3 seed) до оценки:
  ``runs/planned.json`` (файл пишется один раз; повторный вызов сверяет
  идентичность плана).
- ``rankings`` — Gate 2a: рейтинги ``global``/``class_aware`` Chi² и
  ``gradient_x_input`` строго по ``outer_train`` (G×I — на опорном checkpoint
  Gate 1, reference_seed), сохранение полных рейтингов с контрольными суммами
  (``shared/rankings/``, ``shared/attribution/``) и 9 артефактов отбора
  (``shared/selection/<method>_k<K>/``). Возобновляемо: валидные рейтинги
  переиспользуются, бюджеты применяются из сохранённых рейтингов.
- ``refit`` — Gate 2b: fixed-vocabulary refit TF-IDF один раз на метод×K на
  том же снимке корпуса, что полный TF-IDF S02 (``runs/<slug>/tfidf_selected/``
  + ``refit_result.json`` с проверками и хэшами).
- ``train`` — Gate 3: финальные MLP. Конфигурация ``none`` переиспользует
  checkpoint Gate 1 (обучение не повторяется; метрики пересчитываются тем же
  протоколом и сверяются с Gate 1); selected-конфигурации обучаются с нуля на
  outer_train по замороженному протоколу (ровно fixed_epochs эпох,
  restore_best=False) для seed 42/43/44; единая оценка raw top-1 L3 на
  comparison_val с сохранением предсказаний (``runs/predictions/``) и
  per-run результатов (``runs/<slug>/run_result_seed_<s>.json``).
  Безопасное возобновление: существующие результаты не пересчитываются и не
  перезаписываются.
- ``all`` — plan (если ещё нет) → rankings → refit → train.

Предусловия (fail closed, до любого ранжирования/refit/обучения): завершённый
и проверенный Gate 0 (``outputs/S06/splits/manifest.json``, min_count=12),
проверенный full TF-IDF S02 + ``shared/full_tfidf_ref.json``, замороженный
протокол Gate 1, совместимый с фактическими входами, и baseline-прогон Gate 1
(``runs/baseline_full/``). ``sealed_test`` не используется ни для inference,
ни для метрик. Отчёт Gate 4 — отдельной командой ``scripts/run_s06_report.py``.

Примеры (плановый прогон S06, после прохождения Gate 1 и независимой проверки):

    python scripts/run_s06_comparison.py plan
    python scripts/run_s06_comparison.py rankings
    python scripts/run_s06_comparison.py refit
    python scripts/run_s06_comparison.py train
    # или одним запуском:
    python scripts/run_s06_comparison.py all
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf import setup_logging  # noqa: E402
from kkt_clf.s06_comparison import (  # noqa: E402
    DEFAULT_BUDGETS,
    DEFAULT_CHAR_SHARE,
    DEFAULT_SEEDS,
    SELECTION_METHODS,
    S06ComparisonConfig,
    S06ComparisonError,
    run_comparison_all,
    run_rankings,
    run_refit_selected,
    run_train,
    write_plan,
)


def _int_list(text: str) -> list[int]:
    try:
        values = [int(x) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"целые через запятую, напр. 50000,100000,200000: {e}") from e
    if not values:
        raise argparse.ArgumentTypeError("нужен хотя бы один элемент")
    return values


def _json_arg(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"ожидался JSON-объект: {e}") from e
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError(f"ожидался JSON-объект, получено {type(value).__name__}")
    return value


def _json_list_arg(text: str) -> list:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"ожидался JSON-список: {e}") from e
    if not isinstance(value, list):
        raise argparse.ArgumentTypeError(f"ожидался JSON-список, получено {type(value).__name__}")
    return value


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--output-root", default="outputs/S06", help="корень результатов S06")
    p.add_argument("--splits-manifest", default=None,
                   help="manifest сплита Gate 0 (default: <output-root>/splits/manifest.json)")
    p.add_argument("--full-tfidf", default="outputs/S02/tfidf_1m",
                   help="каталог сохранённого full-TFIDF S02 (pretrained)")
    p.add_argument("--corpus", default="data/corpus.pqt",
                   help="корпус full-TFIDF для проверки provenance (sha256)")
    p.add_argument("--methods", default=",".join(SELECTION_METHODS),
                   help="методы через запятую: global,class_aware,gradient_x_input")
    p.add_argument("--budgets", type=_int_list, default=list(DEFAULT_BUDGETS),
                   help="бюджеты K через запятую (default: 50000,100000,200000)")
    p.add_argument("--char-share", type=float, default=DEFAULT_CHAR_SHARE,
                   help="доля char-ветки в бюджете (default: 0.75)")
    p.add_argument("--seeds", type=_int_list, default=list(DEFAULT_SEEDS),
                   help="train seed через запятую (default: 42,43,44)")
    p.add_argument("--reference-seed", type=int, default=42,
                   help="опорный checkpoint Gate 1 для G×I (default: 42)")
    p.add_argument("--features-chunk-rows", type=int, default=50_000)
    p.add_argument("--attribution-batch-size", type=int, default=4096)
    p.add_argument("--preset", default="ods_light",
                   help="базовый пресет Gate 1 (должен совпадать с вызовом Gate 1)")
    p.add_argument("--model", type=_json_arg, default=None, help="JSON-переопределения model.* Gate 1")
    p.add_argument("--loss", type=_json_arg, default=None, help="JSON-переопределения loss.* Gate 1")
    p.add_argument("--train", type=_json_arg, default=None, help="JSON-переопределения train.* Gate 1")
    p.add_argument("--sweep", type=_json_list_arg, default=None,
                   help="JSON-список кандидатов selection Gate 1 [{name, model?, train?}]")
    p.add_argument("--device", default="auto")


def _cfg_from(a: argparse.Namespace) -> S06ComparisonConfig:
    methods = tuple(m.strip() for m in a.methods.split(",") if m.strip())
    return S06ComparisonConfig(
        output_root=a.output_root,
        splits_manifest=a.splits_manifest,
        corpus_path=a.corpus,
        full_tfidf_path=a.full_tfidf,
        methods=methods,
        budgets=tuple(a.budgets),
        char_share=a.char_share,
        seeds=tuple(a.seeds),
        reference_seed=a.reference_seed,
        device=a.device,
        features_chunk_rows=a.features_chunk_rows,
        attribution_batch_size=a.attribution_batch_size,
        preset=a.preset,
        model=a.model or {},
        loss=a.loss or {},
        train=a.train or {},
        sweep=a.sweep or [],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S06 Gate 2–3: сравнительный контур supervised feature selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "rankings", "refit", "train", "all"):
        p = sub.add_parser(name)
        _common(p)
    a = parser.parse_args(argv)
    setup_logging()
    cfg = _cfg_from(a)
    try:
        if a.cmd == "plan":
            plan = write_plan(cfg)
            print(json.dumps({"planned": plan["n_planned"], "path": os.path.abspath(cfg.planned_path)},
                             ensure_ascii=False, indent=2))
            return 0
        if a.cmd == "rankings":
            index = run_rankings(cfg, _resolve_device(cfg))
            print(json.dumps({"methods": list(index["methods"]),
                              "selections": list(index["selections"])},
                             ensure_ascii=False, indent=2))
            return 0
        if a.cmd == "refit":
            results = run_refit_selected(cfg)
            print(json.dumps({"refits": {k: v["vocab"] for k, v in results.items()}},
                             ensure_ascii=False, indent=2))
            return 0
        if a.cmd == "train":
            summary = run_train(cfg, _resolve_device(cfg))
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if a.cmd == "all":
            summary = run_comparison_all(cfg, _resolve_device(cfg))
            print(json.dumps({"train": summary["train"]}, ensure_ascii=False, indent=2))
            return 0
        raise S06ComparisonError(f"неизвестная команда {a.cmd!r}")
    except (S06ComparisonError, RuntimeError) as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2


def _resolve_device(cfg: S06ComparisonConfig):
    from kkt_clf.train import resolve_device

    return resolve_device(cfg.device)


if __name__ == "__main__":
    raise SystemExit(main())
