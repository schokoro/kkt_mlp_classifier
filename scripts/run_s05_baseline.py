"""S05 (D04–D08): обучающий контур baseline на полном TF-IDF (CLI).

Подкоманды:

- ``verify-tfidf`` — D04: проверка единого сохранённого full-TFIDF S02
  (целостность артефакта + совпадение корпуса с provenance), запись
  ``tfidf_verification.json`` (проверенный provenance для gate отчёта).
- ``select`` — D05: train-only выбор числа эпох/конфигурации на
  train_fit/train_monitor; ``selection/selection_report.json`` +
  ``selection/curves_train_monitor.csv``. Перед обучением выполняется
  проверка full-TFIDF (D04, fail closed) и её результат сохраняется в
  ``tfidf_verification.json`` (при уже существующем файле — сверка
  совпадения, не перезапись).
- ``refit`` — D06–D08: финальный refit на outer_train с фиксированными эпохами
  (из selection либо ``--fixed-epochs``) и фиксированным заранее расписанием
  LR (общим с selection: адаптивный ``plateau`` заменяется постоянным LR до
  обучения) для seed 42/43/44, одна оценка raw top-1 на comparison_val после
  всех refit, majority baseline, выбор опорного checkpoint по
  ``--reference-seed`` с round-trip проверкой (полный контракт признаков +
  inference через transform векторизатора checkpoint); ``runs.csv``,
  ``per_class.parquet``, ``history.csv``, ``stages.csv``,
  ``majority_baseline.json``, ``selected_checkpoint.json``,
  ``manifest.json``, ``checkpoints/seed_<s>/``, ``refit/``. При использовании
  selection весь loss выбранного кандидата воспроизводится из сохранённого
  отчёта; несовпадение с текущим ``--loss`` — отказ до обучения. Перед
  обучением выполняется проверка full-TFIDF (D04, fail closed) и её результат
  сохраняется в ``tfidf_verification.json`` (при уже существующем файле —
  сверка совпадения, не перезапись), а при использовании selection сверяется
  его provenance (sha256 сплитов и full_tfidf_id) с текущими входами.
- ``all`` — verify + select + refit одним запуском.

Существующие файлы не перезаписываются: при обнаружении целевого пути —
отказ до любых записей. Sealed test не используется; comparison_val не
используется до завершения всех refit и не используется для выбора checkpoint.

Пример (реальный прогон после ревью, только новые файлы под outputs/S05):

    python scripts/run_s05_baseline.py all \\
        --splits-manifest outputs/S05/splits/manifest.json \\
        --full-tfidf outputs/S02/tfidf_1m \\
        --corpus data/corpus.pqt \\
        --output outputs/S05 \\
        --seeds 42,43,44 --reference-seed 42
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf import setup_logging  # noqa: E402
from kkt_clf.s05_baseline import (  # noqa: E402
    S05BaselineError,
    S05Config,
    load_split_manifest,
    read_selection,
    record_tfidf_verification,
    run_refit,
    run_selection,
    run_training_contour,
    verify_full_tfidf,
)


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


def _seeds_arg(text: str) -> list[int]:
    try:
        seeds = [int(x) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"seed через запятую, напр. 42,43,44: {e}") from e
    if not seeds:
        raise argparse.ArgumentTypeError("нужен хотя бы один seed")
    return seeds


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--splits-manifest", default="outputs/S05/splits/manifest.json",
                   help="manifest замороженного сплита (freeze_split)")
    p.add_argument("--full-tfidf", default="outputs/S02/tfidf_1m",
                   help="каталог сохранённого full-TFIDF S02 (pretrained)")
    p.add_argument("--corpus", default="data/corpus.pqt",
                   help="корпус full-TFIDF для проверки provenance (sha256)")
    p.add_argument("--output", default="outputs/S05", help="каталог результатов (только новые файлы)")
    p.add_argument("--preset", default="ods_light",
                   help="базовый пресет MLP/loss/train (default: ods_light)")
    p.add_argument("--model", type=_json_arg, default=None, help="JSON-переопределения model.*")
    p.add_argument("--loss", type=_json_arg, default=None, help="JSON-переопределения loss.*")
    p.add_argument("--train", type=_json_arg, default=None, help="JSON-переопределения train.*")
    p.add_argument("--features-chunk-rows", type=int, default=50_000)
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--seeds", type=_seeds_arg, default=[42, 43, 44],
                   help="train seed refit через запятую (default: 42,43,44)")
    p.add_argument("--reference-seed", type=int, default=42,
                   help="опорный checkpoint для S06 — train-only правило (default: 42)")
    p.add_argument("--fixed-epochs", type=int, default=None,
                   help="число эпох refit; без флага — из selection_report.json")
    p.add_argument("--device", default="auto")


def _cfg_from(a: argparse.Namespace) -> S05Config:
    return S05Config(
        output_dir=a.output,
        splits_manifest=a.splits_manifest,
        full_tfidf_path=a.full_tfidf,
        corpus_path=a.corpus,
        preset=a.preset,
        model=a.model or {},
        loss=a.loss or {},
        train=a.train or {},
        features_chunk_rows=a.features_chunk_rows,
        selection_seed=a.selection_seed,
        seeds=list(a.seeds),
        reference_seed=a.reference_seed,
        fixed_epochs=a.fixed_epochs,
        device=a.device,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S05 (D04–D08): обучающий контур baseline на полном TF-IDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify-tfidf", help="D04: проверка full-TFIDF + tfidf_verification.json")
    _common(p_verify)

    p_select = sub.add_parser("select", help="D05: train-only выбор эпох/конфигурации")
    _common(p_select)
    p_select.add_argument("--sweep", type=_json_list_arg, default=None,
                          help="JSON-список кандидатов [{name, model?, train?}] для ограниченного выбора по monitor")

    p_refit = sub.add_parser("refit", help="D06–D08: refit на outer_train + оценка + checkpoint")
    _common(p_refit)

    p_all = sub.add_parser("all", help="verify + select + refit одним запуском")
    _common(p_all)
    p_all.add_argument("--sweep", type=_json_list_arg, default=None,
                       help="JSON-список кандидатов selection [{name, model?, train?}]")

    a = parser.parse_args(argv)
    setup_logging()
    try:
        if a.cmd == "verify-tfidf":
            cfg = _cfg_from(a)
            result = verify_full_tfidf(cfg.full_tfidf_path, cfg.corpus_path)
            path = record_tfidf_verification(cfg.output_dir, result)
            print(json.dumps({"decision": result["decision"],
                              "full_tfidf_id": result["full_tfidf_id"],
                              "vocab": result["vocab"], "path": path},
                             ensure_ascii=False, indent=2))
            return 0
        if a.cmd == "select":
            cfg = _cfg_from(a)
            cfg.sweep = a.sweep or []
            cfg.validate()
            from kkt_clf.train import resolve_device
            from kkt_clf.vectorizer import load_vectorizer_artifact

            # Проверка provenance full TF-IDF до любого обучения (D04, fail
            # closed): отдельная команда select не должна обучаться на
            # несовместимом артефакте/корпусе. Результат сохраняется для
            # gate отчёта (при уже существующем файле — сверка совпадения).
            verification = verify_full_tfidf(cfg.full_tfidf_path, cfg.corpus_path)
            record_tfidf_verification(cfg.output_dir, verification)
            manifest = load_split_manifest(cfg.splits_manifest)
            loaded = load_vectorizer_artifact(cfg.full_tfidf_path)
            report = run_selection(cfg, manifest, loaded, resolve_device(cfg.device))
            print(json.dumps({"chosen": report["chosen"],
                              "candidates": [c["name"] for c in report["candidates"]]},
                             ensure_ascii=False, indent=2))
            return 0
        if a.cmd == "refit":
            cfg = _cfg_from(a)
            cfg.validate()
            from kkt_clf.train import resolve_device
            from kkt_clf.vectorizer import load_vectorizer_artifact

            # Проверка provenance full TF-IDF до любого обучения (D04, fail
            # closed): refit с --fixed-epochs может идти без фазы selection,
            # поэтому проверка здесь обязательна, а не только в all. Результат
            # сохраняется для gate отчёта (при уже существующем файле — сверка
            # совпадения) и передаётся в run_refit, которая отказывает при
            # несовпадении full_tfidf_id с загруженным артефактом.
            verification = verify_full_tfidf(cfg.full_tfidf_path, cfg.corpus_path)
            record_tfidf_verification(cfg.output_dir, verification)
            manifest = load_split_manifest(cfg.splits_manifest)
            loaded = load_vectorizer_artifact(cfg.full_tfidf_path)
            selection = None if cfg.fixed_epochs is not None else read_selection(cfg.output_dir)
            manifest_out = run_refit(cfg, manifest, loaded, resolve_device(cfg.device),
                                     selection=selection, tfidf_verification=verification)
            print(json.dumps({"statuses": manifest_out["statuses"],
                              "seed_stats": manifest_out["seed_stats_comparison_val"],
                              "output": os.path.abspath(cfg.output_dir)},
                             ensure_ascii=False, indent=2))
            return 0
        if a.cmd == "all":
            cfg = _cfg_from(a)
            cfg.sweep = a.sweep or []
            manifest_out = run_training_contour(cfg)
            print(json.dumps({"statuses": manifest_out["statuses"],
                              "seed_stats": manifest_out["seed_stats_comparison_val"],
                              "output": os.path.abspath(cfg.output_dir)},
                             ensure_ascii=False, indent=2))
            return 0
        raise S05BaselineError(f"неизвестная команда {a.cmd!r}")
    except S05BaselineError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
