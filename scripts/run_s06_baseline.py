"""S06 Gate 1 (D04–D06): замороженный протокол обучения и новый full baseline (CLI).

Подкоманды:

- ``verify-split`` — проверка provenance сплита Gate 0 (``outputs/S06/splits``):
  целостность частей по sha256, политика ``min_count == 12`` (сплит S05 с
  min_count=10 отклоняется), integrity checks Gate 0. Без обучения.
- ``verify-tfidf`` — проверка full-TFIDF S02 (целостность артефакта +
  совпадение корпуса по sha256) и запись ``shared/full_tfidf_ref.json``
  (при существующем — сверка совпадения, не перезапись; артефакт не копируется).
- ``estimate`` — ресурсная оценка Gate 1: проекция времени/RAM/VRAM от
  измерений S05 и размеров нового сплита, без обучения/transform/inference;
  GPU недоступен — ``null``. ``--write`` сохраняет в
  ``shared/resource_estimate_gate1.json`` (только новый файл).
- ``select`` — train-only выбор числа эпох/конфигурации на
  train_fit/train_monitor (кривые — ``runs/baseline_full/selection/``), затем
  заморозка неизменяемого ``shared/training_protocol.json``. comparison_val
  до этого не читается; повторный select при существующем протоколе — отказ.
- ``refit`` — три полных baseline seed 42/43/44 на outer_train ровно на
  замороженное число эпох (фиксированное заранее расписание LR), опорный
  checkpoint — ``--reference-seed`` (default 42, train-only правило), единственная
  оценка — raw top-1 L3 на comparison_val против train-only majority
  (``runs/baseline_full/{runs.csv,per_class.parquet,history.csv,stages.csv,
  majority_baseline.json,selected_checkpoint.json,manifest.json,checkpoints/}``).
  Перед обучением: verify split + verify TF-IDF + сверка замороженного
  протокола с фактическими входами (fail closed).
- ``all`` — verify-split → verify-tfidf → select → refit одним запуском.

Границы: ``sealed_test`` не используется ни для inference, ни для метрик
(только структурные sha256/n_rows из manifest); старые checkpoint/split S05
не используются; существующие файлы не перезаписываются; selected-сетка
Gate 2+ этим контуром не запускается.

Условие запуска обучения: Gate 0 (``outputs/S06/splits`` + независимая
проверка) завершён; команды ``select``/``refit``/``all`` до этого завершатся
отказом с диагностикой.

Пример (после независимой проверки Gate 0):

    python scripts/run_s06_baseline.py verify-split
    python scripts/run_s06_baseline.py verify-tfidf
    python scripts/run_s06_baseline.py estimate --write
    python scripts/run_s06_baseline.py select
    python scripts/run_s06_baseline.py refit
    # или одним запуском:
    python scripts/run_s06_baseline.py all
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf import setup_logging  # noqa: E402
from kkt_clf.s05_baseline import S05BaselineError  # noqa: E402
from kkt_clf.s06_baseline import (  # noqa: E402
    EXPECTED_FULL_VOCAB,
    S06Gate1Config,
    S06Gate1Error,
    estimate_gate1_resources,
    run_gate1_contour,
    run_gate1_refit,
    verify_gate0_split,
    verify_s06_tfidf,
    freeze_training_protocol,
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
    p.add_argument("--output-root", default="outputs/S06", help="корень результатов S06")
    p.add_argument("--splits-manifest", default=None,
                   help="manifest сплита Gate 0 (default: <output-root>/splits/manifest.json)")
    p.add_argument("--full-tfidf", default="outputs/S02/tfidf_1m",
                   help="каталог сохранённого full-TFIDF S02 (pretrained)")
    p.add_argument("--expect-vocab", type=_json_arg, default=None,
                   help='JSON {"n_char":…,"n_word":…,"n_total":…} — контракт полного '
                        "TF-IDF (default: 300000/100000/400000 — полный S02)")
    p.add_argument("--corpus", default="data/corpus.pqt",
                   help="корпус full-TFIDF для проверки provenance (sha256)")
    p.add_argument("--preset", default="ods_light", help="базовый пресет MLP/loss/train")
    p.add_argument("--model", type=_json_arg, default=None, help="JSON-переопределения model.*")
    p.add_argument("--loss", type=_json_arg, default=None, help="JSON-переопределения loss.*")
    p.add_argument("--train", type=_json_arg, default=None, help="JSON-переопределения train.*")
    p.add_argument("--features-chunk-rows", type=int, default=50_000)
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--seeds", type=_seeds_arg, default=[42, 43, 44],
                   help="train seed refit через запятую (default: 42,43,44)")
    p.add_argument("--reference-seed", type=int, default=42,
                   help="опорный checkpoint G×I — train-only правило (default: 42)")
    p.add_argument("--device", default="auto")


def _cfg_from(a: argparse.Namespace) -> S06Gate1Config:
    return S06Gate1Config(
        output_root=a.output_root,
        splits_manifest=a.splits_manifest or os.path.join(a.output_root, "splits", "manifest.json"),
        full_tfidf_path=a.full_tfidf,
        corpus_path=a.corpus,
        expected_vocab=a.expect_vocab if a.expect_vocab is not None else dict(EXPECTED_FULL_VOCAB),
        preset=a.preset,
        model=a.model or {},
        loss=a.loss or {},
        train=a.train or {},
        features_chunk_rows=a.features_chunk_rows,
        selection_seed=a.selection_seed,
        seeds=list(a.seeds),
        reference_seed=a.reference_seed,
        device=a.device,
    )


def _load_inputs(cfg: S06Gate1Config):
    """verify split (Gate 0) + verify TF-IDF + загрузка артефакта и устройства."""
    from kkt_clf.train import resolve_device
    from kkt_clf.vectorizer import load_vectorizer_artifact

    manifest = verify_gate0_split(cfg)
    verification = verify_s06_tfidf(cfg)
    loaded = load_vectorizer_artifact(cfg.full_tfidf_path)
    device = resolve_device(cfg.device)
    return manifest, verification, loaded, device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S06 Gate 1: замороженный протокол обучения и новый full baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_split = sub.add_parser("verify-split", help="проверка provenance сплита Gate 0")
    _common(p_split)

    p_tfidf = sub.add_parser("verify-tfidf", help="проверка full-TFIDF S02 + shared/full_tfidf_ref.json")
    _common(p_tfidf)

    p_est = sub.add_parser("estimate", help="ресурсная оценка Gate 1 (проекция от S05, без обучения)")
    _common(p_est)
    p_est.add_argument("--write", action="store_true",
                       help="сохранить в shared/resource_estimate_gate1.json (только новый файл)")

    p_select = sub.add_parser("select", help="train-only выбор эпох + заморозка протокола")
    _common(p_select)
    p_select.add_argument("--sweep", type=_json_list_arg, default=None,
                          help="JSON-список кандидатов [{name, model?, train?}] для ограниченного выбора по monitor")

    p_refit = sub.add_parser("refit", help="refit seed 42/43/44 на outer_train + единственная оценка comparison_val")
    _common(p_refit)

    p_all = sub.add_parser("all", help="verify-split → verify-tfidf → select → refit")
    _common(p_all)
    p_all.add_argument("--sweep", type=_json_list_arg, default=None,
                       help="JSON-список кандидатов selection [{name, model?, train?}]")

    a = parser.parse_args(argv)
    setup_logging()
    try:
        if a.cmd == "verify-split":
            cfg = _cfg_from(a)
            manifest = verify_gate0_split(cfg)
            print(json.dumps({
                "ok": True,
                "min_count": manifest["policy"]["min_count"],
                "split_seed": manifest["policy"]["split_seed"],
                "parts": {name: {"n_rows": p["n_rows"], "sha256": p["sha256"][:16] + "…"}
                          for name, p in manifest["parts"].items()},
                "sealed_test": "только структурные sha256/n_rows; inference/eval запрещён",
            }, ensure_ascii=False, indent=2))
            return 0
        if a.cmd == "verify-tfidf":
            cfg = _cfg_from(a)
            verification = verify_s06_tfidf(cfg)
            print(json.dumps({
                "decision": verification["decision"],
                "full_tfidf_id": verification["full_tfidf_id"],
                "vocab": verification["vocab"],
                "corpus_matches_source": verification["corpus_matches_source"],
                "ref": cfg.tfidf_ref_path,
            }, ensure_ascii=False, indent=2))
            return 0
        if a.cmd == "estimate":
            cfg = _cfg_from(a)
            manifest = verify_gate0_split(cfg)
            estimate = estimate_gate1_resources(cfg, manifest, write=a.write)
            print(json.dumps(estimate, ensure_ascii=False, indent=2))
            return 0
        if a.cmd == "select":
            cfg = _cfg_from(a)
            cfg.sweep = a.sweep or []
            cfg.validate()
            manifest, verification, loaded, device = _load_inputs(cfg)
            protocol = freeze_training_protocol(cfg, manifest, verification, loaded, device)
            print(json.dumps({
                "frozen": True,
                "fixed_epochs": protocol["fixed_epochs"],
                "chosen": protocol["chosen"],
                "protocol": cfg.protocol_path,
                "selection_curves": os.path.join(cfg.baseline_dir, "selection", "curves_train_monitor.csv"),
            }, ensure_ascii=False, indent=2))
            return 0
        if a.cmd == "refit":
            cfg = _cfg_from(a)
            cfg.validate()
            manifest, verification, loaded, device = _load_inputs(cfg)
            manifest_out = run_gate1_refit(cfg, manifest, verification, loaded, device)
            print(json.dumps({
                "statuses": manifest_out["statuses"],
                "seed_stats": manifest_out["seed_stats_comparison_val"],
                "output": os.path.abspath(cfg.baseline_dir),
            }, ensure_ascii=False, indent=2))
            return 0
        if a.cmd == "all":
            cfg = _cfg_from(a)
            cfg.sweep = a.sweep or []
            manifest_out = run_gate1_contour(cfg)
            print(json.dumps({
                "statuses": manifest_out["statuses"],
                "seed_stats": manifest_out["seed_stats_comparison_val"],
                "protocol": cfg.protocol_path,
                "output": os.path.abspath(cfg.baseline_dir),
            }, ensure_ascii=False, indent=2))
            return 0
        raise S06Gate1Error(f"неизвестная команда {a.cmd!r}")
    except S06Gate1Error as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2
    except S05BaselineError as e:
        print(f"ОШИБКА (контур S05): {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
