"""Единый эксперимент S03 и агрегированный отчёт (CLI).

Плановые прогоны: контрольный ``none`` (полный TF-IDF) и
``(global | class_aware) × бюджеты K`` на общих splits и label maps.
Selection — только по размеченному train; метрики — raw top-1 L3 без отказа
на validation; метки test не используются. Smoke- и обычный режим — один код,
разные параметры.

Примеры (полный локальный прогон на ODS — при наличии ресурсов):

    python scripts/run_s03_experiment.py run \\
        --output outputs/S03 \\
        --full-tfidf outputs/S02/tfidf_1m \\
        --corpus data/corpus.pqt \\
        --data data/ods_alpha.pqt \\
        --budgets 50000,100000,200000 --char-share 0.75 \\
        --preset ods_light --force

    python scripts/run_s03_experiment.py report --output outputs/S03

Регенерация отчёта из сохранённых результатов без повторного обучения:

    python scripts/run_s03_experiment.py report --output outputs/S03
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf import (  # noqa: E402
    DataConfig,
    ExperimentConfig,
    SelectionPlanConfig,
    build_config,
    build_report,
    planned_runs,
    run_experiment,
    setup_logging,
)


def _split(value: str) -> tuple[float, float, float]:
    parts = [float(x) for x in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("split: три доли через запятую, напр. 0.8,0.1,0.1")
    return tuple(parts)  # type: ignore[return-value]


def _csv_ints(value: str) -> list[int]:
    try:
        return [int(x) for x in value.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"список целых через запятую: {e}")


def _csv_strs(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _build_cfg(a: argparse.Namespace) -> ExperimentConfig:
    rc = build_config(a.preset, json.loads(a.override))
    return ExperimentConfig(
        output_dir=a.output,
        seed=a.seed,
        full_tfidf_path=a.full_tfidf,
        corpus_path=a.corpus,
        data=DataConfig(
            path=a.data,
            max_rows=a.data_max_rows,
            min_count=a.min_count,
            split=a.split,
            n_duplicate_examples=rc.data.n_duplicate_examples,
        ),
        features_chunk_rows=a.features_chunk_rows,
        model=rc.model,
        loss=rc.loss,
        train=rc.train,
        selection=SelectionPlanConfig(
            methods=a.methods,
            budgets=a.budgets,
            char_share=a.char_share,
            include_full=not a.no_full,
        ),
        noisy_support=a.noisy_support,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="выполнить плановые прогоны и построить отчёт")
    pr.add_argument("--output", required=True, help="корневой каталог результатов эксперимента")
    pr.add_argument("--full-tfidf", required=True, help="каталог автономного артефакта полного TF-IDF (S01/S02)")
    pr.add_argument("--corpus", required=True, help="тот же корпус, что в метаданных полного TF-IDF (для повторного fit)")
    pr.add_argument("--data", required=True, help="размеченный Parquet (name, L1, L2, L3)")
    pr.add_argument("--data-max-rows", type=int, default=None, help="ограничение размеченных строк (страт. сэмпл)")
    pr.add_argument("--min-count", type=int, default=10, help="data.min_count (по умолчанию 10)")
    pr.add_argument("--split", type=_split, default=(0.8, 0.1, 0.1), help="доли train,val,test (по умолчанию 0.8,0.1,0.1)")
    pr.add_argument("--methods", type=_csv_strs, default=["global", "class_aware"],
                    help="методы отбора через запятую (по умолчанию global,class_aware)")
    pr.add_argument("--budgets", type=_csv_ints, default=[50_000, 100_000, 200_000],
                    help="бюджеты K через запятую (по умолчанию 50000,100000,200000)")
    pr.add_argument("--char-share", type=float, default=0.75, help="квота char в бюджете (по умолчанию 0.75)")
    pr.add_argument("--no-full", action="store_true", help="не включать контрольный прогон none (полный TF-IDF)")
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--features-chunk-rows", type=int, default=50_000,
                    help="чанкование transform внутри эксперимента")
    pr.add_argument("--noisy-support", type=int, default=2,
                    help="порог support на validation для пометки неустойчивых L3 в HTML")
    pr.add_argument("--preset", default="ods_light", help="источник MLP/loss/train-конфига: default | ods_light")
    pr.add_argument("--override", default="{}",
                    help='JSON с переопределениями model/train/loss, напр. \'{"train": {"epochs": 5}}\'')
    pr.add_argument("--only", type=_csv_strs, default=None,
                    help="выполнить только эти run_id (остальные плановые — skipped)")
    pr.add_argument("--force", action="store_true", help="разрешить запись в непустой каталог --output")

    prep = sub.add_parser("report", help="перестроить отчёт из сохранённых результатов без обучения")
    prep.add_argument("--output", required=True, help="корневой каталог результатов эксперимента")

    a = p.parse_args()
    setup_logging()
    if a.cmd == "run":
        cfg = _build_cfg(a)
        plan = planned_runs(cfg)
        print("Плановые прогоны:", ", ".join(p.run_id for p in plan))
        res = run_experiment(cfg, only=a.only, force=a.force)
        paths = build_report(cfg.output_dir)
        print(json.dumps({
            "output_dir": os.path.abspath(cfg.output_dir),
            "statuses": res["statuses"],
            "report": {k: os.path.abspath(v) if k != "report_dir" else os.path.abspath(v)
                       for k, v in paths.items()},
        }, ensure_ascii=False, indent=2))
    else:
        paths = build_report(a.output)
        print(json.dumps({"report": {k: os.path.abspath(v) for k, v in paths.items()}},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
