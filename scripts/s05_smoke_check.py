"""Синтетическая проверка контура S05 (D04–D08) на малой выборке.

Собирает во временном каталоге: маленький размеченный parquet, корпус имён,
автономный full-TFIDF и замороженный сплит, затем прогоняет CLI
``scripts/run_s05_baseline.py all`` (verify → select → refit 42/43/44 →
оценка на comparison_val → majority → checkpoint seed 42 с round-trip).

Проверяет: файлы созданы, runs.csv/per_class/history/stages/manifest согласованы,
refit действительно идёт ровно fixed_epochs эпох, повторный запуск отказывается
(не перезаписывает), comparison_val не оценивается до завершения refit
(порядок стадий в stages.csv).
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf.config import build_feature_config
from kkt_clf.freeze_split import freeze_split, save_frozen_split
from kkt_clf.vectorizer import fit_and_save_vectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_labeled(n_classes=12, per_class=20, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_classes):
        l3 = f"Класс {c}"
        l2 = f"Группа {c // 3}"
        l1 = "Ветка A" if c < 6 else "Ветка B"
        for i in range(per_class):
            name = f"товар {c} вариант {i} " + (" ".join(rng.choice(["мл", "грамм", "литр"], size=2)))
            rows.append({"name": name, "L1": l1, "L2": l2, "L3": l3})
    return pd.DataFrame(rows)


def make_corpus(n=4000, seed=11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    words = ["молоко", "хлеб", "сахар", "кофе", "чай", "соль", "вода", "сок", "кефир", "рис"]
    names = [" ".join(rng.choice(words, size=3)) for _ in range(n)]
    return pd.DataFrame({"name": names})


def run_cli(args, expect_code=0):
    cmd = [os.path.join(ROOT, ".venv", "bin", "python"),
           os.path.join(ROOT, "scripts", "run_s05_baseline.py"), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != expect_code:
        print(proc.stdout)
        print(proc.stderr)
        raise AssertionError(f"код {proc.returncode} != {expect_code}: {' '.join(cmd)}")
    return proc


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="s05_smoke_")
    labeled_path = os.path.join(tmp, "labeled.parquet")
    corpus_path = os.path.join(tmp, "corpus.parquet")
    tfidf_dir = os.path.join(tmp, "tfidf_full")
    splits_dir = os.path.join(tmp, "splits")
    audit_dir = os.path.join(tmp, "audit")
    out_dir = os.path.join(tmp, "s05_out")

    labeled = make_labeled()
    labeled.to_parquet(labeled_path, index=False)
    make_corpus().to_parquet(corpus_path, index=False)

    feat_cfg = build_feature_config("ods_light", {
        "char": {"max_features": 2000, "min_df": 1},
        "word": {"max_features": 1000, "min_df": 1},
        "vectorizer_fit_rows": None,
    })
    fit_and_save_vectorizer(tfidf_dir, feat_cfg, corpus_path, max_rows=4000, seed=42)

    frozen = freeze_split(labeled, min_count=10, seed=42)
    summary = save_frozen_split(frozen, labeled_path, splits_dir, audit_dir,
                                min_count=10, fractions=(0.8, 0.1, 0.1),
                                monitor_fraction=0.1, seed=42, share_tolerance=0.1)
    assert summary["checks_ok"], summary
    manifest_path = os.path.join(splits_dir, "manifest.json")

    common = [
        "--splits-manifest", manifest_path,
        "--full-tfidf", tfidf_dir,
        "--corpus", corpus_path,
        "--output", out_dir,
        "--seeds", "42,43,44",
        "--reference-seed", "42",
        "--model", json.dumps({"hidden": [32], "dropout": [0.2]}),
        "--train", json.dumps({"epochs": 4, "patience": 3, "batch_size": 64,
                               "scheduler": "cosine", "scheduler_eval_every_steps": 5}),
        "--features-chunk-rows", "1000",
    ]
    run_cli(["all", *common])
    print("== all: ok ==")

    # --- проверка результатов
    for rel in ["tfidf_verification.json", "selection/selection_report.json",
                "selection/curves_train_monitor.csv", "runs.csv", "per_class.parquet",
                "history.csv", "stages.csv", "majority_baseline.json",
                "selected_checkpoint.json", "manifest.json",
                "checkpoints/seed_42/model.pt", "checkpoints/seed_43/model.pt",
                "checkpoints/seed_44/model.pt",
                "refit/history_seed_42.csv", "refit/run_result_seed_42.json"]:
        p = os.path.join(out_dir, rel)
        assert os.path.exists(p), f"нет файла {p}"

    runs = pd.read_csv(os.path.join(out_dir, "runs.csv"))
    assert len(runs) == 3 and (runs["status"] == "success").all(), runs
    assert runs["metric_evaluated_on"].eq("comparison_val").all()
    assert runs["l3_macro_f1"].notna().all() and runs["epochs_completed"].eq(
        json.load(open(os.path.join(out_dir, "manifest.json")))["fixed_epochs"]).all()

    sel = json.load(open(os.path.join(out_dir, "selection", "selection_report.json")))
    fixed_epochs = sel["chosen"]["fixed_epochs"]
    assert fixed_epochs >= 1
    hist = pd.read_csv(os.path.join(out_dir, "history.csv"))
    for seed in (42, 43, 44):
        h = hist[hist.run_id == f"seed_{seed}"]
        assert h["epoch"].max() == fixed_epochs, (seed, h["epoch"].max(), fixed_epochs)
        assert h[h.split == "train"]["loss"].notna().all()
        assert h[h.split == "monitor"]["l3_macro_f1"].notna().all()

    # порядок стадий: все mlp_train идут до vectorize_comparison_val
    stages = pd.read_csv(os.path.join(out_dir, "stages.csv"))
    idx_train = stages[stages.stage == "mlp_train"].index.max()
    idx_val = stages[stages.stage == "vectorize_comparison_val"].index.min()
    assert idx_train < idx_val, "comparison_val трансформировался до завершения refit"

    per_class = pd.read_parquet(os.path.join(out_dir, "per_class.parquet"))
    assert set(per_class.run_id) == {"seed_42", "seed_43", "seed_44"}
    assert len(per_class) == 3 * len(json.load(open(manifest_path))["label_maps"]["L3"])

    maj = json.load(open(os.path.join(out_dir, "majority_baseline.json")))
    assert maj["evaluated_on"] == "comparison_val" and 0.0 <= maj["l3_accuracy"] <= 1.0

    sel_ckpt = json.load(open(os.path.join(out_dir, "selected_checkpoint.json")))
    assert sel_ckpt["run_id"] == "seed_42" and sel_ckpt["selection_rule"]["kind"] == "reference_seed"
    assert sel_ckpt["roundtrip"]["status"] == "success" and sel_ckpt["roundtrip"]["metrics_match"]
    assert sel_ckpt["provenance_checks"]["restored_best"] is False

    manifest = json.load(open(os.path.join(out_dir, "manifest.json")))
    assert manifest["seeds"]["reference_seed"] == 42
    assert manifest["statuses"]["selected_checkpoint"] == "success"
    assert manifest["seed_stats_comparison_val"]["n_seed_success"] == 3

    # majority macro-F1 > 0 и меньше 1 (на синтетике сбалансировано): sanity
    assert 0.0 <= maj["l3_macro_f1"] < 1.0

    # --- повторный запуск обязан отказаться (не перезаписывает)
    proc = run_cli(["all", *common], expect_code=2)
    assert "не перезаписываются" in proc.stderr, proc.stderr
    print("== повторный запуск корректно отказался ==")

    print("SMOKE OK:", out_dir)


if __name__ == "__main__":
    main()
