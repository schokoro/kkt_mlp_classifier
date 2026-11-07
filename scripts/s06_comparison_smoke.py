"""S06 Gate 2–4: recovery-smoke сравнительного контура (10 runs, единый seed 42).

Собирает отдельную песочницу под ``outputs/`` — первый свободный путь
``outputs/S06_recovery_smoke``, ``S06_recovery_smoke_1``, ``_2``, …
(``<unique>``-суффикс; существующие данные не удаляются и не перезаписываются:
fail closed — занятый каталог повод выбрать другой путь, а не чистить его).
Реальный ``outputs/S06`` и реальный ``sealed_test``/test не используются ни
на одном шаге. Внутри песочницы прогоняется полный контур на малых
синтетических данных:

1. детерминированная микрофикстура: 10 L3 × 12 уникальных имён с явным
   стабильным class-токеном ``clsNN`` в каждом имени (различима по ``name``)
   и крошечный корпус (колонка ``name`` = те же имена + filler);
2. малый full TF-IDF (char ≤ 300 / word ≤ 100 признаков) через
   ``scripts/fit_vectorizer.py``;
3. Gate 0 в песочнице: ``scripts/freeze_split_s06.py`` (код Gate 0 не меняется);
4. Gate 1 в песочнице: ``verify-tfidf → select``: протокол обучения замораживается
   **до** любого просмотра ``comparison_val``;
5. сравнительный ``plan``: 10 конфигураций (none + 3 метода × 3 малых бюджета)
   × **один** train seed 42 — фиксируется **до** Gate 1 refit, то есть до
   первой оценки на ``comparison_val`` (Gate 1 при этом реально обучает все
   три baseline seed 42/43/44 — gate сравнительного контура валидируется по
   seed замороженного протокола, а план использует проверенное подмножество);
6. Gate 1 ``refit``: три baseline seed 42/43/44 на outer_train + majority-контроль
   на comparison_val (единственная оценка Gate 1);
7. Gate 2–3: ``scripts/run_s06_comparison.py rankings → refit → train`` —
   ранги (train-only outer_train), 9 refit-артефактов, 10 run results
   (none переиспользует checkpoint Gate 1, обучение не повторяется);
8. Gate 4: ``scripts/run_s06_report.py`` → восемь обязательных файлов отчёта;
9. проверки: full TF-IDF / ранги / refit / ровно 10 run results / отчёт,
   порядок «протокол ≤ план ≤ первый просмотр comparison_val» по mtime,
   paired Δ посчитаны, safe-resume — либо полностью проверенное
   возобновление (reuse-summary + неизменные sha256), либо явный
   fail-closed отказ без изменения любых файлов runs/ и shared/ (resume
   контуром осознанно отключён; отказ — легитимный результат, а не провал
   smoke), table-only HTML re-render
   из изолированного корня с недоступными models/runs/shared (без повторного
   обучения), повторный report в занятый каталог отказывает (нет перезаписи).

Это проверка механики контура на фикстурах, а не оценка качества отбора.
Плановые команды S06 (budgets 50k/100k/200k, seed 42/43/44, реальные
``outputs/S06``) smoke не запускает.

Запуск:

    .venv/bin/python scripts/s06_comparison_smoke.py
"""
from __future__ import annotations

import filecmp
import hashlib
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SMOKE_BASE = os.path.join(ROOT, "outputs", "S06_recovery_smoke")
TINY_N_L3 = 10
TINY_ROWS_PER_L3 = 12
SMOKE_BUDGETS = "60,120,240"
GATE1_SEEDS = "42,43,44"
PLAN_SEEDS = "42"
PLAN_SLUGS = (["none"]
              + [f"{m}_k{k}" for m in ("global", "class_aware", "gradient_x_input")
                 for k in SMOKE_BUDGETS.split(",")])
EXPECTED_RUNS = len(PLAN_SLUGS) * len(PLAN_SEEDS.split(","))  # 10
# Общий протокол обучения Gate 1 и comparison (идентичен в обоих вызовах):
# 120 строк (10×12) при batch 16 дают ~8 шагов на эпоху, 60 эпох — ~480 шагов,
# иначе (batch 256 = 1 шаг/эпоха × 8) seed 44 не успевает обучиться и его
# macro-F1 совпадает с majority — gate справедливо блокирует rankings.
# Проверка механики на синтетической фикстуре, не вывод о качестве на ODS.
TRAIN_CFG = '{"epochs": 60, "batch_size": 16, "eval_batch_size": 512, "scheduler": "none"}'
# Токены явного fail-closed отказа контура (resume отключён осознанно).
REFUSAL_TOKENS = ("resume отключён", "не перезаписываются", "уже существует")


def run(cmd: list[str], timeout: int = 1800) -> None:
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-6000:], file=sys.stderr)
        raise SystemExit(f"команда завершилась с кодом {proc.returncode}: {' '.join(cmd)}")
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if tail:
        print("  →", tail[:300])


def py(script: str, *args: str, timeout: int = 1800) -> None:
    run([PY, os.path.join("scripts", script), *args], timeout=timeout)


def last_json(stdout: str) -> dict:
    """Последний JSON-блок stdout (логи идут вперемешку с финальным JSON)."""
    lines = stdout.splitlines()
    start = max(i for i, ln in enumerate(lines) if ln.startswith("{"))
    return json.loads("\n".join(lines[start:]))


def py_json(script: str, *args: str, timeout: int = 1800) -> dict:
    print("+", PY, os.path.join("scripts", script), *args, flush=True)
    proc = subprocess.run([PY, os.path.join("scripts", script), *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-6000:], file=sys.stderr)
        raise SystemExit(f"команда завершилась с кодом {proc.returncode}: {script} {args}")
    return last_json(proc.stdout)


def make_tiny_data(smoke: str) -> None:
    """Детерминированная микрофикстура вместо сэмпла ODS: 10 L3 × 12 уникальных
    имён, класс различим по явному стабильному токену ``clsNN`` в каждом имени.
    Имена уникальны глобально, поэтому group-aware сплит не даёт конфликтов
    групп и совпадающих имён между частями (train/val/test/monitor). Корпус =
    те же имена + те же filler-строки, что и при исходном fit, — refit на нём
    воспроизводит IDF полного TF-IDF. Проверка механики, не качества."""
    data_dir = os.path.join(smoke, "data")
    os.makedirs(data_dir, exist_ok=True)
    rows = []
    for i in range(TINY_N_L3):
        tok = f"cls{i:02d}"
        for v in range(TINY_ROWS_PER_L3):
            rows.append({
                "name": f"smoke товар {tok} вариант {v:02d} позиция",
                "L1": "SMOKE_L1",
                "L2": f"SMOKE_L2_{i // 3 + 1:02d}",
                "L3": f"SMOKE_C{i:02d}",
            })
    labeled = pd.DataFrame(rows)
    labeled.to_parquet(os.path.join(data_dir, "labeled_tiny.parquet"), index=False)
    names = labeled["name"].tolist()
    filler = [f"smoke filler item {i} тест" for i in range(300)]
    corpus = pd.DataFrame({"name": names + filler})
    corpus.to_parquet(os.path.join(data_dir, "corpus_tiny.parquet"), index=False)
    print(f"tiny data: {len(labeled)} строк, {labeled['L3'].nunique()} L3; корпус {len(corpus)} строк")


def gate0_sandbox(smoke: str) -> None:
    py("freeze_split_s06.py",
       "--input", os.path.join(smoke, "data", "labeled_tiny.parquet"),
       "--min-count", "12",
       "--out-dir", os.path.join(smoke, "splits"),
       "--audit-dir", os.path.join(smoke, "audit"))


def _gate1_common(smoke: str) -> list[str]:
    return [
        "--output-root", smoke,
        "--full-tfidf", os.path.join(smoke, "full_tfidf"),
        "--corpus", os.path.join(smoke, "data", "corpus_tiny.parquet"),
        "--expect-vocab", '{"n_char":300,"n_word":100,"n_total":400}',
        "--device", "cpu",
        "--seeds", GATE1_SEEDS,
        "--reference-seed", "42",
        "--model", '{"hidden": [32], "dropout": [0.1]}',
        "--train", TRAIN_CFG,
        "--features-chunk-rows", "10000",
    ]


def _comparison_common(smoke: str) -> list[str]:
    return [
        "--output-root", smoke,
        "--full-tfidf", os.path.join(smoke, "full_tfidf"),
        "--corpus", os.path.join(smoke, "data", "corpus_tiny.parquet"),
        "--budgets", SMOKE_BUDGETS,
        "--char-share", "0.75",
        "--seeds", PLAN_SEEDS,
        "--reference-seed", "42",
        "--device", "cpu",
        "--features-chunk-rows", "10000",
        # Совпадают с вызовом Gate 1 в песочнице (проверка протокола fail closed).
        "--model", '{"hidden": [32], "dropout": [0.1]}',
        "--train", TRAIN_CFG,
    ]


def check(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"SMOKE FAIL: {message}")
    print(f"  ok: {message}")


def verify_outputs(smoke: str) -> None:
    # --- full TF-IDF (самостоятельный выход этапа fit) ---
    for name in ("features.joblib", "vectorizer_meta.json"):
        check(os.path.isfile(os.path.join(smoke, "full_tfidf", name)),
              f"full_tfidf/{name} существует")

    # --- ранги (train-only) ---
    for method in ("global", "class_aware"):
        d = os.path.join(smoke, "shared", "rankings", method)
        check(os.path.isfile(os.path.join(d, "rankings_meta.json")), f"rankings/{method} сохранены")
    check(os.path.isdir(os.path.join(smoke, "shared", "attribution")),
          "shared/attribution (G×I) существует")

    # --- refit: 9 артефактов на метод×бюджет ---
    for slug in PLAN_SLUGS:
        if slug == "none":
            continue
        d = os.path.join(smoke, "runs", slug)
        check(os.path.isdir(os.path.join(d, "tfidf_selected")), f"runs/{slug}/tfidf_selected существует")
        check(os.path.isfile(os.path.join(d, "refit_result.json")), f"runs/{slug}/refit_result.json существует")

    # --- ровно 10 run results, все success ---
    results = []
    for slug in PLAN_SLUGS:
        for seed in PLAN_SEEDS.split(","):
            path = os.path.join(smoke, "runs", slug, f"run_result_seed_{seed}.json")
            check(os.path.isfile(path), f"run result {slug} seed {seed} существует")
            with open(path, encoding="utf-8") as f:
                res = json.load(f)
            check(res.get("status") == "success", f"run {slug} seed {seed}: status=success")
            results.append(res)

    # --- отчёт: восемь обязательных файлов ---
    report = os.path.join(smoke, "report")
    required = ["report.html", "runs.csv", "per_class.parquet", "stages.csv",
                "manifest.json", "confusion_pairs.csv", "predictions_val.parquet",
                "support_bins.csv"]
    for name in required:
        check(os.path.isfile(os.path.join(report, name)), f"report/{name} существует")

    manifest = json.load(open(os.path.join(report, "manifest.json"), encoding="utf-8"))
    n_planned = manifest["statuses"]["n_planned"]
    check(n_planned == EXPECTED_RUNS,
          f"плановых runs {n_planned} (10 конфигураций × 1 seed, единый seed 42)")
    check({int(s) for s in json.load(open(os.path.join(smoke, "runs", "planned.json"),
                                          encoding="utf-8"))["config"]["seeds"]} == {42},
          "план содержит только seed 42 (Gate 1 обучал 42/43/44, план — подмножество)")
    check(manifest["statuses"]["n_success"] == n_planned, f"успешных runs {n_planned}/{n_planned}")
    check(manifest["statuses"]["n_failed"] == 0, "failed runs 0")

    runs = pd.read_csv(os.path.join(report, "runs.csv"))
    check(len(runs) == n_planned, f"runs.csv содержит {n_planned} строк")
    check(set(runs["seed"]) == {42}, "все runs с единым seed 42")
    selected = runs[runs["method"] != "none"]
    check(selected["delta_macro_f1_vs_none_same_seed"].notna().all(),
          "paired ΔF1 посчитаны для всех selected runs (строго к none seed 42)")
    check(runs[runs["method"] == "none"]["l3_macro_f1"].notna().all(),
          "метрики none (reuse Gate 1) на месте")

    preds = pd.read_parquet(os.path.join(report, "predictions_val.parquet"))
    n_val = manifest["inputs"]["splits_parts"]["comparison_val"]["n_rows"]
    check(len(preds) == n_planned * n_val,
          f"предсказаний n_planned×n_val ({len(preds)} == {n_planned * n_val})")
    check(set(preds["run_id"]) == set(runs["run_id"]), "predictions покрывают все run_id")

    conf = pd.read_csv(os.path.join(report, "confusion_pairs.csv"))
    check(not conf.empty, "confusion_pairs непусты (есть ошибки классификации на smoke)")
    check({"run_id", "true_L3", "predicted_L3", "count", "true_support",
           "error_share"}.issubset(conf.columns), "confusion_pairs: обязательные колонки")

    bins = pd.read_csv(os.path.join(report, "support_bins.csv"))
    check(not bins.empty, "support_bins непусты")
    check(set(bins["support_bin"]) <= {"2-4", "5-9", "10-19", ">=20"},
          "support_bins: только заданные диапазоны")

    per_class = pd.read_parquet(os.path.join(report, "per_class.parquet"))
    check({"run_id", "L3", "support", "precision", "recall", "f1", "delta_f1"}
          .issubset(per_class.columns), "per_class: обязательные колонки")

    stages = pd.read_csv(os.path.join(report, "stages.csv"))
    shared_mask = stages["run_id"].astype(str).str.startswith(("shared", "gate1:shared"))
    check(stages[shared_mask]["stage"].str.contains("refit").any(), "shared refit-этапы в stages.csv")
    check(stages[stages["stage"] == "attribution"].shape[0] >= 1, "атрибуция G×I в stages.csv")

    html = open(os.path.join(report, "report.html"), encoding="utf-8").read()
    for token in ("macro-F1", "paired", "sealed_test", "Формулы", "Ограничения", "support"):
        check(token in html, f"report.html содержит «{token}»")

    # --- порядок фиксации: протокол ≤ план ≤ первый просмотр comparison_val ---
    def mtime(path: str) -> float:
        return os.path.getmtime(path)

    protocol_path = os.path.join(smoke, "shared", "training_protocol.json")
    planned_path = os.path.join(smoke, "runs", "planned.json")
    gate1_val_path = os.path.join(smoke, "runs", "baseline_full", "runs.csv")
    check(mtime(protocol_path) <= mtime(planned_path),
          "протокол Gate 1 зафиксирован до сравнительного плана")
    check(mtime(planned_path) <= mtime(gate1_val_path),
          "план зафиксирован до первой оценки на comparison_val (Gate 1 refit)")


def _py_capture(script: str, *args: str, timeout: int = 1800) -> subprocess.CompletedProcess:
    """Запуск без raise: возвращает процесс целиком (код/stdout/stderr)."""
    print("+", PY, os.path.join("scripts", script), *args, flush=True)
    return subprocess.run([PY, os.path.join("scripts", script), *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def _try_last_json(stdout: str) -> dict | None:
    try:
        return last_json(stdout)
    except (ValueError, StopIteration):
        return None


def _fingerprint_tree(*roots: str) -> dict[str, str]:
    """Карта путь → sha256 содержимого для всех файлов деревьев (fail closed:
    сверяем содержимое, а не только mtime)."""
    fp: dict[str, str] = {}
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                p = os.path.join(dirpath, name)
                digest = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        digest.update(chunk)
                fp[p] = digest.hexdigest()
    return fp


def verify_resume(smoke: str) -> None:
    """Safe resume: допустимы ровно два исхода повторного прогона.

    A. Контур поддерживает проверяемое возобновление: reuse-summary с ключами
       reused/trained/n_planned, ни один run не пересчитан и ни один файл
       runs/ или shared/ не изменён (sha256 до/после).
    B. Контур осознанно отключил resume (fail closed): повторные
       train/rankings/refit завершаются явным отказом с сообщением об
       отключённом resume и **не меняют ни одного файла** выходного каталога.
       Отказ — легитимный результат, а не провал smoke: fingerprints runs/
       и shared/ до/после обязаны совпадать побайтово, после чего smoke
       продолжается с остальными проверками.
    Любое третье исходное состояние (молчаливый пересчёт, отказ с изменёнными
    файлами, успех без reuse-ключей) — SMOKE FAIL/REFUSAL.
    """
    common = _comparison_common(smoke)
    runs_dir = os.path.join(smoke, "runs")
    shared_dir = os.path.join(smoke, "shared")
    before = _fingerprint_tree(runs_dir, shared_dir)

    proc = _py_capture("run_s06_comparison.py", "train", *common, timeout=3600)
    summary = _try_last_json(proc.stdout)

    if proc.returncode == 0 and isinstance(summary, dict) and \
            {"reused", "trained", "n_planned"} <= set(summary):
        # --- Ветка A: полностью проверенное возобновление ---
        check(summary["n_done"] == summary["n_planned"] == EXPECTED_RUNS,
              f"resume train: {EXPECTED_RUNS}/{EXPECTED_RUNS} runs завершено")
        if summary["trained"]:
            raise SystemExit(f"SMOKE REFUSAL: safe-resume неполон — повторный train пересчитал "
                             f"runs {summary['trained']}: перезапись результатов недопустима.")
        check(len(summary["reused"]) == EXPECTED_RUNS,
              f"safe resume: повторный train не пересчитал ни одного run "
              f"({len(summary['reused'])} reused)")
        py_json("run_s06_comparison.py", "rankings", *common, timeout=3600)
        py_json("run_s06_comparison.py", "refit", *common, timeout=3600)
        after = _fingerprint_tree(runs_dir, shared_dir)
        check(set(after) == set(before) and all(after[p] == before[p] for p in before),
              "rankings/refit/train идемпотентны: ни один файл runs/ или shared/ не изменён")
        print("  ok: safe-resume подтверждён контуром (reuse-summary + неизменные sha256)")
        return

    # --- Ветка B: явный fail-closed отказ (resume осознанно отключён) ---
    check(proc.returncode != 0,
          "повторный train отказал (fail closed) либо вернул summary без reuse-ключей")
    text = f"{proc.stdout}\n{proc.stderr}"
    check(any(tok in text for tok in REFUSAL_TOKENS),
          f"отказ явно fail-closed (сообщение: {text.strip()[-300:]!r})")
    for stage in ("rankings", "refit"):
        p = _py_capture("run_s06_comparison.py", stage, *common, timeout=3600)
        check(p.returncode != 0, f"повторный {stage} отказал (fail closed)")
        p_text = f"{p.stdout}\n{p.stderr}"
        check(any(tok in p_text for tok in REFUSAL_TOKENS),
              f"отказ {stage} явно fail-closed (сообщение: {p_text.strip()[-300:]!r})")

    after = _fingerprint_tree(runs_dir, shared_dir)
    check(set(after) == set(before),
          f"fail-closed отказ: набор файлов runs/ и shared/ не изменился "
          f"(добавлено: {sorted(set(after) - set(before))}, удалено: {sorted(set(before) - set(after))})")
    changed = sorted(p for p in before if after.get(p) != before[p])
    check(not changed, f"fail-closed отказ: sha256 всех файлов runs/ и shared/ до/после совпадают")
    print("  ok: контур сознательно отключил resume (fail closed) — отказ явный, "
          "файлы выходного каталога неизменны; smoke продолжается")


def verify_rerender(smoke: str) -> None:
    """Table-only HTML re-render в новом каталоге с недоступными models/runs/shared.

    Изолируем только таблицы отчёта (копия внутри песочницы; оригиналы не
    удаляются и не перемещаются), у корня изоляции нет runs/ и shared/ —
    контур обязан пересобрать report.html из сохранённых таблиц без обучения,
    fit TF-IDF и inference.
    """
    isolated = os.path.join(smoke, "rerender_isolated")
    tables_src = os.path.join(smoke, "report")
    tables_dst = os.path.join(isolated, "report")
    os.makedirs(tables_dst)
    for name in ("runs.csv", "stages.csv", "manifest.json",
                 "confusion_pairs.csv", "support_bins.csv",
                 "per_class.parquet", "predictions_val.parquet"):
        shutil.copyfile(os.path.join(tables_src, name), os.path.join(tables_dst, name))
    check(not os.path.exists(os.path.join(isolated, "runs"))
          and not os.path.exists(os.path.join(isolated, "shared")),
          "изолированный корень: models/runs/shared недоступны")

    rerender_dir = os.path.join(isolated, "report_rerender")
    result = py_json("run_s06_report.py", "--output", isolated, "--report-dir", rerender_dir)
    check(result.get("mode") == "html_rebuild_from_saved_tables",
          f"re-render выполнен в режиме сохранённых таблиц (mode={result.get('mode')!r})")
    check(os.path.isfile(os.path.join(rerender_dir, "report.html")),
          "re-render: report.html создан без runs/shared")

    html = open(os.path.join(rerender_dir, "report.html"), encoding="utf-8").read()
    for token in ("macro-F1", "Ограничения"):
        check(token in html, f"re-render report.html содержит «{token}»")
    for name in ("runs.csv", "confusion_pairs.csv", "support_bins.csv", "stages.csv"):
        check(filecmp.cmp(os.path.join(tables_src, name), os.path.join(rerender_dir, name), shallow=False),
              f"re-render: {name} скопирован без изменений")
    p1 = pd.read_parquet(os.path.join(tables_src, "predictions_val.parquet"))
    p2 = pd.read_parquet(os.path.join(rerender_dir, "predictions_val.parquet"))
    check(p1.equals(p2), "re-render: predictions_val.parquet идентичен")

    # Повторный report в тот же каталог — отказ (нет перезаписи).
    proc = subprocess.run([PY, os.path.join("scripts", "run_s06_report.py"), "--output", smoke],
                          cwd=ROOT, capture_output=True, text=True)
    check(proc.returncode == 2 and "не перезаписываются" in proc.stderr,
          "повторный report в занятый каталог отказывает (без перезаписи)")


def _fresh_sandbox() -> str:
    """Первый свободный путь песочницы <base>[_<unique>]. Существующие данные
    не трогаем: fail closed — занятый каталог никогда не удаляется, выбирается
    новый уникальный путь."""
    candidates = [SMOKE_BASE, *[f"{SMOKE_BASE}_{i}" for i in range(1, 1000)]]
    for path in candidates:
        if not os.path.lexists(path):
            return path
    raise SystemExit("refusing: нет свободного пути песочницы под outputs/")


def main() -> int:
    smoke = _fresh_sandbox()
    outputs_root = os.path.abspath(os.path.join(ROOT, "outputs"))
    if (os.path.dirname(os.path.abspath(smoke)) != outputs_root
            or not os.path.basename(smoke).startswith("S06_recovery_smoke")):
        raise SystemExit("refusing: unexpected sandbox path")
    os.makedirs(smoke)
    print(f"песочница: {smoke}")

    make_tiny_data(smoke)
    py("fit_vectorizer.py",
       "--input", os.path.join(smoke, "data", "corpus_tiny.parquet"),
       "--output", os.path.join(smoke, "full_tfidf"),
       "--features-preset", "ods_light",
       "--override", '{"char": {"max_features": 300, "min_df": 1}, "word": {"max_features": 100, "min_df": 1}}',
       "--source-id", "smoke_recovery_corpus_v1")
    gate0_sandbox(smoke)

    # Gate 1: протокол замораживается до любого просмотра comparison_val.
    g1 = _gate1_common(smoke)
    py("run_s06_baseline.py", "verify-tfidf", *g1)
    py("run_s06_baseline.py", "select", *g1, timeout=3600)

    # Сравнительный план — до Gate 1 refit (первой оценки на comparison_val).
    py("run_s06_comparison.py", "plan", *_comparison_common(smoke))

    # Gate 1 refit: реально обучает все три baseline seed 42/43/44 и проходит
    # majority-контроль на comparison_val.
    py("run_s06_baseline.py", "refit", *g1, timeout=3600)

    # Gate 2–3: ранги → refit → обучение 10 runs (единый seed 42).
    cmp_common = _comparison_common(smoke)
    py("run_s06_comparison.py", "rankings", *cmp_common, timeout=3600)
    py("run_s06_comparison.py", "refit", *cmp_common, timeout=3600)
    py("run_s06_comparison.py", "train", *cmp_common, timeout=3600)

    py("run_s06_report.py", "--output", smoke)
    verify_outputs(smoke)
    verify_resume(smoke)
    verify_rerender(smoke)
    print("\nSMOKE PASS: recovery-контур S06 работоспособен на фикстурах "
          f"(10 runs, единый seed 42; песочница {smoke})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
