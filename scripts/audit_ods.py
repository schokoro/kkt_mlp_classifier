"""S05, этап A: воспроизводимый аудит исходной ODS-разметки до обучения (D02).

Читает исходный ``fixtures/ods.parquet``, воспроизводит шаги подготовки
``kkt_clf.data.prepare_data`` вплоть до фильтра ``min_count`` (без сплита и
обучения) и сверяет результат с историческим ориентиром STATUS.md 2026-10-22
(18 046 строк / 737 L3 → min_count=10 → 16 635 строк / 326 L3, исключено
411 L3 / 1 411 строк). Проверяет метаданные и фактическую идентичность
сохранённого full TF-IDF S02 (``outputs/S02/tfidf_1m``) и его источника
``data/corpus.pqt``. Исследует осуществимость group-aware split с поддержкой
каждой L3 во всех частях (жадный feasibility-probe, разбиение не сохраняется
и нигде не используется).

Метки не исправляются, состав данных не меняется, split/обучение не
выполняются. Результаты пишутся в ``outputs/S05/audit/``; существующие файлы
не перезаписываются.

Пример:
    .venv/bin/python scripts/audit_ods.py \\
        --input fixtures/ods.parquet --min-count 10 \\
        --out-dir outputs/S05/audit \\
        --corpus data/corpus.pqt --s02-dir outputs/S02
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf.data import (  # noqa: E402
    REQUIRED_COLUMNS,
    file_fingerprint,
    taxonomy_violations,
)
from kkt_clf.features import preprocess, strip_whitespace  # noqa: E402
from kkt_clf.vectorizer import VectorizerArtifactError, load_vectorizer_artifact  # noqa: E402

# Исторический ориентир STATUS.md 2026-10-22 (подлежит перепроверке, не факт).
HISTORICAL_EXPECTATION = {
    "source": "artefacts/S05/value_dod.md + STATUS.md 2026-10-22 (ориентир, не измерение)",
    "rows": 18_046,
    "n_l3_raw": 737,
    "rows_after_min_count": 16_635,
    "n_l3_after_min_count": 326,
    "rare_l3_classes": 411,
    "rare_l3_rows": 1_411,
}

DEFAULT_SPLIT = (0.8, 0.1, 0.1)


def _git_info() -> dict:
    def _run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10
            ).stdout.strip() or None
        except Exception:
            return None

    return {"branch": _run("branch", "--show-current"), "head": _run("rev-parse", "HEAD")}


def _versions() -> dict:
    import pyarrow
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
        "platform": platform.platform(),
    }


def _read_meta(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def word_key(text: str) -> str:
    """Ключ word-ветки: preprocess + схлопывание пробелов (эквивалент join токенов)."""
    return " ".join(text.split())


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Три ключа идентичности текста: preprocess(lower), char-ветка, word-ветка."""
    out = df.copy()
    pre = preprocess(out["name"])
    out["_pre"] = pre
    out["_char"] = [strip_whitespace(t) for t in pre]
    out["_word"] = [word_key(t) for t in pre]
    return out


def key_stats(df: pd.DataFrame, key: str) -> dict:
    """Статистика коллизий ключа и конфликтов меток внутри коллизионных групп."""
    sizes = df.groupby(key).size()
    groups = df.groupby(key).agg(
        n_rows=(key, "size"),
        n_paths=("path", "nunique"),
        n_l3=("L3", "nunique"),
        n_l2=("L2", "nunique"),
        n_l1=("L1", "nunique"),
    )
    collided = groups[groups["n_rows"] > 1]
    conflicting = groups[groups["n_paths"] > 1]
    multi_l3 = groups[groups["n_l3"] > 1]
    return {
        "n_unique_keys": int(groups.shape[0]),
        "n_collided_groups": int(len(collided)),
        "n_rows_in_collided_groups": int(collided["n_rows"].sum()),
        "n_extra_rows_vs_unique": int(len(df) - groups.shape[0]),
        "n_groups_multi_path": int(len(conflicting)),
        "n_rows_in_multi_path_groups": int(conflicting["n_rows"].sum()),
        "n_groups_multi_l3": int(len(multi_l3)),
        "n_rows_in_multi_l3_groups": int(multi_l3["n_rows"].sum()),
        "max_rows_per_key": int(sizes.max()),
        "examples_multi_path": [
            {key: idx, "n_rows": int(r.n_rows), "paths": int(r.n_paths), "l3": int(r.n_l3)}
            for idx, r in conflicting.head(10).iterrows()
        ],
    }


def group_split_feasibility(df: pd.DataFrame, key: str, fractions: tuple[float, float, float]) -> dict:
    """Аналитика осуществимости group-aware split с поддержкой каждой L3 в каждой части.

    Выполняется только жадный feasibility-probe: разбиение не сохраняется,
    не возвращается и нигде дальше не используется. Покрытие считается по
    множеству L3, которые группа несёт (у конфликтной группы их несколько).
    """
    g = df.groupby(key).agg(n_rows=(key, "size"), l3s=("L3", lambda s: frozenset(s)))
    l3_counts = df["L3"].value_counts()
    groups_per_l3 = df.groupby("L3")[key].nunique()

    # Жадный probe: для каждой части покрыть все L3, минимизируя число групп.
    names = g.index.to_numpy()
    l3_sets = g["l3s"].tolist()
    sizes = g["n_rows"].to_numpy()
    all_l3 = set(df["L3"].unique())
    order = np.argsort(-sizes)  # крупные группы первыми
    assigned: dict[str, int] = {}
    covered = [set() for _ in fractions]
    target = [f * len(df) for f in fractions]
    part_rows = [0.0 for _ in fractions]
    iter_guard = 0
    while True:
        iter_guard += 1
        if iter_guard > len(g) + 10:
            break
        # часть с наименьшей заполненностью квоты, у которой есть непокрытые L3
        progress = False
        fill = [part_rows[p] / target[p] if target[p] else 1.0 for p in range(3)]
        parts_sorted = np.argsort(fill)
        for p in parts_sorted:
            missing = all_l3 - covered[p]
            if not missing:
                continue
            # кандидаты: нераспределённые группы, несущие хотя бы один missing L3
            best, best_gain = None, 0
            for i in order:
                k = names[i]
                if k in assigned:
                    continue
                gain = len(l3_sets[i] & missing)
                if gain > best_gain or (gain == best_gain and best_gain > 0 and sizes[i] > sizes[best]):
                    best, best_gain = i, gain
            if best is None or best_gain == 0:
                continue
            k = names[best]
            assigned[k] = p
            covered[p] |= l3_sets[best]
            part_rows[p] += sizes[best]
            progress = True
            break
        if not progress:
            break

    # Доброс оставшихся групп в наименее заполненные относительно квоты части:
    # оценка, насколько близко к 80/10/10 можно приблизить group split.
    for i in order:
        k = names[i]
        if k in assigned:
            continue
        fill = [part_rows[p] / target[p] if target[p] else float("inf") for p in range(3)]
        p = int(np.argmin(fill))
        assigned[k] = p
        covered[p] |= l3_sets[i]
        part_rows[p] += sizes[i]

    n_l3 = len(all_l3)
    coverage = {p: len(covered[p]) for p in range(3)}
    feasible = all(c == n_l3 for c in coverage.values())
    unassigned_groups = int(len(g) - len(assigned))
    return {
        "n_groups": int(len(g)),
        "n_conflict_groups_multi_l3": int((g["l3s"].map(len) > 1).sum()),
        "rows_in_conflict_groups": int(g.loc[g["l3s"].map(len) > 1, "n_rows"].sum()),
        "l3_with_lt_3_groups": int((groups_per_l3 < 3).sum()),
        "l3_with_1_group": int((groups_per_l3 == 1).sum()),
        "min_groups_per_l3": int(groups_per_l3.min()),
        "median_groups_per_l3": float(groups_per_l3.median()),
        "top_group_share_max": float(
            df.groupby(["L3", key]).size().groupby("L3").max().div(l3_counts).max()
        ),
        "greedy_probe": {
            "method": "жадное покрытие всех L3 в каждой части крупнейшими нераспределёнными группами, "
            "затем добор оставшихся групп в наименее заполненные относительно квоты части; "
            "разбиение не сохраняется (исследование осуществимости, не сплит)",
            "all_l3_covered_in_all_parts": feasible,
            "coverage_train_val_test": [coverage[0], coverage[1], coverage[2]],
            "n_l3_total": n_l3,
            "probe_part_rows": [int(r) for r in part_rows],
            "probe_part_share": [round(r / len(df), 4) for r in part_rows],
            "target_share": list(fractions),
            "n_unassigned_groups": unassigned_groups,
            "blocking_l3_examples": sorted(all_l3 - covered[min(range(3), key=lambda p: coverage[p])])[:10]
            if not feasible
            else [],
        },
    }


def audit_s02_tfidf(s02_dir: str, corpus_path: str) -> dict:
    """Сверка метаданных S02 full TF-IDF с фактическими файлами (read-only)."""
    out: dict = {}
    for name in ("tfidf_1m", "tfidf_100k"):
        art_dir = os.path.join(s02_dir, name)
        meta_path = os.path.join(art_dir, "vectorizer_meta.json")
        entry: dict = {"artifact_dir": os.path.abspath(art_dir)}
        if not os.path.exists(meta_path):
            entry["status"] = "missing"
            out[name] = entry
            continue
        meta = _read_meta(meta_path)
        entry["meta_created_at"] = meta.get("created_at")
        entry["meta_vocab"] = meta.get("vocab")
        entry["meta_source_id"] = meta.get("source", {}).get("source_id")
        src = meta.get("source", {})
        entry["source_expected"] = {"sha256": src.get("sha256"), "bytes": src.get("bytes"),
                                    "rows_in_file": src.get("rows_in_file"),
                                    "rows_used": src.get("rows_used"),
                                    "max_rows": src.get("max_rows"),
                                    "truncated": src.get("truncated")}
        fp = file_fingerprint(corpus_path)
        entry["corpus_actual"] = {"sha256": fp["sha256"], "bytes": fp["bytes"]}
        entry["corpus_sha256_match"] = fp["sha256"] == src.get("sha256")
        entry["corpus_bytes_match"] = fp["bytes"] == src.get("bytes")
        import pyarrow.parquet as pq

        entry["corpus_rows_in_file_actual"] = int(pq.ParquetFile(corpus_path).metadata.num_rows)
        entry["corpus_rows_in_file_match"] = (
            entry["corpus_rows_in_file_actual"] == src.get("rows_in_file")
        )
        fjob = os.path.join(art_dir, "features.joblib")
        finfo = (meta.get("files") or {}).get("features.joblib", {})
        entry["features_joblib_expected"] = finfo
        entry["features_joblib_actual"] = {
            "sha256": file_fingerprint(fjob)["sha256"],
            "bytes": os.path.getsize(fjob),
        }
        entry["features_joblib_sha256_match"] = (
            entry["features_joblib_actual"]["sha256"] == finfo.get("sha256")
        )
        entry["features_joblib_bytes_match"] = (
            entry["features_joblib_actual"]["bytes"] == finfo.get("bytes")
        )
        try:
            loaded = load_vectorizer_artifact(art_dir)
            entry["load_check"] = "ok"
            entry["loaded_vocab"] = {
                "n_char": loaded.builder.n_char_features,
                "n_word": loaded.builder.n_word_features,
                "n_total": loaded.builder.n_features,
            }
            entry["loaded_vocab_match_meta"] = entry["loaded_vocab"] == meta.get("vocab")
            entry["loaded_fit_report"] = dict(loaded.builder.fit_report_)
        except VectorizerArtifactError as e:
            entry["load_check"] = f"failed: {e}"
        out[name] = entry
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="fixtures/ods.parquet")
    p.add_argument("--min-count", type=int, default=10)
    p.add_argument("--out-dir", default="outputs/S05/audit")
    p.add_argument("--corpus", default="data/corpus.pqt")
    p.add_argument("--s02-dir", default="outputs/S02")
    p.add_argument("--legacy-csv", default="data/ods_labels.csv")
    p.add_argument("--split", type=float, nargs=3, default=list(DEFAULT_SPLIT))
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    audit_path = os.path.join(a.out_dir, "data_audit.json")
    support_path = os.path.join(a.out_dir, "class_support.csv")
    for path in (audit_path, support_path):
        if os.path.exists(path):
            raise SystemExit(f"Отказ: файл уже существует и не будет перезаписан: {path}")

    started = dt.datetime.now(dt.timezone.utc)

    # --- Исходный файл и пустоты (шаги 1–3 prepare_data, без изменения данных).
    fp_in = file_fingerprint(a.input)
    df_raw = pd.read_parquet(a.input, engine="pyarrow")
    df = df_raw[list(REQUIRED_COLUMNS)].copy()
    rows_read = len(df)

    bad = np.zeros(len(df), dtype=bool)
    empty_counts: dict[str, int] = {}
    for c in REQUIRED_COLUMNS:
        col = df[c]
        mask = col.isna().to_numpy()
        notna = ~mask
        if notna.any():
            mask[notna] = col[notna].astype(str).str.strip().eq("").to_numpy()
        empty_counts[c] = int(mask.sum())
        bad |= mask
    empty_counts["total"] = int(bad.sum())
    df = df.loc[~bad].reset_index(drop=True)
    rows_after_drop = len(df)

    # --- Пути меток и конфликты.
    df["path"] = df["L1"] + " / " + df["L2"] + " / " + df["L3"]
    violations = taxonomy_violations(df)
    name_paths = df.groupby("name")["path"].nunique()
    name_l3 = df.groupby("name")["L3"].nunique()
    label_conflicts = {
        "names_with_multi_path": int((name_paths > 1).sum()),
        "names_with_multi_l3": int((name_l3 > 1).sum()),
        "rows_in_multi_path_names": int(df.loc[df["name"].map(name_paths) > 1].shape[0]),
        "examples": [
            {"name": n, "n_paths": int(name_paths[n])}
            for n in name_paths[name_paths > 1].head(10).index
        ],
    }

    # --- Дубли: исходные и нормализованные ключи (тот же preprocessing, что TF-IDF).
    dup_mask = df["name"].duplicated(keep=False)
    dup_names = df.loc[dup_mask, "name"].nunique()
    dups_raw = {
        "n_duplicate_extra_rows": int(df["name"].duplicated(keep="first").sum()),
        "n_names_appearing_more_than_once": int(dup_names),
        "n_names_with_multi_path": int((name_paths[df.loc[dup_mask, "name"].unique()] > 1).sum())
        if dup_names
        else 0,
    }

    dfk = add_keys(df)
    keys_report = {
        "raw_name": {
            "n_unique": int(df["name"].nunique()),
            "n_collided_groups": int((df.groupby("name").size() > 1).sum()),
            "n_extra_rows_vs_unique": int(len(df) - df["name"].nunique()),
        },
        "preprocessed_lower": key_stats(dfk, "_pre"),
        "char_branch_strip_whitespace": key_stats(dfk, "_char"),
        "word_branch_collapse_ws": key_stats(dfk, "_word"),
    }

    # --- Классы и частоты до фильтра.
    l3_counts = df["L3"].value_counts()
    l3_stats = {
        "n_l1": int(df["L1"].nunique()),
        "n_l2": int(df["L2"].nunique()),
        "n_l3": int(df["L3"].nunique()),
        "l3_freq": {
            "min": int(l3_counts.min()),
            "p25": float(l3_counts.quantile(0.25)),
            "median": float(l3_counts.median()),
            "p75": float(l3_counts.quantile(0.75)),
            "p90": float(l3_counts.quantile(0.90)),
            "p99": float(l3_counts.quantile(0.99)),
            "max": int(l3_counts.max()),
        },
        "top10_l3": [
            {"L3": k, "count": int(v)} for k, v in l3_counts.head(10).items()
        ],
        "n_l3_below_min_count": int((l3_counts < a.min_count).sum()),
        "n_l3_equal_1": int((l3_counts == 1).sum()),
    }

    # --- Фильтр min_count (без изменения исходных данных).
    counts = df["L3"].value_counts()
    rare = counts[counts < a.min_count]
    kept_mask = ~df["L3"].isin(rare.index)
    rows_after = int(kept_mask.sum())
    n_l3_after = int(df.loc[kept_mask, "L3"].nunique())
    rare_pairs = (
        df.loc[~kept_mask, ["L1", "L2", "L3"]]
        .drop_duplicates("L3")
        .assign(count=lambda x: x["L3"].map(rare).astype(int))
        .sort_values(["count", "L3"], ascending=[False, True])
    )
    min_count_report = {
        "min_count": a.min_count,
        "n_excluded_l3": int(len(rare)),
        "n_excluded_rows": int(rare.sum()),
        "share_rows_excluded": round(float(rare.sum()) / len(df), 6),
        "rows_after": rows_after,
        "n_l3_after": n_l3_after,
        "excluded_examples_top": rare_pairs.head(20).to_dict("records"),
        "excluded_l3_full_list": rare_pairs[["L1", "L2", "L3", "count"]].to_dict("records"),
    }

    # --- Таблица поддержки всех классов (до и после фильтра).
    support = (
        df.groupby(["L1", "L2", "L3"]).size().reset_index(name="count_raw")
        .sort_values("count_raw", ascending=False, ignore_index=True)
    )
    support["share_raw"] = (support["count_raw"] / len(df)).round(6)
    support["kept_after_min_count"] = support["L3"].isin(set(counts[counts >= a.min_count].index))
    support["count_after_min_count"] = np.where(
        support["kept_after_min_count"], support["count_raw"], 0
    )
    support.to_csv(support_path, index=False)

    # --- Осуществимость group-aware split (после min_count).
    df_kept = df.loc[kept_mask].reset_index(drop=True)
    dfk_kept = dfk.loc[kept_mask].reset_index(drop=True)
    dfk_kept["path"] = df.loc[kept_mask, "path"].to_numpy()
    group_split = {
        "fractions_target": list(a.split),
        "char_branch_key": group_split_feasibility(dfk_kept, "_char", tuple(a.split)),
        "word_branch_key": group_split_feasibility(dfk_kept, "_word", tuple(a.split)),
        "note": "probe не является сплитом; разбиения не сохранялись и нигде не используются",
    }

    # --- Пересечение текстов ODS с неразмеченным корпусом (трансдуктивность).
    corpus_fp = file_fingerprint(a.corpus)
    corpus_names = pd.read_parquet(a.corpus, columns=["name"], engine="pyarrow")["name"]
    corpus_keys = set(strip_whitespace(t) for t in preprocess(corpus_names.tolist()))
    ods_keys = set(dfk_kept["_char"])
    overlap = ods_keys & corpus_keys
    corpus_overlap = {
        "corpus_path": corpus_fp["path"],
        "corpus_sha256": corpus_fp["sha256"],
        "corpus_unique_nonempty_names_approx": int(len(corpus_keys)),
        "ods_kept_unique_char_keys": int(len(ods_keys)),
        "ods_kept_char_keys_found_in_corpus": int(len(overlap)),
        "share_ods_keys_in_corpus": round(len(overlap) / len(ods_keys), 6) if ods_keys else 0.0,
        "interpretation": "совпадение текстов с неразмеченным корпусом не передаёт метки, "
        "но делает оценку частично трансдуктивной (см. S02 notes)",
    }

    # --- S02 full TF-IDF: метаданные против фактических файлов.
    s02_check = audit_s02_tfidf(a.s02_dir, a.corpus)

    # --- Сверка с legacy CSV (provenance перекодировки).
    legacy = {"path": os.path.abspath(a.legacy_csv)}
    if os.path.exists(a.legacy_csv):
        df_csv = pd.read_csv(a.legacy_csv, dtype=str, keep_default_na=False)
        legacy["rows"] = int(len(df_csv))
        legacy["columns"] = list(df_csv.columns)
        key_cols = list(REQUIRED_COLUMNS)
        a_set = set(map(tuple, df[key_cols].astype(str).to_numpy()))
        b_set = set(map(tuple, df_csv[key_cols].astype(str).to_numpy()))
        legacy["set_equal_to_parquet_after_drop_empty"] = a_set == b_set
        legacy["only_in_parquet"] = int(len(a_set - b_set))
        legacy["only_in_csv"] = int(len(b_set - a_set))

    # --- Историческое сопоставление.
    reconciliation = {
        "historical": HISTORICAL_EXPECTATION,
        "actual": {
            "rows": rows_read,
            "n_l3_raw": l3_stats["n_l3"],
            "rows_after_min_count": rows_after,
            "n_l3_after_min_count": n_l3_after,
            "rare_l3_classes": int(len(rare)),
            "rare_l3_rows": int(rare.sum()),
        },
        "match": {
            k: HISTORICAL_EXPECTATION[k] == (
                rows_read if k == "rows"
                else l3_stats["n_l3"] if k == "n_l3_raw"
                else rows_after if k == "rows_after_min_count"
                else n_l3_after if k == "n_l3_after_min_count"
                else int(len(rare)) if k == "rare_l3_classes"
                else int(rare.sum())
            )
            for k in ("rows", "n_l3_raw", "rows_after_min_count", "n_l3_after_min_count",
                      "rare_l3_classes", "rare_l3_rows")
        },
    }

    audit = {
        "generated_at": started.isoformat(timespec="seconds"),
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "generator": {
            "script": os.path.abspath(__file__),
            "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]),
            "cwd": os.getcwd(),
            "git": _git_info(),
            "versions": _versions(),
        },
        "inputs": {
            "ods_parquet": fp_in,
            "legacy_csv": legacy,
            "corpus_pqt": corpus_fp,
        },
        "audit": {
            "rows_read": rows_read,
            "columns": list(df_raw.columns),
            "empty_values": empty_counts,
            "rows_after_drop_empty": rows_after_drop,
            "taxonomy_violations": {
                "n": int(len(violations)),
                "examples": violations.head(20).to_dict("records") if len(violations) else [],
            },
            "label_conflicts_raw_name": label_conflicts,
            "duplicates": {"raw_name": dups_raw, "normalized_keys": keys_report},
            "classes": l3_stats,
            "min_count_filter": min_count_report,
            "group_split_feasibility": group_split,
            "corpus_text_overlap": corpus_overlap,
            "s02_tfidf_check": s02_check,
            "historical_reconciliation": reconciliation,
        },
        "limitations": [
            "Аудит только читает данные; метки не исправлялись, сплит и обучение не выполнялись.",
            "group_split_feasibility — жадный probe покрытия, не сплит; разбиение не сохранялось.",
            "Пересечение с corpus.pqt — наблюдение трансдуктивности, не утечка меток.",
            "Исторические числа сверены как ориентир; расхождения см. в historical_reconciliation.",
        ],
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps({
        "audit_json": os.path.abspath(audit_path),
        "class_support_csv": os.path.abspath(support_path),
        "rows_read": rows_read,
        "rows_after_min_count": rows_after,
        "n_l3_raw": l3_stats["n_l3"],
        "n_l3_after": n_l3_after,
        "historical_match": reconciliation["match"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
