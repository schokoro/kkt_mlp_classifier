"""S06, этап A4 (D04): проверка и фиксация ссылки на полный TF-IDF S02.

Сверяет метаданные выбранного full-артефакта S02 (по умолчанию
``outputs/S02/tfidf_1m`` — 300 000 char + 100 000 word признаков) с
фактическими файлами и с источником fit (``data/corpus.pqt``): sha256, размер,
число строк снимка, размеры словарей, загрузка через kkt_clf.vectorizer и
контрольный transform фиксированных строк. Проверяет, что векторизатор
является именно полным (ожидаемые размеры словарей задаются явно), что fit
выполнен на полном снимке внешнего неразмеченного корпуса (metadata source:
``rows_used == rows_in_file``, ``truncated is False``, sha256 совпадает с
фактическим корпусом), а не на размеченной train или подвыборке, и что после
загрузки inference применяет готовый артефакт через transform без fit.

Утверждения, которые нельзя подтвердить metadata/аудитом, не выдаются за
факт: ``fit_on_labeled_train`` фиксируется как ``false`` только при полном
совпадении source с фактическим корпусом и (при заданном эталоне) несовпадении
с размеченным файлом, иначе — ``null`` с явной оговоркой, что не подтверждено.
Число совпадений текстов ODS с корпусом (трансдуктивность) берётся из аудита
``--audit`` (по умолчанию ``outputs/S06/audit/data_audit.json``) после сверки
sha256 корпуса в аудите с фактическим; без согласованного аудита количество
не заявляется. Эталон размеченного файла: ``--labeled`` либо fingerprint из
замороженного ``outputs/S06/splits/manifest.json``.

Результат — ``full_tfidf_ref.json`` (путь + hashes + provenance); сам full
артефакт не копируется и не изменяется. Существующие файлы не перезаписываются.

Пример:
    .venv/bin/python scripts/s06_tfidf_ref.py \\
        --s02-dir outputs/S02 --full-name tfidf_1m \\
        --corpus data/corpus.pqt --out outputs/S06/shared/full_tfidf_ref.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from kkt_clf.data import file_fingerprint  # noqa: E402
from kkt_clf.vectorizer import VectorizerArtifactError, load_vectorizer_artifact  # noqa: E402

EXPECTED_VOCAB = {"n_char": 300_000, "n_word": 100_000, "n_total": 400_000}

# Фиксированные контрольные строки для transform smoke (заморожены, не из test).
PROBE_TEXTS = [
    "молоко пастеризованное 3.2% 1л",
    "хлеб бородинский 400г",
    "unknown-probe-item-zzz-123",
]


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
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s02-dir", default="outputs/S02")
    p.add_argument("--full-name", default="tfidf_1m", help="каталог full-артефакта внутри --s02-dir")
    p.add_argument("--corpus", default="data/corpus.pqt")
    p.add_argument("--out", default="outputs/S06/shared/full_tfidf_ref.json")
    p.add_argument("--expect-vocab-total", type=int, default=EXPECTED_VOCAB["n_total"])
    p.add_argument("--audit", default=os.path.join("outputs", "S06", "audit", "data_audit.json"),
                   help="data_audit.json S06: источник подтверждённого числа совпадений текстов ODS с корпусом")
    p.add_argument("--labeled", default=None,
                   help="размеченный parquet/CSV для сверки source fit (default: fingerprint из outputs/S06/splits/manifest.json)")
    a = p.parse_args()

    if os.path.exists(a.out):
        print(f"error: файл уже существует и не будет перезаписан: {a.out}", file=sys.stderr)
        return 1

    art_dir = os.path.join(a.s02_dir, a.full_name)
    meta_path = os.path.join(art_dir, "vectorizer_meta.json")
    if not os.path.exists(meta_path):
        print(f"error: нет метаданных артефакта: {meta_path}", file=sys.stderr)
        return 1
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    failures: list[str] = []

    # --- Provenance источника fit против фактического снимка корпуса.
    # Утверждения только от проверенных данных: совпадение sha256/bytes с
    # фактическим корпусом, полный снимок (rows_used == rows_in_file,
    # truncated is False) и несовпадение с размеченным файлом (эталон).
    src = meta.get("source", {}) or {}
    corpus_fp = file_fingerprint(a.corpus)
    corpus_rows_actual = int(pq.ParquetFile(a.corpus).metadata.num_rows)
    sha_match = (
        corpus_fp["sha256"] == src.get("sha256") and corpus_fp["bytes"] == src.get("bytes")
    )
    rows_in_file_meta = src.get("rows_in_file")
    rows_used = src.get("rows_used")
    truncated = src.get("truncated")
    if rows_used is not None and rows_in_file_meta is not None:
        full_snapshot = bool(rows_used == rows_in_file_meta and truncated is False)
    else:
        full_snapshot = None  # metadata не позволяет подтвердить полноту fit

    # Эталон размеченного файла: явный --labeled либо fingerprint из
    # замороженного manifest сплита Gate 0 (без повторного хэширования).
    labeled_fp: dict | None = None
    labeled_origin: str | None = None
    if a.labeled:
        labeled_fp = file_fingerprint(a.labeled)
        labeled_origin = f"--labeled {a.labeled}"
    else:
        default_manifest = os.path.join("outputs", "S06", "splits", "manifest.json")
        if os.path.isfile(default_manifest):
            try:
                with open(default_manifest, encoding="utf-8") as mf:
                    man = json.load(mf)
                lp = ((man.get("inputs") or {}).get("labeled_parquet")) or {}
                if lp.get("sha256"):
                    labeled_fp = dict(lp)
                    labeled_origin = f"inputs.labeled_parquet из {default_manifest}"
            except (OSError, json.JSONDecodeError):
                pass

    fit_on_labeled_train: bool | None = None
    provenance_notes: list[str] = []
    src_sha = src.get("sha256")
    if labeled_fp is not None and src_sha and labeled_fp.get("sha256") == src_sha:
        fit_on_labeled_train = True
        failures.append(
            "source fit совпадает по sha256 с размеченным файлом: fit_on_labeled_train=true, "
            "контракт S02 (fit на внешнем неразмеченном корпусе) нарушен"
        )
    elif sha_match and full_snapshot is True and labeled_fp is not None:
        # Эталон заявлен и source от него отличается — несовпадение проверено.
        fit_on_labeled_train = False
        provenance_notes.append(
            "fit на размеченной train не подтверждён и опровергнут: source совпадает с "
            f"фактическим корпусом ({a.corpus}) по sha256/bytes, fit на полном снимке "
            f"(rows_used={rows_used} == rows_in_file={rows_in_file_meta}, truncated={truncated})"
            f", source отличен от размеченного файла ({labeled_origin})"
        )
    elif sha_match and full_snapshot is True:
        # Эталона нет: несовпадение с разметкой не проверялось — false не утверждается.
        provenance_notes.append(
            "fit на размеченной train не подтверждён, но и не опровергнут: source совпадает с "
            f"фактическим корпусом ({a.corpus}) по sha256/bytes, fit на полном снимке "
            f"(rows_used={rows_used} == rows_in_file={rows_in_file_meta}, truncated={truncated}); "
            f"эталон размеченного файла не заявлен ({labeled_origin or 'manifest Gate 0 не найден'}) — "
            "несовпадение с разметкой по sha256 не проверялось, fit_on_labeled_train=null"
        )
    else:
        # Неподтверждённое не выдаём за факт: значение null с явной оговоркой.
        provenance_notes.append(
            "fit на неразмеченном корпусе НЕ подтверждён полностью: "
            f"sha_match={sha_match}, full_snapshot={full_snapshot} "
            f"(rows_used={rows_used!r}, rows_in_file={rows_in_file_meta!r}, truncated={truncated!r}); "
            "fit_on_labeled_train не заявляется."
        )
        failures.append(
            "provenance полного исходного fit не подтверждён metadata артефакта: "
            "нужны sha256/bytes корпуса, rows_used == rows_in_file и truncated=false"
        )

    # --- Пересечение текстов ODS с корпусом (трансдуктивность) — только из
    # согласованного аудита; без него количество не заявляется.
    overlap_count: int | None = None
    overlap_source: str | None = None
    if a.audit and os.path.isfile(a.audit):
        try:
            with open(a.audit, encoding="utf-8") as af:
                audit = json.load(af)
        except (OSError, json.JSONDecodeError) as e:
            failures.append(f"аудит не читается: {a.audit}: {e}")
        else:
            ao = ((audit.get("audit") or {}).get("corpus_text_overlap")) or {}
            if ao.get("corpus_sha256") and ao["corpus_sha256"] != corpus_fp["sha256"]:
                failures.append(
                    f"аудит {a.audit} относится к иному корпусу (sha256 не совпадает "
                    "с фактическим): пересечение текстов из него неприменимо"
                )
            elif "ods_kept_char_keys_found_in_corpus" in ao:
                overlap_count = ao["ods_kept_char_keys_found_in_corpus"]
                overlap_source = os.path.abspath(a.audit)
            else:
                failures.append(f"аудит {a.audit} не содержит corpus_text_overlap")
    elif a.audit:
        provenance_notes.append(
            f"аудит {a.audit} не найден: число совпадений текстов ODS с корпусом "
            "не подтверждено и не заявляется."
        )

    if overlap_count is not None:
        transductive_note = (
            f"трансдуктивность: {overlap_count} уникальных char-ключов ODS (после фильтра >=12) "
            f"найдено в корпусе fit — подтверждено аудитом {overlap_source}; совпадение текстов "
            "при unsupervised fit метки не переносит, но оценку делает частично трансдуктивной."
        )
    else:
        transductive_note = (
            "трансдуктивность: пересечение текстов ODS с корпусом fit не подтверждено "
            "согласованным аудитом; количество совпадений не заявляется."
        )
    provenance_notes.append(transductive_note)

    provenance = {
        "meta_source": {
            "source_id": src.get("source_id"),
            "path": src.get("path"),
            "sha256": src.get("sha256"),
            "bytes": src.get("bytes"),
            "rows_in_file": rows_in_file_meta,
            "rows_used": rows_used,
            "max_rows": src.get("max_rows"),
            "strategy": src.get("strategy"),
            "truncated": truncated,
        },
        "corpus_actual": {
            "path": corpus_fp["path"],
            "sha256": corpus_fp["sha256"],
            "bytes": corpus_fp["bytes"],
            "rows_in_file": corpus_rows_actual,
        },
        "sha256_match": sha_match,
        "bytes_match": corpus_fp["bytes"] == src.get("bytes"),
        "rows_in_file_match": corpus_rows_actual == rows_in_file_meta,
        "fit_on_full_snapshot": full_snapshot,
        "fit_on_labeled_train": fit_on_labeled_train,
        "labeled_reference": labeled_origin,
        "corpus_text_overlap_ods_keys": overlap_count,
        "corpus_text_overlap_source": overlap_source,
        "notes": provenance_notes,
    }
    if not (provenance["sha256_match"] and provenance["bytes_match"] and provenance["rows_in_file_match"]):
        failures.append("provenance корпуса не совпадает с метаданными артефакта")

    # --- Файлы артефакта против метаданных.
    files_report: dict[str, dict] = {}
    files_meta = meta.get("files") or {}
    for fname in ("features.joblib",):
        fpath = os.path.join(art_dir, fname)
        entry: dict = {"path": os.path.abspath(fpath)}
        if not os.path.exists(fpath):
            entry["status"] = "missing"
            failures.append(f"нет файла артефакта: {fpath}")
        else:
            fp = file_fingerprint(fpath)
            expected = files_meta.get(fname, {})
            entry.update({
                "sha256": fp["sha256"],
                "bytes": fp["bytes"],
                "meta_sha256": expected.get("sha256"),
                "meta_bytes": expected.get("bytes"),
                "sha256_match": fp["sha256"] == expected.get("sha256"),
                "bytes_match": fp["bytes"] == expected.get("bytes"),
            })
            if not (entry["sha256_match"] and entry["bytes_match"]):
                failures.append(f"hash/размер {fname} не совпадает с метаданными")
        files_report[fname] = entry

    # --- Загрузка, словарь и transform smoke.
    load_report: dict = {}
    try:
        loaded = load_vectorizer_artifact(art_dir)
        vocab = {
            "n_char": loaded.builder.n_char_features,
            "n_word": loaded.builder.n_word_features,
            "n_total": loaded.builder.n_features,
        }
        meta_vocab = meta.get("vocab", {})
        vocab_match_meta = vocab == meta_vocab
        if not vocab_match_meta:
            failures.append(f"загруженный словарь {vocab} != метаданным {meta_vocab}")
        if vocab["n_total"] != a.expect_vocab_total:
            failures.append(
                f"артефакт не является полным TF-IDF S02: n_total={vocab['n_total']}, "
                f"ожидалось {a.expect_vocab_total}"
            )
        X = loaded.builder.transform(PROBE_TEXTS)
        transform_ok = X.shape == (len(PROBE_TEXTS), vocab["n_total"])
        if not transform_ok:
            failures.append(f"transform smoke: форма {X.shape}, ожидалось {(len(PROBE_TEXTS), vocab['n_total'])}")
        empty = int((np.diff(X.indptr) == 0).sum())
        load_report = {
            "status": "ok",
            "loaded_vocab": vocab,
            "loaded_vocab_match_meta": vocab_match_meta,
            "fit_report": dict(loaded.builder.fit_report_),
            "transform_probe": {
                "texts_sha256_probe": "frozen-in-script: 3 строки, не из test",
                "shape": [int(X.shape[0]), int(X.shape[1])],
                "dtype": str(X.dtype),
                "empty_rows": empty,
            },
            "inference_contract": "после загрузки inference применяет готовый векторизатор через "
                                  "transform; повторный fit не выполняется",
        }
    except VectorizerArtifactError as e:
        load_report = {"status": f"failed: {e}"}
        failures.append(f"загрузка артефакта не удалась: {e}")

    ref = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "generator": {
            "script": os.path.abspath(__file__),
            "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]),
            "cwd": os.getcwd(),
            "git": _git_info(),
            "versions": _versions(),
        },
        "artifact": {
            "dir": os.path.abspath(art_dir),
            "name": a.full_name,
            "created_at": meta.get("created_at"),
            "vocab_meta": meta.get("vocab"),
        },
        "files": files_report,
        "corpus_provenance": provenance,
        "load_check": load_report,
        "usage_note": "Единственный full TF-IDF для всех runs S06; selected TF-IDF строятся "
                      "refit с фиксированными словарями на том же снимке корпуса; полный "
                      "артефакт не изменяется и не копируется.",
    }

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(ref, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({"ref": os.path.abspath(a.out), "status": ref["status"], "failures": failures,
                      "vocab": load_report.get("loaded_vocab")}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
