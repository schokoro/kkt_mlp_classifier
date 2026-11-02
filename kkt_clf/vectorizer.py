"""Автономный fit и сохранение TF-IDF (S01).

Отделяет построение признакового пространства от обучения классификатора:
``fit_and_save_vectorizer`` обучает ``FeatureBuilder`` на подготовленном
Parquet только с колонкой ``name`` (метки не нужны) и сохраняет автономный
артефакт — каталог::

    features.joblib      обученный FeatureBuilder (предобработка + обе ветки)
    vectorizer_meta.json версия, источник, объём, seed, конфиг TF-IDF,
                         размеры словарей, версии библиотек, ограничения

Чтение корпуса ограничено (``max_rows``): файл читается батчами только
колонки ``name``, размер батча не превышает оставшийся до предела объём;
целиком row-группа в память не загружается. Ограничение воспроизводимо:
«первые ``max_rows`` непустых ``name`` в порядке следования строк файла»;
оно зависит от версии файла и порядка строк.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any

import hashlib

import joblib
import numpy as np
import sklearn

from .artifact import environment_versions
from .config import BranchConfig, FeatureConfig
from .data import DataValidationError, file_fingerprint
from .features import FeatureBuilder

log = logging.getLogger(__name__)

VECTORIZER_ARTIFACT_VERSION = "kkt-tfidf-artifact@1"
ARTIFACT_FILES = ("features.joblib", "vectorizer_meta.json")
DEFAULT_CORPUS_MAX_ROWS = 1_000_000
READ_BATCH_ROWS = 50_000
NAME_COLUMN = "name"

# Секции метаданных, без которых артефакт считается неподготовленным.
REQUIRED_META_KEYS = ("artifact_version", "seed", "source", "tfidf_config", "vocab")

TRANSDUCTIVE_NOTE = (
    "Тексты внешнего корпуса не сравнивались с размеченными train/val/test: "
    "совпадения возможны. При их наличии оценка частично трансдуктивна, "
    "обобщение на полностью невиданный корпус не заявляется."
)

LIMITATION_NOTES = [
    TRANSDUCTIVE_NOTE,
    "Ограничение объёма воспроизводимо как «первые max_rows непустых name в "
    "порядке следования строк файла» и зависит от версии файла и порядка строк; "
    "случайная выборка требует предварительно подготовленного (перемешанного) сэмпла.",
    "Дубликаты текстов не удалялись: частоты IDF отражают исходный состав корпуса.",
    "sha256 источника считается одним проходом по файлу: RAM не растёт, но на "
    "очень больших файлах это заметный объём чтения с диска.",
]


class VectorizerError(RuntimeError):
    """Базовая ошибка standalone-операций с векторизатором."""


class VectorizerArtifactError(VectorizerError):
    """Артефакт повреждён, неподготовлен или несовместим с ожиданиями."""


@dataclass
class NameCorpus:
    """Ограниченное чтение колонки ``name``: тексты + сведения о проходе."""

    names: list[str]
    rows_in_file: int | None
    rows_read: int
    rows_dropped: int
    truncated: bool
    row_groups_read: int
    fingerprint: dict


@dataclass
class LoadedVectorizer:
    """Результат загрузки автономного артефакта."""

    builder: FeatureBuilder
    meta: dict


def load_name_corpus(path: str, max_rows: int) -> NameCorpus:
    """Читает колонку ``name`` из Parquet, набирая не более ``max_rows`` строк.

    Файл читается по row-группам, внутри группы — батчами ограниченного
    размера (не больше остатка до ``max_rows``): даже большая row-группа
    целиком в память не загружается. Пустые и null-значения отбрасываются
    (их число — в отчёте). ``max_rows`` обязателен: публичные standalone-операции
    не читают файл без ограничения (для чтения целиком задайте заведомо
    большое значение). ``truncated`` — признак того, что в файле остались
    непрочитанные строки; он ложен, если предел достигнут ровно на последней
    строке файла.
    """
    if max_rows is None or max_rows <= 0:
        raise ValueError(f"max_rows: обязательное целое > 0, получено {max_rows!r}")
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise VectorizerError(f"Не удалось открыть parquet {path!r}: {e}") from e
    if NAME_COLUMN not in pf.schema_arrow.names:
        raise DataValidationError(
            f"В корпусе {path!r} нет колонки {NAME_COLUMN!r}; есть: {list(pf.schema_arrow.names)}"
        )
    names: list[str] = []
    rows_read = 0
    row_groups_read = 0
    stopped_early = False
    for rg in range(pf.num_row_groups):
        if len(names) >= max_rows:
            stopped_early = True
            break
        row_groups_read += 1
        # Батч не больше остатка до предела: память ограничена даже в огромной группе.
        batch_size = min(READ_BATCH_ROWS, max_rows - len(names))
        try:
            batches = pf.iter_batches(
                batch_size=batch_size, row_groups=[rg], columns=[NAME_COLUMN]
            )
            for batch in batches:
                for v in batch.column(0).to_pylist():
                    rows_read += 1
                    if v is None:
                        continue
                    s = str(v).strip()
                    if not s:
                        continue
                    names.append(str(v))
                    if len(names) >= max_rows:
                        stopped_early = True
                        break
                if stopped_early:
                    break
        except VectorizerError:
            raise
        except Exception as e:
            raise VectorizerError(f"Ошибка чтения row-группы {rg} файла {path!r}: {e}") from e
        if stopped_early:
            break
    rows_in_file = getattr(pf.metadata, "num_rows", None)
    truncated = rows_read < rows_in_file if rows_in_file is not None else stopped_early
    dropped = rows_read - len(names)
    if not names:
        raise DataValidationError(
            f"Корпус {path!r} пуст: прочитано строк {rows_read}, непустых имён нет"
        )
    log.info(
        "Корпус %s: в файле %s строк, прочитано %d (row-групп %d), непустых %d, "
        "отброшено пустых %d, truncated=%s",
        path, rows_in_file, rows_read, row_groups_read, len(names), dropped, truncated,
    )
    return NameCorpus(
        names=names,
        rows_in_file=rows_in_file,
        rows_read=rows_read,
        rows_dropped=dropped,
        truncated=truncated,
        row_groups_read=row_groups_read,
        fingerprint=file_fingerprint(path),
    )


def tfidf_config_dict(feat_cfg: FeatureConfig) -> dict[str, Any]:
    """Конфигурация TF-IDF для метаданных (ветки + внутренние лимиты fit)."""
    def branch(b: BranchConfig) -> dict[str, Any]:
        d = dataclasses.asdict(b)
        d["ngram_range"] = list(b.ngram_range)
        return d

    return {
        "char": branch(feat_cfg.char),
        "word": branch(feat_cfg.word),
        "vectorizer_fit_rows": feat_cfg.vectorizer_fit_rows,
    }


def _same_config(meta_cfg: dict, feat_cfg: FeatureConfig) -> bool:
    return meta_cfg == tfidf_config_dict(feat_cfg)


def fit_vectorizer(
    feat_cfg: FeatureConfig,
    corpus_path: str,
    max_rows: int = DEFAULT_CORPUS_MAX_ROWS,
    seed: int = 42,
    source_id: str | None = None,
) -> tuple[FeatureBuilder, dict]:
    """Обучает FeatureBuilder на корпусе ``name`` без повторного чтения файла.

    ``max_rows`` обязателен (целое > 0): без ограничения standalone-fit
    не выполняется. Возвращает (builder, meta); meta — без
    ``created_at``/``versions``, их добавляет :func:`save_vectorizer_artifact`.
    """
    corpus = load_name_corpus(corpus_path, max_rows)
    fb = FeatureBuilder(feat_cfg).fit(corpus.names, seed=seed)
    meta = {
        "artifact_version": VECTORIZER_ARTIFACT_VERSION,
        "seed": int(seed),
        "source": {
            **corpus.fingerprint,
            "source_id": source_id,
            "rows_in_file": corpus.rows_in_file,
            "rows_read": corpus.rows_read,
            "rows_dropped_empty": corpus.rows_dropped,
            "rows_used": len(corpus.names),
            "max_rows": max_rows,
            "strategy": "head",
            "truncated": corpus.truncated,
            "row_groups_read": corpus.row_groups_read,
        },
        "tfidf_config": tfidf_config_dict(feat_cfg),
        "vocab": {
            "n_char": fb.n_char_features,
            "n_word": fb.n_word_features,
            "n_total": fb.n_features,
        },
        "fit": dict(getattr(fb, "fit_report_", {})),
        "notes": list(LIMITATION_NOTES),
    }
    log.info(
        "TF-IDF fit: корпус %s, непустых прочитано %d, в fit использовано %d (subsampled=%s), "
        "словарь char=%d word=%d",
        corpus.fingerprint["path"], len(corpus.names), fb.fit_report_["rows_fit"],
        fb.fit_report_["subsampled"], fb.n_char_features, fb.n_word_features,
    )
    return fb, meta


def _dump_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _is_int(v: Any) -> bool:
    """Целое число; bool не считается."""
    return isinstance(v, int) and not isinstance(v, bool)


def _require_int(obj: dict, key: str, where: str) -> int:
    v = obj.get(key)
    if not _is_int(v):
        raise VectorizerArtifactError(
            f"{where}: обязательное поле {key!r} (целое число) отсутствует или некорректно: {v!r}"
        )
    return v


def _validate_source_meta(source: dict, path: str) -> None:
    """Обязательные поля источника и согласованность счётчиков строк."""
    where = f"Секция 'source' метаданных артефакта {path!r}"
    src_path = source.get("path")
    if not isinstance(src_path, str) or not src_path:
        raise VectorizerArtifactError(f"{where}: обязательное поле 'path' (непустая строка) отсутствует")
    rows_used = _require_int(source, "rows_used", where)
    if rows_used < 1:
        raise VectorizerArtifactError(f"{where}: 'rows_used' должно быть >= 1, получено {rows_used}")
    rows_read = _require_int(source, "rows_read", where)
    if rows_read < rows_used:
        raise VectorizerArtifactError(
            f"{where}: 'rows_read' ({rows_read}) меньше 'rows_used' ({rows_used})"
        )
    dropped = _require_int(source, "rows_dropped_empty", where)
    if dropped != rows_read - rows_used:
        raise VectorizerArtifactError(
            f"{where}: 'rows_dropped_empty' должно быть rows_read - rows_used = "
            f"{rows_read - rows_used}, получено {dropped}"
        )
    max_rows = _require_int(source, "max_rows", where)
    if max_rows < 1:
        raise VectorizerArtifactError(f"{where}: 'max_rows' должно быть >= 1, получено {max_rows}")
    truncated = source.get("truncated")
    if not isinstance(truncated, bool):
        raise VectorizerArtifactError(
            f"{where}: обязательное поле 'truncated' (булево) отсутствует или некорректно: {truncated!r}"
        )
    rows_in_file = source.get("rows_in_file")
    if rows_in_file is not None and (not _is_int(rows_in_file) or rows_in_file < rows_read):
        raise VectorizerArtifactError(
            f"{where}: 'rows_in_file' должно быть целым >= rows_read={rows_read}, получено {rows_in_file!r}"
        )
    row_groups = source.get("row_groups_read")
    if row_groups is not None and (not _is_int(row_groups) or row_groups < 1):
        raise VectorizerArtifactError(
            f"{where}: 'row_groups_read' должно быть целым >= 1, получено {row_groups!r}"
        )


def _validate_fit_meta(fit: Any, rows_used: int, tfidf_config: dict, path: str) -> None:
    """Секция fit обязательна и согласована с источником и конфигурацией.

    ``rows_used`` источника — прочитанные непустые строки; ``rows_fit`` —
    сколько из них реально дошло до ``TfidfVectorizer.fit`` после ограничения
    ``vectorizer_fit_rows`` (``None`` — без ограничения), поэтому ожидается
    ``rows_fit == min(rows_in, vectorizer_fit_rows)``, а без лимита
    ``rows_fit == rows_in``; ``subsampled`` — ровно ``rows_fit < rows_in``.
    Отсутствующая, null или несогласованная с конфигурацией секция означает,
    что сведения о fit повреждены, и артефакт не принимается.
    """
    where = f"Секция 'fit' метаданных артефакта {path!r}"
    if not isinstance(fit, dict):
        raise VectorizerArtifactError(
            f"{where} отсутствует или некорректна (ожидался объект с полями "
            f"'rows_in', 'rows_fit', 'subsampled', получено {fit!r}): артефакт "
            "неподготовленный или создан другой версией"
        )
    rows_in = _require_int(fit, "rows_in", where)
    if rows_in != rows_used:
        raise VectorizerArtifactError(
            f"{where}: 'rows_in' должно совпадать с source.rows_used={rows_used}, получено {rows_in}"
        )
    rows_fit = _require_int(fit, "rows_fit", where)
    if not 1 <= rows_fit <= rows_in:
        raise VectorizerArtifactError(
            f"{where}: 'rows_fit' должно быть целым в [1, rows_in={rows_in}], получено {rows_fit}"
        )
    subsampled = fit.get("subsampled")
    if not isinstance(subsampled, bool):
        raise VectorizerArtifactError(
            f"{where}: обязательное поле 'subsampled' (булево) отсутствует или некорректно: {subsampled!r}"
        )
    if subsampled != (rows_fit < rows_in):
        raise VectorizerArtifactError(
            f"{where}: 'subsampled'={subsampled} не согласовано с rows_fit={rows_fit}, "
            f"rows_in={rows_in}"
        )
    vfr = tfidf_config.get("vectorizer_fit_rows")
    if vfr is None:
        expected_fit, expected_sub = rows_in, False
    else:
        if not _is_int(vfr) or vfr < 1:
            raise VectorizerArtifactError(
                f"{where}: 'vectorizer_fit_rows' в tfidf_config должно быть None или целым >= 1, "
                f"получено {vfr!r}"
            )
        expected_fit, expected_sub = min(rows_in, vfr), rows_in > vfr
    if rows_fit != expected_fit or subsampled != expected_sub:
        raise VectorizerArtifactError(
            f"{where}: сведения о fit повреждены: при vectorizer_fit_rows={vfr!r} и "
            f"rows_in={rows_in} ожидалось rows_fit={expected_fit}, subsampled={expected_sub}, "
            f"получено rows_fit={rows_fit}, subsampled={subsampled}"
        )


def check_output_dir(out_dir: str, overwrite: bool) -> None:
    """Проверяет, что ``out_dir`` не затирает существующие результаты.

    Вызывается до дорогого fit: непустой существующий каталог без
    ``overwrite`` — ошибка, ничего не создаётся и не перезаписывается.
    """
    if not os.path.exists(out_dir):
        return
    if not os.path.isdir(out_dir):
        raise VectorizerArtifactError(f"{out_dir!r} существует и не является каталогом")
    entries = os.listdir(out_dir)
    if entries and not overwrite:
        raise VectorizerArtifactError(
            f"Каталог {out_dir!r} не пуст ({len(entries)} объектов): не перезаписываю "
            "существующие результаты. Выберите другой путь или передайте overwrite=True / --force."
        )


def save_vectorizer_artifact(
    out_dir: str,
    builder: FeatureBuilder,
    meta: dict,
    overwrite: bool = False,
) -> str:
    """Сохраняет автономный артефакт. Не пустой каталог без ``overwrite`` не трогает."""
    if not builder.fitted:
        raise VectorizerArtifactError("FeatureBuilder не обучен — сохранять нечего")
    existed = os.path.exists(out_dir)
    check_output_dir(out_dir, overwrite)
    if not existed:
        os.makedirs(out_dir)
    fjob_path = os.path.join(out_dir, "features.joblib")
    joblib.dump(builder, fjob_path)
    full_meta = {
        **meta,
        "artifact_version": VECTORIZER_ARTIFACT_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "versions": environment_versions(),
        # Контрольная сумма сериализованного features.joblib: проверяется
        # до десериализации при загрузке.
        "files": {
            "features.joblib": {
                "sha256": _sha256_file(fjob_path),
                "bytes": os.path.getsize(fjob_path),
            }
        },
    }
    _dump_json(full_meta, os.path.join(out_dir, "vectorizer_meta.json"))
    log.info("Артефакт векторизатора сохранён: %s", out_dir)
    return out_dir


def fit_and_save_vectorizer(
    out_dir: str,
    feat_cfg: FeatureConfig,
    corpus_path: str,
    max_rows: int = DEFAULT_CORPUS_MAX_ROWS,
    seed: int = 42,
    source_id: str | None = None,
    overwrite: bool = False,
) -> tuple[FeatureBuilder, dict]:
    """Одним вызовом: bounded fit на корпусе → автономный артефакт на диске.

    Каталог ``out_dir`` проверяется до fit: существующие результаты
    без ``overwrite`` не затираются, а дорогой fit не выполняется.
    """
    check_output_dir(out_dir, overwrite)
    fb, meta = fit_vectorizer(feat_cfg, corpus_path, max_rows=max_rows, seed=seed, source_id=source_id)
    save_vectorizer_artifact(out_dir, fb, meta, overwrite=overwrite)
    return fb, meta


def load_vectorizer_artifact(
    path: str,
    expected_cfg: FeatureConfig | None = None,
) -> LoadedVectorizer:
    """Загружает автономный артефакт с проверкой целостности и совместимости.

    Проверки (в порядке выполнения): каталог и файлы на месте; метаданные
    читаются и содержат обязательные секции; ``seed`` — целое число; вложенные
    секции ``source``, ``tfidf_config``, ``vocab`` — объекты; обязательные поля
    ``source`` (path, rows_used, rows_read, rows_dropped_empty, max_rows,
    truncated) с согласованностью счётчиков; секция ``fit`` обязательна и
    согласована с источником (rows_fit <= rows_in == rows_used) и с
    конфигурацией ``vectorizer_fit_rows`` (rows_fit == min(rows_in, лимит),
    subsampled == rows_fit < rows_in); версия формата совпадает;
    ``versions`` и ``files`` — объекты корректных типов; контрольная сумма и
    размер ``features.joblib`` сверяются до десериализации; объект — обученный
    ``FeatureBuilder``, чей ``fit_report_`` совпадает с секцией ``fit``
    метаданных; размеры словарей и конфигурация TF-IDF совпадают с
    метаданными; IDF обеих веток корректны (длина равна размеру словаря,
    значения конечны и положительны). Версия sklearn сравнивается с сохранённой
    (несовпадение — предупреждение). Если задан ``expected_cfg``, конфигурация
    TF-IDF артефакта должна совпадать с ней (ветки и ``vectorizer_fit_rows``;
    ``transform_chunk_rows`` не учитывается — влияет только на чанкование
    при transform).
    """
    if not os.path.isdir(path):
        raise VectorizerArtifactError(f"Артефакт векторизатора {path!r} не найден (нет каталога)")
    missing = [f for f in ARTIFACT_FILES if not os.path.exists(os.path.join(path, f))]
    if missing:
        raise VectorizerArtifactError(f"В артефакте {path} нет файлов: {missing}")
    try:
        meta = _load_json(os.path.join(path, "vectorizer_meta.json"))
    except (OSError, json.JSONDecodeError) as e:
        raise VectorizerArtifactError(f"vectorizer_meta.json в {path!r} прочитать не удалось: {e}") from e
    if not isinstance(meta, dict):
        raise VectorizerArtifactError(f"vectorizer_meta.json в {path!r}: ожидался объект, получено {type(meta).__name__}")
    missing_keys = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing_keys:
        raise VectorizerArtifactError(
            f"В метаданных артефакта {path!r} нет обязательных секций {missing_keys}: "
            "артефакт неподготовленный или создан другой версией"
        )
    if not _is_int(meta["seed"]):
        raise VectorizerArtifactError(
            f"В метаданных артефакта {path!r} обязательное поле 'seed' (целое число) "
            f"отсутствует или некорректно: {meta['seed']!r} — артефакт неподготовленный или повреждён"
        )
    for k in ("source", "tfidf_config", "vocab"):
        if not isinstance(meta[k], dict):
            raise VectorizerArtifactError(f"Секция метаданных {k!r} в {path!r} должна быть объектом")
    _validate_source_meta(meta["source"], path)
    _validate_fit_meta(meta.get("fit"), meta["source"]["rows_used"], meta["tfidf_config"], path)
    version = meta["artifact_version"]
    if version != VECTORIZER_ARTIFACT_VERSION:
        raise VectorizerArtifactError(
            f"Несовместимая версия артефакта {path!r}: {version!r}, ожидается "
            f"{VECTORIZER_ARTIFACT_VERSION!r} (код этого пакета не умеет его готовить/проверять)"
        )
    versions = meta.get("versions", {})
    if not isinstance(versions, dict):
        raise VectorizerArtifactError(
            f"Секция метаданных 'versions' в {path!r} должна быть объектом, "
            f"получено {type(versions).__name__}"
        )
    for vkey, vval in versions.items():
        if not isinstance(vval, str):
            raise VectorizerArtifactError(
                f"Секция 'versions' в {path!r}: значение {vkey!r} должно быть строкой, "
                f"получено {vval!r}"
            )
    saved_skl = versions.get("sklearn")
    if saved_skl and saved_skl != sklearn.__version__:
        warnings.warn(f"Артефакт сохранён с sklearn {saved_skl}, загружается в {sklearn.__version__}")
    # Целостность сериализованного файла: checksum и размер сверяются ДО joblib.load.
    fjob_path = os.path.join(path, "features.joblib")
    files = meta.get("files", {})
    if not isinstance(files, dict):
        raise VectorizerArtifactError(
            f"Секция метаданных 'files' в {path!r} должна быть объектом, получено {type(files).__name__}"
        )
    finfo = files.get("features.joblib")
    if not isinstance(finfo, dict):
        raise VectorizerArtifactError(
            f"В метаданных {path!r} нет контрольной суммы features.joblib: "
            "артефакт неподготовленный или создан другой версией"
        )
    declared_sha = finfo.get("sha256")
    if (
        not isinstance(declared_sha, str)
        or len(declared_sha) != 64
        or any(c not in "0123456789abcdef" for c in declared_sha)
    ):
        raise VectorizerArtifactError(
            f"Контрольная сумма features.joblib в метаданных {path!r} некорректна: "
            f"ожидался sha256 (64 hex-символа), получено {declared_sha!r}"
        )
    actual_sha = _sha256_file(fjob_path)
    if actual_sha != declared_sha:
        raise VectorizerArtifactError(
            f"Контрольная сумма features.joblib в {path!r} не совпадает "
            f"(в метаданных {declared_sha[:16]}…, фактически {actual_sha[:16]}…): "
            "файл повреждён или подменён, десериализация не выполняется"
        )
    declared_bytes = finfo.get("bytes")
    actual_bytes = os.path.getsize(fjob_path)
    if not _is_int(declared_bytes) or declared_bytes != actual_bytes:
        raise VectorizerArtifactError(
            f"Размер features.joblib в метаданных {path!r} ({declared_bytes!r}) не совпадает "
            f"с фактическим ({actual_bytes} байт): файл повреждён или подменён, "
            "десериализация не выполняется"
        )
    try:
        builder = joblib.load(fjob_path)
    except Exception as e:
        raise VectorizerArtifactError(f"features.joblib в {path!r} повреждён или не читается: {e}") from e
    if not isinstance(builder, FeatureBuilder):
        raise VectorizerArtifactError(
            f"В {path!r} лежит {type(builder).__name__}, а не FeatureBuilder — "
            "это не автономный артефакт TF-IDF (или он неподготовленный)"
        )
    if not builder.fitted:
        raise VectorizerArtifactError(f"FeatureBuilder в {path!r} не обучен (fitted=False)")
    # Сведения о fit в метаданных должны соответствовать фактически
    # сериализованному объекту: без fit_report_ артефакт неподготовленный,
    # расхождение счётчиков — повреждённые метаданные.
    fit_report = getattr(builder, "fit_report_", None)
    if not (
        isinstance(fit_report, dict)
        and _is_int(fit_report.get("rows_in"))
        and _is_int(fit_report.get("rows_fit"))
        and isinstance(fit_report.get("subsampled"), bool)
    ):
        raise VectorizerArtifactError(
            f"Целостность артефакта {path!r} нарушена: у загруженного FeatureBuilder нет "
            "корректных сведений о fit (fit_report_ с целыми 'rows_in'/'rows_fit' и булевым "
            f"'subsampled'), получено {fit_report!r}: объект неподготовленный или создан другой версией"
        )
    for fkey in ("rows_in", "rows_fit", "subsampled"):
        if fit_report[fkey] != meta["fit"][fkey]:
            raise VectorizerArtifactError(
                f"Целостность артефакта {path!r} нарушена: fit[{fkey!r}] в метаданных "
                f"{meta['fit'][fkey]!r} не совпадает с fit_report_ FeatureBuilder "
                f"{fit_report[fkey]!r} (метаданные не соответствуют сериализованному объекту)"
            )
    vocab = meta["vocab"]
    actual_sizes = {
        "n_char": builder.n_char_features,
        "n_word": builder.n_word_features,
        "n_total": builder.n_features,
    }
    for size_key, size_actual in actual_sizes.items():
        size_declared = vocab.get(size_key)
        if not _is_int(size_declared) or size_declared != size_actual:
            raise VectorizerArtifactError(
                f"Целостность артефакта {path!r} нарушена: vocab[{size_key!r}] в метаданных "
                f"{size_declared!r}, фактически {size_actual} "
                "(артефакт неподготовленный или повреждён)"
            )
    actual_cfg = tfidf_config_dict(builder.cfg)
    if actual_cfg != meta["tfidf_config"]:
        raise VectorizerArtifactError(
            f"Целостность артефакта {path!r} нарушена: конфигурация TF-IDF в метаданных "
            "не совпадает с фактическим FeatureBuilder:\n"
            f"  в метаданных:  {json.dumps(meta['tfidf_config'], ensure_ascii=False, default=str)}\n"
            f"  в объекте:     {json.dumps(actual_cfg, ensure_ascii=False, default=str)}"
        )
    for label in ("char", "word"):
        vec = getattr(builder, label)
        idf = getattr(vec, "idf_", None)
        n = len(vec.vocabulary_)
        if idf is None or len(idf) != n or not np.all(np.isfinite(idf)) or not np.all(idf > 0):
            raise VectorizerArtifactError(
                f"IDF ветки {label!r} в {path!r} некорректен: ожидалось {n} конечных "
                f"положительных значений, получено {None if idf is None else len(idf)}"
            )
    if expected_cfg is not None and meta["tfidf_config"] != tfidf_config_dict(expected_cfg):
        raise VectorizerArtifactError(
            f"Конфигурация TF-IDF артефакта {path!r} не совпадает с ожидаемой:\n"
            f"  в артефакте: {json.dumps(meta.get('tfidf_config'), ensure_ascii=False, default=str)}\n"
            f"  ожидалось:   {json.dumps(tfidf_config_dict(expected_cfg), ensure_ascii=False, default=str)}"
        )
    log.info(
        "Векторизатор загружен из %s: корпус %s, прочитано непустых %s, в fit использовано %s, словарь %d",
        path, meta.get("source", {}).get("path"), meta.get("source", {}).get("rows_used"),
        (meta.get("fit") or {}).get("rows_fit"), builder.n_features,
    )
    return LoadedVectorizer(builder=builder, meta=meta)
