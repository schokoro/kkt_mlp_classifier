"""Supervised-отбор признаков TF-IDF (S03): ``global`` и ``class_aware`` Chi².

Отбор выполняется **только по размеченному train** (матрица признаков train и
закодированные метки L3). Метки validation/test сюда передавать нельзя.

Свойства реализации:

- детерминизм: RNG не используется; при равных оценках Chi² порядок задаётся
  возрастанием индекса признака (стабильная сортировка);
- корректный Chi²: ожидаемая сумма признака в классе равна
  ``feature_sum * n_class / n_train`` (доля **объектов** класса, а не доля
  суммы значений). ``global`` — стандартный многоклассовый Chi²
  (как ``sklearn.feature_selection.chi2``: сумма вкладов присутствующих ячеек
  ``(observed - expected)² / expected`` по всем классам). ``class_aware``
  ранжирует признаки внутри категории полным 2×2 OvR-статистикой с тем же
  ожиданием; положительная связь — строго ``observed > expected``: признаки с
  нулевым отклонением не являются кандидатами категории и могут попасть в
  выборку только через детерминированный добор из глобального рейтинга;
- раздельные бюджеты веток char/word без смешения индексов: сначала отбираются
  признаки в пространстве полного словаря (word-смещение ``n_char``), затем
  каждая ветка получает свои ``0..k_branch-1`` новые индексы. Бюджет, дающий
  пустую ветку, отклоняется до вычислений с ясной ошибкой;
- память: плотная матрица ``n_classes × n_features`` не создаётся и
  ``C × K`` кандидатов в RAM не хранится. Полные рейтинги пишутся на диск
  (каталог рейтингов) в точности ``float64`` — сохранённый рейтинг
  воспроизводит порядок отбора, включая близкие оценки, которые float32
  слил бы; class-aware выполняет **точный** равноправный циклический выбор
  (round-robin) из этих потоков с дедупликацией. Число открытых файловых
  дескрипторов не растёт с числом классов: все потоки одной ветки делят
  два дескриптора (индексы + оценки) и читают свои сегменты через
  seek/read; буфер на активный поток — не более ``chunk`` элементов,
  поэтому RAM класса-селектора растёт как ``O(n_classes × chunk)``,
  а не как полный объём рейтингов;
- ``class_aware``: One-vs-Rest по каждой присутствующей в train L3; объединение
  рейтингов категорий — равноправный циклический выбор с дедупликацией; при
  нехватке положительных кандидатов — детерминированный добор из глобального
  рейтинга, число добранных признаков фиксируется в отчёте.

Метод ``none`` здесь не вычисляется: контрольная конфигурация без отбора
использует исходный полный артефакт TF-IDF напрямую.

Полные рейтинги (версия ``kkt-selection-ratings@2``) — самостоятельный
результат отбора и часть версионированного артефакта (см.
:func:`save_selection_artifact`): по ним можно восстановить глобальный рейтинг
и рейтинги кандидатов категорий, проверить позиции невошедших признаков и
применить другой бюджет без пересчёта Chi². Повторное использование каталога
рейтингов проходит fail-closed проверку целостности
:func:`validate_ratings_dir`: наличие **всех** объявленных файлов обеих веток
(включая те, что текущий бюджет не читает), контрольные суммы из секции
``files`` (если записаны write path'ом через :func:`finalize_ratings_manifest`;
присутствующая секция обязана покрывать каждый объявленный файл, пустая —
отказ), согласованность массивов классов, per-class счётчиков (для G×I
``class_counts`` обязателен в обеих ветках, равен между ними поэлементно и
сходится с ``attribution.n_attributed_rows``; у Chi² S03 его нет и это валидно),
длин файлов ранжированных списков и ширин веток; ранжированные списки при этом
в RAM не загружаются. Артефакты отбора хранят рейтинги, исходные и новые индексы,
параметры и происхождение (источник TF-IDF и размеченных данных) и не
переносятся между контурами: в закрытом контуре TF-IDF, MLP и отбор строятся
заново.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import scipy.sparse as sp

from .artifact import environment_versions

log = logging.getLogger(__name__)

SELECTION_METHODS = ("global", "class_aware")
SELECTION_ARTIFACT_VERSION = "kkt-selection-artifact@3"
RATINGS_FORMAT_VERSION = "kkt-selection-ratings@2"
SELECTION_FILES = ("selection.joblib", "selection_meta.json")
RATINGS_DIRNAME = "ratings"
_RATINGS_STREAM_CHUNK = 8192

# Происхождение отобранного признака (записывается в payload["origin"]).
ORIGIN_CLASS = "class"           # выбран из ранжированного списка категории
ORIGIN_GLOBAL_FALLBACK = "global_fallback"  # добор из глобального рейтинга
ORIGIN_REMAINDER_FILL = "remainder_fill"    # добор остатка в стабильном порядке (G×I)


class SelectionError(RuntimeError):
    """Базовая ошибка операций отбора признаков."""


class SelectionArtifactError(SelectionError):
    """Артефакт отбора повреждён, неподготовлен или несовместим."""


@dataclass
class SelectionConfig:
    """Параметры supervised-отбора.

    ``k`` — суммарный бюджет признаков; ветки делят его по ``char_share``
    (char получает ``int(k * char_share)`` — округление вниз, детерминированно,
    остаток достаётся word). Каждая ветка затем усекается по фактически
    доступному числу признаков (без переноса остатка между ветками). Бюджет,
    при котором какая-либо ветка получает 0 признаков, отклоняется.
    ``seed`` в отборе не используется (RNG нет), фиксируется в метаданных
    для связности с прогоном.
    """

    method: str = "global"
    k: int | None = None
    char_share: float = 0.75
    seed: int = 42

    def validate(self) -> None:
        if self.method not in SELECTION_METHODS:
            raise ValueError(f"selection.method: одно из {SELECTION_METHODS} (режим 'none' не вычисляется)")
        if self.k is None or self.k <= 0:
            raise ValueError(f"selection.k: обязательное целое > 0, получено {self.k!r}")
        if not 0.0 <= self.char_share <= 1.0:
            raise ValueError(f"selection.char_share: доля в [0, 1], получено {self.char_share!r}")


def branch_budgets(k: int, char_share: float, n_char: int, n_word: int) -> tuple[int, int]:
    """Детерминированное деление бюджета ``k`` между ветками с усечением по факту.

    Char получает ``int(k * char_share)`` (вниз), word — остаток; затем каждый
    бюджет ограничивается числом доступных признаков ветки (без переноса).
    Комбинация, при которой хотя бы одна ветка получает ноль признаков
    (``char_share`` у границ 0/1 или слишком малое ``k``), отклоняется до
    запуска отбора: пустую ветку невозможно корректно переобучить.
    """
    if k <= 0:
        raise ValueError(f"k: целое > 0, получено {k!r}")
    if not 0.0 <= char_share <= 1.0:
        raise ValueError(f"char_share: в [0, 1], получено {char_share!r}")
    k_char_raw = int(k * char_share)
    k_char = min(k_char_raw, n_char)
    k_word = min(k - k_char_raw, n_word)
    if k_char == 0 or k_word == 0:
        raise ValueError(
            f"деление бюджета k={k}, char_share={char_share}: char→{k_char}, word→{k_word}; "
            "обе ветки обязаны получить ненулевой бюджет — увеличьте K или выберите "
            "char_share строго внутри (0, 1)"
        )
    return k_char, k_word


@dataclass
class BranchSelection:
    """Результат отбора одной ветки (индексы в пространстве полного словаря)."""

    indices: np.ndarray          # int64, отсортированы по возрастанию
    scores: np.ndarray | None    # float64, aligned с indices; None для method-заглушек
    origin: list[str]            # aligned: "class:<id>" | "global_fallback"
    ratings_dir: str | None = None  # каталог полных рейтингов (если создан здесь)

    def __post_init__(self) -> None:
        self.indices = np.asarray(self.indices, dtype=np.int64)
        if self.scores is not None:
            self.scores = np.asarray(self.scores, dtype=np.float64)
        if not (len(self.indices) == len(self.origin) and (self.scores is None or len(self.scores) == len(self.indices))):
            raise SelectionError("BranchSelection: длины indices/scores/origin не совпадают")


@dataclass
class SelectionResult:
    """Результат отбора: обе ветки, параметры, отчёт и полные рейтинги.

    ``ratings_dir`` — каталог с полными рейтингами (``kkt-selection-ratings@1``):
    глобальный рейтинг ветки и, для ``class_aware``, ранжированные списки
    положительных кандидатов по категориям. Каталог создаётся во время отбора
    (временный) и переносится в артефакт :func:`save_selection_artifact`;
    данные намеренно не загружаются в RAM целиком.
    """

    method: str
    k_requested: int
    char_share: float
    seed: int
    char: BranchSelection
    word: BranchSelection
    classes: list[str] | None = None   # метки L3 внутренних id классов OvR (или None)
    report: dict = field(default_factory=dict)
    ratings_dir: str | None = None
    ratings_manifest: dict | None = None

    @property
    def n_char_selected(self) -> int:
        return len(self.char.indices)

    @property
    def n_word_selected(self) -> int:
        return len(self.word.indices)

    def to_payload(self) -> dict:
        return {
            "version": SELECTION_ARTIFACT_VERSION,
            "method": self.method,
            "k_requested": int(self.k_requested),
            "char_share": float(self.char_share),
            "seed": int(self.seed),
            "classes": self.classes,
            "char": {
                "indices": self.char.indices,
                "scores": self.char.scores,
                "origin": list(self.char.origin),
            },
            "word": {
                "indices": self.word.indices,
                "scores": self.word.scores,
                "origin": list(self.word.origin),
            },
            "report": dict(self.report),
            "ratings": self.ratings_manifest,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "SelectionResult":
        if not isinstance(payload, dict):
            raise SelectionArtifactError(f"payload отбора: ожидался объект, получено {type(payload).__name__}")
        try:
            char = BranchSelection(
                indices=payload["char"]["indices"],
                scores=payload["char"].get("scores"),
                origin=list(payload["char"]["origin"]),
            )
            word = BranchSelection(
                indices=payload["word"]["indices"],
                scores=payload["word"].get("scores"),
                origin=list(payload["word"]["origin"]),
            )
            return cls(
                method=str(payload["method"]),
                k_requested=int(payload["k_requested"]),
                char_share=float(payload["char_share"]),
                seed=int(payload["seed"]),
                char=char,
                word=word,
                classes=payload.get("classes"),
                report=dict(payload.get("report") or {}),
                ratings_manifest=payload.get("ratings"),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise SelectionArtifactError(f"payload отбора неполный или некорректен: {e}") from e


def _check_inputs(X: sp.csr_matrix, y: np.ndarray) -> None:
    if X.ndim != 2 or X.shape[0] != len(y):
        raise SelectionError(f"X/y не согласованы: X{X.shape}, len(y)={len(y)}")
    if X.shape[0] == 0:
        raise SelectionError("Пустая выборка train: отбор невозможен")
    if X.shape[1] == 0:
        raise SelectionError("Пустое пространство признаков: отбор невозможен")
    if np.any(X.data < 0):
        raise SelectionError("Chi²-отбор требует неотрицательных признаков (TF-IDF)")
    y = np.asarray(y)
    if np.any(y < 0):
        raise SelectionError("В y есть неразмеченные строки (-1): отбор только по размеченному train")
    if y.dtype.kind not in "iu":
        raise SelectionError(f"y должны быть целочисленными id классов, получено dtype={y.dtype}")


def _rank_topk(scores: np.ndarray, k: int, mask: np.ndarray | None = None) -> np.ndarray:
    """Индексы top-k по убыванию ``scores``; равные оценки — по возрастанию индекса.

    Пороговое значение k-й оценки получается ``argpartition`` (только ускоритель;
    само значение точно при любом порядке равных элементов), а итоговый набор
    восстанавливается явно: все оценки выше порога плюс недостающее число
    равных порогу — с наименьшими индексами. Итог совпадает с полной
    стабильной сортировкой по (-оценка, индекс). O(F) памяти, без плотной
    матрицы и без зависимости от tie-порядка introselect.
    """
    idx = np.flatnonzero(mask) if mask is not None else np.arange(scores.shape[0], dtype=np.int64)
    if k <= 0 or idx.size == 0:
        return np.empty(0, dtype=np.int64)
    if idx.size > k:
        s = scores[idx]
        pos = np.argpartition(-s, kth=k - 1)[:k]
        thr = s[pos].min()
        above = idx[s > thr]
        n_ties_needed = k - len(above)
        if n_ties_needed > 0:
            ties = idx[s == thr]
            ties = ties[np.argsort(ties, kind="stable")][:n_ties_needed]
            idx = np.concatenate([above, ties])
        else:
            idx = above
    order = np.lexsort((idx, -scores[idx]))
    return idx[order]


def _class_chi2(
    X_class: sp.csr_matrix, feature_sum: np.ndarray, n_total: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2×2-статистика Pearson для одной категории против остальных.

    Признаки — суммы TF-IDF по строкам (неотрицательные). Ожидаемая сумма
    признака в классе: ``feature_sum * n_class / n_total``, где ``n_class`` —
    число объектов класса, ``n_total`` — число всех объектов train (стандартный
    Chi² по неотрицательным признакам, как в ``sklearn.feature_selection.chi2``;
    ожидание пропорционально доле **объектов** класса, а не доле суммы значений).
    Возвращает ``(chi2, observed, expected)`` длины F; ячейки с нулевым
    ожиданием дают нулевой вклад. ``observed > expected`` — строго положительная
    связь с категорией.
    """
    n_class = X_class.shape[0]
    observed = np.asarray(X_class.sum(axis=0), dtype=np.float64).ravel()
    expected = feature_sum * (n_class / n_total)
    other_obs = feature_sum - observed
    other_exp = feature_sum - expected
    chi2 = np.zeros(feature_sum.shape[0], dtype=np.float64)
    m1 = expected > 0
    chi2[m1] += (observed[m1] - expected[m1]) ** 2 / expected[m1]
    m2 = other_exp > 0
    chi2[m2] += (other_obs[m2] - other_exp[m2]) ** 2 / other_exp[m2]
    return chi2, observed, expected


def _ovr_parts(X: sp.csr_matrix, y: np.ndarray):
    """Общие суммы, упорядоченные классы и число объектов train для OvR."""
    y = np.asarray(y)
    classes = np.unique(y)
    feature_sum = np.asarray(X.sum(axis=0), dtype=np.float64).ravel()
    if float(feature_sum.sum()) <= 0:
        raise SelectionError("Сумма всех признаков train равна нулю: Chi² не определён")
    return classes, feature_sum, int(X.shape[0])


def _write_global_ratings(ratings_dir: str, branch: str, scores: np.ndarray) -> str:
    """Пишет полный глобальный рейтинг ветки (float64, сырой бинарный .npy).

    float64 обязателен: рейтинг должен точно воспроизводить порядок отбора
    (сортировка по убыванию оценки с разрывом равных по индексу), а float32
    сливал бы различимые оценки и менял top-K при повторном применении бюджета.
    """
    path = os.path.join(ratings_dir, f"{branch}_global.npy")
    np.asarray(scores, dtype=np.float64).tofile(path)
    return path


def _branch_ratings_entry(method: str, branch: str, n_features: int) -> dict:
    """Манифест файлов полных рейтингов одной ветки (имена по соглашению)."""
    entry: dict = {
        "n_features": int(n_features),
        "global_scores": f"{branch}_global.npy",
        "classes": None,
        "rank_indices": None,
        "rank_scores": None,
        "rank_counts": None,
        "class_counts": None,
    }
    if method == "class_aware":
        entry.update(
            classes=f"{branch}_classes.npy",
            rank_indices=f"{branch}_rank_idx.npy",
            rank_scores=f"{branch}_rank_score.npy",
            rank_counts=f"{branch}_rank_counts.npy",
        )
    return entry


def _ratings_file_names(manifest: dict) -> list[str]:
    """Все объявленные имена файлов рейтингов в манифесте."""
    names: list[str] = []
    for entry in (manifest.get("branches") or {}).values():
        if not isinstance(entry, dict):
            continue
        for key in ("global_scores", "classes", "rank_indices", "rank_scores", "rank_counts", "class_counts"):
            name = entry.get(key)
            if name:
                names.append(str(name))
    return names


_RATINGS_DECLARE_KEYS = ("global_scores", "classes", "rank_indices", "rank_scores", "rank_counts", "class_counts")
# Ключи, без которых повторное применение бюджета невозможно в принципе
# (class_counts опционален: его не было у Chi² S03; для G×I он обязателен —
# см. _CLASS_COUNTS_REQUIRED_METHOD).
_RATINGS_REQUIRED_KEYS = ("global_scores", "classes", "rank_indices", "rank_scores", "rank_counts")
# Метод, чьи манифесты обязаны объявлять class_counts (per-class N_c) в обеих
# ветках. Значение совпадает с kkt_clf.attribution.METHOD и продублировано,
# чтобы избежать циклического импорта attribution → selection.
_CLASS_COUNTS_REQUIRED_METHOD = "gradient_x_input"


def _ratings_declared_names(entry: dict) -> dict[str, str]:
    """Имена файлов, объявленные веткой манифеста рейтингов (пропускаются None)."""
    names: dict[str, str] = {}
    for key in _RATINGS_DECLARE_KEYS:
        name = entry.get(key)
        if name:
            names[key] = str(name)
    return names


def finalize_ratings_manifest(ratings_dir: str, manifest: dict) -> dict:
    """Дополняет манифест рейтингов контрольными суммами объявленных файлов.

    Вызывается на **write path** после записи всех файлов рейтингов: добавляет
    (на месте) секцию ``files`` — ``имя файла → {sha256, bytes}`` по каждому
    объявленному файлу обеих веток. :func:`validate_ratings_dir` сверяет суммы
    при повторном использовании, **если секция присутствует**: каталоги,
    записанные до её появления, без неё остаются валидными и проходят
    структурные проверки без криптографических утверждений. Присутствующая
    секция обязана покрывать **каждый** объявленный файл, а пустая секция
    отклоняется как повреждение манифеста (write path всегда пишет её непустой).
    """
    files: dict[str, dict] = {}
    for entry in (manifest.get("branches") or {}).values():
        if not isinstance(entry, dict):
            continue
        for name in _ratings_declared_names(entry).values():
            path = os.path.join(ratings_dir, name)
            files[name] = {"sha256": _sha256_file(path), "bytes": os.path.getsize(path)}
    if files:
        manifest["files"] = files
    return manifest


def _read_ratings_array(ratings_dir: str, name: str, dtype: np.dtype) -> np.ndarray:
    path = os.path.join(ratings_dir, name)
    try:
        return np.fromfile(path, dtype=dtype)
    except (OSError, ValueError) as e:
        raise SelectionArtifactError(f"Файл полных рейтингов {path!r} не читается: {e}") from e


def validate_ratings_dir(
    ratings_dir: str,
    manifest: dict,
    *,
    method: str | None = None,
    widths: dict[str, int] | None = None,
) -> dict[str, dict[str, np.ndarray | None]]:
    """Проверяет целостность каталога полных рейтингов перед повторным использованием.

    Fail closed: отказ при любой несогласованности (:class:`SelectionArtifactError`).
    Отбор из неполного или повреждённого каталога недопустим и не должен молча
    давать результат — в том числе при отсутствии объявленного файла, который
    текущий бюджет формально не читает (``class_counts`` не участвует в
    round-robin, но объявлен и является частью контракта каталога). Проверяется
    для **обеих веток** (char/word):

    - наличие каждого файла, объявленного в манифесте (не-None имена);
    - контрольные суммы из секции ``files`` манифеста. Секция либо отсутствует
      целиком (legacy-манифесты до её появления — только структурные проверки),
      либо присутствует и тогда обязана быть непустой и покрывать **каждый**
      объявленный файл: отсутствие записи о файле, пустая секция или
      несовпадение sha256/размера — отказ;
    - ширина ветки: положительная ``n_features`` манифеста равна фактической
      ширине (``widths``) и длине глобального рейтинга;
    - массивы классов: непустые и одинаковые между ветками (обе построены из
      одной атрибуции / одного train); ``rank_counts`` и (если объявлены)
      ``class_counts» той же длины, без отрицательных значений;
      ``rank_counts`` не превышает ширину ветки;
    - ``class_counts``: декларации обеих веток обязаны быть симметричны; для
      метода ``gradient_x_input`` объявление обязательно в обеих ветках, массивы
      N_c совпадают поэлементно между ветками, а их сумма сходится с заявленным
      ``attribution.n_attributed_rows``, если тот присутствует. Манифесты S03
      без ``class_counts`` в обеих ветках остаются валидными;
    - длины файлов ранжированных списков: размеры ``rank_indices`` и
      ``rank_scores`` совпадают между собой и равны ``sum(rank_counts) × 8``.

    Ранжированные списки в RAM **не загружаются**: длины проверяются по размеру
    файлов, значения индексов не перечитываются — потоковая семантика
    :func:`roundrobin_select_from_rank_files` сохраняется. Малые массивы
    (классы, счётчики, глобальный рейтинг) возвращаются по веткам, чтобы
    вызывающий код не читал их повторно.
    """
    if not isinstance(manifest, dict):
        raise SelectionArtifactError(
            f"Манифест рейтингов {ratings_dir!r}: ожидался объект, получено {type(manifest).__name__}"
        )
    if manifest.get("format") != RATINGS_FORMAT_VERSION:
        raise SelectionArtifactError(
            f"Несовместимая версия рейтингов {ratings_dir!r}: {manifest.get('format')!r}, "
            f"ожидается {RATINGS_FORMAT_VERSION!r}"
        )
    if method is not None and manifest.get("method") != method:
        raise SelectionArtifactError(
            f"Рейтинги в {ratings_dir!r} относятся к методу {manifest.get('method')!r}, ожидается {method!r}"
        )
    branches = manifest.get("branches")
    if not isinstance(branches, dict):
        raise SelectionArtifactError(f"Манифест рейтингов {ratings_dir!r} не объявляет ветки (branches)")
    # Секция контрольных сумм либо отсутствует целиком (legacy-манифесты,
    # записанные до её появления, — только структурные проверки), либо обязана
    # быть непустым объектом, покрывающим каждый объявленный файл. Пустая
    # секция не считается legacy: write path записывает её только непустой,
    # поэтому пустота — повреждение манифеста (fail closed).
    checksums_required = "files" in manifest
    if checksums_required:
        checksums = manifest["files"]
        if not isinstance(checksums, dict) or not checksums:
            raise SelectionArtifactError(
                f"Секция контрольных сумм 'files' в {ratings_dir!r} отсутствует, пуста или не является "
                "объектом: манифест не подтверждает целостность объявленных файлов (fail closed)"
            )
    else:
        checksums = {}

    arrays: dict[str, dict[str, np.ndarray | None]] = {}
    classes_per_branch: dict[str, np.ndarray] = {}
    for branch in ("char", "word"):
        entry = branches.get(branch)
        if not isinstance(entry, dict):
            raise SelectionArtifactError(f"Манифест рейтингов {ratings_dir!r} не объявляет ветку {branch!r}")
        names = _ratings_declared_names(entry)
        absent = [key for key in _RATINGS_REQUIRED_KEYS if key not in names]
        if absent:
            raise SelectionArtifactError(
                f"Манифест рейтингов {ratings_dir!r}: ветка {branch!r} не объявляет обязательные "
                f"файлы {absent}"
            )
        if manifest.get("method") == _CLASS_COUNTS_REQUIRED_METHOD and "class_counts" not in names:
            raise SelectionArtifactError(
                f"Манифест рейтингов {ratings_dir!r}: ветка {branch!r} не объявляет class_counts — "
                "для метода gradient_x_input обязательны per-class счётчики строк N_c обеих веток"
            )
        missing = [name for name in names.values() if not os.path.isfile(os.path.join(ratings_dir, name))]
        if missing:
            raise SelectionArtifactError(
                f"В каталоге рейтингов {ratings_dir!r} отсутствуют объявленные файлы ветки {branch!r}: "
                f"{sorted(missing)} — повторное использование рейтингов запрещено (fail closed)"
            )
        for name in names.values():
            finfo = checksums.get(name)
            if finfo is None:
                if checksums_required:
                    raise SelectionArtifactError(
                        f"В манифесте рейтингов {ratings_dir!r} нет контрольной суммы объявленного файла "
                        f"{name!r}: секция 'files' присутствует, но не покрывает все объявленные файлы "
                        "(fail closed)"
                    )
                continue
            if not isinstance(finfo, dict) or not isinstance(finfo.get("sha256"), str):
                raise SelectionArtifactError(
                    f"В манифесте рейтингов {ratings_dir!r} некорректная запись контрольной суммы "
                    f"файла {name!r}: ожидался sha256"
                )
            path = os.path.join(ratings_dir, name)
            if _sha256_file(path) != finfo["sha256"] or os.path.getsize(path) != finfo.get("bytes"):
                raise SelectionArtifactError(
                    f"Контрольная сумма/размер файла рейтингов {name!r} в {ratings_dir!r} не совпадают: "
                    "файл повреждён или подменён"
                )
        n_features = entry.get("n_features")
        if not isinstance(n_features, int) or isinstance(n_features, bool) or n_features <= 0:
            raise SelectionArtifactError(
                f"Манифест рейтингов {ratings_dir!r}: некорректная ширина ветки {branch!r}: {n_features!r}"
            )
        width = int((widths or {}).get(branch, n_features))
        if width <= 0:
            raise SelectionArtifactError(f"Ширина ветки {branch!r} должна быть > 0, получено {width}")
        if int(n_features) != width:
            raise SelectionArtifactError(
                f"Ширина ветки {branch!r} в манифесте рейтингов {ratings_dir!r} ({n_features}) не "
                f"совпадает с ожидаемой ({width}): каталог несовместим с данным признаковым пространством"
            )
        classes = _read_ratings_array(ratings_dir, names["classes"], np.dtype(np.int64))
        counts = _read_ratings_array(ratings_dir, names["rank_counts"], np.dtype(np.int64))
        global_scores = _read_ratings_array(ratings_dir, names["global_scores"], np.dtype(np.float64))
        class_counts = (
            _read_ratings_array(ratings_dir, names["class_counts"], np.dtype(np.int64))
            if "class_counts" in names else None
        )
        if len(classes) == 0:
            raise SelectionArtifactError(f"Список классов ветки {branch!r} пуст: {ratings_dir!r}")
        if len(counts) != len(classes):
            raise SelectionArtifactError(
                f"rank_counts ветки {branch!r} ({len(counts)} элементов) не согласован с числом "
                f"классов ({len(classes)}): {ratings_dir!r}"
            )
        if class_counts is not None and len(class_counts) != len(classes):
            raise SelectionArtifactError(
                f"class_counts ветки {branch!r} ({len(class_counts)} элементов) не согласован с "
                f"числом классов ({len(classes)}): {ratings_dir!r}"
            )
        if np.any(counts < 0) or np.any(counts > width):
            raise SelectionArtifactError(
                f"rank_counts ветки {branch!r} содержит недопустимые значения (отрицательные или "
                f"превышающие ширину {width}): {ratings_dir!r}"
            )
        if class_counts is not None and np.any(class_counts < 0):
            raise SelectionArtifactError(
                f"class_counts ветки {branch!r} содержит отрицательные значения: {ratings_dir!r}"
            )
        if len(global_scores) != width:
            raise SelectionArtifactError(
                f"Глобальный рейтинг ветки {branch!r}: {len(global_scores)} элементов при ширине "
                f"{width}: {ratings_dir!r}"
            )
        expected_bytes = int(counts.sum()) * np.dtype(np.int64).itemsize
        for key in ("rank_indices", "rank_scores"):
            size = os.path.getsize(os.path.join(ratings_dir, names[key]))
            if size != expected_bytes:
                raise SelectionArtifactError(
                    f"Файл {names[key]!r} ветки {branch!r}: размер {size} байт, ожидалось "
                    f"{expected_bytes} (= sum(rank_counts) × 8): файл оборван или повреждён ({ratings_dir!r})"
                )
        arrays[branch] = {
            "classes": classes,
            "counts": counts,
            "class_counts": class_counts,
            "global_scores": global_scores,
        }
        classes_per_branch[branch] = classes
    if not np.array_equal(classes_per_branch["char"], classes_per_branch["word"]):
        raise SelectionArtifactError(
            f"Списки классов веток не совпадают в {ratings_dir!r}: "
            f"char {classes_per_branch['char'].tolist()} != word {classes_per_branch['word'].tolist()} — "
            "ветки построены из разных данных"
        )
    cc_char = arrays["char"]["class_counts"]
    cc_word = arrays["word"]["class_counts"]
    if (cc_char is None) != (cc_word is None):
        raise SelectionArtifactError(
            f"class_counts объявлен не в обеих ветках {ratings_dir!r} "
            f"(char: {'объявлен' if cc_char is not None else 'отсутствует'}, "
            f"word: {'объявлен' if cc_word is not None else 'отсутствует'}) — декларации веток "
            "несогласованы (fail closed)"
        )
    if cc_char is not None:
        # Обе ветки построены из одной атрибуции: per-class N_c совпадают
        # поэлементно (равенство массивов, а не только сумм).
        if not np.array_equal(cc_char, cc_word):
            raise SelectionArtifactError(
                f"class_counts веток не совпадают в {ratings_dir!r}: "
                f"char {cc_char.tolist()} != word {cc_word.tolist()} — ветки построены из разных данных"
            )
        attribution = manifest.get("attribution")
        if isinstance(attribution, dict) and "n_attributed_rows" in attribution:
            n_attributed = attribution["n_attributed_rows"]
            if not isinstance(n_attributed, int) or isinstance(n_attributed, bool) or n_attributed < 0:
                raise SelectionArtifactError(
                    f"attribution.n_attributed_rows в {ratings_dir!r}: ожидалось целое >= 0, "
                    f"получено {n_attributed!r}"
                )
            if int(cc_char.sum()) != int(n_attributed):
                raise SelectionArtifactError(
                    f"Сумма class_counts ветки char ({int(cc_char.sum())}) не совпадает с "
                    f"attribution.n_attributed_rows ({n_attributed}) в {ratings_dir!r}: каталог "
                    "несогласован с заявленной атрибуцией (fail closed)"
                )
    return arrays


class _RankStream:
    """Последовательное чтение ранжированных кандидатов одной категории из пары
    сырых .npy-файлов (индексы int64 + оценки float64) буфером фиксированного
    размера. Позволяет выполнить точный round-robin, не загружая списки всех
    категорий в RAM. Все потоки одной ветки разделяют **два** открытых файловых
    дескриптора (передаются извне) и читают свои сегменты через seek/read, так
    что число дескрипторов не зависит от числа классов, а при ошибках остаётся
    только два описателя на закрытие внешним ``finally``. Буфер на поток — не
    более ``chunk`` элементов: RAM селектора ``O(n_classes × chunk)``, а не
    ``O(Σ длин списков)``. ``offset`` — позиция сегмента категории внутри
    общих файлов (в элементах)."""

    def __init__(self, fi, fs, n: int, offset: int = 0,
                 chunk: int = _RATINGS_STREAM_CHUNK):
        self._fi = fi
        self._fs = fs
        self._remaining = int(n)
        self._offset = int(offset)
        self._chunk = int(chunk)
        self._idx = np.empty(0, dtype=np.int64)
        self._score = np.empty(0, dtype=np.float64)
        self._pos = 0

    @staticmethod
    def _read_at(f, offset_elems: int, take: int, dtype: np.dtype) -> np.ndarray:
        f.seek(offset_elems * dtype.itemsize)
        buf = f.read(take * dtype.itemsize)
        if len(buf) != take * dtype.itemsize:
            raise SelectionArtifactError(
                "Файл полных рейтингов оборван: целостность артефакта отбора нарушена"
            )
        if len(buf) % dtype.itemsize:
            raise SelectionArtifactError(
                "Файл полных рейтингов повреждён (размер не кратен размеру элемента): "
                "целостность артефакта отбора нарушена"
            )
        return np.frombuffer(buf, dtype=dtype)

    def _refill(self) -> bool:
        take = min(self._chunk, self._remaining)
        if take <= 0:
            return False
        self._idx = self._read_at(self._fi, self._offset, take, np.dtype(np.int64))
        self._score = self._read_at(self._fs, self._offset, take, np.dtype(np.float64))
        self._offset += take
        self._remaining -= take
        self._pos = 0
        return True

    def next_unused(self, used: np.ndarray) -> tuple[int, float] | None:
        """Следующий неиспользованный кандидат в порядке ранжирования."""
        while True:
            if self._pos >= len(self._idx):
                if not self._refill():
                    return None
            f = int(self._idx[self._pos])
            s = float(self._score[self._pos])
            self._pos += 1
            if not used[f]:
                return f, s


def roundrobin_select_from_rank_files(
    idx_path: str,
    score_path: str,
    counts: np.ndarray,
    classes: np.ndarray,
    k: int,
    n_feat: int,
    global_scores: np.ndarray,
    positive_only_fallback: bool = False,
) -> tuple[BranchSelection, dict]:
    """Равноправный циклический выбор из ранжированных списков категорий.

    Единый механизм class-aware объединения S03/S04: классы обходятся в
    детерминированном (отсортированном) порядке, из списка каждой категории за
    цикл берётся не больше одного ещё не выбранного признака; порядок внутри
    списка — по убыванию оценки, равные оценки разрываются возрастанием
    индекса признака. Потоки разделяют два файловых дескриптора
    (``idx_path``/``score_path``), буфер потока — не более ``chunk`` элементов:
    точный round-robin без хранения ``C×K`` кандидатов в RAM.

    Добор до бюджета детерминирован: сначала из глобального рейтинга
    (``positive_only_fallback=False`` — top-k глобального рейтинга целиком,
    поведение S03; ``True`` — только глобально положительные оценки, G×I), и,
    если положительных кандидатов не хватает, из остальных признаков в
    стабильном порядке возрастания индекса с маркировкой
    ``remainder_fill``. Возвращает (BranchSelection, report) с числом
    классов, положительными кандидатами на класс, числом доборов
    (``n_fallback``, ``n_remainder_fill``).
    """
    counts = np.asarray(counts, dtype=np.int64)
    classes = np.asarray(classes, dtype=np.int64)
    global_scores = np.asarray(global_scores, dtype=np.float64)
    n_classes = len(classes)
    if len(counts) != n_classes:
        raise SelectionError("counts/classes не согласованы по длине")
    if global_scores.shape[0] != n_feat:
        raise SelectionError(
            f"global_scores: длина {global_scores.shape[0]} != n_feat {n_feat}"
        )
    used = np.zeros(n_feat, dtype=bool)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]) if n_classes else np.empty(0, dtype=np.int64)
    streams: list[_RankStream] = []
    f_i = f_s = None
    try:
        f_i = open(idx_path, "rb")
        f_s = open(score_path, "rb")
        streams = [
            _RankStream(f_i, f_s, int(counts[ci]), int(offsets[ci]))
            for ci in range(n_classes)
        ]
        selected: list[int] = []
        sel_scores: list[float] = []
        sel_origin: list[str] = []
        while len(selected) < k:
            progressed = False
            for ci in range(n_classes):
                if len(selected) >= k:
                    break
                nxt = streams[ci].next_unused(used)
                if nxt is None:
                    continue
                f, s = nxt
                used[f] = True
                selected.append(f)
                sel_scores.append(s)
                sel_origin.append(f"{ORIGIN_CLASS}:{int(classes[ci])}")
                progressed = True
            if not progressed:
                break
    finally:
        streams = []
        if f_s is not None:
            f_s.close()
        if f_i is not None:
            f_i.close()

    n_fallback = 0
    if len(selected) < k:
        if positive_only_fallback:
            # Добор из глобального рейтинга тех же положительных атрибуций.
            pos_idx = np.flatnonzero(global_scores > 0)
            order_g = np.lexsort((pos_idx, -global_scores[pos_idx]))
            fallback_pool = pos_idx[order_g]
        else:
            fallback_pool = _rank_topk(global_scores, k)
        for f in fallback_pool:
            if len(selected) >= k:
                break
            if used[f]:
                continue
            used[f] = True
            selected.append(int(f))
            sel_scores.append(float(global_scores[f]))
            sel_origin.append(ORIGIN_GLOBAL_FALLBACK)
            n_fallback += 1
    n_remainder_fill = 0
    if positive_only_fallback and len(selected) < k:
        # Положительных кандидатов не хватает: добор остальных признаков в
        # стабильном порядке (возрастание индекса) — обязательно с маркировкой.
        for f in range(n_feat):
            if len(selected) >= k:
                break
            if used[f]:
                continue
            used[f] = True
            selected.append(int(f))
            sel_scores.append(float(global_scores[f]))
            sel_origin.append(ORIGIN_REMAINDER_FILL)
            n_remainder_fill += 1
    order = np.argsort(np.asarray(selected, dtype=np.int64), kind="stable")
    idx = np.asarray(selected, dtype=np.int64)[order]
    scores = np.asarray(sel_scores, dtype=np.float64)[order]
    origin = [sel_origin[i] for i in order]
    report = {
        "n_classes": int(n_classes),
        "n_positive_per_class": [int(c) for c in counts],
        "n_fallback": int(n_fallback),
        "n_remainder_fill": int(n_remainder_fill),
    }
    return BranchSelection(indices=idx, scores=scores, origin=origin), report


def _select_branch_global(
    X_b: sp.csr_matrix, y: np.ndarray, k: int,
    ratings_dir: str | None = None, branch: str | None = None,
) -> BranchSelection:
    """Глобальный Chi²: стандартная многоклассовая статистика (как
    ``sklearn.feature_selection.chi2``) — сумма по классам вкладов
    ``(observed - expected)² / expected`` с ``expected = feature_sum * n_c / n``.
    Полный рейтинг ветки сохраняется на диск при заданном ``ratings_dir``.
    """
    classes, feature_sum, n_total = _ovr_parts(X_b, y)
    scores = np.zeros(X_b.shape[1], dtype=np.float64)
    for c in classes:
        observed = np.asarray(X_b[y == c].sum(axis=0), dtype=np.float64).ravel()
        expected = feature_sum * (float(np.count_nonzero(y == c)) / n_total)
        d = observed - expected
        m = expected > 0
        scores[m] += (d[m] ** 2) / expected[m]
    if ratings_dir is not None and branch is not None:
        _write_global_ratings(ratings_dir, branch, scores)
    top = _rank_topk(scores, k)
    return BranchSelection(
        indices=top,
        scores=scores[top].astype(np.float64),
        origin=[ORIGIN_GLOBAL_FALLBACK for _ in range(len(top))],
        ratings_dir=ratings_dir,
    )


def _select_branch_class_aware(
    X_b: sp.csr_matrix, y: np.ndarray, k: int,
    ratings_dir: str | None = None, branch: str | None = None,
) -> tuple[BranchSelection, dict]:
    """Class-aware Chi²: равноправный циклический выбор из ранжированных
    списков категорий с дедупликацией и детерминированным добором до бюджета.

    Полные рейтинги (глобальный + положительные кандидаты каждой категории,
    отсортированные по убыванию Chi² с разрывом ties по индексу признака)
    пишутся в ``ratings_dir`` и читаются последовательными потоками: в RAM не
    хранится ``C × K`` кандидатов и не создаётся плотная матрица ``C × F``.
    Round-robin точен: порядок выбора совпадает с обходом полных списков.
    """
    own_dir = ratings_dir is None
    if ratings_dir is None:
        ratings_dir = tempfile.mkdtemp(prefix="kkt-ratings-")
    # Файлы рейтингов пишутся всегда (метка ветки либо служебная): round-robin
    # читает их потоками даже при автономном вызове без артефакта.
    label = branch if branch is not None else "sel"
    classes, feature_sum, n_total = _ovr_parts(X_b, y)
    n_classes = len(classes)
    n_feat = X_b.shape[1]
    global_scores = np.zeros(n_feat, dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    idx_path = os.path.join(ratings_dir, f"{label}_rank_idx.npy")
    score_path = os.path.join(ratings_dir, f"{label}_rank_score.npy")
    counts_path = os.path.join(ratings_dir, f"{label}_rank_counts.npy")
    classes_path = os.path.join(ratings_dir, f"{label}_classes.npy")
    f_idx = f_score = None
    try:
        f_idx = open(idx_path, "wb")
        f_score = open(score_path, "wb")
        for ci in range(n_classes):
            chi2, observed, expected = _class_chi2(X_b[y == classes[ci]], feature_sum, n_total)
            # Глобальный многоклассовый рейтинг (presence-ячейки) — тот же,
            # что использует метод global, включая добор до бюджета.
            d = observed - expected
            m = expected > 0
            global_scores[m] += (d[m] ** 2) / expected[m]
            positive = (observed > expected) & (expected > 0)
            pos_idx = np.flatnonzero(positive)
            order = np.lexsort((pos_idx, -chi2[pos_idx]))
            ranked_idx = pos_idx[order]
            counts[ci] = len(ranked_idx)
            ranked_idx.tofile(f_idx)
            np.asarray(chi2[pos_idx][order], dtype=np.float64).tofile(f_score)
    finally:
        if f_score is not None:
            f_score.close()
        if f_idx is not None:
            f_idx.close()
    np.asarray(classes, dtype=np.int64).tofile(classes_path)
    counts.tofile(counts_path)
    _write_global_ratings(ratings_dir, label, global_scores)

    # Равноправный round-robin и добор из глобального рейтинга — общий
    # механизм с G×I (roundrobin_select_from_rank_files): классы в
    # детерминированном порядке, из списка каждой категории за цикл берётся не
    # больше одного признака, потоки разделяют два файловых дескриптора.
    try:
        sel, report = roundrobin_select_from_rank_files(
            idx_path, score_path, counts, classes, k, n_feat, global_scores,
            positive_only_fallback=False,
        )
    except Exception:
        if own_dir:
            shutil.rmtree(ratings_dir, ignore_errors=True)
        raise
    sel.ratings_dir = ratings_dir
    return sel, report


def select_features(
    X_train: sp.csr_matrix,
    y_train: np.ndarray,
    cfg: SelectionConfig,
    n_char: int,
    l3_labels: list[str] | None = None,
) -> SelectionResult:
    """Отбор признаков по размеченному train.

    ``X_train`` — матрица признаков train (CSR, неотрицательная, обычно из
    ``FeatureBuilder.transform`` полного артефакта); первые ``n_char`` столбцов
    — char-ветка, остальные — word. ``y_train`` — целочисленные id L3
    (кодировка ``LabelMaps``). ``l3_labels`` — список меток L3 по id (для
    метаданных; может быть None). Возвращает :class:`SelectionResult` с
    раздельными бюджетами веток, выбранными признаками, происхождением,
    отчётом и полными рейтингами на диске (``ratings_dir``), которые затем
    переносятся в артефект :func:`save_selection_artifact`.
    """
    cfg.validate()
    X = X_train if sp.isspmatrix_csr(X_train) else sp.csr_matrix(X_train)
    y = np.asarray(y_train)
    _check_inputs(X, y)
    if not 0 < n_char < X.shape[1]:
        raise SelectionError(f"n_char: граница веток в (0, {X.shape[1]}), получено {n_char!r}")
    k_char, k_word = branch_budgets(cfg.k, cfg.char_share, n_char, X.shape[1] - n_char)
    Xc = X[:, :n_char]
    Xw = X[:, n_char:]
    ratings_dir = tempfile.mkdtemp(prefix="kkt-ratings-")
    report: dict = {
        "method": cfg.method,
        "k_requested": int(cfg.k),
        "char_share": float(cfg.char_share),
        "seed": int(cfg.seed),
        "tie_policy": "stable sort by (-score, feature_index); no RNG",
        "budgets": {"k_char": int(k_char), "k_word": int(k_word)},
        "n_train_rows": int(X.shape[0]),
        "n_features_total": int(X.shape[1]),
        "expected": "feature_sum * n_class / n_train; global = sklearn multiclass chi2",
        "positive_rule": "observed > expected (expected > 0); zero-deviation features join only via global fallback",
    }
    try:
        if cfg.method == "global":
            sel_c = _select_branch_global(Xc, y, k_char, ratings_dir, "char")
            sel_w = _select_branch_global(Xw, y, k_word, ratings_dir, "word")
            report["char"] = {"n_selected": len(sel_c.indices), "n_fallback": 0}
            report["word"] = {"n_selected": len(sel_w.indices), "n_fallback": 0}
        else:
            sel_c, rep_c = _select_branch_class_aware(Xc, y, k_char, ratings_dir, "char")
            sel_w, rep_w = _select_branch_class_aware(Xw, y, k_word, ratings_dir, "word")
            report["char"] = {"n_selected": len(sel_c.indices), **rep_c}
            report["word"] = {"n_selected": len(sel_w.indices), **rep_w}
        manifest = {
            "format": RATINGS_FORMAT_VERSION,
            "method": cfg.method,
            "branches": {
                "char": _branch_ratings_entry(cfg.method, "char", n_char),
                "word": _branch_ratings_entry(cfg.method, "word", X.shape[1] - n_char),
            },
        }
        # Контракт write path: манифест дополняется контрольными суммами
        # объявленных файлов (проверяются при повторном использовании, если есть).
        finalize_ratings_manifest(ratings_dir, manifest)
        with open(os.path.join(ratings_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except Exception:
        shutil.rmtree(ratings_dir, ignore_errors=True)
        raise
    # Индексы результата — полно-пространные: word-ветка со смещением n_char.
    sel_w = BranchSelection(
        indices=sel_w.indices + n_char if len(sel_w.indices) else sel_w.indices,
        scores=sel_w.scores,
        origin=sel_w.origin,
    )
    classes = None
    if l3_labels is not None:
        present = np.unique(y)
        classes = [str(l3_labels[int(i)]) for i in present]
    log.info(
        "Отбор (%s): k=%d → char %d/%d, word %d/%d (fallback: char %d, word %d)",
        cfg.method, cfg.k, len(sel_c.indices), k_char, len(sel_w.indices), k_word,
        report["char"].get("n_fallback", 0), report["word"].get("n_fallback", 0),
    )
    return SelectionResult(
        method=cfg.method,
        k_requested=int(cfg.k),
        char_share=float(cfg.char_share),
        seed=int(cfg.seed),
        char=sel_c,
        word=sel_w,
        classes=classes,
        report=report,
        ratings_dir=ratings_dir,
        ratings_manifest=manifest,
    )


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_selection_artifact(
    out_dir: str,
    result: SelectionResult,
    provenance: dict,
    overwrite: bool = False,
    ratings_copy: bool = False,
) -> str:
    """Сохраняет версионированный артефакт отбора: payload (выбранные
    признаки/индексы/происхождение + манифест полных рейтингов), метаданные
    (параметры, provenance, версии, контрольные суммы всех файлов, включая
    рейтинги) и сами полные рейтинги (каталог ``ratings/``).

    Полные рейтинги — самостоятельный результат отбора: глобальный рейтинг
    обеих веток и ранжированные списки положительных кандидатов по каждой
    категории (``class_aware`` Chi² и ``gradient_x_input``). По ним можно
    проверить позиции невошедших признаков и применить другой бюджет без
    пересчёта оценок; они не загружаются в RAM целиком.

    ``provenance`` — сведения об источниках: как минимум ``vectorizer``
    (путь/идентификатор полного артефакта TF-IDF и его отпечаток) и ``labeled``
    (путь/отпечаток размеченных данных) — копируется в метаданные как есть.

    ``ratings_copy=True`` копирует рейтинги в артефакт, не забирая исходный
    каталог (нужно для G×I: рейтинги в ``shared/`` переиспользуются всеми
    бюджетами; артефакт каждого прогона остаётся самодостаточным для аудита).

    Сохранение **транзакционно** (в том числе при ``overwrite=True``): payload,
    рейтинги и метаданные собираются в соседнем скрытом staging-каталоге, и
    только после успешной сборки staging атомарно переносится на место
    ``out_dir`` (переименование, одна файловая система). При ошибке на любом
    шаге существующий артефакт остаётся нетронутым, а staging удаляется —
    невозможна ситуация «payload уже заменён, а рейтинги/метаданные старые»
    или вложение ``ratings/`` внутрь прежнего ``ratings/``.
    """
    from .vectorizer import check_output_dir

    if not isinstance(provenance, dict):
        raise SelectionError(f"provenance: ожидался объект, получено {type(provenance).__name__}")
    if not result.ratings_dir or not os.path.isdir(result.ratings_dir):
        raise SelectionError(
            "В SelectionResult нет каталога полных рейтингов: артефакт отбора версии "
            f"{SELECTION_ARTIFACT_VERSION} обязан их содержать. Отбор выполняйте "
            "через select_features (или передайте ratings_dir веткам отбора)."
        )
    check_output_dir(out_dir, overwrite)
    parent = os.path.dirname(os.path.abspath(out_dir)) or "."
    staging = tempfile.mkdtemp(prefix=".kkt-selection-", dir=parent)
    backup: str | None = None
    try:
        payload_path = os.path.join(staging, "selection.joblib")
        joblib.dump(result.to_payload(), payload_path)
        ratings_dest = os.path.join(staging, RATINGS_DIRNAME)
        if ratings_copy:
            shutil.copytree(result.ratings_dir, ratings_dest)
        else:
            shutil.move(result.ratings_dir, ratings_dest)
        files = {
            "selection.joblib": {
                "sha256": _sha256_file(payload_path),
                "bytes": os.path.getsize(payload_path),
            }
        }
        for name in sorted(os.listdir(ratings_dest)):
            fp = os.path.join(ratings_dest, name)
            if os.path.isfile(fp):
                files[f"{RATINGS_DIRNAME}/{name}"] = {"sha256": _sha256_file(fp), "bytes": os.path.getsize(fp)}
        meta = {
            "artifact_version": SELECTION_ARTIFACT_VERSION,
            "method": result.method,
            "k_requested": result.k_requested,
            "char_share": result.char_share,
            "seed": result.seed,
            "n_selected": {"char": result.n_char_selected, "word": result.n_word_selected},
            "report": result.report,
            "provenance": provenance,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "versions": environment_versions(),
            "notes": [
                "Рейтинги вычислены по размеченному train; метки val/test не использовались.",
                "Chi²: ожидание feature_sum * n_class / n_train; global — стандартный "
                "многоклассовый Chi² (sklearn.feature_selection.chi2).",
                "gradient_x_input: score(c,f) = mean_{i∈c} max(0, x·∂m/∂x) по margin "
                "истинной L3 против сильнейшего конкурента (S04); отрицательные/нулевые "
                "вклады только в диагностике; добор остатка маркируется remainder_fill.",
                "Полные рейтинги (каталог ratings/, формат kkt-selection-ratings@2) — "
                "самостоятельный результат отбора: позиции всех кандидатов и повторное "
                "применение бюджета без пересчёта; целиком в RAM не загружаются; "
                "оценки хранятся в float64 и точно воспроизводят порядок отбора.",
                "При равных оценках порядок задаётся индексом признака; RNG не применяется.",
                "class_aware: кандидаты категории — observed > expected (expected > 0); "
                "добор из глобального рейтинга отражён в report.*.n_fallback.",
                "Артефакт отбора не переносится между контурами: там TF-IDF, MLP и отбор "
                "строятся заново.",
            ],
            "files": files,
        }
        with open(os.path.join(staging, "selection_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
        # Атомарная установка: staging полностью собран, переименование в пределах
        # одной файловой системы; прежний каталог сначала убирается в backup, чтобы
        # при сбое второго переименования можно было откатить.
        if os.path.exists(out_dir):
            backup = f"{out_dir}.replaced-{os.getpid()}-{time.time_ns()}"
            os.rename(out_dir, backup)
        try:
            os.rename(staging, out_dir)
        except OSError:
            if backup is not None and not os.path.exists(out_dir):
                os.rename(backup, out_dir)
            raise
        staging = ""
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
        result.ratings_dir = os.path.join(out_dir, RATINGS_DIRNAME)
    finally:
        if staging and os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
    log.info("Артефакт отбора сохранён: %s (рейтинги: %d файлов)", out_dir,
             sum(1 for k in files if k.startswith(f"{RATINGS_DIRNAME}/")))
    return out_dir


def load_selection_artifact(path: str) -> tuple[SelectionResult, dict]:
    """Загружает артефакт отбора с проверкой целостности до десериализации.

    Помимо payload и контрольной суммы ``selection.joblib`` проверяет наличие
    каталога полных рейтингов и контрольные суммы каждого его файла по
    ``meta['files']``. Рейтинги в RAM не загружаются: путь к ним доступен как
    ``result.ratings_dir`` (формат — ``kkt-selection-ratings@1``, манифест —
    ``result.ratings_manifest``).
    """
    if not os.path.isdir(path):
        raise SelectionArtifactError(f"Артефакт отбора {path!r} не найден (нет каталога)")
    missing = [f for f in SELECTION_FILES if not os.path.exists(os.path.join(path, f))]
    if missing:
        raise SelectionArtifactError(f"В артефакте отбора {path} нет файлов: {missing}")
    with open(os.path.join(path, "selection_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise SelectionArtifactError(f"selection_meta.json в {path!r}: ожидался объект")
    if meta.get("artifact_version") != SELECTION_ARTIFACT_VERSION:
        raise SelectionArtifactError(
            f"Несовместимая версия артефакта отбора {path!r}: {meta.get('artifact_version')!r}, "
            f"ожидается {SELECTION_ARTIFACT_VERSION!r}"
        )
    finfo = (meta.get("files") or {}).get("selection.joblib")
    if not isinstance(finfo, dict) or not isinstance(finfo.get("sha256"), str):
        raise SelectionArtifactError(f"В метаданных {path!r} нет контрольной суммы selection.joblib")
    payload_path = os.path.join(path, "selection.joblib")
    if _sha256_file(payload_path) != finfo["sha256"] or os.path.getsize(payload_path) != finfo.get("bytes"):
        raise SelectionArtifactError(
            f"Контрольная сумма/размер selection.joblib в {path!r} не совпадают: файл повреждён или подменён"
        )
    try:
        payload = joblib.load(payload_path)
    except Exception as e:
        raise SelectionArtifactError(f"selection.joblib в {path!r} повреждён или не читается: {e}") from e
    result = SelectionResult.from_payload(payload)
    for name, branch in (("char", result.char), ("word", result.word)):
        if len(np.unique(branch.indices)) != len(branch.indices):
            raise SelectionArtifactError(f"Артефакт отбора {path!r}: дубликаты индексов в ветке {name!r}")
        if len(branch.indices) and (branch.indices.min() < 0):
            raise SelectionArtifactError(f"Артефакт отбора {path!r}: отрицательные индексы в ветке {name!r}")
    ratings_manifest = result.ratings_manifest
    if ratings_manifest:
        ratings_dir = os.path.join(path, RATINGS_DIRNAME)
        if not os.path.isdir(ratings_dir):
            raise SelectionArtifactError(
                f"В артефакте отбора {path!r} нет каталога полных рейтингов {RATINGS_DIRNAME!r}"
            )
        declared = meta.get("files") or {}
        for name in _ratings_file_names(ratings_manifest):
            rel = f"{RATINGS_DIRNAME}/{name}"
            info = declared.get(rel)
            fp = os.path.join(ratings_dir, name)
            if not isinstance(info, dict) or not os.path.isfile(fp):
                raise SelectionArtifactError(
                    f"В артефакте отбора {path!r} нет файла полных рейтингов {rel!r} "
                    "или его контрольной суммы в метаданных"
                )
            if _sha256_file(fp) != info.get("sha256") or os.path.getsize(fp) != info.get("bytes"):
                raise SelectionArtifactError(
                    f"Контрольная сумма/размер {rel!r} в {path!r} не совпадают: файл повреждён или подменён"
                )
        result.ratings_dir = ratings_dir
    return result, meta
