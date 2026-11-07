"""S06 (D09–D11): агрегированный отчёт сравнительного эксперимента (Gate 4).

Собирает восемь обязательных файлов ``outputs/S06/report/`` из сохранённых
машинных результатов контура Gate 2–3 (``kkt_clf.s06_comparison``), Gate 1
(``kkt_clf.s06_baseline``) и аудита Gate 0 — **без обучения, без fit TF-IDF и
без повторного inference**. Модуль не импортирует torch и не загружает
модели: источником чисел являются только JSON/CSV/Parquet, записанные
прогонами (runs/<slug>/run_result_seed_<s>.json, runs/predictions/*.parquet,
runs/stages.csv, shared/*, runs/baseline_full/*).

Файлы отчёта (контракт value_dod D11):

- ``report.html`` — автономный агрегированный отчёт (все 30 runs, paired Δ,
  сводки mean/std/min/max по seed, график K vs macro-F1, support bins,
  top confused pairs, ресурсы shared/per-run, формулы, ограничения,
  provenance и инструкция пересборки);
- ``runs.csv`` — run_id, method, K, seed, status/reason, метрики, paired Δ,
  config/hash;
- ``per_class.parquet`` — run_id, seed, L3, support, precision, recall, f1,
  delta_f1 (Δ к none того же seed);
- ``stages.csv`` — этапы comparison-контура и Gate 1 (source), elapsed,
  peak RSS/VRAM, artefact size; shared-затраты не дублируются по seed;
  для строк ``mlp_train`` проверенных успешных runs (включая переиспользованный
  baseline Gate 1) ``artefact_bytes`` — байтовый размер сериализованного
  ``model.pt`` из проверенного ``artifacts.checkpoint_dir`` результата run
  (sha256 сверен с ``hashes.model_pt_sha256``; размер каталога чекпоинта не
  используется); failed/pending runs и прочие этапы не изменяются;
- ``manifest.json`` — версии/хэши данных, сплитов, TF-IDF, рейтингов,
  отборов, refit-артефактов, чекпоинтов, протокола, условия и статусы;
- ``confusion_pairs.csv`` — run_id, true_L3 → predicted_L3, count,
  true_support, error_share, ранг внутри run;
- ``predictions_val.parquet`` — row_id, run_id, true_L3, predicted_L3/null;
- ``support_bins.csv`` — run_id, диапазон support (2–4, 5–9, 10–19, >=20),
  n_L3, macro-F1 бина, базовый macro-F1 бина (none того же seed) и Δ.

Формулы (единый протокол оценки, совпадает с Gate 1):

- per-class precision/recall/F1 — ``zero_division=0`` по фиксированной
  L3-вселенной label maps; support=0 → метрики null (не 0 и не фиктивный F1);
- macro-F1 run — невзвешенное среднее F1 по L3 с support>0 на comparison_val;
- accuracy — доля строк с top-1 == true L3 (пустые векторы признаков
  не исключаются: модель вызывается, bias-путь; их число фиксируется);
- paired Δ — метрика(run) − метрика(none того же train seed), null если
  baseline того seed неуспешен;
- macro-F1 support-бина — невзвешенное среднее per-L3 F1 внутри бина
  (не micro-accuracy); пустые бины — null с n_L3=0;
- error_share пары (true→predicted) — count / support(true) на comparison_val.

Существующие файлы не перезаписываются: при пересечении — отказ до любых
записей (для повторной пересборки удалите/переименуйте report/ вручную).
Честность статусов: planned runs без result-файла отображаются как
``pending`` (запуск не выполнялся); failed/skipped не скрываются.
Предсказания и per-class метрики включаются только для успешных runs;
``predicted_L3=null`` учитывается как ошибка и попадает в confused pairs.

Fail-closed приёмка предсказаний (status=success сам по себе недостаточен):
перед включением в отчёт результат успешного run сверяется с планом
(``run_id``, ``config_slug``, ``method``, ``budget``, ``seed``), обязан иметь
схему ``s06-run-result@1``, ``comparison_val_eval`` с ``predictions_path`` на
каноничный ``runs/predictions/<run_id>.parquet``, заполненные обязательные
``hashes`` (``model_pt_sha256``, ``tfidf_features_sha256``) и provenance
(``artifacts.checkpoint_dir``/``tfidf_artifact``, для методов отбора также
``selection_artifact``). Сам файл предсказаний принимается только при наличии
проверяемого fingerprint'а в результате run — канонично
``hashes.predictions_sha256`` (допускается
``comparison_val_eval.predictions_sha256``), — сверяемого с фактическим
sha256 parquet; отсутствие fingerprint'а — явный отказ сборки (требуется
запись sha256 предсказаний контуром ``kkt_clf.s06_comparison``). Дополнительно
предсказания привязываются к validation split. Приёмка строгая
(fail closed), без наследованных послаблений: для отчёта с успешными runs
обязательны непустые неизменяемые ``plan.protocol_sha256`` и
``plan.config.splits_manifest`` (план без этих полей не принимается —
отказ, а не legacy-fallback); manifest сплитов разрешается только по пути
из плана; замороженный ``shared/training_protocol.json`` обязан существовать
и совпадать с ``plan.protocol_sha256``; обязательный fingerprint
``comparison_val`` (sha256/bytes/n_rows) сверяется тремя сторонами —
splits manifest, ``protocol.splits.comparison_val`` (все три поля
обязательны, отсутствие — отказ, сравнение None==None запрещено) и
фактический parquet (sha256/bytes/число строк); полные label maps сплита
сверяются с baseline Gate 1 ``runs/baseline_full/manifest.json``
(``label_maps.l3_universe_sha256`` — канонический sha256 JSON label maps —
и ``n_classes`` по уровням L1/L2/L3; отсутствие baseline-полей — отказ).
Проверка выполняется только при наличии успешных runs; отчёт без успешных
runs и изолированная пересборка таблиц этих требований не предъявляют. А
``row_id``/``true_L3`` предсказаний обязаны совпасть со строками
``comparison_val`` (множество row_id, метки в пределах L3-вселенной label
maps, согласованность ``correct``, ``predicted_L3`` в пределах вселенной
либо null).

Автономная пересборка: если ``runs/planned.json`` недоступен, но сохранены
таблицы ``report/`` (runs.csv, stages.csv, manifest.json, confusion_pairs.csv,
support_bins.csv и пр.), ``build_s06_report`` пересоздаёт ``report.html``
**только из этих сохранённых таблиц** — без обращения к runs/, shared/,
models/, training/ и inference. В этом режиме перезаписывается только
``report.html``; таблицы не перезаписываются.
"""
from __future__ import annotations

import datetime as dt
import html
import json
import os
from typing import Any

import numpy as np
import pandas as pd

S06_REPORT_SCHEMA = "s06-report@1"

# Зеркало RUN_RESULT_SCHEMA из kkt_clf.s06_comparison: отчётный контур не
# импортирует comparison (тот тянет torch), поэтому константа продублирована.
RUN_RESULT_SCHEMA = "s06-run-result@1"

# Минимальный совместимый интерфейс fingerprint'а предсказаний в результате
# run: канонично result["hashes"]["predictions_sha256"]; допускается
# result["comparison_val_eval"]["predictions_sha256"]. Значение — sha256
# файла runs/predictions/<run_id>.parquet в момент записи результата.
PREDICTIONS_HASH_KEY = "predictions_sha256"

SUPPORT_BINS: tuple[tuple[str, int, int | None], ...] = (
    ("2-4", 2, 4),
    ("5-9", 5, 9),
    ("10-19", 10, 19),
    (">=20", 20, None),
)

REPORT_FILES = (
    "report.html",
    "runs.csv",
    "per_class.parquet",
    "stages.csv",
    "manifest.json",
    "confusion_pairs.csv",
    "predictions_val.parquet",
    "support_bins.csv",
)

FORMULAS = {
    "per_class": (
        "precision/recall/F1 считаются sklearn-семантикой с zero_division=0 по "
        "фиксированной L3-вселенной label maps; support — число строк L3 на "
        "comparison_val; support=0 → метрики null, класс в средние не входит"
    ),
    "macro_f1": (
        "macro-F1(run) = среднее F1 по L3 с support>0 на comparison_val "
        "(невзвешенное; классы без поддержки не исключаются молча — они null "
        "по явному правилу)"
    ),
    "accuracy": (
        "accuracy = доля строк comparison_val с top-1 предсказанием == true L3; "
        "пустые TF-IDF-векторы не исключаются (модель вызывается, bias-путь), "
        "их число — n_empty_features_val в результате run"
    ),
    "paired_delta": (
        "Δ(run) = metric(run) − metric(none того же train seed); null, если "
        "baseline этого seed неуспешен; три seed на одном split — не "
        "доверительный интервал"
    ),
    "support_bins": (
        "macro-F1 бина = невзвешенное среднее per-L3 F1 по L3, чей support на "
        "comparison_val попадает в диапазон (2–4, 5–9, 10–19, >=20); "
        "это не micro-accuracy; пустой бин → null, n_L3=0"
    ),
    "confusion_pair": (
        "error_share(true→predicted) = count(true→predicted) / support(true) "
        "на comparison_val; пары ранжируются по count внутри run"
    ),
    "chi2": (
        "Chi² (global/class_aware): ожидаемая сумма признака в классе "
        "feature_sum × n_class / n_train; class_aware — полные 2×2 OvR-"
        "статистики, кандидаты категории при observed > expected, объединение "
        "категорий — равноправный round-robin с детерминированным добором"
    ),
    "gxi": (
        "Gradient × Input: m_i = z_{i,y_i} − z_{i,r_i} (r_i — сильнейший другой "
        "класс); a_{i,f} = x_{i,f}·∂m_i/∂x_{i,f} по ненулевым CSR-значениям; "
        "score(c,f) = mean_{i∈c} max(0, a_{i,f}); отрицательные/нулевые "
        "вклады — только диагностика"
    ),
    "budget_split": (
        "k_char = int(K × char_share), k_word = K − k_char (char_share=0.75); "
        "бюджеты применены из сохранённых полных рейтингов без пересчёта оценок"
    ),
}

LIMITATIONS = [
    "ODS — исследовательский PoC: результаты на comparison_val диагностические, "
    "не оценка промышленной применимости и не экстраполяция на реальные чеки "
    "закрытого контура.",
    "sealed_test не использовался ни для inference, ни для метрик; test-метрик "
    "в отчёте нет; выбор «победителя для промышленности» вне S06.",
    "Три seed на одном split — не независимый доверительный интервал; мизерные "
    "Δ не доказывают улучшение.",
    "Выбранные словари зафиксированы для всех seed (рейтинги одни, вариабельность "
    "только обучения MLP) — вариабельность самих рейтингов не измеряется.",
    "Режим pretrained трансдуктивен: тексты корпуса full TF-IDF могут совпадать "
    "с train/val/test; метки при этом не переносятся (см. аудит Gate 0).",
    "Категории с support 2 на comparison_val нестабильны (два примера задают "
    "recall с шагом 0.5); per-class Δ для них трактуются осторожно.",
    "GPU/CPU-характеристики локальны; при недоступности GPU peak_vram=null "
    "(не ноль); ресурсные цифры не переносятся на другие машины.",
    "comparison_val не участвовал в обучении, выборе эпох, checkpoint и "
    "ранжировании; он использован только для сравнения методов.",
]


class S06ReportError(RuntimeError):
    """Ошибка сборки отчёта S06; существующие файлы не перезаписываются."""


# ---------------------------------------------------------------------------
# Загрузка сохранённых результатов (только чтение)
# ---------------------------------------------------------------------------


def _read_json(path: str, required: bool = True) -> dict | None:
    if not os.path.isfile(path):
        if required:
            raise S06ReportError(f"Файл результата не найден: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _check_new(paths: list[str]) -> None:
    existing = [p for p in paths if os.path.exists(p)]
    if existing:
        raise S06ReportError(
            "Существующие файлы не перезаписываются; удалите/переименуйте вручную: "
            + ", ".join(existing)
        )


def _load_run_results(output_root: str, plan: dict) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for r in plan["runs"]:
        slug, seed = r["config_slug"], r["seed"]
        path = os.path.join(output_root, "runs", slug, f"run_result_seed_{seed}.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                results[r["run_id"]] = json.load(f)
    return results


def _verify_run_result_binding(plan_entry: dict, result: dict) -> None:
    """Fail-closed привязка результата успешного run к плану и provenance.

    ``status=success`` сам по себе не принимается: результат обязан иметь
    схему ``s06-run-result@1``, совпадать с записью плана по идентифицирующим
    полям, содержать оценку comparison_val со ссылкой на каноничный файл
    предсказаний, заполненные обязательные хэши и provenance артефактов.
    """
    run_id = plan_entry["run_id"]
    if result.get("schema_version") != RUN_RESULT_SCHEMA:
        raise S06ReportError(
            f"run {run_id}: результат имеет схему {result.get('schema_version')!r}, "
            f"ожидается {RUN_RESULT_SCHEMA!r} — success не подтверждён"
        )
    for key in ("run_id", "config_slug", "method", "budget", "seed"):
        if result.get(key) != plan_entry.get(key):
            raise S06ReportError(
                f"run {run_id}: результат не совпадает с планом по полю {key!r} "
                f"({result.get(key)!r} != {plan_entry.get(key)!r}) — success не подтверждён"
            )
    eval_info = result.get("comparison_val_eval")
    if not isinstance(eval_info, dict):
        raise S06ReportError(
            f"run {run_id}: успешный результат без comparison_val_eval — success не подтверждён"
        )
    recorded_path = eval_info.get("predictions_path")
    if (not isinstance(recorded_path, str)
            or os.path.basename(recorded_path) != f"{run_id}.parquet"):
        raise S06ReportError(
            f"run {run_id}: comparison_val_eval.predictions_path ({recorded_path!r}) не указывает "
            f"на каноничный runs/predictions/{run_id}.parquet — success не подтверждён"
        )
    hashes = result.get("hashes")
    if not isinstance(hashes, dict) or any(
        not hashes.get(k) for k in ("model_pt_sha256", "tfidf_features_sha256")
    ):
        raise S06ReportError(
            f"run {run_id}: обязательные hashes (model_pt_sha256, tfidf_features_sha256) "
            "не заполнены — success не подтверждён"
        )
    artifacts = result.get("artifacts") or {}
    required_artifacts = ["checkpoint_dir", "tfidf_artifact"]
    if plan_entry.get("method") != "none":
        required_artifacts.append("selection_artifact")
    missing = [k for k in required_artifacts if not artifacts.get(k)]
    if missing:
        raise S06ReportError(
            f"run {run_id}: provenance артефактов не полон ({', '.join(missing)}) — "
            "success не подтверждён"
        )


def _sha256_json(payload: Any) -> str:
    """Детерминированный sha256 канонического JSON-представления объекта.

    Зеркало ``kkt_clf.experiment._sha256_json``: модуль отчёта не импортирует
    experiment (тот тянет torch), поэтому канонизация продублирована —
    ``json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)``.
    """
    import hashlib

    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _resolve_splits_manifest_path(output_root: str, plan: dict) -> str:
    """Каноничный путь к manifest сплитов: из неизменяемого плана, fail closed.

    План фиксирует ``config.splits_manifest`` при записи (абсолютный путь) —
    только он и принимается; сторонний/подменённый manifest по другому пути
    не подменяется найденным в ``output_root``. Функция вызывается только при
    наличии успешных runs, поэтому план обязан фиксировать непустое поле, а
    файл по этому пути обязан существовать; наследованный fallback на
    ``output_root/splits/manifest.json`` для успешных отчётов недопустим.
    """
    declared = (plan.get("config") or {}).get("splits_manifest")
    if not isinstance(declared, str) or not declared.strip():
        raise S06ReportError(
            "План runs/planned.json не фиксирует непустой config.splits_manifest — "
            "при наличии успешных runs привязка к validation split без "
            "неизменяемого пути из плана запрещена (fail closed, legacy-fallback "
            "отсутствует)"
        )
    if not os.path.isfile(declared):
        raise S06ReportError(
            f"Splits manifest из plan.config.splits_manifest не найден: {declared!r} — "
            "привязка к validation split невозможна (fail closed)"
        )
    return declared


def _verify_training_protocol(output_root: str, plan: dict) -> dict:
    """Сверяет замороженный ``shared/training_protocol.json`` с ``plan.protocol_sha256``.

    Функция вызывается только при наличии успешных runs, поэтому план обязан
    фиксировать непустый ``protocol_sha256``; файл обязан существовать и
    совпадать с ним побайтово — несовпадение или отсутствие (с одной из
    сторон) — отказ (подмена протокола недопустима). Возвращает распарсенный
    протокол.
    """
    path = os.path.join(output_root, "shared", "training_protocol.json")
    declared = plan.get("protocol_sha256")
    if not isinstance(declared, str) or not declared.strip():
        raise S06ReportError(
            "План runs/planned.json не фиксирует непустой protocol_sha256 — при "
            "наличии успешных runs протокол без сверки с планом не принимается "
            "(fail closed)"
        )
    fp = _fingerprint_file(path)
    if fp is None:
        raise S06ReportError(
            f"План фиксирует protocol_sha256, но замороженный протокол не найден: "
            f"{path} — fail closed"
        )
    if fp["sha256"] != declared:
        raise S06ReportError(
            f"shared/training_protocol.json не совпадает с protocol_sha256 плана "
            f"({fp['sha256']} != {declared}) — подмена протокола недопустима"
        )
    return _read_json(path)


def _part_fingerprint(part: dict, where: str) -> tuple[str, int, int]:
    """Обязательный fingerprint части сплита (sha256/bytes/n_rows), fail closed.

    Отсутствие или пустота любого из трёх полей — отказ: сравнение
    None==None не является проверкой целостности.
    """
    sha = part.get("sha256")
    nbytes = part.get("bytes")
    n_rows = part.get("n_rows")
    if (not isinstance(sha, str) or not sha.strip()) or nbytes is None or n_rows is None:
        raise S06ReportError(
            f"{where}: обязательный fingerprint части (sha256/bytes/n_rows) "
            "отсутствует или пуст — приёмка без проверяемого fingerprint "
            "запрещена (fail closed)"
        )
    try:
        return sha, int(nbytes), int(n_rows)
    except (TypeError, ValueError) as exc:
        raise S06ReportError(
            f"{where}: fingerprint части нечисловой (bytes={nbytes!r}, "
            f"n_rows={n_rows!r}) — fail closed"
        ) from exc


def _load_comparison_val_binding(output_root: str, plan: dict) -> dict[str, Any]:
    """Проверяемая привязка к validation split (comparison_val), fail closed.

    Manifest сплитов разрешается только из неизменяемого плана
    (:func:`_resolve_splits_manifest_path` — без legacy-fallback), замороженный
    протокол обязан существовать и совпадать с обязательным непустым
    ``plan.protocol_sha256`` (:func:`_verify_training_protocol`), fingerprint
    ``comparison_val`` (sha256/bytes/n_rows — все три поля обязательны, без
    None==None) сверяется тремя сторонами: splits manifest, обязательный
    ``protocol.splits.comparison_val`` и фактический parquet (sha256/bytes/
    число строк); полные label maps сверяются с baseline Gate 1
    (:func:`_verify_label_maps_against_baseline`); метки parquet обязаны лежать
    в L3-вселенной label maps. Функция вызывается только при наличии успешных
    runs. Возвращает строки (row_id, true_L3), L3-вселенную label maps и
    fingerprint части.
    """
    manifest_path = _resolve_splits_manifest_path(output_root, plan)
    manifest = _read_json(manifest_path)
    protocol = _verify_training_protocol(output_root, plan)
    part = (manifest.get("parts") or {}).get("comparison_val")
    if not isinstance(part, dict) or not part.get("sha256") or part.get("n_rows") is None:
        raise S06ReportError(
            f"{manifest_path}: нет проверяемого fingerprint comparison_val "
            "(sha256/n_rows) — привязка предсказаний к validation split невозможна"
        )
    proto_part = ((protocol or {}).get("splits") or {}).get("comparison_val")
    if not isinstance(proto_part, dict) or not proto_part:
        raise S06ReportError(
            "shared/training_protocol.json не фиксирует protocol.splits.comparison_val — "
            "при наличии успешных runs fingerprint validation split обязан быть "
            "засвидетельствован замороженным протоколом (fail closed, отсутствие "
            "поля не допускает сборку)"
        )
    declared = _part_fingerprint(part, f"{manifest_path} (parts.comparison_val)")
    recorded = _part_fingerprint(proto_part, "shared/training_protocol.json (splits.comparison_val)")
    if declared != recorded:
        raise S06ReportError(
            f"{manifest_path}: fingerprint comparison_val (sha256/bytes/n_rows) не "
            "совпадает с замороженным протоколом (protocol.splits.comparison_val: "
            f"manifest={declared}, protocol={recorded}) — подмена splits manifest "
            "или протокола недопустима"
        )
    manifest_input = (manifest.get("inputs") or {}).get("labeled_parquet") or {}
    proto_input = (protocol or {}).get("labeled_parquet") or {}
    if (manifest_input.get("sha256") and proto_input.get("sha256")
            and manifest_input["sha256"] != proto_input["sha256"]):
        raise S06ReportError(
            f"{manifest_path}: labeled_parquet не совпадает с замороженным протоколом — "
            "подмена splits manifest недопустима"
        )
    if part.get("file"):
        path = part["file"]
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), path)
        if not os.path.isfile(path):
            raise S06ReportError(
                f"Файл comparison_val из manifest сплитов не найден: {part['file']!r}"
            )
    else:
        path = os.path.join(output_root, "splits", "comparison_val.parquet")
        if not os.path.isfile(path):
            raise S06ReportError(
                f"Файл comparison_val не найден: ни 'file' в manifest, ни {path!r}"
            )
    actual = _fingerprint_file(path)
    frame = pd.read_parquet(path, columns=["row_id", "L3"])
    if (actual["sha256"], actual["bytes"], int(len(frame))) != declared:
        raise S06ReportError(
            f"comparison_val ({path}) не совпадает с fingerprint из manifest сплита "
            f"(фактические sha256/bytes/rows = {(actual['sha256'], actual['bytes'], int(len(frame)))}, "
            f"manifest = {declared}): подмена validation split недопустима"
        )
    if frame["row_id"].duplicated().any():
        raise S06ReportError("comparison_val: дубликаты row_id — привязка невозможна")
    _verify_label_maps_against_baseline(output_root, manifest_path, manifest)
    l3_universe = set((manifest.get("label_maps") or {}).get("L3") or [])
    if not l3_universe:
        raise S06ReportError(
            f"{manifest_path}: пустые label_maps.L3 — L3-вселенная не зафиксирована"
        )
    unknown_true = set(frame["L3"]) - l3_universe
    if unknown_true:
        raise S06ReportError(
            f"comparison_val: метки L3 вне зафиксированной вселенной label maps: "
            f"{sorted(unknown_true)[:5]}"
        )
    return {
        "rows": frame.rename(columns={"L3": "true_L3"}),
        "l3_universe": l3_universe,
        "sha256": declared[0],
        "n_rows": declared[2],
        "path": os.path.abspath(path),
        "manifest_path": os.path.abspath(manifest_path),
    }


def _verify_label_maps_against_baseline(output_root: str, manifest_path: str, manifest: dict) -> None:
    """Сверяет полные label maps сплита с baseline Gate 1, fail closed.

    Канонический sha256 JSON полного словаря ``label_maps`` из splits manifest
    обязан совпадать с ``label_maps.l3_universe_sha256`` baseline-манифеста
    ``runs/baseline_full/manifest.json``, а число классов по уровням L1/L2/L3 —
    с его ``n_classes``. Отсутствие или пустота любого обязательного поля
    (полных label maps сплита, baseline-манифеста, ``l3_universe_sha256``,
    ``n_classes``) — отказ: подмена или «обновление» label maps после Gate 1
    недопустимы, None==None не является проверкой.
    """
    maps = manifest.get("label_maps")
    levels = ("L1", "L2", "L3")
    if not isinstance(maps, dict) or any(
        not isinstance(maps.get(level), list) or not maps.get(level) for level in levels
    ):
        raise S06ReportError(
            f"{manifest_path}: неполные label_maps (обязательны непустые списки "
            "L1/L2/L3) — сверка с baseline Gate 1 невозможна (fail closed)"
        )
    baseline_path = os.path.join(output_root, "runs", "baseline_full", "manifest.json")
    baseline = _read_json(baseline_path)
    base_maps = (baseline or {}).get("label_maps") or {}
    recorded_sha = base_maps.get("l3_universe_sha256")
    if not isinstance(recorded_sha, str) or not recorded_sha.strip():
        raise S06ReportError(
            f"{baseline_path}: нет непустого label_maps.l3_universe_sha256 — "
            "сверка label maps со baseline Gate 1 невозможна (fail closed)"
        )
    recorded_counts = base_maps.get("n_classes")
    if not isinstance(recorded_counts, dict):
        raise S06ReportError(
            f"{baseline_path}: нет label_maps.n_classes — сверка label maps со "
            "baseline Gate 1 невозможна (fail closed)"
        )
    expected_counts = {level: len(maps[level]) for level in levels}
    try:
        counts = {level: int(recorded_counts.get(level, -1)) for level in levels}
    except (TypeError, ValueError) as exc:
        raise S06ReportError(
            f"{baseline_path}: label_maps.n_classes нечисловой "
            f"({recorded_counts!r}) — fail closed"
        ) from exc
    expected_sha = _sha256_json(maps)
    if recorded_sha != expected_sha or counts != expected_counts:
        raise S06ReportError(
            f"{manifest_path}: label maps сплита не совпадают с baseline Gate 1 "
            f"({baseline_path}: l3_universe_sha256={recorded_sha!r} ожидается "
            f"{expected_sha!r}; n_classes={counts} ожидается {expected_counts}) — "
            "подмена label maps недопустима"
        )


def _recorded_predictions_sha256(result: dict) -> str | None:
    hashes = result.get("hashes") or {}
    eval_info = result.get("comparison_val_eval") or {}
    return hashes.get(PREDICTIONS_HASH_KEY) or eval_info.get(PREDICTIONS_HASH_KEY)


def _verify_predictions_file(
    output_root: str,
    plan_entry: dict,
    result: dict,
    val_binding: dict[str, Any],
) -> pd.DataFrame:
    """Fail-closed проверка файла предсказаний успешного run; возвращает кадр.

    Проверяются: наличие каноничного файла, проверяемый fingerprint в
    результате run (``hashes.predictions_sha256`` либо
    ``comparison_val_eval.predictions_sha256``) и его совпадение с фактическим
    sha256, схема колонок, что все строки относятся к этому run_id, и привязка
    к validation split: множество row_id и метки true_L3 совпадают с
    comparison_val, ``correct`` согласован с true/predicted, ``predicted_L3``
    лежит в L3-вселенной label maps либо null.
    """
    run_id = plan_entry["run_id"]
    path = os.path.join(output_root, "runs", "predictions", f"{run_id}.parquet")
    if not os.path.isfile(path):
        raise S06ReportError(
            f"run {run_id}: статус success без файла предсказаний {path} — "
            "success по одному полю статуса не принимается"
        )
    recorded = _recorded_predictions_sha256(result)
    if not recorded:
        raise S06ReportError(
            f"run {run_id}: в результате нет проверяемого fingerprint предсказаний "
            f"(hashes.{PREDICTIONS_HASH_KEY} или comparison_val_eval.{PREDICTIONS_HASH_KEY}); "
            "success не подтверждён. Требуется доработка kkt_clf/s06_comparison.py: "
            "записывать sha256 файла runs/predictions/<run_id>.parquet в run_result "
            f"(hashes.{PREDICTIONS_HASH_KEY}) при сохранении предсказаний."
        )
    actual = _fingerprint_file(path)
    if actual["sha256"] != recorded:
        raise S06ReportError(
            f"run {run_id}: sha256 предсказаний {path} не совпадает с записанным в "
            "результате run — файл подменён или повреждён, success не подтверждён"
        )
    frame = pd.read_parquet(path)
    required = {"row_id", "run_id", "true_L3", "predicted_L3", "correct"}
    missing = required - set(frame.columns)
    if missing:
        raise S06ReportError(
            f"run {run_id}: предсказания без обязательных колонок {sorted(missing)}"
        )
    if len(frame) and not (frame["run_id"] == run_id).all():
        raise S06ReportError(f"run {run_id}: предсказания содержат строки чужого run_id")
    val_rows = val_binding["rows"]
    if len(frame) != len(val_rows) or set(frame["row_id"]) != set(val_rows["row_id"]):
        raise S06ReportError(
            f"run {run_id}: множество row_id предсказаний не совпадает с comparison_val "
            f"(sha256 {val_binding['sha256']}, n_rows={val_binding['n_rows']})"
        )
    joined = frame.merge(val_rows, on="row_id", suffixes=("", "_val"))
    if len(joined) != len(frame) or not (joined["true_L3"] == joined["true_L3_val"]).all():
        raise S06ReportError(
            f"run {run_id}: метки true_L3 предсказаний не совпадают с comparison_val — "
            "привязка к validation split не подтверждена"
        )
    expected_correct = (frame["true_L3"] == frame["predicted_L3"]).fillna(False)
    if not (frame["correct"].fillna(False).astype(bool) == expected_correct).all():
        raise S06ReportError(f"run {run_id}: колонка correct не согласована с true/predicted")
    unknown = set(frame["predicted_L3"].dropna()) - val_binding["l3_universe"]
    if unknown:
        raise S06ReportError(
            f"run {run_id}: predicted_L3 вне L3-вселенной label maps: {sorted(unknown)[:5]}"
        )
    return frame


def _load_predictions(
    output_root: str,
    run_ids: list[str],
    *,
    plan: dict[str, dict] | None = None,
    results: dict[str, dict] | None = None,
    val_binding: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Загружает предсказания перечисленных runs.

    Без контекста (``plan``/``results``/``val_binding`` не переданы) — простое
    чтение существующих файлов (низкоуровневый режим для тестов). С полным
    контекстом каждый файл предварительно проходит fail-closed проверку
    целостности и привязки к плану, результату run и validation split
    (:func:`_verify_run_result_binding`, :func:`_verify_predictions_file`);
    частичный контекст — ошибка.
    """
    verify = bool(run_ids) and (
        plan is not None or results is not None or val_binding is not None
    )
    if verify and (plan is None or results is None or val_binding is None):
        raise S06ReportError(
            "Проверка предсказаний требует одновременно plan, results и val_binding"
        )
    frames = []
    for run_id in run_ids:
        if verify:
            if run_id not in plan:
                raise S06ReportError(f"run {run_id}: отсутствует в плане runs/planned.json")
            result = (results or {}).get(run_id)
            if not isinstance(result, dict) or result.get("status") != "success":
                raise S06ReportError(
                    f"run {run_id}: нет результата со статусом success — "
                    "предсказания непроверенного run в отчёт не включаются"
                )
            frames.append(_verify_predictions_file(output_root, plan[run_id], result, val_binding))
            continue
        path = os.path.join(output_root, "runs", "predictions", f"{run_id}.parquet")
        if os.path.isfile(path):
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame({
            "row_id": pd.Series(dtype="int64"), "run_id": pd.Series(dtype="str"),
            "true_L3": pd.Series(dtype="str"), "predicted_L3": pd.Series(dtype="str"),
            "correct": pd.Series(dtype="bool"),
        })
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Таблицы отчёта
# ---------------------------------------------------------------------------


def _runs_frame(plan: dict, results: dict[str, dict]) -> pd.DataFrame:
    """runs.csv: статусы всех плановых runs, метрики и paired Δ к none того же seed."""
    rows = []
    by_slug_seed = {}
    for run_id, res in results.items():
        if res.get("status") == "success" and res.get("comparison_val_eval"):
            by_slug_seed[(res["config_slug"], res["seed"])] = res["comparison_val_eval"]
    for r in plan["runs"]:
        run_id = r["run_id"]
        res = results.get(run_id)
        status = res["status"] if res else "pending"
        reason = (res or {}).get("reason")
        ev = (res or {}).get("comparison_val_eval") or {}
        f1, acc = ev.get("l3_macro_f1"), ev.get("l3_accuracy")
        delta_f1 = delta_acc = None
        if r["method"] != "none" and res and res.get("status") == "success":
            base = by_slug_seed.get(("none", r["seed"]))
            if base:
                delta_f1 = None if f1 is None or base.get("l3_macro_f1") is None else f1 - base["l3_macro_f1"]
                delta_acc = None if acc is None or base.get("l3_accuracy") is None else acc - base["l3_accuracy"]
        hashes = (res or {}).get("hashes") or {}
        rows.append({
            "run_id": run_id, "config_slug": r["config_slug"], "method": r["method"],
            "budget": r["budget"], "seed": r["seed"], "status": status,
            "reason": reason,
            "l3_macro_f1": f1, "l3_accuracy": acc,
            "delta_macro_f1_vs_none_same_seed": delta_f1,
            "delta_accuracy_vs_none_same_seed": delta_acc,
            "n_features": (res or {}).get("n_features"),
            "n_empty_features_val": ev.get("n_empty_features_val"),
            "model_pt_sha256": hashes.get("model_pt_sha256"),
            "tfidf_features_sha256": hashes.get("tfidf_features_sha256"),
            "checkpoint_dir": ((res or {}).get("artifacts") or {}).get("checkpoint_dir"),
            "selection_artifact": ((res or {}).get("artifacts") or {}).get("selection_artifact"),
            "tfidf_artifact": ((res or {}).get("artifacts") or {}).get("tfidf_artifact"),
        })
    return pd.DataFrame(rows)


def _per_class_frame(results: dict[str, dict], runs_df: pd.DataFrame) -> pd.DataFrame:
    """per_class.parquet: per-L3 метрики каждого успешного run и ΔF1 к none того же seed."""
    base_f1: dict[tuple[Any, str], float] = {}
    for run_id, res in results.items():
        if res.get("config_slug") == "none" and res.get("status") == "success":
            for row in (res.get("comparison_val_eval") or {}).get("per_class") or []:
                if row.get("f1") is not None:
                    base_f1[(res["seed"], row["L3"])] = float(row["f1"])
    frames = []
    for run_id, res in results.items():
        if res.get("status") != "success":
            continue
        per_class = (res.get("comparison_val_eval") or {}).get("per_class") or []
        if not per_class:
            continue
        frame = pd.DataFrame(per_class)
        frame["run_id"] = run_id
        frame["seed"] = res["seed"]
        frame["method"] = res["method"]
        frame["budget"] = res.get("budget")
        frame["delta_f1"] = frame.apply(
            lambda row: (None if row["f1"] is None
                         else float(row["f1"]) - base_f1.get((res["seed"], row["L3"]), float("nan"))),
            axis=1,
        )
        frames.append(frame)
    columns = ["run_id", "seed", "method", "budget", "L3", "support", "precision", "recall", "f1", "delta_f1"]
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
            columns, ["str", "int64", "str", "Int64", "str", "int64", "float64", "float64", "float64", "float64"])})
    out = pd.concat(frames, ignore_index=True)
    return out[columns]


def _support_bins_frame(per_class: pd.DataFrame, runs_df: pd.DataFrame) -> pd.DataFrame:
    """support_bins.csv: macro-F1 по диапазонам support (comparison_val), paired к none.

    Метрики бина — невзвешенное среднее per-L3 F1 внутри бина; базовое значение —
    бин none того же seed; пустые бины — null при n_L3=0.
    """
    rows = []
    if per_class.empty:
        return pd.DataFrame(columns=[
            "run_id", "method", "budget", "seed", "support_bin", "n_L3",
            "macro_f1_bin", "baseline_macro_f1_bin_none_same_seed", "delta_macro_f1_bin",
        ])
    pc = per_class.dropna(subset=["f1"])
    for (run_id, method, budget, seed), group in pc.groupby(
        ["run_id", "method", "budget", "seed"], dropna=False
    ):
        base = pc[(pc["run_id"] == f"none_seed{seed}")]
        for label, lo, hi in SUPPORT_BINS:
            sel = group[(group["support"] >= lo) & (group["support"] <= hi)] if hi is not None else group[group["support"] >= lo]
            bsel = base[(base["support"] >= lo) & (base["support"] <= hi)] if hi is not None else base[base["support"] >= lo]
            macro = float(sel["f1"].mean()) if len(sel) else None
            base_macro = float(bsel["f1"].mean()) if len(bsel) else None
            rows.append({
                "run_id": run_id, "method": method, "budget": budget, "seed": seed,
                "support_bin": label, "n_L3": int(len(sel)),
                "macro_f1_bin": macro,
                "baseline_macro_f1_bin_none_same_seed": base_macro,
                "delta_macro_f1_bin": (None if macro is None or base_macro is None else macro - base_macro),
            })
    return pd.DataFrame(rows)


def _confusion_frame(predictions: pd.DataFrame, per_class: pd.DataFrame) -> pd.DataFrame:
    """confusion_pairs.csv: top путаниц true→predicted по всем runs с предсказаниями.

    ``predicted_L3=null`` считается ошибкой и сохраняется как пустое (NaN)
    значение предсказания в паре; при полностью верных предсказаниях
    возвращается пустая таблица с полной схемой (perfect predictions — не
    ошибка сборки).
    """
    columns = [
        "run_id", "method", "budget", "seed", "true_L3", "predicted_L3",
        "count", "true_support", "error_share", "rank_in_run",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    correct = predictions["correct"].fillna(False).astype(bool)
    wrong = predictions[~correct | predictions["predicted_L3"].isna()]
    if wrong.empty:
        return pd.DataFrame(columns=columns)
    meta = per_class[["run_id", "method", "budget", "seed", "L3", "support"]].drop_duplicates()
    support_map = {(r.run_id, r.L3): int(r.support) for r in meta.itertuples()}
    rows = []
    for (run_id, true_l3, pred_l3), group in wrong.groupby(
        ["run_id", "true_L3", "predicted_L3"], dropna=False
    ):
        rows.append({
            "run_id": run_id, "true_L3": true_l3, "predicted_L3": pred_l3,
            "count": int(len(group)),
            "true_support": support_map.get((run_id, true_l3)),
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    meta_small = meta[["run_id", "method", "budget", "seed"]].drop_duplicates()
    frame = frame.merge(meta_small, on="run_id", how="left")
    frame["error_share"] = frame.apply(
        lambda r: (None if not r["true_support"] else r["count"] / r["true_support"]), axis=1
    )
    frame = frame.sort_values(["run_id", "count"], ascending=[True, False])
    frame["rank_in_run"] = frame.groupby("run_id")["count"].rank(method="first", ascending=False).astype(int)
    return frame[columns]


def _stages_frame(output_root: str) -> pd.DataFrame:
    """stages.csv: этапы comparison-контура + Gate 1 (source-метка, без смешения)."""
    frames = []
    ours = os.path.join(output_root, "runs", "stages.csv")
    if os.path.isfile(ours):
        df = pd.read_csv(ours)
        df["source"] = "comparison"
        frames.append(df)
    gate1 = os.path.join(output_root, "runs", "baseline_full", "stages.csv")
    if os.path.isfile(gate1):
        df = pd.read_csv(gate1)
        df["run_id"] = "gate1:" + df["run_id"].astype(str)
        df["source"] = "gate1_baseline"
        frames.append(df)
    columns = ["run_id", "stage", "status", "seconds", "peak_rss_bytes", "peak_vram_bytes",
               "artefact_bytes", "source", "measurement_note"]
    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True)
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out[columns]


_MODEL_PT_SIZE_NOTE = (
    "artefact_bytes = байтовый размер сериализованного model.pt из проверенного "
    "artifacts.checkpoint_dir результата run (sha256 сверен с hashes.model_pt_sha256; "
    "размер каталога чекпоинта не используется)"
)


def _populate_model_pt_sizes(
    stages: pd.DataFrame,
    results: dict[str, dict],
    success_run_ids: list[str],
) -> pd.DataFrame:
    """Заполняет ``artefact_bytes`` строк ``mlp_train`` успешных runs размером model.pt.

    Comparison-контур для обученных runs размер сериализованной модели не
    записывал (``artefact_bytes=null``), а для переиспользованного baseline
    Gate 1 фиксировал размер каталога чекпоинта. Отчёт требует сопоставимой
    метрики — байтового размера ``model.pt``: для каждого успешного run,
    прошедшего fail-closed привязку (``success_run_ids``), размер берётся из
    ``artifacts.checkpoint_dir/model.pt`` его результата и сверяется
    с ``hashes.model_pt_sha256``. Отсутствие model.pt или записанного хэша,
    несовпадение sha256 — отказ сборки (fail closed, подмена чекпоинта
    недопустима). Строки failed/pending runs, gate1-строки без результата run
    и не-``mlp_train`` этапы не изменяются; измерения для неуспешных runs
    не выдумываются. К заполненным строкам в ``measurement_note`` добавляется
    пометка о происхождении числа.
    """
    if stages.empty or not success_run_ids:
        return stages
    sizes: dict[str, int] = {}
    for run_id in success_run_ids:
        result = results[run_id]
        checkpoint_dir = ((result.get("artifacts") or {}).get("checkpoint_dir"))
        recorded_sha = ((result.get("hashes") or {}).get("model_pt_sha256"))
        if not checkpoint_dir or not recorded_sha:
            raise S06ReportError(
                f"run {run_id}: нет provenance для размера модели "
                "(artifacts.checkpoint_dir / hashes.model_pt_sha256) — fail closed"
            )
        fp = _fingerprint_file(os.path.join(checkpoint_dir, "model.pt"))
        if fp is None:
            raise S06ReportError(
                f"run {run_id}: model.pt не найден в проверенном checkpoint_dir "
                f"{checkpoint_dir!r} — размер модели не подтверждён (fail closed)"
            )
        if fp["sha256"] != recorded_sha:
            raise S06ReportError(
                f"run {run_id}: sha256 model.pt ({fp['sha256']}) не совпадает с "
                f"hashes.model_pt_sha256 результата run ({recorded_sha}) — чекпоинт "
                "подменён или повреждён, размер в отчёт не включается (fail closed)"
            )
        sizes[run_id] = int(fp["bytes"])
    mask = (stages["stage"] == "mlp_train") & stages["run_id"].isin(sizes.keys())
    if not mask.any():
        return stages
    stages = stages.copy()
    stages.loc[mask, "artefact_bytes"] = stages.loc[mask, "run_id"].map(sizes)
    for idx in stages.index[mask]:
        existing = stages.at[idx, "measurement_note"]
        note = _MODEL_PT_SIZE_NOTE
        if isinstance(existing, str) and existing.strip():
            note = f"{existing} | {_MODEL_PT_SIZE_NOTE}"
        stages.at[idx, "measurement_note"] = note
    return stages


def _method_summary(runs_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Сводка method×K (включая baseline none) по успешным seed: mean/std/min/max и средняя парная Δ."""
    rows = []
    for (method, budget), group in runs_df.groupby(["method", "budget"], dropna=False):
        ok = group[group["status"] == "success"]
        f1s = ok["l3_macro_f1"].dropna().astype(float).tolist()
        accs = ok["l3_accuracy"].dropna().astype(float).tolist()
        deltas = ok["delta_macro_f1_vs_none_same_seed"].dropna().astype(float).tolist()

        def stats(values: list[float]) -> dict[str, Any]:
            if not values:
                return {"mean": None, "std": None, "min": None, "max": None, "range": None, "n": 0}
            arr = np.asarray(values, dtype=np.float64)
            return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()),
                    "max": float(arr.max()), "range": float(arr.max() - arr.min()), "n": int(len(arr))}

        rows.append({
            "method": method, "budget": int(budget) if pd.notna(budget) else None,
            "n_seed_success": int(len(ok)),
            "n_seed_planned": int(len(group)),
            "l3_macro_f1": stats(f1s),
            "l3_accuracy": stats(accs),
            "paired_delta_macro_f1": {
                **stats(deltas),
                "mean_pairwise": (float(np.mean(deltas)) if deltas else None),
            },
        })
    return rows


# ---------------------------------------------------------------------------
# manifest.json
# ---------------------------------------------------------------------------


def _fingerprint_file(path: str) -> dict[str, Any] | None:
    import hashlib

    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"path": os.path.abspath(path), "sha256": h.hexdigest(),
            "bytes": os.path.getsize(path)}


def build_manifest(
    output_root: str,
    plan: dict,
    runs_df: pd.DataFrame,
    method_summary: list[dict[str, Any]],
    report_files: dict[str, str],
) -> dict[str, Any]:
    """Машинный manifest отчёта: версии/хэши входов, условия, статусы, файлы."""
    shared = os.path.join(output_root, "shared")
    runs_root = os.path.join(output_root, "runs")
    protocol = _read_json(os.path.join(shared, "training_protocol.json"))
    # Мягкое разрешение: путь из плана (или наследованный), без отказа, —
    # fail-closed сверка выполняется в _load_comparison_val_binding перед
    # включением предсказаний; здесь фиксируется фактический fingerprint.
    declared_manifest = (plan.get("config") or {}).get("splits_manifest")
    manifest_path = declared_manifest or os.path.join(output_root, "splits", "manifest.json")
    splits_manifest = _read_json(manifest_path, required=False)
    rankings_index = _read_json(os.path.join(shared, "rankings_index.json"), required=False)
    refit_results = {}
    for slug in sorted({r["config_slug"] for r in plan["runs"]} - {"none"}):
        refit_results[slug] = _read_json(os.path.join(runs_root, slug, "refit_result.json"), required=False)
    selected_checkpoint = _read_json(os.path.join(runs_root, "baseline_full", "selected_checkpoint.json"), required=False)
    gate1_manifest = _read_json(os.path.join(runs_root, "baseline_full", "manifest.json"), required=False)
    audit = _read_json(os.path.join(output_root, "audit", "data_audit.json"), required=False)
    git_commit = git_dirty = None
    import subprocess

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip() or None
        git_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
        ).stdout.strip())
    except Exception:
        pass
    statuses = {
        "n_planned": int(plan["n_planned"]),
        "n_success": int((runs_df["status"] == "success").sum()),
        "n_failed": int((runs_df["status"] == "failed").sum()),
        "n_pending": int((runs_df["status"] == "pending").sum()),
        "by_run": runs_df[["run_id", "status", "reason"]].to_dict("records"),
    }
    return {
        "schema_version": S06_REPORT_SCHEMA,
        "role": "s06_comparison_report",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "rebuild": {
            "command": f"python scripts/run_s06_report.py --output {output_root}",
            "principle": (
                "пересборка только из сохранённых JSON/CSV/Parquet без обучения, "
                "fit TF-IDF и повторного inference; report/*.parquet/csv являются "
                "производными от runs/** и shared/**; report.html дополнительно "
                "пересоздаётся автономно из сохранённых таблиц report/ без "
                "обращения к runs/**, shared/**, models/, training/ и inference"
            ),
            "report_files": report_files,
        },
        "inputs": {
            "splits_manifest": _fingerprint_file(manifest_path) if os.path.isfile(manifest_path) else None,
            "splits_manifest_source": (
                "путь из неизменяемого plan.config.splits_manifest (замороженные S06-сплиты); "
                "manifest читается только оттуда, без подмены найденным в output_root"
            ),
            "splits_parts": {
                name: {"sha256": part.get("sha256"), "n_rows": part.get("n_rows")}
                for name, part in ((splits_manifest or {}).get("parts") or {}).items()
            },
            "labeled_parquet": (splits_manifest or {}).get("inputs", {}).get("labeled_parquet"),
            "full_tfidf_ref": _fingerprint_file(os.path.join(shared, "full_tfidf_ref.json")),
            "full_tfidf": {
                "path": plan["config"].get("full_tfidf_path"),
                "protocol_full_tfidf_id": (protocol or {}).get("full_tfidf_id"),
                "vocab": (protocol or {}).get("vocab") if protocol else None,
            },
            "corpus": (protocol or {}).get("corpus_fingerprint") if protocol else None,
            "training_protocol": _fingerprint_file(os.path.join(shared, "training_protocol.json")),
            "fixed_epochs": (protocol or {}).get("fixed_epochs") if protocol else None,
            "rankings_index": rankings_index,
            "refit_results": refit_results,
            "gate1_baseline_manifest": _fingerprint_file(os.path.join(runs_root, "baseline_full", "manifest.json")),
            "gate1_selected_checkpoint": selected_checkpoint,
            "data_audit": _fingerprint_file(os.path.join(output_root, "audit", "data_audit.json")),
            "audit_summary": ({
                "rows_before": audit.get("rows_before"), "rows_after": audit.get("rows_after"),
                "l3_before": audit.get("l3_before"), "l3_after": audit.get("l3_after"),
            } if audit else None),
        },
        "protocol": {
            "seeds": plan["config"]["seeds"],
            "reference_seed": plan["config"]["reference_seed"],
            "methods": plan["config"]["methods"],
            "budgets": plan["config"]["budgets"],
            "char_share": plan["config"]["char_share"],
            "selection_seed": 42,
            "evaluation": "raw top-1 L3 без отказа на comparison_val; фиксированная L3-вселенная label maps",
            "sealed_test_boundary": (
                "sealed_test не использовался ни для inference, ни для метрик; только "
                "структурные sha256/n_rows из manifest"
            ),
        },
        "method_summary": method_summary,
        "statuses": statuses,
        "formulas": FORMULAS,
        "limitations": LIMITATIONS,
        "environment": (gate1_manifest or {}).get("environment") if gate1_manifest else None,
        "git": {"commit": git_commit, "dirty": git_dirty},
        "notes": [
            "Плановые runs без result-файла имеют статус pending (запуск не выполнялся); "
            "failed и reused не скрываются.",
            "Shared-затраты (ранжирование, атрибуция, refit, векторизация) учтены "
            "однократно под run_id shared_* и не суммируются повторно по seed.",
            "Predictions включены только для runs, прошедших fail-closed проверку: "
            "привязка результата к плану/provenance, сверка sha256 предсказаний с "
            "fingerprint из run_result (hashes.predictions_sha256) и привязка "
            "row_id/true_L3 к comparison_val по sha256 из splits/manifest.json.",
        ],
    }


# ---------------------------------------------------------------------------
# HTML (автономный, inline SVG)
# ---------------------------------------------------------------------------


def _esc(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "null"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _table_html(df: pd.DataFrame, columns: list[str], max_rows: int | None = None,
                formats: dict[str, Any] | None = None) -> str:
    formats = formats or {}
    view = df[columns].copy()
    if max_rows is not None:
        view = view.head(max_rows)
    head = "".join(f"<th>{_esc(c)}</th>" for c in columns)
    rows = []
    for _, row in view.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            if formats.get(c):
                v = formats[c](v)
            else:
                v = _fmt(v) if isinstance(v, (float, np.floating)) else ("null" if pd.isna(v) else _esc(v))
            cells.append(f"<td>{v}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def _svg_line_chart(series: dict[str, list[tuple[float, float]]], width: int = 720, height: int = 320,
                    x_label: str = "K (суммарный бюджет признаков)", y_label: str = "macro-F1 (comparison_val)",
                    baseline: tuple[float, float] | None = None) -> str:
    """Простой автономный SVG-график: линии методов по бюджетам K."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 16, 40
    all_x = [x for pts in series.values() for x, _ in pts]
    all_y = [y for pts in series.values() for _, y in pts]
    if baseline:
        all_y.extend([baseline[0], baseline[1]])
    if not all_x:
        return "<p class='muted'>нет данных для графика</p>"
    x_min, x_max = min(all_x), max(all_x)
    if x_min == x_max:
        x_min, x_max = x_min * 0.5, x_max * 1.5 + 1
    y_min, y_max = min(all_y), max(all_y)
    margin = (y_max - y_min) * 0.08 + 1e-9
    y_min, y_max = y_min - margin, y_max + margin

    def sx(x: float) -> float:
        return pad_l + (x - x_min) / (x_max - x_min) * (width - pad_l - pad_r)

    def sy(y: float) -> float:
        return height - pad_b - (y - y_min) / (y_max - y_min) * (height - pad_t - pad_b)

    parts = [f"<svg viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg' class='chart'>"]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_min + frac * (y_max - y_min)
        parts.append(f"<line x1='{pad_l}' y1='{sy(y):.1f}' x2='{width - pad_r}' y2='{sy(y):.1f}' class='grid'/>")
        parts.append(f"<text x='{pad_l - 6}' y='{sy(y) + 4:.1f}' class='tick' text-anchor='end'>{y:.3f}</text>")
    for x in sorted(set(all_x)):
        parts.append(f"<text x='{sx(x):.1f}' y='{height - pad_b + 16}' class='tick' text-anchor='middle'>{int(x)}</text>")
    parts.append(f"<text x='{(pad_l + width - pad_r) / 2}' y='{height - 6}' class='label' text-anchor='middle'>{_esc(x_label)}</text>")
    parts.append(f"<text x='14' y='{(pad_t + height - pad_b) / 2}' class='label' text-anchor='middle' transform='rotate(-90 14 {(pad_t + height - pad_b) / 2})'>{_esc(y_label)}</text>")
    if baseline:
        parts.append(
            f"<rect x='{pad_l}' y='{sy(baseline[1]):.1f}' width='{width - pad_l - pad_r}' "
            f"height='{max(0.0, sy(baseline[0]) - sy(baseline[1])):.1f}' class='baseline-band'/>"
        )
        parts.append(f"<line x1='{pad_l}' y1='{sy(sum(baseline) / 2):.1f}' x2='{width - pad_r}' "
                     f"y2='{sy(sum(baseline) / 2):.1f}' class='baseline-line'/>")
        parts.append(f"<text x='{width - pad_r - 4}' y='{sy(baseline[1]) - 5:.1f}' class='tick' text-anchor='end'>baseline none (min–max по seed)</text>")
    for i, (name, pts) in enumerate(sorted(series.items())):
        color = _PALETTE[i % len(_PALETTE)]
        pts_sorted = sorted(pts)
        path = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for j, (x, y) in enumerate(pts_sorted))
        parts.append(f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2'/>")
        for x, y in pts_sorted:
            parts.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3.5' fill='{color}'/>")
        last_x, last_y = pts_sorted[-1]
        parts.append(f"<text x='{sx(last_x):.1f}' y='{sy(last_y) - 8:.1f}' class='legend' text-anchor='middle' fill='{color}'>{_esc(name)}</text>")
    parts.append("</svg>")
    return "".join(parts)


_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1b1b1b;max-width:1180px}
h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}
h3{font-size:14px;margin-top:18px}
table{border-collapse:collapse;font-size:12px;margin:8px 0;width:100%}
th,td{border:1px solid #ccc;padding:3px 6px;text-align:left;vertical-align:top}
th{background:#f3f3f3}
.muted{color:#666;font-size:12px}
.status-success{color:#116611;font-weight:600}.status-failed{color:#a00000;font-weight:600}
.status-pending{color:#8a6d00;font-weight:600}
.chart{background:#fff;border:1px solid #ddd;margin:8px 0;max-width:100%}
.grid{stroke:#e4e4e4;stroke-width:1}.tick{font-size:10px;fill:#555}.label{font-size:11px;fill:#333}
.baseline-band{fill:#1f77b4;opacity:.10}.baseline-line{stroke:#1f77b4;stroke-dasharray:5 4;opacity:.6}
.legend{font-size:11px}
code{background:#f3f3f3;padding:1px 4px}
.box{border:1px solid #ddd;padding:10px 14px;margin:10px 0;background:#fbfbfb}
"""


def _html_report(
    output_root: str,
    plan: dict,
    runs_df: pd.DataFrame,
    method_summary: list[dict[str, Any]],
    support_bins: pd.DataFrame,
    confusion: pd.DataFrame,
    stages: pd.DataFrame,
    predictions: pd.DataFrame,
) -> str:
    ok = runs_df[runs_df["status"] == "success"]
    none_f1 = ok.loc[ok["method"] == "none", "l3_macro_f1"].dropna().astype(float)
    baseline_band = (float(none_f1.min()), float(none_f1.max())) if len(none_f1) else None

    series: dict[str, list[tuple[float, float]]] = {}
    for row in method_summary:
        method, budget, stats = row["method"], row["budget"], row["l3_macro_f1"]
        if budget is None or stats["mean"] is None:
            continue
        series.setdefault(method, []).append((float(budget), stats["mean"]))
    chart = _svg_line_chart(series, baseline=baseline_band)

    summary_rows = []
    for row in method_summary:
        summary_rows.append({
            "method": row["method"], "K": row["budget"],
            "seed ok": f"{row['n_seed_success']}/{row['n_seed_planned']}",
            "macro-F1 mean": row["l3_macro_f1"]["mean"],
            "std": row["l3_macro_f1"]["std"],
            "min": row["l3_macro_f1"]["min"],
            "max": row["l3_macro_f1"]["max"],
            "Δ mean": row["paired_delta_macro_f1"]["mean_pairwise"],
            "Δ range": (None if row["paired_delta_macro_f1"]["range"] is None
                        else f"±{row['paired_delta_macro_f1']['range'] / 2:.4f}"),
        })
    summary_df = pd.DataFrame(summary_rows)

    status_counts = runs_df["status"].value_counts().to_dict()
    runs_view = runs_df.copy()
    runs_view["macro-F1"] = runs_view["l3_macro_f1"]
    runs_view["acc"] = runs_view["l3_accuracy"]
    runs_view["ΔF1"] = runs_view["delta_macro_f1_vs_none_same_seed"]
    runs_view["Δacc"] = runs_view["delta_accuracy_vs_none_same_seed"]

    bins_view = support_bins.copy() if not support_bins.empty else pd.DataFrame(
        columns=["run_id", "method", "seed", "support_bin", "n_L3", "macro_f1_bin",
                 "baseline_macro_f1_bin_none_same_seed", "delta_macro_f1_bin"])
    conf_view = confusion.sort_values(["count"], ascending=False) if not confusion.empty else confusion

    # Ресурсы: shared-этапы отдельно от per-run; без двойного учёта.
    shared_stages = stages[stages["run_id"].astype(str).str.startswith(("shared", "gate1:shared"))] if not stages.empty else stages
    run_stages = stages[~stages["run_id"].astype(str).str.startswith(("shared", "gate1:shared"))] if not stages.empty else stages
    res_agg = None
    if not run_stages.empty:
        mlp = run_stages[run_stages["stage"] == "mlp_train"]
        infer = run_stages[run_stages["stage"] == "comparison_val_inference"]
        res_agg = pd.DataFrame([
            {"этап": "mlp_train (per run)", "n": int(len(mlp)),
             "seconds sum": mlp["seconds"].dropna().astype(float).sum() if len(mlp) else None,
             "seconds mean": mlp["seconds"].dropna().astype(float).mean() if len(mlp) else None,
             "peak RSS max": mlp["peak_rss_bytes"].dropna().max() if len(mlp) and mlp["peak_rss_bytes"].notna().any() else None},
            {"этап": "comparison_val_inference (per run)", "n": int(len(infer)),
             "seconds sum": infer["seconds"].dropna().astype(float).sum() if len(infer) else None,
             "seconds mean": infer["seconds"].dropna().astype(float).mean() if len(infer) else None,
             "peak RSS max": infer["peak_rss_bytes"].dropna().max() if len(infer) and infer["peak_rss_bytes"].notna().any() else None},
        ])

    status_cells = " ".join(
        f"<span class='status-{_esc(k)}'>{_esc(k)}: {v}</span>" for k, v in sorted(status_counts.items())
    )
    formulas_html = "".join(f"<li><b>{_esc(k)}:</b> {_esc(v)}</li>" for k, v in FORMULAS.items())
    limits_html = "".join(f"<li>{_esc(x)}</li>" for x in LIMITATIONS)

    parts = [
        "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>",
        f"<title>S06 — сравнение supervised feature selection</title><style>{_CSS}</style></head><body>",
        "<h1>S06 — сравнение supervised feature selection (Global Chi² / Class-aware Chi² / Gradient × Input)</h1>",
        f"<p class='muted'>Собрано: {_esc(dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'))} · "
        f"корень: <code>{_esc(os.path.abspath(output_root))}</code> · схема {S06_REPORT_SCHEMA}</p>",
        "<h2>0. Статус прогонов</h2>",
        f"<p>Всего плановых runs: {int(plan['n_planned'])}. {status_cells}</p>",
        _table_html(runs_view, ["run_id", "method", "budget", "seed", "status", "reason",
                                "macro-F1", "acc", "ΔF1", "Δacc", "n_features"],
                    formats={"reason": lambda v: _esc(v) if pd.notna(v) else "—",
                             "budget": lambda v: str(int(v)) if pd.notna(v) else "—"}),
        "<h2>1. Главное сравнение: качество L3 vs бюджет K</h2>",
        "<p class='muted'>Точки — средний macro-F1 по успешным seed; голубая лента — диапазон min–max "
        "полного baseline (none) по seed. paired Δ — к none того же seed.</p>",
        chart,
        _table_html(summary_df, list(summary_df.columns)) if not summary_df.empty else "<p class='muted'>нет данных</p>",
        "<h2>2. Метрики по диапазонам support L3 (comparison_val)</h2>",
        "<p class='muted'>macro-F1 бина — невзвешенное среднее per-L3 F1; baseline — none того же seed; "
        "пустые бины — null с n_L3=0. Категории с support 2 нестабильны (шаг recall 0.5).</p>",
        _table_html(bins_view, ["run_id", "support_bin", "n_L3", "macro_f1_bin",
                                "baseline_macro_f1_bin_none_same_seed", "delta_macro_f1_bin"])
        if not bins_view.empty else "<p class='muted'>нет данных (нет успешных runs)</p>",
        "<h2>3. Top confused pairs (true L3 → predicted L3)</h2>",
        "<p class='muted'>Все пары — в report/confusion_pairs.csv с rank_in_run; support истинной L3 и "
        "доля ошибок на comparison_val. Ниже — топ-30 по count среди всех runs.</p>",
        _table_html(conf_view, ["run_id", "true_L3", "predicted_L3", "count", "true_support", "error_share",
                                "rank_in_run"], max_rows=30)
        if not conf_view.empty else "<p class='muted'>предсказаний нет — confused pairs не построены</p>",
        "<h2>4. Ресурсы</h2>",
        "<p class='muted'>Shared-затраты (ранжирование, атрибуция, refit, векторизация) — однократно, "
        "не суммируются по seed. GPU недоступен → peak_vram=null (не ноль).</p>",
        "<h3>Shared-этапы</h3>",
        _table_html(shared_stages, ["run_id", "stage", "status", "seconds", "peak_rss_bytes",
                                    "peak_vram_bytes", "artefact_bytes", "source"])
        if not shared_stages.empty else "<p class='muted'>нет shared-этапов</p>",
        "<h3>Сводка per-run этапов</h3>",
        _table_html(res_agg, list(res_agg.columns)) if res_agg is not None else "<p class='muted'>нет per-run этапов</p>",
        "<h3>Все этапы</h3>",
        _table_html(stages, ["run_id", "stage", "status", "seconds", "peak_rss_bytes",
                             "peak_vram_bytes", "artefact_bytes", "source"], max_rows=400),
        "<h2>5. Формулы и протокол</h2>",
        f"<ul>{formulas_html}</ul>",
        "<div class='box'><b>Границы протокола:</b> рейтинги — только outer_train; refit — один на "
        "метод×K на том же снимке корпуса; обучение — замороженный протокол Gate 1 (train-only выбор "
        "эпох, restore_best=False); оценка — raw top-1 L3 на comparison_val; sealed_test не использовался.</div>",
        "<h2>6. Ограничения</h2>", f"<ul>{limits_html}</ul>",
        "<h2>7. Пересборка отчёта</h2>",
        "<div class='box'><code>python scripts/run_s06_report.py --output " + _esc(output_root) + "</code>"
        "<br>Полная сборка читает сохранённые машинные результаты: "
        "runs/&lt;slug&gt;/run_result_seed_*.json, runs/predictions/*.parquet, runs/stages.csv, "
        "shared/*, runs/baseline_full/*, audit/* — без обучения, fit TF-IDF и inference; существующие "
        "report-файлы не перезаписываются (перед повторной сборкой переименуйте report/)."
        "<br>Автономная пересборка <b>report.html</b>: если runs/planned.json недоступен, HTML "
        "пересоздаётся только из сохранённых таблиц report/ (runs.csv, stages.csv, manifest.json, "
        "confusion_pairs.csv, support_bins.csv) — без обращения к runs/, shared/, models/, training/ "
        "и inference; перезаписывается только report.html.</div>",
        "</body></html>",
    ]
    return "".join(parts)


# ---------------------------------------------------------------------------
# Сборка отчёта (Gate 4)
# ---------------------------------------------------------------------------

# Таблицы, достаточные для автономной пересборки report.html (без runs/shared).
_HTML_SOURCE_TABLES = ("runs.csv", "stages.csv", "manifest.json",
                       "confusion_pairs.csv", "support_bins.csv")


def _rebuild_html_from_saved_tables(output_root: str, report_dir: str) -> dict[str, Any]:
    """Пересоздаёт report.html только из сохранённых таблиц report/.

    Используется, когда исходные runs/** и shared/** недоступны (например,
    перенесён только каталог report/): читаются сохранённые runs.csv,
    stages.csv, manifest.json, confusion_pairs.csv и support_bins.csv;
    обращений к runs/, shared/, models/, training/ и inference нет.
    Перезаписывается только report.html; при указании нового каталога
    сохранённые таблицы копируются туда без изменений (уже существующие
    файлы не перезаписываются).
    """
    source_dir = report_dir
    if not all(os.path.isfile(os.path.join(source_dir, name)) for name in _HTML_SOURCE_TABLES):
        default_dir = os.path.join(output_root, "report")
        if os.path.abspath(default_dir) != os.path.abspath(report_dir):
            source_dir = default_dir
    missing = [name for name in _HTML_SOURCE_TABLES
               if not os.path.isfile(os.path.join(source_dir, name))]
    if missing:
        raise S06ReportError(
            "Автономная пересборка невозможна: не найдены сохранённые таблицы отчёта "
            f"в {source_dir}: {', '.join(missing)}"
        )
    runs_df = pd.read_csv(os.path.join(source_dir, "runs.csv"))
    stages = pd.read_csv(os.path.join(source_dir, "stages.csv"))
    confusion = pd.read_csv(os.path.join(source_dir, "confusion_pairs.csv"))
    support_bins = pd.read_csv(os.path.join(source_dir, "support_bins.csv"))
    manifest = _read_json(os.path.join(source_dir, "manifest.json"))
    method_summary = (manifest or {}).get("method_summary") or []
    n_planned = ((manifest or {}).get("statuses") or {}).get("n_planned")
    plan = {"n_planned": int(n_planned) if n_planned is not None else int(len(runs_df))}

    os.makedirs(report_dir, exist_ok=True)
    same_dir = os.path.abspath(source_dir) == os.path.abspath(report_dir)
    written: dict[str, str] = {}
    if not same_dir:
        import shutil

        for name in REPORT_FILES:
            if name == "report.html":
                continue
            src = os.path.join(source_dir, name)
            dst = os.path.join(report_dir, name)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copyfile(src, dst)
                written[name] = os.path.abspath(dst)
    target = os.path.join(report_dir, "report.html")
    html_doc = _html_report(
        output_root, plan, runs_df, method_summary, support_bins, confusion,
        stages, pd.DataFrame(),
    )
    with open(target, "w", encoding="utf-8") as f:
        f.write(html_doc)
    written["report.html"] = os.path.abspath(target)
    return {
        "report_dir": os.path.abspath(report_dir),
        "mode": "html_rebuild_from_saved_tables",
        "source_dir": os.path.abspath(source_dir),
        "n_runs": plan["n_planned"],
        "files": written,
    }


def build_s06_report(output_root: str = "outputs/S06", report_dir: str | None = None) -> dict[str, Any]:
    """Собирает восемь обязательных файлов отчёта из сохранённых результатов.

    Без обучения, без fit TF-IDF и без повторного inference: читаются только
    JSON/CSV/Parquet, записанные контурами Gate 0–3. Существующие файлы в
    ``report_dir`` не перезаписываются (отказ до любых записей); planned runs
    без result-файла честно получают статус ``pending``. Предсказания и
    per-class метрики включаются только для runs со статусом ``success``,
    прошедших fail-closed проверку: привязка результата к плану и provenance
    (:func:`_verify_run_result_binding`), проверяемый fingerprint предсказаний
    и их привязка к validation split comparison_val
    (:func:`_verify_predictions_file`). Отсутствие проверяемого fingerprint —
    явный отказ сборки, success по одному полю статуса не принимается.
    Для строк ``mlp_train`` verified-успешных runs ``artefact_bytes``
    заполняется байтовым размером сериализованного ``model.pt`` из
    ``artifacts.checkpoint_dir`` результата run со сверкой sha256
    (:func:`_populate_model_pt_sizes`, fail closed); failed/pending runs
    не получают выдуманных измерений.

    Если ``runs/planned.json`` недоступен, выполняется автономная пересборка
    ``report.html`` из сохранённых таблиц ``report/`` (см.
    ``_rebuild_html_from_saved_tables``) без обращения к runs/** и shared/**.
    """
    report_dir = report_dir or os.path.join(output_root, "report")
    plan_path = os.path.join(output_root, "runs", "planned.json")
    if not os.path.isfile(plan_path):
        return _rebuild_html_from_saved_tables(output_root, report_dir)
    plan = _read_json(plan_path)
    results = _load_run_results(output_root, plan)
    plan_by_id = {r["run_id"]: r for r in plan["runs"]}
    # Fail closed: success признаётся только после привязки результата к плану
    # и provenance; predictions — после сверки fingerprint и validation split.
    success_run_ids = []
    for r in plan["runs"]:
        res = results.get(r["run_id"])
        if res is not None and res.get("status") == "success":
            _verify_run_result_binding(r, res)
            success_run_ids.append(r["run_id"])
    val_binding = _load_comparison_val_binding(output_root, plan) if success_run_ids else None
    runs_df = _runs_frame(plan, results)
    per_class = _per_class_frame(results, runs_df)
    support_bins = _support_bins_frame(per_class, runs_df)
    predictions = _load_predictions(
        output_root, success_run_ids,
        plan=plan_by_id, results=results, val_binding=val_binding,
    )
    confusion = _confusion_frame(predictions, per_class)
    stages = _stages_frame(output_root)
    # Размеры сериализованных model.pt для mlp_train-строк успешных runs:
    # только из проверенных checkpoint_dir результатов run, fail closed.
    stages = _populate_model_pt_sizes(stages, results, success_run_ids)
    method_summary = _method_summary(runs_df)

    os.makedirs(report_dir, exist_ok=True)
    targets = {name: os.path.join(report_dir, name) for name in REPORT_FILES}
    _check_new(list(targets.values()))

    runs_csv = targets["runs.csv"]
    runs_df.to_csv(runs_csv, index=False)
    per_class.to_parquet(targets["per_class.parquet"], index=False)
    stages.to_csv(targets["stages.csv"], index=False)
    confusion.to_csv(targets["confusion_pairs.csv"], index=False)
    support_bins.to_csv(targets["support_bins.csv"], index=False)
    predictions.to_parquet(targets["predictions_val.parquet"], index=False)

    report_files = {name: os.path.abspath(path) for name, path in targets.items()}
    manifest = build_manifest(output_root, plan, runs_df, method_summary, report_files)
    with open(targets["manifest.json"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    html_doc = _html_report(
        output_root, plan, runs_df, method_summary, support_bins, confusion, stages, predictions
    )
    with open(targets["report.html"], "w", encoding="utf-8") as f:
        f.write(html_doc)

    log_info = {
        "report_dir": os.path.abspath(report_dir),
        "n_runs": int(plan["n_planned"]),
        "statuses": manifest["statuses"]["by_run"] and {
            k: int((runs_df["status"] == k).sum()) for k in runs_df["status"].unique()
        },
        "files": report_files,
    }
    return log_info


__all__ = [
    "S06_REPORT_SCHEMA",
    "S06ReportError",
    "SUPPORT_BINS",
    "REPORT_FILES",
    "FORMULAS",
    "LIMITATIONS",
    "RUN_RESULT_SCHEMA",
    "PREDICTIONS_HASH_KEY",
    "build_s06_report",
]
