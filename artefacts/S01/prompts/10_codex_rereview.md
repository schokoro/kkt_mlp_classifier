Ты независимый ревьюер, уже запущенный отдельным codex exec. Это второй и последний раунд ревью. Не меняй файлы и не делегируй. Проверь весь diff по D01–D10, особенно исправления первого ревью: защита output_dir и пересечений pretrained, повреждённые метаданные, фактическое число строк fit, документация и изоляция теста ноутбуков. Найди конкретные дефекты с приоритетами и строками, укажи остаточные риски и каждый незакрытый пункт. Тесты на текущем дереве: быстрые 72 passed, 1 deselected; полный набор 73 passed. Пользовательские AGENTS.md и notebooks/train.ipynb и прочие прежние незакоммиченные данные вне diff.

# Исходный промпт
# Промпт оркестратору Codex — S01

Ты — **оркестратор** спринта **S01 «Отделение TF-IDF fit от обучения классификатора»** в репозитории `kkt_mlp_classifier`.

## Цель и источники

Выполнить только требования `artefacts/S01/value_dod.md` (Value, Scope, Out of scope, D01–D10). Сначала прочитай `AGENTS.md`, текущие `artefacts/SPEC_v1.0.md`, `artefacts/DECISION_LOG.md`, `artefacts/STATUS.md`, README и относящийся код (`kkt_clf/features.py`, `pipeline.py`, `config.py`, `artifact.py`, `inference.py`, скрипты и тесты). **Не редактируй `SPEC_v1.0.md`.** Новые указания S01 имеют приоритет над устаревшими частями baseline-спецификации; противоречия явно зафиксируй.

## Организация работы

1. До изменений проверь `git status --short`, ветку, действительные пути и наличие пользовательских изменений. Не стирай их, не коммить и не делай push. Если `AGENTS.md` или `value_dod.md` отсутствует — сообщи о блокере, а не подменяй содержание догадками.
2. Создай каталог `artefacts/S01/prompts/` и **сохраняй в нём точный текст каждого переданного подпроцесcу промпта до запуска подпроцесса**. Не перезаписывай предыдущие промпты: используй порядковые имена (`01_kimi_implementation.md`, `02_codex_tests.md`, `03_codex_review.md`, далее по факту). Внутри исходного `value_dod.md` и `orchestrator_prompt.md` ничего не меняй.
3. Проанализируй минимальный дизайн S01 и коротко перечисли конкретные изменения в контрактах конфигурации, автономного TF-IDF артефакта и пути train/inference. Это рабочие решения реализации, не предрешай дальнейший feature selection.
4. **Реализация кода**: делегируй `kimi -p` отдельным сохранённым промптом. Сам код не пиши. Если для исправления замечаний нужно изменить код, снова делегируй `kimi -p` с конкретным перечнем изменений.
5. **Тесты**: закажи написание/изменение тестов через **отдельный `codex exec`**, который не выполняет код-ревью. Передай DoD, границы спринта, предусмотренные отрицательные сценарии и необходимые интеграционные проверки. Сам тесты не пиши.
6. Запусти предусмотренные тесты и проверь diff. При обнаружении отклонений делегируй исправления соответствующему исполнителю, сохраняй новые промпты.
7. **Независимое ревью**: отдельный `codex exec`, которому передай исходную задачу, `value_dod.md`, текущее состояние diff и результаты тестов. Проверить все пункты D01–D10 и особо: совместимость baseline, отсутствие повторного fit, самостоятельность артефакта, отсутствие утечек, контракт train/inference, ограничение памяти, ошибки и документацию. Ревьюер **не исправляет код и тесты**.
8. Бюджет независимого ревью — **не более двух раундов** (первичное и одно повторное после исправлений). Не имитируй отсутствие замечаний: при исчерпании бюджета сообщи, какие проблемы не закрыты.
9. Обнови документацию в рамках своей роли: README, append-only `DECISION_LOG.md`, `STATUS.md`; создай `artefacts/S01/result.md` с итогом. Не утверждай результаты проверок, которых фактически не было.

## Ограничения реализации

- По умолчанию сохранён текущий алгоритм: `FeatureBuilder.fit` на train после split, две независимые TF-IDF ветки, CSR, входной `EmbeddingBag`, MLP с головами L1/L2/L3. Не менять семантику разделения данных и метрик.
- Нужен отдельный воспроизводимый запуск fit TF-IDF из **Parquet только с `name`**, затем сохранение и загрузка вектора признаков для повторного обучения MLP **без повторного fit**. Артефакт итоговой MLP должен оставаться самодостаточным для инференса.
- Работа на реальном корпусе 300 млн строк **не входит в S01**. Ограниченный подготовленный сэмпл приемлем. Если реализуешь sampling/итеративное чтение, следи за пиком памяти и явно документируй цену прохода по корпусу; не обещай потоковый `TfidfVectorizer.fit`, которого в sklearn нет.
- Не добавляй Chi², Gradient × Input или другие supervised-методы, не сокращай размерности и не запускай подбор гиперпараметров.
- Разрешено явно документировать технические допущения S01, но нельзя незаметно менять требования. Неподтверждённые вопросы (состав будущего корпуса, дедупликация, смешивание с labeled train) вынеси в `result.md`.
- ОДС, синтетика и маленький внешний Parquet служат для проверки механики. Метрики на них не являются доказательством качества будущего классификатора.

## Обязательные проверки

- Режим по умолчанию без регрессии; самостоятельный fit без столбцов меток; повторное использование готового артефакта.
- Согласованность `get_feature_names_out`, IDF, размера первого слоя, CSR-преобразования и результатов после сериализации/загрузки.
- Негативные сценарии: нет `name`, пустой корпус, повреждённый/несовместимый артефакт, некорректная конфигурация, существующий каталог результата; отсутствие использования val/test-меток для fit.
- Выполни доступные быстрые и полные тесты; если тесты не запускались или не прошли — не скрывай этого.

## Формат итогового ответа

Коротко отчитайся: выполненные D01–D10 (с явными невыполненными), какие файлы изменены, какие команды и результаты тестирования получены, где находится `artefacts/S01/result.md` и все промпты, сколько раундов ревью было и какие замечания остались. Отдельно назови непроверенные ограничения, включая большие объёмы данных. **Не приступай к S02.**

# Value и DoD
# S01 — Отделение TF-IDF fit от обучения классификатора

**Статус:** постановка к реализации.  
**Контекст:** исследовательский PoC `kkt_mlp_classifier`. Малый ODS — стенд проверки механики, не представительная выборка основной задачи.  
**Исходная реализация:** `FeatureBuilder` в `kkt_clf/features.py`; `pipeline.run_training` выполняет fit TF-IDF на размеченном train, затем обучает MLP; `features.joblib` уже входит в артефакт модели.

## Value

Можно **однократно построить TF-IDF-признаковое пространство из независимого неразмеченного корпуса** (только строки `name`), сохранить его и повторно использовать в обучающих прогонах MLP без повторного fit. При отсутствии внешнего векторизатора текущий baseline работает как прежде. Это создаёт основу для будущего supervised feature selection, но не реализует его в S01.

## Scope

1. Выделить из `run_training` **самостоятельную операцию fit и сохранения `FeatureBuilder`**, доступную через документированный скрипт/CLI. Вход — Parquet с колонкой `name`, метки не требуются. Поддержать чтение заранее подготовленного ограниченного корпуса/сэмпла; не предполагать, что 300 млн строк помещаются в память.
2. В обучающем пайплайне сделать два явно выбираемых режима:
   - `train` (по умолчанию): fit TF-IDF на размеченном train **с прежней семантикой** и прежними настройками;
   - `pretrained`: загрузка ранее сохранённого `FeatureBuilder` без повторного fit, затем `transform` train/val/test и обычное обучение MLP.
3. Сохранить контракт двух веток TF-IDF (char n-gram + word unigram), предобработку, индексацию и CSR. Новый формат автономного артефакта должен содержать векторизаторы, сведения о fit и возможность проверки целостности/совместимости. Артефакт итоговой MLP должен по-прежнему содержать самодостаточный `features.joblib` для инференса.
4. Записывать в метаданные режим fit, источник/идентификатор подготовленного корпуса, объём использованной выборки, seed, конфигурацию TF-IDF, фактические размеры словарей и ограничения воспроизводимости. Не утверждать наличие полного хеша или полного snapshot большого корпуса, если они не получены.
5. Добавить проверки новых режимов и документацию команд. Выбор способа чтения/ограничения подготовленного внешнего корпуса — ответственность реализации, но он должен быть явным, воспроизводимым и без неограниченного накопления текстов в RAM. Допускается документированное требование предварительно подготовленного сэмпла, если потоковая выборка из огромного корпуса не реализуется в S01.

## Out of scope

- Chi², Gradient × Input, supervised feature selection, подбор `top_k`, сокращение словаря, повторное обучение MLP после отбора.
- Полномасштабная обработка 300 млн строк, Spark ETL, распределённая или инкрементальная реализация IDF, ускорение векторизации.
- Изменение архитектуры MLP, правил отказа, таксономии, состава метрик или стратегии разбиения размеченных данных.
- Перемешивание размеченного train с неразмеченным корпусом по новой стратегии, оценка репрезентативности источника и выбор оптимального объёма корпуса.
- Подтверждение улучшения качества классификации на ODS или переносимости метрик на основной корпус.

## Definition of Done

- [ ] **D01. Baseline:** без новой опции текущий публичный запуск работает как раньше, TF-IDF fit получает только размеченный train после split, настройки `min_df`, `max_features`, `ngram_range`, `vectorizer_fit_rows` сохраняют прежний смысл.
- [ ] **D02. Standalone fit:** документированной командой создаётся отдельный переиспользуемый артефакт TF-IDF из Parquet с `name` без L1/L2/L3. Скрипт и обработка не требуют меток, внешних сервисов или предобученных весов.
- [ ] **D03. Reuse:** два последовательных запуска обучения могут использовать один и тот же заранее сохранённый векторизатор; внутри `run_training` в этом режиме повторный `fit` не происходит.
- [ ] **D04. Compatibility:** `train`, `save`, `load`, `inference` используют идентичную предобработку, IDF и порядок  признаков; совпадают результаты `transform` до и после сериализации (с явным численным допуском), правильна размерность первого слоя.
- [ ] **D05. No leakage:** самостоятельный fit не читает метки, а supervised обучение по-прежнему использует только размеченный train; val/test не влияют на `train`-режим fit. Совпадение текстов внешнего корпуса с val/test документируется как возможный трансдуктивный режим.
- [ ] **D06. Bounded input:** для внешнего источника есть документированный верхний предел объёма, поступающего в `TfidfVectorizer.fit`; нельзя молча прочитать весь огромный Parquet через `pandas.read_parquet` без ограничения. Сэмплирование/ограничение воспроизводимо либо явно зависит от версии подготовленного сэмпла и порядка строк.
- [ ] **D07. Errors:** отсутствие `name`, некорректный формат, пустой корпус, неподготовленный/несовместимый артефакт и ошибки загрузки обрабатываются диагностируемо; существующие артефакты пользователя не затираются без явного решения.
- [ ] **D08. Tests:** новые unit/integration тесты проверяют `train` и `pretrained`, отсутствие повторного fit, сериализацию и round-trip, детерминированность ограничения корпуса и корректность ошибок. Пройдены релевантные тесты; если полный набор нельзя выполнить, указаны причины и непройденные проверки.
- [ ] **D09. Docs:** обновлены README, `artefacts/DECISION_LOG.md` (append-only), `artefacts/STATUS.md`. Созданы `artefacts/S01/prompts/` и сохранены фактически использованные промпты исполнителей и ревьюера; есть краткий итог спринта с проверками и ограничениями.
- [ ] **D10. Review:** независимый reviewer проверил diff против этого DoD. Критичные замечания закрыты либо явно перечислены с незакрытым статусом; не заявлять завершение при невыполненных обязательных критериях.

## Проверка и критерии оценки

- Выполнить старый режим на тестовой фикстуре и убедиться, что путь по умолчанию не изменён.
- Выполнить standalone fit на **внешнем Parquet только с `name`**, затем обучение с ссылкой на сохранённый артефакт и инференс после повторной загрузки модели.
- Сравнить размерности, последовательность имён признаков, IDF и CSR-результаты `transform` до/после load; удостовериться, что `fit` в режиме `pretrained` не вызывается.
- Проверить ошибки и ограничения объёма тестами; запустить быстрый набор `.venv/bin/python -m pytest -q -m 'not slow'`, а при доступном окружении полный `.venv/bin/python -m pytest -q`.
- Фиксировать точные команды, результаты тестов, список изменённых файлов и ограничения масштабирования. **Метрики ODS не использовать как критерий приёмки этого спринта.**

## Открытые вопросы, не блокирующие S01

- Как сформировать репрезентативный подготовленный сэмпл из основного корпуса (в частности, учитывать ли повторы)?
- Нужно ли объединять независимый неразмеченный корпус с размеченным train для покрытия редких признаков?
- Какие варианты объёма корпуса сравнивать при последующей оценке устойчивости словаря и IDF?

Ответы не должны навязываться реализацией S01: достаточно сохранять необходимые границы и метаданные.

# Diff отслеживаемых файлов S01
diff --git a/README.md b/README.md
index f83bec4..6b0efe8 100644
--- a/README.md
+++ b/README.md
@@ -114,3 +114,19 @@ python scripts/run_inference.py --override '{"artifact_dir": "outputs/train_ods/
 5. VRAM: `model.hidden[0]` и `max_features` (первый слой = `n_features × hidden[0]`), `train.batch_size`, `train.eval_batch_size`.
 
 Векторизация однопоточная — это главный кандидат на ускорение в следующих итерациях.
+
+## S01: отдельный fit TF-IDF
+
+По умолчанию `features.mode="train"`: обе ветки обучаются на размеченном train после split, с прежним ограничением `vectorizer_fit_rows`. Для внешнего пространства сначала подготовьте ограниченный Parquet с колонкой `name` и идентификатором версии:
+
+```bash
+.venv/bin/python scripts/fit_vectorizer.py --input sample.parquet --output outputs/tfidf_v1 \
+  --features-preset ods_light --max-rows 100000 --seed 42 --source-id sample-v1
+.venv/bin/python scripts/run_train.py --preset ods_light --override '{"data":{"path":"labeled.parquet"},"output_dir":"outputs/run_v1","features":{"mode":"pretrained","pretrained_path":"outputs/tfidf_v1"}}'
+```
+
+Для следующего обучения задайте новый `output_dir` и тот же `pretrained_path`. Загруженный FeatureBuilder определяет конфигурацию признаков; отличия настроек признаков текущего прогона сопровождаются предупреждением. Повторный fit не выполняется. Модельный артефакт содержит собственный `features.joblib`, поэтому исходный автономный каталог для инференса не нужен. Автономный каталог содержит `features.joblib` и `vectorizer_meta.json` с конфигурацией, происхождением, размерами словарей, версиями и контрольной суммой; при загрузке выполняются проверки. Не загружайте joblib из недоверенных источников.
+
+Берутся первые не более `max_rows` непустых имён в порядке файла; повторы сохраняются. Выборка зависит от версии и порядка подготовленного файла, а seed применяется к дополнительной подвыборке `vectorizer_fit_rows`. Метаданные различают число непустых строк источника и фактическое число строк fit. Чтение выполняется батчами только колонки `name`; fit sklearn не является потоковым и хранит ограниченные тексты и промежуточный словарь в памяти. SHA-256 источника требует отдельного полного чтения файла. Укажите предел, исходя из доступной RAM: обработка 300 млн строк не проверялась. Существующий непустой каталог автономного артефакта отклоняется без `--force`; используйте новый путь. Обучение отклоняет непустой `output_dir` и его пересечение с каталогом внешнего векторизатора: выбирайте отдельный новый каталог каждого прогона.
+
+Внешний корпус может содержать тексты val/test: при совпадениях режим трансдуктивен и не измеряет обобщение на полностью невиданный корпус. Метки внешнего Parquet не читаются. Малые фикстуры и ODS проверяют механику, а не качество будущей модели.
diff --git a/artefacts/DECISION_LOG.md b/artefacts/DECISION_LOG.md
index ca5c6c1..d721882 100644
--- a/artefacts/DECISION_LOG.md
+++ b/artefacts/DECISION_LOG.md
@@ -100,3 +100,9 @@
 - **Решение:** `infer_*` сэмплируются из всего ODS, поэтому пересекаются с train обучающего прогона; метрики на них — только проверка механики (неизвестные метки, пустые признаки, проброс колонок), а не качество.
 - **Альтернативы:** брать вход инференса из `outputs/<run>/splits/test.parquet` — это выход прогона, а не фикстура генератора.
 - **Проверка:** для честной оценки на отложенных данных — указать в ноутбуке инференса `input_path = outputs/train_ods/splits/test.parquet`.
+
+### D-018 · 2026-10-31 · S01 · Независимый TF-IDF fit
+- **Цель и решение:** разрешить повторное обучение MLP на заранее построенных двух TF-IDF ветках. По умолчанию сохраняется fit на размеченном train; `pretrained` загружает автономный FeatureBuilder и копирует его в артефакт модели.
+- **Альтернативы:** потоковый подсчёт IDF и случайная выборка по всему корпусу отложены; выбран ограниченный префикс заранее подготовленного файла. Он зависит от версии и порядка строк, повторы сохраняются.
+- **Ограничения:** sklearn fit не потоковый, SHA-256 требует полного прохода по файлу; совпадение текстов с val/test делает оценку трансдуктивной. Спецификация первой итерации предписывает fit только на train; внешний режим добавлен по прямому требованию S01, исходная спецификация не менялась.
+- **Проверка:** команды и результаты в `artefacts/S01/result.md`; промышленные объёмы и GPU не проверены.
diff --git a/artefacts/STATUS.md b/artefacts/STATUS.md
index 0f7ab8d..7e827a0 100644
--- a/artefacts/STATUS.md
+++ b/artefacts/STATUS.md
@@ -47,3 +47,7 @@ cd ..
 .venv/bin/python scripts/run_train.py --preset ods_light --override '{"data": {"path": "fixtures/ods.parquet"},
   "output_dir": "outputs/train_ods_arch2", "model": {"hidden": [512, 256], "dropout": [0.3, 0.2], "norm": "layernorm"}}'
 ```
+
+## S01 · 2026-10-31
+
+Реализованы автономный ограниченный fit TF-IDF и режим повторного использования в обучении; по умолчанию остаётся fit на train. На текущем дереве быстрый набор: **72 passed, 1 deselected**, полный: **73 passed**. Проверки выполнялись на малых данных; 300 млн строк, промышленный корпус и GPU не проверялись. Итог и статус ревью см. `artefacts/S01/result.md`.
diff --git a/kkt_clf/__init__.py b/kkt_clf/__init__.py
index a406a67..f9aa931 100644
--- a/kkt_clf/__init__.py
+++ b/kkt_clf/__init__.py
@@ -1,9 +1,30 @@
 """Классификатор позиций кассовых чеков L1 → L2 → L3 (итерация 1, baseline)."""
-from .config import InferenceConfig, RunConfig, build_config, default_config, ods_light_config
+from .config import (
+    InferenceConfig,
+    RunConfig,
+    build_config,
+    build_feature_config,
+    default_config,
+    ods_light_config,
+)
 from .inference import Predictor, run_inference
 from .pipeline import TrainResult, run_training, setup_logging, verify_roundtrip
+from .vectorizer import (
+    LoadedVectorizer,
+    VectorizerArtifactError,
+    VectorizerError,
+    fit_and_save_vectorizer,
+    fit_vectorizer,
+    load_vectorizer_artifact,
+    load_name_corpus,
+    save_vectorizer_artifact,
+)
 
 __all__ = [
-    "InferenceConfig", "RunConfig", "build_config", "default_config", "ods_light_config",
+    "InferenceConfig", "RunConfig", "build_config", "build_feature_config",
+    "default_config", "ods_light_config",
     "Predictor", "run_inference", "TrainResult", "run_training", "setup_logging", "verify_roundtrip",
+    "LoadedVectorizer", "VectorizerArtifactError", "VectorizerError",
+    "fit_and_save_vectorizer", "fit_vectorizer", "load_vectorizer_artifact",
+    "load_name_corpus", "save_vectorizer_artifact",
 ]
diff --git a/kkt_clf/config.py b/kkt_clf/config.py
index adcc9bc..baf982a 100644
--- a/kkt_clf/config.py
+++ b/kkt_clf/config.py
@@ -17,6 +17,7 @@ OPTIMIZERS = ("adamw", "adam", "sgd")
 CLASS_WEIGHT_MODES = ("none", "inv_freq")
 TAU_MODES = ("fixed", "target_precision")
 HEADS = ("l1", "l2", "l3")
+FEATURE_MODES = ("train", "pretrained")
 
 
 @dataclass
@@ -38,10 +39,21 @@ class BranchConfig:
 
 @dataclass
 class FeatureConfig:
+    """Признаки TF-IDF.
+
+    ``mode="train"`` (по умолчанию): векторизаторы обучаются на размеченном
+    train внутри ``run_training``. ``mode="pretrained"``: векторизаторы
+    загружаются из автономного артефакта (``pretrained_path``), fit не
+    выполняется; раздел ``features`` конфига прогона в этом режиме не
+    применяется — авторитетна конфигурация, сохранённая в артефакте.
+    """
+
     char: BranchConfig = field(default_factory=lambda: BranchConfig((2, 5), 5, 300_000, True))
     word: BranchConfig = field(default_factory=lambda: BranchConfig((1, 1), 3, 100_000, True))
     vectorizer_fit_rows: int | None = 1_500_000
     transform_chunk_rows: int = 200_000
+    mode: str = "train"
+    pretrained_path: str | None = None
 
 
 @dataclass
@@ -170,8 +182,28 @@ def _normalize(cfg: RunConfig) -> None:
     cfg.model.dropout = list(cfg.model.dropout)
 
 
+def validate_feature_config(f: FeatureConfig) -> None:
+    """Проверки параметров TF-IDF (ветки, режим). Вызывается из validate_config."""
+    if f.mode not in FEATURE_MODES:
+        raise ValueError(f"features.mode: одно из {FEATURE_MODES}")
+    if f.mode == "pretrained" and not f.pretrained_path:
+        raise ValueError("features.pretrained_path: обязателен при features.mode='pretrained'")
+    if f.vectorizer_fit_rows is not None and f.vectorizer_fit_rows <= 0:
+        raise ValueError("features.vectorizer_fit_rows: None или > 0")
+    if f.transform_chunk_rows <= 0:
+        raise ValueError("features.transform_chunk_rows: > 0")
+    for name, b in (("char", f.char), ("word", f.word)):
+        if len(b.ngram_range) != 2 or b.ngram_range[0] < 1 or b.ngram_range[0] > b.ngram_range[1]:
+            raise ValueError(f"features.{name}.ngram_range: пара (min, max), 1 <= min <= max")
+        if isinstance(b.min_df, int) and b.min_df < 1:
+            raise ValueError(f"features.{name}.min_df: >= 1 (int) или доля в (0, 1]")
+        if b.max_features is not None and b.max_features <= 0:
+            raise ValueError(f"features.{name}.max_features: None или > 0")
+
+
 def validate_config(cfg: RunConfig) -> None:
     _normalize(cfg)
+    validate_feature_config(cfg.features)
     m = cfg.model
     if len(m.hidden) == 0:
         raise ValueError("model.hidden: нужен хотя бы один скрытый слой")
@@ -231,6 +263,15 @@ def ods_light_config() -> RunConfig:
 PRESETS = {"default": default_config, "ods_light": ods_light_config}
 
 
+def build_feature_config(preset: str = "default", overrides: dict[str, Any] | None = None) -> FeatureConfig:
+    """FeatureConfig из пресета RunConfig + переопределения (для standalone fit)."""
+    if preset not in PRESETS:
+        raise KeyError(f"Неизвестный пресет {preset!r}, есть: {list(PRESETS)}")
+    cfg = _merge(PRESETS[preset]().features, overrides or {})
+    validate_feature_config(cfg)
+    return cfg
+
+
 def build_config(preset: str = "default", overrides: dict[str, Any] | None = None) -> RunConfig:
     if preset not in PRESETS:
         raise KeyError(f"Неизвестный пресет {preset!r}, есть: {list(PRESETS)}")
diff --git a/kkt_clf/features.py b/kkt_clf/features.py
index 1d8c2c7..73702c6 100644
--- a/kkt_clf/features.py
+++ b/kkt_clf/features.py
@@ -121,16 +121,24 @@ class FeatureBuilder:
 
     def fit(self, texts: Sequence[str], seed: int = 0) -> "FeatureBuilder":
         texts = list(texts)
+        n_in = len(texts)
         n = self.cfg.vectorizer_fit_rows
+        subsampled = False
         if n is not None and len(texts) > n:
             rng = np.random.default_rng(seed)
             sel = np.sort(rng.choice(len(texts), size=n, replace=False))
             texts = [texts[i] for i in sel]
+            subsampled = True
             log.info("Векторизаторы обучаются на подвыборке train: %d строк", n)
         pre = self.preprocess(texts)
         self.char.fit(pre)
         self.word.fit(pre)
         self.fitted = True
+        self.fit_report_ = {
+            "rows_in": int(n_in),
+            "rows_fit": int(len(texts)),
+            "subsampled": subsampled,
+        }
         log.info(
             "Словарь: char=%d, word=%d, всего=%d",
             self.n_char_features, self.n_word_features, self.n_features,
diff --git a/kkt_clf/pipeline.py b/kkt_clf/pipeline.py
index f65f21d..79ce4a9 100644
--- a/kkt_clf/pipeline.py
+++ b/kkt_clf/pipeline.py
@@ -23,6 +23,7 @@ from .metrics import save_eval
 from .model import SparseMLP, param_counts
 from .reject import RejectRule, select_tau, tau_table
 from .train import compute_class_weights, resolve_device, set_seed, train_model
+from .vectorizer import TRANSDUCTIVE_NOTE, load_vectorizer_artifact, tfidf_config_dict
 
 log = logging.getLogger(__name__)
 
@@ -76,6 +77,43 @@ def _y(df: pd.DataFrame, maps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
     return tuple(maps.encode(lvl, df[lvl].to_numpy()) for lvl in LEVELS)
 
 
+def _check_fresh_output_dir(out_dir: str, pretrained_path: str | None) -> None:
+    """D07: прогон обучения не затирает существующие результаты пользователя.
+
+    Проверка выполняется до любых записей и сама ничего не создаёт:
+    существующий не пустой ``output_dir`` отклоняется, равно как и пересечение
+    с ``pretrained_path`` — иначе записи прогона могли бы повредить автономный
+    артефакт векторизатора или смешаться с ним. Автоматической перезаписи нет:
+    явное решение пользователя — вручную освободить каталог или выбрать новый.
+    """
+    out_abs = os.path.abspath(out_dir)
+    if os.path.exists(out_dir):
+        if not os.path.isdir(out_dir):
+            raise ValueError(
+                f"output_dir {out_dir!r} существует и не является каталогом: "
+                "прогон остановлен до любых записей."
+            )
+        entries = os.listdir(out_dir)
+        if entries:
+            raise ValueError(
+                f"output_dir {out_dir!r} уже существует и не пуст ({len(entries)} объектов): "
+                "прогон не затирает результаты пользователя. Укажите новый каталог "
+                "(или вручную освободите этот — автоматической перезаписи нет)."
+            )
+    if pretrained_path:
+        pre_abs = os.path.abspath(str(pretrained_path))
+        if (
+            pre_abs == out_abs
+            or pre_abs.startswith(out_abs + os.sep)
+            or out_abs.startswith(pre_abs + os.sep)
+        ):
+            raise ValueError(
+                f"output_dir {out_dir!r} пересекается с pretrained_path {pretrained_path!r} "
+                "(один каталог содержит другой): прогон не выполняется, чтобы не повредить "
+                "существующий не пустой автономный артефакт; выберите раздельные каталоги."
+            )
+
+
 def run_training(cfg: RunConfig) -> TrainResult:
     validate_config(cfg)
     t = {"start": time.perf_counter()}
@@ -87,6 +125,9 @@ def run_training(cfg: RunConfig) -> TrainResult:
     out = cfg.output_dir
     art_dir = os.path.join(out, "artifact")
     rep_dir = os.path.join(out, "reports")
+    # D07: непустой существующий выход (и пересечение с pretrained_path)
+    # отклоняется до любых записей.
+    _check_fresh_output_dir(out, cfg.features.pretrained_path if cfg.features.mode == "pretrained" else None)
     os.makedirs(rep_dir, exist_ok=True)
 
     # 1. Данные (§4.1–4.3)
@@ -97,8 +138,63 @@ def run_training(cfg: RunConfig) -> TrainResult:
     data.duplicates.examples.to_csv(os.path.join(rep_dir, "duplicate_examples.csv"), index=False)
     t["data"] = time.perf_counter()
 
-    # 2. Признаки (§4.4): обучение только на train
-    fb = FeatureBuilder(cfg.features).fit(data.splits["train"]["name"].tolist(), seed=cfg.seed)
+    # 2. Признаки (§4.4)
+    limitations = list(LIMITATIONS)
+    if cfg.features.mode == "pretrained":
+        # Загрузка ранее обученного FeatureBuilder: повторный fit не выполняется.
+        loaded = load_vectorizer_artifact(cfg.features.pretrained_path)
+        fb = loaded.builder
+        vmeta = loaded.meta
+        artifact_cfg = vmeta.get("tfidf_config")
+        run_cfg = tfidf_config_dict(cfg.features)
+        config_mismatch = artifact_cfg != run_cfg
+        if config_mismatch:
+            log.warning(
+                "Конфиг features прогона отличается от конфигурации артефакта векторизатора "
+                "(%s); применяется конфигурация артефакта, fit не выполняется",
+                cfg.features.pretrained_path,
+            )
+        feature_fit = {
+            "mode": "pretrained",
+            "artifact_dir": os.path.abspath(str(cfg.features.pretrained_path)),
+            "config_mismatch": config_mismatch,
+            "artifact": vmeta,
+        }
+        limitations.append(TRANSDUCTIVE_NOTE)
+        src_meta = vmeta.get("source", {})
+        fit_meta = vmeta.get("fit") or {}
+        log.info(
+            "Признаки: pretrained-артефакт %s (источник %s, seed=%s; источник: прочитано "
+            "непустых %s, truncated=%s; fit: фактически использовано строк %s (subsampled=%s))",
+            cfg.features.pretrained_path,
+            src_meta.get("path"),
+            vmeta.get("seed"),
+            src_meta.get("rows_used"),
+            src_meta.get("truncated"),
+            fit_meta.get("rows_fit"),
+            fit_meta.get("subsampled"),
+        )
+    else:
+        # Режим train (по умолчанию, D01): fit только на размеченном train после split.
+        fb = FeatureBuilder(cfg.features).fit(data.splits["train"]["name"].tolist(), seed=cfg.seed)
+        fr = dict(getattr(fb, "fit_report_", {}))
+        log.info(
+            "Признаки: режим train — fit на размеченном train после split: получено строк %s, "
+            "использовано для fit %s (subsampled=%s), seed=%s",
+            fr.get("rows_in"), fr.get("rows_fit"), fr.get("subsampled"), cfg.seed,
+        )
+        feature_fit = {
+            "mode": "train",
+            "fit_on": "labeled train (post-split)",
+            "seed": cfg.seed,
+            "tfidf_config": tfidf_config_dict(cfg.features),
+            "vocab": {
+                "n_char": fb.n_char_features,
+                "n_word": fb.n_word_features,
+                "n_total": fb.n_features,
+            },
+            **dict(getattr(fb, "fit_report_", {})),
+        }
     t["vectorizer_fit"] = time.perf_counter()
     X = {k: fb.transform(v["name"].tolist()) for k, v in data.splits.items()}
     t["vectorize"] = time.perf_counter()
@@ -174,6 +270,7 @@ def run_training(cfg: RunConfig) -> TrainResult:
         "seed": cfg.seed,
         "input": file_fingerprint(cfg.data.path),
         "data_report": data.report,
+        "feature_fit": feature_fit,
         "feature_stats": feature_stats,
         "best_epoch": best,
         "param_counts": pc.to_dict(orient="records"),
@@ -193,13 +290,14 @@ def run_training(cfg: RunConfig) -> TrainResult:
             "tau_selection": sel, "coverage_test": ev["coverage"],
             "n_empty_features_test": ev["n_empty_features"],
             "n_noisy_l3_classes_test": ev["n_noisy_l3_classes"],
-            "timings": timings, "limitations": LIMITATIONS,
+            "timings": timings, "limitations": limitations,
         }, f, ensure_ascii=False, indent=2, default=str)
     log.info("Артефакт: %s; отчёт: %s; %.1f с, пик RSS %.0f МБ", art_dir, rep_dir, timings["total_s"], timings["peak_rss_mb"])
     return TrainResult(
         config=cfg, output_dir=out, artifact_dir=art_dir, report_dir=rep_dir, data=data,
         feature_stats=feature_stats, param_counts=pc, history=hist, best_epoch=best,
         tau_table=tt, tau_selection=sel, evaluation=ev, test_predictions=test_frame, timings=timings,
+        limitations=limitations,
     )
 
 
diff --git a/tests/test_notebooks.py b/tests/test_notebooks.py
index 735314d..c4ef3a5 100644
--- a/tests/test_notebooks.py
+++ b/tests/test_notebooks.py
@@ -32,14 +32,35 @@ def test_notebook_is_facade(nb):
 def test_notebooks_run_all(tmp_path):
     from nbconvert.preprocessors import ExecutePreprocessor
 
-    fixtures = os.path.join(ROOT, "fixtures")
-    if not os.path.exists(os.path.join(fixtures, "ods.parquet")):
-        build_all(fixtures)
+    fixtures = tmp_path / "fixtures"
+    build_all(str(fixtures))
+    train_output = tmp_path / "outputs" / "train_ods"
+    inference_output = tmp_path / "outputs" / "inference_ods"
     for nb in NOTEBOOKS:
         book = nbformat.read(os.path.join(NB_DIR, nb), as_version=4)
-        ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(book, {"metadata": {"path": NB_DIR}})
+        for cell in book.cells:
+            if cell.cell_type != "code":
+                continue
+            cell.source = cell.source.replace('sys.path.insert(0, "..")', f'sys.path.insert(0, {str(ROOT)!r})')
+            if nb == "train.ipynb":
+                cell.source = cell.source.replace(
+                    'cfg = build_config(PRESET, OVERRIDES)',
+                    f'OVERRIDES["output_dir"] = {str(train_output)!r}\n'
+                    f'OVERRIDES["data"]["path"] = {os.path.join(ROOT, "data", "ods_alpha.pqt")!r}\n'
+                    'cfg = build_config(PRESET, OVERRIDES)',
+                )
+            else:
+                cell.source = cell.source.replace(
+                    'icfg = InferenceConfig.from_dict(CONFIG)',
+                    f'CONFIG.update(artifact_dir={str(train_output / "artifact")!r}, '
+                    f'input_path={str(fixtures / "infer_unknown_labels.parquet")!r}, '
+                    f'output_path={str(inference_output / "predictions.parquet")!r}, '
+                    f'report_dir={str(inference_output / "reports")!r})\n'
+                    'icfg = InferenceConfig.from_dict(CONFIG)',
+                )
+        ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(book, {"metadata": {"path": str(tmp_path)}})
         nbformat.write(book, tmp_path / nb)
-    art = os.path.join(ROOT, "outputs", "train_ods", "artifact")
+    art = train_output / "artifact"
     assert os.path.exists(os.path.join(art, "model.pt"))
-    out = pd.read_parquet(os.path.join(ROOT, "outputs", "inference_ods", "predictions.parquet"))
+    out = pd.read_parquet(inference_output / "predictions.parquet")
     assert set(output_columns(3)) <= set(out.columns)

# Новый файл kkt_clf/vectorizer.py
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


def _validate_fit_meta(fit: Any, rows_used: int, path: str) -> None:
    """Секция fit (если есть): фактический объём fit не превышает прочитанный.

    ``rows_used`` источника — прочитанные непустые строки; ``rows_fit`` —
    сколько из них реально дошло до ``TfidfVectorizer.fit`` после ограничения
    ``vectorizer_fit_rows``, поэтому ``rows_fit <= rows_in == rows_used``.
    """
    if fit is None:
        return
    where = f"Секция 'fit' метаданных артефакта {path!r}"
    if not isinstance(fit, dict):
        raise VectorizerArtifactError(
            f"{where} должна быть объектом, получено {type(fit).__name__}"
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
    читаются и содержат обязательные секции; вложенные секции ``source``,
    ``tfidf_config``, ``vocab`` — объекты; обязательные поля ``source``
    (path, rows_used, rows_read, rows_dropped_empty, max_rows, truncated) с
    согласованностью счётчиков; секция ``fit`` (если есть) согласована с
    источником (rows_fit <= rows_in == rows_used); версия формата совпадает;
    ``versions`` и ``files`` — объекты корректных типов; контрольная сумма и
    размер ``features.joblib`` сверяются до десериализации; объект — обученный
    ``FeatureBuilder``; размеры словарей и конфигурация TF-IDF совпадают с
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
    for k in ("source", "tfidf_config", "vocab"):
        if not isinstance(meta[k], dict):
            raise VectorizerArtifactError(f"Секция метаданных {k!r} в {path!r} должна быть объектом")
    _validate_source_meta(meta["source"], path)
    _validate_fit_meta(meta.get("fit"), meta["source"]["rows_used"], path)
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

# Новый файл scripts/fit_vectorizer.py
"""Standalone fit TF-IDF: Parquet с колонкой ``name`` → автономный артефакт векторизатора.

Метки не нужны; внешние сервисы и предобученные веса не используются.
Чтение корпуса ограничено --max-rows (по умолчанию 1 000 000): файл читается
по row-группам и весь в память не загружается. Ограничение воспроизводимо:
берутся первые max_rows непустых name в порядке следования строк файла
(стратегия «head»); результат зависит от версии файла и порядка строк.
Для случайной выборки заранее подготовьте перемешанный сэмпл.

Пример:
    python scripts/fit_vectorizer.py --input corpus.parquet --output outputs/tfidf_corpus \\
        --max-rows 1000000 --features-preset default --source-id corpus_v1
    python scripts/run_train.py --preset default --override \\
        '{"features": {"mode": "pretrained", "pretrained_path": "outputs/tfidf_corpus"},
          "data": {"path": "labeled.parquet"}}'
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf import build_feature_config, fit_and_save_vectorizer, setup_logging  # noqa: E402
from kkt_clf.vectorizer import DEFAULT_CORPUS_MAX_ROWS  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Parquet с колонкой name (метки не требуются)")
    p.add_argument("--output", required=True, help="Каталог автономного артефакта векторизатора")
    p.add_argument("--max-rows", type=int, default=DEFAULT_CORPUS_MAX_ROWS,
                   help=f"Верхний предел строк, поступающих в fit (по умолчанию {DEFAULT_CORPUS_MAX_ROWS}); "
                        "берутся первые непустые name в порядке следования. 0 или меньше — ошибка; "
                        "чтобы читать файл целиком, задайте заведомо большое число")
    p.add_argument("--features-preset", default="default", help="default | ods_light (источник конфига TF-IDF)")
    p.add_argument("--override", default="{}",
                   help="JSON с переопределениями FeatureConfig (ветки char/word, vectorizer_fit_rows)")
    p.add_argument("--seed", type=int, default=42, help="seed подвыборки vectorizer_fit_rows")
    p.add_argument("--source-id", default=None, help="Идентификатор/версия подготовленного корпуса (в метаданные)")
    p.add_argument("--force", action="store_true", help="Разрешить перезапись непустого каталога --output")
    a = p.parse_args()
    setup_logging()
    feat_cfg = build_feature_config(a.features_preset, json.loads(a.override))
    _, meta = fit_and_save_vectorizer(
        a.output, feat_cfg, a.input,
        max_rows=a.max_rows, seed=a.seed, source_id=a.source_id, overwrite=a.force,
    )
    src = meta["source"]
    print(json.dumps({
        "artifact": os.path.abspath(a.output),
        "source": src["path"],
        "source_id": src["source_id"],
        "rows_read": src["rows_read"],
        "rows_dropped_empty": src["rows_dropped_empty"],
        "rows_used": src["rows_used"],
        "rows_fit": meta["fit"]["rows_fit"],
        "fit_subsampled": meta["fit"]["subsampled"],
        "truncated": src["truncated"],
        "vocab": meta["vocab"],
        "seed": meta["seed"],
    }, ensure_ascii=False, indent=2))

# Новый файл tests/test_s01_vectorizer.py
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from kkt_clf import Predictor, build_config, run_training, verify_roundtrip
from kkt_clf.config import build_feature_config
from kkt_clf.data import DataValidationError
from kkt_clf.features import FeatureBuilder
from kkt_clf.vectorizer import (VectorizerArtifactError, VectorizerError,
    fit_and_save_vectorizer, load_name_corpus, load_vectorizer_artifact)


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / 'corpus.parquet'
    pd.DataFrame({'name': ['Молоко белое', None, '', 'Хлеб ржаной', 'Сыр твёрдый', 'Чай зелёный']}).to_parquet(path, index=False)
    return path


@pytest.fixture
def feature_cfg():
    return build_feature_config('ods_light', {'vectorizer_fit_rows': None})


def assert_features_equal(a, b, names):
    for branch in ('char', 'word'):
        x, y = getattr(a, branch), getattr(b, branch)
        np.testing.assert_array_equal(x.get_feature_names_out(), y.get_feature_names_out())
        np.testing.assert_allclose(x.idf_, y.idf_, rtol=1e-7, atol=1e-7)
    x, y = a.transform(names), b.transform(names)
    assert sp.isspmatrix_csr(x) and sp.isspmatrix_csr(y)
    assert x.shape == y.shape == (len(names), a.n_features)
    np.testing.assert_allclose((x - y).data, 0, atol=1e-7)


def test_standalone_cli_and_bounded_read(corpus, tmp_path, monkeypatch):
    # ParquetFile must request only name even if label columns exist.
    import pyarrow.parquet as pq
    labeled = tmp_path / 'extra.parquet'
    pd.DataFrame({'name': ['one', None, 'two', 'three'], 'L3': ['secret'] * 4}).to_parquet(labeled, index=False)
    original = pq.ParquetFile.iter_batches
    calls = []
    def read(self, *args, **kwargs):
        calls.append(kwargs)
        assert kwargs['columns'] == ['name']
        return original(self, *args, **kwargs)
    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', read)
    a = load_name_corpus(str(labeled), 2)
    b = load_name_corpus(str(labeled), 2)
    assert a.names == b.names == ['one', 'two']
    assert calls and a.rows_read == 3
    out = tmp_path / 'vectorizer'
    cmd = [sys.executable, 'scripts/fit_vectorizer.py', '--input', str(corpus), '--output', str(out),
           '--features-preset', 'ods_light', '--max-rows', '3', '--source-id', 'sample-v1',
           '--override', json.dumps({'char': {'min_df': 1}})]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    loaded = load_vectorizer_artifact(str(out))
    assert loaded.meta['source']['rows_used'] == 3
    assert loaded.meta['source']['source_id'] == 'sample-v1'
    assert loaded.meta['source']['max_rows'] == 3


def test_large_row_group_uses_bounded_batches(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / 'large.parquet'
    pq.write_table(pa.table({'name': [f'name {i}' for i in range(10000)],
                             'L3': ['secret'] * 10000}), path, row_group_size=10000)
    assert pq.ParquetFile(path).num_row_groups == 1
    original = pq.ParquetFile.iter_batches
    batch_sizes = []
    def read(self, *args, **kwargs):
        assert kwargs['columns'] == ['name']
        assert kwargs['batch_size'] <= 7
        for batch in original(self, *args, **kwargs):
            batch_sizes.append(batch.num_rows)
            yield batch
    monkeypatch.setattr(pq.ParquetFile, 'iter_batches', read)
    result = load_name_corpus(str(path), 7)
    assert result.names == [f'name {i}' for i in range(7)]
    assert batch_sizes and max(batch_sizes) <= 7
    assert result.truncated is True


@pytest.mark.parametrize('max_rows', [None, 0, -1])
def test_invalid_corpus_limit(corpus, max_rows):
    with pytest.raises(ValueError, match='max_rows'):
        load_name_corpus(str(corpus), max_rows)


def test_truncated_when_limit_reached_on_last_row(tmp_path):
    path = tmp_path / 'names.parquet'
    pd.DataFrame({'name': ['one', None, 'two']}).to_parquet(path, index=False)
    exact = load_name_corpus(str(path), 2)
    assert exact.rows_read == 3
    assert exact.truncated is False
    assert load_name_corpus(str(path), 1).truncated is True


def test_artifact_metadata_and_checksum(corpus, feature_cfg, tmp_path, monkeypatch):
    import kkt_clf.vectorizer as vectorizer

    path = tmp_path / 'artifact'
    builder, _ = fit_and_save_vectorizer(str(path), feature_cfg, str(corpus),
                                         max_rows=4, seed=123, source_id='sample-v1')
    meta_path = path / 'vectorizer_meta.json'
    meta = json.loads(meta_path.read_text())
    data = (path / 'features.joblib').read_bytes()
    assert meta['files']['features.joblib'] == {
        'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    assert meta['seed'] == 123
    assert meta['source']['source_id'] == 'sample-v1'
    assert meta['source']['rows_used'] == 4
    assert meta['source']['max_rows'] == 4
    assert meta['source']['strategy'] == 'head'
    assert meta['tfidf_config']['char'] and meta['tfidf_config']['word']
    assert meta['vocab'] == {'n_char': builder.n_char_features,
                             'n_word': builder.n_word_features, 'n_total': builder.n_features}
    assert meta['notes'] and meta['versions'] and meta['fit']
    for key in ('artifact_version', 'seed', 'source', 'tfidf_config', 'vocab'):
        modified = dict(meta)
        del modified[key]
        meta_path.write_text(json.dumps(modified))
        with pytest.raises(VectorizerArtifactError, match='обязательн'):
            load_vectorizer_artifact(str(path))
    meta_path.write_text(json.dumps(meta))
    (path / 'features.joblib').write_bytes(data + b'corruption')
    def forbidden(*args, **kwargs):
        pytest.fail('checksum must be checked before deserialization')
    monkeypatch.setattr(vectorizer.joblib, 'load', forbidden)
    with pytest.raises(VectorizerArtifactError, match='Контрольная сумма'):
        load_vectorizer_artifact(str(path))


def test_output_directory_checked_before_fit(corpus, feature_cfg, tmp_path, monkeypatch):
    path = tmp_path / 'existing'
    path.mkdir()
    sentinel = path / 'user.txt'
    sentinel.write_text('keep')
    def forbidden(*args, **kwargs):
        pytest.fail('fit must not run for a nonempty output directory')
    monkeypatch.setattr(FeatureBuilder, 'fit', forbidden)
    with pytest.raises(VectorizerArtifactError):
        fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=3)
    assert sentinel.read_text() == 'keep'


def test_cli_reports_actual_fit_size(corpus, tmp_path):
    out = tmp_path / 'limited'
    result = subprocess.run([
        sys.executable, 'scripts/fit_vectorizer.py', '--input', str(corpus),
        '--output', str(out), '--features-preset', 'ods_light', '--max-rows', '4',
        '--override', json.dumps({'vectorizer_fit_rows': 2}),
    ], check=True, capture_output=True, text=True)
    # Logging precedes the JSON object in stdout.
    report = json.loads(result.stdout[result.stdout.index('{'):])
    meta = json.loads((out / 'vectorizer_meta.json').read_text())
    assert meta['source']['rows_read'] == 6
    assert meta['source']['rows_used'] == 4
    assert meta['fit']['rows_in'] == 4
    assert meta['fit']['rows_fit'] == 2
    assert report['rows_read'] == 6
    assert report['rows_used'] == 4
    assert report['rows_fit'] == 2


@pytest.mark.parametrize('section,key,value', [
    ('files', 'features.joblib', 'invalid'),
    ('files', 'features.joblib', {'sha256': 123, 'bytes': 1}),
    ('versions', 'sklearn', ['invalid']),
    ('source', 'rows_read', 'invalid'),
    ('vocab', 'n_char', 'invalid'),
    ('fit', 'rows_fit', 'invalid'),
    ('tfidf_config', 'char', 'invalid'),
])
def test_corrupt_nested_metadata_types(corpus, feature_cfg, tmp_path, section, key, value):
    path = tmp_path / 'artifact'
    fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
    meta_path = path / 'vectorizer_meta.json'
    meta = json.loads(meta_path.read_text())
    meta[section][key] = value
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(VectorizerArtifactError):
        load_vectorizer_artifact(str(path))


@pytest.mark.parametrize('section,key', [
    ('source', 'rows_read'), ('source', 'rows_used'),
    ('vocab', 'n_char'), ('vocab', 'n_word'), ('vocab', 'n_total'),
    ('fit', 'rows_fit'), ('tfidf_config', 'char'), ('tfidf_config', 'word'),
    ('files', 'features.joblib'),
])
def test_missing_nested_metadata(corpus, feature_cfg, tmp_path, section, key):
    path = tmp_path / 'artifact'
    fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
    meta_path = path / 'vectorizer_meta.json'
    meta = json.loads(meta_path.read_text())
    del meta[section][key]
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(VectorizerArtifactError):
        load_vectorizer_artifact(str(path))


def make_cfg(tmp_path, mode='train', vectorizer=None):
    path = tmp_path / 'labeled.parquet'
    if not path.exists():
        pd.DataFrame([{'name': f'item{c} token{i}', 'L1': 'root', 'L2': f'group{c}', 'L3': f'class{c}'}
                      for c in range(3) for i in range(20)]).to_parquet(path, index=False)
    return build_config('ods_light', {'output_dir': str(tmp_path / ('out_' + mode)),
        'data': {'path': str(path), 'min_count': 5},
        'features': {'mode': mode, 'pretrained_path': vectorizer, 'vectorizer_fit_rows': 20,
                     'char': {'min_df': 1, 'max_features': 100}, 'word': {'min_df': 1, 'max_features': 50}},
        'model': {'hidden': [8], 'dropout': [0.0]},
        'train': {'epochs': 1, 'batch_size': 16, 'device': 'cpu', 'num_threads': 1}})


@pytest.mark.parametrize('same_as_pretrained', [False, True])
def test_training_rejects_nonempty_output_before_writing(corpus, feature_cfg, tmp_path, same_as_pretrained):
    path = tmp_path / 'existing'
    if same_as_pretrained:
        fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
        cfg = make_cfg(tmp_path, 'pretrained', str(path))
    else:
        path.mkdir()
        (path / 'reports').mkdir()
        (path / 'reports' / 'rare_l3_classes.csv').write_bytes(b'user report')
        cfg = make_cfg(tmp_path)
    (path / 'user.txt').write_bytes(b'user data')
    before = {p.relative_to(path): p.read_bytes() for p in path.rglob('*') if p.is_file()}
    directories_before = {p.relative_to(path) for p in path.rglob('*') if p.is_dir()}
    cfg.output_dir = str(path)
    with pytest.raises((ValueError, VectorizerError), match='(?i)(не пуст|non.?empty|exist)'):
        run_training(cfg)
    after = {p.relative_to(path): p.read_bytes() for p in path.rglob('*') if p.is_file()}
    assert after == before
    assert {p.relative_to(path) for p in path.rglob('*') if p.is_dir()} == directories_before


def test_baseline_fit_only_train_and_parameters(tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path)
    original = FeatureBuilder.fit
    calls = []
    def capture(self, texts, seed=0):
        calls.append((list(texts), seed, self.cfg))
        return original(self, texts, seed)
    monkeypatch.setattr(FeatureBuilder, 'fit', capture)
    result = run_training(cfg)
    assert len(calls) == 1
    texts, seed, params = calls[0]
    assert texts == result.data.splits['train']['name'].tolist()
    assert set(texts).isdisjoint(result.data.splits['val']['name'])
    assert set(texts).isdisjoint(result.data.splits['test']['name'])
    assert seed == cfg.seed and params.vectorizer_fit_rows == 20
    p = Predictor.load(result.artifact_dir, 'cpu')
    assert p.features.fit_report_['rows_fit'] == 20
    for branch in ('char', 'word'):
        vec, expected = getattr(p.features, branch), getattr(cfg.features, branch)
        assert vec.min_df == expected.min_df and vec.max_features == expected.max_features
        assert vec.ngram_range == expected.ngram_range
    assert verify_roundtrip(result)


def test_pretrained_reuse_and_model_roundtrip(corpus, feature_cfg, tmp_path, monkeypatch):
    path = tmp_path / 'standalone'
    before, _ = fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
    loaded = load_vectorizer_artifact(str(path)).builder
    names = ['МОЛОКО белое', 'неизвестное', '', None]
    assert_features_equal(before, loaded, names)
    def forbidden(*args, **kwargs):
        pytest.fail('pretrained training must not fit')
    monkeypatch.setattr(FeatureBuilder, 'fit', forbidden)
    cfg = make_cfg(tmp_path, 'pretrained', str(path))
    for i in range(2):
        cfg.output_dir = str(tmp_path / f'run{i}')
        result = run_training(cfg)
        predictor = Predictor.load(result.artifact_dir, 'cpu')
        assert_features_equal(before, predictor.features, names)
        assert predictor.model.input.in_features == before.n_features
        assert verify_roundtrip(result)


@pytest.mark.parametrize('kind', ['missing_name', 'empty', 'invalid'])
def test_bad_corpus(tmp_path, feature_cfg, kind):
    path = tmp_path / 'bad.parquet'
    if kind == 'missing_name':
        pd.DataFrame({'L3': ['x']}).to_parquet(path)
    elif kind == 'empty':
        pd.DataFrame({'name': ['', None]}).to_parquet(path)
    else:
        path.write_text('not parquet')
    with pytest.raises((DataValidationError, VectorizerError)):
        fit_and_save_vectorizer(str(tmp_path / 'out'), feature_cfg, str(path), max_rows=2)


def test_artifact_errors_and_no_overwrite(corpus, feature_cfg, tmp_path):
    path = tmp_path / 'artifact'
    builder, meta = fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
    sentinel = (path / 'features.joblib').read_bytes()
    with pytest.raises(VectorizerArtifactError):
        fit_and_save_vectorizer(str(path), feature_cfg, str(corpus), max_rows=4)
    assert (path / 'features.joblib').read_bytes() == sentinel
    other_cfg = build_feature_config('ods_light', {'char': {'min_df': 1}})
    with pytest.raises(VectorizerArtifactError):
        load_vectorizer_artifact(str(path), expected_cfg=other_cfg)
    meta_path = path / 'vectorizer_meta.json'
    saved = json.loads(meta_path.read_text())
    saved['artifact_version'] = 'incompatible'
    meta_path.write_text(json.dumps(saved))
    with pytest.raises(VectorizerArtifactError):
        load_vectorizer_artifact(str(path))
    saved['artifact_version'] = meta['artifact_version']
    meta_path.write_text(json.dumps(saved))
    (path / 'features.joblib').write_bytes(b'corrupted')
    with pytest.raises(VectorizerArtifactError):
        load_vectorizer_artifact(str(path))
    with pytest.raises(VectorizerArtifactError):
        load_vectorizer_artifact(str(tmp_path / 'missing'))


@pytest.mark.parametrize('overrides', [
    {'mode': 'unknown'}, {'mode': 'pretrained'}, {'vectorizer_fit_rows': 0},
    {'transform_chunk_rows': 0}, {'char': {'ngram_range': [3, 1]}},
    {'word': {'max_features': 0}},
])
def test_invalid_feature_config(overrides):
    with pytest.raises(ValueError):
        build_feature_config('ods_light', overrides)

# Новый файл artefacts/S01/result.md
# S01 — результат

Реализованы отдельный fit TF-IDF на ограниченном Parquet с `name` и повторное использование вектора в обучении MLP. Режим по умолчанию обучает две ветки на train после split. Модельный артефакт содержит FeatureBuilder для самостоятельного инференса. Существующий непустой каталог результата обучения отклоняется; это изменение прежнего поведения сделано для выполнения D07.

Первое независимое ревью выявило риск перезаписи результата, слабую проверку повреждённых метаданных и неоднозначность числа строк fit. Исправления и тесты внесены; повторное ревью выполняется.

Итоговые проверки: `.venv/bin/python -m pytest -q -m 'not slow'` — 72 passed, 1 deselected; `.venv/bin/python -m pytest -q` — 73 passed. Тест ноутбуков исполняет их во временном каталоге без изменения исходных ноутбуков.

Промпты: `artefacts/S01/prompts/`. `SPEC_v1.0.md` не менялась: её требование fit только на train относится к первой итерации, S01 добавляет внешний режим по прямому указанию.

Ограничения: 300 млн строк, промышленный корпус, целевая RAM/CPU/I/O и GPU не проверены. Чтение ограничено числом строк и батчем, но размер текстов и промежуточного словаря не ограничен в байтах; sklearn fit не потоковый. SHA-256 читает весь вход. Совпадение внешних текстов с val/test делает оценку трансдуктивной. ODS и синтетика проверяют механику, не качество классификации.

Открыты вопросы формирования и версии представительного сэмпла, обработки повторов, смешивания с размеченным train и сравнения объёмов для устойчивости словаря и IDF.
