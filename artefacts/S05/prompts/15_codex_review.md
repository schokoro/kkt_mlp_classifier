Ты независимый reviewer S05, read-only. Не исправляй файлы, не запускай вложенных агентов. Ниже содержатся исходный промпт, полный DoD D01–D12 и проверяемый diff реализации; проведи критическое ревью на дефекты и нарушения контракта прежде чем запускать реальное обучение. Проверь утечки validation/test, split и stratification, достоверность fingerprint и checkpoint, точность fixed-epoch refit и scheduler, CSR/память, macro-F1 и support=0, результаты/failed/skipped, HTML re-render, согласованность отчёта с обучением и gate. Не считай smoke доказательством качества. Сообщи замечания по убыванию серьёзности с точными путями/строками и минимальным воспроизведением; явно напиши если блокеров нет. Граница: только изменения S05 в коде/тестах, не включать пользовательский архив и ноутбук. Ниже — полные требования и фактический diff на момент отправки.

# Промпт Codex-оркестратору — S05 Audit & Trained Baseline

Ты — **Codex-оркестратор** S05 проекта `kkt_mlp_classifier`. Организуй аудит исходной ODS-разметки (~18k строк), фиксацию честного протокола сплитов и воспроизводимое обучение MLP на полном TF-IDF, чтобы принять обоснованное решение о переходе к S06 (сравнению трёх методов отбора признаков).

## Источники требований и приоритет

**Главный контракт — `artefacts/S05/value_dod.md`, D01–D12. Прочитай целиком.** Затем `AGENTS.md`, `artefacts/S04/result.md`, `artefacts/S03/value_dod.md`, `artefacts/STATUS.md`, `artefacts/DECISION_LOG.md`, README, фактические `kkt_clf/data.py`, `config.py`, `train.py`, `pipeline.py`, `experiment.py`, `metrics.py`, сохранение checkpoint, fit TF-IDF S02, скрипты/CLI и тесты. Оцени, что можно переиспользовать из S03/S04 без дублирования реализации.

**Историческая оговорка:** `STATUS.md` 2026-10-22 уже приводит результат на ODS: 18 046 строк, после `min_count=10` — 16 635 и 326 L3; `ods_light` baseline macro-F1 0.712–0.732. Это **не** подтверждение качества текущего full-TFIDF протокола. Сверь с исходными данными и не сравнивай несопоставимые runs.

## Ограничения и организация

1. Проверь `git status --short`, текущую основную ветку (предположительно `dev`), интеграцию S04 и сохранность пользовательских файлов. Создай `sprint/S05-audit-baseline` от актуальной основной ветки. Никаких `reset --hard`, `git clean`, force checkout, уничтожения данных; при препятствии — сообщи и предложи `git worktree`.
2. Оркестратор **не пишет код или тесты**: реализацию и исправления делегируй `kimi -p`; создание/изменение тестов — отдельным `codex exec`; независимое code review — ещё одним, независимым read-only `codex exec`. Сам можешь обновлять S05 result/README/STATUS/append-only DECISION_LOG. Каждый фактически переданный полный промпт **заранее** сохрани в `artefacts/S05/prompts/` под последовательным именем.
3. Бюджет независимого ревью — **два раунда**. Исправления после последнего раунда не объявлять прошедшими независимое ревью; фиксировать остаточный риск. Не выходить за пределы S05 и не подменять DoD.
4. Без разрешения пользователя **не делать commit/push/merge**. Не менять `artefacts/SPEC_v1.0.md`, пользовательские корпуса/ноутбуки, S02/S03/S04 outputs. Не размещать модели, parquet, логи и отчёты в `artefacts/` или Git. Все новые крупные файлы — в `outputs/S05/` или конфигурируемом каталоге.

## План исполнения

**A. Аудит до обучения.** Проверь реальную доступность размеченного Parquet (возможно `fixtures/ods.parquet`) и full-TFIDF S02 (`outputs/S02/tfidf_1m`, проверить, не угадывать). Проанализируй количество и частоты L1/L2/L3 до и после `min_count`, дубль `name`, дубли после фактического preprocessing, конфликтующие метки, пустые строки, иерархическую консистентность. Сгенерируй `outputs/S05/audit/{data_audit.json,class_support.csv,split_audit.csv}`. Сопоставь с историческими 18 046 / 16 635 / 326, расхождения объясни. Не исправляй спорные метки автоматически.

**B. Freeze split/protocol.** По итогам аудита реализуй/переиспользуй один воспроизводимый split: `outer train`/`comparison validation`/`sealed test` плюс train-internal monitor для выбора эпох и конфигурации. 80/10/10 и `min_count=10` — исходный ориентир текущего кода, но фактически применённые правила и распределение редких классов должны быть проверены и описаны. Избегай пересечений размеченных идентичных/нормализованных текстов между outer частями; если невозможно совместить с корректной стратификацией, **отметь блокер и предложи минимальную явную политику** вместо молчаливого изменения состава выборки. Зафиксируй fingerprints, labels, поддержку каждой L3, preprocessing, seed и split files. **Не вычисляй test quality ни разу**.

**C. Baseline на full TF-IDF.** Проверив fingerprint/corpus/sample/preprocessing, загрузи один full `FeatureBuilder` в режиме `pretrained`, а не `ods_light`-fit на размеченном train. Исходная MLP-конфигурация `ods_light` (относительно модели, не источника признаков) может быть отправной точкой. Построй train/internal-monitor curves, проверь правильность остановки, параметры оптимизации и отсутствие переобучения. При необходимости разрешён **ограниченный** config sweep; каждая конфигурация фиксируется, выбор по train-internal monitor, **не по comparison_val**. Не требуй определённого F1. Для сравнения рассчитай majority-class predictor из outer train.

**D. Refit final baseline и seed.** После выбора конфигурации/числа эпох по internal monitor обучи отдельные финальные модели на полном outer train по **фиксированному** расписанию, исключающему checkpoint selection на `comparison_val`. По умолчанию выполни ≥3 seed обучения (42/43/44) на одном split и full TF-IDF. Сохраняй model/checkpoint, manifest, curves и метрики на comparison validation; своди mean/std/range и per-L3 support на одном фиксированном наборе классов. Опорную MLP для атрибуции в S06 выбери **детерминированным train-only правилом**, например заранее заданный seed 42, а не максимальный validation F1. Если выполнить эту схему технически невозможно в существующем коде, делегируй минимальное расширение, но не переписывай тренировочный цикл целиком.

**E. Итоговый отчёт.** Строй единый `outputs/S05/report/` с `report.html`, `runs.csv`, `per_class.parquet`, `stages.csv`, `history.csv`, `manifest.json`, плюс аудит и сохранённые split/checkpoint. Переиспользуй существующий репортер S03/S04 там, где уместно; не ломай старые контракты. В HTML: все run статусы, кривые train/monitor, разброс по seed, сравнение с majority, train–monitor gap, per-L3 и support, ошибки, ресурсы, оговорка о трансдуктивном корпусе и решение **`ready/conditional/not_ready` для S06**. Повторная сборка HTML из таблиц — без обучения. Failed/skipped должны быть видны с причиной; missing data ≠ 0.

**F. Проверки.** Делегируй отдельные тесты на: инварианты исходной таксономии и фильтров, детерминизм/изоляцию сплитов, групповые дубли нормализованных имён, отсутствие test в выборе/ранжировании/качествах, inner-monitor checkpoint, фиксированный refit финальной MLP на outer train, воспроизводимость seed, correct macro-F1 по фиксированной L3 universe, корректность пустых векторов и `support=0`, identity full-TFIDF, checkpoint round-trip, raw top-1, manifest/HTML re-render, корректность failed/skipped. Измерь CPU elapsed/RSS и при доступности VRAM; не заявляй промышленные характеристики. Запусти `.venv/bin/python -m pytest -q -m 'not slow'`, затем полный `.venv/bin/python -m pytest -q`, `git diff --check`; если невозможно — объясни.

**G. Ревью и приёмка.** Независимые тесты и максимум два code review. По критичным замечаниям — Kimi fix, codex tests, при наличии бюджета — повторное review. В конце оцени D01–D12 как `done/partial/blocked` на основании фактических файлов/команд, не по обещаниям. Не начинай S06 даже если базовая модель хорошая.

## Ожидаемый результат

`artefacts/S05/result.md` должен содержать: конфигурацию/ветку/проверенные источники, полную статистику аудита, количество и список L3 после фильтрации, схему split с seed+hash, политику rare classes/dup collision, идентификатор full TF-IDF, все реально исполненные training runs и seed, поэпохные кривые, качество на comparison validation (не test), ресурсы, тесты, два ревью и непроверенные исправления, ссылки на 6 файлов отчёта, checkpoint и итоговый **gate для S06**. `STATUS.md` обнови, `DECISION_LOG.md` только дополни, README приведи в соответствие с CLI.

В финальном ответе коротко: что обнаружено в аудите, как изолирован test, сколько seed обучено, какую динамику показала MLP, где отчёт/опорный checkpoint, прошли ли тесты/ревью, что ограничивает переход к S06. **Не коммитить, не пушить и не мержить без моего прямого поручения.**
# S05 — Аудит ODS-разметки и воспроизводимый baseline на полном TF-IDF

**Статус:** постановка к реализации.  
**Проект:** `kkt_mlp_classifier`.  
**Предусловие:** S04 завершён и интегрирован в актуальную основную ветку (`dev` — проверить по фактическому Git). Работать в отдельной ветке `sprint/S05-audit-baseline`.  
**Ключевое разграничение:** S05 готовит честную экспериментальную базу; сравнение методов feature selection и бюджетов — **S06**.

## Value

Получить **проверенный состав размеченной ODS-выборки, неизменяемые разбиения и воспроизводимую достаточно обученную MLP на полном независимом TF-IDF**, пригодные для следующего сравнительного эксперимента (S06). Исследователь должен видеть, чему модель научилась, насколько результат устойчив к seed, какие классы можно оценивать и не нарушены ли границы train/validation/test.

Это исследовательский PoC в открытом контуре, **не оценка** будущего качества на реальных кассовых чеках промышленного масштаба. Из ODS в закрытый контур переносятся только код и методика, а не метрики, веса и рейтинги признаков.

## Исходные данные и исторический ориентир

- Размеченная выборка — прежний ODS, **порядка 18 тыс. строк** (Parquet с `name`, `L1`, `L2`, `L3`). По историческому `STATUS.md` (2026-10-22): 18 046 строк, 737 исходных L3; после `min_count=10` — 16 635 строк и 326 L3, исключено 411 редких L3 / 1 411 строк. **Это подлежащий перепроверке ориентир, не вновь измеренный факт**.
- В прежнем эксперименте на `ods_light` (TF-IDF fit на размеченном train) получали raw L3 macro-F1 ≈0.712–0.732 и accuracy ≈0.804–0.814. Эти результаты **не сопоставимы автоматически** с будущими результатами на полном TF-IDF из неразмеченного корпуса; расхождения необходимо объяснять различиями конфигураций и протокола, а не объявлять регрессией.
- Полный TF-IDF S02: ориентир **300k char + 100k word**, fit на снимке `data/corpus.pqt` (981 850 непустых уникальных строк по отчёту S02). **Путь, fingerprint, состав, реальный размер словаря и доступность артефакта проверить до запуска**. Нельзя подменять его `ods_light` или повторно fit на validation/test.
- Атрибуция Gradient × Input и сравнение Global/Class-aware Chi² реализованы в S03/S04; их результаты одноэпоховых smoke не являются измерением качества методов.

## Scope и последовательность

### A. Аудит входной разметки — до обучения

Проверить фактические число строк, уникальных исходных и нормализованных названий, пустые/невалидные значения, дубли и пересечения одинаковых/нормализованных строк, конфликтующие метки одного имени, непротиворечивость L1→L2→L3, число L1/L2/L3, частоты L3 (включая длинный хвост), доли и список категорий/строк, исключённых существующим `min_count`.

**Не менять молча состав данных.** Зафиксировать историческую политику `min_count=10` как исходную конфигурацию для воспроизведения. По результатам аудита объяснить её влияние на набор оцениваемых классов. Если существующая фильтрация/сплит создаёт нарушение целостности или существенные конфликты меток, не скрывать это «очисткой», а зафиксировать блокер и предложить решение до содержательного сравнения.

### B. Единый протокол разбиения и измерений

- Воспроизводимо зафиксировать **outer train / comparison validation / sealed test** из прошедшей аудит выборки. За исходный ориентир взять существующее стратифицированное `80/10/10` и ту же политику редких классов; если фактическое распределение не допускает корректного протокола, обосновать необходимое изменение, не подменяя его молча. Сохранять row identity/fingerprints, порядок меток, seed, поддержку по каждой L3 во всех частях.
- Проверить пересечения одинаковых `name` и текстов после *того же preprocessing*, что реально использует TF-IDF. Пересечения **между размеченными частями** — отдельный риск supervised leakage, который нельзя скрыть; при необходимости предложить group-aware split, проверив осуществимость стратификации и сохранив новую схему явно. Не путать это с совпадениями **текстов** во внешнем неразмеченном TF-IDF-корпусе: они не передают метки, но могут делать постановку трансдуктивной — отметить в отчёте.
- `sealed test` **не использовать** для ранжирования, выбора модели, настройки эпох, threshold, архитектуры, бюджетов или сравнения гипотез S05/S06. Его содержимое и метрики не открывать в обычном отчёте S05. При необходимости проверки целостности разрешены только count/hash/support, без прогонов качества модели на test.
- Для выбора конфигурации и остановки обучения предусмотреть **внутренний monitor split только внутри outer train** (или равноценный документированный train-only протокол выбора числа эпох). **Comparison validation не использовать для определения checkpoint модели, из которой S06 будет извлекать G×I.** После фиксации настроек допустимо переобучить финальную MLP на всём outer train с выбранным по внутренней проверке фиксированным числом эпох. Границу между `train_fit`, `train_monitor`, `comparison_val` и `sealed_test` отразить в provenance.
- Оценивать **raw top-1 по L3 без отказа**: accuracy, macro-F1 на **зафиксированном наборе классов**, precision/recall/F1/support по L3, матрицу ошибок/топ путаемых пар; не смешивать с метриками покрытия/reject. Отдельно указывать пустые векторы, отсутствие предсказаний и неподдержанные классы. Не дорисовывать метрики для классов с `support=0`.

### C. Содержательный baseline на полном TF-IDF

- Использовать **один и тот же сохранённый full TF-IDF** для всех запусков S05 в режиме pretrained, не запускать supervised selection и не refit векторизатора между seed. Сохранять идентичность словаря, IDF и конфигурации входов MLP.
- Сначала воспроизвести обоснованную существующую конфигурацию обучения (`ods_light` как ориентир *MLP*, но full TF-IDF как **источник признаков**). Если обнаружено недообучение/переобучение, разрешён **ограниченный, протоколируемый подбор** параметров оптимизации/ёмкости модели **только на internal monitor**, без широкого HPO и без изменения архитектурного семейства MLP. Все исследованные конфигурации и причина выбора фиксируются.
- Выполнить **не менее трёх независимых seed обучения** на одинаковых фиксированных outer splits, full TF-IDF и финальной выбранной конфигурации (по умолчанию 42/43/44, конфигурируемо). Отдельно фиксировать seed разбиения и seed обучения/инициализации. Не объявлять недовыполненные прогоны успешными.
- Сохранять learning curves: train/internal-monitor loss и raw L3 macro-F1/accuracy по эпохам, лучший epoch, критерий остановки, разницу train↔monitor, итоговую оценку на `comparison_val` (один зафиксированный протокол). Для ориентира рассчитать тривиальный majority-class baseline **на том же comparison validation** с train-only выбором большинства; не требовать совпадения с историческими F1.
- Подготовить **однозначный финальный checkpoint для S06 Gradient × Input**, обученный на полном TF-IDF без использования `comparison_val`/`sealed_test` меток для выбора его checkpoint. Сохранить параметры, веса, `label_maps`, ссылку/хэш full TF-IDF, train fingerprints, seed, число эпох, критерий выбора и проверку загрузки. Если несколько seed — указать детерминированное правило выбора опорного checkpoint, **не по comparison validation** (например, предзаданный seed 42).
- Для CPU/GPU измерять доступные время, пик RSS/VRAM, размер артефакта. Невозможность CUDA отражать как `unavailable`, а не как пройденную проверку. Входы CSR, без полной dense матрицы.

### D. Решение о готовности к S06

После аудита и обучения подготовить явный вывод `ready / not_ready / conditional` с основаниями:

1. Зафиксированы происхождение данных, единые сплиты, классы и отсутствие необъявленных утечек меток.
2. Baseline обучается содержательно: кривые показывают обучение, а результат на `comparison_val` осмысленно лучше тривиального baseline; характер train–validation gap объяснён. **Численный порог F1 заранее не назначается.**
3. Показан разброс по seed, есть воспроизводимый checkpoint на full TF-IDF и зафиксирована конфигурация для честного будущего сравнения.
4. Известны ограничения редких классов и возможная трансдуктивность внешнего TF-IDF.

Если gate не пройден — S05 может завершиться **диагностическим результатом**, но нельзя объявлять протокол S06 готовым или начинать сравнительные запуски «на всякий случай».

## Out of scope

- Любой эксперимент `global`/`class_aware`/`gradient_x_input` с бюджетами 50k/100k/200k: **это S06**. Не выбирать победителя и не оптимизировать K.
- Новые методы отбора, новая архитектура MLP, изменение таксономии, автоматическое исправление/разметка спорных данных, создание нового корпуса.
- Подбор порога отказа, калибровка, сравнение качества при reject, финальное вскрытие test, промышленный fit на 300 млн строк, Spark/GPU-масштабирование.
- Смешивание признаков из чековых метаданных: доступна только строка `name`.
- Замена/перезапись уже созданных S01–S04 выходов, пользовательских файлов, `data/corpus.pqt`; изменение `artefacts/SPEC_v1.0.md`.
- Commit/push/merge без отдельного разрешения пользователя.

## Обязательные артефакты S05

Все крупные результаты — **вне Git**, в конфигурируемом каталоге `outputs/S05/` (уже существующие файлы не перезаписывать):

```text
outputs/S05/
├── audit/
│   ├── data_audit.json       # источники, количества, фильтры, конфликты, overlap
│   ├── class_support.csv     # L1/L2/L3, исходный/после фильтра, train/val/test support
│   └── split_audit.csv       # части, строки, группы, пересечения, fingerprints
├── splits/                  # замороженные train/val/test (пути и hashes в manifest)
├── checkpoints/             # baseline по seed + выбранный train-only checkpoint
└── report/
    ├── report.html           # автономный агрегированный отчёт по всем плановым runs
    ├── runs.csv              # одна строка на run, включая failed/skipped
    ├── per_class.parquet     # (run_id, L3, precision, recall, f1, support) на comparison_val
    ├── stages.csv            # fit/train/eval: время, RSS/VRAM, status, shared vs per-run
    ├── history.csv           # run_id, epoch, split, loss, L3 macro-F1/accuracy, LR
    └── manifest.json         # входы, конфиги, протокол, hashes, env, seed, статусы
```

**Контракт отчёта:** агрегировать все seed/конфигурации, а не собирать отдельные HTML вручную. `report.html` содержит таблицу каждого run, кривые обучения, mean/std/range macro-F1 и accuracy по seed (без неверного объединения разных конфигураций), train↔monitor gap, сравнение с trivial baseline, распределение support L3/длинный хвост, диаграмму ошибок по категориям, ресурсные затраты и явное заключение о S06 gate. Не показывать test-метрики. HTML автономен (без CDN) и **восстанавливается из сохранённых машинных файлов без обучения**. Для failed/skipped отсутствующие числа обозначаются `null`, а не нулями. Схемы действующего отчёта S03/S04 желательно **переиспользовать**, не ломая их прежние контракты; если поле имеет иную семантику, явно версионировать формат, а не переписывать историю.

**Минимальные поля `runs.csv`:** `run_id`, `config_id`, `train_seed`, `split_seed`, `status`, `reason`, `n_train`, `n_monitor`, `n_val`, `n_classes`, `full_tfidf_id`, `epochs_completed`, `best_epoch`, `l3_macro_f1`, `l3_accuracy`, `train_seconds`, `peak_rss_bytes`, `checkpoint_path`. Для всех метрик указать, по какой выборке они вычислены (`comparison_val` для последних L3 метрик). Показатели ресурсов, не измеренные фактически, — `null`. Метаданные по разделяемым стадиям и fingerprint corpus/feature должны быть в `manifest.json`.

## Definition of Done

- [ ] **D01 — Изоляция.** Ветка `sprint/S05-audit-baseline` от актуальной основной ветки, проверено состояние дерева; S01–S04/пользовательские данные нетронуты. Никаких commit/push/merge.
- [ ] **D02 — Аудит разметки.** Автоматически сформированы полный `data_audit.json` и `class_support.csv`: исходный объём, классы и поддержка, rare-class policy, исключения, пустоты, конфликты иерархии и меток, дубликаты и нормализованные коллизии. Исторические числа проверены, расхождения объяснены.
- [ ] **D03 — Фиксация split.** Заморожены совместимые с данными outer train / comparison validation / sealed test и внутренний протокол выбора модели; hashes/seed/состав L3/support зафиксированы. Test не используется для метрик и выбора. Пересечения train/val/test проверены; реальная утечка не замолчана.
- [ ] **D04 — Полный TF-IDF.** Найден и проверен сохранённый S02 full TF-IDF с достоверным provenance и реальными размерами; все модели S05 используют **один** артефакт в pretrained-режиме. Если full TF-IDF отсутствует или несовместим, явно обозначен блокер либо создан новый версионированный артефакт на воспроизводимом корпусе **без перезаписи S02**.
- [ ] **D05 — Воспроизводимое обучение.** Есть train-only протокол выбора эпох/конфигурации без использования comparison validation для выбора G×I checkpoint; сохраняются learning curves и признаки недо-/переобучения. Ограниченный подбор конфигураций документирован, а не скрыт.
- [ ] **D06 — Несколько seed.** Для выбранной конфигурации **не менее 3 успешных runs с различными train seed** на одних outer splits/TF-IDF, сохранены raw L3 accuracy/macro-F1, mean/std/range и per-L3 support; неуспешные запуски отражены отдельно.
- [ ] **D07 — Проверка содержательности.** Baseline сопоставлен с простым majority-class контролем; train/monitor curves и generalization gap проанализированы. Если обучить устойчивую модель не удалось, причина и `not_ready` записаны, а качество не выдумано. Исторические `ods_light` метрики не выдаются за сопоставимый baseline.
- [ ] **D08 — Checkpoint для S06.** Отдельно сохранён и повторно загружен один опорный checkpoint full-TFIDF MLP, выбранный по train-only правилу (не по comparison_val), и выполнен независимый inference на validation. Label maps, входной словарь, хэши, seed, число эпох, конфигурация и provenance проверяются.
- [ ] **D09 — Агрегированная отчётность.** Сформированы все обязательные файлы `audit/` и `report/` с указанным контрактом; HTML собирается из таблиц без retrain. Все запланированные, failed и skipped runs видны; test-метрик нет; числа, дельты и средние проверены тестами.
- [ ] **D10 — Ресурсы и переносимость.** Замерены локальные CPU/GPU (при доступности), elapsed, peak RSS/VRAM, size checkpoint; нет full dense `N×F` в коде обучения. CPU fallback и невозможность GPU отражены явно; никаких утверждений о промышленной масштабируемости.
- [ ] **D11 — Тесты/ревью.** Реализация делегирована `kimi -p`, тесты — отдельным `codex exec`, независимые ревью — другим `codex exec` с бюджетом **2 раунда**. Полные использованные задания сохранены в `artefacts/S05/prompts/`. Проверены быстрый и полный pytest, `git diff --check`, тесты целостности split, отсутствия leakage, правильности early-stopping, reproducibility/provenance и отчёта; ограничения ревью честно отмечены.
- [ ] **D12 — Документация и решение.** `artefacts/S05/result.md` фиксирует статусы D01–D12 (`done/partial/blocked`), команды и версии, размеры классов, split/feature hashes, итоги seed, ресурсы, результаты тестов и двух ревью, ограничения и однозначный вердикт `ready/conditional/not_ready` для S06. README и STATUS обновлены, DECISION_LOG дополнен append-only; `SPEC_v1.0.md` не изменялся.

## Правила приёмки

S05 **не обязан достичь заранее заданного F1**. Критичны прозрачный аудит, отсутствие утечки меток, воспроизводимость полноценного обучения, корректность оценки, сохранение checkpoint и машинно проверяемый единый отчёт. Недоступность CUDA или ограничения локальной мощности не считаются доказательством непригодности метода, но должны быть документированы. Если 3 seed или другой обязательный DoD не выполнен, указывать `partial`, а не объявлять его выполненным. **Готовность S05 как инженерного спринта и готовность к S06 — разные решения:** работающий диагностический контур не означает, что ранжирование признаков уже стоит сравнивать.
diff --git a/kkt_clf/train.py b/kkt_clf/train.py
index b7e3474..03aed37 100644
--- a/kkt_clf/train.py
+++ b/kkt_clf/train.py
@@ -249,13 +249,17 @@ def train_model(
     class_weights: dict[str, np.ndarray | None],
     device: torch.device,
     seed: int,
+    restore_best: bool = True,
 ) -> tuple[list[dict], dict, list[dict]]:
     """Цикл обучения с early stopping по val и LR scheduler.
 
     Возвращает историю по эпохам, инфо о лучшей эпохе и трассу lr по шагам.
 
     Строки train с пустым вектором признаков в обучение не попадают.
-    Лучшие веса загружаются в ``model`` по окончании.
+    По окончании в ``model`` загружаются лучшие веса (``restore_best=True``,
+    поведение по умолчанию); при ``restore_best=False`` остаются веса
+    финальной эпохи — режим фиксированного refit без выбора checkpoint
+    (S05: финальная MLP на outer train с заранее зафиксированным числом эпох).
     """
     if train_cfg.early_stopping_metric not in ("l3_macro_f1", "l3_accuracy", "val_loss"):
         raise ValueError("train.early_stopping_metric: l3_macro_f1 | l3_accuracy | val_loss")
@@ -328,8 +332,9 @@ def train_model(
             if bad_epochs >= train_cfg.patience:
                 log.info("early stopping: %d эпох без улучшения", bad_epochs)
                 break
-    if best_state is not None:
+    if best_state is not None and restore_best:
         model.load_state_dict(best_state)
+    best["restored_best"] = bool(best_state is not None and restore_best)
     best["metric"] = train_cfg.early_stopping_metric
     best["n_train_rows_used"] = int(len(train_rows))
     best["scheduler"] = train_cfg.scheduler
diff --git a/kkt_clf/freeze_split.py b/kkt_clf/freeze_split.py
new file mode 100644
index 0000000..c33a48d
--- /dev/null
+++ b/kkt_clf/freeze_split.py
@@ -0,0 +1,611 @@
+"""S05 (D03): заморозка воспроизводимого group-aware сплита размеченной выборки.
+
+Строит детерминированный стратифицированный group-aware сплит 80/10/10
+(``outer train`` / ``comparison validation`` / ``sealed test``) и
+train-internal monitor (``train_fit`` / ``train_monitor``) внутри outer train.
+
+Группа — множество строк с одинаковым ключом char-ветки реального
+preprocessing TF-IDF (``strip_whitespace(preprocess(name))``). Равенство
+word-ключа (схлопывание пробелов после ``preprocess``) влечёт равенство
+char-ключа, поэтому при группировке по char-ключу совпадают и строки,
+неразличимые для word-ветки: пересечений нормализованных текстов между
+частями нет ни в одной ветке. Метрики качества не вычисляются; sealed test
+используется только для count/hash/support-проверок целостности.
+
+Отказ (файлы не записываются) при:
+- null/пустых значениях в обязательных колонках;
+- нарушении таксономии (у L3 не один L2, у L2 не один L1);
+- конфликтных метках внутри одной группы;
+- невозможности стратификации (L3 отсутствует в какой-либо части);
+- отклонении фактических долей частей от целевых сверх допуска;
+- пересечении нормализованных ключей между частями (защитная проверка).
+
+Исключение редких L3 по ``min_count`` — объявленная политика, а не молчаливое
+удаление: состав исключённых строк и классов фиксируется в manifest и
+таблицах поддержки. Существующие файлы не перезаписываются.
+"""
+from __future__ import annotations
+
+import datetime as dt
+import json
+import os
+import platform
+import subprocess
+import sys
+from dataclasses import dataclass, field
+
+import numpy as np
+import pandas as pd
+
+from .data import REQUIRED_COLUMNS, check_taxonomy, file_fingerprint, filter_min_count
+from .features import preprocess, strip_whitespace
+from .labels import LEVELS, LabelMaps
+
+PARTS_OUTER = ("outer_train", "comparison_val", "sealed_test")
+PARTS_MONITOR = ("train_fit", "train_monitor")
+PART_NAMES = PARTS_OUTER + PARTS_MONITOR
+ROW_ID_COLUMN = "row_id"
+
+# Столбцы отчётов (единый порядок во всех CSV).
+SPLIT_AUDIT_COLUMNS = [
+    "part", "n_rows", "share_target", "share_actual", "n_groups", "n_l3",
+    "l3_min_support", "l3_max_support", "char_keys_shared_with_other_parts",
+    "word_keys_shared_with_other_parts", "n_row_ids_unique", "sha256", "bytes",
+]
+SUPPORT_COLUMNS = [
+    "L1", "L2", "L3", "kept_after_min_count", "support_outer_train",
+    "support_train_fit", "support_train_monitor", "support_comparison_val",
+    "support_sealed_test",
+]
+
+
+class FreezeSplitError(ValueError):
+    """Отказ заморозки сплита; файлы не записываются."""
+
+
+@dataclass
+class FrozenSplit:
+    parts: dict[str, pd.DataFrame]
+    excluded: pd.DataFrame
+    maps: LabelMaps
+    report: dict = field(default_factory=dict)
+
+
+def add_group_keys(df: pd.DataFrame) -> pd.DataFrame:
+    """Добавляет ключи идентичности текста обеих TF-IDF-веток."""
+    pre = preprocess(df["name"].tolist())
+    out = df.copy()
+    out["_char_key"] = [strip_whitespace(t) for t in pre]
+    out["_word_key"] = [" ".join(t.split()) for t in pre]
+    return out
+
+
+def validate_rows(df: pd.DataFrame) -> None:
+    """Отказ при null/пустых значениях и нарушениях таксономии."""
+    bad = np.zeros(len(df), dtype=bool)
+    counts: dict[str, int] = {}
+    for c in REQUIRED_COLUMNS:
+        col = df[c]
+        mask = col.isna().to_numpy()
+        notna = ~mask
+        if notna.any():
+            mask[notna] = col[notna].astype(str).str.strip().eq("").to_numpy()
+        counts[c] = int(mask.sum())
+        bad |= mask
+    if bad.any():
+        raise FreezeSplitError(
+            f"Отказ freeze split: null/пустые значения в обязательных колонках "
+            f"({counts}); состав данных не изменялся и не будет изменён молча."
+        )
+    try:
+        check_taxonomy(df)
+    except Exception as e:
+        raise FreezeSplitError(f"Отказ freeze split: нарушена таксономия: {e}") from e
+
+
+def check_group_label_consistency(dfk: pd.DataFrame, n_examples: int = 10) -> None:
+    """Отказ, если внутри группы (char-ключ) встречается больше одного пути меток."""
+    paths = dfk.groupby("_char_key")["path"].nunique()
+    bad = paths[paths > 1]
+    if bad.any():
+        examples = [
+            {"char_key": k, "n_paths": int(n)}
+            for k, n in bad.head(n_examples).items()
+        ]
+        raise FreezeSplitError(
+            f"Отказ freeze split: конфликтные метки внутри группы "
+            f"(одинаковый текст после preprocessing обеих TF-IDF-веток несёт "
+            f"несколько путей L1/L2/L3): {len(bad)} групп; примеры: {examples}. "
+            f"Метки не исправлялись."
+        )
+
+
+def make_groups(dfk: pd.DataFrame) -> pd.DataFrame:
+    """Одна строка на группу: ключи, размер, единственный путь меток, row ids."""
+    groups = (
+        dfk.groupby("_char_key", sort=True)
+        .agg(
+            n_rows=(ROW_ID_COLUMN, "size"),
+            L1=("L1", "first"),
+            L2=("L2", "first"),
+            L3=("L3", "first"),
+            n_paths=("path", "nunique"),
+            row_ids=(ROW_ID_COLUMN, list),
+        )
+        .reset_index()
+        .rename(columns={"_char_key": "group_id"})
+        .sort_values("group_id", ignore_index=True)
+    )
+    if (groups["n_paths"] > 1).any():
+        raise FreezeSplitError("Внутри группы несколько путей меток; см. check_group_label_consistency")
+    word_keys = dfk.groupby("_char_key")["_word_key"].nunique()
+    groups["_word_key"] = groups["group_id"].map(
+        dfk.groupby("_char_key")["_word_key"].first()
+    )
+    groups["_word_keys_per_group"] = groups["group_id"].map(word_keys)
+    return groups
+
+
+def _largest_remainder(n: int, weights: np.ndarray) -> np.ndarray:
+    """Целочисленное разбиение ``n`` по весам с суммой ровно ``n``."""
+    raw = n * weights
+    base = np.floor(raw).astype(np.int64)
+    rest = int(n - base.sum())
+    if rest > 0:
+        order = np.argsort(-(raw - base), kind="stable")
+        base[order[:rest]] += 1
+    return base
+
+
+def stratified_group_assignment(
+    groups: pd.DataFrame,
+    targets: dict[str, float],
+    seed_seq: np.random.SeedSequence,
+) -> pd.Series:
+    """Детерминированное стратифицированное назначение групп по частям.
+
+    Для каждой L3 независимо: целевые числа строк частей получаются методом
+    наибольших остатков от долей ``targets``; группы упорядочиваются по
+    убыванию размера (порядок равных — засеянной перестановкой из
+    ``seed_seq``) и каждая целиком идёт в часть с максимальным дефицитом
+    ``целевые строки − уже назначенные строки``; равенство разрешается по
+    индексу части. Возвращает Series ``group_id -> part``.
+    """
+    parts = list(targets)
+    w = np.array([targets[p] for p in parts], dtype=float)
+    classes = sorted(groups["L3"].unique())
+    children = seed_seq.spawn(len(classes))
+    assignment: dict[str, str] = {}
+    for l3, child in zip(classes, children):
+        gdf = groups.loc[groups["L3"] == l3]
+        rng = np.random.default_rng(child)
+        perm = rng.permutation(len(gdf))
+        sizes = gdf["n_rows"].to_numpy()
+        order = np.lexsort((perm, -sizes))
+        quota = _largest_remainder(int(sizes.sum()), w)
+        rows_p = np.zeros(len(parts), dtype=np.int64)
+        gids = gdf["group_id"].to_numpy()
+        for gi in order:
+            deficit = quota - rows_p
+            p = int(np.argmax(deficit))
+            rows_p[p] += int(sizes[gi])
+            assignment[gids[gi]] = parts[p]
+    return pd.Series(assignment, name="part")
+
+
+def coverage_failures(support: pd.DataFrame, part_cols: list[str]) -> list[dict]:
+    """Классы, отсутствующие хотя бы в одной из частей."""
+    fails = []
+    for _, row in support.iterrows():
+        missing = [c for c in part_cols if int(row[c]) == 0]
+        if missing:
+            fails.append({"L1": row["L1"], "L2": row["L2"], "L3": row["L3"], "missing_in": missing})
+    return fails
+
+
+def freeze_split(
+    df: pd.DataFrame,
+    min_count: int = 10,
+    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
+    monitor_fraction: float = 0.1,
+    seed: int = 42,
+    share_tolerance: float = 0.05,
+) -> FrozenSplit:
+    """Замораживает сплиты; при невозможности надёжного протокола — FreezeSplitError.
+
+    ``df`` — исходный размеченный датафрейм (только REQUIRED_COLUMNS);
+    исходные позиционные id сохраняются в колонке ``row_id``.
+    """
+    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
+    if missing:
+        raise FreezeSplitError(f"Нет обязательных колонок: {missing}")
+    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
+        raise FreezeSplitError(f"Доли outer-частей должны быть тремя и суммироваться в 1: {fractions}")
+    if not (0.0 < monitor_fraction < 0.5):
+        raise FreezeSplitError(f"monitor_fraction должен быть в (0, 0.5): {monitor_fraction}")
+
+    df = df[list(REQUIRED_COLUMNS)].copy()
+    df[ROW_ID_COLUMN] = np.arange(len(df), dtype=np.int64)
+    df["path"] = df["L1"] + " / " + df["L2"] + " / " + df["L3"]
+
+    validate_rows(df)
+    dfk = add_group_keys(df)
+    check_group_label_consistency(dfk)
+
+    kept, rare = filter_min_count(dfk, min_count, by="L3")
+    if len(kept) == 0:
+        raise FreezeSplitError(f"После min_count={min_count} не осталось строк; отказ.")
+    groups = make_groups(kept)
+
+    seed_seqs = np.random.SeedSequence(seed).spawn(2)
+    targets_outer = dict(zip(PARTS_OUTER, fractions))
+    assignment_outer = stratified_group_assignment(groups, targets_outer, seed_seqs[0])
+    for gid in groups["group_id"]:
+        if gid not in assignment_outer.index:
+            raise FreezeSplitError(f"Группа не распределена: {gid!r}")
+    part_of = dict(assignment_outer)
+
+    # Train-internal monitor: те же группы, заново назначенные внутри outer train.
+    train_groups = groups.loc[groups["group_id"].map(part_of).eq("outer_train")].copy()
+    targets_monitor = {
+        "train_fit": 1.0 - monitor_fraction,
+        "train_monitor": monitor_fraction,
+    }
+    assignment_monitor = stratified_group_assignment(train_groups, targets_monitor, seed_seqs[1])
+    part_of.update(dict(assignment_monitor))
+
+    kept = kept.copy()
+    kept["_part"] = kept["_char_key"].map(part_of)
+    if kept["_part"].isna().any():
+        raise FreezeSplitError("Есть строки без назначенной части; отказ.")
+
+    # outer_train — объединение train_fit и train_monitor (монитор вырезан
+    # из train заново, поэтому отдельной метки outer_train после update нет).
+    part_rows: dict[str, pd.DataFrame] = {
+        "train_fit": kept.loc[kept["_part"] == "train_fit"],
+        "train_monitor": kept.loc[kept["_part"] == "train_monitor"],
+        "comparison_val": kept.loc[kept["_part"] == "comparison_val"],
+        "sealed_test": kept.loc[kept["_part"] == "sealed_test"],
+    }
+    part_rows["outer_train"] = pd.concat(
+        [part_rows["train_fit"], part_rows["train_monitor"]], ignore_index=True
+    )
+    parts: dict[str, pd.DataFrame] = {}
+    for name in PART_NAMES:
+        d = part_rows[name].sort_values(ROW_ID_COLUMN, ignore_index=True)
+        # Служебные колонки ключей остаются в частях для проверок целостности;
+        # в parquet-файлы они не попадают (save_frozen_split отбрасывает).
+        parts[name] = d.reset_index(drop=True)
+
+    excluded = df.loc[~df[ROW_ID_COLUMN].isin(kept[ROW_ID_COLUMN])].copy()
+
+    maps = LabelMaps.from_df(kept[list(REQUIRED_COLUMNS)])
+
+    # --- Контроль стратификации: каждая L3 во всех частях.
+    kept_labels = kept[list(REQUIRED_COLUMNS)]
+    support = (
+        kept_labels.groupby(["L1", "L2", "L3"]).size().reset_index(name="support_all")
+    )
+    for name in PART_NAMES:
+        support[f"support_{name}"] = (
+            parts[name].groupby(["L1", "L2", "L3"]).size().reindex(
+                pd.MultiIndex.from_frame(support[["L1", "L2", "L3"]]),
+                fill_value=0,
+            ).to_numpy()
+        )
+    part_cols = [f"support_{n}" for n in PART_NAMES]
+    fails = coverage_failures(support, part_cols)
+    if fails:
+        raise FreezeSplitError(
+            f"Отказ freeze split: стратификация невозможна — {len(fails)} L3 отсутствуют "
+            f"в частях (group-aware распределение не покрывает все части); примеры: "
+            f"{fails[:10]}. Состав данных не менялся."
+        )
+
+    # --- Контроль долей частей (actual и target всегда от одной базы).
+    n_total = len(kept)
+    share_report: dict[str, dict] = {}
+    for name in PART_NAMES:
+        if name in PARTS_OUTER:
+            target = targets_outer[name]
+            actual = len(parts[name]) / n_total
+            base = "all_kept_rows"
+        else:
+            target = targets_monitor[name]
+            actual = len(parts[name]) / len(parts["outer_train"])
+            base = "outer_train"
+        share_report[name] = {
+            "target": target,
+            "actual": round(actual, 6),
+            "n_rows": len(parts[name]),
+            "base": base,
+        }
+        if abs(actual - target) > share_tolerance:
+            raise FreezeSplitError(
+                f"Отказ freeze split: доля части {name}={actual:.4f} (база {base}) "
+                f"отклоняется от целевой {target} сверх допуска {share_tolerance}. "
+                f"Распределение: {share_report}."
+            )
+
+    # --- Проверки пересечений и полноты учёта строк.
+    checks = integrity_checks(parts, excluded, n_rows_read=len(df))
+
+    report = {
+        "seed": int(seed),
+        "min_count": int(min_count),
+        "fractions": list(fractions),
+        "monitor_fraction": float(monitor_fraction),
+        "share_tolerance": float(share_tolerance),
+        "n_rows_read": int(len(df)),
+        "n_rows_kept": int(n_total),
+        "n_rows_excluded_min_count": int(len(excluded)),
+        "n_l3_kept": int(kept["L3"].nunique()),
+        "n_l3_excluded": int(len(rare)),
+        "n_groups": int(len(groups)),
+        "shares": share_report,
+        "support": support,
+        "rare_classes": rare,
+        "checks": checks,
+    }
+    return FrozenSplit(parts=parts, excluded=excluded, maps=maps, report=report)
+
+
+def integrity_checks(parts: dict[str, pd.DataFrame], excluded: pd.DataFrame, n_rows_read: int) -> dict:
+    """Пересечения нормализованных ключей между частями и учёт row ids.
+
+    Ожидается, что множество row id трёх outer-частей в точности равно
+    ``range(n_rows_read)`` без исключённых min_count строк, а monitor-части
+    в точности разбивают outer_train.
+    """
+    overlap_pairs = [
+        ("outer_train", "comparison_val"),
+        ("outer_train", "sealed_test"),
+        ("comparison_val", "sealed_test"),
+        ("train_fit", "train_monitor"),
+    ]
+    overlaps = {}
+    ok = True
+    for a, b in overlap_pairs:
+        for key in ("_char_key", "_word_key"):
+            ka = set(parts[a][key])
+            kb = set(parts[b][key])
+            shared = ka & kb
+            overlaps[f"{a}|{b}|{key.strip('_')}"] = {
+                "n_shared_keys": len(shared),
+                "shared_examples": sorted(shared)[:5],
+            }
+            if shared:
+                ok = False
+    kept_expected = set(range(n_rows_read)) - set(int(i) for i in excluded[ROW_ID_COLUMN])
+    outer_ids = pd.concat([parts[n][ROW_ID_COLUMN] for n in PARTS_OUTER])
+    train_ids = pd.concat([parts["train_fit"][ROW_ID_COLUMN], parts["train_monitor"][ROW_ID_COLUMN]])
+    accounting = {
+        "outer_parts_cover_kept_exactly_once": bool(
+            outer_ids.is_unique and set(int(i) for i in outer_ids) == kept_expected
+        ),
+        "monitor_parts_cover_outer_train_exactly_once": bool(
+            train_ids.is_unique
+            and set(int(i) for i in train_ids) == set(int(i) for i in parts["outer_train"][ROW_ID_COLUMN])
+        ),
+        "excluded_disjoint_from_parts": bool(
+            set(int(i) for i in excluded[ROW_ID_COLUMN]).isdisjoint(set(int(i) for i in outer_ids))
+        ),
+    }
+    if not all(accounting.values()):
+        ok = False
+    # Инвариант: равные word-ключи обязаны делить char-ключ (обе ветки покрыты).
+    word_char: dict[str, set] = {}
+    for name in PART_NAMES:
+        for wk, ck in zip(parts[name]["_word_key"], parts[name]["_char_key"]):
+            word_char.setdefault(wk, set()).add(ck)
+    word_key_spans_multi_char = {k: sorted(v) for k, v in word_char.items() if len(v) > 1}
+    if word_key_spans_multi_char:
+        ok = False
+    return {
+        "ok": ok,
+        "overlaps": overlaps,
+        "row_accounting": accounting,
+        "word_key_spans_multiple_char_keys": word_key_spans_multi_char,
+    }
+
+
+def _versions() -> dict:
+    import pyarrow
+    import sklearn
+
+    return {
+        "python": platform.python_version(),
+        "numpy": np.__version__,
+        "pandas": pd.__version__,
+        "sklearn": sklearn.__version__,
+        "pyarrow": pyarrow.__version__,
+        "platform": platform.platform(),
+    }
+
+
+def _git_info() -> dict:
+    def _run(*args: str) -> str | None:
+        try:
+            return subprocess.run(
+                ["git", *args], capture_output=True, text=True, timeout=10
+            ).stdout.strip() or None
+        except Exception:
+            return None
+
+    return {"branch": _run("branch", "--show-current"), "head": _run("rev-parse", "HEAD")}
+
+
+def save_frozen_split(
+    frozen: FrozenSplit,
+    input_path: str,
+    out_dir: str,
+    audit_dir: str,
+    min_count: int,
+    fractions: tuple[float, float, float],
+    monitor_fraction: float,
+    seed: int,
+    share_tolerance: float,
+) -> dict:
+    """Сохраняет части, manifest и отчёты; существующие файлы не перезаписывает.
+
+    Возвращает сводку с путями и sha256 записанных файлов.
+    """
+    os.makedirs(out_dir, exist_ok=True)
+    os.makedirs(audit_dir, exist_ok=True)
+
+    part_files = {name: os.path.join(out_dir, f"{name}.parquet") for name in PART_NAMES}
+    manifest_path = os.path.join(out_dir, "manifest.json")
+    split_audit_path = os.path.join(audit_dir, "split_audit.csv")
+    support_path = os.path.join(audit_dir, "class_support_split.csv")
+    targets = list(part_files.values()) + [manifest_path, split_audit_path, support_path]
+    existing = [p for p in targets if os.path.exists(p)]
+    if existing:
+        raise FreezeSplitError(
+            f"Отказ freeze split: существующие outputs не перезаписываются: {existing}"
+        )
+
+    report = frozen.report
+    checks = report["checks"]
+    if not checks["ok"]:
+        raise FreezeSplitError(f"Отказ freeze split: проверки целостности не пройдены: {checks}")
+
+    fingerprints: dict[str, dict] = {}
+    part_meta: dict[str, dict] = {}
+    for name, path in part_files.items():
+        d = frozen.parts[name][list(REQUIRED_COLUMNS) + [ROW_ID_COLUMN]]
+        d.to_parquet(path, index=False)
+        fp = file_fingerprint(path)
+        fingerprints[name] = fp
+        share = report["shares"][name]
+        part_meta[name] = {
+            "file": fp["path"],
+            "sha256": fp["sha256"],
+            "bytes": fp["bytes"],
+            "n_rows": int(len(d)),
+            "n_groups": int(frozen.parts[name]["_char_key"].nunique()),
+            "share": share["actual"],
+            "share_base": share["base"],
+        }
+
+    support = report["support"].copy()
+    support.insert(3, "kept_after_min_count", True)
+    rare = report["rare_classes"]
+    if len(rare):
+        rare_rows = rare.rename(columns={"count": "support_all"})[["L1", "L2", "L3", "support_all"]].copy()
+        for c in [f"support_{n}" for n in PART_NAMES]:
+            rare_rows[c] = 0
+        rare_rows["kept_after_min_count"] = False
+        support = pd.concat([support, rare_rows[SUPPORT_COLUMNS[:-6] + ["kept_after_min_count"] + [f"support_{n}" for n in PART_NAMES]]], ignore_index=True)
+    support = support[SUPPORT_COLUMNS + (["support_all"] if "support_all" in support.columns else [])]
+    support = support.sort_values(
+        ["kept_after_min_count", "support_outer_train", "L3"],
+        ascending=[False, False, True],
+        ignore_index=True,
+    )
+    support.to_csv(support_path, index=False)
+
+    input_fp = file_fingerprint(input_path)
+    audit_rows = []
+    for name in PART_NAMES:
+        d = frozen.parts[name]
+        sup = report["support"]
+        shared_char = sum(
+            v["n_shared_keys"] for k, v in checks["overlaps"].items()
+            if k.endswith("char_key") and name in k.split("|")[:2]
+        )
+        shared_word = sum(
+            v["n_shared_keys"] for k, v in checks["overlaps"].items()
+            if k.endswith("word_key") and name in k.split("|")[:2]
+        )
+        audit_rows.append({
+            "part": name,
+            "n_rows": int(len(d)),
+            "share_target": report["shares"][name]["target"],
+            "share_actual": report["shares"][name]["actual"],
+            "n_groups": int(d["_char_key"].nunique()),
+            "n_l3": int(d["L3"].nunique()),
+            "l3_min_support": int(sup[f"support_{name}"].min()),
+            "l3_max_support": int(sup[f"support_{name}"].max()),
+            "char_keys_shared_with_other_parts": int(shared_char),
+            "word_keys_shared_with_other_parts": int(shared_word),
+            "n_row_ids_unique": int(d[ROW_ID_COLUMN].nunique()),
+            "sha256": fingerprints[name]["sha256"],
+            "bytes": fingerprints[name]["bytes"],
+        })
+    audit_rows.append({
+        "part": "excluded_by_min_count",
+        "n_rows": int(report["n_rows_excluded_min_count"]),
+        "share_target": None,
+        "share_actual": round(report["n_rows_excluded_min_count"] / report["n_rows_read"], 6),
+        "n_groups": None,
+        "n_l3": int(report["n_l3_excluded"]),
+        "l3_min_support": None,
+        "l3_max_support": None,
+        "char_keys_shared_with_other_parts": None,
+        "word_keys_shared_with_other_parts": None,
+        "n_row_ids_unique": int(frozen.excluded[ROW_ID_COLUMN].nunique()),
+        "sha256": None,
+        "bytes": None,
+    })
+    split_audit = pd.DataFrame(audit_rows, columns=SPLIT_AUDIT_COLUMNS)
+    split_audit.to_csv(split_audit_path, index=False)
+
+    manifest = {
+        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
+        "generator": {
+            "script": os.path.abspath(sys.argv[0] if sys.argv and sys.argv[0] else __file__),
+            "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]) if sys.argv else None,
+            "cwd": os.getcwd(),
+            "git": _git_info(),
+            "versions": _versions(),
+        },
+        "policy": {
+            "min_count": int(min_count),
+            "fractions": list(fractions),
+            "monitor_fraction": float(monitor_fraction),
+            "split_seed": int(seed),
+            "share_tolerance": float(share_tolerance),
+            "group_key": "strip_whitespace(preprocess(name)) — char-ветка TF-IDF; "
+                        "word-ключ (collapse whitespace) равен ⇒ char-ключ равен, "
+                        "поэтому группа покрывает обе ветки",
+            "row_id": "позиционный индекс строки во входном parquet (0-based, порядок файла)",
+            "parts": {
+                "outer_train": "обучение (fit) и внутренний выбор эпох/конфигурации только по train_fit/train_monitor",
+                "comparison_val": "сравнение методов S06; не для выбора checkpoint",
+                "sealed_test": "запечатан; метрики качества не вычислялись, count/hash/support только",
+                "train_fit": "внутренний fit внутри outer train",
+                "train_monitor": "внутренний monitor внутри outer train (early stopping/выбор эпох)",
+            },
+        },
+        "inputs": {"labeled_parquet": input_fp},
+        "label_maps": frozen.maps.to_dict(),
+        "n_classes": {lvl: frozen.maps.n_classes(lvl) for lvl in LEVELS},
+        "parts": part_meta,
+        "excluded_min_count": {
+            "n_rows": int(len(frozen.excluded)),
+            "n_l3": int(report["n_l3_excluded"]),
+            "row_ids": [int(i) for i in frozen.excluded[ROW_ID_COLUMN]],
+        },
+        "checks": checks,
+        "notes": [
+            "Метрики качества не вычислялись; sealed_test не использовался ни для чего, кроме count/hash/support.",
+            "Исключённые min_count строки перечислены по row_id выше; молчаливого удаления нет.",
+            "Пересечения нормализованных ключей между частями проверены для пар "
+            "outer×outer и train_fit×train_monitor (обе TF-IDF-ветки), все нули.",
+        ],
+    }
+    with open(manifest_path, "w", encoding="utf-8") as f:
+        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
+    manifest_fp = file_fingerprint(manifest_path)
+
+    return {
+        "out_dir": os.path.abspath(out_dir),
+        "audit_dir": os.path.abspath(audit_dir),
+        "parts": part_meta,
+        "manifest": manifest_fp,
+        "split_audit": {"path": os.path.abspath(split_audit_path)},
+        "class_support_split": {"path": os.path.abspath(support_path)},
+        "n_rows_kept": report["n_rows_kept"],
+        "n_rows_excluded_min_count": report["n_rows_excluded_min_count"],
+        "checks_ok": checks["ok"],
+    }
diff --git a/kkt_clf/s05_baseline.py b/kkt_clf/s05_baseline.py
new file mode 100644
index 0000000..59bc2f4
--- /dev/null
+++ b/kkt_clf/s05_baseline.py
@@ -0,0 +1,1012 @@
+"""S05 (D04–D08): контур обучения baseline MLP на едином полном TF-IDF.
+
+Реализует обучающий контур спринта S05 поверх замороженных сплитов
+(``outputs/S05/splits/manifest.json``, см. ``kkt_clf.freeze_split``) и одного
+проверенного сохранённого full-TFIDF S02 (по умолчанию ``outputs/S02/tfidf_1m``).
+Feature selection не выполняется; режим векторизатора — только ``pretrained``.
+
+Протокол (границы частей):
+
+- ``train_fit`` / ``train_monitor`` — единственные данные для выбора числа эпох
+  и конфигурации (фаза selection, D05): MLP обучается на train_fit с early
+  stopping по train_monitor, кривые train/monitor сохраняются.
+- ``outer_train`` — финальный refit (D06/D08): отдельное обучение на всём
+  outer_train ровно на зафиксированном числе эпох с фиксированным расписанием
+  LR; checkpoint выбора по monitor **нет** (``restore_best=False`` — веса
+  финальной эпохи). Comparison validation до завершения всех refit не
+  трансформируется и не оценивается; sealed test не используется ни разу.
+- ``comparison_val`` — одна оценка raw top-1 L3 без отказа **после** refit
+  каждого seed: accuracy и macro-F1 по фиксированной L3-вселенной label maps
+  из manifest; классы с support=0 дают null, а не 0. Пустые векторы признаков
+  считаются отдельно. Для контроля считается тривиальный majority-class
+  baseline (большинство по outer train) на том же comparison_val.
+- Опорный checkpoint для S06 (D08) выбирается **заранее заданным детерминированным
+  train-only правилом** — ``reference_seed`` (по умолчанию 42), не по качеству
+  на comparison_val; выполняется round-trip проверка загрузки.
+
+Все результаты пишутся только в новые файлы под ``output_dir`` (по умолчанию
+``outputs/S05``): существующие файлы и каталоги не перезаписываются, при
+обнаружении пересечения — отказ до любых записей. Крупные выходы (checkpoints)
+— вне Git, каталог настраивается.
+
+Реализация переиспользует существующий цикл обучения (``train.train_model``),
+артефакты (``artifact.save_artifact`` / ``load_artifact``), замеры ресурсов и
+метрики raw top-1 (``experiment._Measurement``, ``experiment._l3_val_metrics``);
+тренировочный цикл не дублируется и не переписывается.
+"""
+from __future__ import annotations
+
+import dataclasses
+import datetime as dt
+import json
+import logging
+import os
+import platform
+from dataclasses import dataclass, field
+from typing import Any, Sequence
+
+import numpy as np
+import pandas as pd
+import pyarrow.parquet as pq
+import scipy.sparse as sp
+import torch
+
+from .artifact import environment_versions, load_artifact, save_artifact
+from .config import HEADS, LossConfig, ModelConfig, TrainConfig, _merge, build_config
+from .data import file_fingerprint
+from .experiment import _Measurement, _git_info, _l3_val_metrics, _sha256_file, _sha256_json, _write_json
+from .features import matrix_stats
+from .labels import LEVELS, LabelMaps
+from .model import SparseMLP
+from .reject import RejectRule
+from .train import compute_class_weights, predict, resolve_device, set_seed, train_model
+from .vectorizer import LoadedVectorizer, load_vectorizer_artifact
+
+log = logging.getLogger(__name__)
+
+S05_SCHEMA_VERSION = "s05-training@1"
+
+PROTOCOL = {
+    "selection": (
+        "train-only выбор числа эпох/конфигурации: fit на train_fit, ранний "
+        "останов/выбор best epoch по train_monitor (внутри outer_train)"
+    ),
+    "refit": (
+        "финальный refit на всём outer_train: фиксированное число эпох из "
+        "selection (или --fixed-epochs), фиксированное расписание LR, веса "
+        "финальной эпохи без выбора checkpoint (restore_best=False)"
+    ),
+    "evaluation": (
+        "одна оценка raw top-1 L3 без отказа на comparison_val после refit "
+        "всех seed; macro-F1 по классам с support>0 фиксированной L3-вселенной "
+        "label maps; support=0 → null; τ/отказ не применяются"
+    ),
+    "checkpoint_rule": (
+        "опорный checkpoint для S06 — заранее заданный reference_seed (train-only "
+        "правило), не по качеству на comparison_val; round-trip проверка загрузки"
+    ),
+    "boundaries": (
+        "comparison_val не используется до завершения всех refit и никогда для "
+        "выбора checkpoint; sealed_test не используется ни для чего (ни метрики, "
+        "ни выбор); метки val/test не участвуют в fit TF-IDF (pretrained)"
+    ),
+    "transductive": (
+        "режим pretrained трансдуктивен: тексты корпуса full TF-IDF могут "
+        "совпадать с train/val/test; метки при этом не переносятся"
+    ),
+}
+
+RUNS_COLUMNS = [
+    "run_id", "config_id", "train_seed", "split_seed", "status", "reason",
+    "n_train", "n_monitor", "n_val", "n_classes", "full_tfidf_id",
+    "epochs_completed", "best_epoch", "l3_macro_f1", "l3_accuracy",
+    "metric_evaluated_on", "train_seconds", "peak_rss_bytes", "checkpoint_path",
+]
+STAGES_COLUMNS = [
+    "run_id", "stage", "status", "seconds", "peak_rss_bytes", "peak_vram_bytes",
+    "measurement_note",
+]
+HISTORY_COLUMNS = [
+    "run_id", "epoch", "split", "loss", "l3_macro_f1", "l3_accuracy", "lr",
+]
+PER_CLASS_COLUMNS = ["run_id", "L3", "support", "precision", "recall", "f1"]
+SHARED_VECTORIZATION_RUN_ID = "shared_vectorization"
+
+
+class S05BaselineError(RuntimeError):
+    """Базовая ошибка контура обучения S05; файлы не записываются."""
+
+
+@dataclass
+class S05Config:
+    """Конфигурация контура обучения S05.
+
+    ``preset`` — базовая MLP/loss/train-конфигурация (по умолчанию ``ods_light``
+    как ориентир относительно модели; источник признаков — только полный
+    TF-IDF). ``model``/``loss``/``train`` — рекурсивные переопределения пресета.
+    ``sweep`` — ограниченный список кандидатов фазы selection
+    (``[{"name": ..., "model": {...}, "train": {...}}]``); выбор кандидата —
+    только по train_monitor, каждый кандидат фиксируется в отчёте selection.
+    """
+
+    output_dir: str = "outputs/S05"
+    splits_manifest: str = "outputs/S05/splits/manifest.json"
+    full_tfidf_path: str = "outputs/S02/tfidf_1m"
+    corpus_path: str = "data/corpus.pqt"
+    preset: str = "ods_light"
+    model: dict[str, Any] = field(default_factory=dict)
+    loss: dict[str, Any] = field(default_factory=dict)
+    train: dict[str, Any] = field(default_factory=dict)
+    sweep: list[dict[str, Any]] = field(default_factory=list)
+    features_chunk_rows: int = 50_000
+    selection_seed: int = 42
+    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
+    reference_seed: int = 42
+    fixed_epochs: int | None = None
+    device: str = "auto"
+
+    def build_sections(self, model_over: dict | None = None,
+                       train_over: dict | None = None) -> tuple[ModelConfig, LossConfig, TrainConfig]:
+        base = build_config(self.preset, {})
+        model_merged = {**self.model, **(model_over or {})}
+        train_merged = {**self.train, **(train_over or {})}
+        model_cfg = _merge(base.model, model_merged)
+        loss_cfg = _merge(base.loss, self.loss)
+        train_cfg = _merge(base.train, train_merged)
+        return model_cfg, loss_cfg, train_cfg
+
+    def validate(self) -> None:
+        if self.reference_seed not in self.seeds:
+            raise S05BaselineError(
+                f"reference_seed={self.reference_seed} должен входить в seeds={self.seeds}: "
+                "опорный checkpoint выбирается заранее заданным правилом и обязан быть обучен"
+            )
+        if len(set(self.seeds)) != len(self.seeds):
+            raise S05BaselineError(f"seeds должны быть различны: {self.seeds}")
+        if self.features_chunk_rows <= 0:
+            raise S05BaselineError("features_chunk_rows: > 0")
+        for cand in self.sweep:
+            if not isinstance(cand, dict) or not isinstance(cand.get("name"), str) or not cand["name"]:
+                raise S05BaselineError(f"sweep: каждый кандидат — объект с непустым 'name': {cand!r}")
+        if self.fixed_epochs is not None and self.fixed_epochs < 1:
+            raise S05BaselineError("fixed_epochs: None или >= 1")
+
+    def to_dict(self) -> dict[str, Any]:
+        return dataclasses.asdict(self)
+
+
+# ---------------------------------------------------------------------------
+# Загрузка и проверка входов (D04)
+# ---------------------------------------------------------------------------
+
+
+def load_split_manifest(path: str) -> dict[str, Any]:
+    """Читает manifest замороженного сплита и сверяет части по sha256/числу строк.
+
+    Fail closed: несовпадение хэша/счётчика, отсутствие файла или обязательных
+    секций — S05BaselineError, работа не начинается.
+    """
+    if not os.path.isfile(path):
+        raise S05BaselineError(f"manifest сплитов не найден: {path!r}")
+    with open(path, encoding="utf-8") as f:
+        manifest = json.load(f)
+    for key in ("parts", "label_maps", "policy", "inputs"):
+        if key not in manifest:
+            raise S05BaselineError(f"manifest {path!r}: нет обязательной секции {key!r}")
+    for name, part in manifest["parts"].items():
+        fp = part.get("file")
+        if not fp or not os.path.isfile(fp):
+            raise S05BaselineError(f"часть сплита {name!r}: файл не найден: {fp!r}")
+        actual = file_fingerprint(fp)
+        if actual["sha256"] != part.get("sha256") or actual["bytes"] != part.get("bytes"):
+            raise S05BaselineError(
+                f"часть сплита {name!r}: sha256/bytes изменились (ожидалось "
+                f"{str(part.get('sha256'))[:16]}…/{part.get('bytes')}, фактически "
+                f"{actual['sha256'][:16]}…/{actual['bytes']}): сплит не тот, что заморожен"
+            )
+        n_actual = pq.ParquetFile(fp).metadata.num_rows
+        if n_actual != part.get("n_rows"):
+            raise S05BaselineError(
+                f"часть сплита {name!r}: число строк {n_actual} != зафиксированному {part.get('n_rows')}"
+            )
+    return manifest
+
+
+def load_part(manifest: dict[str, Any], name: str) -> pd.DataFrame:
+    return pd.read_parquet(manifest["parts"][name]["file"])
+
+
+def verify_full_tfidf(full_tfidf_path: str, corpus_path: str) -> dict[str, Any]:
+    """D04: загружает единый full-TFIDF с полной проверкой целостности и provenance.
+
+    Проверки (отказ при любом нарушении, молчаливая подмена недопустима):
+    стандартная верификация артефакта (метаданные, контрольные суммы файлов,
+    fit-отчёт, словари, IDF) через ``load_vectorizer_artifact``; фактический
+    корпус по ``corpus_path`` совпадает с зафиксированным источником по
+    sha256/bytes. Возвращает сводку с идентификатором артефакта
+    (sha256 features.joblib) для ссылок из runs/manifest.
+    """
+    loaded = load_vectorizer_artifact(full_tfidf_path)
+    meta = loaded.meta
+    if not os.path.isfile(corpus_path):
+        raise S05BaselineError(f"D04: корпус full TF-IDF не найден: {corpus_path!r}")
+    corpus_fp = file_fingerprint(corpus_path)
+    src = meta.get("source") or {}
+    corpus_matches = corpus_fp["sha256"] == src.get("sha256") and corpus_fp["bytes"] == src.get("bytes")
+    if not corpus_matches:
+        raise S05BaselineError(
+            f"D04: корпус {corpus_path!r} не совпадает с источником full TF-IDF "
+            f"(sha256/bytes: ожидалось {str(src.get('sha256'))[:16]}…/{src.get('bytes')}, "
+            f"фактически {corpus_fp['sha256'][:16]}…/{corpus_fp['bytes']}): артефакт "
+            "несовместим или корпус подменён, обучение не начинается."
+        )
+    artifact_id = (meta.get("files") or {}).get("features.joblib", {}).get("sha256")
+    return {
+        "schema_version": S05_SCHEMA_VERSION,
+        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
+        "artifact_dir": os.path.abspath(full_tfidf_path),
+        "full_tfidf_id": artifact_id,
+        "artifact_version": meta.get("artifact_version"),
+        "seed": meta.get("seed"),
+        "vocab": meta.get("vocab"),
+        "fit": meta.get("fit"),
+        "source": {k: v for k, v in src.items()},
+        "corpus_fingerprint": corpus_fp,
+        "corpus_matches_source": True,
+        "files": {
+            name: {"sha256": _sha256_file(os.path.join(full_tfidf_path, name)),
+                   "bytes": os.path.getsize(os.path.join(full_tfidf_path, name))}
+            for name in ("features.joblib", "vectorizer_meta.json")
+            if os.path.exists(os.path.join(full_tfidf_path, name))
+        },
+        "mode": "pretrained (fit не выполняется, transform без повторного обучения)",
+        "decision": "ok",
+        "notes": [
+            "Используется один сохранённый артефакт для всех seed/прогонов S05.",
+            PROTOCOL["transductive"],
+        ],
+    }
+
+
+# ---------------------------------------------------------------------------
+# Общие helpers
+# ---------------------------------------------------------------------------
+
+
+def _check_new(paths: Sequence[str]) -> None:
+    """Отказ, если любой целевой путь уже существует (ничего не перезаписываем)."""
+    existing = [p for p in paths if os.path.exists(p)]
+    if existing:
+        raise S05BaselineError(
+            f"Существующие outputs не перезаписываются; отказ до любых записей: {existing}"
+        )
+
+
+def _encode_labels(maps: LabelMaps, df: pd.DataFrame, part: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
+    out = []
+    for lvl in LEVELS:
+        ids = maps.encode(lvl, df[lvl].to_numpy(dtype=object))
+        if (ids < 0).any():
+            bad = sorted(set(df.loc[ids < 0, lvl].astype(str)))[:5]
+            raise S05BaselineError(
+                f"Часть {part!r}: метки вне зафиксированной вселенной {lvl}: {bad}"
+            )
+        out.append(ids)
+    return tuple(out)  # type: ignore[return-value]
+
+
+def _transform(loaded: LoadedVectorizer, names: list[str], chunk_rows: int) -> sp.csr_matrix:
+    return loaded.builder.transform(names, chunk_rows=chunk_rows)
+
+
+def _empty_rows(X: sp.csr_matrix) -> int:
+    return int((np.diff(X.indptr) == 0).sum())
+
+
+def _reject_stub() -> dict[str, Any]:
+    return {
+        "note": "S05: τ не калибровался; сравнение по raw top-1 на comparison_val без отказа",
+        "tau_mode": "fixed",
+    }
+
+
+def _history_rows(run_id: str, history: list[dict]) -> list[dict[str, Any]]:
+    """История эпох train_model → длинный формат train/monitor кривых."""
+    rows: list[dict[str, Any]] = []
+    for rec in history:
+        rows.append({
+            "run_id": run_id, "epoch": int(rec["epoch"]), "split": "train",
+            "loss": rec["train_loss"], "l3_macro_f1": None, "l3_accuracy": None,
+            "lr": rec["lr"],
+        })
+        rows.append({
+            "run_id": run_id, "epoch": int(rec["epoch"]), "split": "monitor",
+            "loss": rec["val_loss"], "l3_macro_f1": rec["val_l3_macro_f1"],
+            "l3_accuracy": rec["val_l3_accuracy"], "lr": rec["lr"],
+        })
+    return rows
+
+
+# ---------------------------------------------------------------------------
+# D05: train-only выбор числа эпох/конфигурации (train_fit / train_monitor)
+# ---------------------------------------------------------------------------
+
+
+def run_selection(
+    cfg: S05Config,
+    manifest: dict[str, Any],
+    loaded: LoadedVectorizer,
+    device: torch.device,
+) -> dict[str, Any]:
+    """Фаза selection: кандидаты обучаются на train_fit с early stopping по
+    train_monitor; выбранное число эпох — best_epoch лучшего кандидата по
+    метрике train.* (по умолчанию val_l3_macro_f1 на train_monitor).
+
+    Записывает ``selection/selection_report.json`` и
+    ``selection/curves_train_monitor.csv`` (оба — только новые файлы).
+    Возвращает сводку для последующего refit.
+    """
+    cfg.validate()
+    maps = LabelMaps.from_dict(manifest["label_maps"])
+    fit_df = load_part(manifest, "train_fit")
+    mon_df = load_part(manifest, "train_monitor")
+    y_fit = _encode_labels(maps, fit_df, "train_fit")
+    y_mon = _encode_labels(maps, mon_df, "train_monitor")
+
+    out_dir = cfg.output_dir
+    sel_dir = os.path.join(out_dir, "selection")
+    report_path = os.path.join(sel_dir, "selection_report.json")
+    curves_path = os.path.join(sel_dir, "curves_train_monitor.csv")
+    _check_new([report_path, curves_path])
+    os.makedirs(sel_dir, exist_ok=True)
+
+    stage_rows: list[dict[str, Any]] = []
+    with _Measurement(device, f"transform train_fit/train_monitor полным TF-IDF, chunk_rows={cfg.features_chunk_rows}") as m:
+        X_fit = _transform(loaded, fit_df["name"].tolist(), cfg.features_chunk_rows)
+        X_mon = _transform(loaded, mon_df["name"].tolist(), cfg.features_chunk_rows)
+    stage_rows.append(m.stage_row("selection", "vectorize", "success"))
+    log.info(
+        "Selection: train_fit %d строк (пустых %d), train_monitor %d (пустых %d), признаков %d",
+        X_fit.shape[0], _empty_rows(X_fit), X_mon.shape[0], _empty_rows(X_mon), X_fit.shape[1],
+    )
+
+    candidates = [{"name": "base", "model": {}, "train": {}}] + [dict(c) for c in cfg.sweep]
+    names = [c["name"] for c in candidates]
+    if len(set(names)) != len(names):
+        raise S05BaselineError(f"Имена кандидатов selection должны быть различны: {names}")
+
+    results: list[dict[str, Any]] = []
+    curves: list[dict[str, Any]] = []
+    for cand in candidates:
+        model_cfg, loss_cfg, train_cfg = cfg.build_sections(cand.get("model"), cand.get("train"))
+        metric = train_cfg.early_stopping_metric
+        set_seed(cfg.selection_seed)
+        model = SparseMLP(
+            loaded.builder.n_features,
+            tuple(maps.n_classes(lvl) for lvl in LEVELS),
+            model_cfg,
+        )
+        class_weights = {
+            h: compute_class_weights(y_fit[i], maps.n_classes(lvl), loss_cfg.class_weight_mode[h],
+                                     loss_cfg.class_weight_alpha)
+            for i, (h, lvl) in enumerate(zip(HEADS, LEVELS))
+        }
+        run_id = f"selection:{cand['name']}"
+        with _Measurement(
+            device,
+            f"обучение кандидата {cand['name']!r} на train_fit (n={X_fit.shape[0]}), "
+            f"early stopping по train_monitor, метрика {metric}; выбор только по train-internal monitor",
+        ) as meas:
+            history, best, lr_trace = train_model(
+                model, X_fit, y_fit, X_mon, y_mon,
+                loss_cfg, train_cfg, class_weights, device, cfg.selection_seed,
+            )
+        stage_rows.append(meas.stage_row(run_id, "selection_train", "success"))
+        curves.extend(_history_rows(run_id, history))
+        last = history[-1] if history else {}
+        results.append({
+            "name": cand["name"],
+            "model": dataclasses.asdict(model_cfg),
+            "train": dataclasses.asdict(train_cfg),
+            "loss": dataclasses.asdict(loss_cfg),
+            "epochs_run": len(history),
+            "best_epoch": int(best["epoch"]),
+            "best_score": float(best["score"]),
+            "early_stopping_metric": metric,
+            "final_monitor_l3_macro_f1": last.get("val_l3_macro_f1"),
+            "final_monitor_l3_accuracy": last.get("val_l3_accuracy"),
+            "train_seconds": meas.seconds,
+            "peak_rss_bytes": meas.peak_rss_bytes,
+            "peak_vram_bytes": meas.peak_vram_bytes,
+            "n_train_rows_used": int(best.get("n_train_rows_used", 0)),
+        })
+        log.info(
+            "Кандидат %s: best_epoch=%d (%s=%.4f), эпох выполнено %d",
+            cand["name"], best["epoch"], metric, best["score"], len(history),
+        )
+
+    higher_better = metric != "val_loss"
+    chosen = max(results, key=lambda r: r["best_score"]) if higher_better else min(
+        results, key=lambda r: r["best_score"]
+    )
+    report = {
+        "schema_version": S05_SCHEMA_VERSION,
+        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
+        "protocol": PROTOCOL["selection"],
+        "metric": metric,
+        "higher_better": higher_better,
+        "selection_seed": int(cfg.selection_seed),
+        "candidates": results,
+        "chosen": {
+            "name": chosen["name"],
+            "fixed_epochs": int(chosen["best_epoch"]),
+            "criterion": f"best {metric} на train_monitor (train-only; comparison_val не использовался)",
+        },
+        "splits": {
+            "train_fit": manifest["parts"]["train_fit"]["sha256"],
+            "train_monitor": manifest["parts"]["train_monitor"]["sha256"],
+        },
+        "full_tfidf_id": (loaded.meta.get("files") or {}).get("features.joblib", {}).get("sha256"),
+        "stages": stage_rows,
+    }
+    _write_json(report, report_path)
+    pd.DataFrame(curves, columns=HISTORY_COLUMNS).to_csv(curves_path, index=False)
+    log.info("Selection: выбран кандидат %s, фиксированные эпохи %d", chosen["name"], chosen["best_epoch"])
+    return report
+
+
+def read_selection(output_dir: str) -> dict[str, Any]:
+    path = os.path.join(output_dir, "selection", "selection_report.json")
+    if not os.path.isfile(path):
+        raise S05BaselineError(
+            f"Нет {path!r}: сначала выполните фазу selection (train-only выбор эпох/конфигурации)"
+        )
+    with open(path, encoding="utf-8") as f:
+        return json.load(f)
+
+
+# ---------------------------------------------------------------------------
+# D06–D08: финальный refit на outer_train (seed 42/43/44) + оценка + checkpoint
+# ---------------------------------------------------------------------------
+
+
+def run_refit(
+    cfg: S05Config,
+    manifest: dict[str, Any],
+    loaded: LoadedVectorizer,
+    device: torch.device,
+    selection: dict[str, Any] | None = None,
+) -> dict[str, Any]:
+    """Финальный контур refit+оценка (D06–D08).
+
+    Порядок гарантирует границы протокола: сначала transform outer_train и
+    обучение всех seed ровно на фиксированное число эпох (comparison_val до
+    этого момента не читается и не оценивается), затем одна оценка raw top-1
+    каждого seed на comparison_val, majority baseline и выбор опорного
+    checkpoint по правилу reference_seed с round-trip проверкой.
+
+    Записывает (только новые файлы): ``checkpoints/seed_<s>/`` (артефакты
+    save_artifact по каждому seed), ``refit/history_seed_<s>.csv``,
+    ``refit/run_result_seed_<s>.json``, ``runs.csv``, ``per_class.parquet``,
+    ``history.csv``, ``stages.csv``, ``majority_baseline.json``,
+    ``selected_checkpoint.json``, ``manifest.json``.
+    """
+    cfg.validate()
+    maps = LabelMaps.from_dict(manifest["label_maps"])
+    split_seed = int(manifest["policy"]["split_seed"])
+    chosen = (selection or {}).get("chosen") or {}
+    fixed_epochs = int(cfg.fixed_epochs or chosen.get("fixed_epochs") or 0)
+    if fixed_epochs < 1:
+        raise S05BaselineError(
+            "Фиксированное число эпох не задано: передайте fixed_epochs в конфиге "
+            "или выполните фазу selection (selection/selection_report.json)"
+        )
+    if selection is not None and cfg.fixed_epochs is None:
+        cand = next(
+            (c for c in selection["candidates"] if c["name"] == chosen["name"]), None
+        )
+        if cand is None:
+            raise S05BaselineError(f"chosen-кандидат {chosen['name']!r} не найден в selection")
+        model_over, train_over = cand["model"], cand["train"]
+    else:
+        model_over, train_over = {}, {}
+    model_cfg, loss_cfg, train_cfg = cfg.build_sections(model_over, train_over)
+    config_id = chosen.get("name", "base")
+    log.info(
+        "Refit: config_id=%s, фиксированные эпохи=%d, seed=%s, reference_seed=%d",
+        config_id, fixed_epochs, cfg.seeds, cfg.reference_seed,
+    )
+
+    train_df = load_part(manifest, "outer_train")
+    mon_df = load_part(manifest, "train_monitor")
+    y_train = _encode_labels(maps, train_df, "outer_train")
+    y_mon = _encode_labels(maps, mon_df, "train_monitor")
+    l3_labels = list(maps.l3)
+    n_classes = tuple(maps.n_classes(lvl) for lvl in LEVELS)
+
+    out_dir = cfg.output_dir
+    ckpt_root = os.path.join(out_dir, "checkpoints")
+    refit_dir = os.path.join(out_dir, "refit")
+    targets = [
+        os.path.join(out_dir, "runs.csv"),
+        os.path.join(out_dir, "per_class.parquet"),
+        os.path.join(out_dir, "history.csv"),
+        os.path.join(out_dir, "stages.csv"),
+        os.path.join(out_dir, "majority_baseline.json"),
+        os.path.join(out_dir, "selected_checkpoint.json"),
+        os.path.join(out_dir, "manifest.json"),
+    ]
+    for s in cfg.seeds:
+        targets.append(os.path.join(refit_dir, f"history_seed_{s}.csv"))
+        targets.append(os.path.join(refit_dir, f"run_result_seed_{s}.json"))
+        targets.append(os.path.join(ckpt_root, f"seed_{s}"))
+    _check_new(targets)
+    os.makedirs(refit_dir, exist_ok=True)
+
+    stage_rows: list[dict[str, Any]] = []
+    run_rows: list[dict[str, Any]] = []
+    per_class_frames: list[pd.DataFrame] = []
+    history_rows: list[dict[str, Any]] = []
+    run_results: dict[int, dict[str, Any]] = {}
+    class_weights = {
+        h: compute_class_weights(y_train[i], n_classes[i], loss_cfg.class_weight_mode[h],
+                                 loss_cfg.class_weight_alpha)
+        for i, h in enumerate(HEADS)
+    }
+    cw_json = {
+        "mode": loss_cfg.class_weight_mode,
+        "alpha": float(loss_cfg.class_weight_alpha),
+        "weights": {h: (None if w is None else w.tolist()) for h, w in class_weights.items()},
+        "train_counts": {
+            lvl: train_df[lvl].value_counts().sort_index().astype(int).to_dict() for lvl in LEVELS
+        },
+    }
+
+    # --- transform обучающих частей (comparison_val НЕ трогаем до конца refit)
+    with _Measurement(device, f"transform outer_train/train_monitor полным TF-IDF, chunk_rows={cfg.features_chunk_rows}") as m:
+        X_train = _transform(loaded, train_df["name"].tolist(), cfg.features_chunk_rows)
+        X_mon = _transform(loaded, mon_df["name"].tolist(), cfg.features_chunk_rows)
+    stage_rows.append(m.stage_row(SHARED_VECTORIZATION_RUN_ID, "vectorize_train", "success"))
+    log.info(
+        "Refit: outer_train %d строк (пустых %d), monitor %d (пустых %d), признаков %d",
+        X_train.shape[0], _empty_rows(X_train), X_mon.shape[0], _empty_rows(X_mon), X_train.shape[1],
+    )
+
+    # --- refit каждого seed на фиксированное число эпох
+    for seed in cfg.seeds:
+        run_id = f"seed_{seed}"
+        meas: _Measurement | None = None
+        try:
+            set_seed(seed)
+            model = SparseMLP(loaded.builder.n_features, n_classes, model_cfg)
+            fixed_cfg = dataclasses.replace(train_cfg, epochs=fixed_epochs, patience=fixed_epochs)
+            with _Measurement(
+                device,
+                f"seed {seed}: финальный refit на outer_train (n={X_train.shape[0]}), ровно "
+                f"{fixed_epochs} эпох, scheduler={fixed_cfg.scheduler}, restore_best=False "
+                "(веса финальной эпохи; comparison_val не используется)",
+            ) as meas:
+                history, best, lr_trace = train_model(
+                    model, X_train, y_train, X_mon, y_mon,
+                    loss_cfg, fixed_cfg, class_weights, device, seed,
+                    restore_best=False,
+                )
+            stage_rows.append(meas.stage_row(run_id, "mlp_train", "success"))
+            ckpt_dir = os.path.join(ckpt_root, run_id)
+            save_artifact(
+                ckpt_dir, model, model_cfg, loaded.builder, maps,
+                RejectRule(0.5, False, False), _reject_stub(), cw_json,
+                {
+                    "experiment_run": run_id,
+                    "role": "s05_final_refit",
+                    "config": {
+                        "preset": cfg.preset, "model": dataclasses.asdict(model_cfg),
+                        "loss": dataclasses.asdict(loss_cfg), "train": dataclasses.asdict(fixed_cfg),
+                    },
+                },
+                {
+                    "seed": int(seed),
+                    "split_seed": int(split_seed),
+                    "n_features": int(loaded.builder.n_features),
+                    "device": str(device),
+                    "protocol": S05_SCHEMA_VERSION,
+                    "train": {
+                        "epochs_completed": len(history),
+                        "fixed_epochs": int(fixed_epochs),
+                        "best_epoch_monitor": int(best["epoch"]),
+                        "early_stopping_metric": best["metric"],
+                        "best_score_monitor": float(best["score"]),
+                        "restored_best": bool(best.get("restored_best")),
+                        "n_train_rows_used": int(best.get("n_train_rows_used", 0)),
+                        "final_monitor_l3_macro_f1": (history[-1]["val_l3_macro_f1"] if history else None),
+                        "final_monitor_l3_accuracy": (history[-1]["val_l3_accuracy"] if history else None),
+                    },
+                    "feature_stats": {
+                        "outer_train": matrix_stats(X_train),
+                        "train_monitor": matrix_stats(X_mon),
+                    },
+                    "train_fingerprints": {
+                        "outer_train_sha256": manifest["parts"]["outer_train"]["sha256"],
+                        "train_monitor_sha256": manifest["parts"]["train_monitor"]["sha256"],
+                    },
+                    "full_tfidf": {
+                        "path": os.path.abspath(cfg.full_tfidf_path),
+                        "artifact_version": loaded.meta.get("artifact_version"),
+                        "full_tfidf_id": (loaded.meta.get("files") or {})
+                        .get("features.joblib", {}).get("sha256"),
+                    },
+                    "selection": {
+                        "selection_seed": int(cfg.selection_seed),
+                        "config_id": config_id,
+                        "criterion": chosen.get("criterion"),
+                    },
+                },
+            )
+            history_rows.extend(_history_rows(run_id, history))
+            run_results[seed] = {
+                "run_id": run_id, "seed": seed, "status": "trained",
+                "checkpoint_dir": os.path.abspath(ckpt_dir),
+                "history": history, "best": best,
+                "train_seconds": meas.seconds,
+                "peak_rss_bytes": meas.peak_rss_bytes,
+                "peak_vram_bytes": meas.peak_vram_bytes,
+                "lr_trace": lr_trace,
+            }
+            log.info(
+                "seed %d: refit завершён (%d эпох, %.1f с), checkpoint %s",
+                seed, len(history), meas.seconds or 0.0, ckpt_dir,
+            )
+        except Exception as e:  # noqa: BLE001 — фиксируем failed seed, остальные продолжают
+            log.exception("seed %d: refit failed", seed)
+            if meas is not None and meas.seconds is not None:
+                row = meas.stage_row(run_id, "mlp_train", "failed")
+                row["measurement_note"] += f" Этап прерван: {type(e).__name__}: {e}"
+                stage_rows.append(row)
+            else:
+                stage_rows.append({
+                    "run_id": run_id, "stage": "mlp_train", "status": "failed",
+                    "seconds": None, "peak_rss_bytes": None, "peak_vram_bytes": None,
+                    "measurement_note": f"refit прерван: {type(e).__name__}: {e}"[:500],
+                })
+            run_rows.append({
+                "run_id": f"seed_{seed}", "config_id": config_id, "train_seed": seed,
+                "split_seed": split_seed, "status": "failed",
+                "reason": f"{type(e).__name__}: {e}"[:500],
+                "n_train": int(len(train_df)), "n_monitor": int(len(mon_df)),
+                "n_val": int(manifest["parts"]["comparison_val"]["n_rows"]), "n_classes": int(n_classes[2]),
+                "full_tfidf_id": (loaded.meta.get("files") or {})
+                .get("features.joblib", {}).get("sha256"),
+                "epochs_completed": None, "best_epoch": None,
+                "l3_macro_f1": None, "l3_accuracy": None,
+                "metric_evaluated_on": "comparison_val",
+                "train_seconds": None, "peak_rss_bytes": None, "checkpoint_path": None,
+            })
+            run_results[seed] = {"run_id": f"seed_{seed}", "seed": seed, "status": "failed",
+                                 "reason": f"{type(e).__name__}: {e}"[:500]}
+
+    trained = [s for s in cfg.seeds if run_results[s]["status"] == "trained"]
+    if not trained:
+        # Фиксируем провал до ошибки: все planned seed остаются видимыми в
+        # runs.csv (status=failed, null-метрики), стадии — в stages.csv;
+        # comparison_val не читался и не оценивался (успешных моделей нет).
+        pd.DataFrame(run_rows, columns=RUNS_COLUMNS).to_csv(os.path.join(out_dir, "runs.csv"), index=False)
+        pd.DataFrame(history_rows, columns=HISTORY_COLUMNS).to_csv(os.path.join(out_dir, "history.csv"), index=False)
+        pd.DataFrame(stage_rows, columns=STAGES_COLUMNS).to_csv(os.path.join(out_dir, "stages.csv"), index=False)
+        raise S05BaselineError(
+            "Ни один seed не обучен; failed-запуски зафиксированы в runs.csv/stages.csv, "
+            "comparison_val не читался: см. логи"
+        )
+
+    # --- Одна оценка raw top-1 на comparison_val ПОСЛЕ refit всех seed
+    # (первое чтение/кодирование val-данных — только здесь, и только если
+    # есть успешно обученные seed)
+    val_df = load_part(manifest, "comparison_val")
+    y_val = _encode_labels(maps, val_df, "comparison_val")
+    with _Measurement(device, "transform comparison_val полным TF-IDF (первое обращение к val — после всех refit)") as m:
+        X_val = _transform(loaded, val_df["name"].tolist(), cfg.features_chunk_rows)
+    stage_rows.append(m.stage_row(SHARED_VECTORIZATION_RUN_ID, "vectorize_comparison_val", "success"))
+    n_empty_val = _empty_rows(X_val)
+
+    eval_rows: dict[int, dict[str, Any]] = {}
+    for seed in trained:
+        run_id = f"seed_{seed}"
+        with _Measurement(device, f"seed {seed}: predict raw top-1 на comparison_val (n={X_val.shape[0]}), без отказа") as meas:
+            model = run_results[seed]["checkpoint_dir"]
+            art = load_artifact(model, device)
+            preds, _ = predict(art.model, X_val, device, train_cfg.eval_batch_size, top_k=1)
+            macro, acc, per_class = _l3_val_metrics(y_val[2], preds.l3_top1, l3_labels)
+        stage_rows.append(meas.stage_row(run_id, "comparison_val_inference", "success"))
+        frame = pd.DataFrame(per_class)[PER_CLASS_COLUMNS[1:]]
+        frame.insert(0, "run_id", run_id)
+        per_class_frames.append(frame)
+        rr = run_results[seed]
+        eval_rows[seed] = {
+            "l3_macro_f1": macro, "l3_accuracy": acc,
+            "val_inference_seconds": meas.seconds,
+            "n_empty_features_val": n_empty_val,
+            "per_class": per_class,
+        }
+        run_rows.append({
+            "run_id": run_id, "config_id": config_id, "train_seed": seed,
+            "split_seed": split_seed, "status": "success", "reason": None,
+            "n_train": int(len(train_df)), "n_monitor": int(len(mon_df)),
+            "n_val": int(len(val_df)), "n_classes": int(n_classes[2]),
+            "full_tfidf_id": (loaded.meta.get("files") or {})
+            .get("features.joblib", {}).get("sha256"),
+            "epochs_completed": len(rr["history"]),
+            "best_epoch": int(rr["best"]["epoch"]),
+            "l3_macro_f1": macro, "l3_accuracy": acc,
+            "metric_evaluated_on": "comparison_val",
+            "train_seconds": rr["train_seconds"],
+            "peak_rss_bytes": rr["peak_rss_bytes"],
+            "checkpoint_path": rr["checkpoint_dir"],
+        })
+        log.info("seed %d: comparison_val raw top-1 — acc=%.4f macro-F1=%.4f", seed, acc, macro)
+
+    # --- Majority-class baseline (большинство по outer train) на том же comparison_val
+    counts_l3 = np.bincount(y_train[2], minlength=n_classes[2])
+    majority_id = int(np.argmax(counts_l3))
+    majority_label = l3_labels[majority_id]
+    majority_pred = np.full(len(y_val[2]), majority_id, dtype=np.int64)
+    maj_macro, maj_acc, maj_per_class = _l3_val_metrics(y_val[2], majority_pred, l3_labels)
+    majority_report = {
+        "run_id": "majority_outer_train",
+        "rule": "наиболее частая L3 в outer_train (ties → наименьший id; train-only)",
+        "majority_label": majority_label,
+        "majority_id": majority_id,
+        "majority_count_outer_train": int(counts_l3[majority_id]),
+        "n_outer_train": int(len(train_df)),
+        "evaluated_on": "comparison_val",
+        "l3_accuracy": maj_acc,
+        "l3_macro_f1": maj_macro,
+        "n_empty_features_val": n_empty_val,
+        "per_class": maj_per_class,
+        "note": "тривиальный контроль; качество моделей сопоставляется с ним, а не с историческими ods_light метриками",
+    }
+    log.info("Majority (%s): comparison_val acc=%.4f macro-F1=%.4f", majority_label, maj_acc, maj_macro)
+
+    # --- Опорный checkpoint для S06: правило reference_seed, round-trip проверка
+    selected = _select_and_verify_checkpoint(
+        cfg, manifest, loaded, maps, ckpt_root, run_results, eval_rows,
+        X_val, y_val, l3_labels, device, n_classes,
+    )
+
+    # --- Сводные агрегаты по seed (mean/std/range)
+    f1s = [eval_rows[s]["l3_macro_f1"] for s in trained]
+    accs = [eval_rows[s]["l3_accuracy"] for s in trained]
+    seed_stats = {
+        "n_seed_success": len(trained),
+        "n_seed_failed": len(cfg.seeds) - len(trained),
+        "l3_macro_f1": _mean_std_range(f1s),
+        "l3_accuracy": _mean_std_range(accs),
+    }
+
+    # --- Записи результатов (все пути проверены на отсутствие до любых записей)
+    for seed in trained:
+        rr = run_results[seed]
+        hist_df = pd.DataFrame(rr["history"])
+        hist_df.to_csv(os.path.join(refit_dir, f"history_seed_{seed}.csv"), index=False)
+        ev = eval_rows[seed]
+        _write_json({
+            "run_id": rr["run_id"], "seed": seed, "status": "success",
+            "config_id": config_id,
+            "fixed_epochs": fixed_epochs,
+            "selection": {"selection_seed": cfg.selection_seed, "criterion": chosen.get("criterion")},
+            "train": {
+                "epochs_completed": len(rr["history"]),
+                "best_epoch_monitor": int(rr["best"]["epoch"]),
+                "early_stopping_metric": rr["best"]["metric"],
+                "best_score_monitor": float(rr["best"]["score"]),
+                "restored_best": bool(rr["best"].get("restored_best")),
+                "final_monitor_l3_macro_f1": rr["history"][-1]["val_l3_macro_f1"] if rr["history"] else None,
+                "final_monitor_l3_accuracy": rr["history"][-1]["val_l3_accuracy"] if rr["history"] else None,
+                "n_train_rows_used": int(rr["best"].get("n_train_rows_used", 0)),
+            },
+            "train_seconds": rr["train_seconds"],
+            "peak_rss_bytes": rr["peak_rss_bytes"],
+            "peak_vram_bytes": rr["peak_vram_bytes"],
+            "checkpoint_dir": rr["checkpoint_dir"],
+            "comparison_val_eval": {
+                "evaluated_after": "все refit seed завершены",
+                "protocol": "raw top-1 L3 без отказа",
+                "metric_evaluated_on": "comparison_val",
+                "l3_macro_f1": ev["l3_macro_f1"],
+                "l3_accuracy": ev["l3_accuracy"],
+                "n_empty_features_val": ev["n_empty_features_val"],
+                "val_inference_seconds": ev["val_inference_seconds"],
+                "fixed_l3_universe": len(l3_labels),
+            },
+            "device": str(device),
+        }, os.path.join(refit_dir, f"run_result_seed_{seed}.json"))
+
+    pd.DataFrame(run_rows, columns=RUNS_COLUMNS).to_csv(os.path.join(out_dir, "runs.csv"), index=False)
+    pd.concat(per_class_frames, ignore_index=True).to_parquet(os.path.join(out_dir, "per_class.parquet"), index=False)
+    pd.DataFrame(history_rows, columns=HISTORY_COLUMNS).to_csv(os.path.join(out_dir, "history.csv"), index=False)
+    pd.DataFrame(stage_rows, columns=STAGES_COLUMNS).to_csv(os.path.join(out_dir, "stages.csv"), index=False)
+    _write_json(majority_report, os.path.join(out_dir, "majority_baseline.json"))
+    _write_json(selected, os.path.join(out_dir, "selected_checkpoint.json"))
+
+    git_commit, git_dirty = _git_info()
+    manifest_out = {
+        "schema_version": S05_SCHEMA_VERSION,
+        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
+        "protocol": dict(PROTOCOL),
+        "config": cfg.to_dict(),
+        "config_id": config_id,
+        "fixed_epochs": fixed_epochs,
+        "selection": (selection or {}).get("chosen"),
+        "seeds": {"planned": list(cfg.seeds), "trained": trained,
+                  "failed": [s for s in cfg.seeds if s not in trained],
+                  "reference_seed": int(cfg.reference_seed)},
+        "seed_stats_comparison_val": seed_stats,
+        "inputs": {
+            "splits_manifest": file_fingerprint(cfg.splits_manifest),
+            "splits": {
+                name: {"sha256": part["sha256"], "bytes": part["bytes"], "n_rows": part["n_rows"]}
+                for name, part in manifest["parts"].items()
+            },
+            "labeled_parquet": manifest["inputs"]["labeled_parquet"],
+            "full_tfidf": {
+                "path": os.path.abspath(cfg.full_tfidf_path),
+                "artifact_version": loaded.meta.get("artifact_version"),
+                "full_tfidf_id": (loaded.meta.get("files") or {})
+                .get("features.joblib", {}).get("sha256"),
+                "vocab": loaded.meta.get("vocab"),
+                "fit": loaded.meta.get("fit"),
+                "files": (verify_full_tfidf(cfg.full_tfidf_path, cfg.corpus_path)
+                          if os.path.isfile(cfg.corpus_path) else {}).get("files"),
+            },
+            "corpus": file_fingerprint(cfg.corpus_path) if os.path.isfile(cfg.corpus_path) else None,
+        },
+        "label_maps": {
+            "n_classes": {lvl: maps.n_classes(lvl) for lvl in LEVELS},
+            "l3_universe_sha256": _sha256_json(maps.to_dict()),
+        },
+        "environment": {
+            "versions": environment_versions(),
+            "platform": platform.platform(),
+            "cpu_count": os.cpu_count(),
+            "device": str(device),
+            "torch_cuda_available": torch.cuda.is_available(),
+        },
+        "git": {"commit": git_commit, "dirty": git_dirty},
+        "statuses": {
+            "refit": {f"seed_{s}": run_results[s]["status"] for s in cfg.seeds},
+            "comparison_val_eval": {f"seed_{s}": "success" for s in trained},
+            "selected_checkpoint": selected["roundtrip"]["status"],
+        },
+        "limitations": [
+            "ODS/PoC: результаты на comparison_val диагностические, не оценка промышленной применимости.",
+            "Сеaled_test не использовался ни для чего; test-метрик нет.",
+            PROTOCOL["transductive"],
+            "Режим pretrained: идентичность словаря/IDF всех seed обеспечена одним артефактом; "
+            "fit векторизатора между seed не выполнялся.",
+        ],
+    }
+    _write_json(manifest_out, os.path.join(out_dir, "manifest.json"))
+    log.info("Refit контур завершён: %d/%d seed успешно, отчёты в %s",
+             len(trained), len(cfg.seeds), out_dir)
+    return manifest_out
+
+
+def _mean_std_range(values: list[float]) -> dict[str, float]:
+    arr = np.asarray(values, dtype=np.float64)
+    return {
+        "mean": float(arr.mean()), "std": float(arr.std()),
+        "min": float(arr.min()), "max": float(arr.max()),
+        "range": float(arr.max() - arr.min()), "n": int(len(arr)),
+    }
+
+
+def _select_and_verify_checkpoint(
+    cfg: S05Config,
+    manifest: dict[str, Any],
+    loaded: LoadedVectorizer,
+    maps: LabelMaps,
+    ckpt_root: str,
+    run_results: dict[int, dict[str, Any]],
+    eval_rows: dict[int, dict[str, Any]],
+    X_val: sp.csr_matrix,
+    y_val: tuple[np.ndarray, np.ndarray, np.ndarray],
+    l3_labels: list[str],
+    device: torch.device,
+    n_classes: tuple[int, int, int],
+) -> dict[str, Any]:
+    """D08: выбор опорного checkpoint заранее заданным правилом reference_seed.
+
+    Round-trip: артефакт загружается с диска через load_artifact, сверяются
+    seed/размерность/label maps/число эпох, выполняется независимый inference
+    на comparison_val; метрики должны совпасть с сохранёнными (allclose).
+    Правило — train-only, качество на comparison_val в выборе не участвует.
+    """
+    seed = int(cfg.reference_seed)
+    rr = run_results.get(seed)
+    if rr is None or rr["status"] != "trained":
+        raise S05BaselineError(
+            f"reference_seed={seed} не обучен (status={rr and rr['status']}): опорный "
+            "checkpoint заранее заданным правилом получить нельзя"
+        )
+    ckpt_dir = os.path.join(ckpt_root, f"seed_{seed}")
+    art = load_artifact(ckpt_dir, device)
+    checks: dict[str, Any] = {}
+    checks["seed_matches"] = bool((art.meta or {}).get("seed") == seed)
+    checks["n_features_matches"] = bool(art.model.n_features == loaded.builder.n_features)
+    checks["n_classes_match"] = bool(tuple(art.model.n_classes) == tuple(n_classes))
+    checks["label_maps_match"] = bool(art.maps.to_dict() == maps.to_dict())
+    checks["protocol"] = (art.meta or {}).get("protocol")
+    checks["epochs_completed"] = int(((art.meta or {}).get("train") or {}).get("epochs_completed", -1))
+    checks["restored_best"] = bool(((art.meta or {}).get("train") or {}).get("restored_best"))
+    identity_checks = {
+        "seed_matches": checks["seed_matches"],
+        "n_features_matches": checks["n_features_matches"],
+        "n_classes_match": checks["n_classes_match"],
+        "label_maps_match": checks["label_maps_match"],
+    }
+    if not all(identity_checks.values()):
+        raise S05BaselineError(f"Round-trip проверка provenance не пройдена: {checks}")
+    if checks["restored_best"] is not False:
+        raise S05BaselineError(
+            f"Опорный checkpoint обязан быть финальным refit без выбора по monitor "
+            f"(restored_best=False), получено {checks['restored_best']!r}"
+        )
+
+    preds, _ = predict(art.model, X_val, device, 8192, top_k=1)
+    macro, acc, _ = _l3_val_metrics(y_val[2], preds.l3_top1, l3_labels)
+    saved = eval_rows[seed]
+    metrics_match = bool(
+        np.isclose(macro, saved["l3_macro_f1"], rtol=1e-6, atol=1e-8)
+        and np.isclose(acc, saved["l3_accuracy"], rtol=1e-6, atol=1e-8)
+    )
+    result = {
+        "run_id": f"seed_{seed}",
+        "selection_rule": {
+            "kind": "reference_seed",
+            "value": seed,
+            "note": "заранее заданное детерминированное train-only правило; "
+                    "качество на comparison_val в выборе не участвует",
+        },
+        "checkpoint_dir": os.path.abspath(ckpt_dir),
+        "checkpoint_files_sha256": {
+            name: _sha256_file(os.path.join(ckpt_dir, name))
+            for name in sorted(os.listdir(ckpt_dir))
+        },
+        "provenance_checks": checks,
+        "roundtrip": {
+            "status": "success" if metrics_match else "failed",
+            "inference": "load_artifact → transform comparison_val готовым full TF-IDF → predict",
+            "l3_macro_f1_recomputed": macro,
+            "l3_accuracy_recomputed": acc,
+            "l3_macro_f1_saved": saved["l3_macro_f1"],
+            "l3_accuracy_saved": saved["l3_accuracy"],
+            "metrics_match": metrics_match,
+        },
+        "expected_use": "опорная MLP для Gradient × Input (S06); G×I извлекается из этой модели",
+    }
+    if not metrics_match:
+        raise S05BaselineError(f"Round-trip метрики не совпали: {result['roundtrip']}")
+    return result
+
+
+# ---------------------------------------------------------------------------
+# Полный контур (verify → select → refit+eval)
+# ---------------------------------------------------------------------------
+
+
+def run_training_contour(cfg: S05Config) -> dict[str, Any]:
+    """verify (D04) → selection (D05) → refit+eval+checkpoint (D06–D08).
+
+    Записывает также ``tfidf_verification.json`` (только новый файл).
+    """
+    cfg.validate()
+    device = resolve_device(cfg.device)
+    manifest = load_split_manifest(cfg.splits_manifest)
+    verification = verify_full_tfidf(cfg.full_tfidf_path, cfg.corpus_path)
+    loaded = load_vectorizer_artifact(cfg.full_tfidf_path)
+
+    vpath = os.path.join(cfg.output_dir, "tfidf_verification.json")
+    _check_new([vpath])
+    os.makedirs(cfg.output_dir, exist_ok=True)
+    _write_json(verification, vpath)
+
+    selection = run_selection(cfg, manifest, loaded, device)
+    return run_refit(cfg, manifest, loaded, device, selection=selection)
diff --git a/kkt_clf/s05_report.py b/kkt_clf/s05_report.py
new file mode 100644
index 0000000..b191377
--- /dev/null
+++ b/kkt_clf/s05_report.py
@@ -0,0 +1,758 @@
+"""S05 (D09): агрегированный отчёт из сохранённых машинных результатов обучения.
+
+Строит ``report/report.html``, ``report/runs.csv``, ``report/per_class.parquet``,
+``report/stages.csv``, ``report/history.csv``, ``report/manifest.json`` **без
+обучения и без пересчёта метрик** — исключительно по файлам, записанным
+обучающим контуром ``kkt_clf.s05_baseline`` (D04–D08) и аудитом (D02/D03).
+
+Источники (все опциональны, кроме обязательной пятёрки):
+
+- обязательные: ``runs.csv``, ``history.csv``, ``stages.csv``,
+  ``per_class.parquet``, ``manifest.json`` в ``output_dir``;
+- опциональные: ``majority_baseline.json``, ``selected_checkpoint.json``,
+  ``tfidf_verification.json``, ``selection/selection_report.json``,
+  ``audit/split_audit_corrected.csv`` (предпочитается исправленный аудит
+  сплита; исходный ``audit/split_audit.csv`` используется только как
+  запасной вариант), ``audit/class_support.csv``,
+  ``audit/class_support_split.csv``, ``audit/data_audit.json``.
+
+Гарантии контракта D09:
+
+- числа берутся из файлов как есть; отсутствующие значения остаются ``null``
+  (в HTML — «—»), ничего не дорисовывается нулями;
+- запланированные seed из ``manifest.seeds.planned``, не имеющие строки в
+  ``runs.csv``, добавляются как ``skipped`` — плановые failed/skipped видны;
+- test-метрик нет: sealed test не читается и не оценивается;
+- средние/разброс считаются внутри одинаковых ``config_id``;
+- HTML автономен (inline CSS/SVG, без CDN), кривые train/monitor, gap,
+  majority-сравнение, per-class/support и ресурсы — из сохранённых таблиц;
+- файл сплит-аудита: в исходном ``split_audit.csv`` колонка
+  ``n_row_ids_unique`` была ошибочно записана постоянной ``1``; используется
+  ``split_audit_corrected.csv``, ошибка явно указывается в отчёте.
+
+Ничего не перезаписывается: если любой целевой файл в ``report_dir`` уже
+существует — отказ до любых записей.
+"""
+from __future__ import annotations
+
+import datetime as dt
+import os
+from typing import Any
+
+import numpy as np
+import pandas as pd
+
+from .experiment import (
+    _esc,
+    _fmt,
+    _fmt_mb,
+    _read_json,
+    _sha256_file,
+    _svg_bar_chart,
+    _svg_line_chart,
+    _table_html,
+    _write_json,
+)
+
+log = __import__("logging").getLogger(__name__)
+
+S05_REPORT_SCHEMA_VERSION = "s05-report@1"
+REPORT_DIRNAME = "report"
+REPORT_FILES = ("report.html", "runs.csv", "per_class.parquet",
+                "stages.csv", "history.csv", "manifest.json")
+
+SPLIT_AUDIT_ORIGINAL_ERROR = (
+    "Исходный audit/split_audit.csv содержал ошибку в колонке n_row_ids_unique: "
+    "вместо фактического числа уникальных row_id во всех частях было записано "
+    "постоянное значение 1 (остальные колонки корректны). В отчёте используется "
+    "исправленный audit/split_audit_corrected.csv; sha256/bytes частей сплита "
+    "в обоих файлах совпадают."
+)
+
+RUNS_REQUIRED = [
+    "run_id", "config_id", "train_seed", "split_seed", "status", "reason",
+    "n_train", "n_monitor", "n_val", "n_classes", "full_tfidf_id",
+    "epochs_completed", "best_epoch", "l3_macro_f1", "l3_accuracy",
+    "metric_evaluated_on", "train_seconds", "peak_rss_bytes", "checkpoint_path",
+]
+STAGES_REQUIRED = [
+    "run_id", "stage", "status", "seconds", "peak_rss_bytes", "peak_vram_bytes",
+    "measurement_note",
+]
+HISTORY_REQUIRED = [
+    "run_id", "epoch", "split", "loss", "l3_macro_f1", "l3_accuracy", "lr",
+]
+PER_CLASS_REQUIRED = ["run_id", "L3", "support", "precision", "recall", "f1"]
+
+
+class S05ReportError(RuntimeError):
+    """Ошибка построения отчёта S05; файлы не записываются."""
+
+
+# ---------------------------------------------------------------------------
+# Загрузка машинных файлов
+# ---------------------------------------------------------------------------
+
+
+def _required_frame(path: str, columns: list[str], kind: str) -> pd.DataFrame:
+    if not os.path.isfile(path):
+        raise S05ReportError(
+            f"{kind}: обязательный машинный файл не найден: {path!r}. "
+            "Сначала выполните обучающий контур S05 (scripts/run_s05_baseline.py "
+            "refit); отчёт не подменяет несуществующие данные."
+        )
+    if path.endswith(".parquet"):
+        df = pd.read_parquet(path)
+    else:
+        df = pd.read_csv(path)
+    missing = [c for c in columns if c not in df.columns]
+    if missing:
+        raise S05ReportError(f"{kind}: в {path!r} нет обязательных колонок {missing}")
+    return df
+
+
+def _optional_json(output_dir: str, name: str) -> dict[str, Any] | None:
+    path = os.path.join(output_dir, name)
+    if not os.path.isfile(path):
+        return None
+    try:
+        return _read_json(path)
+    except Exception as e:  # noqa: BLE001 — битый опциональный файл не роняет отчёт
+        log.warning("Не удалось прочитать %s: %s", path, e)
+        return None
+
+
+def _load_split_audit(output_dir: str) -> tuple[pd.DataFrame | None, str | None]:
+    """Исправленный аудит сплита предпочтительнее; ошибка исходного файла
+    фиксируется для показа в отчёте."""
+    corrected = os.path.join(output_dir, "audit", "split_audit_corrected.csv")
+    original = os.path.join(output_dir, "audit", "split_audit.csv")
+    if os.path.isfile(corrected):
+        return pd.read_csv(corrected), "split_audit_corrected.csv"
+    if os.path.isfile(original):
+        return pd.read_csv(original), "split_audit.csv (с ошибкой в n_row_ids_unique, см. примечание)"
+    return None, None
+
+
+def _check_new(paths: list[str]) -> None:
+    existing = [p for p in paths if os.path.exists(p)]
+    if existing:
+        raise S05ReportError(
+            "Существующие файлы отчёта не перезаписываются; отказ до любых записей: "
+            f"{existing}. Удалите/перенесите их или укажите другой --report-dir."
+        )
+
+
+# ---------------------------------------------------------------------------
+# Агрегаты: seed-статистика, train–monitor gap, majority
+# ---------------------------------------------------------------------------
+
+
+def _mean_std_range(values: list[float]) -> dict[str, Any]:
+    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
+    if len(arr) == 0:
+        return {"mean": None, "std": None, "min": None, "max": None,
+                "range": None, "n": 0}
+    return {
+        "mean": float(arr.mean()), "std": float(arr.std()),
+        "min": float(arr.min()), "max": float(arr.max()),
+        "range": float(arr.max() - arr.min()), "n": int(len(arr)),
+    }
+
+
+def _with_skipped_runs(runs_df: pd.DataFrame, training_manifest: dict | None) -> pd.DataFrame:
+    """Запланированные seed без строки в runs.csv → status=skipped (null-метрики)."""
+    planned = ((training_manifest or {}).get("seeds") or {}).get("planned") or []
+    present = set(runs_df["run_id"].astype(str))
+    rows = []
+    split_seed = None
+    if training_manifest:
+        split_seed = ((training_manifest.get("config") or {}).get("selection_seed"))
+    for seed in planned:
+        run_id = f"seed_{seed}"
+        if run_id not in present:
+            rows.append({
+                "run_id": run_id, "config_id": None, "train_seed": int(seed),
+                "split_seed": split_seed, "status": "skipped",
+                "reason": "запланирован в manifest.seeds.planned, но обучающий "
+                          "контур не записал результат (прогон не выполнен или "
+                          "контур прерван до его записи)",
+                "n_train": None, "n_monitor": None, "n_val": None,
+                "n_classes": None, "full_tfidf_id": None,
+                "epochs_completed": None, "best_epoch": None,
+                "l3_macro_f1": None, "l3_accuracy": None,
+                "metric_evaluated_on": "comparison_val",
+                "train_seconds": None, "peak_rss_bytes": None, "checkpoint_path": None,
+            })
+    if not rows:
+        return runs_df
+    skipped_df = pd.DataFrame(rows).reindex(columns=runs_df.columns)
+    for col in runs_df.columns:
+        if not skipped_df[col].notna().any():
+            if pd.api.types.is_numeric_dtype(runs_df[col].dtype):
+                skipped_df[col] = np.nan
+            else:
+                skipped_df[col] = None
+        elif pd.api.types.is_numeric_dtype(runs_df[col].dtype) and \
+                pd.api.types.is_numeric_dtype(skipped_df[col].dropna().dtype):
+            skipped_df[col] = skipped_df[col].astype(runs_df[col].dtype)
+    return pd.concat([runs_df, skipped_df], ignore_index=True)
+
+
+def _seed_stats(runs_df: pd.DataFrame) -> list[dict[str, Any]]:
+    """mean/std/range внутри групп одинакового config_id (успешные runs)."""
+    out = []
+    ok = runs_df[runs_df["status"] == "success"]
+    for config_id, grp in ok.groupby(ok["config_id"].astype(str)):
+        out.append({
+            "config_id": config_id,
+            "n_success": int(len(grp)),
+            "l3_macro_f1": _mean_std_range(grp["l3_macro_f1"].tolist()),
+            "l3_accuracy": _mean_std_range(grp["l3_accuracy"].tolist()),
+            "evaluated_on": "comparison_val",
+        })
+    return out
+
+
+def _train_monitor_gap(history_df: pd.DataFrame, runs_df: pd.DataFrame) -> pd.DataFrame:
+    """Итоговый train↔monitor gap по loss и метрикам monitor (последняя эпоха)."""
+    rows = []
+    for run_id, grp in history_df.groupby("run_id"):
+        if run_id not in set(runs_df["run_id"]):
+            continue
+        last_epoch = grp["epoch"].max()
+        final = grp[grp["epoch"] == last_epoch]
+        train_loss = final.loc[final["split"] == "train", "loss"]
+        mon = final[final["split"] == "monitor"]
+        mon_loss = mon["loss"]
+        tr = float(train_loss.iloc[0]) if len(train_loss) else None
+        mo = float(mon_loss.iloc[0]) if len(mon_loss) else None
+        gap = (mo - tr) if (tr is not None and mo is not None) else None
+        rows.append({
+            "run_id": run_id,
+            "final_train_loss": tr,
+            "final_monitor_loss": mo,
+            "gap_monitor_minus_train": gap,
+            "final_monitor_l3_macro_f1": (float(mon["l3_macro_f1"].iloc[0])
+                                          if len(mon) and pd.notna(mon["l3_macro_f1"].iloc[0]) else None),
+            "final_monitor_l3_accuracy": (float(mon["l3_accuracy"].iloc[0])
+                                          if len(mon) and pd.notna(mon["l3_accuracy"].iloc[0]) else None),
+        })
+    return pd.DataFrame(rows)
+
+
+# ---------------------------------------------------------------------------
+# Честный S06 gate (без заранее назначенного порога F1)
+# ---------------------------------------------------------------------------
+
+
+def _decide_gate(
+    runs_df: pd.DataFrame,
+    training_manifest: dict | None,
+    tfidf_verification: dict | None,
+    selected: dict | None,
+    majority: dict | None,
+    gap_df: pd.DataFrame | None,
+    split_audit: pd.DataFrame | None,
+) -> dict[str, Any]:
+    reasons: list[str] = []
+    ok = runs_df[runs_df["status"] == "success"]
+    n_planned = len(runs_df)
+    n_ok = len(ok)
+    n_failed = int((runs_df["status"] == "failed").sum())
+    n_skipped = int((runs_df["status"] == "skipped").sum())
+
+    if n_ok == 0:
+        reasons.append("Нет ни одного успешного прогона обучения: baseline не продемонстрирован.")
+        return {"decision": "not_ready", "reasons": reasons,
+                "n_planned": n_planned, "n_success": 0,
+                "n_failed": n_failed, "n_skipped": n_skipped}
+
+    if split_audit is None:
+        reasons.append("Файл аудита сплита (audit/split_audit_corrected.csv) отсутствует: "
+                       "фиксация разбиения (D03) не подтверждена файлами.")
+    if tfidf_verification is not None and tfidf_verification.get("decision") != "ok":
+        reasons.append(f"Проверка full TF-IDF (D04): decision={tfidf_verification.get('decision')!r} ≠ 'ok'.")
+    if tfidf_verification is None:
+        reasons.append("tfidf_verification.json отсутствует: проверка единого full TF-IDF (D04) файлами не подтверждена.")
+
+    maj_f1 = (majority or {}).get("l3_macro_f1")
+    if maj_f1 is None:
+        reasons.append("Majority-class baseline не записан (majority_baseline.json отсутствует): "
+                       "содержательность baseline (D07) не подтверждена.")
+    else:
+        mean_f1 = float(ok["l3_macro_f1"].astype(float).mean())
+        if not mean_f1 > float(maj_f1):
+            reasons.append(
+                f"Средний macro-F1 успешных seed ({mean_f1:.4f}) не превышает "
+                f"majority-class baseline ({float(maj_f1):.4f}) на comparison_val: "
+                "baseline не показал содержательного обучения (D07).")
+
+    if n_ok < 3:
+        reasons.append(f"Успешных seed {n_ok} < 3 (D06): разброс по seed недостаточен для S06.")
+    if n_failed or n_skipped:
+        reasons.append(f"Плановые прогоны неуспешны: failed={n_failed}, skipped={n_skipped} — см. runs.csv.")
+    if len(ok["config_id"].astype(str).unique()) > 1:
+        reasons.append("Успешные прогоны относятся к разным config_id: средние по seed "
+                       "необъединимы в один протокол.")
+
+    if selected is None:
+        reasons.append("selected_checkpoint.json отсутствует: опорный checkpoint для S06 (D08) не зафиксирован.")
+    else:
+        rt = (selected.get("roundtrip") or {}).get("status")
+        if rt != "success":
+            reasons.append(f"Round-trip проверка опорного checkpoint: status={rt!r} (D08).")
+
+    if gap_df is not None and len(gap_df):
+        gaps = gap_df["gap_monitor_minus_train"].dropna()
+        if len(gaps):
+            worst = float(gaps.max())
+            if worst > 1.0:
+                reasons.append(
+                    f"Существенный train–monitor gap по loss (max monitor−train = {worst:.4f}): "
+                    "признаки переобучения; характер gap требует объяснения до S06.")
+    decision = "ready" if not reasons else "conditional"
+    return {"decision": decision, "reasons": reasons,
+            "n_planned": n_planned, "n_success": n_ok,
+            "n_failed": n_failed, "n_skipped": n_skipped}
+
+
+# ---------------------------------------------------------------------------
+# HTML (автономный, inline SVG через переиспользуемые helpers S03/S04)
+# ---------------------------------------------------------------------------
+
+_CSS = """
+body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 24px;
+       color: #1a1a2e; max-width: 1180px; }
+h1 { font-size: 22px; } h2 { font-size: 17px; margin-top: 32px;
+     border-bottom: 2px solid #4a69bd; padding-bottom: 4px; }
+table { border-collapse: collapse; margin: 12px 0; font-size: 13px; width: 100%; }
+th, td { border: 1px solid #ccd; padding: 4px 8px; text-align: left; }
+th { background: #eef1f8; }
+.caption, .muted { color: #667; font-size: 12px; }
+.tick { font-size: 10px; fill: #556; }
+.chart-title { font-size: 13px; font-weight: 600; fill: #1a1a2e; }
+.grid { stroke: #dde; stroke-width: 1; }
+.axis { stroke: #99a; stroke-width: 1.5; }
+.baseline { stroke: #b71540; stroke-dasharray: 5 4; stroke-width: 1.5; }
+.baseline-label { fill: #b71540; }
+.badge { display: inline-block; padding: 2px 12px; border-radius: 10px;
+         font-weight: 600; font-size: 13px; }
+.badge.ready { background: #d8f5d0; color: #1e6b2f; }
+.badge.conditional { background: #fdf0c8; color: #8a6100; }
+.badge.not_ready { background: #f8d7da; color: #8a1c24; }
+.note { background: #f4f6fb; border-left: 4px solid #4a69bd; padding: 8px 12px;
+        font-size: 13px; margin: 10px 0; }
+.warn { background: #fdf0c8; border-left: 4px solid #b71540; padding: 8px 12px;
+        font-size: 13px; margin: 10px 0; }
+code { background: #f0f0f4; padding: 1px 4px; border-radius: 3px; }
+"""
+
+
+def _fmt_run_metric(v: Any) -> str:
+    if v is None or (isinstance(v, float) and np.isnan(v)):
+        return "—"
+    if isinstance(v, (int, np.integer)):
+        return str(int(v))
+    if isinstance(v, (float, np.floating)):
+        return f"{float(v):.4f}"
+    return str(v)
+
+
+def _html_report(
+    output_dir: str,
+    runs_df: pd.DataFrame,
+    history_df: pd.DataFrame,
+    stages_df: pd.DataFrame,
+    per_class_df: pd.DataFrame,
+    training_manifest: dict | None,
+    majority: dict | None,
+    selected: dict | None,
+    tfidf_verification: dict | None,
+    selection_report: dict | None,
+    split_audit: pd.DataFrame | None,
+    split_audit_source: str | None,
+    class_support: pd.DataFrame | None,
+    class_support_split: pd.DataFrame | None,
+    data_audit: dict | None,
+    seed_stats: list[dict[str, Any]],
+    gap_df: pd.DataFrame | None,
+    gate: dict[str, Any],
+    report_manifest: dict[str, Any],
+) -> str:
+    p: list[str] = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
+                    f"<title>S05 baseline report ({_esc(S05_REPORT_SCHEMA_VERSION)})</title>",
+                    f"<style>{_CSS}</style></head><body>"]
+    p.append(f"<h1>S05 — аудит ODS и воспроизводимый baseline: агрегированный отчёт</h1>")
+    p.append(f"<p class='muted'>Схема {_esc(S05_REPORT_SCHEMA_VERSION)} · сформирован "
+             f"{_esc(report_manifest['generated_at'])} из машинных файлов "
+             f"<code>{_esc(os.path.abspath(output_dir))}</code> без обучения. "
+             "Sealed test не использовался; test-метрик в отчёте нет.</p>")
+    p.append(f"<p>Вердикт S06 gate: <span class='badge {_esc(gate['decision'])}'>"
+             f"{_esc(gate['decision'])}</span> "
+             f"(плановых прогонов {gate['n_planned']}: success {gate['n_success']}, "
+             f"failed {gate['n_failed']}, skipped {gate['n_skipped']})</p>")
+
+    # --- протокол и границы ---
+    p.append("<h2>Протокол и границы частей</h2>")
+    proto = (training_manifest or {}).get("protocol") or {}
+    if proto:
+        p.append(_table_html(
+            pd.DataFrame([{"раздел": k, "описание": v} for k, v in proto.items()]),
+            ["раздел", "описание"]))
+    p.append("<div class='note'>Метрики качества вычислены на <b>comparison_val</b> "
+             "(raw top-1 L3 без отказа), однократно, после завершения всех refit. "
+             "Метки comparison_val/sealed_test не использовались для выбора checkpoint "
+             "и числа эпох. Режим full TF-IDF — <b>pretrained</b>, трансдуктивен: тексты "
+             "корпуса могут совпадать с train/val/test, метки не переносятся.</div>")
+
+    # --- аудит сплита ---
+    p.append("<h2>Аудит разбиения (D03)</h2>")
+    p.append(f"<div class='warn'>{_esc(SPLIT_AUDIT_ORIGINAL_ERROR)} "
+             f"Источник таблицы: <code>{_esc(split_audit_source or 'файл не найден')}</code>.</div>")
+    if split_audit is not None:
+        p.append(_table_html(split_audit, list(split_audit.columns),
+                             caption="Части сплита: строки, доли, группы, поддержка L3, "
+                                     "пересечения ключей TF-IDF между частями, fingerprints."))
+    else:
+        p.append("<p class='muted'>Файл аудита сплита не найден.</p>")
+
+    # --- аудит данных ---
+    if data_audit is not None:
+        p.append("<h2>Аудит входной разметки (D02)</h2>")
+        audit = data_audit.get("audit") or {}
+        summary_rows = [{"показатель": k, "значение": jsonable(v)}
+                        for k, v in list(audit.items())[:40]
+                        if not isinstance(v, (dict, list))]
+        if summary_rows:
+            p.append(_table_html(pd.DataFrame(summary_rows), ["показатель", "значение"],
+                                 caption=f"Источник: audit/data_audit.json "
+                                         f"(сформирован {data_audit.get('generated_at', '—')}). "
+                                         "Полные детали — в файле."))
+
+    # --- runs ---
+    p.append("<h2>Прогоны обучения (runs.csv)</h2>")
+    p.append(_table_html(runs_df, RUNS_REQUIRED,
+                         formats={c: _fmt_run_metric for c in RUNS_REQUIRED},
+                         caption="Одна строка на плановый прогон, включая failed/skipped; "
+                                 "'—' = null (не измерено/не записано), не ноль. "
+                                 "Метрики — на comparison_val."))
+
+    # --- seed stats ---
+    p.append("<h2>Разброс по seed (mean/std/range на comparison_val)</h2>")
+    if seed_stats:
+        rows = []
+        for st in seed_stats:
+            for metric in ("l3_macro_f1", "l3_accuracy"):
+                s = st[metric]
+                rows.append({"config_id": st["config_id"], "metric": metric,
+                             "n": st["n_success"], "mean": s["mean"], "std": s["std"],
+                             "min": s["min"], "max": s["max"], "range": s["range"]})
+        p.append(_table_html(pd.DataFrame(rows),
+                             ["config_id", "metric", "n", "mean", "std", "min", "max", "range"],
+                             formats={c: _fmt_run_metric for c in
+                                      ("mean", "std", "min", "max", "range")},
+                             caption="Средние считаются только внутри одинакового config_id "
+                                     "(без объединения разных конфигураций)."))
+    else:
+        p.append("<p class='muted'>Нет успешных прогонов — статистика по seed недоступна.</p>")
+
+    # --- кривые обучения ---
+    p.append("<h2>Кривые обучения (history.csv)</h2>")
+    ok_ids = runs_df.loc[runs_df["status"] == "success", "run_id"].tolist()
+    palette = ["#4a69bd", "#b71540", "#0a8754", "#e58e26", "#6c3483", "#117a8b"]
+    if ok_ids:
+        epochs = sorted(history_df["epoch"].unique().tolist())
+        mon_f1 = []
+        for i, rid in enumerate(ok_ids):
+            grp = history_df[(history_df["run_id"] == rid) & (history_df["split"] == "monitor")]
+            mon_f1.append((rid, palette[i % len(palette)],
+                           {int(r.epoch): (None if pd.isna(r.l3_macro_f1) else float(r.l3_macro_f1))
+                            for r in grp.itertuples()}))
+        p.append(_svg_line_chart("Monitor L3 macro-F1 по эпохам (train_monitor)", epochs, mon_f1))
+        train_loss, mon_loss = [], []
+        for i, rid in enumerate(ok_ids):
+            grp = history_df[history_df["run_id"] == rid]
+            tr = grp[grp["split"] == "train"]
+            mo = grp[grp["split"] == "monitor"]
+            train_loss.append((f"{rid} train", palette[i % len(palette)],
+                               {int(r.epoch): float(r.loss) for r in tr.itertuples()
+                                if pd.notna(r.loss)}))
+            mon_loss.append((f"{rid} monitor", palette[i % len(palette)],
+                             {int(r.epoch): (None if pd.isna(r.loss) else float(r.loss))
+                              for r in mo.itertuples()}))
+        p.append(_svg_line_chart("Loss по эпохам: train и monitor", epochs,
+                                 train_loss + mon_loss, y_fmt="{:.4f}"))
+    else:
+        p.append("<p class='muted'>Нет успешных прогонов — кривые недоступны.</p>")
+
+    # --- train–monitor gap ---
+    p.append("<h2>Train ↔ monitor gap</h2>")
+    if gap_df is not None and len(gap_df):
+        p.append(_table_html(gap_df, list(gap_df.columns),
+                             formats={c: _fmt_run_metric for c in gap_df.columns
+                                      if c != "run_id"},
+                             caption="Итоговая эпоха каждого прогона: loss train vs monitor, "
+                                     "разница monitor−train и финальные метрики monitor."))
+    else:
+        p.append("<p class='muted'>Нет данных history.csv.</p>")
+
+    # --- majority ---
+    p.append("<h2>Сравнение с тривиальным majority-class baseline (D07)</h2>")
+    if majority is not None:
+        maj_rows = [{
+            "метод": "majority (outer_train)", "run_id": majority.get("run_id"),
+            "majority_label": majority.get("majority_label"),
+            "l3_accuracy": majority.get("l3_accuracy"),
+            "l3_macro_f1": majority.get("l3_macro_f1"),
+            "evaluated_on": majority.get("evaluated_on"),
+        }]
+        for _, r in runs_df[runs_df["status"] == "success"].iterrows():
+            maj_rows.append({"метод": "MLP seed", "run_id": r["run_id"],
+                             "majority_label": None, "l3_accuracy": r["l3_accuracy"],
+                             "l3_macro_f1": r["l3_macro_f1"],
+                             "evaluated_on": r["metric_evaluated_on"]})
+        maj_df = pd.DataFrame(maj_rows)
+        p.append(_table_html(maj_df, list(maj_df.columns),
+                             formats={c: _fmt_run_metric for c in ("l3_accuracy", "l3_macro_f1")},
+                             caption="Качество моделей сопоставляется с majority на том же "
+                                     "comparison_val, а не с историческими ods_light метриками."))
+        groups = list(range(len(maj_rows)))
+        series = [("l3_macro_f1", "#4a69bd", {i: (None if pd.isna(r["l3_macro_f1"]) else float(r["l3_macro_f1"])) for i, r in enumerate(maj_rows)}),
+                  ("l3_accuracy", "#0a8754", {i: (None if pd.isna(r["l3_accuracy"]) else float(r["l3_accuracy"])) for i, r in enumerate(maj_rows)})]
+        p.append(_svg_bar_chart("Majority vs seed: macro-F1 и accuracy (comparison_val)",
+                                groups, series, y_fmt="{:.3f}"))
+    else:
+        p.append("<p class='muted'>majority_baseline.json не найден — сравнение недоступно.</p>")
+
+    # --- per-class ---
+    p.append("<h2>Per-class качество и поддержка L3 (comparison_val)</h2>")
+    pc_ok = per_class_df[per_class_df["run_id"].isin(ok_ids)]
+    if len(pc_ok):
+        agg = (pc_ok.groupby("L3", as_index=False)
+               .agg(support=("support", "max"),
+                    precision=("precision", "mean"),
+                    recall=("recall", "mean"),
+                    f1=("f1", "mean")))
+        agg["error_rate"] = 1.0 - agg["recall"]
+        agg = agg.sort_values("support", ascending=False)
+        p.append(_table_html(agg.head(15), ["L3", "support", "precision", "recall", "f1", "error_rate"],
+                             formats={c: _fmt_run_metric for c in
+                                      ("precision", "recall", "f1", "error_rate")},
+                             caption="Топ-15 классов по support; агрегат по успешным seed "
+                                     "(mean). Классы с support=0 имеют null-метрики (—), "
+                                     "не нули."))
+        tail = agg[agg["support"] <= max(2, int(agg["support"].quantile(0.1)))]
+        p.append(f"<p class='muted'>Длинный хвост: {len(tail)} из {len(agg)} классов имеют "
+                 f"support ≤ {max(2, int(agg['support'].quantile(0.1)))} на comparison_val.</p>")
+        worst = agg.sort_values("f1", na_position="last").head(15)
+        p.append(_table_html(worst, ["L3", "support", "precision", "recall", "f1", "error_rate"],
+                             formats={c: _fmt_run_metric for c in
+                                      ("precision", "recall", "f1", "error_rate")},
+                             caption="Классы с наихудшим mean F1 (диаграмма ошибок по "
+                                     "категориям: error_rate = 1 − recall)."))
+        p.append(_svg_bar_chart("Error rate (1 − recall) по категориям: 15 худших",
+                                list(range(len(worst))),
+                                [("error_rate", "#b71540",
+                                  {i: float(r.error_rate) for i, r in enumerate(worst.itertuples())
+                                   if pd.notna(r.error_rate)})],
+                                y_fmt="{:.2f}"))
+    else:
+        p.append("<p class='muted'>Нет per-class данных по успешным прогонам.</p>")
+    if class_support_split is not None:
+        sup = class_support_split
+        kept = sup[sup["kept_after_min_count"].astype(str) == "True"] if "kept_after_min_count" in sup else sup
+        p.append(f"<p class='muted'>Классов после min_count: {len(kept)}; исключено редких L3: "
+                 f"{len(sup) - len(kept)}. Детали — audit/class_support_split.csv "
+                 f"({len(sup)} строк).</p>")
+    elif class_support is not None:
+        p.append(f"<p class='muted'>Классов в аудите разметки: {len(class_support)} "
+                 "(audit/class_support.csv).</p>")
+
+    # --- ресурсы ---
+    p.append("<h2>Ресурсные затраты (stages.csv)</h2>")
+    if len(stages_df):
+        st = stages_df.copy()
+        st["seconds"] = pd.to_numeric(st["seconds"], errors="coerce")
+        st["peak_rss_bytes"] = pd.to_numeric(st["peak_rss_bytes"], errors="coerce")
+        st["peak_vram_bytes"] = pd.to_numeric(st["peak_vram_bytes"], errors="coerce")
+        per_run = (st.groupby(["run_id", "stage", "status"], as_index=False)
+                   .agg(seconds=("seconds", "sum"),
+                        peak_rss_bytes=("peak_rss_bytes", "max"),
+                        peak_vram_bytes=("peak_vram_bytes", "max")))
+        p.append(_table_html(per_run, list(per_run.columns),
+                             formats={"seconds": _fmt_run_metric,
+                                      "peak_rss_bytes": _fmt_mb,
+                                      "peak_vram_bytes": _fmt_mb},
+                             caption="RSS/VRAM в МиБ; '—' = не измерено (null), а не ноль. "
+                                     "run_id=shared_vectorization — разделяемые этапы."))
+        tot = st[st["status"] == "success"]["seconds"].sum()
+        p.append(f"<p class='muted'>Суммарное измеренное время успешных этапов: "
+                 f"{_fmt_run_metric(float(tot)) if pd.notna(tot) else '—'} с. GPU: "
+                 f"{_esc(str(((training_manifest or {}).get('environment') or {}).get('device', '—')))}; "
+                 "недоступность CUDA — 'unavailable', не пройденная проверка.</p>")
+    else:
+        p.append("<p class='muted'>stages.csv пуст.</p>")
+
+    # --- selection / tfidf / checkpoint ---
+    p.append("<h2>Selection (D05), full TF-IDF (D04), checkpoint (D08)</h2>")
+    if selection_report is not None:
+        ch = selection_report.get("chosen") or {}
+        p.append("<div class='note'>Selection (train-only): выбран кандидат "
+                 f"<code>{_esc(str(ch.get('name')))}</code>, фиксированные эпохи "
+                 f"{_esc(str(ch.get('fixed_epochs')))}; критерий: "
+                 f"{_esc(str(ch.get('criterion')))}. Кандидатов: "
+                 f"{len(selection_report.get('candidates') or [])}.</div>")
+    else:
+        p.append("<div class='note'>selection/selection_report.json не найден — "
+                 "протокол selection файлами не подтверждён.</div>")
+    if tfidf_verification is not None:
+        p.append("<div class='note'>Full TF-IDF (D04): decision="
+                 f"<code>{_esc(str(tfidf_verification.get('decision')))}</code>, "
+                 f"artifact_version={_esc(str(tfidf_verification.get('artifact_version')))}, "
+                 f"full_tfidf_id={_esc(str((tfidf_verification.get('full_tfidf_id') or '')[:16]))}…, "
+                 f"corpus_matches_source={_esc(str(tfidf_verification.get('corpus_matches_source')))}.</div>")
+    if selected is not None:
+        rt = selected.get("roundtrip") or {}
+        rule = selected.get("selection_rule") or {}
+        p.append("<div class='note'>Опорный checkpoint для S06: "
+                 f"<code>{_esc(str(selected.get('checkpoint_dir')))}</code>; правило "
+                 f"{_esc(str(rule.get('kind')))}={_esc(str(rule.get('value')))} "
+                 "(train-only, не по comparison_val); round-trip: "
+                 f"<b>{_esc(str(rt.get('status')))}</b>.</div>")
+
+    # --- gate ---
+    p.append("<h2>Решение о готовности к S06 (D)</h2>")
+    if gate["reasons"]:
+        p.append("<ul>" + "".join(f"<li>{_esc(r)}</li>" for r in gate["reasons"]) + "</ul>")
+    else:
+        p.append("<div class='note'>Все проверки gate пройдены: сплиты и TF-IDF зафиксированы, "
+                 "≥3 успешных seed, baseline содержательно лучше majority, checkpoint "
+                 "проверен round-trip.</div>")
+    p.append("<div class='note'>Ограничения: ODS/PoC — результаты диагностические, не оценка "
+             "промышленной применимости; трансдуктивность внешнего full TF-IDF отмечена; "
+             "sealed test не вскрывался, test-метрик нет.</div>")
+    p.append("</body></html>")
+    return "".join(p)
+
+
+def jsonable(v: Any) -> Any:
+    if isinstance(v, (dict, list)):
+        try:
+            import json
+            return json.dumps(v, ensure_ascii=False, default=str)[:400]
+        except Exception:  # noqa: BLE001
+            return str(v)[:400]
+    return v
+
+
+# ---------------------------------------------------------------------------
+# Публичный вход
+# ---------------------------------------------------------------------------
+
+
+def build_s05_report(output_dir: str = "outputs/S05",
+                     report_dir: str | None = None) -> dict[str, str]:
+    """Формирует агрегированный отчёт D09 из машинных результатов S05.
+
+    Читает обязательные ``runs.csv``/``history.csv``/``stages.csv``/
+    ``per_class.parquet``/``manifest.json`` и опциональные сопутствующие файлы
+    из ``output_dir``; пишет шесть файлов отчёта в ``report_dir`` (по умолчанию
+    ``<output_dir>/report``). Существующие файлы (включая исходные CSV обучения
+    в ``output_dir``) не изменяются; при любом пересечении — отказ до записей.
+    """
+    report_dir = report_dir or os.path.join(output_dir, REPORT_DIRNAME)
+    targets = [os.path.join(report_dir, name) for name in REPORT_FILES]
+    _check_new(targets)
+
+    runs_df = _required_frame(os.path.join(output_dir, "runs.csv"), RUNS_REQUIRED, "runs.csv")
+    history_df = _required_frame(os.path.join(output_dir, "history.csv"), HISTORY_REQUIRED, "history.csv")
+    stages_df = _required_frame(os.path.join(output_dir, "stages.csv"), STAGES_REQUIRED, "stages.csv")
+    per_class_df = _required_frame(os.path.join(output_dir, "per_class.parquet"), PER_CLASS_REQUIRED, "per_class.parquet")
+    manifest_path = os.path.join(output_dir, "manifest.json")
+    if not os.path.isfile(manifest_path):
+        raise S05ReportError(
+            "manifest.json не найден в output_dir: агрегированный отчёт требует "
+            "записанного обучающим контуром манифеста (входы, seed, статусы)."
+        )
+    training_manifest = _read_json(manifest_path)
+
+    majority = _optional_json(output_dir, "majority_baseline.json")
+    selected = _optional_json(output_dir, "selected_checkpoint.json")
+    tfidf_verification = _optional_json(output_dir, "tfidf_verification.json")
+    selection_report = _optional_json(output_dir, os.path.join("selection", "selection_report.json"))
+    data_audit = _optional_json(output_dir, os.path.join("audit", "data_audit.json"))
+    split_audit, split_audit_source = _load_split_audit(output_dir)
+    cs_path = os.path.join(output_dir, "audit", "class_support.csv")
+    css_path = os.path.join(output_dir, "audit", "class_support_split.csv")
+    class_support = pd.read_csv(cs_path) if os.path.isfile(cs_path) else None
+    class_support_split = pd.read_csv(css_path) if os.path.isfile(css_path) else None
+
+    runs_full = _with_skipped_runs(runs_df, training_manifest)
+    seed_stats = _seed_stats(runs_full)
+    gap_df = _train_monitor_gap(history_df, runs_full)
+    gate = _decide_gate(runs_full, training_manifest, tfidf_verification,
+                        selected, majority, gap_df, split_audit)
+
+    report_manifest: dict[str, Any] = {
+        "schema_version": S05_REPORT_SCHEMA_VERSION,
+        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
+        "output_dir": os.path.abspath(output_dir),
+        "report_dir": os.path.abspath(report_dir),
+        "gate": gate,
+        "seed_stats_comparison_val": seed_stats,
+        "sources": {},
+        "split_audit_source": split_audit_source,
+        "split_audit_original_error": SPLIT_AUDIT_ORIGINAL_ERROR,
+        "notes": [
+            "Отчёт построен без обучения из машинных файлов обучающего контура.",
+            "Отсутствующие значения сохранены как null, нулями не подменяются.",
+            "Sealed test не использовался; test-метрик нет.",
+            "Метрики качества — raw top-1 L3 на comparison_val без отказа.",
+        ],
+    }
+    for name in ("runs.csv", "history.csv", "stages.csv", "per_class.parquet",
+                 "manifest.json", "majority_baseline.json", "selected_checkpoint.json",
+                 "tfidf_verification.json"):
+        path = os.path.join(output_dir, name)
+        if os.path.isfile(path):
+            report_manifest["sources"][name] = _sha256_file(path)
+    audit_corr = os.path.join(output_dir, "audit", "split_audit_corrected.csv")
+    audit_orig = os.path.join(output_dir, "audit", "split_audit.csv")
+    if os.path.isfile(audit_corr):
+        report_manifest["sources"]["audit/split_audit_corrected.csv"] = _sha256_file(audit_corr)
+    if os.path.isfile(audit_orig):
+        report_manifest["sources"]["audit/split_audit.csv"] = _sha256_file(audit_orig)
+    # Полный контекст обучения — как есть из контура (входы, конфиги, протокол,
+    # hashes, env, seed, статусы).
+    report_manifest["training_manifest"] = training_manifest
+
+    doc = _html_report(
+        output_dir, runs_full, history_df, stages_df, per_class_df,
+        training_manifest, majority, selected, tfidf_verification,
+        selection_report, split_audit, split_audit_source,
+        class_support, class_support_split, data_audit,
+        seed_stats, gap_df, gate, report_manifest,
+    )
+
+    os.makedirs(report_dir, exist_ok=True)
+    runs_full.to_csv(os.path.join(report_dir, "runs.csv"), index=False)
+    per_class_df.to_parquet(os.path.join(report_dir, "per_class.parquet"), index=False)
+    stages_df.to_csv(os.path.join(report_dir, "stages.csv"), index=False)
+    history_df.to_csv(os.path.join(report_dir, "history.csv"), index=False)
+    _write_json(report_manifest, os.path.join(report_dir, "manifest.json"))
+    with open(os.path.join(report_dir, "report.html"), "w", encoding="utf-8") as f:
+        f.write(doc)
+    log.info("Отчёт S05 (D09): %s; gate=%s (%d runs: %d success, %d failed, %d skipped)",
+             report_dir, gate["decision"], gate["n_planned"], gate["n_success"],
+             gate["n_failed"], gate["n_skipped"])
+    return {
+        "report_dir": report_dir,
+        "report_html": os.path.join(report_dir, "report.html"),
+        "runs_csv": os.path.join(report_dir, "runs.csv"),
+        "per_class_parquet": os.path.join(report_dir, "per_class.parquet"),
+        "stages_csv": os.path.join(report_dir, "stages.csv"),
+        "history_csv": os.path.join(report_dir, "history.csv"),
+        "manifest_json": os.path.join(report_dir, "manifest.json"),
+        "gate": gate["decision"],
+    }
diff --git a/scripts/audit_ods.py b/scripts/audit_ods.py
new file mode 100644
index 0000000..a4d9c29
--- /dev/null
+++ b/scripts/audit_ods.py
@@ -0,0 +1,554 @@
+"""S05, этап A: воспроизводимый аудит исходной ODS-разметки до обучения (D02).
+
+Читает исходный ``fixtures/ods.parquet``, воспроизводит шаги подготовки
+``kkt_clf.data.prepare_data`` вплоть до фильтра ``min_count`` (без сплита и
+обучения) и сверяет результат с историческим ориентиром STATUS.md 2026-10-22
+(18 046 строк / 737 L3 → min_count=10 → 16 635 строк / 326 L3, исключено
+411 L3 / 1 411 строк). Проверяет метаданные и фактическую идентичность
+сохранённого full TF-IDF S02 (``outputs/S02/tfidf_1m``) и его источника
+``data/corpus.pqt``. Исследует осуществимость group-aware split с поддержкой
+каждой L3 во всех частях (жадный feasibility-probe, разбиение не сохраняется
+и нигде не используется).
+
+Метки не исправляются, состав данных не меняется, split/обучение не
+выполняются. Результаты пишутся в ``outputs/S05/audit/``; существующие файлы
+не перезаписываются.
+
+Пример:
+    .venv/bin/python scripts/audit_ods.py \\
+        --input fixtures/ods.parquet --min-count 10 \\
+        --out-dir outputs/S05/audit \\
+        --corpus data/corpus.pqt --s02-dir outputs/S02
+"""
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import json
+import os
+import platform
+import subprocess
+import sys
+
+import numpy as np
+import pandas as pd
+
+sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+
+from kkt_clf.data import (  # noqa: E402
+    REQUIRED_COLUMNS,
+    file_fingerprint,
+    taxonomy_violations,
+)
+from kkt_clf.features import preprocess, strip_whitespace  # noqa: E402
+from kkt_clf.vectorizer import VectorizerArtifactError, load_vectorizer_artifact  # noqa: E402
+
+# Исторический ориентир STATUS.md 2026-10-22 (подлежит перепроверке, не факт).
+HISTORICAL_EXPECTATION = {
+    "source": "artefacts/S05/value_dod.md + STATUS.md 2026-10-22 (ориентир, не измерение)",
+    "rows": 18_046,
+    "n_l3_raw": 737,
+    "rows_after_min_count": 16_635,
+    "n_l3_after_min_count": 326,
+    "rare_l3_classes": 411,
+    "rare_l3_rows": 1_411,
+}
+
+DEFAULT_SPLIT = (0.8, 0.1, 0.1)
+
+
+def _git_info() -> dict:
+    def _run(*args: str) -> str | None:
+        try:
+            return subprocess.run(
+                ["git", *args], capture_output=True, text=True, timeout=10
+            ).stdout.strip() or None
+        except Exception:
+            return None
+
+    return {"branch": _run("branch", "--show-current"), "head": _run("rev-parse", "HEAD")}
+
+
+def _versions() -> dict:
+    import pyarrow
+    import sklearn
+
+    return {
+        "python": platform.python_version(),
+        "numpy": np.__version__,
+        "pandas": pd.__version__,
+        "sklearn": sklearn.__version__,
+        "pyarrow": pyarrow.__version__,
+        "platform": platform.platform(),
+    }
+
+
+def _read_meta(path: str) -> dict:
+    with open(path, encoding="utf-8") as f:
+        return json.load(f)
+
+
+def word_key(text: str) -> str:
+    """Ключ word-ветки: preprocess + схлопывание пробелов (эквивалент join токенов)."""
+    return " ".join(text.split())
+
+
+def add_keys(df: pd.DataFrame) -> pd.DataFrame:
+    """Три ключа идентичности текста: preprocess(lower), char-ветка, word-ветка."""
+    out = df.copy()
+    pre = preprocess(out["name"])
+    out["_pre"] = pre
+    out["_char"] = [strip_whitespace(t) for t in pre]
+    out["_word"] = [word_key(t) for t in pre]
+    return out
+
+
+def key_stats(df: pd.DataFrame, key: str) -> dict:
+    """Статистика коллизий ключа и конфликтов меток внутри коллизионных групп."""
+    sizes = df.groupby(key).size()
+    groups = df.groupby(key).agg(
+        n_rows=(key, "size"),
+        n_paths=("path", "nunique"),
+        n_l3=("L3", "nunique"),
+        n_l2=("L2", "nunique"),
+        n_l1=("L1", "nunique"),
+    )
+    collided = groups[groups["n_rows"] > 1]
+    conflicting = groups[groups["n_paths"] > 1]
+    multi_l3 = groups[groups["n_l3"] > 1]
+    return {
+        "n_unique_keys": int(groups.shape[0]),
+        "n_collided_groups": int(len(collided)),
+        "n_rows_in_collided_groups": int(collided["n_rows"].sum()),
+        "n_extra_rows_vs_unique": int(len(df) - groups.shape[0]),
+        "n_groups_multi_path": int(len(conflicting)),
+        "n_rows_in_multi_path_groups": int(conflicting["n_rows"].sum()),
+        "n_groups_multi_l3": int(len(multi_l3)),
+        "n_rows_in_multi_l3_groups": int(multi_l3["n_rows"].sum()),
+        "max_rows_per_key": int(sizes.max()),
+        "examples_multi_path": [
+            {key: idx, "n_rows": int(r.n_rows), "paths": int(r.n_paths), "l3": int(r.n_l3)}
+            for idx, r in conflicting.head(10).iterrows()
+        ],
+    }
+
+
+def group_split_feasibility(df: pd.DataFrame, key: str, fractions: tuple[float, float, float]) -> dict:
+    """Аналитика осуществимости group-aware split с поддержкой каждой L3 в каждой части.
+
+    Выполняется только жадный feasibility-probe: разбиение не сохраняется,
+    не возвращается и нигде дальше не используется. Покрытие считается по
+    множеству L3, которые группа несёт (у конфликтной группы их несколько).
+    """
+    g = df.groupby(key).agg(n_rows=(key, "size"), l3s=("L3", lambda s: frozenset(s)))
+    l3_counts = df["L3"].value_counts()
+    groups_per_l3 = df.groupby("L3")[key].nunique()
+
+    # Жадный probe: для каждой части покрыть все L3, минимизируя число групп.
+    names = g.index.to_numpy()
+    l3_sets = g["l3s"].tolist()
+    sizes = g["n_rows"].to_numpy()
+    all_l3 = set(df["L3"].unique())
+    order = np.argsort(-sizes)  # крупные группы первыми
+    assigned: dict[str, int] = {}
+    covered = [set() for _ in fractions]
+    target = [f * len(df) for f in fractions]
+    part_rows = [0.0 for _ in fractions]
+    iter_guard = 0
+    while True:
+        iter_guard += 1
+        if iter_guard > len(g) + 10:
+            break
+        # часть с наименьшей заполненностью квоты, у которой есть непокрытые L3
+        progress = False
+        fill = [part_rows[p] / target[p] if target[p] else 1.0 for p in range(3)]
+        parts_sorted = np.argsort(fill)
+        for p in parts_sorted:
+            missing = all_l3 - covered[p]
+            if not missing:
+                continue
+            # кандидаты: нераспределённые группы, несущие хотя бы один missing L3
+            best, best_gain = None, 0
+            for i in order:
+                k = names[i]
+                if k in assigned:
+                    continue
+                gain = len(l3_sets[i] & missing)
+                if gain > best_gain or (gain == best_gain and best_gain > 0 and sizes[i] > sizes[best]):
+                    best, best_gain = i, gain
+            if best is None or best_gain == 0:
+                continue
+            k = names[best]
+            assigned[k] = p
+            covered[p] |= l3_sets[best]
+            part_rows[p] += sizes[best]
+            progress = True
+            break
+        if not progress:
+            break
+
+    # Доброс оставшихся групп в наименее заполненные относительно квоты части:
+    # оценка, насколько близко к 80/10/10 можно приблизить group split.
+    for i in order:
+        k = names[i]
+        if k in assigned:
+            continue
+        fill = [part_rows[p] / target[p] if target[p] else float("inf") for p in range(3)]
+        p = int(np.argmin(fill))
+        assigned[k] = p
+        covered[p] |= l3_sets[i]
+        part_rows[p] += sizes[i]
+
+    n_l3 = len(all_l3)
+    coverage = {p: len(covered[p]) for p in range(3)}
+    feasible = all(c == n_l3 for c in coverage.values())
+    unassigned_groups = int(len(g) - len(assigned))
+    return {
+        "n_groups": int(len(g)),
+        "n_conflict_groups_multi_l3": int((g["l3s"].map(len) > 1).sum()),
+        "rows_in_conflict_groups": int(g.loc[g["l3s"].map(len) > 1, "n_rows"].sum()),
+        "l3_with_lt_3_groups": int((groups_per_l3 < 3).sum()),
+        "l3_with_1_group": int((groups_per_l3 == 1).sum()),
+        "min_groups_per_l3": int(groups_per_l3.min()),
+        "median_groups_per_l3": float(groups_per_l3.median()),
+        "top_group_share_max": float(
+            df.groupby(["L3", key]).size().groupby("L3").max().div(l3_counts).max()
+        ),
+        "greedy_probe": {
+            "method": "жадное покрытие всех L3 в каждой части крупнейшими нераспределёнными группами, "
+            "затем добор оставшихся групп в наименее заполненные относительно квоты части; "
+            "разбиение не сохраняется (исследование осуществимости, не сплит)",
+            "all_l3_covered_in_all_parts": feasible,
+            "coverage_train_val_test": [coverage[0], coverage[1], coverage[2]],
+            "n_l3_total": n_l3,
+            "probe_part_rows": [int(r) for r in part_rows],
+            "probe_part_share": [round(r / len(df), 4) for r in part_rows],
+            "target_share": list(fractions),
+            "n_unassigned_groups": unassigned_groups,
+            "blocking_l3_examples": sorted(all_l3 - covered[min(range(3), key=lambda p: coverage[p])])[:10]
+            if not feasible
+            else [],
+        },
+    }
+
+
+def audit_s02_tfidf(s02_dir: str, corpus_path: str) -> dict:
+    """Сверка метаданных S02 full TF-IDF с фактическими файлами (read-only)."""
+    out: dict = {}
+    for name in ("tfidf_1m", "tfidf_100k"):
+        art_dir = os.path.join(s02_dir, name)
+        meta_path = os.path.join(art_dir, "vectorizer_meta.json")
+        entry: dict = {"artifact_dir": os.path.abspath(art_dir)}
+        if not os.path.exists(meta_path):
+            entry["status"] = "missing"
+            out[name] = entry
+            continue
+        meta = _read_meta(meta_path)
+        entry["meta_created_at"] = meta.get("created_at")
+        entry["meta_vocab"] = meta.get("vocab")
+        entry["meta_source_id"] = meta.get("source", {}).get("source_id")
+        src = meta.get("source", {})
+        entry["source_expected"] = {"sha256": src.get("sha256"), "bytes": src.get("bytes"),
+                                    "rows_in_file": src.get("rows_in_file"),
+                                    "rows_used": src.get("rows_used"),
+                                    "max_rows": src.get("max_rows"),
+                                    "truncated": src.get("truncated")}
+        fp = file_fingerprint(corpus_path)
+        entry["corpus_actual"] = {"sha256": fp["sha256"], "bytes": fp["bytes"]}
+        entry["corpus_sha256_match"] = fp["sha256"] == src.get("sha256")
+        entry["corpus_bytes_match"] = fp["bytes"] == src.get("bytes")
+        import pyarrow.parquet as pq
+
+        entry["corpus_rows_in_file_actual"] = int(pq.ParquetFile(corpus_path).metadata.num_rows)
+        entry["corpus_rows_in_file_match"] = (
+            entry["corpus_rows_in_file_actual"] == src.get("rows_in_file")
+        )
+        fjob = os.path.join(art_dir, "features.joblib")
+        finfo = (meta.get("files") or {}).get("features.joblib", {})
+        entry["features_joblib_expected"] = finfo
+        entry["features_joblib_actual"] = {
+            "sha256": file_fingerprint(fjob)["sha256"],
+            "bytes": os.path.getsize(fjob),
+        }
+        entry["features_joblib_sha256_match"] = (
+            entry["features_joblib_actual"]["sha256"] == finfo.get("sha256")
+        )
+        entry["features_joblib_bytes_match"] = (
+            entry["features_joblib_actual"]["bytes"] == finfo.get("bytes")
+        )
+        try:
+            loaded = load_vectorizer_artifact(art_dir)
+            entry["load_check"] = "ok"
+            entry["loaded_vocab"] = {
+                "n_char": loaded.builder.n_char_features,
+                "n_word": loaded.builder.n_word_features,
+                "n_total": loaded.builder.n_features,
+            }
+            entry["loaded_vocab_match_meta"] = entry["loaded_vocab"] == meta.get("vocab")
+            entry["loaded_fit_report"] = dict(loaded.builder.fit_report_)
+        except VectorizerArtifactError as e:
+            entry["load_check"] = f"failed: {e}"
+        out[name] = entry
+    return out
+
+
+def main() -> None:
+    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
+    p.add_argument("--input", default="fixtures/ods.parquet")
+    p.add_argument("--min-count", type=int, default=10)
+    p.add_argument("--out-dir", default="outputs/S05/audit")
+    p.add_argument("--corpus", default="data/corpus.pqt")
+    p.add_argument("--s02-dir", default="outputs/S02")
+    p.add_argument("--legacy-csv", default="data/ods_labels.csv")
+    p.add_argument("--split", type=float, nargs=3, default=list(DEFAULT_SPLIT))
+    a = p.parse_args()
+
+    os.makedirs(a.out_dir, exist_ok=True)
+    audit_path = os.path.join(a.out_dir, "data_audit.json")
+    support_path = os.path.join(a.out_dir, "class_support.csv")
+    for path in (audit_path, support_path):
+        if os.path.exists(path):
+            raise SystemExit(f"Отказ: файл уже существует и не будет перезаписан: {path}")
+
+    started = dt.datetime.now(dt.timezone.utc)
+
+    # --- Исходный файл и пустоты (шаги 1–3 prepare_data, без изменения данных).
+    fp_in = file_fingerprint(a.input)
+    df_raw = pd.read_parquet(a.input, engine="pyarrow")
+    df = df_raw[list(REQUIRED_COLUMNS)].copy()
+    rows_read = len(df)
+
+    bad = np.zeros(len(df), dtype=bool)
+    empty_counts: dict[str, int] = {}
+    for c in REQUIRED_COLUMNS:
+        col = df[c]
+        mask = col.isna().to_numpy()
+        notna = ~mask
+        if notna.any():
+            mask[notna] = col[notna].astype(str).str.strip().eq("").to_numpy()
+        empty_counts[c] = int(mask.sum())
+        bad |= mask
+    empty_counts["total"] = int(bad.sum())
+    df = df.loc[~bad].reset_index(drop=True)
+    rows_after_drop = len(df)
+
+    # --- Пути меток и конфликты.
+    df["path"] = df["L1"] + " / " + df["L2"] + " / " + df["L3"]
+    violations = taxonomy_violations(df)
+    name_paths = df.groupby("name")["path"].nunique()
+    name_l3 = df.groupby("name")["L3"].nunique()
+    label_conflicts = {
+        "names_with_multi_path": int((name_paths > 1).sum()),
+        "names_with_multi_l3": int((name_l3 > 1).sum()),
+        "rows_in_multi_path_names": int(df.loc[df["name"].map(name_paths) > 1].shape[0]),
+        "examples": [
+            {"name": n, "n_paths": int(name_paths[n])}
+            for n in name_paths[name_paths > 1].head(10).index
+        ],
+    }
+
+    # --- Дубли: исходные и нормализованные ключи (тот же preprocessing, что TF-IDF).
+    dup_mask = df["name"].duplicated(keep=False)
+    dup_names = df.loc[dup_mask, "name"].nunique()
+    dups_raw = {
+        "n_duplicate_extra_rows": int(df["name"].duplicated(keep="first").sum()),
+        "n_names_appearing_more_than_once": int(dup_names),
+        "n_names_with_multi_path": int((name_paths[df.loc[dup_mask, "name"].unique()] > 1).sum())
+        if dup_names
+        else 0,
+    }
+
+    dfk = add_keys(df)
+    keys_report = {
+        "raw_name": {
+            "n_unique": int(df["name"].nunique()),
+            "n_collided_groups": int((df.groupby("name").size() > 1).sum()),
+            "n_extra_rows_vs_unique": int(len(df) - df["name"].nunique()),
+        },
+        "preprocessed_lower": key_stats(dfk, "_pre"),
+        "char_branch_strip_whitespace": key_stats(dfk, "_char"),
+        "word_branch_collapse_ws": key_stats(dfk, "_word"),
+    }
+
+    # --- Классы и частоты до фильтра.
+    l3_counts = df["L3"].value_counts()
+    l3_stats = {
+        "n_l1": int(df["L1"].nunique()),
+        "n_l2": int(df["L2"].nunique()),
+        "n_l3": int(df["L3"].nunique()),
+        "l3_freq": {
+            "min": int(l3_counts.min()),
+            "p25": float(l3_counts.quantile(0.25)),
+            "median": float(l3_counts.median()),
+            "p75": float(l3_counts.quantile(0.75)),
+            "p90": float(l3_counts.quantile(0.90)),
+            "p99": float(l3_counts.quantile(0.99)),
+            "max": int(l3_counts.max()),
+        },
+        "top10_l3": [
+            {"L3": k, "count": int(v)} for k, v in l3_counts.head(10).items()
+        ],
+        "n_l3_below_min_count": int((l3_counts < a.min_count).sum()),
+        "n_l3_equal_1": int((l3_counts == 1).sum()),
+    }
+
+    # --- Фильтр min_count (без изменения исходных данных).
+    counts = df["L3"].value_counts()
+    rare = counts[counts < a.min_count]
+    kept_mask = ~df["L3"].isin(rare.index)
+    rows_after = int(kept_mask.sum())
+    n_l3_after = int(df.loc[kept_mask, "L3"].nunique())
+    rare_pairs = (
+        df.loc[~kept_mask, ["L1", "L2", "L3"]]
+        .drop_duplicates("L3")
+        .assign(count=lambda x: x["L3"].map(rare).astype(int))
+        .sort_values(["count", "L3"], ascending=[False, True])
+    )
+    min_count_report = {
+        "min_count": a.min_count,
+        "n_excluded_l3": int(len(rare)),
+        "n_excluded_rows": int(rare.sum()),
+        "share_rows_excluded": round(float(rare.sum()) / len(df), 6),
+        "rows_after": rows_after,
+        "n_l3_after": n_l3_after,
+        "excluded_examples_top": rare_pairs.head(20).to_dict("records"),
+        "excluded_l3_full_list": rare_pairs[["L1", "L2", "L3", "count"]].to_dict("records"),
+    }
+
+    # --- Таблица поддержки всех классов (до и после фильтра).
+    support = (
+        df.groupby(["L1", "L2", "L3"]).size().reset_index(name="count_raw")
+        .sort_values("count_raw", ascending=False, ignore_index=True)
+    )
+    support["share_raw"] = (support["count_raw"] / len(df)).round(6)
+    support["kept_after_min_count"] = support["L3"].isin(set(counts[counts >= a.min_count].index))
+    support["count_after_min_count"] = np.where(
+        support["kept_after_min_count"], support["count_raw"], 0
+    )
+    support.to_csv(support_path, index=False)
+
+    # --- Осуществимость group-aware split (после min_count).
+    df_kept = df.loc[kept_mask].reset_index(drop=True)
+    dfk_kept = dfk.loc[kept_mask].reset_index(drop=True)
+    dfk_kept["path"] = df.loc[kept_mask, "path"].to_numpy()
+    group_split = {
+        "fractions_target": list(a.split),
+        "char_branch_key": group_split_feasibility(dfk_kept, "_char", tuple(a.split)),
+        "word_branch_key": group_split_feasibility(dfk_kept, "_word", tuple(a.split)),
+        "note": "probe не является сплитом; разбиения не сохранялись и нигде не используются",
+    }
+
+    # --- Пересечение текстов ODS с неразмеченным корпусом (трансдуктивность).
+    corpus_fp = file_fingerprint(a.corpus)
+    corpus_names = pd.read_parquet(a.corpus, columns=["name"], engine="pyarrow")["name"]
+    corpus_keys = set(strip_whitespace(t) for t in preprocess(corpus_names.tolist()))
+    ods_keys = set(dfk_kept["_char"])
+    overlap = ods_keys & corpus_keys
+    corpus_overlap = {
+        "corpus_path": corpus_fp["path"],
+        "corpus_sha256": corpus_fp["sha256"],
+        "corpus_unique_nonempty_names_approx": int(len(corpus_keys)),
+        "ods_kept_unique_char_keys": int(len(ods_keys)),
+        "ods_kept_char_keys_found_in_corpus": int(len(overlap)),
+        "share_ods_keys_in_corpus": round(len(overlap) / len(ods_keys), 6) if ods_keys else 0.0,
+        "interpretation": "совпадение текстов с неразмеченным корпусом не передаёт метки, "
+        "но делает оценку частично трансдуктивной (см. S02 notes)",
+    }
+
+    # --- S02 full TF-IDF: метаданные против фактических файлов.
+    s02_check = audit_s02_tfidf(a.s02_dir, a.corpus)
+
+    # --- Сверка с legacy CSV (provenance перекодировки).
+    legacy = {"path": os.path.abspath(a.legacy_csv)}
+    if os.path.exists(a.legacy_csv):
+        df_csv = pd.read_csv(a.legacy_csv, dtype=str, keep_default_na=False)
+        legacy["rows"] = int(len(df_csv))
+        legacy["columns"] = list(df_csv.columns)
+        key_cols = list(REQUIRED_COLUMNS)
+        a_set = set(map(tuple, df[key_cols].astype(str).to_numpy()))
+        b_set = set(map(tuple, df_csv[key_cols].astype(str).to_numpy()))
+        legacy["set_equal_to_parquet_after_drop_empty"] = a_set == b_set
+        legacy["only_in_parquet"] = int(len(a_set - b_set))
+        legacy["only_in_csv"] = int(len(b_set - a_set))
+
+    # --- Историческое сопоставление.
+    reconciliation = {
+        "historical": HISTORICAL_EXPECTATION,
+        "actual": {
+            "rows": rows_read,
+            "n_l3_raw": l3_stats["n_l3"],
+            "rows_after_min_count": rows_after,
+            "n_l3_after_min_count": n_l3_after,
+            "rare_l3_classes": int(len(rare)),
+            "rare_l3_rows": int(rare.sum()),
+        },
+        "match": {
+            k: HISTORICAL_EXPECTATION[k] == (
+                rows_read if k == "rows"
+                else l3_stats["n_l3"] if k == "n_l3_raw"
+                else rows_after if k == "rows_after_min_count"
+                else n_l3_after if k == "n_l3_after_min_count"
+                else int(len(rare)) if k == "rare_l3_classes"
+                else int(rare.sum())
+            )
+            for k in ("rows", "n_l3_raw", "rows_after_min_count", "n_l3_after_min_count",
+                      "rare_l3_classes", "rare_l3_rows")
+        },
+    }
+
+    audit = {
+        "generated_at": started.isoformat(timespec="seconds"),
+        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
+        "generator": {
+            "script": os.path.abspath(__file__),
+            "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]),
+            "cwd": os.getcwd(),
+            "git": _git_info(),
+            "versions": _versions(),
+        },
+        "inputs": {
+            "ods_parquet": fp_in,
+            "legacy_csv": legacy,
+            "corpus_pqt": corpus_fp,
+        },
+        "audit": {
+            "rows_read": rows_read,
+            "columns": list(df_raw.columns),
+            "empty_values": empty_counts,
+            "rows_after_drop_empty": rows_after_drop,
+            "taxonomy_violations": {
+                "n": int(len(violations)),
+                "examples": violations.head(20).to_dict("records") if len(violations) else [],
+            },
+            "label_conflicts_raw_name": label_conflicts,
+            "duplicates": {"raw_name": dups_raw, "normalized_keys": keys_report},
+            "classes": l3_stats,
+            "min_count_filter": min_count_report,
+            "group_split_feasibility": group_split,
+            "corpus_text_overlap": corpus_overlap,
+            "s02_tfidf_check": s02_check,
+            "historical_reconciliation": reconciliation,
+        },
+        "limitations": [
+            "Аудит только читает данные; метки не исправлялись, сплит и обучение не выполнялись.",
+            "group_split_feasibility — жадный probe покрытия, не сплит; разбиение не сохранялось.",
+            "Пересечение с corpus.pqt — наблюдение трансдуктивности, не утечка меток.",
+            "Исторические числа сверены как ориентир; расхождения см. в historical_reconciliation.",
+        ],
+    }
+    with open(audit_path, "w", encoding="utf-8") as f:
+        json.dump(audit, f, ensure_ascii=False, indent=2, default=str)
+
+    print(json.dumps({
+        "audit_json": os.path.abspath(audit_path),
+        "class_support_csv": os.path.abspath(support_path),
+        "rows_read": rows_read,
+        "rows_after_min_count": rows_after,
+        "n_l3_raw": l3_stats["n_l3"],
+        "n_l3_after": n_l3_after,
+        "historical_match": reconciliation["match"],
+    }, ensure_ascii=False, indent=2))
+
+
+if __name__ == "__main__":
+    main()
diff --git a/scripts/freeze_split.py b/scripts/freeze_split.py
new file mode 100644
index 0000000..baa67be
--- /dev/null
+++ b/scripts/freeze_split.py
@@ -0,0 +1,90 @@
+"""S05, этап B (D03): заморозка воспроизводимого group-aware сплита ODS.
+
+Читает размеченный parquet, применяет объявленную политику ``min_count=10``
+без молчаливого удаления (исключённые row ids фиксируются в manifest) и
+строит детерминированный стратифицированный group-aware сплит 80/10/10
+(outer train / comparison validation / sealed test) плюс train-internal
+monitor (train_fit / train_monitor) внутри outer train. Группировка — по
+реальному preprocessing обеих TF-IDF-веток (kkt_clf.features): одинаковые
+после ``strip_whitespace(preprocess(name))`` строки всегда попадают в одну
+часть, поэтому пересечений нормализованных текстов между частями нет.
+
+Метрики качества не вычисляются; sealed test используется только для
+count/hash/support-проверок целостности. При конфликтных метках, пустых
+значениях, нарушении таксономии, недостижимой стратификации или недопустимом
+отклонении долей частей — отказ без записи файлов. Существующие outputs не
+перезаписываются.
+
+Пример:
+    .venv/bin/python scripts/freeze_split.py \\
+        --input fixtures/ods.parquet --min-count 10 \\
+        --out-dir outputs/S05/splits --audit-dir outputs/S05/audit
+"""
+from __future__ import annotations
+
+import argparse
+import json
+import os
+import sys
+
+import pandas as pd
+
+sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+
+from kkt_clf.data import REQUIRED_COLUMNS, read_parquet  # noqa: E402
+from kkt_clf.freeze_split import (  # noqa: E402
+    FreezeSplitError,
+    freeze_split,
+    save_frozen_split,
+)
+
+DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)
+
+
+def main() -> int:
+    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
+    p.add_argument("--input", default="fixtures/ods.parquet", help="размеченный parquet (name, L1, L2, L3)")
+    p.add_argument("--min-count", type=int, default=10)
+    p.add_argument("--fractions", type=float, nargs=3, default=list(DEFAULT_FRACTIONS),
+                   help="доли outer train/val/test")
+    p.add_argument("--monitor-fraction", type=float, default=0.1,
+                   help="доля train_monitor внутри outer train")
+    p.add_argument("--seed", type=int, default=42, help="seed разбиения (train seed — отдельно)")
+    p.add_argument("--share-tolerance", type=float, default=0.05,
+                   help="допуск отклонения фактических долей частей от целевых")
+    p.add_argument("--out-dir", default="outputs/S05/splits")
+    p.add_argument("--audit-dir", default="outputs/S05/audit")
+    a = p.parse_args()
+
+    try:
+        df = read_parquet(a.input)
+        df = df[list(REQUIRED_COLUMNS)]
+        frozen = freeze_split(
+            df,
+            min_count=a.min_count,
+            fractions=tuple(a.fractions),
+            monitor_fraction=a.monitor_fraction,
+            seed=a.seed,
+            share_tolerance=a.share_tolerance,
+        )
+        summary = save_frozen_split(
+            frozen,
+            input_path=a.input,
+            out_dir=a.out_dir,
+            audit_dir=a.audit_dir,
+            min_count=a.min_count,
+            fractions=tuple(a.fractions),
+            monitor_fraction=a.monitor_fraction,
+            seed=a.seed,
+            share_tolerance=a.share_tolerance,
+        )
+    except FreezeSplitError as e:
+        print(f"error: {e}", file=sys.stderr)
+        return 1
+
+    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
diff --git a/scripts/run_s05_baseline.py b/scripts/run_s05_baseline.py
new file mode 100644
index 0000000..5028610
--- /dev/null
+++ b/scripts/run_s05_baseline.py
@@ -0,0 +1,202 @@
+"""S05 (D04–D08): обучающий контур baseline на полном TF-IDF (CLI).
+
+Подкоманды:
+
+- ``verify-tfidf`` — D04: проверка единого сохранённого full-TFIDF S02
+  (целостность артефакта + совпадение корпуса с provenance) и запись
+  ``tfidf_verification.json``.
+- ``select`` — D05: train-only выбор числа эпох/конфигурации на
+  train_fit/train_monitor; ``selection/selection_report.json`` +
+  ``selection/curves_train_monitor.csv``.
+- ``refit`` — D06–D08: финальный refit на outer_train с фиксированными эпохами
+  (из selection либо ``--fixed-epochs``) для seed 42/43/44, одна оценка
+  raw top-1 на comparison_val после всех refit, majority baseline, выбор
+  опорного checkpoint по ``--reference-seed`` с round-trip проверкой;
+  ``runs.csv``, ``per_class.parquet``, ``history.csv``, ``stages.csv``,
+  ``majority_baseline.json``, ``selected_checkpoint.json``, ``manifest.json``,
+  ``checkpoints/seed_<s>/``, ``refit/``.
+- ``all`` — verify + select + refit одним запуском.
+
+Существующие файлы не перезаписываются: при обнаружении целевого пути —
+отказ до любых записей. Sealed test не используется; comparison_val не
+используется до завершения всех refit и не используется для выбора checkpoint.
+
+Пример (реальный прогон после ревью, только новые файлы под outputs/S05):
+
+    python scripts/run_s05_baseline.py all \\
+        --splits-manifest outputs/S05/splits/manifest.json \\
+        --full-tfidf outputs/S02/tfidf_1m \\
+        --corpus data/corpus.pqt \\
+        --output outputs/S05 \\
+        --seeds 42,43,44 --reference-seed 42
+"""
+import argparse
+import json
+import os
+import sys
+
+sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+
+from kkt_clf import setup_logging  # noqa: E402
+from kkt_clf.s05_baseline import (  # noqa: E402
+    S05BaselineError,
+    S05Config,
+    load_split_manifest,
+    read_selection,
+    run_refit,
+    run_selection,
+    run_training_contour,
+    verify_full_tfidf,
+)
+
+
+def _json_arg(text: str) -> dict:
+    try:
+        value = json.loads(text)
+    except json.JSONDecodeError as e:
+        raise argparse.ArgumentTypeError(f"ожидался JSON-объект: {e}") from e
+    if not isinstance(value, dict):
+        raise argparse.ArgumentTypeError(f"ожидался JSON-объект, получено {type(value).__name__}")
+    return value
+
+
+def _seeds_arg(text: str) -> list[int]:
+    try:
+        seeds = [int(x) for x in text.split(",") if x.strip()]
+    except ValueError as e:
+        raise argparse.ArgumentTypeError(f"seed через запятую, напр. 42,43,44: {e}") from e
+    if not seeds:
+        raise argparse.ArgumentTypeError("нужен хотя бы один seed")
+    return seeds
+
+
+def _common(p: argparse.ArgumentParser) -> None:
+    p.add_argument("--splits-manifest", default="outputs/S05/splits/manifest.json",
+                   help="manifest замороженного сплита (freeze_split)")
+    p.add_argument("--full-tfidf", default="outputs/S02/tfidf_1m",
+                   help="каталог сохранённого full-TFIDF S02 (pretrained)")
+    p.add_argument("--corpus", default="data/corpus.pqt",
+                   help="корпус full-TFIDF для проверки provenance (sha256)")
+    p.add_argument("--output", default="outputs/S05", help="каталог результатов (только новые файлы)")
+    p.add_argument("--preset", default="ods_light",
+                   help="базовый пресет MLP/loss/train (default: ods_light)")
+    p.add_argument("--model", type=_json_arg, default=None, help="JSON-переопределения model.*")
+    p.add_argument("--loss", type=_json_arg, default=None, help="JSON-переопределения loss.*")
+    p.add_argument("--train", type=_json_arg, default=None, help="JSON-переопределения train.*")
+    p.add_argument("--features-chunk-rows", type=int, default=50_000)
+    p.add_argument("--selection-seed", type=int, default=42)
+    p.add_argument("--seeds", type=_seeds_arg, default=[42, 43, 44],
+                   help="train seed refit через запятую (default: 42,43,44)")
+    p.add_argument("--reference-seed", type=int, default=42,
+                   help="опорный checkpoint для S06 — train-only правило (default: 42)")
+    p.add_argument("--fixed-epochs", type=int, default=None,
+                   help="число эпох refit; без флага — из selection_report.json")
+    p.add_argument("--device", default="auto")
+
+
+def _cfg_from(a: argparse.Namespace) -> S05Config:
+    return S05Config(
+        output_dir=a.output,
+        splits_manifest=a.splits_manifest,
+        full_tfidf_path=a.full_tfidf,
+        corpus_path=a.corpus,
+        preset=a.preset,
+        model=a.model or {},
+        loss=a.loss or {},
+        train=a.train or {},
+        features_chunk_rows=a.features_chunk_rows,
+        selection_seed=a.selection_seed,
+        seeds=list(a.seeds),
+        reference_seed=a.reference_seed,
+        fixed_epochs=a.fixed_epochs,
+        device=a.device,
+    )
+
+
+def main(argv: list[str] | None = None) -> int:
+    parser = argparse.ArgumentParser(
+        description="S05 (D04–D08): обучающий контур baseline на полном TF-IDF",
+        formatter_class=argparse.RawDescriptionHelpFormatter,
+        epilog=__doc__,
+    )
+    sub = parser.add_subparsers(dest="cmd", required=True)
+
+    p_verify = sub.add_parser("verify-tfidf", help="D04: проверка full-TFIDF + tfidf_verification.json")
+    _common(p_verify)
+
+    p_select = sub.add_parser("select", help="D05: train-only выбор эпох/конфигурации")
+    _common(p_select)
+    p_select.add_argument("--sweep", type=_json_arg, default=None,
+                          help="JSON-список кандидатов [{name, model?, train?}] для ограниченного выбора по monitor")
+
+    p_refit = sub.add_parser("refit", help="D06–D08: refit на outer_train + оценка + checkpoint")
+    _common(p_refit)
+
+    p_all = sub.add_parser("all", help="verify + select + refit одним запуском")
+    _common(p_all)
+    p_all.add_argument("--sweep", type=_json_arg, default=None,
+                       help="JSON-список кандидатов selection [{name, model?, train?}]")
+
+    a = parser.parse_args(argv)
+    setup_logging()
+    try:
+        if a.cmd == "verify-tfidf":
+            cfg = _cfg_from(a)
+            result = verify_full_tfidf(cfg.full_tfidf_path, cfg.corpus_path)
+            from kkt_clf.s05_baseline import _check_new, _write_json  # локально, чтобы не экспортировать
+
+            path = os.path.join(cfg.output_dir, "tfidf_verification.json")
+            _check_new([path])
+            os.makedirs(cfg.output_dir, exist_ok=True)
+            _write_json(result, path)
+            print(json.dumps({"decision": result["decision"],
+                              "full_tfidf_id": result["full_tfidf_id"],
+                              "vocab": result["vocab"], "path": path},
+                             ensure_ascii=False, indent=2))
+            return 0
+        if a.cmd == "select":
+            cfg = _cfg_from(a)
+            cfg.sweep = a.sweep or []
+            cfg.validate()
+            from kkt_clf.train import resolve_device
+            from kkt_clf.vectorizer import load_vectorizer_artifact
+
+            manifest = load_split_manifest(cfg.splits_manifest)
+            loaded = load_vectorizer_artifact(cfg.full_tfidf_path)
+            report = run_selection(cfg, manifest, loaded, resolve_device(cfg.device))
+            print(json.dumps({"chosen": report["chosen"],
+                              "candidates": [c["name"] for c in report["candidates"]]},
+                             ensure_ascii=False, indent=2))
+            return 0
+        if a.cmd == "refit":
+            cfg = _cfg_from(a)
+            cfg.validate()
+            from kkt_clf.train import resolve_device
+            from kkt_clf.vectorizer import load_vectorizer_artifact
+
+            manifest = load_split_manifest(cfg.splits_manifest)
+            loaded = load_vectorizer_artifact(cfg.full_tfidf_path)
+            selection = None if cfg.fixed_epochs is not None else read_selection(cfg.output_dir)
+            manifest_out = run_refit(cfg, manifest, loaded, resolve_device(cfg.device), selection=selection)
+            print(json.dumps({"statuses": manifest_out["statuses"],
+                              "seed_stats": manifest_out["seed_stats_comparison_val"],
+                              "output": os.path.abspath(cfg.output_dir)},
+                             ensure_ascii=False, indent=2))
+            return 0
+        if a.cmd == "all":
+            cfg = _cfg_from(a)
+            cfg.sweep = a.sweep or []
+            manifest_out = run_training_contour(cfg)
+            print(json.dumps({"statuses": manifest_out["statuses"],
+                              "seed_stats": manifest_out["seed_stats_comparison_val"],
+                              "output": os.path.abspath(cfg.output_dir)},
+                             ensure_ascii=False, indent=2))
+            return 0
+        raise S05BaselineError(f"неизвестная команда {a.cmd!r}")
+    except S05BaselineError as e:
+        print(f"ОШИБКА: {e}", file=sys.stderr)
+        return 2
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
diff --git a/scripts/run_s05_report.py b/scripts/run_s05_report.py
new file mode 100644
index 0000000..1b7a508
--- /dev/null
+++ b/scripts/run_s05_report.py
@@ -0,0 +1,51 @@
+"""S05 (D09): агрегированный отчёт из машинных результатов обучения (CLI).
+
+Без обучения и без пересчёта метрик строит автономный ``report/report.html``
+и обязательные ``report/runs.csv``, ``report/per_class.parquet``,
+``report/stages.csv``, ``report/history.csv``, ``report/manifest.json`` из
+файлов, записанных обучающим контуром (``scripts/run_s05_baseline.py``) и
+аудитом в ``--output`` (по умолчанию ``outputs/S05``).
+
+Контракт D09: planned failed/skipped видны; отсутствующие числа — ``null``,
+не нули; test-метрик нет; кривые train/monitor, train–monitor gap, разброс
+seed, majority-сравнение, per-class/support и ресурсы — из сохранённых
+таблиц; для сплит-аудита используется ``audit/split_audit_corrected.csv``
+(ошибка исходного ``split_audit.csv`` в ``n_row_ids_unique`` указана в
+отчёте явно). Существующие файлы не перезаписываются — при пересечении
+отказ до любых записей.
+
+Пример:
+
+    python scripts/run_s05_report.py --output outputs/S05
+"""
+import argparse
+import json
+import os
+import sys
+
+sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+
+from kkt_clf import setup_logging  # noqa: E402
+from kkt_clf.s05_report import S05ReportError, build_s05_report  # noqa: E402
+
+
+def main() -> int:
+    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
+    ap.add_argument("--output", default="outputs/S05",
+                    help="каталог машинных результатов обучающего контура S05")
+    ap.add_argument("--report-dir", default=None,
+                    help="куда писать отчёт (по умолчанию <output>/report); "
+                         "существующие файлы не перезаписываются")
+    args = ap.parse_args()
+    setup_logging()
+    try:
+        result = build_s05_report(output_dir=args.output, report_dir=args.report_dir)
+    except S05ReportError as e:
+        print(f"ОШИБКА: {e}", file=sys.stderr)
+        return 2
+    print(json.dumps(result, ensure_ascii=False, indent=2))
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
diff --git a/scripts/s05_smoke_check.py b/scripts/s05_smoke_check.py
new file mode 100644
index 0000000..666b0b6
--- /dev/null
+++ b/scripts/s05_smoke_check.py
@@ -0,0 +1,166 @@
+"""Синтетическая проверка контура S05 (D04–D08) на малой выборке.
+
+Собирает во временном каталоге: маленький размеченный parquet, корпус имён,
+автономный full-TFIDF и замороженный сплит, затем прогоняет CLI
+``scripts/run_s05_baseline.py all`` (verify → select → refit 42/43/44 →
+оценка на comparison_val → majority → checkpoint seed 42 с round-trip).
+
+Проверяет: файлы созданы, runs.csv/per_class/history/stages/manifest согласованы,
+refit действительно идёт ровно fixed_epochs эпох, повторный запуск отказывается
+(не перезаписывает), comparison_val не оценивается до завершения refit
+(порядок стадий в stages.csv).
+"""
+import json
+import os
+import subprocess
+import sys
+import tempfile
+
+import numpy as np
+import pandas as pd
+
+sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+
+from kkt_clf.config import build_feature_config
+from kkt_clf.freeze_split import freeze_split, save_frozen_split
+from kkt_clf.vectorizer import fit_and_save_vectorizer
+
+ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
+
+
+def make_labeled(n_classes=12, per_class=20, seed=7) -> pd.DataFrame:
+    rng = np.random.default_rng(seed)
+    rows = []
+    for c in range(n_classes):
+        l3 = f"Класс {c}"
+        l2 = f"Группа {c // 3}"
+        l1 = "Ветка A" if c < 6 else "Ветка B"
+        for i in range(per_class):
+            name = f"товар {c} вариант {i} " + (" ".join(rng.choice(["мл", "грамм", "литр"], size=2)))
+            rows.append({"name": name, "L1": l1, "L2": l2, "L3": l3})
+    return pd.DataFrame(rows)
+
+
+def make_corpus(n=4000, seed=11) -> pd.DataFrame:
+    rng = np.random.default_rng(seed)
+    words = ["молоко", "хлеб", "сахар", "кофе", "чай", "соль", "вода", "сок", "кефир", "рис"]
+    names = [" ".join(rng.choice(words, size=3)) for _ in range(n)]
+    return pd.DataFrame({"name": names})
+
+
+def run_cli(args, expect_code=0):
+    cmd = [os.path.join(ROOT, ".venv", "bin", "python"),
+           os.path.join(ROOT, "scripts", "run_s05_baseline.py"), *args]
+    proc = subprocess.run(cmd, capture_output=True, text=True)
+    if proc.returncode != expect_code:
+        print(proc.stdout)
+        print(proc.stderr)
+        raise AssertionError(f"код {proc.returncode} != {expect_code}: {' '.join(cmd)}")
+    return proc
+
+
+def main() -> None:
+    tmp = tempfile.mkdtemp(prefix="s05_smoke_")
+    labeled_path = os.path.join(tmp, "labeled.parquet")
+    corpus_path = os.path.join(tmp, "corpus.parquet")
+    tfidf_dir = os.path.join(tmp, "tfidf_full")
+    splits_dir = os.path.join(tmp, "splits")
+    audit_dir = os.path.join(tmp, "audit")
+    out_dir = os.path.join(tmp, "s05_out")
+
+    labeled = make_labeled()
+    labeled.to_parquet(labeled_path, index=False)
+    make_corpus().to_parquet(corpus_path, index=False)
+
+    feat_cfg = build_feature_config("ods_light", {
+        "char": {"max_features": 2000, "min_df": 1},
+        "word": {"max_features": 1000, "min_df": 1},
+        "vectorizer_fit_rows": None,
+    })
+    fit_and_save_vectorizer(tfidf_dir, feat_cfg, corpus_path, max_rows=4000, seed=42)
+
+    frozen = freeze_split(labeled, min_count=10, seed=42)
+    summary = save_frozen_split(frozen, labeled_path, splits_dir, audit_dir,
+                                min_count=10, fractions=(0.8, 0.1, 0.1),
+                                monitor_fraction=0.1, seed=42, share_tolerance=0.1)
+    assert summary["checks_ok"], summary
+    manifest_path = os.path.join(splits_dir, "manifest.json")
+
+    common = [
+        "--splits-manifest", manifest_path,
+        "--full-tfidf", tfidf_dir,
+        "--corpus", corpus_path,
+        "--output", out_dir,
+        "--seeds", "42,43,44",
+        "--reference-seed", "42",
+        "--model", json.dumps({"hidden": [32], "dropout": [0.2]}),
+        "--train", json.dumps({"epochs": 4, "patience": 3, "batch_size": 64,
+                               "scheduler": "cosine", "scheduler_eval_every_steps": 5}),
+        "--features-chunk-rows", "1000",
+    ]
+    run_cli(["all", *common])
+    print("== all: ok ==")
+
+    # --- проверка результатов
+    for rel in ["tfidf_verification.json", "selection/selection_report.json",
+                "selection/curves_train_monitor.csv", "runs.csv", "per_class.parquet",
+                "history.csv", "stages.csv", "majority_baseline.json",
+                "selected_checkpoint.json", "manifest.json",
+                "checkpoints/seed_42/model.pt", "checkpoints/seed_43/model.pt",
+                "checkpoints/seed_44/model.pt",
+                "refit/history_seed_42.csv", "refit/run_result_seed_42.json"]:
+        p = os.path.join(out_dir, rel)
+        assert os.path.exists(p), f"нет файла {p}"
+
+    runs = pd.read_csv(os.path.join(out_dir, "runs.csv"))
+    assert len(runs) == 3 and (runs["status"] == "success").all(), runs
+    assert runs["metric_evaluated_on"].eq("comparison_val").all()
+    assert runs["l3_macro_f1"].notna().all() and runs["epochs_completed"].eq(
+        json.load(open(os.path.join(out_dir, "manifest.json")))["fixed_epochs"]).all()
+
+    sel = json.load(open(os.path.join(out_dir, "selection", "selection_report.json")))
+    fixed_epochs = sel["chosen"]["fixed_epochs"]
+    assert fixed_epochs >= 1
+    hist = pd.read_csv(os.path.join(out_dir, "history.csv"))
+    for seed in (42, 43, 44):
+        h = hist[hist.run_id == f"seed_{seed}"]
+        assert h["epoch"].max() == fixed_epochs, (seed, h["epoch"].max(), fixed_epochs)
+        assert h[h.split == "train"]["loss"].notna().all()
+        assert h[h.split == "monitor"]["l3_macro_f1"].notna().all()
+
+    # порядок стадий: все mlp_train идут до vectorize_comparison_val
+    stages = pd.read_csv(os.path.join(out_dir, "stages.csv"))
+    idx_train = stages[stages.stage == "mlp_train"].index.max()
+    idx_val = stages[stages.stage == "vectorize_comparison_val"].index.min()
+    assert idx_train < idx_val, "comparison_val трансформировался до завершения refit"
+
+    per_class = pd.read_parquet(os.path.join(out_dir, "per_class.parquet"))
+    assert set(per_class.run_id) == {"seed_42", "seed_43", "seed_44"}
+    assert len(per_class) == 3 * len(json.load(open(manifest_path))["label_maps"]["L3"])
+
+    maj = json.load(open(os.path.join(out_dir, "majority_baseline.json")))
+    assert maj["evaluated_on"] == "comparison_val" and 0.0 <= maj["l3_accuracy"] <= 1.0
+
+    sel_ckpt = json.load(open(os.path.join(out_dir, "selected_checkpoint.json")))
+    assert sel_ckpt["run_id"] == "seed_42" and sel_ckpt["selection_rule"]["kind"] == "reference_seed"
+    assert sel_ckpt["roundtrip"]["status"] == "success" and sel_ckpt["roundtrip"]["metrics_match"]
+    assert sel_ckpt["provenance_checks"]["restored_best"] is False
+
+    manifest = json.load(open(os.path.join(out_dir, "manifest.json")))
+    assert manifest["seeds"]["reference_seed"] == 42
+    assert manifest["statuses"]["selected_checkpoint"] == "success"
+    assert manifest["seed_stats_comparison_val"]["n_seed_success"] == 3
+
+    # majority macro-F1 > 0 и меньше 1 (на синтетике сбалансировано): sanity
+    assert 0.0 <= maj["l3_macro_f1"] < 1.0
+
+    # --- повторный запуск обязан отказаться (не перезаписывает)
+    proc = run_cli(["all", *common], expect_code=2)
+    assert "не перезаписываются" in proc.stderr, proc.stderr
+    print("== повторный запуск корректно отказался ==")
+
+    print("SMOKE OK:", out_dir)
+
+
+if __name__ == "__main__":
+    main()
diff --git a/tests/test_s05_audit_independent.py b/tests/test_s05_audit_independent.py
new file mode 100644
index 0000000..19d753c
--- /dev/null
+++ b/tests/test_s05_audit_independent.py
@@ -0,0 +1,96 @@
+"""Contract checks for the S05 audit using only disposable synthetic inputs."""
+
+import json
+import subprocess
+import sys
+from pathlib import Path
+
+import pandas as pd
+import pytest
+
+
+SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_ods.py"
+
+
+@pytest.fixture
+def audit_case(tmp_path):
+    rows = [
+        ("Молоко  цельное", "Food", "Dairy", "Milk"),
+        ("молоко\tцельное", "Food", "Dairy", "Milk"),
+        ("МОЛОКОЦЕЛЬНОЕ", "Food", "Dairy", "Cheese"),
+        ("same", "Food", "Dairy", "Cheese"),
+        ("same", "Food", "Dairy", "Cheese"),
+        ("other", "Other", "Dairy", "Rare"),
+        ("third", "Food", "Bakery", "Milk"),
+        (None, "Food", "Dairy", "Milk"),
+        (" ", None, "Dairy", "Milk"),
+        ("invalid", "Food", "  ", "Milk"),
+        ("invalid2", "Food", "Dairy", ""),
+    ]
+    source = tmp_path / "input.parquet"
+    corpus = tmp_path / "corpus.parquet"
+    pd.DataFrame(rows, columns=["name", "L1", "L2", "L3"]).to_parquet(source)
+    pd.DataFrame({"name": ["молоко цельное", "unrelated"]}).to_parquet(corpus)
+    out = tmp_path / "audit"
+    command = [sys.executable, str(SCRIPT), "--input", str(source), "--corpus", str(corpus),
+               "--s02-dir", str(tmp_path / "missing_s02"), "--legacy-csv", str(tmp_path / "missing.csv"),
+               "--out-dir", str(out), "--min-count", "3"]
+    return source, corpus, out, command
+
+
+def run(command):
+    return subprocess.run(command, text=True, capture_output=True, check=False)
+
+
+def test_audit_reports_actual_data_and_preserves_inputs(audit_case):
+    source, corpus, out, command = audit_case
+    before = (source.read_bytes(), corpus.read_bytes())
+    result = run(command)
+    assert result.returncode == 0, result.stderr
+    assert (source.read_bytes(), corpus.read_bytes()) == before
+    audit = json.loads((out / "data_audit.json").read_text(encoding="utf-8"))["audit"]
+    assert audit["rows_read"] == 11
+    assert audit["rows_after_drop_empty"] == 7
+    assert audit["empty_values"] == {"name": 2, "L1": 1, "L2": 1, "L3": 1, "total": 4}
+
+    violations = {(v["level"], v["child"], tuple(v["parents"]))
+                  for v in audit["taxonomy_violations"]["examples"]}
+    assert violations == {("L3", "Milk", ("Bakery", "Dairy")),
+                          ("L2", "Dairy", ("Food", "Other"))}
+    assert audit["taxonomy_violations"]["n"] == 2
+    duplicates = audit["duplicates"]
+    assert duplicates["raw_name"]["n_duplicate_extra_rows"] == 1
+    assert duplicates["raw_name"]["n_names_appearing_more_than_once"] == 1
+    keys = duplicates["normalized_keys"]
+    assert keys["char_branch_strip_whitespace"]["n_groups_multi_l3"] == 1
+    assert keys["char_branch_strip_whitespace"]["n_rows_in_multi_l3_groups"] == 3
+    assert keys["word_branch_collapse_ws"]["n_groups_multi_l3"] == 0
+    assert keys["word_branch_collapse_ws"]["n_collided_groups"] == 2
+
+    rare = audit["min_count_filter"]
+    assert (rare["n_excluded_l3"], rare["n_excluded_rows"], rare["rows_after"], rare["n_l3_after"]) == (1, 1, 6, 2)
+    assert rare["excluded_l3_full_list"] == [{"L1": "Other", "L2": "Dairy", "L3": "Rare", "count": 1}]
+    support = pd.read_csv(out / "class_support.csv").set_index(["L1", "L2", "L3"])
+    expected = {("Food", "Dairy", "Milk"): (2, 2), ("Food", "Bakery", "Milk"): (1, 1),
+                ("Food", "Dairy", "Cheese"): (3, 3), ("Other", "Dairy", "Rare"): (1, 0)}
+    assert set(support.index) == set(expected)
+    for key, (raw, after) in expected.items():
+        row = support.loc[key]
+        assert row["count_raw"] == raw
+        assert row["count_after_min_count"] == after
+        assert bool(row["kept_after_min_count"]) == (after > 0)
+        assert row["share_raw"] == pytest.approx(raw / 7, abs=1e-6)
+
+
+@pytest.mark.parametrize("existing", ["data_audit.json", "class_support.csv"])
+def test_existing_output_refuses_overwrite_without_writing_other_file(audit_case, existing):
+    source, corpus, out, command = audit_case
+    out.mkdir()
+    sentinel = b"existing user data\n"
+    (out / existing).write_bytes(sentinel)
+    before = (source.read_bytes(), corpus.read_bytes())
+    result = run(command)
+    assert result.returncode != 0
+    assert (out / existing).read_bytes() == sentinel
+    assert sorted(p.name for p in out.iterdir()) == [existing]
+    assert (source.read_bytes(), corpus.read_bytes()) == before
diff --git a/tests/test_s05_split_independent.py b/tests/test_s05_split_independent.py
new file mode 100644
index 0000000..aafcd81
--- /dev/null
+++ b/tests/test_s05_split_independent.py
@@ -0,0 +1,108 @@
+import hashlib
+import json
+from itertools import combinations
+
+import pandas as pd
+import pytest
+
+from kkt_clf.freeze_split import FreezeSplitError, freeze_split, save_frozen_split
+
+
+@pytest.fixture
+def rows():
+    data = []
+    for label in ('a', 'b'):
+        for i in range(40):
+            name = f'{label} item {i}'
+            data.append((name, 'root', f'parent-{label}', label))
+            if i == 0:
+                data.extend([(name, 'root', f'parent-{label}', label),
+                             (f' {label.upper()}  item\t{i} ', 'root', f'parent-{label}', label),
+                             (f'{label}i tem {i}', 'root', f'parent-{label}', label)])
+    data.extend((f'rare {i}', 'root', 'parent-r', 'rare') for i in range(3))
+    return pd.DataFrame(data, columns=['name', 'L1', 'L2', 'L3'])
+
+
+def split(rows):
+    return freeze_split(rows, min_count=10, seed=42, share_tolerance=0.1)
+
+
+def test_determinism_identity_isolation_and_support(rows):
+    first, second = split(rows), split(rows)
+    for part in first.parts:
+        pd.testing.assert_frame_equal(first.parts[part], second.parts[part])
+    outer = [first.parts[p] for p in ('outer_train', 'comparison_val', 'sealed_test')]
+    ids = [set(p.row_id) for p in outer]
+    assert all(a.isdisjoint(b) for a, b in combinations(ids, 2))
+    assert set.union(*ids) | set(first.excluded.row_id) == set(range(len(rows)))
+    assert set.union(*ids).isdisjoint(first.excluded.row_id)
+    for part in outer:
+        for row in part.itertuples():
+            assert (row.name, row.L1, row.L2, row.L3) == tuple(rows.iloc[row.row_id])
+        assert set(part.L3) == {'a', 'b'}
+    fit, monitor = first.parts['train_fit'], first.parts['train_monitor']
+    assert set(fit.row_id).isdisjoint(monitor.row_id)
+    assert set(fit.row_id) | set(monitor.row_id) == ids[0]
+    assert set(fit.L3) == set(monitor.L3) == {'a', 'b'}
+    for a, b in [*combinations(outer, 2), (fit, monitor)]:
+        for normalize in (lambda s: ''.join(s.lower().split()),
+                          lambda s: ' '.join(s.lower().split())):
+            assert {normalize(s) for s in a.name}.isdisjoint(normalize(s) for s in b.name)
+    assert set(first.excluded.L3) == {'rare'}
+    assert set(first.excluded.row_id) == set(range(len(rows) - 3, len(rows)))
+
+
+def test_conflicting_normalized_names_rejected(rows):
+    rows.loc[1, 'name'] = ' A  ITEM\t0 '
+    rows.loc[1, ['L2', 'L3']] = ['parent-b', 'b']
+    with pytest.raises(FreezeSplitError, match='конфликт'):
+        split(rows)
+
+
+def test_impossible_group_stratification_rejected():
+    rows = pd.DataFrame([('same', 'root', 'parent', 'class')] * 20,
+                        columns=['name', 'L1', 'L2', 'L3'])
+    with pytest.raises(FreezeSplitError, match='стратификац'):
+        split(rows)
+
+
+def test_manifest_hashes_support_and_no_overwrite(rows, tmp_path):
+    source = tmp_path / 'input.parquet'
+    rows.to_parquet(source, index=False)
+    out, audit = tmp_path / 'splits', tmp_path / 'audit'
+    frozen = split(rows)
+    args = (frozen, str(source), str(out), str(audit), 10, (0.8, 0.1, 0.1), 0.1, 42, 0.1)
+    save_frozen_split(*args)
+    manifest = json.loads((out / 'manifest.json').read_text())
+    assert manifest['policy']['split_seed'] == 42
+    assert manifest['policy']['min_count'] == 10
+    assert manifest['excluded_min_count']['row_ids'] == list(range(len(rows) - 3, len(rows)))
+    assert manifest['inputs']['labeled_parquet']['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
+    for part, meta in manifest['parts'].items():
+        path = out / f'{part}.parquet'
+        assert meta['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
+        assert meta['bytes'] == path.stat().st_size
+        saved = pd.read_parquet(path)
+        assert meta['n_rows'] == len(saved) == len(frozen.parts[part])
+        assert set(saved.row_id) == set(frozen.parts[part].row_id)
+    support = pd.read_csv(audit / 'class_support_split.csv').set_index('L3')
+    for label in ('a', 'b'):
+        for part in frozen.parts:
+            assert support.loc[label, f'support_{part}'] == (frozen.parts[part].L3 == label).sum()
+    assert not support.loc['rare', 'kept_after_min_count']
+    before = {p: p.read_bytes() for directory in (out, audit) for p in directory.iterdir()}
+    with pytest.raises(FreezeSplitError, match='перезапис'):
+        save_frozen_split(*args)
+    assert before == {p: p.read_bytes() for directory in (out, audit) for p in directory.iterdir()}
+
+
+def test_split_audit_reports_number_of_unique_row_ids(rows, tmp_path):
+    source = tmp_path / 'input.parquet'
+    rows.to_parquet(source, index=False)
+    frozen = split(rows)
+    save_frozen_split(frozen, str(source), str(tmp_path / 'splits'), str(tmp_path / 'audit'),
+                      10, (0.8, 0.1, 0.1), 0.1, 42, 0.1)
+    audit = pd.read_csv(tmp_path / 'audit' / 'split_audit.csv').set_index('part')
+    for part, frame in frozen.parts.items():
+        assert audit.loc[part, 'n_row_ids_unique'] == frame.row_id.nunique()
+    assert audit.loc['excluded_by_min_count', 'n_row_ids_unique'] == frozen.excluded.row_id.nunique()
diff --git a/tests/test_s05_baseline_independent.py b/tests/test_s05_baseline_independent.py
new file mode 100644
index 0000000..5dc00e9
--- /dev/null
+++ b/tests/test_s05_baseline_independent.py
@@ -0,0 +1,166 @@
+"""S05 protocol checks on synthetic data; no ODS or external model artifacts."""
+import json
+from types import SimpleNamespace
+
+import numpy as np
+import pandas as pd
+import pytest
+import scipy.sparse as sp
+import torch
+
+from kkt_clf import s05_baseline as s05
+from kkt_clf.config import BranchConfig, FeatureConfig, LossConfig, ModelConfig, TrainConfig
+from kkt_clf.features import FeatureBuilder
+from kkt_clf.labels import LabelMaps
+from kkt_clf.model import SparseMLP
+from kkt_clf.train import set_seed, train_model
+
+
+@pytest.fixture
+def synthetic(tmp_path, monkeypatch):
+    torch.set_num_threads(1)
+    df = pd.DataFrame({'name': ['alpha milk', 'beta bread', 'alpha food', 'beta food', 'gamma'],
+                       'L1': ['root'] * 5, 'L2': ['group'] * 5,
+                       'L3': ['a', 'b', 'a', 'b', 'c']})
+    maps = LabelMaps.from_df(df)
+    parts = {'outer_train': df.iloc[:4], 'train_fit': df.iloc[:2],
+             'train_monitor': df.iloc[2:4], 'comparison_val': df.iloc[[0, 1]].copy()}
+    parts['comparison_val'].loc[parts['comparison_val'].index[1], 'name'] = ''
+    manifest = {'label_maps': maps.to_dict(), 'policy': {'split_seed': 7},
+                'parts': {k: {'sha256': k, 'bytes': 1, 'n_rows': len(v)} for k, v in parts.items()},
+                'inputs': {'labeled_parquet': {'sha256': 'source'}}}
+    manifest['parts']['sealed_test'] = {'sha256': 'sealed', 'bytes': 1, 'n_rows': 1}
+    builder = FeatureBuilder(FeatureConfig(BranchConfig((2, 3), 1), BranchConfig((1, 1), 1), None, 10))
+    builder.fit(['alpha milk', 'beta bread', 'alpha food', 'beta food', 'gamma'], seed=3)
+    loaded = SimpleNamespace(builder=builder, meta={'files': {'features.joblib': {'sha256': 'full-id'}}})
+    cfg = s05.S05Config(output_dir=str(tmp_path / 'result'), splits_manifest=str(tmp_path / 'split.json'),
+                        seeds=[42, 43], reference_seed=42, fixed_epochs=2, device='cpu',
+                        model={'hidden': [4], 'dropout': [0.0]},
+                        train={'scheduler': 'none', 'batch_size': 2, 'epochs': 2, 'patience': 2})
+    (tmp_path / 'split.json').write_text('{}')
+    events = []
+    def read_part(_manifest, name):
+        events.append(('read', name))
+        if name == 'sealed_test':
+            pytest.fail('sealed test data were read')
+        return parts[name].copy()
+    monkeypatch.setattr(s05, 'load_part', read_part)
+    original_train = s05.train_model
+    def record_train(*args, **kwargs):
+        events.append(('train', args[9]))
+        return original_train(*args, **kwargs)
+    monkeypatch.setattr(s05, 'train_model', record_train)
+    return cfg, manifest, loaded, maps, events
+
+
+def test_selection_uses_only_internal_train(synthetic):
+    cfg, manifest, loaded, _, events = synthetic
+    report = s05.run_selection(cfg, manifest, loaded, torch.device('cpu'))
+    assert [x[1] for x in events if x[0] == 'read'] == ['train_fit', 'train_monitor']
+    assert report['chosen']['fixed_epochs'] == report['candidates'][0]['best_epoch']
+    assert report['full_tfidf_id'] == 'full-id'
+
+
+def test_refit_does_not_read_validation_until_all_seeds_finished(synthetic):
+    cfg, manifest, loaded, _, events = synthetic
+    s05.run_refit(cfg, manifest, loaded, torch.device('cpu'))
+    val_index = events.index(('read', 'comparison_val'))
+    assert [e[1] for e in events[:val_index] if e[0] == 'train'] == cfg.seeds
+    assert ('read', 'sealed_test') not in events
+
+
+def test_refit_metrics_checkpoint_and_feature_identity(synthetic, monkeypatch):
+    cfg, manifest, loaded, maps, events = synthetic
+    def no_fit(*args, **kwargs):
+        pytest.fail('pretrained TF-IDF was fitted again')
+    monkeypatch.setattr(FeatureBuilder, 'fit', no_fit)
+    before = loaded.builder.transform(['alpha milk', '']).copy()
+    s05.run_refit(cfg, manifest, loaded, torch.device('cpu'))
+    from kkt_clf.artifact import load_artifact
+    out = __import__('pathlib').Path(cfg.output_dir)
+    runs = pd.read_csv(out / 'runs.csv')
+    assert list(runs.status) == ['success', 'success']
+    assert set(runs.full_tfidf_id) == {'full-id'}
+    assert set(runs.epochs_completed) == {2}
+    classes = pd.read_parquet(out / 'per_class.parquet')
+    for _, frame in classes.groupby('run_id'):
+        assert set(frame.L3) == set(maps.l3)
+        absent = frame.set_index('L3').loc['c']
+        assert absent.support == 0 and pd.isna(absent.f1)
+    selected = json.loads((out / 'selected_checkpoint.json').read_text())
+    assert selected['run_id'] == 'seed_42' and selected['roundtrip']['metrics_match']
+    art = load_artifact(str(out / 'checkpoints' / 'seed_42'), torch.device('cpu'))
+    assert art.meta['train']['n_train_rows_used'] == 4
+    assert art.meta['train']['restored_best'] is False
+    assert art.maps.to_dict() == maps.to_dict()
+    assert (art.features.transform(['alpha milk', '']) != before).nnz == 0
+    result = json.loads((out / 'refit' / 'run_result_seed_42.json').read_text())
+    assert result['comparison_val_eval']['n_empty_features_val'] == 1
+    assert result['comparison_val_eval']['l3_accuracy'] == runs.iloc[0].l3_accuracy
+
+
+def test_fixed_training_final_epoch_and_seed_reproducibility():
+    X = sp.csr_matrix(np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32))
+    y = (np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64), np.array([0, 1, 0, 1]))
+    model_cfg = ModelConfig(hidden=[4], dropout=[0.0])
+    def run(seed, epochs, restore):
+        set_seed(seed)
+        model = SparseMLP(2, (1, 1, 2), model_cfg)
+        cfg = TrainConfig(epochs=epochs, batch_size=2, patience=10, scheduler='none', lr=0.01)
+        history, best, _ = train_model(model, X, y, X, y, LossConfig(), cfg,
+                                        {'l1': None, 'l2': None, 'l3': None}, torch.device('cpu'), seed,
+                                        restore_best=restore)
+        return model, history, best
+    a, hist, best = run(42, 3, False)
+    b, hist2, _ = run(42, 3, False)
+    c, _, _ = run(43, 3, False)
+    assert len(hist) == 3 and best['n_train_rows_used'] == 3 and not best['restored_best']
+    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
+    assert any(not torch.equal(a.state_dict()[k], c.state_dict()[k]) for k in a.state_dict())
+    assert [x['val_l3_accuracy'] for x in hist] == [x['val_l3_accuracy'] for x in hist2]
+    if best['epoch'] < 3:
+        restored, _, _ = run(42, 3, True)
+        assert any(not torch.equal(a.state_dict()[k], restored.state_dict()[k]) for k in a.state_dict())
+
+
+def test_failed_seed_records_null_metrics(synthetic, monkeypatch):
+    cfg, manifest, loaded, _, _ = synthetic
+    original = s05.train_model
+    def fail_second(*args, **kwargs):
+        if args[9] == 43:
+            raise RuntimeError('synthetic failure')
+        return original(*args, **kwargs)
+    monkeypatch.setattr(s05, 'train_model', fail_second)
+    s05.run_refit(cfg, manifest, loaded, torch.device('cpu'))
+    runs = pd.read_csv(__import__('pathlib').Path(cfg.output_dir) / 'runs.csv').set_index('train_seed')
+    assert runs.loc[43, 'status'] == 'failed'
+    assert pd.isna(runs.loc[43, 'l3_macro_f1']) and pd.isna(runs.loc[43, 'l3_accuracy'])
+    assert pd.isna(runs.loc[43, 'epochs_completed']) and pd.isna(runs.loc[43, 'checkpoint_path'])
+
+
+def test_raw_top1_metrics_keep_unsupported_class_null():
+    # Class b is never predicted; c is absent in validation but remains in the universe.
+    macro, accuracy, rows = s05._l3_val_metrics(np.array([0, 0, 1]), np.array([0, 1, 0]), ['a', 'b', 'c'])
+    by_class = {row['L3']: row for row in rows}
+    assert accuracy == pytest.approx(1 / 3)
+    assert macro == pytest.approx(0.25)
+    assert by_class['c']['support'] == 0
+    assert by_class['c']['f1'] is None
+
+
+def test_all_failed_seeds_still_have_report_rows(synthetic, monkeypatch):
+    cfg, manifest, loaded, _, _ = synthetic
+    def fail(*args, **kwargs):
+        raise RuntimeError('all seeds failed')
+    monkeypatch.setattr(s05, 'train_model', fail)
+    try:
+        s05.run_refit(cfg, manifest, loaded, torch.device('cpu'))
+    except s05.S05BaselineError:
+        pass
+    from pathlib import Path
+    path = Path(cfg.output_dir) / 'runs.csv'
+    assert path.exists(), 'all planned failed runs must remain visible in runs.csv'
+    runs = pd.read_csv(path)
+    assert set(runs.train_seed) == set(cfg.seeds)
+    assert set(runs.status) == {'failed'}
+    assert runs.l3_macro_f1.isna().all() and runs.l3_accuracy.isna().all()
