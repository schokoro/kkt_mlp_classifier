# DECISION_LOG

Журнал решений, только дописывается. Формат: номер, дата, раздел спеки, решение, альтернативы, обоснование, как проверить.

---

### D-001 · 2026-10-22 · §8 · Имя файла журнала
- **Решение:** `artefacts/DECISON_LOG.md` (опечатка) переименован в `artefacts/DECISION_LOG.md`, как требует §8.
- **Альтернативы:** оставить имя с опечаткой.
- **Обоснование:** спека ссылается на `DECISION_LOG.md`; файл был пустым, история не теряется (`git mv`).
- **Проверка:** `ls artefacts/`.

### D-002 · 2026-10-22 · §2, §5.2 · Python и установка torch в облачной сессии
- **Решение:** в облаке `python3` = 3.13.16, поэтому разработка и все прогоны — в `.venv` на `/usr/bin/python3.12`. Индекс `download.pytorch.org` закрыт сетевой политикой окружения (403 на CONNECT), поэтому torch поставлен с PyPI (linux-колесо с CUDA-зависимостями; на CPU работает). `scripts/setup_cloud.sh` сначала пробует CPU-индекс по спеке, при неудаче откатывается на PyPI. **Отступление от §5.2** (CUDA-колёса в облаке).
- **Альтернативы:** ждать открытия хоста; `pip install torch --no-deps` с PyPI (не импортируется без CUDA-библиотек).
- **Обоснование:** другого способа получить torch в этой сессии нет; в целевую среду переносится только код, на неё отступление не влияет.
- **Проверка:** `curl -sI https://download.pytorch.org/whl/cpu/` → 403; `.venv/bin/python --version` → 3.12.x. Чтобы вернуться к спеке — добавить `download.pytorch.org` в allowlist окружения.

### D-003 · 2026-10-22 · §5.2 · Структура кода
- **Решение:** пакет `kkt_clf/`: `config` (dataclass-конфиги, пресеты), `data` (загрузка, проверки, сэмпл, `min_count`, сплит), `labels` (маппинги, родители), `features` (preprocess, ветки TF-IDF), `model` (SparseMLP), `train` (веса классов, лосс, цикл, предсказание), `reject` (отказ, выбор τ), `metrics`, `inference` (общий код предсказания/отказа/метрик + `Predictor`, `run_inference`), `artifact` (save/load), `pipeline` (`run_training`), `plots`, `fixtures` (генератор). Ноутбуки вызывают `run_training(cfg)` / `run_inference(cfg)`; `scripts/run_train.py` и `scripts/run_inference.py` делают те же вызовы.
- **Альтернативы:** один модуль; классы-оркестраторы с шагами.
- **Обоснование:** обучение и инференс используют одну функцию `inference.predict_frame` (предсказание + отказ) и одну `evaluate_frame` (метрики) — исключает расхождение кода.
- **Проверка:** `tests/test_notebooks.py::test_notebook_is_facade`, `python scripts/run_train.py`.

### D-004 · 2026-10-22 · §4.4 · Подача разреженных батчей в первый слой
- **Решение:** первый слой — `nn.EmbeddingBag(n_features, hidden[0], mode="sum")` + отдельный bias. CSR-срез батча → `indices` (input), `indptr[:-1]` (offsets), `data` (per_sample_weights) = ровно `X @ W + b`. Инициализация как у `nn.Linear`.
- **Альтернативы:** `torch.sparse.mm(CSR/COO, W)` — autograd для sparse CSR на CPU неполный и медленнее; dense-батч — запрещён спекой и неэффективен при 400k признаков.
- **Обоснование:** fused kernel на CPU и CUDA, простая пересылка (3 тензора), стабильный autograd.
- **Проверка:** `tests/test_model_loss_reject.py::test_sparse_linear_equals_dense`.

### D-005 · 2026-10-22 · §4.4, §5.4 · Хранение признаковой матрицы
- **Решение:** train/val/test векторизуются один раз, чанками (`transform_chunk_rows`), в CSR float32 с int32-индексами; склейка блоков вручную (`features.vstack_csr`). Батчи — срезы CSR по перестановке. Dense не материализуется нигде. Инференс векторизует батчами `batch_size`.
- **Альтернативы:** векторизация на лету в каждой эпохе (дорого по CPU на 6M); memmap на диске.
- **Обоснование:** при ~100 nnz/строку CSR на 6M строк ≈ 4.8 ГБ — помещается в 24 ГБ; векторизация не повторяется по эпохам.
- **Проверка:** `features.matrix_stats` в `run_report.json` (`bytes`, `nnz_per_row_mean`).

### D-006 · 2026-10-22 · §2, §5.4 · Дефолты конфига под ~6 млн строк (пресет `default`)
- **Решение:** char `(2,5)`, `min_df=5`, `max_features=300k`; word `min_df=3`, `max_features=100k`; `sublinear_tf=True`; `vectorizer_fit_rows=1.5M`; `hidden=[512,256]`, `dropout=[0.3,0.3]`, relu, без нормализации; AdamW lr=1e-3, wd=0; batch 2048, eval batch 8192, 10 эпох, patience 2, early stopping по val macro-F1 L3; `head_weights` 0.3/0.3/1.0; class weights `none`, alpha 0.5; split 0.8/0.1/0.1, seed 42, `min_count=10`; `tau_mode=fixed`, `tau=0.5`, mismatch-правило выключено.
- **Альтернативы:** `hidden[0]=1024` при 200k признаков (тот же размер слоя, меньше словарь); больше эпох.
- **Обоснование:** первый слой 400k × 512 ≈ 205M параметров (< «нескольких сотен млн», §5.4), с AdamW ≈ 3.3 ГБ на GPU. `vectorizer_fit_rows` ограничивает главный пик RAM — словарь всех n-грамм до отсечения `max_features`. Гиперпараметры не подбирались (§1).
- **Проверка:** `config.default_config()`; оценки памяти — README.

### D-007 · 2026-10-22 · §6 · Пресет `ods_light`
- **Решение:** для ODS: char `min_df=2`, `max_features=50k`; word `min_df=1`, `max_features=20k`; fit на всём train; `hidden=[256]`, `dropout=[0.3]`; batch 256, 15 эпох, patience 3.
- **Альтернативы:** гонять дефолты 6M на ODS (словарь почти пустой при `min_df=5`, модель избыточна).
- **Обоснование:** разрешено §2; быстрый прогон на CPU.
- **Проверка:** `config.ods_light_config()`, обучающий ноутбук.

### D-008 · 2026-10-22 · §4.1 · Стратифицированный сэмпл `max_rows`
- **Решение:** квоты по L3 пропорционально частоте, целые части + метод наибольших остатков (ровно `max_rows` строк); выбор строк — случайная перестановка с seed, ранг внутри класса < квоты.
- **Альтернативы:** `train_test_split(stratify=...)` — падает на классах из одного примера.
- **Обоснование:** детерминированно, векторизовано (быстро на 6M), не падает на редких классах; они затем отсекаются `min_count`.
- **Проверка:** `tests/test_data.py::test_stratified_sample`.

### D-009 · 2026-10-22 · §4.3 [Г] · Редкие классы: `min_count=10`
- **Решение:** гипотеза принята без изменений: L3 с < 10 примерами (после сэмпла) исключаются из обучения и оценки, их список — `reports/rare_l3_classes.csv`, объём — `run_report.json`. На ODS это 411 из 737 классов L3 (1 411 строк из 18 046).
- **Альтернативы:** меньший порог (на 0.1-сплите классы с < 10 примерами дают 0–1 пример в val/test, стратификация ломается).
- **Обоснование:** спека ожидает отсечение на обоих наборах; порог < 2 запрещён валидацией конфига, т. к. ломает стратифицированный сплит.
- **Проверка:** `tests/test_data.py::test_min_count_and_split`.

### D-010 · 2026-10-22 · §4.3 · Сплит
- **Решение:** два `train_test_split(stratify=L3, random_state=seed)`: train | (val+test), затем val | test. Сплиты сохраняются по содержимому: `outputs/<run>/splits/{train,val,test}.parquet` (`name, L1..L3`).
- **Альтернативы:** `StratifiedShuffleSplit` — то же самое по сути.
- **Проверка:** `tests/test_data.py::test_min_count_and_split`, `tests/test_pipeline.py::test_reproducible`.

### D-011 · 2026-10-22 · §4.5 · Веса голов и прочие детали обучения
- **Решение:** `head_weights` по умолчанию L1=0.3, L2=0.3, L3=1.0. Строки train с пустым вектором признаков в обучение не попадают (их число — в `best_epoch.n_train_rows_used`). Лучшие по val веса восстанавливаются после early stopping. Классы без примеров в train при `inv_freq` получают вес 0 (нормировка на средний 1 — по классам, присутствующим в train).
- **Альтернативы:** равные веса голов.
- **Обоснование:** L1/L2 — вспомогательные, решение по L3 (§4.5).
- **Проверка:** `tests/test_model_loss_reject.py::test_class_weights`, `test_multitask_loss_weighted`.

### D-012 · 2026-10-22 · §4.6, §4.9 · Детали отказа
- **Решение:** приоритет причин: `empty_features` → `low_confidence` → `mismatch`. Для строк с пустыми признаками модель не вызывается: все выходы модели (`l3_top1_raw`, головы, вероятности, top-k) — null, флаги рассогласования — False. Таблица τ на val строится с учётом текущего mismatch-правила. В режиме `target_precision` «точность принятых» = accuracy L3 на принятых.
- **Альтернативы:** выдавать сырые выходы модели и для пустых строк.
- **Обоснование:** спека: «модель не вызывается, либо выход игнорируется» — выбран первый вариант, он честнее.
- **Проверка:** `tests/test_model_loss_reject.py::test_apply_reject_rules`, `tests/test_pipeline.py::test_inference_without_labels`.

### D-013 · 2026-10-22 · §4.7, §4.9 · Детали метрик
- **Решение:** вариант «без отказа»: строки без предсказания (`empty_features`) считаются ошибкой (не исключаются), отдельным классом в отчёт не входят. Вариант «с отказом»: L1/L2/L3 по пути от принятого L3. Метки, которых нет в маппинге модели, исключаются из метрик своего уровня и показываются строкой `__unknown_label__` с `support`. Доля рассогласований считается по строкам с предсказанием. Метрики top-errors — по L3.
- **Альтернативы:** исключать пустые строки и из «без отказа».
- **Проверка:** `tests/test_pipeline.py::test_inference_unknown_labels`.

### D-014 · 2026-10-22 · §6 · Фикстуры
- **Решение:** генератор `kkt_clf/fixtures.py` (+ `scripts/make_fixtures.py`) → `fixtures/` (в .gitignore): `ods` (весь ODS), `ods_small` (40 частых L3 × 50 строк, для быстрых тестов), `defect_dups_empty`, `defect_taxonomy`, `infer_no_labels`, `infer_unknown_labels`, `infer_empty_features`. Seed 42. В ODS нарушений путей таксономии и дубликатов `name` не найдено — для основной фикстуры ничего не отфильтровано (`fixtures/ods_info.json`).
- **Проверка:** `python scripts/make_fixtures.py`; тесты генерируют фикстуры во временный каталог.

### D-015 · 2026-10-22 · §4.5, §5.4 · Fused AdamW
- **Решение:** Adam/AdamW создаются с `fused=True` (с откатом на обычную реализацию, если сборка torch её не поддерживает). Требование `torch>=2.4`.
- **Альтернативы:** foreach-реализация по умолчанию; sparse-градиенты EmbeddingBag + SparseAdam (другая семантика «ленивого» Adam).
- **Обоснование:** обновление ~200M параметров первого слоя — основная стоимость шага. На CPU замер шага (70k × 256): 108 мс → 32 мс; на CUDA fused тоже быстрее. Результаты на ODS не изменились.
- **Проверка:** `kkt_clf/train.py::_make_optimizer`; время эпохи в `history.csv`.

### D-016 · 2026-10-22 · §4.5 · Инициализация первого слоя
- **Решение:** оставлена как у `nn.Linear` (`U(±1/√n_features)`), bias тоже.
- **Альтернативы (замер на ODS, `ods_light`, 15 эпох, test L3 acc / macro-F1 без отказа):** `nn.Linear`-like — 0.804 / 0.712; N(0, 0.1) — 0.803 / 0.711; N(0, 0.7) — 0.726 / 0.601.
- **Обоснование:** разницы с N(0, 0.1) нет, крупная инициализация хуже; без подбора (§1) берём стандартную.
- **Проверка:** скрипт эксперимента не коммитился; повтор — подменить инициализацию в `SparseLinear.__init__` и запустить `scripts/run_train.py`.

### D-017 · 2026-10-22 · §6 · Фикстуры инференса и метрики на них
- **Решение:** `infer_*` сэмплируются из всего ODS, поэтому пересекаются с train обучающего прогона; метрики на них — только проверка механики (неизвестные метки, пустые признаки, проброс колонок), а не качество.
- **Альтернативы:** брать вход инференса из `outputs/<run>/splits/test.parquet` — это выход прогона, а не фикстура генератора.
- **Проверка:** для честной оценки на отложенных данных — указать в ноутбуке инференса `input_path = outputs/train_ods/splits/test.parquet`.

### D-018 · 2026-10-31 · S01 · Независимый TF-IDF fit
- **Цель и решение:** разрешить повторное обучение MLP на заранее построенных двух TF-IDF ветках. По умолчанию сохраняется fit на размеченном train; `pretrained` загружает автономный FeatureBuilder и копирует его в артефакт модели.
- **Альтернативы:** потоковый подсчёт IDF и случайная выборка по всему корпусу отложены; выбран ограниченный префикс заранее подготовленного файла. Он зависит от версии и порядка строк, повторы сохраняются.
- **Ограничения:** sklearn fit не потоковый, SHA-256 требует полного прохода по файлу; совпадение текстов с val/test делает оценку трансдуктивной. Спецификация первой итерации предписывает fit только на train; внешний режим добавлен по прямому требованию S01, исходная спецификация не менялась.
- **Проверка:** команды и результаты в `artefacts/S01/result.md`; промышленные объёмы и GPU не проверены.

## S03 — supervised Chi² и воспроизводимое сравнение

- Цель: сравнивать full, global и class-aware на единых train/validation при повторном fit выбранных словарей на исходной выборке корпуса. Альтернативы: срез нормированной матрицы и общий рейтинг вместо циклического выбора отклонены как нарушающие контракт признаков и гипотезу равноправия классов.
- Рейтинг вычисляется по train; для class-aware используются только положительные OvR-ассоциации и циклический выбор уникальных признаков, при исчерпании — отмечаемый global-добор. Рейтинги хранятся на диске, что снижает RAM ценой дискового пространства и I/O; оценки сохраняются с точностью float64. Версия артефакта selection `@3` не загружает старые версии.
- Проверка: синтетические Chi², идентичность корпуса и IDF, разделённая нормировка, train/save/load, общий validation, статусы и регенерация отчёта. Значения на ODS диагностические; высокая масштабируемость и качество не доказаны. Замечания двух независимых раундов и отсутствие независимого ревью последних исправлений фиксируются в отчёте спринта.

## S04 — незавершённая проверка Gradient × Input

- Цель: проверить margin true L3 против сильнейшего конкурента на активных CSR значениях, с положительной class-aware агрегацией и повторным fit выбранных словарей. Альтернативы: dense матрицы и срез нормированного TF-IDF противоречат контракту.
- Ограничения: первый smoke имеет недостоверные G×I рейтинги из-за исправленного впоследствии дефекта; независимое ревью выявило дополнительные открытые дефекты совместимости, памяти и отчёта. Не делать вывод о качестве и масштабировании.
- Проверка: воспроизведение и текущие статусы приведены в `artefacts/S04/result.md`; приёмка и повторное ревью не завершены.

### S04 — продолжение после независимого ревью

- Исправлены замечания о fingerprint параметров преобразования и весов, общих затратах векторизации, незавершённых бюджетах, пустых строках и seed 0; неизвестная стоимость внешнего checkpoint остаётся null. Альтернатива скрывать неизвестные затраты нулём отвергнута.
- Проверка: независимые регрессионные тесты и общий smoke на фикстуре, см. `artefacts/S04/result.md`. Исправления внесены после второго, последнего раунда независимого ревью; подтверждать их третьим ревью без изменения согласованного бюджета нельзя. Промышленная применимость и GPU не проверены.

### S04 — целостность повторно используемых рейтингов

- Каталог объявленных рейтингов проверяется до повторного отбора целиком: обе ветки, class counts, размеры файлов и checksum-секция, если она присутствует; новые манифесты содержат хэши и размеры. Для старых манифестов без checksum-секции сохраняется структурная проверка без утверждений о криптографической целостности.
- Основание: `S04_review.zip` содержал неполную выгрузку рейтингов, а отдельное воспроизведение показало, что прежний прямой загрузчик не отклонял отсутствующий объявленный файл. Проверки и границы приёмки — `artefacts/S04/result.md`; последние исправления независимого ревью не имеют.

### S04 — пользовательская приёмка с оговорками (2026-11-06)

- Пользователь принял S04 с зафиксированными ограничениями и явно принял риск отсутствия независимого повторного ревью исправлений, внесённых после второго раунда. Дополнительный раунд не проводится.
- Основание: полный pytest 242 passed, 1 skipped; общий smoke 10/10; повторная загрузка 7/7 selection-артефактов. Частичные DoD остаются partial, GPU и промышленный масштаб не проверены; одноэпоховый smoke не доказывает качество. Полные сведения — `artefacts/S04/result.md`.

### S05 — изоляция размеченных текстов (2026-11-06)

- Цель: зафиксировать сравнимые части без пересечения совпадающих нормализованных названий. Выбран групповой стратифицированный split с ключом char-предобработки, seed 42, ориентиром 80/10/10 и monitor внутри outer train; конфликты меток и невозможность покрытия классов приводят к отказу, а не к исправлению данных. Строковый стратифицированный split не обеспечивает изоляцию повторов на произвольных данных.
- Аудит исходных 18 046 строк не обнаружил нормализованных коллизий; `min_count=10` исключает 411 L3 и 1 411 строк. Найдены два совпадения сохранённых нормализованных названий с неразмеченным корпусом S02; качество на полностью невиданном корпусе этими данными не устанавливается.
- Проверка: повторная генерация пяти частей дала идентичные хэши; независимый тест обнаружил ошибочный счётчик в исходном `split_audit.csv`. Код исправлен, исходный файл сохранён, исправленная версия — `outputs/S05/audit/split_audit_corrected.csv`. Независимое ревью всего спринта и обучение ещё не завершены.

### S05 — обученный baseline и условный gate

- Три refit seed на полном TF-IDF выполнены после train-only выбора 10 эпох; сравнение только на comparison validation, reference seed 42 задан заранее. Средний macro-F1 0.703132, majority 0.000248; кривая monitor вышла на плато при сильном train–monitor gap. Тестовая часть не вскрывалась для качества.
- Альтернативы: выдавать `ready` по одному высокому F1 или сопоставлять напрямую с историческим `ods_light` отвергнуты из-за gap, различий в признаках и протоколах. Gate `conditional`; последнее исправление протокола и отчёта после второго ревью не имеет третьей независимой проверки. Подробности и незакрытые DoD — `artefacts/S05/result.md`.

### Решение пользователя для S06 — новый протокол разметки

- До разбиения исключать L3 с поддержкой менее 12 строк; граница включительная: `count >= 12`. Затем воспроизводимо и стратифицированно сформировать train, comparison validation и закрытый test с сохранением исходных индексов, seed, fingerprint и поддержки категорий; в validation и test должно быть не менее двух строк каждого сохранённого L3.
- Выбор эпох и гиперпараметров выполняется на внутреннем monitor только внутри train; comparison validation используется для сравнения конфигураций, test остаётся закрытым до окончательного выбора. Проверять пересечения одинаковых/нормализованных названий и утечки между выборками. Сохранять статистику строк, L3 и распределения support до/после фильтра и размеров частей.
- Точные доли не зафиксированы: сначала S06 проверит выполнимость на фактическом распределении и группах. Baseline необходимо обучить заново на новом split; S05 остаётся историческим ориентиром и не смешивается с новым экспериментом. Реализация S06 в S05 не входит.

### S05 — пользовательская приёмка с оговорками

- Пользователь принял S05 с неполностью выполненными DoD и явно принял риск отсутствия независимого ревью последних исправлений; новые раунды ревью запрещены. Разрешены commit и fast-forward merge в dev без push. Подробности фактических результатов и ограничений — `artefacts/S05/result.md`.

### S06 — фиксация наблюдаемого состояния (2026-11-06)

Цель — честное сравнение трёх методов отбора на общем новом split ODS. Выбор эпох: 15 по внутреннему train_monitor, до comparison validation; постоянный LR. Альтернатива — использовать исторический S05 baseline — отклонена, поскольку фильтр и split отличаются. Проверка: полные baseline seed 42/43/44 обучены и оценены только на comparison validation; sealed test не вскрывался для качества. Два совпадения имён с неразмеченным корпусом делают режим частично трансдуктивным. Текущий план сетки сохранён после оценки baseline, а второе ревью выявило блокирующие риски provenance и resume: вывод о сравнении методов пока невозможен. Ограничения и актуальный прогресс — `artefacts/S06/result.md`.

### S06 — восстановление, частичный результат (2026-11-07)

Для безопасного восстановления не запускались финальные 30 обучений. Отчёт и фиксация протокола исправлены частично; альтернативу считать старый baseline частью нового сравнительного эксперимента исключили из-за поздней фиксации плана. Проверка после добавления тестов: полный pytest 375 passed, 9 failed, 1 skipped; gate и provenance/resume пока не обеспечивают fail-closed. Kimi недоступен из-за лимита провайдера, поэтому интеграционный smoke не проведён. Две независимые проверки уже исчерпаны; исправления после последней не считать независимо принятыми.

### S06 — механическая проверка восстановления

После возобновления решено временно отключить resume, поскольку проверенное повторное использование всех артефактов сложнее надёжно подтвердить; повторный вызов отказывает без перезаписи. Синтетический CPU smoke в новом каталоге подтвердил 10 конфигураций, fixed-vocabulary refit, paired Δ и восстановление отчёта из таблиц; полный pytest 421 passed, 1 skipped. Это проверка механики, не сравнение методов на ODS. Полная сетка и её ресурсы не проверены, исторический baseline не переносится в новый протокол, последние исправления не проходили дополнительное независимое ревью.

### 2026-11-07 — S06 final comparison and external split report binding

The final comparison uses a new CPU experiment directory with a frozen 30-run plan and 14-epoch train-only schedule, preserving earlier outputs. The original external split manifest is bound by fingerprints instead of copied or assumed to live under the new output root. Alternatives were overwriting prior outputs, copying the split without provenance, or omitting the verification; all were rejected to preserve provenance and prior work. GPU initialization failed; CPU retains the same model protocol. All 30 runs and nine shared refits completed, and report reconstruction from report tables alone succeeded. The comparison validation was previously examined, so this is an ODS PoC comparison rather than an untouched evaluation or industrial model selection. External-split reporting fixes were made after the second independent review and remain independently unreviewed; regression and final test outcomes are in `artefacts/S06/result.md`.

### S06 — пользовательская приёмка с оговорками и закрытие

Пользователь принял сравнительный ODS-эксперимент S06 с ограничениями и разрешил commit и fast-forward merge в `dev` без push. Завершено 30/30 runs; итоговый полный pytest — 436 passed, 1 skipped. Сохранены оговорки о ранее просмотренной validation, ненадёжности оценок редких L3, CPU-only измерениях и отсутствии независимого ревью последних исправлений отчёта. Результат не является оценкой промышленного качества; sealed test не использовался для метрик. Полные доказательства и артефакты перечислены в `artefacts/S06/result.md`.


### 2026-11-07 — S07 development resampling and frozen protocol

Use exactly S06 outer_train + comparison_val (14,652 rows), retain all 293 L3
without a new support filter, and structurally exclude the sealed 1,637 rows.
Normalized development names form singleton groups; actual assignments for
seeds 101/202/303 satisfy 6 fit / 2 monitor / 2 validation minima per L3. Reuse
the verified final S06 14-epoch constant-LR configuration, not historical
15-epoch attempts, and rank independently per split with reference model seed42.
Alternatives of reusing S06 supervised rankings, dropping rare development
classes or retuning on S07 validation are outside scope. Verification uses
pinned sources, deterministic split witnesses, an immutable 36-run plan,
independent synthetic/negative tests and at most two read-only Kimi rounds.
Resume is deliberately unsupported: preserve interrupted roots and create new
plans. New development splits remain previously explored ODS data, not an
external holdout; the S06 +0.0139 observation remains a motivating hypothesis.


### 2026-11-07 — S07 strict LR diagnostic serialization after final review

The first formal attempt `outputs/S07/stability_20261107_final/` stopped with
0 success / 1 failed / 35 pending before validation inference. The constant-LR
controller emits an uncomputed `val_loss_subset=NaN` sentinel every ten steps;
strict JSON rejected the training record. Saved model tensors were finite, but
unsaved epoch history cannot be retrospectively verified. Preserve this root
and its partial report. The alternative of allowing NaN globally or scrubbing
scientific metrics was rejected. Delegated S07-only fix converts this specific
uncomputed diagnostic to null and explicitly rejects nonfinite model tensors;
training/metric values remain strict. Independent test author added 53 cases,
including >=10 real optimizer steps, save/reload/inference/report and negative
loss/weight/measured-diagnostic checks. All S07 tests: 143 passed; full suite:
579 passed, 1 skipped. Both Kimi review rounds were already exhausted: this
serialization/finite-model fix is independently unreviewed, despite regression
coverage. No third review. V2 uses a fresh immutable plan/root, no reuse of the
failed model, and identical scientific protocol. Plan file SHA256:
`cafcabcdf5e33688d11cc2dfd31ce70743d8a6d09683efda3dc505524fff8deb`.


### 2026-11-07 — S07 observed robustness and conditional next-study recommendation

Final v2 completed36/36 and nine exact-corpus refits under the frozen protocol.
Class-aware100k mean pairedΔmacro-F1 +0.000873 (split means negative/negative/
positive;5/9 seed gains), G×I +0.000085 (positive/negative/negative;6/9), global
-0.003572 (positive/negative/negative;3/9). S06 +0.013905 does not generalize
as a stable gain across these development assignments. Prefer reporting all
three split effects rather than choosing the best split or declaring a winner
from the correlated nine pairs. No confidence intervals based on nine
independent observations. Source ODS remains explored and unrepresentative.

100k reduces model bytes ~4x and mean CPU training ~3.6x, so it merits further
research as a cost-saving configuration alongside full baseline. The alternative
of promoting it on improved accuracy is unsupported; class-aware's small F1
gain concentrates in the2–4-support bin (200classes), with negative effects in
other bins. G×I has lower macro-F1 than class-aware by0.000788 and additional
pipeline cost, so the study does not justify its attribution premium. Larger
independently held-out evaluation is the proposed next research step, not work
performed in this sprint. No industrial recommendation.

Verification: full pytest579passed/1skipped,143 S07 tests, independent numerical
recomputation of36metrics/27deltas and18feature-name overlaps, per-class/bin/
confusion checks, byte-identical HTML from report-only tables. Two Kimi reviews
exhausted; post-review serialization fix remains independently unreviewed and
requires explicit acceptance of that limitation. See S07/result.md. No commits,
merges or pushes; sprint acceptance remains with the user.


### 2026-11-07 — S07 user acceptance with limitations and integration authorization

The user explicitly accepted S07 and its documented limitations, including
small L3 validation support, prior exposure to development/validation results,
non-independent development splits, CPU-only measurements, and no independent
review of the final tested LR-diagnostics serialization fix. Industrial quality
and scalability remain unestablished. The final36/36 study with three splits,
three model seeds and nine refits is the accepted evidence; earlier failed
attempts remain historical and preserved. No method showed a consistent
split-level improvement; the S06 Class-aware advantage did not reproduce,
while100k materially reduces model bytes and training time.

Commit and fast-forward-only merge to `dev` are explicitly authorized after
mandatory final tests and Git/report verification. If tests regress or a
fast-forward is impossible, stop rather than change scientific gates or rewrite
history. Push, branch deletion, cleanup of outputs, new experiments, new review
rounds and S08 are not authorized. Stage only S07 implementation, independent
tests, technical documentation and the revised role instructions; preserve the
unrelated user ZIP archives/notebook untracked. SPEC_v1.0 and prior decisions
remain unchanged. Acceptance is final; integration is gated by verification.


### 2026-11-07 — S07 closure verification passed

Fresh full pytest:579passed/1skipped in77.99 s (CUDA warning). Frozen-plan and
saved-report checks confirm36successes, complete manifest, fixed splits/seeds,
nine refit receipts. Both `git diff --check` and `git diff --cached --check`
pass. Staging contains28 intended S07 files only, including revised AGENTS;
no datasets, models, outputs, archives, unrelated notebook or S08 documents.
All3,717 existing experiment files retain size/mtime, protected source/notebook
hashes unchanged. Concurrent ZIP disappearance and untracked S08 document
arrival were observed; closure performs no deletion/restoration or S08 work.
These unrelated changes do not intersect the S07 commit/fast-forward paths.
Proceed with the authorized commit and fast-forward-only merge, without push.
