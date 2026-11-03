# S02 — результат, 2026-11-02

Проверена техническая интеграция внешнего TF-IDF; код и тесты не изменялись. Ветка `sprint/S02-corpus-tfidf-validation` создана от `master` c853b53 (равен локальному `origin/master`; S01 8a46d37 входит в историю). До создания имелись изменённый `AGENTS.md` и неотслеживаемые S02, корпус, `data/ods_alpha.pqt` и два ноутбука; они сохранены. Удалённый репозиторий не опрашивался. Коммитов, push и merge не выполнялось. Ограничение S01: исправления после второго ревью не имели повторного независимого ревью; S01 принят владельцем с этой оговоркой.

## Вход и измерения

`data/corpus.pqt`: 24 816 862 байта, Parquet `name: large_string`, 981 850 строк, null 0, пустых/пробельных 0, точных повторов непустых значений 0; длина name: медиана 39, 95-й перцентиль 74, максимум 170 символов. Инвентаризация через PyArrow, файл не изменялся и не добавлялся в Git. SHA-256: `f9a057a00db10ed115dd5f47e035f3a3c56c710d9463f47d38812d4cceef3dbc`. `source_id=corpus_1m_v1`, seed 42. Префикс непустых строк, без дедупликации; это не случайный сэмпл. Совпадает с контрактом метаданных S01. Полный доступный корпус меньше миллиона.

| Прогон | max_rows | rows_read / rows_used / rows_fit | truncated | char / word | время | пик RSS |
|---|---:|---|---|---|---:|---:|
| 100k | 100 000 | 100 000 / 100 000 / 100 000 | true | 231 308 / 25 240 | 14,332 с | 1 409 184 КиБ |
| весь корпус | 1 000 000 | 981 850 / 981 850 / 981 850 | false | 300 000 / 100 000 | 100,709 с | 5 280 804 КиБ |

Оба fit завершены, `subsampled=false`, `rows_dropped_empty=0`. Ограничение `vectorizer_fit_rows=1 500 000` не сработало. В полном прогоне обе ветки достигли `max_features`; это не оценка полноты или качества словаря. Настройки и версии записаны в `outputs/S02/tfidf_{100k,1m}/vectorizer_meta.json`. `/usr/bin/time` отсутствует: время измерено `time.monotonic`, пик дочернего процесса — `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` на Linux. Полный fit запущен в systemd user scope с `MemoryMax=8G`, `MemorySwapMax=0`; на старте доступно около 15 ГБ RAM. Это ограничение снижает риск, но не оценка ресурсов других корпусов.

Основные фактически выполненные команды (измерительная обёртка вызывает указанную команду через `subprocess.run`):

```bash
.venv/bin/python scripts/fit_vectorizer.py --input data/corpus.pqt --output outputs/S02/tfidf_100k --max-rows 100000 --features-preset default --seed 42 --source-id corpus_1m_v1
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 .venv/bin/python -c 'import subprocess,time,resource; t=time.monotonic(); p=subprocess.run([".venv/bin/python","scripts/fit_vectorizer.py","--input","data/corpus.pqt","--output","outputs/S02/tfidf_1m","--max-rows","1000000","--features-preset","default","--seed","42","--source-id","corpus_1m_v1"]); print("MEASURE elapsed_seconds=%.3f maxrss_kib=%d exit=%d"%(time.monotonic()-t,resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,p.returncode),flush=True); raise SystemExit(p.returncode)'
.venv/bin/python scripts/make_fixtures.py --out outputs/S02/fixtures
.venv/bin/python scripts/run_train.py --preset default --override '{"data":{"path":"outputs/S02/fixtures/ods_small.parquet"},"features":{"mode":"pretrained","pretrained_path":"outputs/S02/tfidf_1m"},"model":{"hidden":[64],"dropout":[0.1]},"train":{"epochs":1,"device":"cpu","scheduler":"none"},"output_dir":"outputs/S02/smoke_train"}'
.venv/bin/python scripts/run_inference.py --override '{"artifact_dir":"outputs/S02/smoke_train/artifact","input_path":"outputs/S02/fixtures/infer_no_labels.parquet","output_path":"outputs/S02/smoke_inference/predictions.parquet","device":"cpu"}'
.venv/bin/python -m pytest -q -m 'not slow'
.venv/bin/python -m pytest -q
git diff --check
```

Для обоих артефактов отдельной одноразовой проверкой через `load_vectorizer_artifact`, `joblib.dump/load` в памяти и NumPy/SciPy проверены идентичность порядка имён признаков и IDF, CSR transform четырёх одинаковых строк до/после повторной сериализации, размеры и равенство с отдельными преобразованиями веток по границе char/word. Для полного артефакта shape `(4, 400000)`, граница 300 000, nnz 120; для 100k `(4, 256548)`, граница 231 308, nnz 112. Существующий `tests/test_s01_vectorizer.py` дополнительно проверяет исходный fit → save → load и запрещает `FeatureBuilder.fit` при pretrained train.

Smoke: 2 000 размеченных строк, split 1600/200/200, одна эпоха CPU; `run_train.py` сообщил `round-trip OK`. Отдельный инференс загрузил сохранённую модель и обработал 307 строк. При пороге 0,5 отказаны все 307; это технический прогон, качество не оценивается. Код pretrained-ветки `pipeline.py` загружает готовый builder и только вызывает transform; существующий тест подменяет fit ошибкой и проходит.

## D01–D08

- D01 выполнен: отдельная ветка, исходное состояние проверено.
- D02 выполнен: реальный файл инвентаризирован, не изменён.
- D03 выполнен: 100k fit и артефакт сохранены.
- D04 выполнен: fit всех 981 850 доступных строк; ровно 1 млн строк в файле нет.
- D05 выполнен: метаданные и контракт проверены, включая повторную сериализацию; исходный fit → save → load также проверяется имеющимся тестом на малой фикстуре.
- D06 выполнен: pretrained train, round-trip и отдельный inference; отсутствие повторного fit подтверждено кодом и существующим тестом.
- D07 выполнен: быстрые тесты 88 passed, 1 deselected; полный набор 89 passed; оба с предупреждением инициализации CUDA, CPU проверки прошли. `git diff --check` без ошибок (не проверяет неотслеживаемые файлы). Код не менялся, ревью и делегирование не требовались.
- D08 выполнен: этот отчёт; `prompts/` создан, пуст, поскольку делегирования не было. Каталоги `outputs/S02/` игнорируются Git и остаются локальными.

Ограничения: совпадения внешних текстов с train/val/test не измерялись; если они есть, оценка частично трансдуктивна. Источник представляет подготовленный локальный корпус, а не репрезентативную оценку основного массива. GPU, большие корпуса и промышленная масштабируемость не проверены. Отбор признаков и оценка качества вне scope. Новых технических решений, требующих записи в DECISION_LOG, нет.
