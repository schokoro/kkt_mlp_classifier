# S04 — Принят с оговорками

Ветка `sprint/S04-gradient-x-input`, база `dev` f974016 (интегрированный S03). Пользователь принял S04 **с оговорками** 2026-11-06 и принял риск того, что последние исправления выполнены **после второго и последнего разрешённого ревью и не имеют независимого повторного ревью**. Это пользовательское решение о приёмке не меняет фактические статусы частично выполненных DoD. `SPEC_v1.0.md`, пользовательские данные и S02/S03 outputs не изменялись. Реализация и исправления — `kimi -p`, тесты — отдельные `codex exec`, два независимых ревью — отдельные read-only `codex exec`; полные задания сохранены в `prompts/`.

| DoD | Статус | Факт и ограничения |
|---|---|---|
| D01 | done | Ветка создана от f974016; неотслеживаемые пользовательские файлы и data/corpus.pqt сохранены. |
| D02 | partial | Конфиг, два режима checkpoint и train-only реализованы; проверки параметров преобразования, весов и сплитов расширены после последнего ревью и не проверены независимым ревьюером. |
| D03 | done | Аналитический margin, конкурент и конечные разности проверены независимыми тестами. |
| D04 | partial | Sparse CSR и CPU, активации и отсутствие обновления весов проверены; CUDA недоступна для PyTorch и не проверена. |
| D05 | done | Positive-only class means с пустыми строками, round-robin и fallback покрыты тестами; ключевые исправления проверялись вторым ревью. |
| D06 | partial | Рейтинги, индексы и provenance сохраняются; повторная загрузка трёх наборов и контроль файлов проверены. Исправление проверки целостности внесено после последнего ревью; масштаб ~1000 L3 не измерен. |
| D07 | partial | Refit, IDF, нормировка и round-trip проверены на фикстуре; refit одного бюджета каждого метода выполнен на исходных 981850 строках `data/corpus.pqt`. Бюджеты 50k/100k/200k не проверены. |
| D08 | partial | Train/save/load/inference и baseline проверены на малых данных; последние изменения не проходили повторное независимое ревью. |
| D09 | done | Все 10 конфигураций ресурсно ограниченного общего smoke успешно выполнены на единых split/label maps; это одноэпоховый smoke, не диагностическая кривая. |
| D10 | partial | Все пять файлов созданы, статусы и пересоздание HTML проверены; исправления учёта ресурсов после второго ревью независимо не проверены. |
| D11 | partial | CPU время/RSS этапов и синтетический тест памяти выполнены; GPU/VRAM и промышленный масштаб не проверены. |
| D12 | partial | Быстрый и полный pytest прошли, `git diff --check` выполнен, оба раунда ревью завершены; последние исправления, включая проверку рейтингов, не прошли дополнительное независимое ревью в пределах бюджета. |
| D13 | done | Данный отчёт, README, STATUS и append-only DECISION_LOG обновлены с ограничениями; SPEC не менялась. |

## Реализованный контракт

Добавлены `kkt_clf/attribution.py`, `scripts/run_s04_experiment.py`, расширены `kkt_clf/experiment.py`, `selection.py` и экспорты `__init__.py`. Метод `gradient_x_input` использует margin истинной L3 к сильнейшей другой L3 и градиент по активным значениям CSR в eval; положительные атрибуции усредняются по классу и объединяются общим S03 round-robin. Selected TF-IDF строится повторным fit фиксированных словарей; конечные MLP обучаются с нуля. Реализованы сохранение рейтингов/индексов, checkpoint train/load с provenance, общие и отдельные стадии, единый отчёт. Возможность использовать свой full TF-IDF, корпус, budgets, char share, seed и sampling задаётся CLI. Полный diff и новые тесты находятся в рабочем дереве, не закоммичены.

## Выполненные runs и воспроизведение

В итоговом общем smoke все 10 runs получили `success`: `none`, `global_{200,400,800}`, `class_aware_{200,400,800}`, `gradient_x_input_{200,400,800}`. Размеченный train/val: 1600/200 строк, 40 L3; словарь full 3000 char + 1000 word. Одноэпоховая attribution MLP: best epoch 1, val raw top-1 accuracy 0.03, macro-F1 0.00485. Это проверка механики, не содержательное сравнение методов. Default budgets 50k/100k/200k и реальный корпус в этом общем прогоне не проверялись.

Отдельный минимальный общий smoke на фактическом корпусе выполнен командой:

```bash
.venv/bin/python scripts/run_s04_experiment.py run --output outputs/S04_corpus_smoke --full-tfidf outputs/S02/tfidf_1m --corpus data/corpus.pqt --data fixtures/ods_small.parquet --budgets 200,400,800 --char-share 0.75 --preset ods_light --override '{"model":{"hidden":[32],"dropout":[0.1]},"train":{"epochs":1,"device":"cpu","scheduler":"none","num_threads":2}}' --only none,global_200,class_aware_200,gradient_x_input_200
```

`none`, `global_200`, `class_aware_200`, `gradient_x_input_200` — success; остальные шесть плановых runs — skipped с причиной `--only`. Пять файлов отчёта находятся в `outputs/S04_corpus_smoke/report/`. Full TF-IDF S02 содержит 300000 char и 100000 word, исходный корпус SHA-256 `f9a057a00db10ed115dd5f47e035f3a3c56c710d9463f47d38812d4cceef3dbc`, 981850 строк; full `features.joblib` SHA-256 `7ff42885729d0bcfb2b6e9e0893b1be4ddf59562705e3ed048614183cd952052`. Каждая выбранная ветка повторно fitted на том же снимке корпуса. Attribution MLP: 1 эпоха, val accuracy 0.055, macro-F1 0.04056 — только механика, не диагностическая кривая.

Фактически выполнены:

```bash
.venv/bin/python scripts/fit_vectorizer.py --input fixtures/ods.parquet --output /tmp/s04-final.m3wJ1o/full --max-rows 20000 --features-preset ods_light --override '{"char":{"max_features":3000},"word":{"max_features":1000},"vectorizer_fit_rows":null}' --seed 42 --source-id s04-fixture-smoke
.venv/bin/python scripts/run_s04_experiment.py run --output outputs/S04_smoke_final --full-tfidf /tmp/s04-final.m3wJ1o/full --corpus fixtures/ods.parquet --data fixtures/ods_small.parquet --budgets 200,400,800 --char-share 0.75 --preset ods_light --override '{"model":{"hidden":[32],"dropout":[0.1]},"train":{"epochs":1,"device":"cpu","scheduler":"none","num_threads":2}}'
```

Для повтора использовать новые каталоги: CLI не перезаписывает существующие результаты. Векторизатор находится в `/tmp` и может быть удалён системой; команда выше создаёт новый из зафиксированной фикстуры. Фикстура ODS может пересекаться текстами с validation/test при unsupervised fit — оценка трансдуктивна.

Отчёт: `outputs/S04_smoke_final/report/{report.html,runs.csv,per_class.parquet,stages.csv,manifest.json}`. Все пять файлов существуют. Копия четырёх табличных файлов в `/tmp/s04-final.m3wJ1o/report_only/report` использована для команды `.venv/bin/python scripts/run_s04_experiment.py report --output /tmp/s04-final.m3wJ1o/report_only`: HTML построен без retrain. Исходный старый `outputs/S04/report` получен до исправления дефекта рейтингов и не годится для интерпретации. `outputs/S04_final` — неуспешная попытка до обучения из-за исчезнувшего временного TF-IDF; не смешивать её с итоговым smoke.

Provenance: корпус `fixtures/ods.parquet` SHA-256 `6dba600b6f5d2437e225d19f19e77938543537d91b395cca6c0948279122795d`, full features `fba4fa0eb0491eb8fb0d537076e4cdc11e281d5839b8c7013506687d40da79df`, train split `81967b6e97e350244291b5b70765016e6fdd3a973a17990f514599cbc14e8b1a`, val split `d0212082ab7828e9f390e0fa1a46e44191874be5f8e24b22b85846db8b4854ba`; конфигурации, веса, label maps и прочие хэши — в manifest и checkpoint provenance. Рейтинги и selected индексы лежат вне report в `outputs/S04_smoke_final/shared/attribution/` и `outputs/S04_smoke_final/runs/*/selection/`.

## Время и память

CPU smoke: общий transform 0.051 с, RSS 707956736 байт; attribution checkpoint training 0.729 с, RSS 870735872 байт; вычисление атрибуций 0.018 с, RSS 873996288 байт. G×I selection 0.0028–0.0048 с; selected TF-IDF refit 0.424–0.462 с; финальное обучение 0.0116–0.0133 с; validation inference 0.0016–0.0017 с. Максимальный RSS G×I этапов 921006080 байт. Это абсолютные RSS процесса, пики этапов не складываются. Значения локальны и не являются промышленными прогнозами. VRAM не измерена: GPU виден через `nvidia-smi`, но PyTorch не смог инициализировать CUDA даже при отдельном тесте вне песочницы. Синтетический RSS-тест рейтингов прошёл после последовательной записи; число классов и признаки в нём ограничены, промышленная масштабируемость не проверена.

Отдельно на `data/corpus.pqt`: shared vectorization 0.092 с / RSS 846979072 байт; attribution model train 3.565 с / 1384804352 байт; attribution 0.078 с / 1174511616 байт; G×I selection 0.007 с / 1302597632 байт; refit 21.464 с / 1660473344 байт; финальная MLP 0.010 с и inference 0.0017 с / 1331572736 байт. Эти абсолютные RSS не складывать; данные и размеры относятся только к указанному smoke.

## Тесты и ревью

- `.venv/bin/python -m pytest -q -m 'not slow'`: **175 passed, 1 skipped, 1 deselected**; `.venv/bin/python -m pytest -q`: **176 passed, 1 skipped** (включая ноутбуки, выполнен исполнителем после последних изменений). Skip — CUDA. Отдельная GPU-попытка: 1 skipped.
- Независимые тесты после второго ревью: `tests/test_s04_review2_independent.py` — 15 passed; проверены параметры TF-IDF и веса, shared cost, failed/skipped, пустые строки и seed 0. Дополнительно интеграционный smoke train/load подтвердил отказ при подмене и null неизвестной стоимости внешнего checkpoint.
- `git diff --check` — без замечаний; `git diff --no-index --check` для новых файлов не вывел whitespace-ошибок (exit 1 означает наличие diff).
- Первое ревью — `review_1.md`, второе — `review_2.md`. Все пять замечаний второго раунда переданы Kimi (`prompts/13_kimi_review2_fixes.md`) и покрыты тестами (`prompts/14_codex_review2_tests.md`), но **независимое повторное ревью последних исправлений не проводилось**. Результаты ревью нельзя считать независимым подтверждением исправлений, внесённых после него.

Не проверены полномасштабный корпус, GPU, переносимость рейтингов и промышленное качество. Победитель по PoC не выбирается.

## Дополнительная проверка пакета и целостности рейтингов

Внешняя проверка использовала `S04_review.zip`. Причина расхождения — неполная комплектация ZIP: он содержит 15 записей, в том числе отчёты обоих smoke и `outputs/S04_corpus_smoke/shared/attribution/ratings/meta.json`, но не содержит бинарных файлов рейтингов ни одного прогона и отдельного `ratings/meta.json` для `S04_smoke_final`. Метаданные рейтингов smoke доступны внутри manifest отчёта. Архив сам по себе не является полным переносимым набором рейтингов; он не изменялся. В исходных каталогах `outputs/S04_smoke_final/shared/attribution/ratings` и `outputs/S04_corpus_smoke/shared/attribution/ratings` оба файла `char_class_counts.npy` и `word_class_counts.npy` физически присутствуют. Для обеих веток каждого набора: 40 классов, сумма N_c = 1600, размеры индексов и оценок согласованы с суммой rank_counts. Существующие результаты не переписывались.

При проверке обнаружен самостоятельный дефект прямой повторной загрузки: копия каталога без объявленного `char_class_counts.npy` принималась `select_features_gradient_x_input`. Аналогичная проблема относилась к word. Исправления `kkt_clf/selection.py` и `kkt_clf/attribution.py` проверяют наличие всех объявленных файлов, размеры и счётчики обеих веток, полноту и значения checksum-секции при её наличии. Новые рейтинги получают SHA-256 и размеры каждого из 12 файлов в `ratings/meta.json`; в старых наборах без этой секции выполняется структурная проверка, а не предполагаемая криптографическая проверка. Сохранённые selection-артефакты дополнительно проверяют свои хэши. Удаление файла, неполная/пустая секция хэшей, повреждённые массивы, несовпадающие N_c и ошибочная сумма строк отклоняются. Старые корректные форматы S03/S04 продолжают загружаться.

Повторный smoke на актуальном коде выполнен в новом каталоге:

```bash
.venv/bin/python scripts/run_s04_experiment.py run --output outputs/S04_ratings_smoke --full-tfidf /tmp/s04-final.m3wJ1o/full --corpus fixtures/ods.parquet --data fixtures/ods_small.parquet --budgets 200,400,800 --char-share 0.75 --preset ods_light --override '{"model":{"hidden":[32],"dropout":[0.1]},"train":{"epochs":1,"device":"cpu","scheduler":"none","num_threads":2}}'
```

Все 10 runs — success; `outputs/S04_ratings_smoke/report/` содержит все пять отчётных файлов. В новых общих рейтингах — 12 записей SHA-256, файлы присутствуют в обеих ветках, сумма N_c = 1600. На трёх каталогах (`S04_smoke_final`, `S04_corpus_smoke`, `S04_ratings_smoke`) заново проверены целостность метаданных, `load_selection_artifact` и повторный отбор для всех семи сохранённых G×I selection-артефактов: char/word индексы и оценки совпали. Для старых общих ratings отсутствуют встроенные хэши; проверки хэшей selection-артефактов и структурные проверки не доказывают криптографическую целостность первоначального общего каталога до сохранения артефакта.

Независимые тестовые задания сохранены в `prompts/16_codex_ratings_tests.md` и `prompts/18_codex_integrity_followup_tests.md`; реализации — `prompts/15_kimi_ratings_integrity.md` и `prompts/17_kimi_integrity_followup.md`. Регрессии: `tests/test_s04_ratings_integrity.py` и `tests/test_s04_manifest_completeness_independent.py`. Последний быстрый прогон `.venv/bin/python -m pytest -q -m 'not slow'`: **241 passed, 1 skipped, 1 deselected**; полный `.venv/bin/python -m pytest -q`: **242 passed, 1 skipped**. Skip — CUDA. `git diff --check` выполнен без замечаний. Третий независимый code review не запускался: новые исправления проверки рейтингов и все ранее внесённые после второго ревью изменения остаются **без независимого повторного ревью**. Пользователь принял этот риск при закрытии спринта. С учётом фактических ограничений D02/D04/D06/D07/D08/D10/D11/D12 остаются partial.

### Финальная проверка перед интеграцией

При добавлении ранее неотслеживаемых документов в индекс `git diff --cached --check` сообщил о конечных пробелах в исходном `value_dod.md` и двух полных, фактически отправленных промптах независимого ревью (`prompts/04_codex_review.md`, `prompts/12_codex_rereview.md`). Они сохранены без изменений, поскольку являются исходным контрактом и проверяемым следом заданий. Проверка staged diff с исключением этих трёх неизменяемых документов прошла; проверка рабочего diff `git diff --check` также прошла. Корпуса, архивы, временные файлы и outputs в индекс не включены; ранее отслеживаемый `data/ods_labels.csv` не изменён.
