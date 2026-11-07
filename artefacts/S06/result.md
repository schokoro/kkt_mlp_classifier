# S06 — Accepted with limitations

The user accepted S06 with documented limitations. The final ODS experiment completed **30/30 successful runs**, and the final full test suite passed **436 tests, with 1 CUDA-dependent test skipped**. The verified aggregate report is `outputs/S06_final_cpu_20261107/report_modelpt_bytes/`; its HTML was rebuilt from saved report tables alone.

Class-aware Chi² with 100,000 features achieved mean macro-F1 **0.7368**, versus **0.7229** for the full 400,000-feature baseline. Serialized model size was 102,778,289 versus 409,978,289 bytes, and mean CPU training time was approximately 18.18 versus 71.31 seconds per run. These measurements apply to this experiment, not to industrial deployment.

Limitations: comparison validation was previously inspected; estimates for rare L3 classes are unreliable because some have only two validation examples; this final experiment was CPU-only; and reporting fixes after the second independent review have not received another independent review. The sealed test was not used for inference or metrics. The entries below the final status preserve earlier failed attempts and recovery as historical records.

Final DoD assessment:

| ID | Status | Evidence / qualification |
|---|---|---|
| D01 | done | S05 ancestry, sprint branch and working tree checked; user files retained. |
| D02 | done | 16,289 rows / 293 L3 after support ≥12; audit and exclusions preserved. |
| D03 | done | Frozen group-aware split, row identity and support verified; sealed test structurally checked only. |
| D04 | done | Full S02 artifact and exact corpus snapshot fingerprints verified; partial transductivity disclosed. |
| D05 | done | Train-only 14-epoch selection, three fresh full baselines and Gate 1 passed; prior validation exposure disclosed. |
| D06 | done | Three train-only rankings persisted with provenance; G×I uses this experiment's seed-42 checkpoint. |
| D07 | done | Nine fixed-vocabulary refits, sizes, IDF, norms, provenance and reload verified. |
| D08 | done | 30/30 runs succeeded under the frozen common plan. |
| D09 | done | Fixed-label macro-F1, accuracy, same-seed deltas and seed summaries independently checked from saved predictions. |
| D10 | done | Per-class, support-bin, validation-prediction and confusion tables saved; support limitations disclosed. |
| D11 | done | Eight report files, timings, RSS, null VRAM and model sizes recorded; table-only HTML rebuild verified. |
| D12 | done | Final full pytest: 436 passed, 1 skipped (CUDA unavailable); final fast pytest: 435 passed, 1 skipped, 1 deselected; CPU experiment and prior integration smoke completed. |
| D13 | done | Kimi implementation and separate Codex tests; two earlier independent reviews; final report fixes after review explicitly remain independently unreviewed. |
| D14 | done | Final measurements, limitations, decisions, status and commands documented; acceptance remains a user decision. |


## Historical record: initial incomplete experiment

Состояние на момент записи: сравнительный эксперимент не завершён. S06 — исследование на ODS, не оценка промышленного качества; sealed test не использовался для inference или метрик. Исторический baseline S05 не использован.

| DoD | Статус | Факты и ограничения |
|---|---|---|
| D01 | done | Ветка `sprint/S06-feature-selection-comparison` создана от `dev`, содержащей `e3c3475`; незакоммиченные файлы пользователя сохранены. |
| D02 | done | 18 046 строк и 737 L3 до фильтра; после support ≥12 — 16 289 строк и 293 L3; исключено 1 757 строк и 444 L3. Аудит: `outputs/S06/audit/data_audit.json`. |
| D03 | done | Seed 42: outer_train 12 960, comparison_val 1 692, sealed_test 1 637, train_fit 11 600, train_monitor 1 360. Минимумы L3: 2 в val/test/monitor, 6 в fit. Manifest SHA-256 `ff44250867e864af23c72da94c38ed4d5a2796af9a658b83787193432ee2cd5d`; повторная генерация дала побитово одинаковые файлы. |
| D04 | done | `outputs/S02/tfidf_1m`: 300 000 char + 100 000 word; SHA-256 `7ff42885729d0bcfb2b6e9e0893b1be4ddf59562705e3ed048614183cd952052`. Корпус SHA-256 `f9a057a00db10ed115dd5f47e035f3a3c56c710d9463f47d38812d4cceef3dbc`, 981 850 строк fit без подвыборки. Два совпадения с ODS после фильтра — частично трансдуктивный режим. |
| D05 | done | 15 эпох выбрано на train_fit/train_monitor до evaluation; baseline seed 42/43/44 на comparison_val: macro-F1 0.717861 / 0.721763 / 0.727254, среднее 0.722292; majority около 0.0003. Опорный checkpoint — seed 42. Monitor при финальном refit входит в outer_train и не является независимой оценкой. |
| D06 | blocked | Сравнительные рейтинги не запускались до исправления замечаний по gate/provenance. |
| D07 | blocked | Девять refit на корпусе не выполнены. |
| D08 | partial | 3 из 30 итоговых запусков завершены; 27 selected ещё не выполнены. Общий машинный план создан после оценки baseline, поэтому требование его фиксации до этой оценки не выполнено. |
| D09 | partial | Метрики baseline получены, парного сравнения методов нет. |
| D10 | partial | Per-class baseline доступен; обязательная сравнительная диагностика и predictions отсутствуют. |
| D11 | blocked | Полный отчёт `outputs/S06/report/` не сформирован из-за ошибок происхождения и сборки отчёта. Частичные таблицы baseline находятся в `outputs/S06/runs/baseline_full/`. |
| D12 | partial | Текущий быстрый набор `.venv/bin/python -m pytest -q -m 'not slow'`: 339 passed, 16 failed, 1 deselected; полный `.venv/bin/python -m pytest -q`: 340 passed, 16 failed. Все падения в тестах сравнительного контура. Регрессии первого ревью: 14 passed; `git diff --check` проходит. Smoke исполнителя выполнен на малых фикстурах, но не заменяет проверку реального эксперимента. |
| D13 | partial | Выполнены два независимых read-only ревью: `review1.md`, `review2.md`. Исправления после второго не проходят третье ревью по установленному пределу. |
| D14 | partial | Промежуточный отчёт; сравнения методов и итогового решения нет. |

Фактические команды включают `git status --short --branch`, `git branch -a --contains e3c3475`, `git switch -c sprint/S06-feature-selection-comparison`, `sha256sum data/corpus.pqt outputs/S02/tfidf_1m/features.joblib`, `scripts/audit_ods.py` с порогом 12, `scripts/freeze_split_s06.py`, `scripts/s06_tfidf_ref.py`, `.venv/bin/python scripts/run_s06_baseline.py select --device cuda`, `.venv/bin/python scripts/run_s06_baseline.py refit --device cuda`, `.venv/bin/python scripts/run_s06_baseline.py verify-split`, `.venv/bin/python scripts/run_s06_comparison.py plan`, `.venv/bin/python -m pytest -q`, `.venv/bin/python -m pytest -q tests/test_s06_review1_regressions_independent.py`, `.venv/bin/python -m pytest -q tests/test_s06_review2_regressions_independent.py --tb=short`. Полные промпты делегирования находятся в `artefacts/S06/prompts/`; обе команды Kimi для исправления сравнительного контура были прерваны до записи изменений, поскольку предложенные решения нарушали совместимость S03 либо fail-closed.

Локальный GPU — NVIDIA RTX 3050, 4 ГБ VRAM; внутри sandbox устройство не обнаруживалось, на хосте использовалось для baseline. Финальное обучение трёх моделей заняло 17.14 / 16.22 / 16.26 с; измеренный пиковый RSS обучения 1.49–1.95 ГБ, VRAM около 1.68–1.70 ГБ. Замеры этапов — `outputs/S06/runs/baseline_full/stages.csv`; пики разных этапов не суммируются. Скорости и потребление ресурсов относятся только к данной машине. Три seed на одном split не являются независимым доверительным интервалом; редкие классы с двумя примерами validation особенно нестабильны.

## Восстановление сравнительного контура (2026-11-07)

**Статус: blocked; к запуску 30 runs контур не готов.** Исторический baseline выше сохранён только как результат прежней попытки: план 30 запусков был зафиксирован после его оценки, поэтому baseline не является частью нового формально корректного сравнительного эксперимента. Ни один из десяти запрошенных smoke runs не выполнен; его нельзя подменять старым smoke в `outputs/S06_comparison_smoke`.

Начальная диагностика воспроизвела 16 ошибок и классифицирована в `recovery_diagnosis.md`. Исправлены включение baseline в сводку, обработка null и пустых confusion pairs, фильтрация failed/pending predictions и table-only восстановление HTML. Добавлена проверка целостности predictions и привязки к строкам comparison validation; целевые положительные и отрицательные проверки проводились на временных фикстурах. Формирование результата сравнительным контуром пока не записывает необходимый hash predictions, поэтому сквозная совместимость отчёта не подтверждена. Функция записи плана теперь требует hash протокола и сверяет сохранённую конфигурацию при повторной фиксации; это не доказывает проверку плана на всех этапах. Общий контракт усечения бюджета S03/S04 не менялся.

Повторный полный `.venv/bin/python -m pytest -q --tb=no` после добавления тестов отчёта: **375 passed, 9 failed, 1 skipped**; CUDA в тестовом sandbox не инициализировалась. Отдельный `tests/test_s06_report_hash_independent.py`: **8 passed**. Оставшиеся падения: точный бюджет S06; допуск трёх baseline; отсутствующий или изменённый hash опорного checkpoint; provenance cached refit, ranking и selection; неверный учёт failed run при возобновлении. Негативные случаи происхождения и безопасного resume в целом не закрыты. `git diff --check` прошёл. Новый интеграционный smoke, результаты всех десяти конфигураций и построение/восстановление полного smoke-отчёта не проверены.

Две прежние независимые проверки кода исчерпаны; третья не проводилась, последние исправления независимо не ревьюированы. Kimi вернул `provider.api_error: 429`, указав на исчерпание лимита пятичасового окна; поскольку реализацию по AGENTS.md выполняет Kimi, завершить оставшиеся изменения в текущей среде не удалось. Сохранены полные промпты `17`–`28` в `artefacts/S06/prompts/`. При диагностике один исполнитель выполнил `git stash` и сразу восстановил его; по проверкам `git status`, `git diff --stat`, пустому `git stash list` и `git diff --check` отслеживаемые изменения возвращены, пользовательские untracked-файлы сохранены. Commit, push и merge не выполнялись.

## Продолжение восстановления и интеграционный smoke

После сброса лимита Kimi восстановление продолжено. Gate проверяет три baseline seed 42/43/44, majority, fingerprints и согласованность checkpoint и метрик; сравнительный план может использовать проверенное подмножество seed. Исправлены отказ при недостаточном бюджете ветви и сохранение hash predictions для отчёта. **Resume намеренно отключён**: повторное использование рейтингов, отбора, refit и успешных run results без полного доказательства provenance отклоняется; существующие файлы не перезаписываются. Отчёт включает только проверенные успешные predictions, обрабатывает null и пустые confusion pairs, включает baseline в сводку и восстанавливает HTML из таблиц.

Проверка: `.venv/bin/python -m pytest -q` — **421 passed, 1 skipped**, предупреждение об инициализации CUDA в sandbox. Новые проверки gate: 24 passed, seed-подмножества: 13 passed; сохранены тесты повреждения predictions и split. `.venv/bin/python scripts/s06_comparison_smoke.py` — **SMOKE PASS** в `outputs/S06_recovery_smoke_3/`: синтетическая фикстура 120 строк, 10 L3, CPU; полный TF-IDF 300 char + 100 word. Протокол выбран по train/monitor, 18 эпох, план зафиксирован до оценки; три микро-baseline gate и 10 сравнительных запусков с seed 42. `none`, `global`, `class_aware`, `gradient_x_input` с K=60/120/240 — все 10 `success`. Девять выбранных TF-IDF переобучены на том же snapshot; проверены парные Δ, сохранение и загрузка результатов. Повторные этапы явно отказали при отключённом resume, а SHA-256 всех файлов `runs/` и `shared/` до/после совпали. Восемь файлов отчёта — в `outputs/S06_recovery_smoke_3/report/`; HTML пересоздан из таблиц без доступа к `runs/` и `shared/` в `outputs/S06_recovery_smoke_3/rerender_isolated/report_rerender/`. `git diff --check` прошёл.

Первые три smoke-попытки сохранены: `outputs/S06_recovery_smoke` остановилась на несовместимости seed, `_1` и `_2` — на majority gate; `_3` прошла после исправления проверки seed и увеличения числа шагов обучения синтетической fixture. Эти изменения fixture и hyperparameters предназначены исключительно для проверки механики, не являются оценкой отбора на ODS. Sealed test не использовался для inference или метрик. Исторический baseline на ODS остаётся отделённым: план был зафиксирован после его оценки; для нового формального сравнения потребуется свежий baseline с планом до оценки. 30 итоговых runs не запускались. Последние исправления после второго ревью независимо не ревьюированы; полноразмерный запуск и ресурсные требования refit не проверены.

## Final comparative experiment — 2026-11-07

This section supersedes the earlier D01–D14 status table; that table and the earlier baseline remain a historical record. The comparison validation set had already been examined during prior S06 work. The complete new plan was frozen before the first validation inference **in this experiment**, not before all prior exposure to the set. No conclusion here establishes industrial performance.

The final experiment is `outputs/S06_final_cpu_20261107/`. Frozen split manifest SHA-256 `ff44250867e864af23c72da94c38ed4d5a2796af9a658b83787193432ee2cd5d`; protocol SHA-256 `caffae8c2609d9f9cff28c47cf368cedddf7c38e7c4a4c8801bb553802c5cb0d`; 30-run plan SHA-256 `69bc5489137e6b0da8296b4461eb8b9e8b9886756a70ee46e936e59d12eb960f`. The retained dataset has 16,289 rows, 293 L3 classes, and split sizes 12,960 outer_train / 1,692 comparison_val / 1,637 sealed_test, including 11,600 train_fit / 1,360 train_monitor. The original group-aware quota audit, minimum supports and split fingerprints were reverified. The sealed test was used only for structural validation, not inference or quality metrics. The full S02 TF-IDF has 300,000 char + 100,000 word features, SHA-256 `7ff42885729d0bcfb2b6e9e0893b1be4ddf59562705e3ed048614183cd952052`; the 981,850-row corpus snapshot has SHA-256 `f9a057a00db10ed115dd5f47e035f3a3c56c710d9463f47d38812d4cceef3dbc`. Two matching ODS texts in the unlabeled corpus make the feature fit partially transductive.

A train_fit/train_monitor procedure selected 14 epochs before validation; the fixed plan specifies the same schedule and seeds 42/43/44 throughout. All three new full baselines trained successfully, beat majority and passed Gate 1; their validation macro-F1 mean is 0.722928. G×I used the new seed-42 checkpoint, chosen independently of validation. All 30 final runs succeeded: 3 fresh full baselines and 27 fresh selected models. There were no failed, pending or skipped runs. Every fixed-vocabulary refit used the original corpus snapshot, preserved the 75/25 branch allocation and achieved zero maximum IDF discrepancy against corresponding full features; maximum norm discrepancy was about 2.38e-7, and all nine artifacts reloaded. At K=200,000 the class-aware word branch used 29,421 fallback features and G×I used 29,439 remainder features; hence not every selected feature in those cases has positive class-aware evidence or positive attribution.

| Method | Features | Macro-F1 mean | Std (population) | Paired Δ vs full, mean |
|---|---:|---:|---:|---:|
| Full TF-IDF | 400,000 | 0.722928 | 0.001938 | 0 |
| Global Chi² | 50,000 | 0.717083 | 0.005652 | -0.005844 |
| Global Chi² | 100,000 | 0.722231 | 0.002482 | -0.000697 |
| Global Chi² | 200,000 | 0.719005 | 0.006326 | -0.003923 |
| Class-aware Chi² | 50,000 | 0.722022 | 0.003697 | -0.000905 |
| Class-aware Chi² | 100,000 | 0.736833 | 0.003529 | +0.013905 |
| Class-aware Chi² | 200,000 | 0.717978 | 0.007024 | -0.004949 |
| Gradient × Input | 50,000 | 0.727261 | 0.002738 | +0.004333 |
| Gradient × Input | 100,000 | 0.729647 | 0.000910 | +0.006719 |
| Gradient × Input | 200,000 | 0.719289 | 0.004704 | -0.003638 |

The class-aware 100k configuration had the largest observed mean gain on this split; small differences and three seeds on one previously examined validation set do not establish a general winner. Some L3 classes have only two validation examples, limiting per-class conclusions. The saved predictions independently reproduce all 30 F1 and accuracy values exactly; paired deltas agree within 1e-16. `report_modelpt_bytes/` contains the eight required files, including per-class, support-bin and confusion diagnostics. A separate directory `/tmp/s06-final-tables-20261107/` containing only copies of the corrected saved report tables was used to regenerate HTML without reading models, `runs/`, `shared/`, training or inference.

CPU-only timings from `stages.csv`: three fresh full baseline training stages total 213.925 s; 27 selected training stages total 608.342 s; nine shared TF-IDF refits total 343.790 s; two Chi² ranking stages total 8.846 s and G×I attribution 0.702 s. Comparison-val inference stages total 5.529 s, including the separate Gate 1 inferences. Stages have their own measurement scopes and are not necessarily additive end-to-end wall time. Maximum recorded stage peak RSS was 3,358,146,560 bytes; serialized `model.pt` sizes were 51,578,225–409,978,289 bytes. VRAM is null because CUDA initialization failed in this environment; `outputs/S06_final_20261107/` preserves the earlier unsuccessful GPU setup with no comparison-validation inference. CPU was used for the final experiment without changing the frozen model methodology.


Executed final-experiment commands (all output roots new; existing results preserved):

```text
.venv/bin/python scripts/run_s06_baseline.py verify-split --output-root outputs/S06_final_20261107 --splits-manifest outputs/S06/splits/manifest.json
.venv/bin/python scripts/run_s06_comparison.py plan --output-root outputs/S06_final_cpu_20261107 --splits-manifest outputs/S06/splits/manifest.json --device cpu
.venv/bin/python scripts/run_s06_baseline.py refit --output-root outputs/S06_final_cpu_20261107 --splits-manifest outputs/S06/splits/manifest.json --device cpu
.venv/bin/python scripts/run_s06_comparison.py rankings --output-root outputs/S06_final_cpu_20261107 --splits-manifest outputs/S06/splits/manifest.json --device cpu
.venv/bin/python scripts/run_s06_comparison.py refit --output-root outputs/S06_final_cpu_20261107 --splits-manifest outputs/S06/splits/manifest.json --device cpu
.venv/bin/python scripts/run_s06_comparison.py train --output-root outputs/S06_final_cpu_20261107 --splits-manifest outputs/S06/splits/manifest.json --device cpu
.venv/bin/python scripts/run_s06_report.py --output outputs/S06_final_cpu_20261107
.venv/bin/python scripts/run_s06_report.py --output /tmp/s06-report-rerender-bKPxw5 --report-dir /tmp/s06-report-rerender-bKPxw5/rerender
.venv/bin/python -m pytest -q tests/test_s06_report_hash_independent.py tests/test_s06_external_split_report_independent.py
```

The report's external-split resolution was fixed after the two allocated independent reviews: it now validates frozen protocol and manifest fingerprints, validation split hash/size/rows and label-map contract. Its regression tests passed (20 tests); these final fixes have not received a third independent review. Fresh-run isolation is mandatory and unsafe partial resume remains disabled.

Final report size correction: `outputs/S06_final_cpu_20261107/report_modelpt_bytes/` preserves the original report and records verified `model.pt` bytes for all 30 comparison runs. Missing or modified checkpoints cause a fail-closed error. These changes were made after the two independent reviews and are not independently reviewed. The final full suite after this change passed: `436 passed, 1 skipped` (CUDA unavailable); the final fast suite passed `435 passed, 1 skipped, 1 deselected`. The final full suite included the added size regression tests.

```text
.venv/bin/python scripts/run_s06_report.py --output outputs/S06_final_cpu_20261107 --report-dir outputs/S06_final_cpu_20261107/report_modelpt_bytes
.venv/bin/python scripts/run_s06_report.py --output /tmp/s06-final-tables-20261107 --report-dir /tmp/s06-final-tables-20261107/rebuilt
.venv/bin/python -m pytest -q -rs
.venv/bin/python -m pytest -q -m 'not slow'
git diff --check
```

Acceptance verification: `git diff --check` passed. The stricter staged check flags eight intentional Markdown hard-break spaces in the immutable original `value_dod.md` and the exact saved review prompt containing its contents; these files are preserved verbatim. `git -c core.whitespace=-blank-at-eol diff --cached --check` passed for the staged content. No datasets, models, outputs or unrelated user files were staged.
