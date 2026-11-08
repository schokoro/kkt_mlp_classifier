# kkt_mlp_classifier

Классификатор наименований позиций кассовых чеков по таксономии L1 → L2 → L3. Итерация 1 — baseline:
TF-IDF (char n-gram + word unigram) → MLP с общим стволом и тремя головами → отказ по порогу τ.
Требования — `artefacts/SPEC_v1.0.md`, решения — `artefacts/DECISION_LOG.md`, состояние — `artefacts/STATUS.md`.

## Установка

Целевая среда (Python 3.12, GPU): torch с CUDA — из внутреннего зеркала, затем
```bash
pip install -r requirements.txt          # рантайм
pip install -r requirements-dev.txt      # + pytest, nbconvert, ipykernel
```
Облачная сессия / локально на CPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt
# или всё сразу: scripts/setup_cloud.sh  (venv .venv на python3.12)
```
Код не ходит в сеть и не использует предобученных весов: в целевую среду переносятся только файлы репозитория.

## Запуск

```bash
python scripts/make_fixtures.py          # fixtures/*.parquet из data/ods_labels.csv
```
1. **Обучение** — `notebooks/train.ipynb`, Run All. В конфиг-ячейке: `PRESET` (`default` — под ~6 млн строк,
   `ods_light` — ODS) и `OVERRIDES` (путь к parquet в `data.path`, каталог выхода `output_dir`, всё остальное).
   Итоговый конфиг печатается и сохраняется в артефакт.
2. **Инференс** — `notebooks/inference.ipynb`, Run All. В конфиг-ячейке: путь к артефакту, вход, выход,
   batch size, переопределение τ и правил отказа, `top_k`.

Без ноутбуков, теми же вызовами:
```bash
python scripts/run_train.py --preset ods_light \
  --override '{"data": {"path": "fixtures/ods.parquet"}, "output_dir": "outputs/train_ods"}'
python scripts/run_inference.py --override '{"artifact_dir": "outputs/train_ods/artifact",
  "input_path": "fixtures/infer_unknown_labels.parquet", "output_path": "outputs/inference_ods/predictions.parquet",
  "report_dir": "outputs/inference_ods/reports"}'
```
Тесты: `pytest -q` (включая исполнение обоих ноутбуков через nbconvert), быстро без ноутбуков: `pytest -q -m "not slow"`.

## Конфиг

`kkt_clf/config.py`, `RunConfig` (обучение) и `InferenceConfig` (инференс). Разделы `RunConfig`:

| раздел | параметры |
|---|---|
| верх | `output_dir`, `seed` |
| `data` | `path`, `max_rows` (None — всё, иначе стратифицированный по L3 сэмпл), `min_count`, `split` (train/val/test) |
| `features.char` / `features.word` | `ngram_range`, `min_df`, `max_features`, `sublinear_tf` (dtype всегда float32, `lowercase=False`) |
| `features` | `vectorizer_fit_rows` (подвыборка train для fit), `transform_chunk_rows` |
| `model` | `hidden` (список размеров), `dropout` (той же длины), `activation` (relu/gelu/silu/tanh), `norm` (None/layernorm/batchnorm) |
| `loss` | `head_weights` {l1,l2,l3}, `class_weight_mode` {l1,l2,l3}: `none` \| `inv_freq`, `class_weight_alpha` |
| `train` | `optimizer` (adamw/adam/sgd), `lr`, `weight_decay`, `batch_size`, `eval_batch_size`, `epochs`, `early_stopping_metric` (l3_macro_f1/l3_accuracy/val_loss), `patience`, `device` (auto/cpu/cuda), `num_threads` |
| `train` (LR scheduler) | `scheduler`: `plateau` (по умолчанию) \| `cosine` \| `onecycle` \| `none`; `scheduler_eval_every_steps`, `scheduler_val_rows`, `plateau_factor`, `plateau_patience` (в проверках), `plateau_threshold`, `plateau_cooldown`, `min_lr`, `warmup_frac` (cosine) |
| `reject` | `tau_mode` (`fixed` \| `target_precision`), `tau`, `target_precision`, `tau_grid`, `reject_on_mismatch_l2`, `reject_on_mismatch_l1` |
| `report` | `top_n_errors`, `noisy_support` |

Архитектура меняется только конфигом: например `"model": {"hidden": [1024, 512, 256], "dropout": [0.3, 0.3, 0.1], "norm": "layernorm"}`.
Батч-нормализация — `"norm": "batchnorm"` (после каждого скрытого слоя, включая разреженный первый), по умолчанию выключена.

**LR scheduler.** `plateau` — ReduceLROnPlateau, но не по эпохам: каждые `scheduler_eval_every_steps` шагов считается val loss на фиксированной подвыборке val (`scheduler_val_rows`), и при застое на `plateau_patience` проверок lr умножается на `plateau_factor` (не ниже `min_lr`). Дефолт под 6M — проверка каждые 250 шагов (~9 раз за эпоху) на 50k строк val. `cosine` — линейный разогрев (`warmup_frac`) и косинус до `min_lr`, `onecycle` — OneCycleLR с пиком `lr`. Трасса lr по шагам — `reports/lr_trace.csv`, график — в обучающем ноутбуке. Early stopping остаётся по эпохам.

## Выходы

```
<output_dir>/
  artifact/
    model.pt            state_dict
    model_spec.json     n_features, число классов, архитектура
    config.json         полный конфиг прогона
    features.joblib     предобработка + оба TF-IDF векторизатора
    label_maps.json     label↔id для L1/L2/L3, parent_l2_of_l3, parent_l1_of_l2
    reject.json         τ, правила отказа, как выбран τ
    class_weights.json  веса CE и частоты классов в train
    meta.json           версии Python/torch/sklearn/…, seed, дата, размеры выборок, sha256 входа, тайминги
  splits/{train,val,test}.parquet   сплиты по содержимому (name, L1..L3)
  reports/
    test__report__<variant>__<level>__<source>.csv   classification_report (с support)
    test__summary.csv, test__top_errors.csv, test__mismatch_rates.csv, test__f1_vs_freq.csv
    history.csv, lr_trace.csv, tau_table_val.csv, param_counts.csv, rare_l3_classes.csv, duplicate_examples.csv
    run_report.json
```
Загрузка: `Predictor.load(artifact_dir).predict(names, top_k=3)` — предобработка, векторизация, модель, отказ.

Выход инференса: входные колонки + `l3_pred` (null при отказе), `l3_prob`, `rejected`, `reject_reason`
(`empty_features` / `low_confidence` / `mismatch` / null), `l3_top1_raw`, `l2_from_l3`, `l1_from_l3`, `l2_head`,
`l2_head_prob`, `l1_head`, `l1_head_prob`, `mismatch_l2`, `mismatch_l1`, `l3_top{i}`, `l3_top{i}_prob`.

## Память и время

**Замер** (облачная сессия, 4 vCPU, без GPU, Python 3.12, torch 2.14 CPU-режим):

| что | значение |
|---|---|
| ODS, пресет `ods_light`, `hidden=[256]` (18M параметров): весь прогон | ~37 с, пик RSS 1.3 ГБ, эпоха ~1.8 с (13.3k строк train) |
| ODS, `hidden=[512,256]` + layernorm (36M) | ~43 с, пик RSS 1.7 ГБ, эпоха ~3 с |
| векторизация с дефолтами 6M (char 2–5, 300k + word) | fit ~3.7k строк/с, transform ~8k строк/с (один поток sklearn) |
| nnz на строку (дефолты) | ~120 → ~950 Б/строку в CSR (float32 + int32) |
| первый слой дефолтной модели | 400k × 512 = 204.8M параметров, всего ~205.5M |

**Экстраполяция на ~6 млн строк, пресет `default`** (оценка, не замер):

| этап | RAM / VRAM | время |
|---|---|---|
| чтение parquet + проверки | ~1.5–2 ГБ (строки в pandas) | 1–2 мин |
| fit векторизаторов на 1.5M (`vectorizer_fit_rows`) | пик ~5–7 ГБ (словарь n-грамм до отсечения) | ~7 мин |
| transform 6M | CSR train ≈ 4.6 ГБ, val+test ≈ 1.1 ГБ; транзиентно ×2 при склейке | ~12–13 мин |
| обучение на GPU | VRAM ≈ 3.3 ГБ (веса + градиент + 2 момента AdamW) + активации; RAM: +0.8 ГБ на копию лучших весов | ~1–2 мин/эпоха, 10 эпох ≈ 10–20 мин |
| итого пик RAM | ~12–14 ГБ из 24 | ~35–50 мин |

Если не хватает памяти, крутить по порядку:
1. `features.vectorizer_fit_rows` (пик при fit) — 0.5–1M;
2. `features.char.ngram_range` → `(2, 4)` (≈ −25% nnz, т. е. и CSR) и `features.char.max_features`;
3. `features.transform_chunk_rows` (пик при transform);
4. `data.max_rows` — стратифицированный сэмпл;
5. VRAM: `model.hidden[0]` и `max_features` (первый слой = `n_features × hidden[0]`), `train.batch_size`, `train.eval_batch_size`.

Векторизация однопоточная — это главный кандидат на ускорение в следующих итерациях.

## S01: отдельный fit TF-IDF

По умолчанию `features.mode="train"`: обе ветки обучаются на размеченном train после split, с прежним ограничением `vectorizer_fit_rows`. Для внешнего пространства сначала подготовьте ограниченный Parquet с колонкой `name` и идентификатором версии:

```bash
.venv/bin/python scripts/fit_vectorizer.py --input sample.parquet --output outputs/tfidf_v1 \
  --features-preset ods_light --max-rows 100000 --seed 42 --source-id sample-v1
.venv/bin/python scripts/run_train.py --preset ods_light --override '{"data":{"path":"labeled.parquet"},"output_dir":"outputs/run_v1","features":{"mode":"pretrained","pretrained_path":"outputs/tfidf_v1"}}'
```

Для следующего обучения задайте новый `output_dir` и тот же `pretrained_path`. Загруженный FeatureBuilder определяет конфигурацию признаков; отличия настроек признаков текущего прогона сопровождаются предупреждением. Повторный fit не выполняется. Модельный артефакт содержит собственный `features.joblib`, поэтому исходный автономный каталог для инференса не нужен. Автономный каталог содержит `features.joblib` и `vectorizer_meta.json` с конфигурацией, происхождением, размерами словарей, версиями и контрольной суммой; при загрузке выполняются проверки. Не загружайте joblib из недоверенных источников.

Берутся первые не более `max_rows` непустых имён в порядке файла; повторы сохраняются. Выборка зависит от версии и порядка подготовленного файла, а seed применяется к дополнительной подвыборке `vectorizer_fit_rows`. Метаданные различают число непустых строк источника и фактическое число строк fit. Чтение выполняется батчами только колонки `name`; fit sklearn не является потоковым и хранит ограниченные тексты и промежуточный словарь в памяти. SHA-256 источника требует отдельного полного чтения файла. Укажите предел, исходя из доступной RAM: обработка 300 млн строк не проверялась. Существующий непустой каталог автономного артефакта отклоняется без `--force`; используйте новый путь. Обучение отклоняет непустой `output_dir` и его пересечение с каталогом внешнего векторизатора: выбирайте отдельный новый каталог каждого прогона.

Внешний корпус может содержать тексты val/test: при совпадениях режим трансдуктивен и не измеряет обобщение на полностью невиданный корпус. Метки внешнего Parquet не читаются. Малые фикстуры и ODS проверяют механику, а не качество будущей модели.

## S03: supervised feature selection и сравнение

`run_s03_experiment.py` использует автономный полный TF-IDF и отдельный корпус из его метаданных. Конфигурации `none`, `global` и `class_aware` обучаются на общих train/validation и сравниваются по raw top-1 L3 на validation. Пример облегчённого CPU-прогона (каталог выхода должен быть новым):

```bash
.venv/bin/python scripts/run_s03_experiment.py run --output outputs/S03 \
  --full-tfidf outputs/S02/tfidf_1m --corpus data/corpus.pqt \
  --data outputs/S02/fixtures/ods_small.parquet --preset ods_light \
  --override '{"model":{"hidden":[32],"dropout":[0.1]},"train":{"epochs":1,"device":"cpu","scheduler":"none","num_threads":2}}'
.venv/bin/python scripts/run_s03_experiment.py report --output outputs/S03
```

Бюджеты, доля char/word, seed и настройки модели задаются CLI; справка — `run_s03_experiment.py run --help`. Для выбранных признаков создаются новые фиксированные словари, обученные на проверенной исходной подвыборке корпуса; IDF пересчитывается. Отбор использует train-метки, а не метки validation/test. Каталог отчёта содержит `report.html`, `runs.csv`, `per_class.parquet`, `stages.csv`, `manifest.json`; все запланированные прогоны и их статусы учитываются. Время и RSS этапов следует интерпретировать по `measurement_note`, пиковые значения не суммировать. Дисковые рейтинги могут занимать значительное место; версия артефакта выбора `@3` несовместима со старыми версиями. Малые ODS-прогоны проверяют механику, а не качество или масштабы промышленной системы.

## S04: предварительная реализация Gradient × Input

S04 добавляет экспериментальный метод `gradient_x_input`: полный TF-IDF → attribution MLP → class-aware отбор → повторный fit выбранных словарей → новая MLP. Справка: `.venv/bin/python scripts/run_s04_experiment.py run --help`; отчёт из сохранённых результатов: `.venv/bin/python scripts/run_s04_experiment.py report --output <каталог>`. Пути входов и выхода, бюджеты, доля char, выбор train-строк и checkpoint настраиваются. Новые каталоги не должны содержать прежние результаты.

Общий CPU smoke выполнен; последние исправления внесены после второго и последнего независимого ревью и не прошли повторное независимое ревью. CUDA, полномасштабный корпус и диагностическое сравнение не проверены. Команды, отчёт и ограничения: `artefacts/S04/result.md`.

Повторный отбор G×I требует полный каталог `shared/attribution/ratings`: перед использованием проверяются обе ветки и все объявленные файлы. Новые манифесты содержат размеры и SHA-256 файлов; старые манифесты без хэшей проверяются структурно. Для переноса рейтингов одного HTML-отчёта или `meta.json` недостаточно.
# S05: аудит и baseline на полном TF-IDF

S05 использует замороженные group-aware сплиты и уже обученный внешний TF-IDF;
качество на sealed test не вычисляется. Выбор числа эпох происходит только на
`train_fit`/`train_monitor`, затем модели переобучаются на всём outer train по
фиксированному расписанию. Сравнение выполняется на `comparison_val`; результаты
ODS проверяют механику PoC и не оценивают промышленное качество.

```bash
.venv/bin/python scripts/audit_ods.py
.venv/bin/python scripts/freeze_split.py
.venv/bin/python scripts/run_s05_baseline.py select --device cuda
.venv/bin/python scripts/run_s05_baseline.py refit --device cuda --seeds 42,43,44 --reference-seed 42
.venv/bin/python scripts/run_s05_report.py --output outputs/S05
.venv/bin/python scripts/run_s05_report.py --output outputs/S05/report --report-dir /tmp/s05-render
```

Команды записи требуют новых целевых файлов; пути, устройство, seed и
конфигурация доступны через `--help`. При отсутствии CUDA задайте `--device cpu`.
В имеющемся S05 первоначальные `split_audit.csv` и `class_support_split.csv`
содержат выявленные ошибки счётчиков; проверенные версии имеют суффикс
`_corrected.csv`. Подробности, источники и ограничения — в `artefacts/S05/result.md`.
При закрытии исправленный отчёт сохранён отдельно в `outputs/S05/report_closed/`;
предыдущая версия сохранена, таблица частых L3-путаниц отсутствует из-за
отсутствия сохранённых предсказаний.

## S06: сравнительный эксперимент (принят с оговорками)

Split использует порог L3 support ≥12, отдельный внутренний monitor и
закрытый test. Сравнительный эксперимент завершён: 30 из 30 запусков успешны;
отчёт — `outputs/S06_final_cpu_20261107/report_modelpt_bytes/`. Команды аудита и baseline:

```bash
.venv/bin/python scripts/freeze_split_s06.py --help
.venv/bin/python scripts/run_s06_baseline.py select --device cuda
.venv/bin/python scripts/run_s06_baseline.py refit --device cuda
```

Для новых запусков выбирайте отдельные каталоги, чтобы не перезаписать данные;
при отсутствии GPU используйте `--device cpu`. Состояние, замечания ревью и
ограничения эксперимента: `artefacts/S06/result.md`. Результаты ODS не являются
оценкой качества промышленной системы.

## S07: repeated development splits

S07 compares full TF-IDF with three 100k selections on development splits
101/202/303 and model seeds 42/43/44 (36 runs). It reuses the verified S02
features/corpus and final S06 14-epoch training protocol; S06 sealed test stays
excluded. These ODS resamples are not an independent external holdout.

```bash
.venv/bin/python scripts/run_s07.py preflight --output-root outputs/S07/preflight_NEW_ID
.venv/bin/python scripts/run_s07.py plan --output-root outputs/S07/study_NEW_ID --device cpu --num-threads 2
.venv/bin/python scripts/run_s07.py execute --output-root outputs/S07/study_NEW_ID
.venv/bin/python scripts/run_s07.py report --output-root outputs/S07/study_NEW_ID
.venv/bin/python scripts/run_s07.py rebuild --report-dir outputs/S07/study_NEW_ID/report --output-html /tmp/s07_NEW_ID_rebuilt.html
```

Use fresh directories. Execution is single-use; interrupted runs require a new
plan/root. Source, split, dependency, runtime and implementation fingerprints
are checked before use. `report/` contains HTML, manifest and eight CSV/parquet
tables, including paired effects, feature stability and stage costs. Copy only
the manifest and eight tables to an isolated directory to rebuild HTML there;
models, source data, `runs/` and `shared/` are unnecessary. Actual study status,
commands, review limitations and results are recorded in `artefacts/S07/result.md`.


S07 study output: `outputs/S07/stability_20261107_v2/report/report.html`
(36/36 success, nine selected refits). Class-aware100k paired mean macro-F1 gain
is only +0.000873 and positive in one of three split means; S06's +0.0139 is not
a stable repeated-split result. The 100k model is ~4x smaller; further independent
testing is warranted for resource savings. S07 is accepted with documented limitations; the post-review
serialization fix remains independently unreviewed. Full results and limitations:
`artefacts/S07/result.md`.

### S08: accepted non-inferiority study (holdout consumed)

S08 is **accepted with documented limitations**. The one-time original S06 holdout
assessment completed nine frozen models: Full400k mean L3 macro-F1 **0.730355**,
Class-aware100k **0.731956**, Global100k exploratory **0.721845**. Approved
`delta_NI=0.01`, `alpha=0.025` and the one-sided97.5% lower bound yield primary
paired Delta **+0.00160175**, lower bound **-0.00984237 > -0.01**:
**non-inferiority established**, with only **0.00015763** margin clearance.
Superiority was not established. The original sealed test is now consumed;
do not reuse it for adaptive model selection or repeat its inference.

Final report: `outputs/S08/noninferiority_20261107_v1/report_evaluated_20261108_v1/`.
Results, accepted limitations, authorization and independent table-only verification:
`artefacts/S08/result.md`, `evaluation_authorization_20261108.md` and
`evaluation_verification.md` in the same directory. Sparse L3 support, previously
examined development validation, three fixed seeds rather than independent datasets,
unrepresentative ODS data and CPU-only measurements preclude production quality or
scalability guarantees. Future adaptive confirmation requires new holdout labels.

The following commands document the preparation/evaluation interfaces; they do not
authorize further experiments or reuse of the consumed holdout.

S08 preserves the S06 source pins, S02 full TF-IDF/corpus, 293 L3 and the
S06/S07 14-epoch architecture/optimizer contract. Final training uses exactly
`outer_train + comparison_val` (14,652 development rows), methods `none`,
`class_aware`, `global`, seeds `42/43/44`. Each selected dictionary is independently
ranked on development labels and refitted on the identical original unlabeled
corpus sample. Training diagnostics are explicitly in-sample; no validation or
test scores are emitted. Fresh directories and single-use execution are required.

```bash
.venv/bin/python scripts/run_s08.py preflight --output-root outputs/S08/<fresh-preflight-id>
.venv/bin/python scripts/run_s08.py plan --output-root outputs/S08/<fresh-experiment-id> --device cpu --num-threads 2
# Execute only after independent tests and review; nine final training runs:
.venv/bin/python scripts/run_s08.py execute --output-root outputs/S08/<fresh-experiment-id>
.venv/bin/python scripts/run_s08.py report --output-root outputs/S08/<fresh-experiment-id>
.venv/bin/python scripts/run_s08.py rebuild --report-dir outputs/S08/<fresh-experiment-id>/report --output-html /tmp/s08-rebuilt.html
```

`protocol.json`, `plan.json`, `pre_registration.md`, and `feasibility.json` freeze
inputs, training settings and decision rules before execution. A completed
`training_freeze.json` recursively binds weights, label maps, feature vocabularies,
IDF, training receipts and prediction interface. Source/code/runtime changes or
resume attempts fail closed. `report` can also generate an honest pending report
before training; use a fresh `--report-dir` for each later snapshot. Exit zero
means successful report construction, not completion of training or an NI claim.
HTML rebuild needs only `manifest.json`, `runs.csv`, `stages.csv`, and
`ni_decision.json` copied into one directory; it never loads models or data.

A new unsigned preparation plan keeps primary `class_aware` and illustrative
margin `.01` **UNAPPROVED**, with `delta_NI` null until a separate actual decision.
The completed S08 assessment instead has the explicit detached approval described above. Feasibility numerically explores explicitly
simulated categorical prediction scenarios with development-derived support
composition; these are not holdout predictions or performance estimates. The
row-level paired bootstrap recomputes fixed-universe macro-F1, preserves each
class support and uses identical sampled rows across all three seed pairs:
B=5000, PCG64 seed 808, alpha=.025, linear percentile lower bound, zero_division=0,
strict `lower_bound > -delta_NI`. Non-singleton leakage groups fail closed.

The evaluation implementation was available **before formal freeze** and the
completed S08 run used actual user consent. For any separately authorized study, a detached
JSON decision must bind an explicit margin, practical rationale, owner, actual
consent evidence, primary `class_aware`, one-time S06 use, and exact protocol/plan/
training-freeze/source/model/maps/features identities. It does not modify the
unsigned frozen protocol or supply a default margin. Global stays descriptive;
GxI is forbidden.

```bash
# ONLY the orchestrator following actual user consent may supply the trusted digest:
.venv/bin/python scripts/run_s08.py evaluate --output-root <frozen-root> \
  --approval-record <detached-decision.json> \
  --trusted-approval-sha256 <independently-trusted-raw-file-sha256> \
  --report-dir <fresh-versioned-report-directory>
```

Python: `kkt_clf.s08.evaluate(root, approval_record_path, trusted_anchor,
report_dir=fresh_path)`. `kkt_clf.s08_evaluation.approval_bindings(root)` returns
only an identity template; it cannot grant consent. Exact decision schema is in
`artefacts/S08/review_fixes_handoff.md`. The anchor is separate **trusted caller
input**, supplied by the orchestrator only after actual user consent. Never
compute that argument automatically from an untrusted record: SHA256 integrity
alone is not consent. No fictitious signer, external issuer or service is assumed.
The local single-use claim assumes trusted local storage/operator; deliberate
ledger deletion/rollback by the account owner is outside this boundary.

All nine checkpoints/features are verified before an atomic `O_EXCL` claim in
`evaluation_claim.json`. Any partial/failed/interrupted evaluation consumes use
permanently; no retry/resume. Inference uses verified checkpoint byte snapshots,
with live freeze revalidation before each read/inference. Row predictions include
truth/method/seed/model hash, and are re-read and rowID-aligned before statistics.
Evaluated/failed reports go to fresh versioned paths. Offline `rebuild` uses only
manifest-listed tables, the detached decision and NI decision, with no model or
source access. Interrupted stages may have a claim and partial report; that is
consumed and must never be rerun. Pending reports retain strict nulls and omit all
five holdout-dependent files. This remains an internal ODS PoC, not industrial
validation or actual authorization.

For independent mechanics checks, Python `S08Config(fixture=True, fixture_pins=...)`
accepts explicitly pinned synthetic sources and tiny branch budgets while keeping
the 75/25 allocation. Fixture manifests must declare all five original parts,
`n_rows`, `n_groups`, hashes and byte sizes, with inner partitions covering
outer_train exactly. Fixture epochs/universe may be small; production pins cannot
be overridden. Fixture provenance is always marked synthetic. Synthetic decisions may exercise
authorized mechanics on generated data only and cannot authorize production use. Public API details and implementation limits are in
`artefacts/S08/implementation_handoff.md`.
