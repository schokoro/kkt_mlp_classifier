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
| `reject` | `tau_mode` (`fixed` \| `target_precision`), `tau`, `target_precision`, `tau_grid`, `reject_on_mismatch_l2`, `reject_on_mismatch_l1` |
| `report` | `top_n_errors`, `noisy_support` |

Архитектура меняется только конфигом: например `"model": {"hidden": [1024, 512, 256], "dropout": [0.3, 0.3, 0.1], "norm": "layernorm"}`.

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
    history.csv, tau_table_val.csv, param_counts.csv, rare_l3_classes.csv, duplicate_examples.csv
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
