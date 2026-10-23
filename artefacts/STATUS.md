# STATUS

Обновлено: 2026-10-23. Итерация 1 (baseline) реализована; добавлен LR scheduler (по умолчанию plateau по шагам), BatchNorm — опция `model.norm`. Итоговая ветка: **`master`** (рабочая — `claude/ecstatic-bardeen-yd2ggd`, то же состояние).

## DoD (§9)

| # | пункт | статус | комментарий |
|---|---|---|---|
| 1 | Обучающий ноутбук на ODS, Run All, CPU | done | `notebooks/train.ipynb` → `outputs/train_ods/{artifact,reports,splits}`, ~40 с |
| 2 | Ноутбук инференса на фикстуре | done | `notebooks/inference.ipynb` на `fixtures/infer_unknown_labels.parquet`, все колонки §4.9 + top-3 |
| 3 | Тесты §6 зелёные | done | `pytest -q`: 38 passed (включая исполнение ноутбуков) |
| 4 | Архитектура меняется только конфигом, ≥2 конфигурации | done | ODS: `[256]` и `[512,256]`+layernorm (ниже); в тестах `[64]`, `[96,48]`+layernorm+onecycle, `[32,32,16]`+batchnorm+cosine, `[64]`+batchnorm+plateau |
| 5 | Ноутбуки — фасад, пайплайн из скрипта | done | тест AST без def/class/lambda; `scripts/run_train.py`, `scripts/run_inference.py` |
| 6 | Без сети и предобученных весов, перенос копированием | done | зависимости — `requirements.txt` (диапазоны под 3.12); сетевых вызовов в коде нет |
| 7 | README | done | запуск, конфиг, структура артефакта, память/время |
| 8 | STATUS и DECISION_LOG | done | D-001…D-018, включая [Г] `min_count` (D-009) |
| 9 | Итог в одной ветке | done | `master` |

## Метрики на ODS (проверка работоспособности, не цель)

Данные: 18 046 строк → после `min_count=10` 16 635 строк, 326 L3 / 72 L2 / 3 L1 (411 редких L3 исключены); test 1 664.
τ = 0.5 (fixed), калибровки нет.

| конфиг | L3 acc / macro-F1 без отказа | L2 из L3 acc | с отказом: L3 acc / macro-F1 | покрытие |
|---|---|---|---|---|
| `ods_light`, `hidden=[256]` | 0.804 / 0.712 | 0.856 | 0.936 / 0.873 | 0.693 |
| `hidden=[512,256]`, dropout 0.3/0.2, layernorm | 0.814 / 0.732 | 0.867 | 0.883 / 0.812 | 0.877 |

Наблюдения для следующих итераций: головы L1/L2 сами по себе слабее пути от L3 (L2 head 0.838 vs from_l3 0.856); без калибровки покрытие при одном τ сильно зависит от архитектуры.

### LR scheduler и BatchNorm (ODS, `ods_light`)

| конфиг | эпох (лучшая) | L3 acc / macro-F1 без отказа | с отказом acc / macro-F1 | покрытие |
|---|---|---|---|---|
| `[256]`, scheduler `none` | 15 (12) | 0.804 / 0.712 | 0.936 / 0.873 | 0.693 |
| `[256]`, `plateau` (дефолт) | 15 (12) | 0.804 / 0.712 | 0.936 / 0.873 | 0.693 |
| `[256]`, `cosine` | 15 (13) | 0.787 / 0.637 | 0.951 / 0.889 | 0.599 |
| `[256]` + batchnorm, `plateau` | 11 (8) | 0.811 / 0.720 | 0.936 / 0.885 | 0.728 |
| `[512,256]`+layernorm, `none` | 11 (8) | 0.814 / 0.732 | 0.883 / 0.812 | 0.877 |
| `[512,256]`+layernorm, `plateau` | 10 (7) | 0.813 / 0.719 | 0.890 / 0.826 | 0.862 |

На `[256]` val loss падает все 15 эпох — plateau не срабатывает, результат совпадает с `none`. На `[512,256]` plateau снижает lr с 6-й эпохи (шаг 290 из ~52/эпоху), каждые 40 шагов до 1.6e-5; качество в пределах шума ODS. Ощутимого выигрыша на ODS нет — эффект ждём на 6M, где эпоха ≈ 2.3k шагов.

## Блокеры и вопросы к Сергею

- `download.pytorch.org` закрыт сетевой политикой облачного окружения → torch поставлен с PyPI (CUDA-колёса, отступление от §5.2, D-002). Если нужно строго по спеке — добавить хост в allowlist.
- Целевая версия не проверена на реальном GPU и 6M строк: ориентиры в README — экстраполяция.

## Как воспроизвести последний зелёный прогон

```bash
scripts/setup_cloud.sh                      # .venv на python3.12 + зависимости
.venv/bin/python scripts/make_fixtures.py   # fixtures/
.venv/bin/python -m pytest -q               # 38 тестов, включая ноутбуки (~1 мин)
cd notebooks
../.venv/bin/jupyter nbconvert --to notebook --execute train.ipynb --output-dir ../outputs/nb
../.venv/bin/jupyter nbconvert --to notebook --execute inference.ipynb --output-dir ../outputs/nb
cd ..
# вторая архитектура (DoD 4):
.venv/bin/python scripts/run_train.py --preset ods_light --override '{"data": {"path": "fixtures/ods.parquet"},
  "output_dir": "outputs/train_ods_arch2", "model": {"hidden": [512, 256], "dropout": [0.3, 0.2], "norm": "layernorm"}}'
```
