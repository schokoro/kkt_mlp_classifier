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

## S01 · 2026-10-31

Реализованы автономный ограниченный fit TF-IDF и режим повторного использования в обучении; по умолчанию остаётся fit на train. На текущем дереве быстрый набор: **84 passed, 1 deselected**, полный: **85 passed**. После второго ревью его замечания исправлены и проверены тестами; независимого ревью этих последних исправлений не было ввиду лимита двух раундов. Проверки выполнялись на малых данных; 300 млн строк, промышленный корпус и GPU не проверялись. Итог см. `artefacts/S01/result.md`.

**Статус: ACCEPTED (2026-11-02).**
Спринт закрыт с принятым ограничением по независимому ревью.
## S02 · 2026-11-02

Проверочный спринт на ветке `sprint/S02-corpus-tfidf-validation`: внешний TF-IDF обучен на 100 000 и всех 981 850 строках подготовленного корпуса; полный fit 100,709 с, пик RSS 5 280 804 КиБ, словари char 300 000 / word 100 000. Pretrained train и inference smoke прошли; быстрый набор 88 passed, полный 89 passed. Это проверка механики и ресурсов конкретного корпуса, не оценка качества. Подробности и ограничения: `artefacts/S02/result.md`. Изменения не закоммичены и не интегрированы.

## S03 · промежуточное состояние

Спринт не принят. Ветка `sprint/S03-feature-selection` от `dev`; после двух ревью остались блокирующие замечания, а реализация через Kimi остановлена лимитом HTTP 429. Быстрый pytest 116 passed, 1 deselected; полный 117 passed, с предупреждением CUDA. Семь плановых запусков и итоговый отчёт не выполнены. Подробности: `artefacts/S03/result.md`, `artefacts/S03/review_2.md`.

### S03 — продолжение

Исправления замечаний второго ревью внесены и покрыты тестами, но независимое ревью этих исправлений не проводилось из-за лимита двух раундов. Плановый локальный эксперимент 7/7 success; отчёт `outputs/S03/report/`, быстрые тесты 121 passed, полный набор 122 passed. S03 остаётся без приёмки и без merge. Подробности и ограничения — `artefacts/S03/result.md`.

### S03 — дополнительное ревью

Разрешённое пользователем дополнительное ревью подтвердило результаты и нашло два дефекта отчётности; исправления покрыты тестами. Быстрый набор 126 passed, 1 deselected; полный 127 passed. Эти последние исправления не проходили отдельного независимого ревью. Семь успешных run и ограничения описаны в `artefacts/S03/result.md`.

### S03 — принят

2026-11-05 пользователь принял S03 с оговорками и ограничениями, зафиксированными в `artefacts/S03/result.md`.

## S04 — в работе, приёмка не пройдена

Ветка `sprint/S04-gradient-x-input` от `dev` f974016. Реализован предварительный путь Gradient × Input и выполнен общий CPU smoke, но результаты G×I получены до исправления ошибки рейтингов. Первое независимое ревью выявило открытые дефекты идентичности checkpoint/сравнений, расхода памяти, GPU и отчёта. Второй раунд ревью не проводился. Подробности, фактические проверки и ограничения — `artefacts/S04/result.md`.

### S04 — обновление после двух раундов ревью

Общий CPU smoke `outputs/S04_smoke_final` выполнил 10/10 runs, пять файлов отчёта созданы; HTML восстановлен из табличных файлов. Дополнительный `outputs/S04_corpus_smoke` выполнил baseline и по одному run каждого метода на `data/corpus.pqt`: 4 success, 6 skipped. Быстрый pytest: 175 passed, 1 skipped, 1 deselected; полный: 176 passed, 1 skipped. Пять замечаний второго ревью исправлены и покрыты тестами **после последнего разрешённого раунда**; независимого повторного ревью этих исправлений нет. GPU и промышленный масштаб не проверены. Полный статус D01–D13 и команды: `artefacts/S04/result.md`.

### S04 — проверка рейтингов после внешней проверки архива

`S04_review.zip` не содержит бинарных рейтингов; в обоих исходных каталогах counts-файлы присутствуют. Исправлен прямой повторный отбор, который принимал отсутствие объявленного файла; добавлены контрольные суммы новых рейтингов и регрессионные тесты. `outputs/S04_ratings_smoke` — 10/10 success; старые результаты сохранены. Быстрый pytest 241 passed, полный 242 passed; CUDA skipped. Бюджет ревью исчерпан, последние исправления независимо не ревьюировались; детали и статусы DoD — `artefacts/S04/result.md`.

### S04 — принят с оговорками (2026-11-06)

Пользователь принял S04 и риск отсутствия независимого повторного ревью исправлений после второго раунда. Результаты проверок: 242 passed, 1 skipped; 10/10 smoke; 7/7 повторных загрузок G×I selection. Частично выполненные DoD и ограничения GPU, масштабирования, бюджетов и интерпретации ODS сохраняются; подробности — `artefacts/S04/result.md`.
# S05 — audit и full-TFIDF baseline (2026-11-06)

Ветка `sprint/S05-audit-baseline` от dev; без commit/push/merge. Аудит подтвердил 18 046 строк / 737 L3 и 16 635 строк / 326 L3 после `min_count=10`; дубликатов после preprocessing и нарушений иерархии нет. Freeze split зафиксирован без пересечений, test не оценивался. Два нормализованных названия встречаются в неразмеченном корпусе S02 — возможна частичная трансдуктивность.

Три финальных full-TFIDF refit seed 42/43/44 успешны: comparison-val raw macro-F1 0.6984–0.7087, accuracy 0.7749–0.7819; majority macro-F1 0.000248. Train-only выбор — 10 эпох; опорный checkpoint seed 42 прошёл round-trip. Исправленный отчёт `outputs/S05/report_closed/report.html` создан из сохранённых результатов без обучения; прежний сохранён. Gate **conditional**: максимум независимого selection gap по loss 1.644991 на эпохе 9; исправления после второго и последнего независимого review не имеют дополнительного review. Confusion pairs не сформированы: сохранённых предсказаний нет, повторный inference не выполнялся. Исходные audit-таблицы с выявленными ошибками сохранены, рядом созданы `_corrected.csv`. Это диагностика ODS, не подтверждение промышленного качества.

Проверки финального кода до обучения: быстрый pytest 301 passed, 1 deselected; полный 302 passed; `git diff --check` чист. Полные статусы D01–D12, команды, hashes, ресурсы и ограничения — `artefacts/S05/result.md`. S06 не запускался.

После исправления gap при закрытии: быстрый pytest 303 passed, 1 deselected; полный pytest 304 passed. Новых независимых ревью не проводилось по указанию пользователя.

Пользователь закрыл S05 **с оговорками**, приняв риск отсутствия независимого ревью последних исправлений. Для S06 согласован новый протокол: до split оставить L3 с `count >= 12`, обеспечить не менее двух примеров каждого класса в validation и test, внутренний monitor только из train, test закрыт до окончательного выбора; фиксировать индексы, seed, fingerprints, поддержку и проверки утечек. Доли будут выбраны после проверки выполнимости. Baseline S06 необходимо переобучить на новом split; показатели S05 остаются историческими и не смешиваются с новым экспериментом. S06 ещё не реализован.

## S06 — промежуточное состояние (2026-11-06)

Ветка `sprint/S06-feature-selection-comparison`. На новом ODS-фильтре L3 support ≥12 сохранены 16 289 строк и 293 L3; получен воспроизводимый split с минимумом 2 на каждую L3 в validation, sealed test и monitor. Три полных baseline на comparison validation дали L3 macro-F1 0.717861, 0.721763 и 0.727254; sealed test не оценивался. Завершено 3 из 30 итоговых запусков, сравнение методов отсутствует. Второе независимое ревью выявило блокирующие дефекты gate/provenance и resume; на текущем коде полный pytest дал 340 passed, 16 failed, поэтому широкая сетка не запущена. Машинный план 30 запусков зафиксирован после оценки baseline. Подробности и ограничения: `artefacts/S06/result.md`, `artefacts/S06/review1.md`, `artefacts/S06/review2.md`.

Восстановление 2026-11-07: отчёт и проверка неизменяемости плана частично исправлены; последний полный pytest — 375 passed, 9 failed, 1 skipped. Kimi вернул 429 из-за лимита пятичасового окна. Gate, checkpoint и provenance/resume остаются блокирующими, новый smoke 10 конфигураций не запускался. Исторический baseline не считается частью нового формального сравнения. Подробности: `artefacts/S06/result.md`.

После сброса лимита восстановление продолжено: полный pytest **421 passed, 1 skipped**; интеграционный smoke в `outputs/S06_recovery_smoke_3/` — **10/10 success**, отчёт восстановлен из сохранённых таблиц без runs/shared. Resume намеренно отключён и отказывает без изменения файлов. 30 итоговых runs не запускались; для нового формального сравнения нужен свежий baseline с планом до оценки. Последние исправления не проходили третье независимое ревью. Подробности: `artefacts/S06/result.md`.

2026-11-07 — S06 final ODS comparison completed 30/30 runs in `outputs/S06_final_cpu_20261107/`; Gate 1 passed and sealed test was not evaluated. The aggregate report and table-only HTML rebuild are available; conclusions are limited to the previously examined ODS comparison validation. Full results and independently unreviewed final report fixes are documented in `artefacts/S06/result.md`. The user accepted S06 with documented limitations.

The corrected final report is `outputs/S06_final_cpu_20261107/report_modelpt_bytes/`; it includes SHA-256-verified serialized model sizes for all 30 runs and was rebuilt from its saved tables alone. Final full pytest after the report fix: 436 passed, 1 skipped (CUDA unavailable). The reporting fixes after the second review have not had another independent review.

## S06 — принято с оговорками

Пользователь принял S06 и разрешил commit и fast-forward merge в `dev` без push. Финальный эксперимент завершил 30/30 runs; полный pytest — 436 passed, 1 skipped (CUDA unavailable). Class-aware Chi² 100k: macro-F1 0.7368 против 0.7229 у full 400k, при меньшем размере модели и времени CPU-обучения. Ранее просмотренная validation, малая поддержка редких L3, CPU-only запуск и отсутствие независимого ревью последних исправлений отчёта остаются ограничениями. Sealed test не оценивался; подробности — `artefacts/S06/result.md`.


### S07 — active, pre-workload gates (2026-11-07)

Branch `sprint/S07-stability-validation` starts at integrated `dev` `c7df0d4`.
Revised AGENTS roles verified; user changes preserved. Verified S06 development
pool: 14,652 rows, 293 L3, no duplicate normalized groups; feasible fresh seeds
101/202/303, per split fit/monitor/validation 11,766/1,378/1,508 with minima 6/2/2.
S02 corpus/features and final S06 14-epoch protocol hashes match. Codex authored
implementation; a separate Codex authored tests. First read-only Kimi review
completed, M1/M2 fixes delegated, new independent regressions and second review
pending. Full pytest before review fixes: 504 passed, 1 skipped. Formal ODS
workload remains 0/36; diagnostic pending reports are not experimental results.
See `artefacts/S07/preflight.md` and `implementation_handoff.md`.


### S07 — completed ODS study, acceptance pending (2026-11-07)

Final experiment `outputs/S07/stability_20261107_v2/`: **36/36 success**, zero
failed/skipped/pending, nine split-local refits. Seeds101/202/303 × methods
none/global/class_aware/gradient_x_input × model seeds42/43/44, frozen14 epochs.
Full macro-F1 **0.724313**; global100k **0.720741** (pairedΔ -0.003572),
class-aware100k **0.725186** (+0.000873), G×I100k **0.724398** (+0.000085).
Each selected method gains in only1/3 split means; Class-aware gains in5/9 seed
pairs, G×I6/9, global3/9. S06's +0.0139 was not reproduced as a stable gain.
Nine correlated pairs are not independent holdouts. 200/293 validation classes
have only2–4 examples. Larger independent testing is warranted for resource
savings, without an accuracy/industrial superiority claim; G×I's extra cost is
not justified by macro-F1 evidence here.

Report `outputs/S07/stability_20261107_v2/report/report.html` includes all ten
files; saved-prediction metrics,27 paired deltas, per-class/bin/confusion tables
and18 name-based vocabulary overlaps independently verified. Isolated table-only
HTML rebuild is byte-identical. Full pytest **579 passed,1 skipped**; targeted
S07 **143 passed**. CPU execution2215.612 s, measured max stageRSS3,322,728,448
bytes; VRAMnull. Full/selected model.pt409,978,289/102,778,289 bytes.

Two read-only Kimi rounds completed, final PASS. A subsequent S07-only fix for
uncomputed LR diagnostic NaN serialization remains **independently unreviewed**;
53 added independent regressions passed, no third review. First attempt remains
preserved with0 success/1 failed/35 pending and no validation inference; v2 used
a new plan and retrained all models. All limitations, stage costs, hashes,
commands and final D01–D14 statuses: `artefacts/S07/result.md`. Sealed test and
old outputs preserved; no commit/push/merge. Await separate user acceptance.


## S07 — Accepted with limitations; closure authorized (2026-11-07)

The user accepted S07 with documented limitations and authorized commit and
fast-forward-only integration into `dev`, without push. Final verified study:
36/36 successes, splits101/202/303, model seeds42/43/44 and nine TF-IDF refits;
complete report/manifest `outputs/S07/stability_20261107_v2/report/`. Full400k
macro-F1=0.724313; pairedΔ Global100k=-0.003572, Class-aware100k=+0.000873,
G×I100k=+0.000085. No selected method consistently improved all three splits;
S06 Class-aware advantage was not reproduced. Models shrink ~410 MB→103 MB,
with substantially lower training time.

Accepted limits: sparse per-L3 validation, previously examined development/
validation data, correlated splits rather than external validation, CPU-only
measurements, tested final LR serialization fix without further independent
review, no established industrial quality/scalability. Historical attempts and
all experiment outputs remain intact. Closure full pytest **579 passed, 1 skipped** (77.99 s); frozen-plan/report
verification and both unstaged/staged Git whitespace checks passed. All 3,717
existing output files retain their sizes/mtimes; protected source/notebook
hashes unchanged. Concurrent unrelated ZIP disappearance/S08-document arrival
was observed without any closure deletion or S08 work; these files are excluded
from staging. The accepted S07 change set is ready for the authorized commit
and fast-forward integration; no push or branch/artifact cleanup.


### S08 — active preparation, holdout CLOSED (2026-11-07)

Branch `sprint/S08-noninferiority` created from integrated S07 `dev` `dd7559a`. Original S08 documents and unrelated untracked notebook preserved. Source fingerprints verified against accepted S07, development union14,652 rows/293 L3, original sealed identity1,637 rows; only structural projection/counts/hashes used. Original support audit has114 holdout classes with2 examples,186 with2–4. Separate Codex implementation/tests completed initial preparation and synthetic9-model smoke; first read-only Kimi review in progress, max2 rounds. Formal training0/9. Margin and primary candidate unapproved, no sealed labels/inference/metrics. Implementation of an authenticated future evaluation path remains a preparation issue before freezing final models; current evaluation stub always rejects. Full suite sandbox notebook socket failure is being rechecked with permitted sockets. See `artefacts/S08/preflight.md` and worker handoffs; no NI result exists.


### S08 — prepared study awaiting_user_approval (2026-11-07)

`outputs/S08/noninferiority_20261107_v1/` completed **9/9 final models**, Full/Class-aware/Global × seeds42/43/44, full14,652-row development union/293 L3, fixed14 epochs. Two exact981,850-row corpus refits,100k=75k+25k, IDF differences0; Full400k unchanged. Execution625.756s CPU/two threads, max measured stageRSS2,864,926,720bytes, model.pt409,978,289/102,778,289bytes; no GPU/inference measurements. Final fullpytest712passed/1skipped; quick711passed/1skipped/1deselected;133 S08 tests. Separate Codex implementation/tests/fixes/regressions; two read-only Kimi rounds final PASS, no post-review code changes. H1 permanently closed stub resolved before freeze through authenticated detached decision and atomic one-time evaluation path.

Pre-unseal report `report/report.html` and four saved reporting inputs rebuilt byte-identically in isolation. Complete training, confirmatory_complete=false. **Gate awaiting_user_approval; delta_NI=null; Class-aware primary unapproved; .01 planning-only.** Original sealed labels/inference/metrics were NOT accessed. No five test-dependent report files, approval or consumption claim. Existing3,717 outputs/user notebook/SPEC/original S08 documents preserved. Numerical exploratory feasibility: downward uncertainty .00944–.01707 under two hypothetical error scenarios; sparse classes mean a one-point margin may be unresolved. Next: one explicit stakeholder decision on margin+rationale, primary and one-time original sealed use. D03partial, D09blocked; no NI claim, commit/merge/push or acceptance. Full commands, hashes, D01–D14 and residual risks: `artefacts/S08/result.md`.


### S08 — authorization received; confidence-level conflict before evaluation (2026-11-08)

User explicitly approved delta_NI=.01 with practical ODS research rationale, Class-aware100k primary, Full400k reference, Global exploratory and one-time original S06 sealed use for9 frozen models. Authorization persists; no repeat consent required. Latest instruction also requests one-sided95% while prohibiting changes to reviewed procedure. Frozen alpha=.025 means one-sided97.5%; alpha=.05 would change/reject the frozen procedure. Status `evaluation_blocked_pending_confidence_level_clarification`, before consumption. All9 checkpoints/features/maps/training/source hashes reverified; original sealed hash unchanged.0/9 test evaluations, no sealed labels/predictions/metrics, no claim or executable approval record. Existing models/preparation report preserved, no code/refit/training/review/commit/merge/push. Fullpytest sandbox run711passed/1skipped/1failed (notebook socket); permitted-socket rerun pending. See S08/evaluation_authorization_20261108.md and latest result.md section.

S08 continuation verification completed: fresh socket-enabled fullpytest **712passed/1skipped** (187.68s, CUDA warning), `git diff --check` clean. Protocol/models/preparation report unchanged;0/9 test evaluations, no claim/access. Statistical confidence-level clarification remains unresolved; stop before sealed access. No completed-NI acceptance recommendation.


### S08 — final authorized NI evaluation complete; user acceptance pending (2026-11-08)

The user corrected the confidence wording to frozenalpha=.025/one-sided97.5%, resolving the pre-access blocker without protocol/code/hash changes; prior explicit margin.01/primary/one-time-use approval remained valid. Repeated integrity gate passed. **9/9 original S06 sealed-test evaluations succeeded once**,1,637 rows/293 L3, seeds42/43/44; use consumed permanently, no retry/resume. Final report `outputs/S08/noninferiority_20261107_v1/report_evaluated_20261108_v1/report.html`; original preparation report preserved.

Mean macro-F1 Full **.730354516**, Class-aware100k **.731956271**; primary pairedDelta **+.001601754916**, one-sided97.5% lower bound **-.009842370585 > -.01**, formal **non_inferior**, narrow clearance .000157629415. Global exploratory mean .721844764/Delta-.008509751678; no secondary confirmatory decision/superiority claim. Accuracy Full.793728365 versus Class-aware/Global.789655875.114 classes have2 examples,186 have2–4; Class-aware gains concentrated in sparse2–4 bin, negative in larger bins.

Separate Codex saved-output numerical audit **2,843 assertions PASS**, exact CI/metrics, all14,733 prediction rows/2,637 class rows/27 bin rows/2,751 confusion rows,33 measured stages and bindings; table-only HTML byte-identical. Bundle `outputs/S08/verification_20261108_v1/`. Latest fullpytest on unchanged source **712passed/1skipped**187.68s; whitespace clean;2 completed Kimi reviews remain final, no third. Training/model/feature/protocol/preparation fingerprints and3,717 old outputs/user notebook/SPEC unchanged. No retrain/refit/selection/code changes/commit/merge/push.

AllD01–D14 completed within measured/PoC scope; user acceptance remains separate. Recommend accepting S08 with sparse-support/limited-power/finiteB5000/three-fixed-seed/prior-dev-exposure/partly-transductive/internal-ODS/CPU-only/no-production-guarantee limits. Future adaptive confirmation requires new independent holdout labels; original test is no longer fresh. Full commands, hashes and exact limits: `artefacts/S08/result.md`; independent audit `evaluation_verification.md`.


## S08 — Accepted with limitations; commit/fast-forward closure authorized

The user explicitly accepts S08 with documented limitations and authorizes intended S08 commit(s) followed by fast-forward-only integration into `dev`. Push, branch deletion, cleanup and further experiments are not authorized. Final evaluated report `outputs/S08/noninferiority_20261107_v1/report_evaluated_20261108_v1/`:9/9 one-time original holdout evaluations; Full400k macro-F1.730355, Class-aware100k.731956, Global100k exploratory.721845. Approved margin.01/alpha.025/one-sided97.5%; primaryDelta+.00160175, lower-.00984237 > -.01, **non-inferiority established** with only.00015763 clearance. Superiority not established.

Acceptance retains near-boundary NI, sparse L3 holdout support, previously examined development results, three fixed seeds rather than independent datasets, permanently consumed sealed test, unrepresentative ODS PoC and CPU-only/no-industrial-quality-or-scalability guarantees. Authorization/verification/preparation history preserved; D01–D14 consolidated in S08/result.md. Fresh closure tests, manifest/hash verification and staging-scope checks gate commit/integration. Preserve unrelated ZIP/notebook, datasets/models/predictions and all generated outputs; no original evaluation/bootstrap/training rerun or additional Kimi review.

S08 closure verification passed: fresh fullpytest **712passed/1skipped**173.97s (CUDA warning); approved final report/manifest/claim and original protocol/model/feature hashes unchanged, no original inference/bootstrap/training. All3,987 current output files retain sizes/mtimes and protected user ZIP/notebook/SPEC/source-doc/AGENTS hashes match.34-file staging allowlist excludes outputs/data/models/predictions/unrelated files; tracked/new-file whitespace checks clean. Read-only remote lookup confirms origin/dev=localdev=dd7559a before integration. Proceed with authorized sprint commit and ff-only merge after exact staged-set check; no push/cleanup. Final SHAs and integration outcome are reported to the user.
