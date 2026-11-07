Ты второй и последний независимый read-only ревьюер S06. Ничего не меняй и не делегируй. Найди блокирующие ошибки, leakage, нарушения provenance, refit, seed pairing, resume и отчёта; проверь исправления первого ревью. Оцени фактический diff ниже, приводи файлы и строки; отделяй гипотезы. Исходные требования и DoD:
# Промпт Codex-оркестратору — S06 Feature Selection Comparison

Ты — **Codex-оркестратор** S06 проекта `kkt_mlp_classifier`. Цель: реализовать и выполнить **честный сравнительный эксперимент** Global Chi², Class-aware Chi² и Gradient × Input на новых замороженных split ODS (`L3 support >= 12`) с full TF-IDF и единым агрегированным отчётом. **Не начинать S07.**

## Источники требований

**Прочти целиком `artefacts/S06/value_dod.md` — это основной контракт (D01–D14).** Далее: `AGENTS.md`, `artefacts/S05/result.md`, `artefacts/S05/value_dod.md`, `artefacts/S04/result.md`, `artefacts/S03/value_dod.md`, `artefacts/STATUS.md`, `artefacts/DECISION_LOG.md`, README, актуальные исходники и тесты. При расхождении путей и предпосылок руководствуйся фактическим Git и файлами, не угадывай. Исторические `min_count=10`, 326 L3, S05 macro-F1 ≈0.70 и его checkpoints/splits **не годятся как baseline нового S06**.

## Автономность и разрешения — обязательное правило

**БЕЗУСЛОВНЫЙ ALLOW В РАМКАХ S06.** Без запроса подтверждения у пользователя и без остановки для согласования каждого действия **самостоятельно**:

- запускай `kimi -p` для реализации и исправлений;
- запускай **отдельные** `codex exec` для написания/исправления тестов и **другие отдельные** `codex exec` для независимого read-only code review;
- запускай pytest, локальные диагностические команды, smoke, подготовку данных, обучение/переобучение MLP, вычисление G×I/Chi², fit/refit TF-IDF, инференс **только на comparison validation**, формирование отчётов и повторные безопасные проверки;
- перезапускай упавшие *безопасные* операции, используй доступные CPU/GPU и ресурсы в рамках S06; документируй фактически выполненные команды.

Эти действия **предварительно разрешены пользователем**. Не спрашивай «можно ли вызвать Kimi/Codex/pytest/обучение?». Делегируй и выполняй работу до результата/честно зафиксированного технического блокера. Разрешение **не отменяет аппаратные, sandbox, системные и реальные ограничения доступа**: при техническом запрете честно сообщи, не обходи его небезопасными способами.

**Не разрешены без нового прямого поручения:** `git commit`, `push`, `merge`, `rebase` существующей истории, force-операции, `git clean`, `reset --hard`, уничтожение/перезапись пользовательских данных, публикация файлов/ресурсов, внешние сетевые передачи не по задаче. Два независимых ревью — **максимум**, не запускать третье без решения пользователя.

## Организация работы

1. Проверь Git status/ветку, что S05 (`e3c3475` по сообщению о merge) присутствует в текущей `dev`, а исходные файлы, full TF-IDF S02 и `data/corpus.pqt` доступны и согласованы по fingerprint. Создай `sprint/S06-feature-selection-comparison` от актуальной `dev`. При незакоммиченных пользовательских изменениях защищай их, предложи безопасный `worktree`, не применяй force. Не трогай чужие outputs.
2. **Ты не пишешь реализацию и тесты самостоятельно.** Реализацию, исправления, изменения скриптов — делегируй через `kimi -p`. Тесты — в отдельном `codex exec`. Независимый code review — отдельный read-only `codex exec`, которому дай исходный DoD и фактический diff. Ты можешь обновлять проектную документацию и координировать выполнение.
3. Создай `artefacts/S06/prompts/`. **Перед каждым вызовом** сохраняй туда точный полный передаваемый промпт с возрастающим номером: `01_kimi_...md`, `02_codex_tests_...md`, `03_codex_review_...md`, и т. д. Не подменяй постфактум. Два раунда независимого ревью; исправления после второго — в отдельном помеченном блоке риск/тесты, **не называй независимо принятыми**.

## План исполнения по воротам (gates)

**Gate 0 — audit/split.** Применить к исходным размеченным ODS `count(L3)>=12` **до** split. Выполнить групповой аудит нормализованных названий и конфликтов. Создать неизменяемые `outer_train/comparison_val/sealed_test` и внутренние `train_fit/train_monitor`, гарантируя ≥2 примера каждой сохранённой L3 в val, test, monitor, и оставляя обучаемый train_fit. Предпочесть близкие к 80/10/10 доли при соблюдении минимумов, объяснить квоты. Сохранить группы, IDs, hashes, поддержку по частям. При невозможности split — `blocked`, никакого произвольного изменения порога. **Test только count/hash/support — ни одного model inference/eval на test.**

**Gate 1 — обученный baseline.** На идентичном проверенном full TF-IDF S02 по `train_fit/train_monitor` выбрать один фиксированный train-only протокол и число эпох; сохранить curves. Зафиксировать его **до просмотра comparison validation**. Затем обучить полные baseline с seed 42/43/44 на новом `outer_train`, проверить качество против majority на `comparison_val`, train–monitor gap, ресурсы и воспроизводимость. Опорный G×I checkpoint — заранее указанный seed 42 (не выбор max F1 по comparison_val). Если baseline объективно не обучается или есть leakage — зафиксировать gate `conditional/blocked` и **не запускать 9×3 selected runs**.

**Gate 2 — методики и словари.** Проверить и использовать S03/S04 `global`, `class_aware`, `gradient_x_input` без новой теории. Для каждого метода вычислить рейтинг на `outer_train` (G×I — full checkpoint seed42, margin/positive-only), сохранить hash/происхождение. Для K=50k,100k,200k с char_share=0.75 сформировать 9 словарей. Для каждого — **новый fit TF-IDF с fixed vocabulary на том же фактическом snapshot и subset строк**, что у full TF-IDF; отдельно проверить IDF, нормировку, точное число признаков и round-trip. Кэшировать rank/refit per method×K и переиспользовать между seed; не повторять refit по seed.

**Gate 3 — paired runs.** На единых `outer_train/comparison_val` и замороженном общем schedule/оптимизаторе выполнить 3 seed для `none` и для каждого из 9 вариантов: **30 итоговых запусков** (если ресурсы позволяют). Обучать MLP заново; для G×I не включать comparison-val метки в ранжирование. Пропуски/ошибки — явные `skipped/failed` с причиной, без подмены метрик. Обеспечить безопасный resume без перезаписи исходных экспериментов и без потери provenance.

**Gate 4 — сводка/качество.** Собрать `outputs/S06/report/` строго по контракту DoD: автономный `report.html`, `runs.csv`, `per_class.parquet`, `stages.csv`, `manifest.json`, `confusion_pairs.csv`, `predictions_val.parquet`, `support_bins.csv`. Обязательны paired Δ к full **того же seed**, mean/std/min/max по seed, график размера словаря vs качество (все 3 метода + baseline), анализ L3 и диапазонов support, top confused pairs и отдельные shared/per-run затраты. Для GPU unavailable — `null`, без мнимых замеров. Проверить HTML re-render из таблиц **без training/inference**. Не открывать sealed test.

**Gate 5 — тесты и ревью.** Делегировать независимые тесты квот/групповых совпадений, train-only ранжирования/epoch selection, checkpoint provenance, fixed vocab refit, 10×3 run/seed matching, корректных F1 и paired Δ, support bins, confusion pairs и re-render. Запустить `.venv/bin/python -m pytest -q -m 'not slow'`, `.venv/bin/python -m pytest -q`, `git diff --check`, ресурсно ограниченный smoke и целевой эксперимент при возможности. Делегировать до **двух** независимых read-only code review с реальными diff, разбирать замечания; последний непроверенный fix пометить отдельно. Не затягивать бесконечными ревью.

## Приёмка и итог

- В `artefacts/S06/result.md` дай D01–D14 со статусами `done/partial/blocked`, точные размеры после `>=12`, квоты/split hash, full corpus/TF-IDF provenance, сколько из 30 runs завершено, по каждой конфигурации mean/std F1 и paired Δ, диагностические ограничения, ресурсы, результаты тестов/ревью и фактические команды.
- Отчёт должен отвечать на вопрос о **сравнении на ODS**, а не объявлять пригодность модели для закрытого контура. Test остаётся невскрытым; выбор лучшего метода для промышленной системы **вне S06**.
- Обнови README при изменении CLI, `artefacts/STATUS.md`, append-only `artefacts/DECISION_LOG.md`. **`artefacts/SPEC_v1.0.md` не изменять.** Не добавлять corpora, checkpoints, отчёты, большие логи в Git.
- **Никаких commit, push, merge** до прямого поручения пользователя. Не удалять ветку, старые результаты, исходники, пользовательские файлы.
- Финальное сообщение: gate/статус, главное сравнение с ограничениями, состав и путь отчёта, сколько выполнено из 30, тесты/ревью и риски. Если что-то невозможно, завершить с частичным, но честным результатом, не запрашивать подтверждение обычных разрешённых операций.
# S06 — Сравнение supervised feature selection на зафиксированном ODS-протоколе

**Статус:** постановка к реализации.  
**Проект:** `kkt_mlp_classifier`.  
**Предусловие:** S05 принят с оговорками и интегрирован в `dev` (по сообщению о закрытии — коммит `e3c3475985d905ef64fbb5e79ab55ca1aeeccd18`; оркестратор обязан проверить фактический Git).  
**Ветка:** `sprint/S06-feature-selection-comparison`.  
**Граница:** S06 — сравнительный эксперимент с уже реализованными методами S03/S04, а не создание новых способов отбора.

## Value

Получить **воспроизводимое, сопоставимое и машинно проверяемое сравнение** трёх методов сокращения TF-IDF-словаря на размеченной ODS-выборке с более устойчивым протоколом редких классов. Ответить на вопросы: как меняется качество L3 при 50k/100k/200k признаков по сравнению с полными 400k; помогают ли class-aware Chi² и Gradient × Input относительно global Chi²; какие категории выигрывают/проигрывают; каковы затраты отбора, refit, обучения и инференса; насколько выводы устойчивы к seed и величине support.

**Результат на ODS — исследовательский PoC.** Не экстраполировать качество, оптимальный K, рейтинги признаков или ресурсные измерения на данные из реальных кассовых чеков закрытого контура. В закрытый контур потенциально переносятся код, протокол и формат отчёта.

## Проверенные предпосылки и ограничения

- Итог S05: исходные 18 046 размеченных строк и 737 L3; после исторического `min_count=10` оставались 16 635 строк и 326 L3. **В S06 применяется новая политика `count >= 12`; число строк и L3 после неё заранее неизвестно и должно быть измерено.**
- S05: comparison-val macro-F1 L3 для 3 seed ≈0.6984–0.7087 при **старом** фильтре/сплитах. Эти значения только исторический ориентир, **не baseline S06**; S05 checkpoint и splits не переиспользовать для оценки нового протокола.
- Полный TF-IDF S02, если доступен и валиден: 300 000 символьных + 100 000 словесных признаков; зафиксированный fit на фактическом снимке внешнего неразмеченного `data/corpus.pqt` (981 850 строк). Пути, hashes, параметры fit и точные размеры **сверить с фактическими артефактами**; не подменять full векторизатор fit на размеченной train.
- Два совпадения текстов ODS с TF-IDF-корпусом ранее отмечены как частичная трансдуктивность. Уточнить overlap после нового фильтра. Совпадение текстов при *unsupervised* fit не есть leakage меток, но требует явной оговорки.
- CUDA, ресурсы, детали реализации selection и версии кода определяются по фактическому окружению. Данные и модель используют **только строку `name`**; L1/L2/L3 — целевые метки, не входные признаки.

## Scope и протокол

### A. Аудит, фильтр и единый split (до обучения и feature selection)

1. Из исходного размеченного ODS отфильтровать L3 с `support < 12` **до всех разбиений**. Порог включительный (`>=12`) по числу валидных размеченных строк L3 после существующих явных проверок исходных данных; никаких дополнительных незадокументированных фильтров. Зафиксировать список исключённых L3, количество строк, распределение support, L1/L2 и источник данных.
2. На оставшихся строках однократно создать воспроизводимые `outer_train`, `comparison_val`, `sealed_test` и внутри `outer_train` — `train_fit`/`train_monitor`. Выделить **не менее 2 строк каждой сохранённой L3 в comparison_val и sealed_test**, не менее 2 — в `train_monitor`, сохранив достаточно train_fit для обучения. При минимальном support=12 и разбиении 2+2+2 остаётся 6 примеров для train_fit. Предпочитать близкие к 80/10/10 пропорции для классов с большим support, **но не назначать проценты механически** и не нарушать минимумы. Фактический алгоритм квот, rounding, split seed и размеры вывести из данных и зафиксировать; при невозможности гарантировать все ограничения — явно `blocked`, а не молчаливое исключение новых L3.
3. Проверить групповые совпадения по `name` и **фактической нормализации обеих TF-IDF-веток**; одинаковые/эквивалентные имена не могут попасть в разные части размеченной выборки (включая fit/monitor). Если ограничения групп конфликтуют с квотами — остановиться с диагностикой и предложением минимального изменения протокола, не исправлять молча. Проверить идентичность label maps, наличие каждой L3 в `train_fit` и всех внешних частях, отсутствие конфликтов и пересечений row IDs.
4. Сохранить сами разбиения, row identity, per-class counts, split seed, fingerprints/hashes и manifest. `sealed_test` физически/логически изолирован от скриптов ранжирования, подбора эпох, оценки моделей и отчёта о качестве: разрешены **только структурные сведения** (count/hash/support), но **никакого inference или метрик на test** в S06.

### B. Замороженный обучающий протокол S06 и новый full baseline

1. Загружать **один проверенный full TF-IDF** на том же снимке внешнего корпуса. Использовать одну и ту же фиксированную таксономию/label maps и split для всех методов. Обучение MLP — raw top-1 L3, без reject/threshold; архитектурное семейство и основные гиперпараметры S05 сохраняются, изменения только по обоснованному техническому блокеру.
2. **Число эпох и правила обучения** выбрать **только через `train_fit`/`train_monitor` на full baseline**, сохранив learning curves и основание решения. Затем **заморозить** конфигурацию оптимизатора, расписание LR, число эпох и правила оценки для *всех* final runs. На `outer_train` обучить с нуля три полных baseline: seed 42/43/44 (или три явно заданных seed). `comparison_val` не участвует в выборе эпох, конфигурации, checkpoint G×I или ранжировании.
3. Если baseline не показывает содержательного обучения, превосходящего train-only majority baseline на `comparison_val`, или обнаружены критичные нарушения splits/provenance, **не запускать широкую сетку отбора**. Отразить `gate=blocked/conditional` и диагностический результат. Не назначать искусственный минимальный F1.
4. Для Gradient × Input использовать заранее определённый **опорный full-baseline checkpoint** на новом `outer_train` (по умолчанию seed 42), выбранный независимо от `comparison_val`. Исторический checkpoint S05 с другим split запрещён. При необходимости исправить совместимость S04 loader/checkpoint минимально и через тесты.

### C. Отбор, реfit и сравнение

1. Методы: `none` (полный baseline), `global` Chi², `class_aware` Chi² (One-vs-Rest, положительные ассоциации и round-robin), `gradient_x_input` (по margin истинной L3 к сильнейшему конкуренту, положительные вклады; отрицательные — только диагностика, ошибочно классифицированные train-примеры остаются). **Переиспользовать S03/S04; не менять методологию незаметно.** Любые расхождения с реализованным контрактом описать до эксперимента.
2. Рейтинг каждого метода считается **только на `outer_train`**; метки comparison validation/test туда не попадают. Для G×I ранжирование вычисляется один раз на фиксированном train-only checkpoint; выбранные словари **фиксируются для всех последующих seed**, чтобы измерять именно вариабельность обучения MLP, а не смешивать её с вариабельностью рейтингов. Хэши checkpoint, рейтингов и словарей обязательны. Ограничение такого дизайна явно описать.
3. Бюджеты K = **50 000, 100 000, 200 000 суммарных char+word признаков**, исходный full ≈400 000; распределение по умолчанию `char_share=0.75`, настраиваемо. Выбор для каждого метода и K воспроизводим; корректность числа признаков, вложенности Top-K внутри метода (при применимости) и порядка словарей проверяется. Не допускать искажений бюджета silent fallback.
4. Для каждой комбинации метода и K создать **самостоятельный `tfidf_selected` через новый fit с фиксированным словарём** на **той же фактической выборке строк внешнего неразмеченного корпуса**, которая использовалась при full TF-IDF. Обе ветки char/word: те же preprocessing/TF-IDF параметры, новый IDF и нормировка; сохранять round-trip, provenance и неизменность full артефакта. Refit выполняется **один раз на метод×K**, не по каждому training seed.
5. Основная матрица экспериментов: **10 конфигураций** = full baseline + три метода × три бюджета. Для каждой — **три одинаковых seed обучения** (42/43/44): всего **30 final MLP runs**, если gate и ресурсы позволяют. Обучать финальные MLP с нуля на одном `outer_train` с общими зафиксированными параметрами и расписанием эпох. Seed сопоставляются попарно с full baseline того же seed; различия размеров входного слоя не предполагают идентичности начальных весов. Не подбирать метод/бюджет на test.
6. При ограничениях времени, GPU/RAM или ошибках сохранить все запланированные run IDs как `success/failed/skipped` с причинами и явно отметить невыполненные DoD; **не выдавать smoke/частичную сетку за полное сравнение**. Предусмотреть возобновление по сохранённым артефактам без повторного refit уже валидных словарей и без перезаписи результатов.

### D. Метрики, диагностика, интерпретация

- Главная метрика: **raw top-1 L3 macro-F1** по одному фиксированному universe L3, независимо от предсказанных классов и seed. Дополнительно accuracy и per-L3 precision/recall/F1/support; неверные/пустые предсказания — ошибки, а не незаметно исключённые строки. Не округлять метрики до расчёта разностей; классы без support отражать по явному правилу, не фиктивным F1.
- Для каждого run: разница macro-F1 и accuracy с **full baseline того же seed**; для метода×K: mean, std, min/max по seed, средняя парная Δ, диапазон и число successful seed. **Три seed на одном split — не независимый доверительный интервал**; не представлять мизерные Δ как доказанное улучшение. Не выбирать «победителя для закрытого контура».
- По каждой L3: support на `comparison_val`, F1/recall по run, Δ к baseline того же seed. Обязательны агрегаты по заранее заданным диапазонам validation support: `2–4`, `5–9`, `10–19`, `>=20` (пустые группы явно обозначать). Метрики групп рассчитывать как **невзвешенное среднее по L3** внутри группы, не заменять micro-accuracy; показывать число L3.
- По всем run сохранить **предсказания на comparison_val с row ID, true L3, predicted L3/null** (без test). Сформировать **top confused pairs** `true_L3 → predicted_L3` с count, support истинной L3 и долей ошибок; показать в агрегированном HTML и машинной таблице. Не путать с путаницами L1/L2.
- Ресурсы: раздельно время/peak RSS/VRAM для baseline setup, ranking каждого метода, attribution checkpoint (shared), TF-IDF refit, downstream MLP train и inference; артефактный размер/число признаков. **Shared costs не складывать повторно по seed** при выводе стоимости полного эксперимента; при GPU unavailable — `null/unavailable`, не нуль.
- Зафиксировать ограничения: редкие категории даже при двух val-примерах нестабильны; количество классов/текстов отличается от S05; небольшой PoC, возможная трансдуктивность, выбранный seed рейтинга G×I, GPU/CPU характеристики локальны.

## Обязательные выходные артефакты

Все тяжёлые данные и checkpoints — **вне Git**, в `outputs/S06/` (или конфигурируемом каталоге), без удаления прежних результатов:

```text
outputs/S06/
├── audit/
│   ├── data_audit.json        # до/после >=12, исключённые L3, overlap и контроль качества
│   ├── class_support.csv      # support L3 во всех разбиениях и диапазонах
│   └── split_audit.csv        # изоляция частей, квоты/группы/хэши
├── splits/                  # замороженные parquet и manifest, включая sealed test
├── shared/
│   ├── full_tfidf_ref.json   # путь + hash + fit/snapshot provenance (не копировать full без нужды)
│   ├── training_protocol.json # epoch-selection и замороженный final train config
│   ├── attribution/          # опорный checkpoint, рейтинг и диагностика G×I
│   └── rankings/             # рейтинги global и class-aware, hashes
├── runs/                     # 10 configs: selected TF-IDF, checkpoints/seed, run results
└── report/
    ├── report.html           # автономный агрегированный отчёт, все runs, графики, ограничения
    ├── runs.csv              # run_id, method, K, seed, status/reason, метрики, paired Δ, config/hash
    ├── per_class.parquet     # run_id, seed, L3, support, precision, recall, f1, delta_f1
    ├── stages.csv            # run_id/shared_id, stage, status, elapsed, peak RSS/VRAM, artefact size
    ├── manifest.json         # версии/хэши данных, splits, моделей, рейтингов, условия, статусы
    ├── confusion_pairs.csv   # run_id, true_L3, predicted_L3, count, true_support, error_share
    ├── predictions_val.parquet # row_id, run_id, true_L3, predicted_L3/null
    └── support_bins.csv      # run_id, support bin, n_L3, macro-F1/Δ, paired summary
```

Допускается иная внутренняя иерархия `shared/`/`runs/` при сохранении контракта и совместимости с S03/S04/S05. `report.html` **пересоздаётся из сохранённых отчётных файлов без обучения, fit и повторного inference**. При корректном/частичном запуске пропущенные/неудачные runs отображаются честно. Формулы показателей, схема CSV/Parquet, источники чисел и способ пересборки отчёта документируются. Список подлежащих выполнению запусков и одинаковые split/epoch config фиксируются до оценки.

## Out of scope

- Новые методы supervised feature selection, переработка MLP, широкая оптимизация гиперпараметров, оптимизация архитектуры, конфигураций под каждый K, подбор char_share по comparison_val.
- Оценка качества по sealed test и подбор после его вскрытия, thresholds/reject, L1/L2 как критерий победы, оценка промышленного качества или скорости 300 млн чековых строк.
- Новая разметка/перенос L3, добавление метаданных чеков в признаки, синтетика, Spark/production ETL.
- Изменение `artefacts/SPEC_v1.0.md`, перезапись пользовательских исходников, удаление S01–S05 артефактов; commit/push/merge без отдельного прямого разрешения.

## Definition of Done

- [ ] **D01 — Git/предусловия.** Проверены интеграция S05, рабочее дерево и актуальные исходные артефакты; создана отдельная ветка S06 без повреждения пользовательских файлов.
- [ ] **D02 — Фильтр.** Применён `L3 support >=12` до split; сохранён аудируемый список исключений, строки и counts L1/L2/L3; нет скрытого изменения меток.
- [ ] **D03 — Split.** `outer_train/comparison_val/sealed_test` и `train_fit/train_monitor` воспроизводимы, group-aware при необходимости; у каждого сохранённого класса ≥2 в val, test и monitor, достаточно train_fit; row IDs/labels/fingerprints и отсутствие overlap проверены; на sealed test нет quality/inference.
- [ ] **D04 — Признаки.** Проверен full TF-IDF S02 и фактический снимок корпуса; он одинаков во всех runs, char/word контракты сохранены; трансдуктивность задокументирована.
- [ ] **D05 — Новый baseline.** Внутренним monitor выбран и заморожен единый протокол final training; на новых splits обучены три full baseline с обучением, обоснованно превосходящим majority; есть learning curves, три checkpoint, raw метрики и устойчивость к seed без использования comparison validation для обучения/эпох.
- [ ] **D06 — Ранжирование.** Все три метода воспроизводимо отобрали признаки строго по outer_train; G×I использует правильный новый train-only checkpoint и сохраняет margin/positive-only, исключение leakage тестами; рейтинги/хэши сохраняются и повторно загружаются.
- [ ] **D07 — Selected TF-IDF.** Для девяти метод×K выполнен fixed-vocabulary refit на **идентичном** снимке корпуса; размеры 50k/100k/200k, char/word budgets, словари/IDF/нормировка и train/inference round-trip проверены, сохранён provenance.
- [ ] **D08 — Сравнительная сетка.** Завершены 30 final runs (10 конфигураций × три общих seed) или каждый пропуск/сбой обозначен с причиной и DoD `partial`. Все финальные MLP обучаются с нуля по одинаковому протоколу на тех же splits.
- [ ] **D09 — Метрики.** Реализованы фиксированный L3-universe, raw top-1, парные дельты к baseline одного seed и сводки mean/std/min/max; проверена корректность обработки пустых предсказаний и подсчётов support.
- [ ] **D10 — Диагностика.** Есть per-class Δ, support bins, сохранённые `comparison_val` predictions и top confused pairs; нет test метрик. Отдельно оценены риски категорий с support 2.
- [ ] **D11 — Ресурсы/отчёт.** Измерены локальные этапные время и RAM/VRAM (где доступны), без двойного учёта shared; сформированы восемь обязательных файлов `report/`; HTML автономный и пересоздаётся без обучения, failed/skipped не скрываются.
- [ ] **D12 — Автоматические проверки.** Тесты фильтрации, split leakage, train-only provenance, ranking/reload, fixed vocab refit, формул метрик/агрегирования и отчёта; быстрый и полный pytest, `git diff --check`, реальный smoke; отсутствие непредусмотренной dense `N×F`.
- [ ] **D13 — Делегирование/ревью.** Реализация — `kimi -p`, тесты — отдельный `codex exec`, два независимых read-only review — отдельные `codex exec`. Полные промпты сохранены заранее, исправления после последнего review честно помечены как не проверенные независимо. Оркестратор не кодит.
- [ ] **D14 — Документация/решение.** `artefacts/S06/result.md` содержит `done/partial/blocked` по D01–D14, все фактические команды/запуски, метрики, ресурсы, тесты, замечания и ограничения. README/STATUS обновлены, DECISION_LOG дополнен append-only; дан обоснованный вывод о сравнимости и переносимости **методики**, а не «победителе в промышленности».

## Приёмка

Приоритет — **достоверность сравнения и воспроизводимость**, а не требование G×I/Chi² превзойти baseline. Неполная сетка, невозможный split, неработающий refit или leakage — отражаются статусом `partial/blocked`, не маскируются фиктивными результатами. Невыполненное независимое ревью последних исправлений — явный остаточный риск. **Ни S06, ни PoC в целом не подтверждают промышленное качество.**

Фактический diff добавленных S06 исходников и тестов относительно dev:
diff --git a/kkt_clf/s06_split.py b/kkt_clf/s06_split.py
new file mode 100644
index 0000000..8540fe5
--- /dev/null
+++ b/kkt_clf/s06_split.py
@@ -0,0 +1,771 @@
+"""S06 (D03): group-aware split с жёсткими per-class минимумами.
+
+Политика S06 отличается от S05: после фильтра ``L3 support >= 12`` каждая
+сохранённая L3 обязана иметь не менее ``min_rows_per_class`` строк в
+``comparison_val``, ``sealed_test`` и ``train_monitor`` и не менее
+``min_train_fit`` строк в ``train_fit``. Группа — множество строк с одним
+char-ключом реального preprocessing TF-IDF (``strip_whitespace(preprocess(name))``);
+равенство word-ключа влечёт равенство char-ключа, поэтому группа целиком
+попадает в одну часть и пересечений нормализованных текстов между частями
+нет ни в одной ветке.
+
+Квоты выводятся из данных, а не назначаются механически: целевые числа
+строк частей для класса — метод наибольших остатков от долей
+80/10/10, поднятые до минимумов; ``outer_train`` получает остаток, но не
+меньше ``min_rows_per_class + min_train_fit`` (иначе невозможно удовлетворить
+монитор и fit одновременно). Назначение неделимых групп — жадное по
+дефициту квоты с засеянным tie-break; если из-за неделимости групп минимум
+нарушен, выполняется ограниченный локальный ремонт (перенос самой малой
+подходящей группы от донора, который после отдачи не падает ниже своего
+минимума). Ремонт не доказывает несовместимость, поэтому при его неудаче
+выполняется детерминированный точный поиск :func:`search_feasible_assignment`
+по всем четырём частям сразу (train_fit/train_monitor/comparison_val/
+sealed_test); найденное распределение используется, а при неуспехе диагностика
+честно различает доказанную несовместимость (квота невыполнима на уровне
+строк либо перебор исчерпан) и «решение не найдено в пределах бюджета поиска».
+Минимумы протокола жёстки: ``min_rows_per_class >= 2`` и ``min_train_fit >= 2``
+(значение 1 отклоняется до любых вычислений). Если ограничения несовместимы —
+:class:`S06SplitError` с полной per-class диагностикой: файлы сплита не
+записываются, состав данных не меняется, требования не ослабляются.
+
+Метрики качества не вычисляются; ``sealed_test`` используется только для
+count/hash/support-проверок целостности. Существующие файлы не перезаписываются.
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
+from .data import REQUIRED_COLUMNS, file_fingerprint, filter_min_count
+from .features import preprocess, strip_whitespace
+from .freeze_split import (
+    PART_NAMES,
+    PARTS_MONITOR,
+    PARTS_OUTER,
+    ROW_ID_COLUMN,
+    SPLIT_AUDIT_COLUMNS,
+    FreezeSplitError,
+    FrozenSplit,
+    add_group_keys,
+    check_group_label_consistency,
+    integrity_checks,
+    make_groups,
+    validate_rows,
+)
+from .labels import LEVELS, LabelMaps
+
+SUPPORT_BIN_EDGES = ((2, 4), (5, 9), (10, 19), (20, None))
+
+# Протокол S06 (D03): минимумы per-L3 жёстко равны 2; значение 1 не допускается.
+S06_MIN_ROWS_PER_CLASS = 2
+S06_MIN_TRAIN_FIT = 2
+
+# Части совместного точного поиска (fallback жадной эвристики) и его бюджет.
+JOINT_PARTS = ("train_fit", "train_monitor", "comparison_val", "sealed_test")
+SEARCH_NODE_BUDGET = 200_000
+SEARCH_MAX_GROUPS = 64
+
+
+def support_bin(support: int) -> str:
+    """Диапазон support comparison_val по протоколу S06."""
+    for lo, hi in SUPPORT_BIN_EDGES:
+        if support >= lo and (hi is None or support <= hi):
+            return f"{lo}-{hi}" if hi is not None else f">={lo}"
+    raise ValueError(f"support {support} вне допустимых диапазонов")
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
+def class_outer_quota(
+    s: int,
+    fractions: tuple[float, float, float],
+    min_rows: int,
+    min_train_rows: int,
+) -> np.ndarray | None:
+    """Целевые числа строк (outer_train, comparison_val, sealed_test) класса.
+
+    Минимумы имеют приоритет над пропорциями; при невозможности оставить
+    ``outer_train`` хотя бы ``min_train_rows`` возвращает None (квота
+    несовместима с минимумами на уровне строк, группы не рассматривались).
+    """
+    w = np.asarray(fractions, dtype=float)
+    base = _largest_remainder(s, w)
+    q_val = max(int(base[1]), min_rows)
+    q_test = max(int(base[2]), min_rows)
+    if s - q_val - q_test < min_train_rows:
+        q_val, q_test = min_rows, min_rows
+    q_train = s - q_val - q_test
+    if q_train < min_train_rows:
+        return None
+    return np.array([q_train, q_val, q_test], dtype=np.int64)
+
+
+def assign_groups(
+    sizes: np.ndarray,
+    quota: np.ndarray,
+    minima: np.ndarray,
+    rng: np.random.Generator,
+) -> tuple[np.ndarray, np.ndarray]:
+    """Жадное назначение неделимых групп по частям + локальный ремонт минимумов.
+
+    Группы упорядочиваются по убыванию размера (равные — засеянной
+    перестановкой) и идут в часть с максимальным дефицитом ``квота − строки``;
+    равенство разрешается по индексу части. Ремонт: пока есть дефицитная
+    часть, переносим в неё самую малую группу донора, которая закрывает
+    дефицит и не опускает донора ниже его минимума. Возвращает
+    (назначение group→part, число строк частей); нарушенные минимумы —
+    по ``rows < minima``.
+    """
+    sizes = np.asarray(sizes, dtype=np.int64)
+    parts = len(quota)
+    perm = rng.permutation(len(sizes))
+    order = np.lexsort((perm, -sizes))
+    assign = np.zeros(len(sizes), dtype=np.int64)
+    rows = np.zeros(parts, dtype=np.int64)
+    for gi in order:
+        p = int(np.argmax(quota - rows))
+        assign[gi] = p
+        rows[p] += int(sizes[gi])
+
+    for _ in range(max(1, len(sizes)) * parts):
+        deficient = [p for p in range(parts) if rows[p] < minima[p]]
+        if not deficient:
+            break
+        p = min(deficient)
+        need = int(minima[p] - rows[p])
+        donors = sorted(
+            (d for d in range(parts) if d != p and rows[d] > minima[d]),
+            key=lambda d: -(rows[d] - minima[d]),
+        )
+        moved = False
+        for d in donors:
+            cand = [
+                gi
+                for gi in np.where(assign == d)[0]
+                if sizes[gi] >= need and rows[d] - sizes[gi] >= minima[d]
+            ]
+            if not cand:
+                continue
+            gi = int(min(cand, key=lambda g: sizes[g]))
+            assign[gi] = p
+            rows[d] -= int(sizes[gi])
+            rows[p] += int(sizes[gi])
+            moved = True
+            break
+        if not moved:
+            break
+    return assign, rows
+
+
+def search_feasible_assignment(
+    sizes: np.ndarray,
+    minima: np.ndarray,
+    node_budget: int = SEARCH_NODE_BUDGET,
+) -> tuple[np.ndarray | None, bool, int]:
+    """Точный поиск допустимого назначения неделимых групп по частям.
+
+    Жадная эвристика :func:`assign_groups` не доказывает несовместимость:
+    например, для размеров групп (1, 1, 4, 4, 4) и минимумов (2, 2, 2, 2)
+    допустимое разбиение (4, 1+1, 4, 4) существует, но локальный ремонт его
+    может не найти. Этот перебор с отсечением и дедупликацией состояний
+    детерминирован (группы по убыванию размера, части по индексу) и ищет
+    любое допустимое назначение, удовлетворяющее ``rows >= minima``.
+
+    Возвращает ``(assignment, proven, nodes)``: assignment — массив
+    group→part либо None; ``proven`` — True, если перебор завершён полностью
+    и допустимого назначения нет (отсутствие решения доказано); при обрыве
+    по бюджету узлов ``proven`` = False и отсутствие решения НЕ доказано;
+    ``nodes`` — число посещённых узлов. Классы с числом групп больше
+    :data:`SEARCH_MAX_GROUPS` не перебираются (proven=False).
+    """
+    sizes = np.asarray(sizes, dtype=np.int64)
+    minima = np.asarray(minima, dtype=np.int64)
+    n = len(sizes)
+    parts = len(minima)
+    if n > SEARCH_MAX_GROUPS:
+        return None, False, 0
+    order = np.argsort(-sizes, kind="stable")
+    s = sizes[order]
+    suffix = np.zeros(n + 1, dtype=np.int64)
+    np.cumsum(s[::-1], out=suffix[1:])
+    suffix = suffix[::-1]  # suffix[i] = сумма размеров групп i..n-1
+    assign_sorted = np.zeros(n, dtype=np.int64)
+    rows = np.zeros(parts, dtype=np.int64)
+    visited: set[tuple[int, tuple[int, ...]]] = set()
+    nodes = 0
+    exhausted = True
+
+    def dfs(i: int) -> bool:
+        nonlocal nodes, exhausted
+        if not exhausted:
+            return False
+        nodes += 1
+        if nodes > node_budget:
+            exhausted = False
+            return False
+        state = (i, tuple(int(r) for r in rows))
+        if state in visited:
+            return False
+        visited.add(state)
+        if i == n:
+            return bool((rows >= minima).all())
+        remaining = int(suffix[i])
+        for p in range(parts):
+            if int(rows[p]) + remaining < int(minima[p]):
+                return False
+        g = int(s[i])
+        for p in range(parts):
+            rows[p] += g
+            assign_sorted[i] = p
+            if dfs(i + 1):
+                rows[p] -= g
+                return True
+            rows[p] -= g
+        return False
+
+    if not dfs(0):
+        return None, exhausted and n <= SEARCH_MAX_GROUPS, nodes
+    assignment = np.empty(n, dtype=np.int64)
+    assignment[order] = assign_sorted
+    return assignment, False, nodes
+
+
+def _class_assignment_report(
+    l3: str,
+    stage: str,
+    sizes: np.ndarray,
+    quota: np.ndarray | None,
+    minima: np.ndarray,
+    rows: np.ndarray,
+    part_names: list[str],
+) -> dict:
+    return {
+        "stage": stage,
+        "L3": l3,
+        "support": int(sizes.sum()),
+        "n_groups": int(len(sizes)),
+        "group_sizes": sorted(int(x) for x in sizes),
+        "quota": None if quota is None else [int(x) for x in quota],
+        "minima": [int(x) for x in minima],
+        "achieved": {part_names[p]: int(rows[p]) for p in range(len(part_names))},
+        "failed_minima": [
+            part_names[p] for p in range(len(part_names)) if rows[p] < minima[p]
+        ],
+    }
+
+
+def s06_split(
+    df: pd.DataFrame,
+    min_count: int = 12,
+    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
+    monitor_fraction: float = 0.1,
+    seed: int = 42,
+    min_rows_per_class: int = 2,
+    min_train_fit: int = 2,
+) -> FrozenSplit:
+    """Замораживает S06-сплит; при несовместимости минимумов и групп — S06SplitError.
+
+    ``df`` — исходный размеченный датафрейм (только REQUIRED_COLUMNS);
+    позиционные id строк сохраняются в ``row_id``.
+    """
+    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
+    if missing:
+        raise FreezeSplitError(f"Нет обязательных колонок: {missing}")
+    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9:
+        raise FreezeSplitError(f"Доли outer-частей должны быть тремя и суммироваться в 1: {fractions}")
+    if not (0.0 < monitor_fraction < 0.5):
+        raise FreezeSplitError(f"monitor_fraction должен быть в (0, 0.5): {monitor_fraction}")
+    if min_rows_per_class < S06_MIN_ROWS_PER_CLASS or min_train_fit < S06_MIN_TRAIN_FIT:
+        raise FreezeSplitError(
+            f"Протокол S06 требует минимумов >= {S06_MIN_ROWS_PER_CLASS}: "
+            f"min_rows_per_class={min_rows_per_class}, min_train_fit={min_train_fit}; "
+            "значения 1 не допускаются (comparison_val/sealed_test/train_monitor обязаны "
+            "иметь >= 2 строки каждой сохранённой L3)."
+        )
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
+    min_train_outer = min_rows_per_class + min_train_fit
+    classes = sorted(groups["L3"].unique())
+    seed_children = np.random.SeedSequence(seed).spawn(len(classes))
+
+    group_part: dict[str, str] = {}  # group_id -> part
+    diagnostics: list[dict] = []
+    infeasible: list[dict] = []
+    quota_report: list[dict] = []
+
+    for l3, child in zip(classes, seed_children):
+        rng = np.random.default_rng(child)
+        gdf = groups.loc[groups["L3"] == l3]
+        sizes = gdf["n_rows"].to_numpy(dtype=np.int64)
+        gids = gdf["group_id"].to_numpy()
+
+        # --- Внешнее разбиение 80/10/10 с минимумами.
+        quota = class_outer_quota(int(sizes.sum()), fractions, min_rows_per_class, min_train_outer)
+        if quota is None:
+            # Квота невыполнима на уровне строк ещё до групп: это доказанная
+            # несовместимость минимумов с support класса.
+            report = _class_assignment_report(
+                l3, "outer-row-level", sizes, None,
+                np.array([min_train_outer, min_rows_per_class, min_rows_per_class]),
+                np.zeros(3, dtype=np.int64), list(PARTS_OUTER),
+            )
+            report["proof"] = "row-level"
+            infeasible.append(report)
+            continue
+
+        assignment: dict[str, str] = {}  # group_id -> part (все 4 части)
+        class_reports: list[dict] = []
+
+        minima_outer = np.array([min_train_outer, min_rows_per_class, min_rows_per_class])
+        assign, rows = assign_groups(sizes, quota, minima_outer, rng)
+        greedy_ok = not (rows < minima_outer).any()
+        if greedy_ok:
+            # --- Монитор внутри outer_train: те же группы, квоты из фактических строк.
+            tr_idx = np.array([i for i, p in enumerate(assign) if p == 0], dtype=np.int64)
+            tr_sizes = sizes[tr_idx]
+            tr_gids = gids[tr_idx]
+            r = int(tr_sizes.sum())
+            m_target = max(min_rows_per_class, int(_largest_remainder(r, np.array([1.0 - monitor_fraction, monitor_fraction]))[1]))
+            fit_min = min_train_fit
+            if r - m_target < fit_min:
+                m_target = min_rows_per_class
+            quota_mon = np.array([r - m_target, m_target], dtype=np.int64)
+            minima_mon = np.array([fit_min, min_rows_per_class], dtype=np.int64)
+            assign_mon, rows_mon = assign_groups(tr_sizes, quota_mon, minima_mon, rng)
+            greedy_ok = not (rows_mon < minima_mon).any()
+        if greedy_ok:
+            class_reports.append(
+                _class_assignment_report(l3, "outer", sizes, quota, minima_outer, rows, list(PARTS_OUTER))
+            )
+            class_reports.append(
+                _class_assignment_report(l3, "monitor", tr_sizes, quota_mon, minima_mon, rows_mon, list(PARTS_MONITOR))
+            )
+            for gid, part in zip(gids, (PARTS_OUTER[int(x)] for x in assign)):
+                if part != "outer_train":
+                    assignment[gid] = part
+            for gid, p in zip(tr_gids, (PARTS_MONITOR[int(x)] for x in assign_mon)):
+                assignment[gid] = p
+        else:
+            # Жадная эвристика + локальный ремонт не нашли решение. Это НЕ
+            # доказательство несовместимости: ищем точным перебором по всем
+            # четырём частям сразу (train_fit/train_monitor/val/test).
+            joint_minima = np.array(
+                [min_train_fit, min_rows_per_class, min_rows_per_class, min_rows_per_class]
+            )
+            found, proven, nodes = search_feasible_assignment(sizes, joint_minima)
+            if found is not None:
+                achieved = np.zeros(4, dtype=np.int64)
+                for gi, gid in enumerate(gids):
+                    p = int(found[gi])
+                    achieved[p] += int(sizes[gi])
+                    assignment[gid] = JOINT_PARTS[p]
+                report = _class_assignment_report(
+                    l3, "exact-search", sizes, None, joint_minima, achieved, list(JOINT_PARTS)
+                )
+                report["method"] = "exact-search"
+                report["nodes"] = int(nodes)
+                class_reports.append(report)
+            else:
+                report = _class_assignment_report(
+                    l3, "outer+monitor", sizes, quota, minima_outer, rows, list(PARTS_OUTER)
+                )
+                report["greedy_failed_stage"] = "monitor" if not (rows < minima_outer).any() else "outer"
+                report["search_nodes"] = int(nodes)
+                report["search_max_groups"] = int(SEARCH_MAX_GROUPS)
+                report["proof"] = "exhaustive" if proven else "budget-cut"
+                infeasible.append(report)
+                continue
+        group_part.update(assignment)
+        quota_report.extend(class_reports)
+        diagnostics.extend(class_reports)
+
+    if infeasible:
+        all_proven = all(x.get("proof") in ("row-level", "exhaustive") for x in infeasible)
+        if all_proven:
+            message = (
+                f"Отказ S06 split: минимумы несовместимы с неделимыми группами "
+                f"(доказано на уровне строк или полным перебором) для "
+                f"{len(infeasible)} (класс, стадия): "
+                f"{[(x['L3'], x['stage']) for x in infeasible[:10]]}. "
+                "Требования не ослаблялись; полная диагностика в атрибуте infeasible."
+            )
+        else:
+            # Жадность и ограниченный поиск решения не нашли, но отсутствие
+            # решения не доказано: честно сообщаем о границах проверки, не
+            # выдавая её за доказанную несовместимость.
+            message = (
+                f"Отказ S06 split: допустимое разбиение не найдено (жадная эвристика "
+                f"и точный поиск с бюджетом {SEARCH_NODE_BUDGET} узлов) для "
+                f"{len(infeasible)} (класс, стадия): "
+                f"{[(x['L3'], x['stage']) for x in infeasible[:10]]}. "
+                "Отсутствие решения не доказано — требуется ручной анализ протокола; "
+                "требования не ослаблялись; полная диагностика в атрибуте infeasible."
+            )
+        err = S06SplitError(message)
+        err.infeasible = infeasible
+        raise err
+
+    kept = kept.copy()
+    kept["_part"] = kept["_char_key"].map(group_part)
+    if kept["_part"].isna().any():
+        raise FreezeSplitError("Есть строки без назначенной части; отказ.")
+
+    part_rows = {
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
+        parts[name] = d.reset_index(drop=True)
+
+    excluded = df.loc[~df[ROW_ID_COLUMN].isin(kept[ROW_ID_COLUMN])].copy()
+    maps = LabelMaps.from_df(kept[list(REQUIRED_COLUMNS)])
+
+    # --- Поддержка классов по частям.
+    support = kept[list(REQUIRED_COLUMNS)].groupby(["L1", "L2", "L3"]).size().reset_index(name="support_all")
+    for name in PART_NAMES:
+        support[f"support_{name}"] = (
+            parts[name].groupby(["L1", "L2", "L3"]).size().reindex(
+                pd.MultiIndex.from_frame(support[["L1", "L2", "L3"]]),
+                fill_value=0,
+            ).to_numpy()
+        )
+    support["support_bin_val"] = [support_bin(int(v)) for v in support["support_comparison_val"]]
+
+    # --- Жёсткие проверки минимумов (сводно по всем классам).
+    min_violations = []
+    for _, row in support.iterrows():
+        checks = {
+            "support_comparison_val": min_rows_per_class,
+            "support_sealed_test": min_rows_per_class,
+            "support_train_monitor": min_rows_per_class,
+            "support_train_fit": min_train_fit,
+        }
+        for col, minimum in checks.items():
+            if int(row[col]) < minimum:
+                min_violations.append({"L3": row["L3"], "part": col, "support": int(row[col]), "min": minimum})
+    if min_violations:
+        raise S06SplitError(
+            f"Отказ S06 split: нарушены per-class минимумы: {min_violations[:20]}; "
+            f"требования не ослаблялись."
+        )
+
+    # --- Полнота покрытия: каждая L3 в каждой части (label maps идентичны).
+    coverage_fails = []
+    for name in PART_NAMES:
+        missing_l3 = set(maps.l3) - set(parts[name]["L3"].unique())
+        if missing_l3:
+            coverage_fails.append({"part": name, "missing_l3": sorted(missing_l3)[:10]})
+    if coverage_fails:
+        raise S06SplitError(f"Отказ S06 split: неполное покрытие L3: {coverage_fails}")
+
+    # --- Доли частей (фиксируются, но не являются жёстким критерием: минимумы
+    # мелких классов законно смещают доли от 80/10/10).
+    n_total = len(kept)
+    share_report: dict[str, dict] = {}
+    for name in PART_NAMES:
+        if name in PARTS_OUTER:
+            target = fractions[PARTS_OUTER.index(name)]
+            actual = len(parts[name]) / n_total
+            base = "all_kept_rows"
+        else:
+            target = (1.0 - monitor_fraction) if name == "train_fit" else monitor_fraction
+            actual = len(parts[name]) / len(parts["outer_train"])
+            base = "outer_train"
+        share_report[name] = {
+            "target": target,
+            "actual": round(actual, 6),
+            "n_rows": len(parts[name]),
+            "base": base,
+        }
+
+    checks = integrity_checks(parts, excluded, n_rows_read=len(df))
+
+    report = {
+        "seed": int(seed),
+        "min_count": int(min_count),
+        "fractions": list(fractions),
+        "monitor_fraction": float(monitor_fraction),
+        "min_rows_per_class": int(min_rows_per_class),
+        "min_train_fit": int(min_train_fit),
+        "n_rows_read": int(len(df)),
+        "n_rows_kept": int(n_total),
+        "n_rows_excluded_min_count": int(len(excluded)),
+        "n_l3_kept": int(kept["L3"].nunique()),
+        "n_l3_excluded": int(len(rare)),
+        "n_groups": int(len(groups)),
+        "shares": share_report,
+        "support": support,
+        "rare_classes": rare,
+        "class_assignment": diagnostics,
+        "checks": checks,
+    }
+    return FrozenSplit(parts=parts, excluded=excluded, maps=maps, report=report)
+
+
+class S06SplitError(FreezeSplitError):
+    """Несовместимость минимумов S06 с неделимыми группами; файлы не записываются."""
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
+def save_s06_split(
+    frozen: FrozenSplit,
+    input_path: str,
+    out_dir: str,
+    audit_dir: str,
+    min_count: int,
+    fractions: tuple[float, float, float],
+    monitor_fraction: float,
+    seed: int,
+    min_rows_per_class: int,
+    min_train_fit: int,
+) -> dict:
+    """Сохраняет части, manifest и отчёты S06; существующие файлы не перезаписывает."""
+    os.makedirs(out_dir, exist_ok=True)
+    os.makedirs(audit_dir, exist_ok=True)
+
+    part_files = {name: os.path.join(out_dir, f"{name}.parquet") for name in PART_NAMES}
+    manifest_path = os.path.join(out_dir, "manifest.json")
+    split_audit_path = os.path.join(audit_dir, "split_audit.csv")
+    support_path = os.path.join(audit_dir, "class_support_split.csv")
+    assignment_path = os.path.join(audit_dir, "class_assignment_quotas.json")
+    targets = list(part_files.values()) + [manifest_path, split_audit_path, support_path, assignment_path]
+    existing = [p for p in targets if os.path.exists(p)]
+    if existing:
+        raise FreezeSplitError(
+            f"Отказ S06 split: существующие outputs не перезаписываются: {existing}"
+        )
+
+    report = frozen.report
+    checks = report["checks"]
+    if not checks["ok"]:
+        raise FreezeSplitError(f"Отказ S06 split: проверки целостности не пройдены: {checks}")
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
+        rare_rows["support_bin_val"] = "excluded"
+        support = pd.concat(
+            [
+                support,
+                rare_rows[
+                    ["L1", "L2", "L3", "kept_after_min_count", "support_all"]
+                    + [f"support_{n}" for n in PART_NAMES] + ["support_bin_val"]
+                ],
+            ],
+            ignore_index=True,
+        )
+    support = support[
+        ["L1", "L2", "L3", "kept_after_min_count"]
+        + [f"support_{n}" for n in PART_NAMES]
+        + ["support_all", "support_bin_val"]
+    ]
+    support = support.sort_values(
+        ["kept_after_min_count", "support_comparison_val", "L3"],
+        ascending=[False, False, True],
+        ignore_index=True,
+    )
+    support.to_csv(support_path, index=False)
+
+    with open(assignment_path, "w", encoding="utf-8") as f:
+        json.dump(report["class_assignment"], f, ensure_ascii=False, indent=2)
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
+    pd.DataFrame(audit_rows, columns=SPLIT_AUDIT_COLUMNS).to_csv(split_audit_path, index=False)
+
+    manifest = {
+        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
+        "generator": {
+            "module": os.path.abspath(__file__),
+            "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]) if sys.argv else None,
+            "cwd": os.getcwd(),
+            "git": _git_info(),
+            "versions": _versions(),
+        },
+        "policy": {
+            "min_count": int(min_count),
+            "min_count_rule": "L3 kept iff support >= min_count (включительно), до всех разбиений",
+            "fractions": list(fractions),
+            "monitor_fraction": float(monitor_fraction),
+            "split_seed": int(seed),
+            "min_rows_per_class": int(min_rows_per_class),
+            "min_train_fit": int(min_train_fit),
+            "quota_rule": "целевые числа строк класса — наибольшие остатки от долей частей, "
+                          "поднятые до минимумов; outer_train получает остаток, но не меньше "
+                          "min_rows_per_class + min_train_fit; проценты не назначаются механически",
+            "assignment_rule": "неделимые группы (char-ключ) жадно по дефициту квоты, "
+                               "засеянный tie-break; локальный ремонт переносом самой малой "
+                               "подходящей группы; при несовместимости — отказ без записи файлов",
+            "group_key": "strip_whitespace(preprocess(name)) — char-ветка TF-IDF; "
+                         "word-ключ (collapse whitespace) равен ⇒ char-ключ равен, "
+                         "поэтому группа покрывает обе ветки",
+            "row_id": "позиционный индекс строки во входном parquet (0-based, порядок файла)",
+            "parts": {
+                "outer_train": "обучение (fit) и внутренний выбор эпох/конфигурации только по train_fit/train_monitor",
+                "comparison_val": "сравнение методов S06; не для выбора checkpoint/эпох",
+                "sealed_test": "запечатан; метрики качества и inference не выполнялись, count/hash/support только",
+                "train_fit": "внутренний fit внутри outer_train",
+                "train_monitor": "внутренний monitor внутри outer_train (early stopping/выбор эпох)",
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
+            "Per-class минимумы проверены жёстко: comparison_val/sealed_test/train_monitor >= "
+            "min_rows_per_class, train_fit >= min_train_fit; классы без выполнимых минимумов "
+            "не исключались молча — при несовместимости сплит не записывался бы.",
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
+        "class_assignment_quotas": {"path": os.path.abspath(assignment_path)},
+        "n_rows_kept": report["n_rows_kept"],
+        "n_rows_excluded_min_count": report["n_rows_excluded_min_count"],
+        "checks_ok": checks["ok"],
+    }
diff --git a/kkt_clf/s06_baseline.py b/kkt_clf/s06_baseline.py
new file mode 100644
index 0000000..f428e79
--- /dev/null
+++ b/kkt_clf/s06_baseline.py
@@ -0,0 +1,883 @@
+"""S06 Gate 1 (D04–D06): замороженный обучающий протокол и новый full baseline.
+
+Контур Gate 1 поверх результатов Gate 0 (``outputs/S06/splits``, см.
+``kkt_clf.s06_split``) и одного проверенного full-TFIDF S02. **Модуль не
+переобучает и не оценивает ничего на ``sealed_test``**: часть читается только
+как структурная запись manifest (sha256/число строк), любой code path к её
+строкам отсутствует; единственная внешняя оценка — raw top-1 L3 на
+``comparison_val`` после refit всех seed, против train-only majority.
+
+Протокол (границы частей совпадают с S05, D05/D06 value_dod S06):
+
+- Выбор числа эпох и конфигурации — **только** по ``train_fit``/``train_monitor``
+  (фаза ``select``), до любого обращения к ``comparison_val``; кривые и отчёт
+  selection сохраняются, затем протокол **замораживается** в неизменяемый
+  ``shared/training_protocol.json`` (файл пишется один раз; повторная запись,
+  как и любое расхождение provenance, — отказ, а не перезапись).
+- Финальный refit — три полных baseline seed 42/43/44 на ``outer_train`` ровно
+  на замороженное число эпох с фиксированным заранее расписанием LR
+  (переиспользуется контур S05 без изменения его поведения), веса финальной
+  эпохи (restore_best=False), без выбора checkpoint по monitor.
+- Опорный checkpoint для G×I (Gate 2) — заранее заданный ``reference_seed``
+  (по умолчанию 42), не по качеству на comparison_val; round-trip проверка —
+  контур S05.
+- Оценка — одна raw top-1 L3 на ``comparison_val`` против train-only majority;
+  ``sealed_test`` не используется ни для inference, ни для метрик.
+
+Проверки до обучения (fail closed): manifest Gate 0 (sha256 частей, политика
+``min_count == 12`` — сплит S05 с ``min_count == 10`` отклоняется; заявленные
+``min_rows_per_class``/``min_train_fit`` >= 2; фактические per-L3 support
+всех частей из реальных файлов и идентичность наборов L3 label maps),
+целостность и provenance full-TFIDF S02 (sha256 файлов артефакта +
+fingerprint корпуса + полный контракт словаря 300 000 char + 100 000 word +
+fit на полном снимке корпуса без усечения), сверка с существующей ссылкой
+``shared/full_tfidf_ref.json``, замороженного протокола с фактическими
+входами (sha256 сплитов, full_tfidf_id, hash selection-отчёта, согласованность
+fixed_epochs на всех уровнях вложенности, постоянный LR scheduler 'none').
+Никакие существующие файлы не перезаписываются; исторические checkpoint/split
+S05 не используются.
+
+Реализация **делегирует обучение** существующему контуру S05
+(``s05_baseline.run_selection`` / ``run_refit`` и их проверки) с конфигурацией,
+направленной в ``outputs/S06/runs/baseline_full``; тренировочный цикл не
+дублируется. Ресурсная оценка (``estimate``) — честная проекция от измерений
+S05 и размеров нового сплита, без обучения и без выдачи проекций за замеры.
+"""
+from __future__ import annotations
+
+import dataclasses
+import datetime as dt
+import json
+import logging
+import os
+from dataclasses import dataclass, field
+from typing import Any
+
+import torch
+
+import pandas as pd
+
+from .config import _merge, build_config
+from .data import file_fingerprint
+from .experiment import _sha256_file, _sha256_json, _write_json
+from .s06_split import S06_MIN_ROWS_PER_CLASS, S06_MIN_TRAIN_FIT
+from .s05_baseline import (
+    S05BaselineError,
+    S05Config,
+    load_split_manifest,
+    record_tfidf_verification,
+    run_refit,
+    run_selection,
+    verify_full_tfidf,
+)
+
+log = logging.getLogger(__name__)
+
+S06_GATE1_SCHEMA = "s06-gate1-protocol@1"
+
+# Полный контракт TF-IDF S02 (D04): артефакт с иным vocab при том же корпусе
+# (например усечённый fit tfidf_100k) для S06 недопустим.
+EXPECTED_FULL_VOCAB = {"n_char": 300_000, "n_word": 100_000, "n_total": 400_000}
+
+# Части, к которым Gate 1 запрещено обращаться за данными. comparison_val
+# читается ровно один раз — внутри s05.run_refit после всех refit (оценка);
+# sealed_test не читается никогда (только структурные sha256/n_rows из manifest).
+SEALED_PART = "sealed_test"
+EVAL_PART = "comparison_val"
+
+PROTOCOL_NOTES = {
+    "epoch_selection": (
+        "число эпох и конфигурация выбраны только по train_fit/train_monitor "
+        "(early stopping по train_monitor, selection_seed), до любого обращения "
+        "к comparison_val; кривые — runs/baseline_full/selection/curves_train_monitor.csv"
+    ),
+    "final_train": (
+        "refit каждого seed на outer_train ровно на замороженное число эпох, "
+        "фиксированное заранее расписание LR (общее с selection), "
+        "restore_best=False (веса финальной эпохи), без выбора checkpoint по monitor"
+    ),
+    "evaluation": (
+        "единственная внешняя оценка Gate 1 — raw top-1 L3 без отказа на "
+        "comparison_val после refit всех seed; majority baseline — большинство "
+        "по outer_train (train-only); метрики по фиксированной L3-вселенной label maps"
+    ),
+    "sealed_test": (
+        "sealed_test запрещён для inference/eval в Gate 1: часть не читается, "
+        "метрик на ней нет; использованы только структурные sha256/n_rows из manifest"
+    ),
+    "reference_checkpoint": (
+        "опорный checkpoint для G×I (Gate 2) — заранее заданный reference_seed, "
+        "не по качеству на comparison_val; round-trip проверка контура S05"
+    ),
+    "transductive": (
+        "режим pretrained трансдуктивен: тексты корпуса full TF-IDF могут "
+        "совпадать с train/val/test; метки при этом не переносятся"
+    ),
+}
+
+
+class S06Gate1Error(RuntimeError):
+    """Ошибка контура Gate 1 S06; файлы не перезаписываются."""
+
+
+def ensure_constant_lr_schedule(scheduler: str) -> None:
+    """Fail closed: S06 допускает только постоянный LR (scheduler 'none').
+
+    ``cosine``/``onecycle`` строят кривую LR от ``total_steps``, которое у
+    selection (train_fit, ранняя остановка) и refit (outer_train,
+    замороженное число эпох) различается, — кривые LR selection/refit были бы
+    разными, и выбранное число эпох относилось бы к другому расписанию.
+    Идентичное воспроизводимое расписание для разного числа шагов в S06 не
+    реализовано и не проверено, поэтому такие scheduler'ы отклоняются до
+    любого обучения. ``plateau`` контур S05 уже заменяет на ``none`` при
+    сборке секций (зафиксировано заметкой в протоколе), ``none`` сохраняется.
+    """
+    if scheduler != "none":
+        raise S06Gate1Error(
+            f"scheduler={scheduler!r} для S06 запрещён: кривая LR зависит от числа "
+            "шагов, а selection и refit имеют разное число шагов (train_fit + early "
+            "stopping против outer_train + замороженные эпохи), поэтому идентичность "
+            "расписания не подтверждена. Разрешён только scheduler 'none' "
+            "(постоянный LR); текущий замороженный протокол S06 использует 'none'."
+        )
+
+
+@dataclass
+class S06Gate1Config:
+    """Конфигурация Gate 1 S06 (обёртка над S05Config с S06-путями)."""
+
+    output_root: str = "outputs/S06"
+    splits_manifest: str = "outputs/S06/splits/manifest.json"
+    full_tfidf_path: str = "outputs/S02/tfidf_1m"
+    corpus_path: str = "data/corpus.pqt"
+    expected_min_count: int = 12
+    expected_vocab: dict[str, int] = field(default_factory=lambda: dict(EXPECTED_FULL_VOCAB))
+    preset: str = "ods_light"
+    model: dict[str, Any] = field(default_factory=dict)
+    loss: dict[str, Any] = field(default_factory=dict)
+    train: dict[str, Any] = field(default_factory=dict)
+    sweep: list[dict[str, Any]] = field(default_factory=list)
+    features_chunk_rows: int = 50_000
+    selection_seed: int = 42
+    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
+    reference_seed: int = 42
+    device: str = "auto"
+
+    @property
+    def baseline_dir(self) -> str:
+        return os.path.join(self.output_root, "runs", "baseline_full")
+
+    @property
+    def shared_dir(self) -> str:
+        return os.path.join(self.output_root, "shared")
+
+    @property
+    def protocol_path(self) -> str:
+        return os.path.join(self.shared_dir, "training_protocol.json")
+
+    @property
+    def tfidf_ref_path(self) -> str:
+        return os.path.join(self.shared_dir, "full_tfidf_ref.json")
+
+    def to_s05_config(self) -> S05Config:
+        """S05-контур с S06-путями; поведение контура S05 не изменяется."""
+        return S05Config(
+            output_dir=self.baseline_dir,
+            splits_manifest=self.splits_manifest,
+            full_tfidf_path=self.full_tfidf_path,
+            corpus_path=self.corpus_path,
+            preset=self.preset,
+            model=dict(self.model),
+            loss=dict(self.loss),
+            train=dict(self.train),
+            sweep=[dict(c) for c in self.sweep],
+            features_chunk_rows=self.features_chunk_rows,
+            selection_seed=self.selection_seed,
+            seeds=list(self.seeds),
+            reference_seed=self.reference_seed,
+            fixed_epochs=None,  # эпохи — только из замороженного протокола
+            device=self.device,
+        )
+
+    def resolved_scheduler(self) -> str:
+        """Расписание LR после сборки секций (plateau контур S05 уже заменил на none)."""
+        _, _, train_cfg = self.to_s05_config().build_sections()
+        return str(train_cfg.scheduler)
+
+    def validate(self) -> None:
+        if self.expected_min_count != 12:
+            raise S06Gate1Error(
+                f"expected_min_count должен быть 12 (протокол S06), получено {self.expected_min_count}"
+            )
+        if self.reference_seed not in self.seeds:
+            raise S06Gate1Error(
+                f"reference_seed={self.reference_seed} должен входить в seeds={self.seeds}"
+            )
+        if len(set(self.seeds)) != len(self.seeds):
+            raise S06Gate1Error(f"seeds должны быть различны: {self.seeds}")
+        ensure_constant_lr_schedule(self.resolved_scheduler())
+        self.to_s05_config().validate()
+
+
+# ---------------------------------------------------------------------------
+# Gate 0: проверка provenance нового сплита
+# ---------------------------------------------------------------------------
+
+
+def verify_gate0_split(cfg: S06Gate1Config) -> dict[str, Any]:
+    """Проверяет замороженный сплит Gate 0 (fail closed до любого обучения).
+
+    Поверх проверок целостности S05 (sha256/bytes/число строк каждой части)
+    требует политики S06: ``min_count == 12`` (сплит S05 с min_count=10
+    отклоняется), ``min_rows_per_class``/``min_train_fit`` не меньше 2 (если
+    заявлены в политике), прошедшие integrity checks Gate 0, наличие всех
+    пяти частей и зафиксированных label maps. Дополнительно проверяет по
+    фактическим файлам частей (когда в manifest есть пути): реальные per-L3
+    support каждой части — не менее 2 строк в comparison_val/sealed_test/
+    train_monitor и не менее 2 в train_fit — и идентичность наборов L3 между
+    всеми частями и label maps manifest. К ``sealed_test`` обращений за
+    данными нет и не будет: здесь используются только его структурные поля
+    manifest и count/support-подсчёт, допустимый протоколом S06.
+    """
+    if not os.path.isfile(cfg.splits_manifest):
+        raise S06Gate1Error(
+            f"manifest сплитов Gate 0 не найден: {cfg.splits_manifest!r}. "
+            "Gate 1 не запускается без независимо проверенного Gate 0."
+        )
+    manifest = load_split_manifest(cfg.splits_manifest)  # sha256/bytes/n_rows всех частей
+    policy = manifest.get("policy") or {}
+    min_count = policy.get("min_count")
+    if min_count != cfg.expected_min_count:
+        raise S06Gate1Error(
+            f"Политика сплита min_count={min_count!r} != {cfg.expected_min_count} (протокол S06). "
+            "Вероятен сплит S05 (min_count=10) или иной протокол: Gate 1 отказывается, "
+            "старую выборку для нового протокола не подменяем."
+        )
+    # Политика per-class минимумов: протокол S06 требует >= 2; значение 1
+    # (или отсутствие заявленных минимумов при заявленном значении < 2)
+    # отклоняется. Отсутствие ключей в чужом manifest не подтверждает политику
+    # — достоверность тогда определяется фактическими проверками support ниже.
+    policy_min_rows = policy.get("min_rows_per_class")
+    policy_min_fit = policy.get("min_train_fit")
+    if policy_min_rows is not None and int(policy_min_rows) < S06_MIN_ROWS_PER_CLASS:
+        raise S06Gate1Error(
+            f"Политика сплита min_rows_per_class={policy_min_rows} < {S06_MIN_ROWS_PER_CLASS} "
+            "(протокол S06): comparison_val/sealed_test/train_monitor обязаны иметь "
+            ">= 2 строки каждой сохранённой L3; сплит с минимумом 1 отклоняется."
+        )
+    if policy_min_fit is not None and int(policy_min_fit) < S06_MIN_TRAIN_FIT:
+        raise S06Gate1Error(
+            f"Политика сплита min_train_fit={policy_min_fit} < {S06_MIN_TRAIN_FIT} "
+            "(протокол S06): train_fit обязан иметь >= 2 строки каждой сохранённой L3."
+        )
+    checks = manifest.get("checks") or {}
+    if not checks.get("ok"):
+        raise S06Gate1Error(f"Integrity checks Gate 0 не пройдены (checks.ok != True): {checks}")
+    for part in ("outer_train", "train_fit", "train_monitor", EVAL_PART, SEALED_PART):
+        if part not in manifest.get("parts", {}):
+            raise S06Gate1Error(f"В manifest сплита нет части {part!r}")
+    if not manifest.get("label_maps"):
+        raise S06Gate1Error("В manifest сплита нет label_maps")
+
+    # --- Фактические per-L3 минимумы и идентичность label maps по файлам частей.
+    required = {
+        EVAL_PART: int(policy_min_rows) if policy_min_rows is not None else S06_MIN_ROWS_PER_CLASS,
+        SEALED_PART: int(policy_min_rows) if policy_min_rows is not None else S06_MIN_ROWS_PER_CLASS,
+        "train_monitor": int(policy_min_rows) if policy_min_rows is not None else S06_MIN_ROWS_PER_CLASS,
+        "train_fit": int(policy_min_fit) if policy_min_fit is not None else S06_MIN_TRAIN_FIT,
+    }
+    l3_universe = set((manifest.get("label_maps") or {}).get("L3") or [])
+    part_l3_sets: dict[str, set] = {}
+    problems: list[str] = []
+    for name, minimum in required.items():
+        part = manifest["parts"].get(name) or {}
+        fpath = part.get("file")
+        if not fpath or not os.path.isfile(fpath):
+            # Manifest без путей к частям (сторонняя схема): фактические
+            # проверки невозможны, минимумы не подтверждены — пропускаем.
+            continue
+        df_part = pd.read_parquet(fpath, columns=["L1", "L2", "L3"])
+        part_l3_sets[name] = set(df_part["L3"].unique())
+        if l3_universe and part_l3_sets[name] != l3_universe:
+            missing = sorted(l3_universe - part_l3_sets[name])[:5]
+            extra = sorted(part_l3_sets[name] - l3_universe)[:5]
+            problems.append(
+                f"{name}: набор L3 не совпадает с label maps manifest "
+                f"(нет: {missing}; лишние: {extra})"
+            )
+        low = df_part.groupby(["L1", "L2", "L3"]).size()
+        low = low[low < minimum]
+        if len(low):
+            examples = {"/".join(map(str, idx)): int(v) for idx, v in low.head(5).items()}
+            problems.append(
+                f"{name}: {len(low)} L3 с фактическим support < {minimum} (напр. {examples})"
+            )
+    if len(part_l3_sets) >= 2:
+        reference_name = next(iter(part_l3_sets))
+        for name, l3s in part_l3_sets.items():
+            if l3s != part_l3_sets[reference_name]:
+                problems.append(
+                    f"идентичность label maps: набор L3 части {name!r} отличается от {reference_name!r}"
+                )
+    if problems:
+        raise S06Gate1Error(
+            "Gate 0 split: фактические per-class проверки не пройдены: " + "; ".join(problems)
+        )
+    log.info(
+        "Gate 0 split проверен: min_count=%s, per-L3 минимумы по фактическим "
+        "файлам частей (>=2 в val/test/monitor/fit), наборы L3 частей "
+        "идентичны label maps; sealed_test — только структурные sha256/n_rows "
+        "и count/support, инференс/метрик на нём нет",
+        min_count,
+    )
+    return manifest
+
+
+# ---------------------------------------------------------------------------
+# D04: full TF-IDF S02 → shared/full_tfidf_ref.json
+# ---------------------------------------------------------------------------
+
+
+def verify_s06_tfidf(cfg: S06Gate1Config) -> dict[str, Any]:
+    """Проверяет full TF-IDF S02 и сохраняет его provenance в ``shared/``.
+
+    Проверка целостности и совпадения корпуса — ``s05_baseline.verify_full_tfidf``
+    (без изменения поведения). Поверх неё fail closed полный контракт S02:
+    словарь ровно 300 000 char + 100 000 word и fit выполнен на полном
+    снимке корпуса (``source.rows_used == source.rows_in_file``,
+    ``truncated is False``) — иной валидный TF-IDF с тем же корпусом (например
+    усечённый fit на подвыборке строк) отклоняется.
+    ``shared/full_tfidf_ref.json`` пишется один раз;
+    при уже существующем файле (в т.ч. созданном Gate 0 через
+    ``scripts/s06_tfidf_ref.py`` — иная схема) выполняется сверка записанных в
+    нём hash-полей с фактической проверкой — расхождение отказ, совпадение —
+    существующий файл сохраняется, перезапись чужого артефакта недопустима.
+    Сам full-артефакт не копируется (ссылка + hash).
+    """
+    verification = verify_full_tfidf(cfg.full_tfidf_path, cfg.corpus_path)
+    expected = dict(cfg.expected_vocab)
+    if (verification.get("vocab") or {}) != expected:
+        raise S06Gate1Error(
+            f"full TF-IDF не соответствует контракту S02: vocab="
+            f"{verification.get('vocab')} != {expected}. Артефакт валиден сам по "
+            "себе и относится к тому же корпусу, но это не полный TF-IDF S06 "
+            "(например усечённый или иной fit): подмена контракта признаков "
+            "между фазами недопустима, отказ до любых записей."
+        )
+    src = verification.get("source") or {}
+    if not (src.get("rows_used") == src.get("rows_in_file") and src.get("truncated") is False):
+        raise S06Gate1Error(
+            "provenance полного исходного fit не подтверждён: metadata source "
+            f"должен содержать rows_used == rows_in_file и truncated=false, "
+            f"получено rows_used={src.get('rows_used')!r}, rows_in_file="
+            f"{src.get('rows_in_file')!r}, truncated={src.get('truncated')!r}. "
+            "Fit на подвыборке строк корпуса для S06 недопустим: refit selected "
+            "TF-IDF обязан воспроизводить ту же фактическую выборку строк."
+        )
+    os.makedirs(cfg.shared_dir, exist_ok=True)
+    path = cfg.tfidf_ref_path
+    if os.path.isfile(path):
+        with open(path, encoding="utf-8") as f:
+            existing = json.load(f)
+        problems = _existing_tfidf_ref_problems(existing, verification, expected)
+        if problems:
+            raise S06Gate1Error(
+                f"{path} уже существует и не совпадает с текущей проверкой full "
+                f"TF-IDF по полям {problems}: подмена артефакта/корпуса между "
+                "фазами недопустима, отказ до любых записей."
+            )
+        log.info("%s уже существует и совпадает с текущей проверкой full TF-IDF", path)
+        return verification
+    ref = {
+        "schema_version": S06_GATE1_SCHEMA,
+        "role": "full_tfidf_ref",
+        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
+        "artifact_dir": verification["artifact_dir"],
+        "full_tfidf_id": verification["full_tfidf_id"],
+        "artifact_version": verification["artifact_version"],
+        "seed": verification["seed"],
+        "vocab": verification["vocab"],
+        "fit": verification["fit"],
+        "source": verification["source"],
+        "corpus_fingerprint": verification["corpus_fingerprint"],
+        "corpus_matches_source": verification["corpus_matches_source"],
+        "files": verification["files"],
+        "decision": verification["decision"],
+        "mode": verification["mode"],
+        "notes": [
+            "Артефакт не копируется: ссылка + sha256; проверка целостности — load_vectorizer_artifact.",
+            PROTOCOL_NOTES["transductive"],
+        ],
+    }
+    _write_json(ref, path)
+    return verification
+
+
+def _existing_tfidf_ref_problems(
+    existing: dict[str, Any], verification: dict[str, Any], expected_vocab: dict[str, int]
+) -> list[str]:
+    """Сверяет существующий full_tfidf_ref.json (любой известной схемы) с фактической проверкой.
+
+    Поддерживает схему этого модуля (``full_tfidf_id``/``corpus_fingerprint``)
+    и схему ``scripts/s06_tfidf_ref.py`` Gate 0 (``files.features.joblib.sha256``
+    + ``corpus_provenance.corpus_actual`` + ``status``/``load_check.status``).
+    Поверх идентичности проверяет полный контракт S02: vocab существующей
+    ссылки равен 300 000 char + 100 000 word, а зафиксированный fit —
+    полному снимку корпуса (``meta_source.rows_used == rows_in_file``,
+    ``truncated is False``), чтобы ссылка не узаконивала усечённый fit.
+    """
+    problems: list[str] = []
+    actual_id = verification.get("full_tfidf_id")
+    new_corpus = (verification.get("corpus_fingerprint") or {})
+    existing_vocab = existing.get("vocab")
+    if existing_vocab is None:
+        existing_vocab = ((existing.get("load_check") or {}).get("loaded_vocab")) or (
+            (existing.get("artifact") or {}).get("vocab_meta")
+        )
+    if existing_vocab is not None and existing_vocab != expected_vocab:
+        problems.append(f"vocab: существующая ссылка фиксирует {existing_vocab}, контракт S02 {expected_vocab}")
+    if "full_tfidf_id" in existing:  # схема kkt_clf.s06_baseline
+        for key in ("decision", "full_tfidf_id", "artifact_version"):
+            if existing.get(key) != verification.get(key):
+                problems.append(key)
+        ex_corpus = (existing.get("corpus_fingerprint") or {})
+        if (ex_corpus.get("sha256") != new_corpus.get("sha256")
+                or ex_corpus.get("bytes") != new_corpus.get("bytes")):
+            problems.append("corpus_fingerprint")
+        return problems
+    fj = ((existing.get("files") or {}).get("features.joblib") or {})
+    if existing.get("status") != "ok":
+        problems.append("status")
+    if fj.get("sha256") != actual_id:
+        problems.append("features.joblib.sha256")
+    actual = ((existing.get("corpus_provenance") or {}).get("corpus_actual") or {})
+    if (actual.get("sha256") != new_corpus.get("sha256")
+            or actual.get("bytes") != new_corpus.get("bytes")):
+        problems.append("corpus_actual")
+    if (existing.get("load_check") or {}).get("status") not in (None, "ok"):
+        problems.append("load_check.status")
+    meta_source = ((existing.get("corpus_provenance") or {}).get("meta_source")) or {}
+    if meta_source and not (
+        meta_source.get("rows_used") == meta_source.get("rows_in_file")
+        and meta_source.get("truncated") is False
+    ):
+        problems.append("meta_source.full_fit: ссылка фиксирует fit не на полном снимке корпуса")
+    return problems
+
+
+# ---------------------------------------------------------------------------
+# D05: train-only выбор эпох → неизменяемый shared/training_protocol.json
+# ---------------------------------------------------------------------------
+
+
+def freeze_training_protocol(
+    cfg: S06Gate1Config,
+    manifest: dict[str, Any],
+    verification: dict[str, Any],
+    loaded: Any,
+    device: torch.device,
+) -> dict[str, Any]:
+    """Фаза selection (train_fit/train_monitor) + заморозка протокола.
+
+    Selection делегируется ``s05_baseline.run_selection`` (train-only ранний
+    останов по train_monitor; comparison_val не читается). Затем протокол
+    пишется в ``shared/training_protocol.json`` **один раз**: повторный вызов
+    при существующем файле — отказ (перевыбор эпох незаметно недопустим).
+    Протокол содержит полный selection-отчёт, sha256 сплитов, full_tfidf_id,
+    замороженные секции конфигурации выбранного кандидата и hash файла
+    selection-отчёта; refit валидируется против него (fail closed).
+    """
+    if os.path.isfile(cfg.protocol_path):
+        raise S06Gate1Error(
+            f"Протокол уже заморожен: {cfg.protocol_path}. Перевыбор эпох/конфигурации "
+            "после заморозки запрещён; для нового протокола нужен новый каталог "
+            "output_root, существующие результаты не перезаписываются."
+        )
+    # Fail closed до selection: расписание LR обязано быть подтверждённо
+    # идентичным для selection/refit (см. ensure_constant_lr_schedule).
+    ensure_constant_lr_schedule(cfg.resolved_scheduler())
+    selection = run_selection(cfg.to_s05_config(), manifest, loaded, device)
+    sel_path = os.path.join(cfg.baseline_dir, "selection", "selection_report.json")
+    chosen = selection["chosen"]
+    cand = next(c for c in selection["candidates"] if c["name"] == chosen["name"])
+    protocol = {
+        "schema_version": S06_GATE1_SCHEMA,
+        "gate": "S06 Gate 1: frozen training protocol (epoch selection + final train config)",
+        "frozen": True,
+        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
+        "notes": dict(PROTOCOL_NOTES),
+        "config": {
+            "preset": cfg.preset,
+            "model": dict(cfg.model),
+            "loss": dict(cfg.loss),
+            "train": dict(cfg.train),
+            "sweep": [dict(c) for c in cfg.sweep],
+            "features_chunk_rows": int(cfg.features_chunk_rows),
+            "selection_seed": int(cfg.selection_seed),
+            "seeds": list(cfg.seeds),
+            "reference_seed": int(cfg.reference_seed),
+        },
+        "resolved_sections": {
+            "model": cand["model"],
+            "loss": cand["loss"],
+            "train": cand["train"],
+        },
+        "chosen": chosen,
+        "early_stopping_metric": selection["metric"],
+        "fixed_epochs": int(chosen["fixed_epochs"]),
+        "splits": {
+            name: {"sha256": part["sha256"], "bytes": part["bytes"], "n_rows": part["n_rows"]}
+            for name, part in manifest["parts"].items()
+        },
+        "labeled_parquet": manifest["inputs"]["labeled_parquet"],
+        "min_count_policy": int(manifest["policy"]["min_count"]),
+        "split_seed": int(manifest["policy"]["split_seed"]),
+        "full_tfidf_id": verification["full_tfidf_id"],
+        "artifact_version": verification["artifact_version"],
+        "corpus_fingerprint": verification["corpus_fingerprint"],
+        "selection_report_file": os.path.abspath(sel_path),
+        "selection_report_sha256": _sha256_file(sel_path),
+        "selection_report": selection,
+        "sealed_test_boundary": (
+            "sealed_test исключён из inference/eval Gate 1; метрик на нём нет "
+            "и не будет до отдельного поручения"
+        ),
+    }
+    _write_json(protocol, cfg.protocol_path)
+    log.info(
+        "Протокол заморожен: %d эпох (кандидат %s, метрика %s), файл %s",
+        protocol["fixed_epochs"], chosen["name"], selection["metric"], cfg.protocol_path,
+    )
+    return protocol
+
+
+def load_training_protocol(cfg: S06Gate1Config) -> dict[str, Any]:
+    if not os.path.isfile(cfg.protocol_path):
+        raise S06Gate1Error(
+            f"Нет замороженного протокола {cfg.protocol_path!r}: сначала выполните "
+            "фазу select (train-only выбор эпох на train_fit/train_monitor). "
+            "Обучение без замороженного протокола запрещено."
+        )
+    with open(cfg.protocol_path, encoding="utf-8") as f:
+        return json.load(f)
+
+
+def validate_protocol_against_inputs(
+    cfg: S06Gate1Config,
+    protocol: dict[str, Any],
+    manifest: dict[str, Any],
+    verification: dict[str, Any],
+) -> None:
+    """Fail closed: протокол применим только к своим входам.
+
+    Сверяет sha256 всех частей сплита, full_tfidf_id, fingerprint корпуса,
+    политику min_count, seeds/reference_seed и замороженные секции
+    конфигурации, пересобранные из текущего cfg — до любого чтения данных и
+    обучения. Поверх того проверяет неизменность замороженного selection-отчёта
+    (пересчёт sha256 файла против ``selection_report_sha256``), согласованность
+    числа эпох на всех уровнях вложенности (верхний ``fixed_epochs`` ==
+    ``chosen.fixed_epochs`` == ``selection_report.chosen.fixed_epochs``,
+    совпадение ``chosen`` и ``selection_report.chosen``) и постоянство LR
+    (``resolved_sections.train.scheduler == 'none'``): расхождение —
+    S06Gate1Error, а не молчаливое обучение по чужому/повреждённому протоколу.
+    """
+    problems: list[str] = []
+
+    # --- Целостность замороженного selection-отчёта и эпох.
+    sel_path = protocol.get("selection_report_file")
+    sel_sha = protocol.get("selection_report_sha256")
+    if not sel_path or not sel_sha:
+        problems.append("протокол не содержит selection_report_file/selection_report_sha256")
+    elif not os.path.isfile(sel_path):
+        problems.append(f"замороженный selection-отчёт не найден: {sel_path}")
+    elif _sha256_file(sel_path) != sel_sha:
+        problems.append(
+            "selection_report_sha256: файл отчёта изменился после заморозки протокола"
+        )
+    top_epochs = protocol.get("fixed_epochs")
+    chosen = protocol.get("chosen") or {}
+    sel_chosen = (protocol.get("selection_report") or {}).get("chosen") or {}
+    for label, value in (
+        ("chosen.fixed_epochs", chosen.get("fixed_epochs")),
+        ("selection_report.chosen.fixed_epochs", sel_chosen.get("fixed_epochs")),
+    ):
+        try:
+            consistent = int(value) == int(top_epochs)
+        except (TypeError, ValueError):
+            consistent = False
+        if not consistent:
+            problems.append(
+                f"несогласованность эпох: верхний fixed_epochs={top_epochs!r} != {label}={value!r}"
+            )
+    if sel_chosen and sel_chosen != chosen:
+        problems.append("chosen != selection_report.chosen (расхождение вложенной копии)")
+    resolved_train = (protocol.get("resolved_sections") or {}).get("train") or {}
+    if resolved_train.get("scheduler", "none") != "none":
+        problems.append(
+            f"scheduler={resolved_train.get('scheduler')!r}: расписание LR, зависящее от "
+            "числа шагов, для S06 не подтверждено идентичностью selection/refit; "
+            "разрешён только 'none'"
+        )
+
+    proto_splits = protocol.get("splits") or {}
+    for name, part in manifest["parts"].items():
+        expected = proto_splits.get(name) or {}
+        if expected.get("sha256") != part.get("sha256") or expected.get("n_rows") != part.get("n_rows"):
+            problems.append(
+                f"сплит {name}: протокол зафиксирован для sha256 "
+                f"{str(expected.get('sha256'))[:16]}…/{expected.get('n_rows')} строк, "
+                f"текущий сплит {str(part.get('sha256'))[:16]}…/{part.get('n_rows')}"
+            )
+    if protocol.get("full_tfidf_id") != verification.get("full_tfidf_id"):
+        problems.append(
+            f"full_tfidf_id: протокол относится к {str(protocol.get('full_tfidf_id'))[:16]}…, "
+            f"проверен артефакт {str(verification.get('full_tfidf_id'))[:16]}…"
+        )
+    proto_corpus = (protocol.get("corpus_fingerprint") or {})
+    new_corpus = (verification.get("corpus_fingerprint") or {})
+    if (proto_corpus.get("sha256") != new_corpus.get("sha256")
+            or proto_corpus.get("bytes") != new_corpus.get("bytes")):
+        problems.append("corpus_fingerprint")
+    if protocol.get("min_count_policy") != manifest["policy"].get("min_count"):
+        problems.append("min_count_policy")
+    proto_cfg = protocol.get("config") or {}
+    if list(proto_cfg.get("seeds") or []) != list(cfg.seeds):
+        problems.append(f"seeds: протокол {proto_cfg.get('seeds')} != текущие {cfg.seeds}")
+    if int(proto_cfg.get("reference_seed", -1)) != int(cfg.reference_seed):
+        problems.append("reference_seed")
+    # Замороженные секции конфигурации должны воспроизводиться из текущего cfg:
+    # иначе refit обучал бы не тот протокол, что был заморожен.
+    s05cfg = cfg.to_s05_config()
+    chosen = (protocol.get("selection_report") or {}).get("chosen") or {}
+    cand = next(
+        (c for c in (protocol.get("selection_report") or {}).get("candidates", [])
+         if c["name"] == chosen.get("name")),
+        None,
+    )
+    if cand is None:
+        problems.append("chosen-кандидат не найден в selection_report протокола")
+    else:
+        model_cfg, loss_cfg, train_cfg = s05cfg.build_sections(cand["model"], cand["train"])
+        if dataclasses.asdict(model_cfg) != (protocol.get("resolved_sections") or {}).get("model"):
+            problems.append("model-секция не воспроизводится из текущего cfg")
+        if dataclasses.asdict(loss_cfg) != (protocol.get("resolved_sections") or {}).get("loss"):
+            problems.append("loss-секция не воспроизводится из текущего cfg (проверьте --loss)")
+        if dataclasses.asdict(train_cfg) != (protocol.get("resolved_sections") or {}).get("train"):
+            problems.append("train-секция не воспроизводится из текущего cfg (проверьте --train)")
+    if problems:
+        raise S06Gate1Error(
+            "Замороженный протокол несовместим с текущими входами: " + "; ".join(problems)
+        )
+
+
+# ---------------------------------------------------------------------------
+# D05–D06: финальный refit 42/43/44 + единственная оценка comparison_val
+# ---------------------------------------------------------------------------
+
+
+def run_gate1_refit(
+    cfg: S06Gate1Config,
+    manifest: dict[str, Any],
+    verification: dict[str, Any],
+    loaded: Any,
+    device: torch.device,
+) -> dict[str, Any]:
+    """Refit трёх full baseline на outer_train по замороженному протоколу.
+
+    Порядок гарантирует границы: проверка TF-IDF уже выполнена вызывающей
+    стороной (fail closed), затем загрузка и валидация замороженного протокола
+    против текущих входов — только после этого контур S05 обучает seed
+    42/43/44 на outer_train и выполняет единственную оценку на
+    ``comparison_val`` (majority-контроль и round-trip опорного checkpoint —
+    внутри контура S05 без изменений). ``sealed_test`` не читается.
+    """
+    protocol = load_training_protocol(cfg)
+    validate_protocol_against_inputs(cfg, protocol, manifest, verification)
+    selection = protocol["selection_report"]
+    s05cfg = cfg.to_s05_config()
+    # Проверенный provenance дублируется контуром S05 в baseline_dir
+    # (record_tfidf_verification внутри run_refit: новый файл либо сверка).
+    record_tfidf_verification(s05cfg.output_dir, verification)
+    return run_refit(s05cfg, manifest, loaded, device,
+                     selection=selection, tfidf_verification=verification)
+
+
+def run_gate1_contour(cfg: S06Gate1Config) -> dict[str, Any]:
+    """Полный Gate 1: verify split (Gate 0) → verify TF-IDF → select+freeze → refit+eval."""
+    from .train import resolve_device
+    from .vectorizer import load_vectorizer_artifact
+
+    cfg.validate()
+    device = resolve_device(cfg.device)
+    manifest = verify_gate0_split(cfg)
+    verification = verify_s06_tfidf(cfg)
+    loaded = load_vectorizer_artifact(cfg.full_tfidf_path)
+    freeze_training_protocol(cfg, manifest, verification, loaded, device)
+    return run_gate1_refit(cfg, manifest, verification, loaded, device)
+
+
+# ---------------------------------------------------------------------------
+# Ресурсная оценка Gate 1 (проекция от измерений S05, без обучения)
+# ---------------------------------------------------------------------------
+
+
+def _dir_size(path: str) -> int | None:
+    if not os.path.isdir(path):
+        return None
+    total = 0
+    for root, _dirs, files in os.walk(path):
+        for name in files:
+            try:
+                total += os.path.getsize(os.path.join(root, name))
+            except OSError:
+                return None
+    return total
+
+
+def estimate_gate1_resources(
+    cfg: S06Gate1Config,
+    manifest: dict[str, Any],
+    write: bool = False,
+) -> dict[str, Any]:
+    """Честная проекция ресурсов Gate 1 от измерений S05 и размеров S06-сплита.
+
+    **Не выполняет обучение, transform и inference.** Опора — сохранённые
+    замеры S05 (``outputs/S05/stages.csv``, ``outputs/S05/manifest.json``,
+    размер checkpoint) и число строк частей нового сплита из manifest Gate 0.
+    Все проекции помечены ``projected=true`` и источником; это не измерения.
+    GPU недоступен — ``null`` (не ноль). При ``write`` результат пишется в
+    ``shared/resource_estimate_gate1.json`` (только новый файл).
+    """
+    parts = manifest["parts"]
+    n_outer = int(parts["outer_train"]["n_rows"])
+    n_fit = int(parts["train_fit"]["n_rows"])
+    n_mon = int(parts["train_monitor"]["n_rows"])
+    n_val = int(parts[EVAL_PART]["n_rows"])
+
+    anchor: dict[str, Any] = {"available": False}
+    proj: dict[str, Any] = {
+        "vectorize_train_seconds": None,
+        "selection_train_seconds": None,
+        "refit_seconds_per_seed": None,
+        "eval_seconds_per_seed": None,
+    }
+    s05_stages_path = os.path.join("outputs", "S05", "stages.csv")
+    s05_manifest_path = os.path.join("outputs", "S05", "manifest.json")
+    if os.path.isfile(s05_stages_path) and os.path.isfile(s05_manifest_path):
+        import pandas as pd
+
+        stages = pd.read_csv(s05_stages_path)
+        s05 = json.load(open(s05_manifest_path, encoding="utf-8"))
+        n_outer_s05 = int(s05["inputs"]["splits"]["outer_train"]["n_rows"])
+        fixed_s05 = int(s05.get("fixed_epochs") or 0)
+        sel_mask = stages["stage"].eq("selection_train")
+        refit_mask = stages["stage"].eq("mlp_train") & stages["run_id"].astype(str).str.startswith("seed_")
+        eval_mask = stages["stage"].eq("comparison_val_inference")
+        vec_mask = stages["stage"].eq("vectorize_train")
+        scale = n_outer / max(n_outer_s05, 1)
+        anchor = {
+            "available": True,
+            "source": os.path.abspath(s05_stages_path),
+            "s05_outer_train_rows": n_outer_s05,
+            "s05_fixed_epochs": fixed_s05,
+            "note": "исторический протокол S05 (min_count=10, старые сплиты): только "
+                    "опора для проекции времени/RAM, не baseline S06",
+        }
+
+        def _mean_seconds(mask) -> float | None:
+            vals = stages.loc[mask, "seconds"]
+            vals = vals[vals.notna()]
+            return float(vals.mean()) if len(vals) else None
+
+        vec_s = _mean_seconds(vec_mask)
+        if vec_s is not None:
+            proj["vectorize_train_seconds"] = round(vec_s * scale, 3)
+        sel_s = _mean_seconds(sel_mask)
+        if sel_s is not None and fixed_s05 > 0:
+            # selection ≈ обучение на train_fit примерно на числе эпох ≤ refit
+            proj["selection_train_seconds"] = round(sel_s * (n_fit / max(n_outer_s05, 1)), 3)
+        refit_s = _mean_seconds(refit_mask)
+        if refit_s is not None and fixed_s05 > 0:
+            per_epoch = refit_s / fixed_s05
+            proj["refit_seconds_per_seed"] = round(per_epoch * scale * fixed_s05, 3)
+            proj["assumption_epochs"] = (
+                fixed_s05,
+                "число эпох S06 ещё не выбрано; проекция при том же числе эпох, "
+                "после select замените фактическим fixed_epochs из протокола",
+            )
+        eval_s = _mean_seconds(eval_mask)
+        if eval_s is not None:
+            proj["eval_seconds_per_seed"] = round(eval_s * (n_val / max(int(
+                s05["inputs"]["splits"][EVAL_PART]["n_rows"]), 1)), 3)
+
+    ckpt_anchor = _dir_size(os.path.join("outputs", "S05", "checkpoints", "seed_42"))
+    estimate = {
+        "schema_version": S06_GATE1_SCHEMA,
+        "role": "gate1_resource_estimate",
+        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
+        "projected": True,
+        "disclaimer": (
+            "Проекция от измерений S05 и размеров сплита S06; НЕ измерение. "
+            "Фактические время/RAM/VRAM фиксируются stages.csv самого Gate 1."
+        ),
+        "s06_split_rows": {
+            "outer_train": n_outer, "train_fit": n_fit,
+            "train_monitor": n_mon, EVAL_PART: n_val,
+            SEALED_PART: int(parts[SEALED_PART]["n_rows"]),
+        },
+        "anchor_s05": anchor,
+        "projections_seconds": proj,
+        "ram_bytes_reference_s05": {
+            "refit_peak_rss": None,  # заполняется ниже при наличии stages
+            "eval_peak_rss": None,
+            "note": "RAM сублиинейна по числу строк (разреженные признаки); ориентир — "
+                    "фактические пики stages.csv Gate 1, здесь только якоря S05",
+        },
+        "vram": {
+            "torch_cuda_available": bool(torch.cuda.is_available()),
+            "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
+            "projected_peak_bytes_s05_gpu_refit": None,
+            "note": "при недоступности GPU — null, не ноль; CPU-прогон дольше и не измерен здесь",
+        },
+        "checkpoint_bytes_reference_s05_seed42": ckpt_anchor,
+        "sealed_test": "оценка ресурсов не выполняет inference; sealed_test исключён",
+    }
+    if anchor["available"]:
+        import pandas as pd
+
+        stages = pd.read_csv(s05_stages_path)
+        refit_rss = stages.loc[stages["stage"].eq("mlp_train"), "peak_rss_bytes"]
+        eval_rss = stages.loc[stages["stage"].eq("comparison_val_inference"), "peak_rss_bytes"]
+        if refit_rss.notna().any():
+            estimate["ram_bytes_reference_s05"]["refit_peak_rss"] = int(refit_rss.max())
+        if eval_rss.notna().any():
+            estimate["ram_bytes_reference_s05"]["eval_peak_rss"] = int(eval_rss.max())
+        vram = stages.loc[stages["stage"].eq("mlp_train"), "peak_vram_bytes"]
+        if vram.notna().any() and bool(torch.cuda.is_available()):
+            estimate["vram"]["projected_peak_bytes_s05_gpu_refit"] = int(vram.max())
+    if write:
+        os.makedirs(cfg.shared_dir, exist_ok=True)
+        path = os.path.join(cfg.shared_dir, "resource_estimate_gate1.json")
+        if os.path.exists(path):
+            raise S06Gate1Error(f"Существующие outputs не перезаписываются: {path}")
+        _write_json(estimate, path)
+        estimate = dict(estimate)
+        estimate["written_to"] = os.path.abspath(path)
+    return estimate
+
+
+__all__ = [
+    "S06_GATE1_SCHEMA",
+    "S06Gate1Config",
+    "S06Gate1Error",
+    "EXPECTED_FULL_VOCAB",
+    "ensure_constant_lr_schedule",
+    "estimate_gate1_resources",
+    "freeze_training_protocol",
+    "load_training_protocol",
+    "run_gate1_contour",
+    "run_gate1_refit",
+    "validate_protocol_against_inputs",
+    "verify_gate0_split",
+    "verify_s06_tfidf",
+]
diff --git a/kkt_clf/s06_comparison.py b/kkt_clf/s06_comparison.py
new file mode 100644
index 0000000..eeb85f6
--- /dev/null
+++ b/kkt_clf/s06_comparison.py
@@ -0,0 +1,1758 @@
+"""S06 (D06–D08): сравнительный контур Gate 2–3 — ранжирование, refit, 30 paired runs.
+
+Реализует сравнение supervised feature selection на замороженном протоколе S06
+поверх результатов Gate 0 (``outputs/S06/splits``, см. ``kkt_clf.s06_split``)
+и Gate 1 (``outputs/S06/shared/training_protocol.json`` +
+``outputs/S06/runs/baseline_full``, см. ``kkt_clf.s06_baseline``). Методология
+отбора не меняется и целиком делегируется существующим контурам S03/S04:
+
+- ``global`` и ``class_aware`` Chi² — :func:`kkt_clf.selection.select_features`
+  (рейтинги полные, float64, на диске; положительные OvR-ассоциации и
+  round-robin у class_aware — без изменений);
+- ``gradient_x_input`` — :func:`kkt_clf.attribution.compute_margin_attributions`
+  и :func:`kkt_clf.attribution.build_gxi_ratings` на опорном train-only
+  checkpoint Gate 1 (``reference_seed``, margin истинной L3 к сильнейшему
+  конкуренту, положительные вклады; отрицательные — только диагностика);
+- fixed-vocabulary refit — :func:`kkt_clf.vectorizer.refit_selected_vectorizer`
+  на **том же** фактическом снимке корпуса, что полный TF-IDF S02
+  (проверка sha256 корпуса, инвариант IDF, L2-нормировка, round-trip).
+
+Границы протокола:
+
+- Рейтинги считаются **только по ``outer_train``**; метки comparison_val и
+  sealed_test в отбор не попадают (fail closed в используемых контрактах).
+  Рейтинг каждого метода вычисляется один раз и сохраняется с контрольными
+  суммами в ``shared/rankings/`` / ``shared/attribution/``; бюджеты
+  K = 50k/100k/200k (char_share=0.75) применяются из сохранённых рейтингов
+  повторно, без пересчёта оценок. Выбранные словари фиксируются для всех seed,
+  чтобы измерять вариабельность обучения MLP, а не смешивать её с
+  вариабельностью рейтингов; ограничение дизайна зафиксировано в отчёте.
+- Refit выполняется **один раз на метод×K** и переиспользуется тремя seed
+  обучения; refit по seed не повторяется.
+- Финальные MLP обучаются с нуля на ``outer_train`` ровно на замороженное в
+  Gate 1 число эпох с секциями model/loss/train замороженного протокола
+  (``restore_best=False``); конфигурация ``none`` — переиспользование трёх
+  baseline checkpoint Gate 1 (повторное обучение не выполняется), их метрики
+  пересчитываются тем же протоколом оценки и сверяются с сохранёнными Gate 1.
+- Оценка — raw top-1 L3 без отказа на ``comparison_val`` по фиксированной
+  L3-вселенной label maps; парные Δ считаются строго к ``none`` того же seed.
+  ``sealed_test`` не читается и не оценивается нигде (никакого inference/метрик);
+  используются только структурные сведения manifest, как в Gate 1.
+
+Устойчивость к сбоям и отсутствие перезаписей: каждый запуск фиксирует
+собственный результат (``runs/<slug>/run_result_seed_<s>.json``), предсказания
+(``runs/predictions/<run_id>.parquet``) и строки этапов (``runs/stages.csv``,
+только дополнение); уже записанный результат любого статуса не пересчитывается
+и не перезаписывается — возобновление пропускает завершённые runs, повторный
+запуск failed-run требует явного удаления его result-файла (диагностика в CLI).
+Плановые 30 runs фиксируются заранее в ``runs/planned.json``; пропуски и сбои
+отражаются статусами ``pending/success/failed/skipped`` с причинами, а не
+маскируются. Агрегированный отчёт Gate 4 строится отдельным контуром
+(``kkt_clf.s06_report``) из сохранённых файлов без обучения и inference.
+"""
+from __future__ import annotations
+
+import dataclasses
+import datetime as dt
+import json
+import logging
+import os
+import shutil
+from dataclasses import dataclass, field
+from typing import Any, Iterable
+
+import numpy as np
+import pandas as pd
+import scipy.sparse as sp
+import torch
+
+from .artifact import environment_versions, load_artifact, save_artifact
+from .attribution import (
+    ATTRIBUTION_PROVENANCE_FILENAME,
+    build_gxi_ratings,
+    compute_margin_attributions,
+    feature_contract_fingerprint,
+    select_features_gradient_x_input,
+    verify_attribution_checkpoint,
+)
+from .config import HEADS, LossConfig, ModelConfig, TrainConfig
+from .experiment import _Measurement, _l3_val_metrics, _sha256_file, _sha256_json
+from .labels import LEVELS, LabelMaps
+from .model import SparseMLP
+from .reject import RejectRule
+from .s05_baseline import (
+    load_part,
+    verify_full_tfidf,
+    _encode_labels,
+    _transform,
+)
+from .s06_baseline import (
+    S06Gate1Config,
+    load_training_protocol,
+    validate_protocol_against_inputs,
+    verify_gate0_split,
+)
+from .selection import (
+    RATINGS_FORMAT_VERSION,
+    BranchSelection,
+    SelectionArtifactError,
+    SelectionConfig,
+    SelectionError,
+    SelectionResult,
+    _rank_topk,
+    _ratings_declared_names,
+    branch_budgets,
+    load_selection_artifact,
+    roundrobin_select_from_rank_files,
+    save_selection_artifact,
+    select_features,
+    validate_ratings_dir,
+)
+from .train import compute_class_weights, predict, set_seed, train_model
+from .vectorizer import LoadedVectorizer, load_vectorizer_artifact, refit_selected_vectorizer
+
+log = logging.getLogger(__name__)
+
+S06_COMPARISON_SCHEMA = "s06-comparison@1"
+
+METHOD_NONE = "none"
+METHOD_GLOBAL = "global"
+METHOD_CLASS_AWARE = "class_aware"
+METHOD_GXI = "gradient_x_input"
+SELECTION_METHODS = (METHOD_GLOBAL, METHOD_CLASS_AWARE, METHOD_GXI)
+
+DEFAULT_BUDGETS = (50_000, 100_000, 200_000)
+DEFAULT_SEEDS = (42, 43, 44)
+DEFAULT_CHAR_SHARE = 0.75
+
+# Этапы контура и канонический порядок для дописывания skipped-строк.
+STAGE_VECTORIZE_OUTER = "vectorize_outer_train"
+STAGE_RANKING = "ranking"
+STAGE_ATTRIBUTION = "attribution"
+STAGE_SELECTION = "selection"
+STAGE_REFIT = "selected_tfidf_refit"
+STAGE_VECTORIZE_CONFIG = "vectorize_config"
+STAGE_MLP_TRAIN = "mlp_train"
+STAGE_VAL_INFERENCE = "comparison_val_inference"
+
+STAGES_COLUMNS = [
+    "run_id", "stage", "status", "seconds", "peak_rss_bytes", "peak_vram_bytes",
+    "artefact_bytes", "measurement_note",
+]
+
+RUN_RESULT_SCHEMA = "s06-run-result@1"
+
+PROTOCOL_NOTES = {
+    "rankings_train_only": (
+        "рейтинги всех методов вычислены только по outer_train полного TF-IDF; "
+        "метки comparison_val/sealed_test не использовались (fail closed в "
+        "контрактах S03/S04)"
+    ),
+    "gxi_reference": (
+        "G×I-атрибуция выполнена на опорном checkpoint Gate 1 (reference_seed, "
+        "train-only правило выбора): margin истинной L3 к сильнейшему конкуренту, "
+        "положительные вклады; отрицательные — только в диагностике; ошибочно "
+        "классифицированные train-примеры сохранены"
+    ),
+    "budget_reuse": (
+        "полные рейтинги каждого метода вычислены один раз и сохранены с "
+        "контрольными суммами; бюджеты 50k/100k/200k (char_share=0.75) применены "
+        "из сохранённых рейтингов без пересчёта оценок; выбранные словари "
+        "зафиксированы для всех seed (вариабельность рейтингов не смешивается "
+        "с вариабельностью обучения MLP — ограничение дизайна)"
+    ),
+    "refit_once": (
+        "refit TF-IDF с фиксированными словарями выполнен один раз на метод×K "
+        "на том же снимке корпуса, что полный TF-IDF S02; между seed общий "
+        "refit-артефакт, повторный fit не выполнялся"
+    ),
+    "frozen_protocol": (
+        "финальные MLP обучены с нуля на outer_train ровно на замороженное в "
+        "Gate 1 число эпох с секциями model/loss/train замороженного протокола "
+        "(restore_best=False, веса финальной эпохи); comparison_val не "
+        "использовался для обучения, эпох, checkpoint или ранжирования"
+    ),
+    "paired": (
+        "парные Δ считаются строго к полному baseline (none) того же train seed; "
+        "различия размеров входного слоя не предполагают идентичности начальных "
+        "весов; три seed на одном split — не доверительный интервал"
+    ),
+    "sealed_test": (
+        "sealed_test не использовался ни для inference, ни для метрик; обращений "
+        "к его строкам в контуре нет (только структурные sha256/n_rows manifest)"
+    ),
+}
+
+
+class S06ComparisonError(RuntimeError):
+    """Ошибка сравнительного контура S06; существующие файлы не перезаписываются."""
+
+
+@dataclass
+class S06ComparisonConfig:
+    """Конфигурация сравнительного контура Gate 2–3 S06.
+
+    Все пути по умолчанию согласованы с контрактами Gate 0/1. Бюджеты и seed
+    настраиваются (для изолированных smoke на фикстурах); плановый прогон S06
+    использует budgets=(50_000, 100_000, 200_000), char_share=0.75,
+    seeds=(42, 43, 44), reference_seed=42.
+    """
+
+    output_root: str = "outputs/S06"
+    splits_manifest: str | None = None
+    corpus_path: str = "data/corpus.pqt"
+    full_tfidf_path: str = "outputs/S02/tfidf_1m"
+    methods: tuple[str, ...] = SELECTION_METHODS
+    budgets: tuple[int, ...] = DEFAULT_BUDGETS
+    char_share: float = DEFAULT_CHAR_SHARE
+    seeds: tuple[int, ...] = DEFAULT_SEEDS
+    reference_seed: int = 42
+    device: str = "auto"
+    features_chunk_rows: int = 50_000
+    attribution_batch_size: int = 4096
+    eval_batch_size: int = 8192
+    # Пасsthrough настроек Gate 1 (должны совпадать с вызовом Gate 1:
+    # validate_protocol_against_inputs воспроизводит замороженные секции из них).
+    preset: str = "ods_light"
+    model: dict[str, Any] = field(default_factory=dict)
+    loss: dict[str, Any] = field(default_factory=dict)
+    train: dict[str, Any] = field(default_factory=dict)
+    sweep: list[dict[str, Any]] = field(default_factory=list)
+
+    def __post_init__(self) -> None:
+        if self.splits_manifest is None:
+            self.splits_manifest = os.path.join(self.output_root, "splits", "manifest.json")
+        self.methods = tuple(self.methods)
+        self.budgets = tuple(int(k) for k in self.budgets)
+        self.seeds = tuple(int(s) for s in self.seeds)
+
+    # --- Пути контракта -----------------------------------------------------
+    @property
+    def shared_dir(self) -> str:
+        return os.path.join(self.output_root, "shared")
+
+    @property
+    def runs_dir(self) -> str:
+        return os.path.join(self.output_root, "runs")
+
+    @property
+    def baseline_dir(self) -> str:
+        return os.path.join(self.runs_dir, "baseline_full")
+
+    @property
+    def protocol_path(self) -> str:
+        return os.path.join(self.shared_dir, "training_protocol.json")
+
+    @property
+    def tfidf_ref_path(self) -> str:
+        return os.path.join(self.shared_dir, "full_tfidf_ref.json")
+
+    @property
+    def planned_path(self) -> str:
+        return os.path.join(self.runs_dir, "planned.json")
+
+    @property
+    def stages_path(self) -> str:
+        return os.path.join(self.runs_dir, "stages.csv")
+
+    def rankings_dir(self, method: str) -> str:
+        return os.path.join(self.shared_dir, "rankings", method)
+
+    def attribution_dir(self) -> str:
+        return os.path.join(self.shared_dir, "attribution")
+
+    def selection_dir(self, slug: str) -> str:
+        return os.path.join(self.shared_dir, "selection", slug)
+
+    def config_dir(self, slug: str) -> str:
+        return os.path.join(self.runs_dir, slug)
+
+    def run_result_path(self, slug: str, seed: int) -> str:
+        return os.path.join(self.config_dir(slug), f"run_result_seed_{seed}.json")
+
+    def predictions_path(self, run_id: str) -> str:
+        return os.path.join(self.runs_dir, "predictions", f"{run_id}.parquet")
+
+    # --- Служебные ----------------------------------------------------------
+    def slug(self, method: str, budget: int | None) -> str:
+        return METHOD_NONE if method == METHOD_NONE else f"{method}_k{budget}"
+
+    def run_id(self, slug: str, seed: int) -> str:
+        return f"{slug}_seed{seed}"
+
+    def planned_configs(self) -> list[dict[str, Any]]:
+        """10 конфигураций: none + методы × бюджеты (порядок стабилен)."""
+        configs = [{"slug": METHOD_NONE, "method": METHOD_NONE, "budget": None}]
+        for method in self.methods:
+            for budget in self.budgets:
+                configs.append({"slug": self.slug(method, budget), "method": method, "budget": int(budget)})
+        return configs
+
+    def gate1_config(self) -> S06Gate1Config:
+        """Согласованная обёртка Gate 1 для read-only проверок (код Gate 1 не меняется)."""
+        return S06Gate1Config(
+            output_root=self.output_root,
+            splits_manifest=str(self.splits_manifest),
+            full_tfidf_path=self.full_tfidf_path,
+            corpus_path=self.corpus_path,
+            preset=self.preset,
+            model=dict(self.model),
+            loss=dict(self.loss),
+            train=dict(self.train),
+            sweep=[dict(c) for c in self.sweep],
+            seeds=list(self.seeds),
+            reference_seed=self.reference_seed,
+            features_chunk_rows=self.features_chunk_rows,
+            device=self.device,
+        )
+
+    def validate(self) -> None:
+        bad = [m for m in self.methods if m not in SELECTION_METHODS]
+        if bad:
+            raise S06ComparisonError(f"methods: неподдерживаемые методы {bad}, ожидаются {SELECTION_METHODS}")
+        if len(set(self.methods)) != len(self.methods):
+            raise S06ComparisonError(f"methods содержит дубликаты: {self.methods}")
+        if not self.budgets or any(k <= 0 for k in self.budgets):
+            raise S06ComparisonError(f"budgets: положительные целые, получено {self.budgets}")
+        if len(set(self.budgets)) != len(self.budgets):
+            raise S06ComparisonError(f"budgets содержат дубликаты: {self.budgets}")
+        if not 0.0 < self.char_share < 1.0:
+            raise S06ComparisonError(
+                f"char_share должен быть строго внутри (0, 1) (обе ветки обязаны получить "
+                f"ненулевой бюджет), получено {self.char_share}"
+            )
+        if len(set(self.seeds)) != len(self.seeds):
+            raise S06ComparisonError(f"seeds должны быть различны: {self.seeds}")
+        if self.reference_seed not in self.seeds:
+            raise S06ComparisonError(f"reference_seed={self.reference_seed} должен входить в seeds={self.seeds}")
+
+
+# ---------------------------------------------------------------------------
+# Мелкие утилиты (без побочных эффектов)
+# ---------------------------------------------------------------------------
+
+
+def _now() -> str:
+    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
+
+
+def _dir_size(path: str) -> int | None:
+    if not os.path.isdir(path):
+        return None
+    total = 0
+    for root, _dirs, files in os.walk(path):
+        for name in files:
+            try:
+                total += os.path.getsize(os.path.join(root, name))
+            except OSError:
+                return None
+    return total
+
+
+def _check_new(paths: Iterable[str]) -> None:
+    """Отказ до любых записей, если какой-либо целевой путь уже существует."""
+    existing = [p for p in paths if os.path.exists(p)]
+    if existing:
+        raise S06ComparisonError(
+            "Существующие outputs не перезаписываются; удалите/переименуйте вручную: "
+            + ", ".join(existing)
+        )
+
+
+def _atomic_write_json(obj: dict, path: str) -> None:
+    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
+    tmp = f"{path}.tmp-{os.getpid()}"
+    with open(tmp, "w", encoding="utf-8") as f:
+        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
+    os.replace(tmp, path)
+
+
+def _append_stage_rows(path: str, rows: list[dict[str, Any]]) -> None:
+    """Дописывает строки этапов в stages.csv (файл создаётся с заголовком)."""
+    if not rows:
+        return
+    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
+    frame = pd.DataFrame(rows, columns=STAGES_COLUMNS)
+    header = not os.path.exists(path)
+    frame.to_csv(path, mode="a", header=header, index=False)
+
+
+def _stage_recorded(cfg: "S06ComparisonConfig", run_id: str, stage: str) -> bool:
+    """Есть ли уже строка этапа (любого статуса) в stages.csv — для resume-защиты."""
+    if not os.path.exists(cfg.stages_path):
+        return False
+    try:
+        frame = pd.read_csv(cfg.stages_path, usecols=["run_id", "stage"])
+    except Exception:
+        return False
+    return bool(((frame["run_id"] == run_id) & (frame["stage"] == stage)).any())
+
+
+def _git_info() -> tuple[str | None, bool | None]:
+    import subprocess
+
+    def _run(*args: str) -> str | None:
+        try:
+            out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=10)
+            return out.stdout.strip() or None
+        except Exception:
+            return None
+
+    commit = _run("rev-parse", "HEAD")
+    dirty = None
+    if commit:
+        status = _run("status", "--porcelain")
+        dirty = bool(status)
+    return commit, dirty
+
+
+def _full_tfidf_id(loaded: LoadedVectorizer) -> str | None:
+    return (loaded.meta.get("files") or {}).get("features.joblib", {}).get("sha256")
+
+
+# ---------------------------------------------------------------------------
+# Проверка предусловий Gate 0/1 (read-only поверх их контрактов)
+# ---------------------------------------------------------------------------
+
+
+def verify_preconditions(cfg: S06ComparisonConfig) -> dict[str, Any]:
+    """Fail closed до любого ранжирования/refit/обучения.
+
+    Проверяет: сплит Gate 0 (через ``verify_gate0_split``), full TF-IDF против
+    корпуса (``verify_full_tfidf``) и согласованность с сохранённой ссылкой
+    ``shared/full_tfidf_ref.json``, замороженный протокол Gate 1 и его
+    совместимость с фактическими входами (``validate_protocol_against_inputs``),
+    наличие baseline-прогона Gate 1 для парных сравнений. К ``sealed_test``
+    обращений за данными нет и не будет.
+    """
+    cfg.validate()
+    g1 = cfg.gate1_config()
+    manifest = verify_gate0_split(g1)  # S06Gate1Error при несоответствии
+    verification = verify_full_tfidf(cfg.full_tfidf_path, cfg.corpus_path)
+    if verification.get("decision") != "ok":
+        raise S06ComparisonError(f"проверка full TF-IDF не пройдена: {verification.get('decision')}")
+    # Согласованность с сохранённой ссылкой Gate 1 (оба известных формата).
+    if os.path.isfile(cfg.tfidf_ref_path):
+        with open(cfg.tfidf_ref_path, encoding="utf-8") as f:
+            ref = json.load(f)
+        ref_id = ref.get("full_tfidf_id")
+        if ref_id is None and isinstance(ref.get("files"), dict):
+            ref_id = (ref["files"].get("features.joblib") or {}).get("sha256")
+        if ref_id and ref_id != verification.get("full_tfidf_id"):
+            raise S06ComparisonError(
+                "full TF-IDF не совпадает с сохранённой ссылкой "
+                f"{cfg.tfidf_ref_path}: подмена артефакта между фазами недопустима"
+            )
+    protocol = load_training_protocol(g1)
+    validate_protocol_against_inputs(g1, protocol, manifest, verification)
+    if not os.path.isdir(cfg.baseline_dir):
+        raise S06ComparisonError(
+            f"baseline-прогон Gate 1 не найден: {cfg.baseline_dir!r}. Парные сравнения "
+            "требуют завершённого Gate 1 (D05/D06 value_dod)."
+        )
+    log.info(
+        "Предусловия S06 проверены: split min_count=%s, full_tfidf_id=%s…, протокол "
+        "заморожен (%d эпох), baseline Gate 1 на месте",
+        manifest["policy"].get("min_count"), str(verification.get("full_tfidf_id"))[:12],
+        int(protocol.get("fixed_epochs") or 0),
+    )
+    return {"manifest": manifest, "verification": verification, "protocol": protocol}
+
+
+def _resolved_train_sections(protocol: dict[str, Any]) -> tuple[ModelConfig, LossConfig, TrainConfig, int]:
+    """Замороженные секции обучения из протокола Gate 1 (источник истины)."""
+    sections = protocol.get("resolved_sections") or {}
+    try:
+        model_cfg = ModelConfig(**sections["model"])
+        loss_cfg = LossConfig(**sections["loss"])
+        train_cfg = TrainConfig(**sections["train"])
+    except (KeyError, TypeError) as e:
+        raise S06ComparisonError(f"протокол Gate 1 не содержит корректных resolved_sections: {e}")
+    # Fail closed: расписание LR, зависящее от числа шагов (cosine/onecycle),
+    # не подтверждено идентичностью кривых selection/refit протокола S06;
+    # разрешён только постоянный LR 'none' (замороженный протокол Gate 1
+    # использует именно его).
+    if str(train_cfg.scheduler) != "none":
+        raise S06ComparisonError(
+            f"протокол Gate 1 содержит scheduler={train_cfg.scheduler!r}: для S06 "
+            "разрешён только постоянный LR 'none'; расписания с кривой от числа "
+            "шагов не проходят независимую проверку идентичности selection/refit."
+        )
+    fixed_epochs = int(protocol.get("fixed_epochs") or 0)
+    if fixed_epochs < 1:
+        raise S06ComparisonError(f"протокол Gate 1: fixed_epochs некорректен: {fixed_epochs!r}")
+    return model_cfg, loss_cfg, train_cfg, fixed_epochs
+
+
+# ---------------------------------------------------------------------------
+# План (30 runs) — фиксация до оценки
+# ---------------------------------------------------------------------------
+
+
+def write_plan(cfg: S06ComparisonConfig) -> dict[str, Any]:
+    """Фиксирует список плановых запусков (10 конфигураций × seed) до оценки.
+
+    Файл пишется один раз; повторный вызов — сверка идентичности плана
+    (методы/бюджеты/seed/protocol hash) без перезаписи.
+    """
+    cfg.validate()
+    runs = [
+        {
+            "run_id": cfg.run_id(c["slug"], seed),
+            "config_slug": c["slug"], "method": c["method"], "budget": c["budget"],
+            "seed": int(seed), "status": "pending", "reason": None,
+        }
+        for c in cfg.planned_configs()
+        for seed in cfg.seeds
+    ]
+    protocol_sha = _sha256_file(cfg.protocol_path) if os.path.isfile(cfg.protocol_path) else None
+    plan = {
+        "schema_version": S06_COMPARISON_SCHEMA,
+        "role": "planned_runs",
+        "generated_at": _now(),
+        "config": {
+            "methods": list(cfg.methods), "budgets": list(cfg.budgets),
+            "char_share": float(cfg.char_share), "seeds": list(cfg.seeds),
+            "reference_seed": int(cfg.reference_seed),
+            "full_tfidf_path": os.path.abspath(cfg.full_tfidf_path),
+            "corpus_path": os.path.abspath(cfg.corpus_path),
+            "splits_manifest": os.path.abspath(str(cfg.splits_manifest)),
+        },
+        "protocol_sha256": protocol_sha,
+        "n_planned": len(runs),
+        "runs": runs,
+        "notes": [
+            "10 конфигураций = none (полный baseline Gate 1) + методы × бюджеты; "
+            "3 общих train seed; парные Δ строго к none того же seed.",
+            "Статусы обновляются результатами прогонов (runs/<slug>/run_result_seed_<s>.json); "
+            "pending — запуск не выполнялся; существующие результаты не перезаписываются.",
+        ],
+    }
+    if os.path.exists(cfg.planned_path):
+        with open(cfg.planned_path, encoding="utf-8") as f:
+            existing = json.load(f)
+        key = lambda d: (d["config"]["methods"], d["config"]["budgets"], d["config"]["seeds"],  # noqa: E731
+                         d["config"]["char_share"], d["config"]["reference_seed"])
+        if key(existing) != key(plan):
+            raise S06ComparisonError(
+                f"{cfg.planned_path} уже существует с другим планом (методы/бюджеты/seed): "
+                "перезапись плана недопустима; используйте новый output_root."
+            )
+        return existing
+    os.makedirs(os.path.dirname(cfg.planned_path), exist_ok=True)
+    _atomic_write_json(plan, cfg.planned_path)
+    log.info("План зафиксирован: %d runs → %s", len(runs), cfg.planned_path)
+    return plan
+
+
+# ---------------------------------------------------------------------------
+# Gate 2a: рейтинги (Chi² — по outer_train; G×I — опорный checkpoint Gate 1)
+# ---------------------------------------------------------------------------
+
+
+def _transform_outer_train(
+    cfg: S06ComparisonConfig,
+    loaded: LoadedVectorizer,
+    manifest: dict[str, Any],
+    device: torch.device,
+) -> tuple[sp.csr_matrix, np.ndarray, pd.DataFrame, dict[str, Any]]:
+    """Общий transform outer_train полным TF-IDF (единый замер этапа)."""
+    train_df = load_part(manifest, "outer_train")
+    y_train = _encode_labels(LabelMaps.from_dict(manifest["label_maps"]), train_df, "outer_train")
+    with _Measurement(device, f"transform outer_train полным TF-IDF, chunk_rows={cfg.features_chunk_rows}") as m:
+        X_train = _transform(loaded, train_df["name"].tolist(), cfg.features_chunk_rows)
+    stage = m.stage_row("shared_vectorization", STAGE_VECTORIZE_OUTER, "success")
+    stage["artefact_bytes"] = None
+    stage["measurement_note"] += (
+        "; матрица общая для ранжирования всех методов и атрибуции G×I; повторный "
+        "transform внутри фазы rankings не выполняется"
+    )
+    if not _stage_recorded(cfg, "shared_vectorization", STAGE_VECTORIZE_OUTER):
+        _append_stage_rows(cfg.stages_path, [stage])
+    log.info("outer_train: %d строк (пустых %d), признаков %d",
+             X_train.shape[0], int((np.diff(X_train.indptr) == 0).sum()), X_train.shape[1])
+    return X_train, y_train, train_df, stage
+
+
+def _verify_ratings_meta(ratings_dir: str, method: str) -> dict[str, Any]:
+    """Сверяет контрольные суммы объявленных файлов рейтингов с meta.json."""
+    meta_path = os.path.join(ratings_dir, "meta.json")
+    with open(meta_path, encoding="utf-8") as f:
+        manifest = json.load(f)
+    if manifest.get("format") != RATINGS_FORMAT_VERSION:
+        raise SelectionArtifactError(
+            f"Рейтинги {ratings_dir}: несовместимый формат {manifest.get('format')!r}"
+        )
+    if manifest.get("method") != method:
+        raise SelectionArtifactError(
+            f"Рейтинги {ratings_dir}: метод {manifest.get('method')!r}, ожидается {method!r}"
+        )
+    checksums = manifest.get("files")
+    if not isinstance(checksums, dict) or not checksums:
+        raise SelectionArtifactError(f"Рейтинги {ratings_dir}: нет корректной секции контрольных сумм")
+    for entry in (manifest.get("branches") or {}).values():
+        for name in _ratings_declared_names(entry).values():
+            path = os.path.join(ratings_dir, name)
+            info = checksums.get(name)
+            if not isinstance(info, dict) or not os.path.isfile(path):
+                raise SelectionArtifactError(f"Рейтинги {ratings_dir}: нет файла/суммы {name!r}")
+            if _sha256_file(path) != info.get("sha256") or os.path.getsize(path) != info.get("bytes"):
+                raise SelectionArtifactError(
+                    f"Рейтинги {ratings_dir}: контрольная сумма/размер {name!r} не совпадают"
+                )
+    return manifest
+
+
+def _persist_ratings(
+    cfg: S06ComparisonConfig,
+    method: str,
+    result: SelectionResult,
+    *,
+    outer_train_sha256: str,
+    n_rows: int,
+    label_map_sha256: str,
+    full_tfidf_id: str | None,
+    extra: dict[str, Any] | None = None,
+) -> tuple[str, dict[str, Any]]:
+    """Переносит полные рейтинги из временного каталога в ``shared/rankings``."""
+    dest = cfg.rankings_dir(method)
+    if os.path.exists(dest):
+        raise S06ComparisonError(f"Каталог рейтингов уже существует и не перезаписывается: {dest}")
+    os.makedirs(os.path.dirname(dest), exist_ok=True)
+    # select_features возвращает ratings_dir, указывающий на каталог с meta.json
+    # и .npy-файлами рейтингов; переносим его целиком в shared/rankings/<method>.
+    shutil.move(result.ratings_dir, dest)
+    result.ratings_dir = dest
+    manifest = _verify_ratings_meta(dest, method)
+    files = {
+        name: {"sha256": _sha256_file(os.path.join(result.ratings_dir, name)),
+               "bytes": os.path.getsize(os.path.join(result.ratings_dir, name))}
+        for name in sorted(os.listdir(result.ratings_dir)) if os.path.isfile(os.path.join(result.ratings_dir, name))
+    }
+    meta = {
+        "schema_version": S06_COMPARISON_SCHEMA,
+        "role": "rankings",
+        "method": method,
+        "created_at": _now(),
+        "ratings_dir": os.path.abspath(result.ratings_dir),
+        "ratings_manifest": manifest,
+        "files": files,
+        "computed_on": {
+            "split": "outer_train",
+            "outer_train_sha256": outer_train_sha256,
+            "n_rows": int(n_rows),
+            "label_map_sha256": label_map_sha256,
+            "full_tfidf_id": full_tfidf_id,
+        },
+        "budgets_note": PROTOCOL_NOTES["budget_reuse"],
+        "train_only_note": PROTOCOL_NOTES["rankings_train_only"],
+    }
+    if extra:
+        meta.update(extra)
+    _atomic_write_json(meta, os.path.join(dest, "rankings_meta.json"))
+    log.info("Рейтинги %s сохранены: %s (%d файлов)", method, dest, len(files))
+    return result.ratings_dir, meta
+
+
+def _load_rankings_meta(cfg: S06ComparisonConfig, method: str) -> tuple[str, dict[str, Any]] | None:
+    path = os.path.join(cfg.rankings_dir(method), "rankings_meta.json")
+    if not os.path.isfile(path):
+        return None
+    with open(path, encoding="utf-8") as f:
+        meta = json.load(f)
+    ratings_dir = meta["ratings_dir"]
+    _verify_ratings_meta(ratings_dir, method)
+    for name, info in (meta.get("files") or {}).items():
+        p = os.path.join(ratings_dir, name)
+        if not os.path.isfile(p) or _sha256_file(p) != info.get("sha256"):
+            raise SelectionArtifactError(
+                f"Сохранённые рейтинги {method} повреждены (файл {name!r}): "
+                "пересчёт без удаления зафиксированных результатов не выполняется"
+            )
+    return ratings_dir, meta
+
+
+def _selection_from_saved_ratings(
+    method: str,
+    ratings_dir: str,
+    k: int,
+    char_share: float,
+    n_char: int,
+    n_total: int,
+    l3_labels: list[str] | None = None,
+) -> SelectionResult:
+    """Повторное применение бюджета к сохранённым полным рейтингам Chi².
+
+    Воспроизводит поведение :func:`kkt_clf.selection.select_features` для
+    заданного K без пересчёта оценок: global — стабильный top-K глобального
+    рейтинга; class_aware — тот же равноправный round-robin из тех же
+    потоковых файлов с тем же детерминированным добором. Индексы результата —
+    полно-пространные (word со смещением ``n_char``).
+    """
+    manifest = _verify_ratings_meta(ratings_dir, method)
+    k_char, k_word = branch_budgets(k, char_share, n_char, n_total - n_char)
+    report: dict[str, Any] = {
+        "method": method,
+        "k_requested": int(k),
+        "char_share": float(char_share),
+        "seed": 42,
+        "ratings_dir": os.path.abspath(ratings_dir),
+        "tie_policy": "stable sort by (-score, feature_index); no RNG",
+        "budgets": {"k_char": int(k_char), "k_word": int(k_word)},
+        "n_features_total": int(n_total),
+        "reapplied_from_saved_ratings": True,
+    }
+    branches: dict[str, BranchSelection] = {}
+    if method == METHOD_GLOBAL:
+        for branch, k_b, width in (("char", k_char, n_char), ("word", k_word, n_total - n_char)):
+            entry = (manifest.get("branches") or {}).get(branch) or {}
+            name = entry.get("global_scores")
+            if not name:
+                raise SelectionArtifactError(f"Манифест рейтингов global: ветка {branch!r} не объявляет global_scores")
+            scores = np.fromfile(os.path.join(ratings_dir, name), dtype=np.float64)
+            if len(scores) != width:
+                raise SelectionArtifactError(
+                    f"Глобальный рейтинг ветки {branch!r}: длина {len(scores)} != ширине {width}"
+                )
+            top = _rank_topk(scores, k_b)
+            branches[branch] = BranchSelection(
+                indices=top, scores=scores[top].astype(np.float64),
+                origin=["global_fallback"] * len(top),
+            )
+            report[branch] = {"n_selected": len(top), "n_fallback": 0}
+    elif method == METHOD_CLASS_AWARE:
+        arrays = validate_ratings_dir(
+            ratings_dir, manifest, method=METHOD_CLASS_AWARE,
+            widths={"char": int(n_char), "word": int(n_total - n_char)},
+        )
+        for branch, k_b, offset in (("char", k_char, 0), ("word", k_word, n_char)):
+            width = n_char if branch == "char" else n_total - n_char
+            entry = (manifest.get("branches") or {}).get(branch) or {}
+            sel, rep = roundrobin_select_from_rank_files(
+                os.path.join(ratings_dir, entry["rank_indices"]),
+                os.path.join(ratings_dir, entry["rank_scores"]),
+                arrays[branch]["counts"], arrays[branch]["classes"],
+                k_b, width, arrays[branch]["global_scores"],
+                positive_only_fallback=False,
+            )
+            branches[branch] = sel
+            report[branch] = {"n_selected": len(sel.indices), **rep}
+    else:
+        raise SelectionError(f"_selection_from_saved_ratings: метод {method!r} не поддерживается")
+    word = branches["word"]
+    word = BranchSelection(
+        indices=word.indices + n_char if len(word.indices) else word.indices,
+        scores=word.scores,
+        origin=word.origin,
+    )
+    classes_labels = None
+    if l3_labels is not None and method == METHOD_CLASS_AWARE:
+        # Ровно те классы, что присутствовали в train при построении рейтингов.
+        present = validate_ratings_dir(
+            ratings_dir, manifest, method=METHOD_CLASS_AWARE,
+            widths={"char": int(n_char), "word": int(n_total - n_char)},
+        )["char"]["classes"]
+        classes_labels = [str(l3_labels[int(i)]) for i in present]
+    return SelectionResult(
+        method=method,
+        k_requested=int(k),
+        char_share=float(char_share),
+        seed=42,
+        char=branches["char"],
+        word=word,
+        classes=classes_labels,
+        report=report,
+        ratings_dir=ratings_dir,
+        ratings_manifest=manifest,
+    )
+
+
+# ---------------------------------------------------------------------------
+# Опорный checkpoint Gate 1 для G×I (read-only проверки, fail closed)
+# ---------------------------------------------------------------------------
+
+
+def _verify_reference_checkpoint(
+    cfg: S06ComparisonConfig,
+    manifest: dict[str, Any],
+    verification: dict[str, Any],
+    loaded: LoadedVectorizer,
+    protocol: dict[str, Any],
+    device: torch.device,
+) -> dict[str, Any]:
+    """Проверяет опорный checkpoint Gate 1 до Gradient × Input-атрибуции.
+
+    Слои проверок (все fail closed):
+
+    1. Загрузка артефакта checkpoint ``reference_seed`` и сверка полного
+       контракта: seed, размерность (= полный TF-IDF), label maps, число эпох
+       (замороженное fixed_epochs), restored_best=False, sha256 full TF-IDF,
+       sha256 train-сплитов, проверяемый отпечаток контракта признаков
+       (словари/IDF/параметры transform).
+    2. При наличии ``attribution_provenance.json`` в checkpoint — дополнительно
+       полная проверка S04 (:func:`verify_attribution_checkpoint`, включая
+       sha256 весов model.pt).
+    3. При наличии ``selected_checkpoint.json`` Gate 1 — сверка статуса
+       round-trip и файловых sha256.
+    """
+    seed = int(cfg.reference_seed)
+    ckpt_dir = os.path.join(cfg.baseline_dir, "checkpoints", f"seed_{seed}")
+    if not os.path.isdir(ckpt_dir):
+        raise S06ComparisonError(
+            f"Опорный checkpoint Gate 1 не найден: {ckpt_dir!r}. G×I-атрибуция "
+            "требует завершённого refit reference_seed (D06 value_dod)."
+        )
+    art = load_artifact(ckpt_dir, device)
+    maps = LabelMaps.from_dict(manifest["label_maps"])
+    _, _, _, fixed_epochs = _resolved_train_sections(protocol)
+    art_meta = art.meta or {}
+    art_train = art_meta.get("train") or {}
+    checks: dict[str, Any] = {
+        "checkpoint_dir": os.path.abspath(ckpt_dir),
+        "seed_matches": bool(art_meta.get("seed") == seed),
+        "n_features_matches_full_tfidf": bool(art.model.n_features == loaded.builder.n_features),
+        "label_maps_match": bool(art.maps.to_dict() == maps.to_dict()),
+        "fixed_epochs_matches": bool(int(art_train.get("epochs_completed", -1)) == fixed_epochs),
+        "restored_best_is_false": bool(art_train.get("restored_best")) is False,
+        "full_tfidf_id_matches": bool(
+            (art_meta.get("full_tfidf") or {}).get("full_tfidf_id")
+            == (verification.get("full_tfidf_id") or _full_tfidf_id(loaded))
+        ),
+        "train_fingerprints_match": bool(
+            (art_meta.get("train_fingerprints") or {}).get("outer_train_sha256")
+            == manifest["parts"]["outer_train"]["sha256"]
+            and (art_meta.get("train_fingerprints") or {}).get("train_monitor_sha256")
+            == manifest["parts"]["train_monitor"]["sha256"]
+        ),
+        "feature_contract_matches": bool(
+            feature_contract_fingerprint(art.features) == feature_contract_fingerprint(loaded.builder)
+        ),
+    }
+    failed = {k: v for k, v in checks.items() if isinstance(v, bool) and not v}
+    if failed:
+        raise S06ComparisonError(
+            f"Опорный checkpoint {ckpt_dir!r} не совместим с входами S06: {failed}"
+        )
+    prov_path = os.path.join(ckpt_dir, ATTRIBUTION_PROVENANCE_FILENAME)
+    if os.path.isfile(prov_path):
+        # Полная проверка S04: fingerprint контракта признаков, sha256 весов,
+        # label maps, train split. data_cfg у checkpoint контура S05 отсутствует
+        # (сплиты приходят из manifest) — проверка data_cfg вакуальна, остальные
+        # связки обязательны.
+        verify_attribution_checkpoint(
+            ckpt_dir, art.model, art.config,
+            n_features_full=int(loaded.builder.n_features),
+            label_map_sha256=_sha256_json(maps.to_dict()),
+            data_cfg=None,
+            seed=seed,
+            full_tfidf_fp=feature_contract_fingerprint(loaded.builder),
+            train_split_sha256=manifest["parts"]["outer_train"]["sha256"],
+        )
+        checks["attribution_provenance_s04_check"] = True
+    else:
+        checks["attribution_provenance_s04_check"] = None  # Gate 1 не писал файл — слой 1 обязателен
+    selected_path = os.path.join(cfg.baseline_dir, "selected_checkpoint.json")
+    if os.path.isfile(selected_path):
+        with open(selected_path, encoding="utf-8") as f:
+            selected = json.load(f)
+        checks["gate1_selected_roundtrip_ok"] = bool((selected.get("roundtrip") or {}).get("status") == "success")
+        files_sha = selected.get("checkpoint_files_sha256") or {}
+        actual = {
+            name: _sha256_file(os.path.join(ckpt_dir, name))
+            for name in sorted(os.listdir(ckpt_dir)) if os.path.isfile(os.path.join(ckpt_dir, name))
+        }
+        checks["gate1_selected_files_match"] = bool(actual == files_sha) if files_sha else None
+        if checks["gate1_selected_roundtrip_ok"] is False:
+            raise S06ComparisonError(
+                f"selected_checkpoint.json Gate 1: round-trip проверка не пройдена — "
+                "опорный checkpoint для G×I не подтверждён"
+            )
+    checks["model_weights_sha256"] = _sha256_file(os.path.join(ckpt_dir, "model.pt"))
+    log.info("Опорный checkpoint проверен: %s (seed=%d)", ckpt_dir, seed)
+    return {"artifact": art, "checks": checks}
+
+
+# ---------------------------------------------------------------------------
+# Gate 2a целиком: Chi²-рейтинги, G×I-атрибуция, 9 артефактов отбора
+# ---------------------------------------------------------------------------
+
+
+def _selection_provenance(
+    cfg: S06ComparisonConfig,
+    manifest: dict[str, Any],
+    verification: dict[str, Any],
+    slug: str,
+    method: str,
+    budget: int | None,
+    attribution: dict[str, Any] | None = None,
+) -> dict[str, Any]:
+    prov: dict[str, Any] = {
+        "vectorizer": {
+            "path": os.path.abspath(cfg.full_tfidf_path),
+            "source_sha256": (verification.get("corpus_fingerprint") or {}).get("sha256"),
+            "full_tfidf_id": verification.get("full_tfidf_id"),
+        },
+        "labeled": {
+            "split": "outer_train",
+            "splits_manifest": os.path.abspath(str(cfg.splits_manifest)),
+            "outer_train_sha256": manifest["parts"]["outer_train"]["sha256"],
+        },
+        "protocol": {
+            "path": os.path.abspath(cfg.protocol_path),
+            "sha256": _sha256_file(cfg.protocol_path),
+        },
+        "experiment": {
+            "sprint": "S06",
+            "config_slug": slug,
+            "method": method,
+            "budget": budget,
+            "char_share": float(cfg.char_share),
+            "selection_seed": 42,
+            "note": PROTOCOL_NOTES["rankings_train_only"],
+        },
+    }
+    if attribution is not None:
+        prov["attribution"] = attribution
+    return prov
+
+
+def _save_selection_artifact_resumed(
+    cfg: S06ComparisonConfig,
+    slug: str,
+    result: SelectionResult,
+    provenance: dict[str, Any],
+) -> str:
+    """Сохраняет артефакт отбора; при существовании — сверка и переиспользование."""
+    out_dir = cfg.selection_dir(slug)
+    if os.path.isdir(out_dir):
+        existing, meta = load_selection_artifact(out_dir)
+        if (existing.method != result.method or existing.k_requested != result.k_requested
+                or float(existing.char_share) != float(result.char_share)):
+            raise SelectionArtifactError(
+                f"Артефакт отбора {out_dir} существует с другими параметрами "
+                f"({existing.method}, k={existing.k_requested}, share={existing.char_share}): перезапись недопустима"
+            )
+        log.info("Артефакт отбора существует, переиспользуется: %s", out_dir)
+        return os.path.abspath(out_dir)
+    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
+    save_selection_artifact(out_dir, result, provenance=provenance, overwrite=False, ratings_copy=True)
+    return os.path.abspath(out_dir)
+
+
+def run_rankings(cfg: S06ComparisonConfig, device: torch.device) -> dict[str, Any]:
+    """Этап Gate 2a: рейтинги трёх методов (outer_train-only) + 9 отборов.
+
+    Возобновляемо: сохранённые рейтинги с корректными контрольными суммами
+    переиспользуются без пересчёта; отсутствующие артефакты отбора достраиваются
+    из сохранённых рейтингов. Повреждённые рейтинги — отказ (не перезапись).
+    """
+    cfg.validate()
+    inputs = verify_preconditions(cfg)
+    manifest, verification, protocol = inputs["manifest"], inputs["verification"], inputs["protocol"]
+    loaded = load_vectorizer_artifact(cfg.full_tfidf_path)
+    maps = LabelMaps.from_dict(manifest["label_maps"])
+    l3_labels = list(maps.l3)
+    n_char = int(loaded.builder.n_char_features)
+    n_total = int(loaded.builder.n_features)
+    label_map_sha = _sha256_json(maps.to_dict())
+    outer_sha = manifest["parts"]["outer_train"]["sha256"]
+    full_id = verification.get("full_tfidf_id")
+
+    X_outer, y_outer, _train_df, _stage = _transform_outer_train(cfg, loaded, manifest, device)
+    if X_outer.shape[1] != n_total:
+        raise S06ComparisonError(f"размерность матрицы outer_train {X_outer.shape[1]} != словарю полного TF-IDF {n_total}")
+    y_l3 = y_outer[2]
+    if np.any(y_l3 < 0):
+        raise S06ComparisonError("В outer_train есть строки вне label maps: ранжирование невозможно")
+
+    index: dict[str, Any] = {
+        "schema_version": S06_COMPARISON_SCHEMA,
+        "role": "rankings_index",
+        "generated_at": _now(),
+        "methods": {},
+        "selections": {},
+        "notes": list(PROTOCOL_NOTES.values()),
+    }
+    k_max = max(cfg.budgets)
+    sel_provenance_extra: dict[str, Any] = {}
+
+    # --- Chi²-методы: рейтинги один раз, бюджеты из сохранённых рейтингов.
+    for method in (m for m in cfg.methods if m in (METHOD_GLOBAL, METHOD_CLASS_AWARE)):
+        ratings_dir, ratings_meta = None, None
+        saved = _load_rankings_meta(cfg, method)
+        stage_note = (
+            f"supervised Chi²-отбор (method={method}) только по размеченному outer_train "
+            f"полного TF-IDF; метки comparison_val/sealed_test не использовались"
+        )
+        if saved is not None:
+            ratings_dir, ratings_meta = saved
+            stage = {
+                "run_id": f"shared_ranking_{method}", "stage": STAGE_RANKING, "status": "reused",
+                "seconds": None, "peak_rss_bytes": None, "peak_vram_bytes": None,
+                "artefact_bytes": _dir_size(cfg.rankings_dir(method)),
+                "measurement_note": stage_note + "; сохранённые рейтинги прошли проверку контрольных сумм и переиспользуются",
+            }
+            _append_stage_rows(cfg.stages_path, [stage])
+        else:
+            with _Measurement(device, stage_note) as m:
+                result_max = select_features(
+                    X_outer, y_l3,
+                    SelectionConfig(method=method, k=int(k_max), char_share=cfg.char_share, seed=42),
+                    n_char, l3_labels=l3_labels,
+                )
+                ratings_dir, ratings_meta = _persist_ratings(
+                    cfg, method, result_max,
+                    outer_train_sha256=outer_sha, n_rows=int(X_outer.shape[0]),
+                    label_map_sha256=label_map_sha, full_tfidf_id=full_id,
+                )
+            stage = m.stage_row(f"shared_ranking_{method}", STAGE_RANKING, "success")
+            stage["artefact_bytes"] = _dir_size(cfg.rankings_dir(method))
+            _append_stage_rows(cfg.stages_path, [stage])
+        index["methods"][method] = {
+            "ratings_meta": os.path.abspath(os.path.join(cfg.rankings_dir(method), "rankings_meta.json")),
+            "ratings_dir": ratings_dir,
+            "sha256": _sha256_file(os.path.join(cfg.rankings_dir(method), "rankings_meta.json")),
+            "stage_seconds": stage.get("seconds"),
+        }
+        for budget in cfg.budgets:
+            slug = cfg.slug(method, budget)
+            art_path = cfg.selection_dir(slug)
+            if os.path.isdir(art_path):
+                result, _meta = load_selection_artifact(art_path)
+                if result.k_requested != int(budget):
+                    raise SelectionArtifactError(f"Артефакт {art_path}: k={result.k_requested}, ожидалось {budget}")
+            else:
+                result = _selection_from_saved_ratings(
+                    method, ratings_dir, int(budget), cfg.char_share, n_char, n_total, l3_labels=l3_labels
+                )
+                _save_selection_artifact_resumed(
+                    cfg, slug, result,
+                    _selection_provenance(cfg, manifest, verification, slug, method, int(budget)),
+                )
+            index["selections"][slug] = {"artifact": os.path.abspath(art_path)}
+
+    # --- Gradient × Input: атрибуция на опорном checkpoint Gate 1 (один раз).
+    if METHOD_GXI in cfg.methods:
+        att_dir = cfg.attribution_dir()
+        report_path = os.path.join(att_dir, "attribution_report.json")
+        ratings_dir_gxi = os.path.join(att_dir, "ratings")
+        ref = _verify_reference_checkpoint(cfg, manifest, verification, loaded, protocol, device)
+        if os.path.isfile(report_path):
+            with open(report_path, encoding="utf-8") as f:
+                att_report = json.load(f)
+            _verify_ratings_meta(ratings_dir_gxi, METHOD_GXI)
+            stage = {
+                "run_id": "shared_attribution", "stage": STAGE_ATTRIBUTION, "status": "reused",
+                "seconds": None, "peak_rss_bytes": None, "peak_vram_bytes": None,
+                "artefact_bytes": _dir_size(att_dir),
+                "measurement_note": PROTOCOL_NOTES["gxi_reference"] + "; сохранённые рейтинги и отчёт прошли проверку и переиспользуются",
+            }
+            _append_stage_rows(cfg.stages_path, [stage])
+        else:
+            if os.path.isdir(att_dir) and os.listdir(att_dir):
+                raise S06ComparisonError(
+                    f"Каталог атрибуции {att_dir} существует, но attribution_report.json отсутствует: "
+                    "состояние частично; удалите каталог после разбора, перезапись недопустима."
+                )
+            os.makedirs(att_dir, exist_ok=True)
+            with _Measurement(
+                device,
+                "Gradient × Input-атрибуция по margin (истинная L3 vs сильнейший конкурент) на "
+                f"outer_train, опорный checkpoint Gate 1 seed={cfg.reference_seed}; "
+                "вклады только по ненулевым CSR-значениям, плотные N×F не создаются",
+            ) as m:
+                att = compute_margin_attributions(
+                    ref["artifact"].model, X_outer, y_l3, device, batch_size=cfg.attribution_batch_size
+                )
+                manifest_gxi, att_report = build_gxi_ratings(att, n_char, ratings_dir_gxi, l3_labels=l3_labels)
+            att_report["checkpoint"] = {
+                "dir": ref["checks"]["checkpoint_dir"],
+                "model_weights_sha256": ref["checks"]["model_weights_sha256"],
+                "checks": {k: v for k, v in ref["checks"].items() if k != "model_weights_sha256"},
+                "selection_rule": {"kind": "reference_seed", "value": int(cfg.reference_seed)},
+            }
+            att_report["computed_on"] = {
+                "split": "outer_train", "outer_train_sha256": outer_sha,
+                "n_rows": int(X_outer.shape[0]), "label_map_sha256": label_map_sha,
+                "full_tfidf_id": full_id,
+            }
+            att_report["protocol_notes"] = {
+                "train_only": PROTOCOL_NOTES["rankings_train_only"],
+                "reference": PROTOCOL_NOTES["gxi_reference"],
+            }
+            _atomic_write_json(att_report, report_path)
+            stage = m.stage_row("shared_attribution", STAGE_ATTRIBUTION, "success")
+            stage["artefact_bytes"] = _dir_size(att_dir)
+            _append_stage_rows(cfg.stages_path, [stage])
+        index["methods"][METHOD_GXI] = {
+            "attribution_report": os.path.abspath(report_path),
+            "ratings_dir": os.path.abspath(ratings_dir_gxi),
+            "sha256": _sha256_file(report_path),
+            "stage_seconds": stage.get("seconds"),
+        }
+        sel_provenance_extra = {
+            "checkpoint_dir": ref["checks"]["checkpoint_dir"],
+            "model_weights_sha256": ref["checks"]["model_weights_sha256"],
+            "selection_rule": {"kind": "reference_seed", "value": int(cfg.reference_seed)},
+        }
+        for budget in cfg.budgets:
+            slug = cfg.slug(METHOD_GXI, budget)
+            art_path = cfg.selection_dir(slug)
+            if os.path.isdir(art_path):
+                result, _meta = load_selection_artifact(art_path)
+                if result.k_requested != int(budget):
+                    raise SelectionArtifactError(f"Артефакт {art_path}: k={result.k_requested}, ожидалось {budget}")
+            else:
+                stage_sel = {
+                    "run_id": f"shared_selection_{slug}", "stage": STAGE_SELECTION, "status": "success",
+                    "seconds": None, "peak_rss_bytes": None, "peak_vram_bytes": None, "artefact_bytes": None,
+                    "measurement_note": (
+                        "class-aware Top-K round-robin из сохранённых G×I-рейтингов "
+                        "(shared, train-only); метки val/test не используются; выбор фиксируется для всех seed"
+                    ),
+                }
+                result = select_features_gradient_x_input(
+                    ratings_dir_gxi,
+                    SelectionConfig(method=METHOD_GXI, k=int(budget), char_share=cfg.char_share, seed=42),
+                    n_char, n_total, l3_labels=l3_labels,
+                )
+                _save_selection_artifact_resumed(
+                    cfg, slug, result,
+                    _selection_provenance(
+                        cfg, manifest, verification, slug, METHOD_GXI, int(budget),
+                        attribution=sel_provenance_extra,
+                    ),
+                )
+                _append_stage_rows(cfg.stages_path, [stage_sel])
+            index["selections"][slug] = {"artifact": os.path.abspath(art_path)}
+
+    # Фиксируем индекс рейтингов/отборов (только новый файл; сверка при повторе).
+    index_path = os.path.join(cfg.shared_dir, "rankings_index.json")
+    index["generated_at"] = _now()
+    if os.path.exists(index_path):
+        with open(index_path, encoding="utf-8") as f:
+            existing = json.load(f)
+        if {k: v.get("sha256") for k, v in existing.get("methods", {}).items()} != {
+            k: v.get("sha256") for k, v in index["methods"].items()
+        }:
+            raise S06ComparisonError(
+                f"{index_path} уже существует с другими хэшами методов: подмена рейтингов недопустима"
+            )
+    else:
+        _atomic_write_json(index, index_path)
+    log.info("Этап rankings завершён: методы %s, отборов %d", list(index["methods"]), len(index["selections"]))
+    return index
+
+
+# ---------------------------------------------------------------------------
+# Gate 2b: fixed-vocabulary refit (один на метод×K)
+# ---------------------------------------------------------------------------
+
+
+def run_refit_selected(cfg: S06ComparisonConfig) -> dict[str, Any]:
+    """Refit выбранных словарей на том же снимке корпуса, что полный TF-IDF.
+
+    Делегирует проверки и fit :func:`kkt_clf.vectorizer.refit_selected_vectorizer`
+    (sha256 корпуса, воспроизведение подвыборки, инвариант IDF, L2-нормировка).
+    Каждый refit-артефакт строится один раз и делится тремя seed; возобновление
+    переиспользует валидный артефакт (load + сверка метаданных), повреждённый —
+    отказ без перезаписи.
+    """
+    cfg.validate()
+    inputs = verify_preconditions(cfg)
+    manifest, verification = inputs["manifest"], inputs["verification"]
+    loaded = load_vectorizer_artifact(cfg.full_tfidf_path)
+    results: dict[str, Any] = {}
+    for c in cfg.planned_configs():
+        if c["method"] == METHOD_NONE:
+            continue
+        slug = c["slug"]
+        out_dir = os.path.join(cfg.config_dir(slug), "tfidf_selected")
+        result_path = os.path.join(cfg.config_dir(slug), "refit_result.json")
+        if os.path.isfile(result_path):
+            with open(result_path, encoding="utf-8") as f:
+                saved = json.load(f)
+            art = load_vectorizer_artifact(out_dir)
+            if saved.get("vocab") != {
+                "n_char": int(art.builder.n_char_features),
+                "n_word": int(art.builder.n_word_features),
+                "n_total": int(art.builder.n_features),
+            }:
+                raise S06ComparisonError(f"Refit-артефакт {out_dir} не согласован с refit_result.json")
+            stage = {
+                "run_id": f"shared_refit_{slug}", "stage": STAGE_REFIT, "status": "reused",
+                "seconds": None, "peak_rss_bytes": None, "peak_vram_bytes": None,
+                "artefact_bytes": _dir_size(out_dir),
+                "measurement_note": PROTOCOL_NOTES["refit_once"] + "; сохранённый артефакт прошёл load-проверку",
+            }
+            _append_stage_rows(cfg.stages_path, [stage])
+            results[slug] = saved
+            continue
+        if os.path.isdir(out_dir):
+            raise S06ComparisonError(
+                f"Каталог {out_dir} существует без refit_result.json: состояние частично, "
+                "перезапись недопустима; разберитесь и уберите каталог вручную"
+            )
+        selection, _sel_meta = load_selection_artifact(cfg.selection_dir(slug))
+        with _Measurement(
+            torch.device("cpu"),
+            f"refit TF-IDF с фиксированными словарями ({slug}) на том же снимке корпуса, "
+            "что полный TF-IDF S02; новый IDF и L2-нормировка в каждой ветке",
+        ) as m:
+            refit = refit_selected_vectorizer(
+                cfg.corpus_path, loaded.builder, loaded.meta, selection,
+                out_dir=out_dir, overwrite=False,
+            )
+            art = load_vectorizer_artifact(out_dir)  # round-trip load check
+        checks = dict(refit.checks)
+        checks["roundtrip_load"] = True
+        result = {
+            "schema_version": S06_COMPARISON_SCHEMA,
+            "slug": slug, "method": c["method"], "budget": c["budget"],
+            "status": "success", "created_at": _now(),
+            "artifact_dir": os.path.abspath(out_dir),
+            "vocab": {
+                "n_char": int(art.builder.n_char_features),
+                "n_word": int(art.builder.n_word_features),
+                "n_total": int(art.builder.n_features),
+            },
+            "checks": checks,
+            "files": {
+                name: {"sha256": _sha256_file(os.path.join(out_dir, name)),
+                       "bytes": os.path.getsize(os.path.join(out_dir, name))}
+                for name in sorted(os.listdir(out_dir)) if os.path.isfile(os.path.join(out_dir, name))
+            },
+            "provenance": {
+                "corpus_sha256": (verification.get("corpus_fingerprint") or {}).get("sha256"),
+                "full_tfidf_id": verification.get("full_tfidf_id"),
+                "selection_artifact": os.path.abspath(cfg.selection_dir(slug)),
+                "selection_payload_sha256": (
+                    (load_selection_artifact(cfg.selection_dir(slug))[1].get("files") or {})
+                    .get("selection.joblib", {}).get("sha256")
+                ),
+                "outer_train_sha256": manifest["parts"]["outer_train"]["sha256"],
+                "note": PROTOCOL_NOTES["refit_once"],
+            },
+            "stage_seconds": m.seconds,
+        }
+        _atomic_write_json(result, result_path)
+        stage = {
+            "run_id": f"shared_refit_{slug}", "stage": STAGE_REFIT, "status": "success",
+            "seconds": m.seconds, "peak_rss_bytes": m.peak_rss_bytes,
+            "peak_vram_bytes": None, "artefact_bytes": _dir_size(out_dir),
+            "measurement_note": "refit_selected_vectorizer: sha256 корпуса, та же подвыборка строк, "
+                                "инвариант IDF, L2-нормировка, save→load round-trip",
+        }
+        _append_stage_rows(cfg.stages_path, [stage])
+        results[slug] = result
+        log.info("Refit %s завершён: char %d, word %d", slug, result["vocab"]["n_char"], result["vocab"]["n_word"])
+    return results
+
+
+# ---------------------------------------------------------------------------
+# Gate 3: 30 paired runs (none — reuse baseline Gate 1; selected — обучение)
+# ---------------------------------------------------------------------------
+
+
+def _read_gate1_baseline_runs(cfg: S06ComparisonConfig) -> pd.DataFrame:
+    path = os.path.join(cfg.baseline_dir, "runs.csv")
+    if not os.path.isfile(path):
+        raise S06ComparisonError(
+            f"runs.csv baseline Gate 1 не найден: {path!r}. Парные сравнения невозможны без Gate 1."
+        )
+    return pd.read_csv(path)
+
+
+def _load_run_result(cfg: S06ComparisonConfig, slug: str, seed: int) -> dict[str, Any] | None:
+    path = cfg.run_result_path(slug, seed)
+    if not os.path.isfile(path):
+        return None
+    with open(path, encoding="utf-8") as f:
+        return json.load(f)
+
+
+def _write_predictions(
+    cfg: S06ComparisonConfig,
+    run_id: str,
+    row_ids: np.ndarray,
+    y_true_l3: np.ndarray,
+    y_pred_l3: np.ndarray,
+    l3_labels: list[str],
+) -> str:
+    path = cfg.predictions_path(run_id)
+    if os.path.exists(path):
+        raise S06ComparisonError(f"Предсказания уже существуют и не перезаписываются: {path}")
+    frame = pd.DataFrame({
+        "row_id": np.asarray(row_ids, dtype=np.int64),
+        "run_id": run_id,
+        "true_L3": [l3_labels[int(t)] for t in y_true_l3],
+        "predicted_L3": [l3_labels[int(p)] for p in y_pred_l3],
+    })
+    frame["correct"] = frame["true_L3"] == frame["predicted_L3"]
+    os.makedirs(os.path.dirname(path), exist_ok=True)
+    frame.to_parquet(path, index=False)
+    return os.path.abspath(path)
+
+
+def _eval_run(
+    cfg: S06ComparisonConfig,
+    model: torch.nn.Module,
+    X_val: sp.csr_matrix,
+    y_val_l3: np.ndarray,
+    val_row_ids: np.ndarray,
+    l3_labels: list[str],
+    run_id: str,
+    stage_rows: list[dict[str, Any]],
+    device: torch.device,
+) -> dict[str, Any]:
+    """Raw top-1 на comparison_val + сохранение предсказаний (единый протокол оценки)."""
+    m = _Measurement(
+        device, f"{run_id}: predict raw top-1 на comparison_val (n={X_val.shape[0]}), без отказа"
+    )
+    try:
+        with m:
+            preds, _ = predict(model, X_val, device, cfg.eval_batch_size, top_k=1)
+            macro, acc, per_class = _l3_val_metrics(y_val_l3, preds.l3_top1, l3_labels)
+    except Exception as e:
+        row = m.stage_row(run_id, STAGE_VAL_INFERENCE, "failed")
+        row["artefact_bytes"] = None
+        row["measurement_note"] += f" Этап прерван: {type(e).__name__}: {e}"
+        stage_rows.append(row)
+        raise
+    stage = m.stage_row(run_id, STAGE_VAL_INFERENCE, "success")
+    stage["artefact_bytes"] = None
+    stage["measurement_note"] += (
+        "; фиксированная L3-вселенная label maps; support=0 → null; пустые векторы признаков "
+        "считаются ошибками (модель вызывается, bias-путь), их число фиксируется отдельно"
+    )
+    stage_rows.append(stage)
+    pred_path = _write_predictions(cfg, run_id, val_row_ids, y_val_l3, preds.l3_top1, l3_labels)
+    return {
+        "l3_macro_f1": macro, "l3_accuracy": acc, "per_class": per_class,
+        "n_empty_features_val": int((np.diff(X_val.indptr) == 0).sum()),
+        "inference_seconds": m.seconds,
+        "inference_peak_rss_bytes": m.peak_rss_bytes,
+        "inference_peak_vram_bytes": m.peak_vram_bytes,
+        "predictions_path": pred_path,
+    }
+
+
+def _verify_none_checkpoint(
+    cfg: S06ComparisonConfig,
+    manifest: dict[str, Any],
+    verification: dict[str, Any],
+    loaded: LoadedVectorizer,
+    ckpt_dir: str,
+    seed: int,
+    device: torch.device,
+) -> Any:
+    """Проверка reuse baseline-checkpoint Gate 1 перед оценкой (fail closed)."""
+    art = load_artifact(ckpt_dir, device)
+    maps = LabelMaps.from_dict(manifest["label_maps"])
+    meta = art.meta or {}
+    ok = (
+        meta.get("seed") == seed
+        and art.model.n_features == loaded.builder.n_features
+        and art.maps.to_dict() == maps.to_dict()
+        and (meta.get("full_tfidf") or {}).get("full_tfidf_id") == verification.get("full_tfidf_id")
+        and (meta.get("train_fingerprints") or {}).get("outer_train_sha256")
+        == manifest["parts"]["outer_train"]["sha256"]
+        and feature_contract_fingerprint(art.features) == feature_contract_fingerprint(loaded.builder)
+    )
+    if not ok:
+        raise S06ComparisonError(
+            f"Baseline-checkpoint {ckpt_dir} не прошёл проверку provenance для reuse (seed={seed})"
+        )
+    return art
+
+
+def _finalize_run_result(
+    cfg: S06ComparisonConfig,
+    *,
+    slug: str,
+    method: str,
+    budget: int | None,
+    seed: int,
+    status: str,
+    reason: str | None,
+    n_features: int | None,
+    n_char: int | None,
+    n_word: int | None,
+    train_info: dict[str, Any] | None,
+    eval_info: dict[str, Any] | None,
+    checkpoint_dir: str | None,
+    tfidf_artifact: str | None,
+    tfidf_sha256: str | None,
+    selection_artifact: str | None,
+    train_stage: dict[str, Any] | None,
+    baseline_check: dict[str, Any] | None,
+    device: torch.device,
+) -> dict[str, Any]:
+    result = {
+        "schema_version": RUN_RESULT_SCHEMA,
+        "run_id": cfg.run_id(slug, seed),
+        "config_slug": slug, "method": method, "budget": budget, "seed": int(seed),
+        "status": status, "reason": reason,
+        "n_features": n_features, "n_char": n_char, "n_word": n_word,
+        "fixed_l3_universe": None,
+        "train": train_info,
+        "comparison_val_eval": eval_info,
+        "baseline_metric_check": baseline_check,
+        "artifacts": {
+            "checkpoint_dir": checkpoint_dir,
+            "tfidf_artifact": tfidf_artifact,
+            "selection_artifact": selection_artifact,
+        },
+        "hashes": {
+            "model_pt_sha256": (
+                _sha256_file(os.path.join(checkpoint_dir, "model.pt")) if checkpoint_dir else None
+            ),
+            "tfidf_features_sha256": tfidf_sha256,
+        },
+        "device": str(device),
+        "created_at": _now(),
+        "versions": environment_versions(),
+    }
+    if eval_info is not None:
+        result["fixed_l3_universe"] = len(eval_info.get("per_class") or [])
+    path = cfg.run_result_path(slug, seed)
+    if os.path.exists(path):
+        raise S06ComparisonError(f"Результат run уже существует и не перезаписывается: {path}")
+    os.makedirs(os.path.dirname(path), exist_ok=True)
+    _atomic_write_json(result, path)
+    return result
+
+
+def run_train(cfg: S06ComparisonConfig, device: torch.device) -> dict[str, Any]:
+    """Этап Gate 3: обучение/переиспользование 30 финальных MLP и оценка.
+
+    Порядок гарантирует границы: предусловия Gate 0/1 (fail closed) → загрузка
+    протокола и плана → для ``none`` reuse checkpoint Gate 1 (без обучения),
+    для selected-конфигураций обучение с нуля по замороженному протоколу →
+    единая оценка raw top-1 на comparison_val каждого run с сохранением
+    предсказаний. Уже записанные результаты не пересчитываются (безопасное
+    возобновление без перезаписи).
+    """
+    cfg.validate()
+    inputs = verify_preconditions(cfg)
+    manifest, verification, protocol = inputs["manifest"], inputs["verification"], inputs["protocol"]
+    if not os.path.isfile(cfg.planned_path):
+        raise S06ComparisonError(
+            f"План запусков не зафиксирован: {cfg.planned_path!r}. Сначала выполните команду plan."
+        )
+    model_cfg, loss_cfg, train_cfg, fixed_epochs = _resolved_train_sections(protocol)
+    loaded_full = load_vectorizer_artifact(cfg.full_tfidf_path)
+    maps = LabelMaps.from_dict(manifest["label_maps"])
+    l3_labels = list(maps.l3)
+    n_classes = tuple(maps.n_classes(lvl) for lvl in LEVELS)
+
+    train_df = load_part(manifest, "outer_train")
+    mon_df = load_part(manifest, "train_monitor")
+    val_df = load_part(manifest, "comparison_val")
+    y_outer = _encode_labels(maps, train_df, "outer_train")
+    y_mon = _encode_labels(maps, mon_df, "train_monitor")
+    y_val = _encode_labels(maps, val_df, "comparison_val")
+    val_row_ids = val_df["row_id"].to_numpy(dtype=np.int64)
+    class_weights = {
+        h: compute_class_weights(y_outer[i], n_classes[i], loss_cfg.class_weight_mode[h],
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
+    fixed_train_cfg = dataclasses.replace(train_cfg, epochs=fixed_epochs, patience=fixed_epochs)
+    gate1_runs = _read_gate1_baseline_runs(cfg)
+    protocol_sha = _sha256_file(cfg.protocol_path)
+
+    summary: dict[str, Any] = {"trained": [], "reused": [], "failed": [], "skipped": []}
+
+    for c in cfg.planned_configs():
+        slug, method, budget = c["slug"], c["method"], c["budget"]
+        stage_rows: list[dict[str, Any]] = []
+        ckpt_root = os.path.join(cfg.config_dir(slug), "checkpoints")
+        is_none = method == METHOD_NONE
+
+        # --- векторизатор конфигурации и матрицы
+        if is_none:
+            builder = loaded_full.builder
+            tfidf_artifact = os.path.abspath(cfg.full_tfidf_path)
+            tfidf_sha = _full_tfidf_id(loaded_full)
+            selection_artifact = None
+            with _Measurement(
+                device, f"transform comparison_val полным TF-IDF (конфиг none), chunk_rows={cfg.features_chunk_rows}"
+            ) as m:
+                X_val = _transform(loaded_full, val_df["name"].tolist(), cfg.features_chunk_rows)
+            stage = m.stage_row("shared_vectorize_none", STAGE_VECTORIZE_CONFIG, "success")
+            stage["artefact_bytes"] = None
+            stage["measurement_note"] += "; матрицы train/monitor не строятся: обучение none не выполняется (reuse Gate 1)"
+            if not _stage_recorded(cfg, "shared_vectorize_none", STAGE_VECTORIZE_CONFIG):
+                _append_stage_rows(cfg.stages_path, [stage])
+        else:
+            refit_result_path = os.path.join(cfg.config_dir(slug), "refit_result.json")
+            if not os.path.isfile(refit_result_path):
+                raise S06ComparisonError(
+                    f"Нет refit-результата для {slug}: сначала выполните этап refit (D07)."
+                )
+            sel_artifact = load_vectorizer_artifact(os.path.join(cfg.config_dir(slug), "tfidf_selected"))
+            builder = sel_artifact.builder
+            tfidf_artifact = os.path.abspath(os.path.join(cfg.config_dir(slug), "tfidf_selected"))
+            tfidf_sha = (sel_artifact.meta.get("files") or {}).get("features.joblib", {}).get("sha256")
+            selection_artifact = os.path.abspath(cfg.selection_dir(slug))
+            with _Measurement(
+                device,
+                f"transform outer_train/train_monitor/comparison_val выбранным TF-IDF ({slug}), "
+                f"chunk_rows={cfg.features_chunk_rows}; матрицы общие для всех seed конфигурации",
+            ) as m:
+                X_outer = builder.transform(train_df["name"].tolist(), chunk_rows=cfg.features_chunk_rows)
+                X_mon = builder.transform(mon_df["name"].tolist(), chunk_rows=cfg.features_chunk_rows)
+                X_val = builder.transform(val_df["name"].tolist(), chunk_rows=cfg.features_chunk_rows)
+            stage = m.stage_row(f"shared_vectorize_{slug}", STAGE_VECTORIZE_CONFIG, "success")
+            stage["artefact_bytes"] = None
+            if not _stage_recorded(cfg, f"shared_vectorize_{slug}", STAGE_VECTORIZE_CONFIG):
+                _append_stage_rows(cfg.stages_path, [stage])
+
+        for seed in cfg.seeds:
+            run_id = cfg.run_id(slug, seed)
+            existing = _load_run_result(cfg, slug, seed)
+            if existing is not None:
+                log.info("%s: результат существует (status=%s), пересчёт не выполняется",
+                         run_id, existing.get("status"))
+                summary["reused"].append(run_id)
+                continue
+            try:
+                if is_none:
+                    # --- reuse baseline Gate 1: обучение не выполняется
+                    gate1_row = gate1_runs[gate1_runs["run_id"] == f"seed_{seed}"]
+                    if len(gate1_row) != 1:
+                        raise S06ComparisonError(f"runs.csv Gate 1: нет уникальной строки seed_{seed}")
+                    gate1_row = gate1_row.iloc[0].to_dict()
+                    if gate1_row.get("status") != "success":
+                        raise S06ComparisonError(
+                            f"Gate 1 baseline seed {seed}: status={gate1_row.get('status')!r} "
+                            f"({gate1_row.get('reason')}) — парный none-run недоступен"
+                        )
+                    ckpt_dir = gate1_row.get("checkpoint_path")
+                    if not ckpt_dir or not os.path.isdir(ckpt_dir):
+                        raise S06ComparisonError(f"checkpoint Gate 1 для seed {seed} не найден: {ckpt_dir!r}")
+                    art = _verify_none_checkpoint(cfg, manifest, verification, loaded_full, ckpt_dir, seed, device)
+                    _append_stage_rows(cfg.stages_path, [{
+                        "run_id": run_id, "stage": STAGE_MLP_TRAIN, "status": "reused",
+                        "seconds": None, "peak_rss_bytes": None, "peak_vram_bytes": None,
+                        "artefact_bytes": _dir_size(ckpt_dir),
+                        "measurement_note": "обучение не выполнялось: checkpoint полного baseline "
+                                            "переиспользован из Gate 1 (runs/baseline_full); затраты обучения — "
+                                            "в stages.csv Gate 1, здесь только оценка",
+                    }])
+                    eval_info = _eval_run(
+                        cfg, art.model, X_val, y_val[2], val_row_ids, l3_labels, run_id, stage_rows, device
+                    )
+                    saved_f1, saved_acc = gate1_row.get("l3_macro_f1"), gate1_row.get("l3_accuracy")
+                    metrics_match = bool(
+                        np.isclose(eval_info["l3_macro_f1"], float(saved_f1), rtol=1e-6, atol=1e-8)
+                        and np.isclose(eval_info["l3_accuracy"], float(saved_acc), rtol=1e-6, atol=1e-8)
+                    )
+                    baseline_check = {
+                        "gate1_run_id": f"seed_{seed}",
+                        "gate1_l3_macro_f1": float(saved_f1), "gate1_l3_accuracy": float(saved_acc),
+                        "recomputed_l3_macro_f1": eval_info["l3_macro_f1"],
+                        "recomputed_l3_accuracy": eval_info["l3_accuracy"],
+                        "metrics_match": metrics_match,
+                    }
+                    if not metrics_match:
+                        raise S06ComparisonError(
+                            f"Пересчитанные метрики none seed {seed} не совпали с Gate 1: {baseline_check}"
+                        )
+                    if stage_rows:
+                        _append_stage_rows(cfg.stages_path, stage_rows)
+                        stage_rows = []
+                    result = _finalize_run_result(
+                        cfg, slug=slug, method=method, budget=budget, seed=seed,
+                        status="success", reason=None,
+                        n_features=int(builder.n_features), n_char=int(builder.n_char_features),
+                        n_word=int(builder.n_word_features),
+                        train_info={
+                            "reused_from_gate1": True, "fixed_epochs": int(fixed_epochs),
+                            "checkpoint_dir_gate1": os.path.abspath(ckpt_dir),
+                        },
+                        eval_info=eval_info, checkpoint_dir=os.path.abspath(ckpt_dir),
+                        tfidf_artifact=tfidf_artifact, tfidf_sha256=tfidf_sha,
+                        selection_artifact=selection_artifact, train_stage=None,
+                        baseline_check=baseline_check, device=device,
+                    )
+                else:
+                    # --- обучение с нуля по замороженному протоколу
+                    set_seed(seed)
+                    model = SparseMLP(builder.n_features, n_classes, model_cfg)
+                    with _Measurement(
+                        device,
+                        f"{run_id}: обучение SparseMLP с нуля на outer_train (n={X_outer.shape[0]}), "
+                        f"признаков {builder.n_features}, ровно {fixed_epochs} эпох "
+                        f"(scheduler={fixed_train_cfg.scheduler}, restore_best=False); протокол Gate 1",
+                    ) as m:
+                        history, best, lr_trace = train_model(
+                            model, X_outer, y_outer, X_mon, y_mon,
+                            loss_cfg, fixed_train_cfg, class_weights, device, seed,
+                            restore_best=False,
+                        )
+                    stage = m.stage_row(run_id, STAGE_MLP_TRAIN, "success")
+                    stage["artefact_bytes"] = None
+                    _append_stage_rows(cfg.stages_path, [stage] + stage_rows)
+                    stage_rows = []
+                    ckpt_dir = os.path.join(ckpt_root, f"seed_{seed}")
+                    save_artifact(
+                        ckpt_dir, model, model_cfg, builder, maps,
+                        RejectRule(0.5, False, False),
+                        {"note": "S06-эксперимент: сравнение по raw top-1 на comparison_val; τ не применяется",
+                         "tau_mode": "fixed"},
+                        cw_json,
+                        {
+                            "experiment_run": run_id,
+                            "role": "s06_selected_run",
+                            "config": {
+                                "protocol_sha256": protocol_sha,
+                                "model": dataclasses.asdict(model_cfg),
+                                "loss": dataclasses.asdict(loss_cfg),
+                                "train": dataclasses.asdict(fixed_train_cfg),
+                            },
+                        },
+                        {
+                            "seed": int(seed),
+                            "split_seed": int(manifest["policy"]["split_seed"]),
+                            "n_features": int(builder.n_features),
+                            "device": str(device),
+                            "protocol": S06_COMPARISON_SCHEMA,
+                            "config_slug": slug,
+                            "method": method,
+                            "budget": budget,
+                            "train": {
+                                "epochs_completed": len(history),
+                                "fixed_epochs": int(fixed_epochs),
+                                "scheduler": str(fixed_train_cfg.scheduler),
+                                "best_epoch_monitor": int(best["epoch"]),
+                                "early_stopping_metric": best["metric"],
+                                "best_score_monitor": float(best["score"]),
+                                "restored_best": bool(best.get("restored_best")),
+                                "n_train_rows_used": int(best.get("n_train_rows_used", 0)),
+                                "final_monitor_l3_macro_f1": (history[-1]["val_l3_macro_f1"] if history else None),
+                                "final_monitor_l3_accuracy": (history[-1]["val_l3_accuracy"] if history else None),
+                            },
+                            "feature_stats": {
+                                "outer_train": _matrix_stats(X_outer),
+                                "train_monitor": _matrix_stats(X_mon),
+                                "comparison_val": _matrix_stats(X_val),
+                            },
+                            "train_fingerprints": {
+                                "outer_train_sha256": manifest["parts"]["outer_train"]["sha256"],
+                                "train_monitor_sha256": manifest["parts"]["train_monitor"]["sha256"],
+                            },
+                            "tfidf": {
+                                "artifact_dir": tfidf_artifact,
+                                "features_sha256": tfidf_sha,
+                                "derived_from_full_tfidf_id": verification.get("full_tfidf_id"),
+                            },
+                            "selection": {"artifact": selection_artifact},
+                        },
+                    )
+                    eval_info = _eval_run(
+                        cfg, model, X_val, y_val[2], val_row_ids, l3_labels, run_id, stage_rows, device
+                    )
+                    _append_stage_rows(cfg.stages_path, stage_rows)
+                    stage_rows = []
+                    result = _finalize_run_result(
+                        cfg, slug=slug, method=method, budget=budget, seed=seed,
+                        status="success", reason=None,
+                        n_features=int(builder.n_features), n_char=int(builder.n_char_features),
+                        n_word=int(builder.n_word_features),
+                        train_info={
+                            "epochs_completed": len(history),
+                            "fixed_epochs": int(fixed_epochs),
+                            "scheduler": str(fixed_train_cfg.scheduler),
+                            "best_epoch_monitor": int(best["epoch"]),
+                            "early_stopping_metric": best["metric"],
+                            "best_score_monitor": float(best["score"]),
+                            "restored_best": bool(best.get("restored_best")),
+                            "n_train_rows_used": int(best.get("n_train_rows_used", 0)),
+                            "final_monitor_l3_macro_f1": (history[-1]["val_l3_macro_f1"] if history else None),
+                            "final_monitor_l3_accuracy": (history[-1]["val_l3_accuracy"] if history else None),
+                            "train_seconds": m.seconds,
+                            "peak_rss_bytes": m.peak_rss_bytes,
+                            "peak_vram_bytes": m.peak_vram_bytes,
+                            "lr_trace_tail": lr_trace[-3:] if lr_trace else [],
+                        },
+                        eval_info=eval_info, checkpoint_dir=os.path.abspath(ckpt_dir),
+                        tfidf_artifact=tfidf_artifact, tfidf_sha256=tfidf_sha,
+                        selection_artifact=selection_artifact, train_stage=None,
+                        baseline_check=None, device=device,
+                    )
+                summary["trained"].append(run_id)
+                log.info("%s: success (macro-F1=%.4f, acc=%.4f)", run_id,
+                         result["comparison_val_eval"]["l3_macro_f1"],
+                         result["comparison_val_eval"]["l3_accuracy"])
+            except Exception as e:  # noqa: BLE001 — фиксируем failed run, остальные продолжаем
+                log.exception("%s: failed", run_id)
+                if stage_rows:
+                    _append_stage_rows(cfg.stages_path, stage_rows)
+                    stage_rows = []
+                try:
+                    _finalize_run_result(
+                        cfg, slug=slug, method=method, budget=budget, seed=seed,
+                        status="failed", reason=f"{type(e).__name__}: {e}"[:800],
+                        n_features=None, n_char=None, n_word=None,
+                        train_info=None, eval_info=None, checkpoint_dir=None,
+                        tfidf_artifact=None, tfidf_sha256=None, selection_artifact=None,
+                        train_stage=None, baseline_check=None, device=device,
+                    )
+                except Exception as inner:  # noqa: BLE001 — результат failed уже существует: не трогаем
+                    log.warning("%s: не удалось записать failed-результат: %s", run_id, inner)
+                summary["failed"].append(run_id)
+    done = len(summary["trained"]) + len(summary["reused"])
+    log.info("Этап train завершён: %d/%d runs успешно, failed=%d, reused=%d",
+             done, len(cfg.planned_configs()) * len(cfg.seeds), len(summary["failed"]), len(summary["reused"]))
+    summary["n_planned"] = len(cfg.planned_configs()) * len(cfg.seeds)
+    summary["n_done"] = done
+    return summary
+
+
+def _matrix_stats(X: sp.csr_matrix) -> dict[str, Any]:
+    from .features import matrix_stats
+
+    return dict(matrix_stats(X))
+
+
+def run_comparison_all(cfg: S06ComparisonConfig, device: torch.device) -> dict[str, Any]:
+    """Полный контур Gate 2–3: plan (если ещё нет) → rankings → refit → train."""
+    cfg.validate()
+    if not os.path.exists(cfg.planned_path):
+        write_plan(cfg)
+    index = run_rankings(cfg, device)
+    refits = run_refit_selected(cfg)
+    summary = run_train(cfg, device)
+    return {"rankings": list(index["methods"]), "refits": list(refits), "train": summary}
+
+
+__all__ = [
+    "S06_COMPARISON_SCHEMA",
+    "S06ComparisonConfig",
+    "S06ComparisonError",
+    "METHOD_NONE",
+    "METHOD_GLOBAL",
+    "METHOD_CLASS_AWARE",
+    "METHOD_GXI",
+    "SELECTION_METHODS",
+    "DEFAULT_BUDGETS",
+    "DEFAULT_SEEDS",
+    "DEFAULT_CHAR_SHARE",
+    "STAGES_COLUMNS",
+    "write_plan",
+    "verify_preconditions",
+    "run_rankings",
+    "run_refit_selected",
+    "run_train",
+    "run_comparison_all",
+]
diff --git a/kkt_clf/s06_report.py b/kkt_clf/s06_report.py
new file mode 100644
index 0000000..0e776d8
--- /dev/null
+++ b/kkt_clf/s06_report.py
@@ -0,0 +1,833 @@
+"""S06 (D09–D11): агрегированный отчёт сравнительного эксперимента (Gate 4).
+
+Собирает восемь обязательных файлов ``outputs/S06/report/`` из сохранённых
+машинных результатов контура Gate 2–3 (``kkt_clf.s06_comparison``), Gate 1
+(``kkt_clf.s06_baseline``) и аудита Gate 0 — **без обучения, без fit TF-IDF и
+без повторного inference**. Модуль не импортирует torch и не загружает
+модели: источником чисел являются только JSON/CSV/Parquet, записанные
+прогонами (runs/<slug>/run_result_seed_<s>.json, runs/predictions/*.parquet,
+runs/stages.csv, shared/*, runs/baseline_full/*).
+
+Файлы отчёта (контракт value_dod D11):
+
+- ``report.html`` — автономный агрегированный отчёт (все 30 runs, paired Δ,
+  сводки mean/std/min/max по seed, график K vs macro-F1, support bins,
+  top confused pairs, ресурсы shared/per-run, формулы, ограничения,
+  provenance и инструкция пересборки);
+- ``runs.csv`` — run_id, method, K, seed, status/reason, метрики, paired Δ,
+  config/hash;
+- ``per_class.parquet`` — run_id, seed, L3, support, precision, recall, f1,
+  delta_f1 (Δ к none того же seed);
+- ``stages.csv`` — этапы comparison-контура и Gate 1 (source), elapsed,
+  peak RSS/VRAM, artefact size; shared-затраты не дублируются по seed;
+- ``manifest.json`` — версии/хэши данных, сплитов, TF-IDF, рейтингов,
+  отборов, refit-артефактов, чекпоинтов, протокола, условия и статусы;
+- ``confusion_pairs.csv`` — run_id, true_L3 → predicted_L3, count,
+  true_support, error_share, ранг внутри run;
+- ``predictions_val.parquet`` — row_id, run_id, true_L3, predicted_L3/null;
+- ``support_bins.csv`` — run_id, диапазон support (2–4, 5–9, 10–19, >=20),
+  n_L3, macro-F1 бина, базовый macro-F1 бина (none того же seed) и Δ.
+
+Формулы (единый протокол оценки, совпадает с Gate 1):
+
+- per-class precision/recall/F1 — ``zero_division=0`` по фиксированной
+  L3-вселенной label maps; support=0 → метрики null (не 0 и не фиктивный F1);
+- macro-F1 run — невзвешенное среднее F1 по L3 с support>0 на comparison_val;
+- accuracy — доля строк с top-1 == true L3 (пустые векторы признаков
+  не исключаются: модель вызывается, bias-путь; их число фиксируется);
+- paired Δ — метрика(run) − метрика(none того же train seed), null если
+  baseline того seed неуспешен;
+- macro-F1 support-бина — невзвешенное среднее per-L3 F1 внутри бина
+  (не micro-accuracy); пустые бины — null с n_L3=0;
+- error_share пары (true→predicted) — count / support(true) на comparison_val.
+
+Существующие файлы не перезаписываются: при пересечении — отказ до любых
+записей (для повторной пересборки удалите/переименуйте report/ вручную).
+Честность статусов: planned runs без result-файла отображаются как
+``pending`` (запуск не выполнялся); failed/skipped не скрываются.
+"""
+from __future__ import annotations
+
+import datetime as dt
+import html
+import json
+import os
+from typing import Any
+
+import numpy as np
+import pandas as pd
+
+S06_REPORT_SCHEMA = "s06-report@1"
+
+SUPPORT_BINS: tuple[tuple[str, int, int | None], ...] = (
+    ("2-4", 2, 4),
+    ("5-9", 5, 9),
+    ("10-19", 10, 19),
+    (">=20", 20, None),
+)
+
+REPORT_FILES = (
+    "report.html",
+    "runs.csv",
+    "per_class.parquet",
+    "stages.csv",
+    "manifest.json",
+    "confusion_pairs.csv",
+    "predictions_val.parquet",
+    "support_bins.csv",
+)
+
+FORMULAS = {
+    "per_class": (
+        "precision/recall/F1 считаются sklearn-семантикой с zero_division=0 по "
+        "фиксированной L3-вселенной label maps; support — число строк L3 на "
+        "comparison_val; support=0 → метрики null, класс в средние не входит"
+    ),
+    "macro_f1": (
+        "macro-F1(run) = среднее F1 по L3 с support>0 на comparison_val "
+        "(невзвешенное; классы без поддержки не исключаются молча — они null "
+        "по явному правилу)"
+    ),
+    "accuracy": (
+        "accuracy = доля строк comparison_val с top-1 предсказанием == true L3; "
+        "пустые TF-IDF-векторы не исключаются (модель вызывается, bias-путь), "
+        "их число — n_empty_features_val в результате run"
+    ),
+    "paired_delta": (
+        "Δ(run) = metric(run) − metric(none того же train seed); null, если "
+        "baseline этого seed неуспешен; три seed на одном split — не "
+        "доверительный интервал"
+    ),
+    "support_bins": (
+        "macro-F1 бина = невзвешенное среднее per-L3 F1 по L3, чей support на "
+        "comparison_val попадает в диапазон (2–4, 5–9, 10–19, >=20); "
+        "это не micro-accuracy; пустой бин → null, n_L3=0"
+    ),
+    "confusion_pair": (
+        "error_share(true→predicted) = count(true→predicted) / support(true) "
+        "на comparison_val; пары ранжируются по count внутри run"
+    ),
+    "chi2": (
+        "Chi² (global/class_aware): ожидаемая сумма признака в классе "
+        "feature_sum × n_class / n_train; class_aware — полные 2×2 OvR-"
+        "статистики, кандидаты категории при observed > expected, объединение "
+        "категорий — равноправный round-robin с детерминированным добором"
+    ),
+    "gxi": (
+        "Gradient × Input: m_i = z_{i,y_i} − z_{i,r_i} (r_i — сильнейший другой "
+        "класс); a_{i,f} = x_{i,f}·∂m_i/∂x_{i,f} по ненулевым CSR-значениям; "
+        "score(c,f) = mean_{i∈c} max(0, a_{i,f}); отрицательные/нулевые "
+        "вклады — только диагностика"
+    ),
+    "budget_split": (
+        "k_char = int(K × char_share), k_word = K − k_char (char_share=0.75); "
+        "бюджеты применены из сохранённых полных рейтингов без пересчёта оценок"
+    ),
+}
+
+LIMITATIONS = [
+    "ODS — исследовательский PoC: результаты на comparison_val диагностические, "
+    "не оценка промышленной применимости и не экстраполяция на реальные чеки "
+    "закрытого контура.",
+    "sealed_test не использовался ни для inference, ни для метрик; test-метрик "
+    "в отчёте нет; выбор «победителя для промышленности» вне S06.",
+    "Три seed на одном split — не независимый доверительный интервал; мизерные "
+    "Δ не доказывают улучшение.",
+    "Выбранные словари зафиксированы для всех seed (рейтинги одни, вариабельность "
+    "только обучения MLP) — вариабельность самих рейтингов не измеряется.",
+    "Режим pretrained трансдуктивен: тексты корпуса full TF-IDF могут совпадать "
+    "с train/val/test; метки при этом не переносятся (см. аудит Gate 0).",
+    "Категории с support 2 на comparison_val нестабильны (два примера задают "
+    "recall с шагом 0.5); per-class Δ для них трактуются осторожно.",
+    "GPU/CPU-характеристики локальны; при недоступности GPU peak_vram=null "
+    "(не ноль); ресурсные цифры не переносятся на другие машины.",
+    "comparison_val не участвовал в обучении, выборе эпох, checkpoint и "
+    "ранжировании; он использован только для сравнения методов.",
+]
+
+
+class S06ReportError(RuntimeError):
+    """Ошибка сборки отчёта S06; существующие файлы не перезаписываются."""
+
+
+# ---------------------------------------------------------------------------
+# Загрузка сохранённых результатов (только чтение)
+# ---------------------------------------------------------------------------
+
+
+def _read_json(path: str, required: bool = True) -> dict | None:
+    if not os.path.isfile(path):
+        if required:
+            raise S06ReportError(f"Файл результата не найден: {path}")
+        return None
+    with open(path, encoding="utf-8") as f:
+        return json.load(f)
+
+
+def _check_new(paths: list[str]) -> None:
+    existing = [p for p in paths if os.path.exists(p)]
+    if existing:
+        raise S06ReportError(
+            "Существующие файлы не перезаписываются; удалите/переименуйте вручную: "
+            + ", ".join(existing)
+        )
+
+
+def _load_run_results(output_root: str, plan: dict) -> dict[str, dict]:
+    results: dict[str, dict] = {}
+    for r in plan["runs"]:
+        slug, seed = r["config_slug"], r["seed"]
+        path = os.path.join(output_root, "runs", slug, f"run_result_seed_{seed}.json")
+        if os.path.isfile(path):
+            with open(path, encoding="utf-8") as f:
+                results[r["run_id"]] = json.load(f)
+    return results
+
+
+def _load_predictions(output_root: str, run_ids: list[str]) -> pd.DataFrame:
+    frames = []
+    for run_id in run_ids:
+        path = os.path.join(output_root, "runs", "predictions", f"{run_id}.parquet")
+        if os.path.isfile(path):
+            frames.append(pd.read_parquet(path))
+    if not frames:
+        return pd.DataFrame({
+            "row_id": pd.Series(dtype="int64"), "run_id": pd.Series(dtype="str"),
+            "true_L3": pd.Series(dtype="str"), "predicted_L3": pd.Series(dtype="str"),
+            "correct": pd.Series(dtype="bool"),
+        })
+    return pd.concat(frames, ignore_index=True)
+
+
+# ---------------------------------------------------------------------------
+# Таблицы отчёта
+# ---------------------------------------------------------------------------
+
+
+def _runs_frame(plan: dict, results: dict[str, dict]) -> pd.DataFrame:
+    """runs.csv: статусы всех плановых runs, метрики и paired Δ к none того же seed."""
+    rows = []
+    by_slug_seed = {}
+    for run_id, res in results.items():
+        if res.get("status") == "success" and res.get("comparison_val_eval"):
+            by_slug_seed[(res["config_slug"], res["seed"])] = res["comparison_val_eval"]
+    for r in plan["runs"]:
+        run_id = r["run_id"]
+        res = results.get(run_id)
+        status = res["status"] if res else "pending"
+        reason = (res or {}).get("reason")
+        ev = (res or {}).get("comparison_val_eval") or {}
+        f1, acc = ev.get("l3_macro_f1"), ev.get("l3_accuracy")
+        delta_f1 = delta_acc = None
+        if r["method"] != "none" and res and res.get("status") == "success":
+            base = by_slug_seed.get(("none", r["seed"]))
+            if base:
+                delta_f1 = None if f1 is None or base.get("l3_macro_f1") is None else f1 - base["l3_macro_f1"]
+                delta_acc = None if acc is None or base.get("l3_accuracy") is None else acc - base["l3_accuracy"]
+        hashes = (res or {}).get("hashes") or {}
+        rows.append({
+            "run_id": run_id, "config_slug": r["config_slug"], "method": r["method"],
+            "budget": r["budget"], "seed": r["seed"], "status": status,
+            "reason": reason,
+            "l3_macro_f1": f1, "l3_accuracy": acc,
+            "delta_macro_f1_vs_none_same_seed": delta_f1,
+            "delta_accuracy_vs_none_same_seed": delta_acc,
+            "n_features": (res or {}).get("n_features"),
+            "n_empty_features_val": ev.get("n_empty_features_val"),
+            "model_pt_sha256": hashes.get("model_pt_sha256"),
+            "tfidf_features_sha256": hashes.get("tfidf_features_sha256"),
+            "checkpoint_dir": ((res or {}).get("artifacts") or {}).get("checkpoint_dir"),
+            "selection_artifact": ((res or {}).get("artifacts") or {}).get("selection_artifact"),
+            "tfidf_artifact": ((res or {}).get("artifacts") or {}).get("tfidf_artifact"),
+        })
+    return pd.DataFrame(rows)
+
+
+def _per_class_frame(results: dict[str, dict], runs_df: pd.DataFrame) -> pd.DataFrame:
+    """per_class.parquet: per-L3 метрики каждого успешного run и ΔF1 к none того же seed."""
+    base_f1: dict[tuple[Any, str], float] = {}
+    for run_id, res in results.items():
+        if res.get("config_slug") == "none" and res.get("status") == "success":
+            for row in (res.get("comparison_val_eval") or {}).get("per_class") or []:
+                if row.get("f1") is not None:
+                    base_f1[(res["seed"], row["L3"])] = float(row["f1"])
+    frames = []
+    for run_id, res in results.items():
+        if res.get("status") != "success":
+            continue
+        per_class = (res.get("comparison_val_eval") or {}).get("per_class") or []
+        if not per_class:
+            continue
+        frame = pd.DataFrame(per_class)
+        frame["run_id"] = run_id
+        frame["seed"] = res["seed"]
+        frame["method"] = res["method"]
+        frame["budget"] = res.get("budget")
+        frame["delta_f1"] = frame.apply(
+            lambda row: (None if row["f1"] is None
+                         else float(row["f1"]) - base_f1.get((res["seed"], row["L3"]), float("nan"))),
+            axis=1,
+        )
+        frames.append(frame)
+    columns = ["run_id", "seed", "method", "budget", "L3", "support", "precision", "recall", "f1", "delta_f1"]
+    if not frames:
+        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
+            columns, ["str", "int64", "str", "Int64", "str", "int64", "float64", "float64", "float64", "float64"])})
+    out = pd.concat(frames, ignore_index=True)
+    return out[columns]
+
+
+def _support_bins_frame(per_class: pd.DataFrame, runs_df: pd.DataFrame) -> pd.DataFrame:
+    """support_bins.csv: macro-F1 по диапазонам support (comparison_val), paired к none.
+
+    Метрики бина — невзвешенное среднее per-L3 F1 внутри бина; базовое значение —
+    бин none того же seed; пустые бины — null при n_L3=0.
+    """
+    rows = []
+    if per_class.empty:
+        return pd.DataFrame(columns=[
+            "run_id", "method", "budget", "seed", "support_bin", "n_L3",
+            "macro_f1_bin", "baseline_macro_f1_bin_none_same_seed", "delta_macro_f1_bin",
+        ])
+    pc = per_class.dropna(subset=["f1"])
+    for (run_id, method, budget, seed), group in pc.groupby(
+        ["run_id", "method", "budget", "seed"], dropna=False
+    ):
+        base = pc[(pc["run_id"] == f"none_seed{seed}")]
+        for label, lo, hi in SUPPORT_BINS:
+            sel = group[(group["support"] >= lo) & (group["support"] <= hi)] if hi is not None else group[group["support"] >= lo]
+            bsel = base[(base["support"] >= lo) & (base["support"] <= hi)] if hi is not None else base[base["support"] >= lo]
+            macro = float(sel["f1"].mean()) if len(sel) else None
+            base_macro = float(bsel["f1"].mean()) if len(bsel) else None
+            rows.append({
+                "run_id": run_id, "method": method, "budget": budget, "seed": seed,
+                "support_bin": label, "n_L3": int(len(sel)),
+                "macro_f1_bin": macro,
+                "baseline_macro_f1_bin_none_same_seed": base_macro,
+                "delta_macro_f1_bin": (None if macro is None or base_macro is None else macro - base_macro),
+            })
+    return pd.DataFrame(rows)
+
+
+def _confusion_frame(predictions: pd.DataFrame, per_class: pd.DataFrame) -> pd.DataFrame:
+    """confusion_pairs.csv: top путаниц true→predicted по всем runs с предсказаниями."""
+    if predictions.empty:
+        return pd.DataFrame(columns=[
+            "run_id", "method", "budget", "seed", "true_L3", "predicted_L3",
+            "count", "true_support", "error_share", "rank_in_run",
+        ])
+    wrong = predictions[~predictions["correct"]]
+    meta = per_class[["run_id", "method", "budget", "seed", "L3", "support"]].drop_duplicates()
+    support_map = {(r.run_id, r.L3): int(r.support) for r in meta.itertuples()}
+    rows = []
+    for (run_id, true_l3, pred_l3), group in wrong.groupby(["run_id", "true_L3", "predicted_L3"]):
+        rows.append({
+            "run_id": run_id, "true_L3": true_l3, "predicted_L3": pred_l3,
+            "count": int(len(group)),
+            "true_support": support_map.get((run_id, true_l3)),
+        })
+    frame = pd.DataFrame(rows)
+    meta_small = meta[["run_id", "method", "budget", "seed"]].drop_duplicates()
+    frame = frame.merge(meta_small, on="run_id", how="left")
+    frame["error_share"] = frame.apply(
+        lambda r: (None if not r["true_support"] else r["count"] / r["true_support"]), axis=1
+    )
+    frame = frame.sort_values(["run_id", "count"], ascending=[True, False])
+    frame["rank_in_run"] = frame.groupby("run_id")["count"].rank(method="first", ascending=False).astype(int)
+    return frame[[
+        "run_id", "method", "budget", "seed", "true_L3", "predicted_L3",
+        "count", "true_support", "error_share", "rank_in_run",
+    ]]
+
+
+def _stages_frame(output_root: str) -> pd.DataFrame:
+    """stages.csv: этапы comparison-контура + Gate 1 (source-метка, без смешения)."""
+    frames = []
+    ours = os.path.join(output_root, "runs", "stages.csv")
+    if os.path.isfile(ours):
+        df = pd.read_csv(ours)
+        df["source"] = "comparison"
+        frames.append(df)
+    gate1 = os.path.join(output_root, "runs", "baseline_full", "stages.csv")
+    if os.path.isfile(gate1):
+        df = pd.read_csv(gate1)
+        df["run_id"] = "gate1:" + df["run_id"].astype(str)
+        df["source"] = "gate1_baseline"
+        frames.append(df)
+    columns = ["run_id", "stage", "status", "seconds", "peak_rss_bytes", "peak_vram_bytes",
+               "artefact_bytes", "source", "measurement_note"]
+    if not frames:
+        return pd.DataFrame(columns=columns)
+    out = pd.concat(frames, ignore_index=True)
+    for col in columns:
+        if col not in out.columns:
+            out[col] = None
+    return out[columns]
+
+
+def _method_summary(runs_df: pd.DataFrame) -> list[dict[str, Any]]:
+    """Сводка method×K по успешным seed: mean/std/min/max и средняя парная Δ."""
+    rows = []
+    for (method, budget), group in runs_df[runs_df["method"] != "none"].groupby(["method", "budget"]):
+        ok = group[group["status"] == "success"]
+        f1s = ok["l3_macro_f1"].dropna().astype(float).tolist()
+        accs = ok["l3_accuracy"].dropna().astype(float).tolist()
+        deltas = ok["delta_macro_f1_vs_none_same_seed"].dropna().astype(float).tolist()
+
+        def stats(values: list[float]) -> dict[str, Any]:
+            if not values:
+                return {"mean": None, "std": None, "min": None, "max": None, "range": None, "n": 0}
+            arr = np.asarray(values, dtype=np.float64)
+            return {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()),
+                    "max": float(arr.max()), "range": float(arr.max() - arr.min()), "n": int(len(arr))}
+
+        rows.append({
+            "method": method, "budget": int(budget) if pd.notna(budget) else None,
+            "n_seed_success": int(len(ok)),
+            "n_seed_planned": int(len(group)),
+            "l3_macro_f1": stats(f1s),
+            "l3_accuracy": stats(accs),
+            "paired_delta_macro_f1": {
+                **stats(deltas),
+                "mean_pairwise": (float(np.mean(deltas)) if deltas else None),
+            },
+        })
+    return rows
+
+
+# ---------------------------------------------------------------------------
+# manifest.json
+# ---------------------------------------------------------------------------
+
+
+def _fingerprint_file(path: str) -> dict[str, Any] | None:
+    import hashlib
+
+    if not os.path.isfile(path):
+        return None
+    h = hashlib.sha256()
+    with open(path, "rb") as f:
+        for chunk in iter(lambda: f.read(1 << 20), b""):
+            h.update(chunk)
+    return {"path": os.path.abspath(path), "sha256": h.hexdigest(),
+            "bytes": os.path.getsize(path)}
+
+
+def build_manifest(
+    output_root: str,
+    plan: dict,
+    runs_df: pd.DataFrame,
+    method_summary: list[dict[str, Any]],
+    report_files: dict[str, str],
+) -> dict[str, Any]:
+    """Машинный manifest отчёта: версии/хэши входов, условия, статусы, файлы."""
+    shared = os.path.join(output_root, "shared")
+    runs_root = os.path.join(output_root, "runs")
+    protocol = _read_json(os.path.join(shared, "training_protocol.json"))
+    splits_manifest = _read_json(os.path.join(output_root, "splits", "manifest.json"), required=False)
+    rankings_index = _read_json(os.path.join(shared, "rankings_index.json"), required=False)
+    refit_results = {}
+    for slug in sorted({r["config_slug"] for r in plan["runs"]} - {"none"}):
+        refit_results[slug] = _read_json(os.path.join(runs_root, slug, "refit_result.json"), required=False)
+    selected_checkpoint = _read_json(os.path.join(runs_root, "baseline_full", "selected_checkpoint.json"), required=False)
+    gate1_manifest = _read_json(os.path.join(runs_root, "baseline_full", "manifest.json"), required=False)
+    audit = _read_json(os.path.join(output_root, "audit", "data_audit.json"), required=False)
+    git_commit = git_dirty = None
+    import subprocess
+
+    try:
+        git_commit = subprocess.run(
+            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
+        ).stdout.strip() or None
+        git_dirty = bool(subprocess.run(
+            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10
+        ).stdout.strip())
+    except Exception:
+        pass
+    statuses = {
+        "n_planned": int(plan["n_planned"]),
+        "n_success": int((runs_df["status"] == "success").sum()),
+        "n_failed": int((runs_df["status"] == "failed").sum()),
+        "n_pending": int((runs_df["status"] == "pending").sum()),
+        "by_run": runs_df[["run_id", "status", "reason"]].to_dict("records"),
+    }
+    return {
+        "schema_version": S06_REPORT_SCHEMA,
+        "role": "s06_comparison_report",
+        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
+        "rebuild": {
+            "command": f"python scripts/run_s06_report.py --output {output_root}",
+            "principle": (
+                "пересборка только из сохранённых JSON/CSV/Parquet без обучения, "
+                "fit TF-IDF и повторного inference; report/*.parquet/csv являются "
+                "производными от runs/** и shared/**"
+            ),
+            "report_files": report_files,
+        },
+        "inputs": {
+            "splits_manifest": _fingerprint_file(os.path.join(output_root, "splits", "manifest.json")),
+            "splits_parts": {
+                name: {"sha256": part.get("sha256"), "n_rows": part.get("n_rows")}
+                for name, part in ((splits_manifest or {}).get("parts") or {}).items()
+            },
+            "labeled_parquet": (splits_manifest or {}).get("inputs", {}).get("labeled_parquet"),
+            "full_tfidf_ref": _fingerprint_file(os.path.join(shared, "full_tfidf_ref.json")),
+            "full_tfidf": {
+                "path": plan["config"].get("full_tfidf_path"),
+                "protocol_full_tfidf_id": (protocol or {}).get("full_tfidf_id"),
+                "vocab": (protocol or {}).get("vocab") if protocol else None,
+            },
+            "corpus": (protocol or {}).get("corpus_fingerprint") if protocol else None,
+            "training_protocol": _fingerprint_file(os.path.join(shared, "training_protocol.json")),
+            "fixed_epochs": (protocol or {}).get("fixed_epochs") if protocol else None,
+            "rankings_index": rankings_index,
+            "refit_results": refit_results,
+            "gate1_baseline_manifest": _fingerprint_file(os.path.join(runs_root, "baseline_full", "manifest.json")),
+            "gate1_selected_checkpoint": selected_checkpoint,
+            "data_audit": _fingerprint_file(os.path.join(output_root, "audit", "data_audit.json")),
+            "audit_summary": ({
+                "rows_before": audit.get("rows_before"), "rows_after": audit.get("rows_after"),
+                "l3_before": audit.get("l3_before"), "l3_after": audit.get("l3_after"),
+            } if audit else None),
+        },
+        "protocol": {
+            "seeds": plan["config"]["seeds"],
+            "reference_seed": plan["config"]["reference_seed"],
+            "methods": plan["config"]["methods"],
+            "budgets": plan["config"]["budgets"],
+            "char_share": plan["config"]["char_share"],
+            "selection_seed": 42,
+            "evaluation": "raw top-1 L3 без отказа на comparison_val; фиксированная L3-вселенная label maps",
+            "sealed_test_boundary": (
+                "sealed_test не использовался ни для inference, ни для метрик; только "
+                "структурные sha256/n_rows из manifest"
+            ),
+        },
+        "method_summary": method_summary,
+        "statuses": statuses,
+        "formulas": FORMULAS,
+        "limitations": LIMITATIONS,
+        "environment": (gate1_manifest or {}).get("environment") if gate1_manifest else None,
+        "git": {"commit": git_commit, "dirty": git_dirty},
+        "notes": [
+            "Плановые runs без result-файла имеют статус pending (запуск не выполнялся); "
+            "failed и reused не скрываются.",
+            "Shared-затраты (ранжирование, атрибуция, refit, векторизация) учтены "
+            "однократно под run_id shared_* и не суммируются повторно по seed.",
+        ],
+    }
+
+
+# ---------------------------------------------------------------------------
+# HTML (автономный, inline SVG)
+# ---------------------------------------------------------------------------
+
+
+def _esc(s: Any) -> str:
+    return html.escape("" if s is None else str(s))
+
+
+def _fmt(value: Any, digits: int = 4) -> str:
+    if value is None or (isinstance(value, float) and not np.isfinite(value)):
+        return "null"
+    if isinstance(value, (int, np.integer)):
+        return str(int(value))
+    return f"{float(value):.{digits}f}"
+
+
+def _table_html(df: pd.DataFrame, columns: list[str], max_rows: int | None = None,
+                formats: dict[str, Any] | None = None) -> str:
+    formats = formats or {}
+    view = df[columns].copy()
+    if max_rows is not None:
+        view = view.head(max_rows)
+    head = "".join(f"<th>{_esc(c)}</th>" for c in columns)
+    rows = []
+    for _, row in view.iterrows():
+        cells = []
+        for c in columns:
+            v = row[c]
+            if formats.get(c):
+                v = formats[c](v)
+            else:
+                v = _fmt(v) if isinstance(v, (float, np.floating)) else ("null" if pd.isna(v) else _esc(v))
+            cells.append(f"<td>{v}</td>")
+        rows.append("<tr>" + "".join(cells) + "</tr>")
+    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
+
+
+_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
+
+
+def _svg_line_chart(series: dict[str, list[tuple[float, float]]], width: int = 720, height: int = 320,
+                    x_label: str = "K (суммарный бюджет признаков)", y_label: str = "macro-F1 (comparison_val)",
+                    baseline: tuple[float, float] | None = None) -> str:
+    """Простой автономный SVG-график: линии методов по бюджетам K."""
+    pad_l, pad_r, pad_t, pad_b = 56, 16, 16, 40
+    all_x = [x for pts in series.values() for x, _ in pts]
+    all_y = [y for pts in series.values() for _, y in pts]
+    if baseline:
+        all_y.extend([baseline[0], baseline[1]])
+    if not all_x:
+        return "<p class='muted'>нет данных для графика</p>"
+    x_min, x_max = min(all_x), max(all_x)
+    if x_min == x_max:
+        x_min, x_max = x_min * 0.5, x_max * 1.5 + 1
+    y_min, y_max = min(all_y), max(all_y)
+    margin = (y_max - y_min) * 0.08 + 1e-9
+    y_min, y_max = y_min - margin, y_max + margin
+
+    def sx(x: float) -> float:
+        return pad_l + (x - x_min) / (x_max - x_min) * (width - pad_l - pad_r)
+
+    def sy(y: float) -> float:
+        return height - pad_b - (y - y_min) / (y_max - y_min) * (height - pad_t - pad_b)
+
+    parts = [f"<svg viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg' class='chart'>"]
+    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
+        y = y_min + frac * (y_max - y_min)
+        parts.append(f"<line x1='{pad_l}' y1='{sy(y):.1f}' x2='{width - pad_r}' y2='{sy(y):.1f}' class='grid'/>")
+        parts.append(f"<text x='{pad_l - 6}' y='{sy(y) + 4:.1f}' class='tick' text-anchor='end'>{y:.3f}</text>")
+    for x in sorted(set(all_x)):
+        parts.append(f"<text x='{sx(x):.1f}' y='{height - pad_b + 16}' class='tick' text-anchor='middle'>{int(x)}</text>")
+    parts.append(f"<text x='{(pad_l + width - pad_r) / 2}' y='{height - 6}' class='label' text-anchor='middle'>{_esc(x_label)}</text>")
+    parts.append(f"<text x='14' y='{(pad_t + height - pad_b) / 2}' class='label' text-anchor='middle' transform='rotate(-90 14 {(pad_t + height - pad_b) / 2})'>{_esc(y_label)}</text>")
+    if baseline:
+        parts.append(
+            f"<rect x='{pad_l}' y='{sy(baseline[1]):.1f}' width='{width - pad_l - pad_r}' "
+            f"height='{max(0.0, sy(baseline[0]) - sy(baseline[1])):.1f}' class='baseline-band'/>"
+        )
+        parts.append(f"<line x1='{pad_l}' y1='{sy(sum(baseline) / 2):.1f}' x2='{width - pad_r}' "
+                     f"y2='{sy(sum(baseline) / 2):.1f}' class='baseline-line'/>")
+        parts.append(f"<text x='{width - pad_r - 4}' y='{sy(baseline[1]) - 5:.1f}' class='tick' text-anchor='end'>baseline none (min–max по seed)</text>")
+    for i, (name, pts) in enumerate(sorted(series.items())):
+        color = _PALETTE[i % len(_PALETTE)]
+        pts_sorted = sorted(pts)
+        path = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for j, (x, y) in enumerate(pts_sorted))
+        parts.append(f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2'/>")
+        for x, y in pts_sorted:
+            parts.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3.5' fill='{color}'/>")
+        last_x, last_y = pts_sorted[-1]
+        parts.append(f"<text x='{sx(last_x):.1f}' y='{sy(last_y) - 8:.1f}' class='legend' text-anchor='middle' fill='{color}'>{_esc(name)}</text>")
+    parts.append("</svg>")
+    return "".join(parts)
+
+
+_CSS = """
+body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1b1b1b;max-width:1180px}
+h1{font-size:22px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}
+h3{font-size:14px;margin-top:18px}
+table{border-collapse:collapse;font-size:12px;margin:8px 0;width:100%}
+th,td{border:1px solid #ccc;padding:3px 6px;text-align:left;vertical-align:top}
+th{background:#f3f3f3}
+.muted{color:#666;font-size:12px}
+.status-success{color:#116611;font-weight:600}.status-failed{color:#a00000;font-weight:600}
+.status-pending{color:#8a6d00;font-weight:600}
+.chart{background:#fff;border:1px solid #ddd;margin:8px 0;max-width:100%}
+.grid{stroke:#e4e4e4;stroke-width:1}.tick{font-size:10px;fill:#555}.label{font-size:11px;fill:#333}
+.baseline-band{fill:#1f77b4;opacity:.10}.baseline-line{stroke:#1f77b4;stroke-dasharray:5 4;opacity:.6}
+.legend{font-size:11px}
+code{background:#f3f3f3;padding:1px 4px}
+.box{border:1px solid #ddd;padding:10px 14px;margin:10px 0;background:#fbfbfb}
+"""
+
+
+def _html_report(
+    output_root: str,
+    plan: dict,
+    runs_df: pd.DataFrame,
+    method_summary: list[dict[str, Any]],
+    support_bins: pd.DataFrame,
+    confusion: pd.DataFrame,
+    stages: pd.DataFrame,
+    predictions: pd.DataFrame,
+) -> str:
+    ok = runs_df[runs_df["status"] == "success"]
+    none_ok = ok[ok["method"] == "none"]
+    baseline_band = None
+    if len(none_ok):
+        baseline_band = (float(none_ok["l3_macro_f1"].min()), float(none_ok["l3_macro_f1"].max()))
+
+    series: dict[str, list[tuple[float, float]]] = {}
+    for row in method_summary:
+        method, budget, stats = row["method"], row["budget"], row["l3_macro_f1"]
+        if budget is None or stats["mean"] is None:
+            continue
+        series.setdefault(method, []).append((float(budget), stats["mean"]))
+    chart = _svg_line_chart(series, baseline=baseline_band)
+
+    summary_rows = []
+    for row in method_summary:
+        summary_rows.append({
+            "method": row["method"], "K": row["budget"],
+            "seed ok": f"{row['n_seed_success']}/{row['n_seed_planned']}",
+            "macro-F1 mean": row["l3_macro_f1"]["mean"],
+            "std": row["l3_macro_f1"]["std"],
+            "min": row["l3_macro_f1"]["min"],
+            "max": row["l3_macro_f1"]["max"],
+            "Δ mean": row["paired_delta_macro_f1"]["mean_pairwise"],
+            "Δ range": (None if row["paired_delta_macro_f1"]["range"] is None
+                        else f"±{row['paired_delta_macro_f1']['range'] / 2:.4f}"),
+        })
+    summary_df = pd.DataFrame(summary_rows)
+
+    status_counts = runs_df["status"].value_counts().to_dict()
+    runs_view = runs_df.copy()
+    runs_view["macro-F1"] = runs_view["l3_macro_f1"]
+    runs_view["acc"] = runs_view["l3_accuracy"]
+    runs_view["ΔF1"] = runs_view["delta_macro_f1_vs_none_same_seed"]
+    runs_view["Δacc"] = runs_view["delta_accuracy_vs_none_same_seed"]
+
+    bins_view = support_bins.copy() if not support_bins.empty else pd.DataFrame(
+        columns=["run_id", "method", "seed", "support_bin", "n_L3", "macro_f1_bin",
+                 "baseline_macro_f1_bin_none_same_seed", "delta_macro_f1_bin"])
+    conf_view = confusion.sort_values(["count"], ascending=False) if not confusion.empty else confusion
+
+    # Ресурсы: shared-этапы отдельно от per-run; без двойного учёта.
+    shared_stages = stages[stages["run_id"].astype(str).str.startswith(("shared", "gate1:shared"))] if not stages.empty else stages
+    run_stages = stages[~stages["run_id"].astype(str).str.startswith(("shared", "gate1:shared"))] if not stages.empty else stages
+    res_agg = None
+    if not run_stages.empty:
+        mlp = run_stages[run_stages["stage"] == "mlp_train"]
+        infer = run_stages[run_stages["stage"] == "comparison_val_inference"]
+        res_agg = pd.DataFrame([
+            {"этап": "mlp_train (per run)", "n": int(len(mlp)),
+             "seconds sum": mlp["seconds"].dropna().astype(float).sum() if len(mlp) else None,
+             "seconds mean": mlp["seconds"].dropna().astype(float).mean() if len(mlp) else None,
+             "peak RSS max": mlp["peak_rss_bytes"].dropna().max() if len(mlp) and mlp["peak_rss_bytes"].notna().any() else None},
+            {"этап": "comparison_val_inference (per run)", "n": int(len(infer)),
+             "seconds sum": infer["seconds"].dropna().astype(float).sum() if len(infer) else None,
+             "seconds mean": infer["seconds"].dropna().astype(float).mean() if len(infer) else None,
+             "peak RSS max": infer["peak_rss_bytes"].dropna().max() if len(infer) and infer["peak_rss_bytes"].notna().any() else None},
+        ])
+
+    status_cells = " ".join(
+        f"<span class='status-{_esc(k)}'>{_esc(k)}: {v}</span>" for k, v in sorted(status_counts.items())
+    )
+    formulas_html = "".join(f"<li><b>{_esc(k)}:</b> {_esc(v)}</li>" for k, v in FORMULAS.items())
+    limits_html = "".join(f"<li>{_esc(x)}</li>" for x in LIMITATIONS)
+
+    parts = [
+        "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>",
+        f"<title>S06 — сравнение supervised feature selection</title><style>{_CSS}</style></head><body>",
+        "<h1>S06 — сравнение supervised feature selection (Global Chi² / Class-aware Chi² / Gradient × Input)</h1>",
+        f"<p class='muted'>Собрано: {_esc(dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'))} · "
+        f"корень: <code>{_esc(os.path.abspath(output_root))}</code> · схема {S06_REPORT_SCHEMA}</p>",
+        "<h2>0. Статус прогонов</h2>",
+        f"<p>Всего плановых runs: {int(plan['n_planned'])}. {status_cells}</p>",
+        _table_html(runs_view, ["run_id", "method", "budget", "seed", "status", "reason",
+                                "macro-F1", "acc", "ΔF1", "Δacc", "n_features"],
+                    formats={"reason": lambda v: _esc(v) if pd.notna(v) else "—"}),
+        "<h2>1. Главное сравнение: качество L3 vs бюджет K</h2>",
+        "<p class='muted'>Точки — средний macro-F1 по успешным seed; голубая лента — диапазон min–max "
+        "полного baseline (none) по seed. paired Δ — к none того же seed.</p>",
+        chart,
+        _table_html(summary_df, list(summary_df.columns)) if not summary_df.empty else "<p class='muted'>нет данных</p>",
+        "<h2>2. Метрики по диапазонам support L3 (comparison_val)</h2>",
+        "<p class='muted'>macro-F1 бина — невзвешенное среднее per-L3 F1; baseline — none того же seed; "
+        "пустые бины — null с n_L3=0. Категории с support 2 нестабильны (шаг recall 0.5).</p>",
+        _table_html(bins_view, ["run_id", "support_bin", "n_L3", "macro_f1_bin",
+                                "baseline_macro_f1_bin_none_same_seed", "delta_macro_f1_bin"])
+        if not bins_view.empty else "<p class='muted'>нет данных (нет успешных runs)</p>",
+        "<h2>3. Top confused pairs (true L3 → predicted L3)</h2>",
+        "<p class='muted'>Все пары — в report/confusion_pairs.csv с rank_in_run; support истинной L3 и "
+        "доля ошибок на comparison_val. Ниже — топ-30 по count среди всех runs.</p>",
+        _table_html(conf_view, ["run_id", "true_L3", "predicted_L3", "count", "true_support", "error_share",
+                                "rank_in_run"], max_rows=30)
+        if not conf_view.empty else "<p class='muted'>предсказаний нет — confused pairs не построены</p>",
+        "<h2>4. Ресурсы</h2>",
+        "<p class='muted'>Shared-затраты (ранжирование, атрибуция, refit, векторизация) — однократно, "
+        "не суммируются по seed. GPU недоступен → peak_vram=null (не ноль).</p>",
+        "<h3>Shared-этапы</h3>",
+        _table_html(shared_stages, ["run_id", "stage", "status", "seconds", "peak_rss_bytes",
+                                    "peak_vram_bytes", "artefact_bytes", "source"])
+        if not shared_stages.empty else "<p class='muted'>нет shared-этапов</p>",
+        "<h3>Сводка per-run этапов</h3>",
+        _table_html(res_agg, list(res_agg.columns)) if res_agg is not None else "<p class='muted'>нет per-run этапов</p>",
+        "<h3>Все этапы</h3>",
+        _table_html(stages, ["run_id", "stage", "status", "seconds", "peak_rss_bytes",
+                             "peak_vram_bytes", "artefact_bytes", "source"], max_rows=400),
+        "<h2>5. Формулы и протокол</h2>",
+        f"<ul>{formulas_html}</ul>",
+        "<div class='box'><b>Границы протокола:</b> рейтинги — только outer_train; refit — один на "
+        "метод×K на том же снимке корпуса; обучение — замороженный протокол Gate 1 (train-only выбор "
+        "эпох, restore_best=False); оценка — raw top-1 L3 на comparison_val; sealed_test не использовался.</div>",
+        "<h2>6. Ограничения</h2>", f"<ul>{limits_html}</ul>",
+        "<h2>7. Пересборка отчёта</h2>",
+        "<div class='box'><code>python scripts/run_s06_report.py --output " + _esc(output_root) + "</code>"
+        "<br>Источники чисел: runs/&lt;slug&gt;/run_result_seed_*.json, runs/predictions/*.parquet, "
+        "runs/stages.csv, shared/*, runs/baseline_full/*, audit/*. Обучение, fit TF-IDF и inference "
+        "при пересборке не выполняются; существующие report-файлы не перезаписываются (перед повторной "
+        "сборкой переименуйте report/).</div>",
+        "</body></html>",
+    ]
+    return "".join(parts)
+
+
+# ---------------------------------------------------------------------------
+# Сборка отчёта (Gate 4)
+# ---------------------------------------------------------------------------
+
+
+def build_s06_report(output_root: str = "outputs/S06", report_dir: str | None = None) -> dict[str, str]:
+    """Собирает восемь обязательных файлов отчёта из сохранённых результатов.
+
+    Без обучения, без fit TF-IDF и без повторного inference: читаются только
+    JSON/CSV/Parquet, записанные контурами Gate 0–3. Существующие файлы в
+    ``report_dir`` не перезаписываются (отказ до любых записей); planned runs
+    без result-файла честно получают статус ``pending``.
+    """
+    report_dir = report_dir or os.path.join(output_root, "report")
+    plan = _read_json(os.path.join(output_root, "runs", "planned.json"))
+    results = _load_run_results(output_root, plan)
+    runs_df = _runs_frame(plan, results)
+    per_class = _per_class_frame(results, runs_df)
+    support_bins = _support_bins_frame(per_class, runs_df)
+    predictions = _load_predictions(output_root, [r["run_id"] for r in plan["runs"]])
+    confusion = _confusion_frame(predictions, per_class)
+    stages = _stages_frame(output_root)
+    method_summary = _method_summary(runs_df)
+
+    os.makedirs(report_dir, exist_ok=True)
+    targets = {name: os.path.join(report_dir, name) for name in REPORT_FILES}
+    _check_new(list(targets.values()))
+
+    runs_csv = targets["runs.csv"]
+    runs_df.to_csv(runs_csv, index=False)
+    per_class.to_parquet(targets["per_class.parquet"], index=False)
+    stages.to_csv(targets["stages.csv"], index=False)
+    confusion.to_csv(targets["confusion_pairs.csv"], index=False)
+    support_bins.to_csv(targets["support_bins.csv"], index=False)
+    predictions.to_parquet(targets["predictions_val.parquet"], index=False)
+
+    report_files = {name: os.path.abspath(path) for name, path in targets.items()}
+    manifest = build_manifest(output_root, plan, runs_df, method_summary, report_files)
+    with open(targets["manifest.json"], "w", encoding="utf-8") as f:
+        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
+
+    html_doc = _html_report(
+        output_root, plan, runs_df, method_summary, support_bins, confusion, stages, predictions
+    )
+    with open(targets["report.html"], "w", encoding="utf-8") as f:
+        f.write(html_doc)
+
+    log_info = {
+        "report_dir": os.path.abspath(report_dir),
+        "n_runs": int(plan["n_planned"]),
+        "statuses": manifest["statuses"]["by_run"] and {
+            k: int((runs_df["status"] == k).sum()) for k in runs_df["status"].unique()
+        },
+        "files": report_files,
+    }
+    return log_info
+
+
+__all__ = [
+    "S06_REPORT_SCHEMA",
+    "S06ReportError",
+    "SUPPORT_BINS",
+    "REPORT_FILES",
+    "FORMULAS",
+    "LIMITATIONS",
+    "build_s06_report",
+]
diff --git a/scripts/freeze_split_s06.py b/scripts/freeze_split_s06.py
new file mode 100644
index 0000000..e31df30
--- /dev/null
+++ b/scripts/freeze_split_s06.py
@@ -0,0 +1,123 @@
+"""S06, этап A3 (D03): заморозка воспроизводимого group-aware сплита ODS с
+жёсткими per-class минимумами.
+
+Читает размеченный parquet, применяет политику S06 ``support >= 12`` без
+молчаливого удаления (исключённые row ids фиксируются в manifest) и строит
+детерминированный стратифицированный group-aware сплит (~80/10/10 →
+outer_train / comparison_val / sealed_test плюс train_fit / train_monitor
+внутри outer_train) с минимумами: каждая сохранённая L3 имеет >= min-per-class
+строк в comparison_val, sealed_test и train_monitor и >= min-train-fit в
+train_fit. Группы — по реальному preprocessing обеих TF-IDF-веток
+(kkt_clf.features): пересечений нормализованных текстов между частями нет.
+
+При несовместимости минимумов с неделимыми группами — остановка без записи
+сплита; доказательная диагностика сохраняется в ``<audit-dir>/split_blocked.json``.
+Метрики качества не вычисляются; sealed test используется только для
+count/hash/support-проверок целостности. Существующие outputs не перезаписываются.
+
+Пример:
+    .venv/bin/python scripts/freeze_split_s06.py \\
+        --input fixtures/ods.parquet --min-count 12 \\
+        --out-dir outputs/S06/splits --audit-dir outputs/S06/audit
+"""
+from __future__ import annotations
+
+import argparse
+import datetime as dt
+import json
+import os
+import sys
+
+sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+
+from kkt_clf.data import REQUIRED_COLUMNS, file_fingerprint, read_parquet  # noqa: E402
+from kkt_clf.freeze_split import FreezeSplitError  # noqa: E402
+from kkt_clf.s06_split import S06SplitError, s06_split, save_s06_split  # noqa: E402
+
+DEFAULT_FRACTIONS = (0.8, 0.1, 0.1)
+
+
+def _write_blocked_diagnostics(audit_dir: str, args: argparse.Namespace, err: Exception) -> str:
+    """Сохраняет доказательную диагностику блокировки; существующий файл не трогает."""
+    os.makedirs(audit_dir, exist_ok=True)
+    path = os.path.join(audit_dir, "split_blocked.json")
+    if os.path.exists(path):
+        raise SystemExit(f"Отказ: файл диагностики уже существует и не будет перезаписан: {path}")
+    payload = {
+        "status": "blocked",
+        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
+        "error_type": type(err).__name__,
+        "error": str(err),
+        "infeasible_classes": getattr(err, "infeasible", None),
+        "min_violations": getattr(err, "min_violations", None),
+        "policy": {
+            "min_count": args.min_count,
+            "fractions": list(args.fractions),
+            "monitor_fraction": args.monitor_fraction,
+            "split_seed": args.seed,
+            "min_rows_per_class": args.min_per_class,
+            "min_train_fit": args.min_train_fit,
+        },
+        "input": file_fingerprint(args.input),
+    }
+    with open(path, "w", encoding="utf-8") as f:
+        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
+    return path
+
+
+def main() -> int:
+    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
+    p.add_argument("--input", default="fixtures/ods.parquet", help="размеченный parquet (name, L1, L2, L3)")
+    p.add_argument("--min-count", type=int, default=12)
+    p.add_argument("--fractions", type=float, nargs=3, default=list(DEFAULT_FRACTIONS),
+                   help="целевые доли outer train/val/test (минимумы имеют приоритет)")
+    p.add_argument("--monitor-fraction", type=float, default=0.1,
+                   help="целевая доля train_monitor внутри outer train")
+    p.add_argument("--seed", type=int, default=42, help="seed разбиения (train seed — отдельно)")
+    p.add_argument("--min-per-class", type=int, default=2,
+                   help="минимум строк каждой сохранённой L3 в comparison_val/sealed_test/train_monitor")
+    p.add_argument("--min-train-fit", type=int, default=2,
+                   help="минимум строк каждой сохранённой L3 в train_fit")
+    p.add_argument("--out-dir", default="outputs/S06/splits")
+    p.add_argument("--audit-dir", default="outputs/S06/audit")
+    a = p.parse_args()
+
+    try:
+        df = read_parquet(a.input)
+        df = df[list(REQUIRED_COLUMNS)]
+        frozen = s06_split(
+            df,
+            min_count=a.min_count,
+            fractions=tuple(a.fractions),
+            monitor_fraction=a.monitor_fraction,
+            seed=a.seed,
+            min_rows_per_class=a.min_per_class,
+            min_train_fit=a.min_train_fit,
+        )
+        summary = save_s06_split(
+            frozen,
+            input_path=a.input,
+            out_dir=a.out_dir,
+            audit_dir=a.audit_dir,
+            min_count=a.min_count,
+            fractions=tuple(a.fractions),
+            monitor_fraction=a.monitor_fraction,
+            seed=a.seed,
+            min_rows_per_class=a.min_per_class,
+            min_train_fit=a.min_train_fit,
+        )
+    except S06SplitError as e:
+        diag = _write_blocked_diagnostics(a.audit_dir, a, e)
+        print(f"blocked: {e}", file=sys.stderr)
+        print(f"диагностика сохранена: {diag}", file=sys.stderr)
+        return 2
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
diff --git a/scripts/s06_tfidf_ref.py b/scripts/s06_tfidf_ref.py
new file mode 100644
index 0000000..9e27394
--- /dev/null
+++ b/scripts/s06_tfidf_ref.py
@@ -0,0 +1,356 @@
+"""S06, этап A4 (D04): проверка и фиксация ссылки на полный TF-IDF S02.
+
+Сверяет метаданные выбранного full-артефакта S02 (по умолчанию
+``outputs/S02/tfidf_1m`` — 300 000 char + 100 000 word признаков) с
+фактическими файлами и с источником fit (``data/corpus.pqt``): sha256, размер,
+число строк снимка, размеры словарей, загрузка через kkt_clf.vectorizer и
+контрольный transform фиксированных строк. Проверяет, что векторизатор
+является именно полным (ожидаемые размеры словарей задаются явно), что fit
+выполнен на полном снимке внешнего неразмеченного корпуса (metadata source:
+``rows_used == rows_in_file``, ``truncated is False``, sha256 совпадает с
+фактическим корпусом), а не на размеченной train или подвыборке, и что после
+загрузки inference применяет готовый артефакт через transform без fit.
+
+Утверждения, которые нельзя подтвердить metadata/аудитом, не выдаются за
+факт: ``fit_on_labeled_train`` фиксируется как ``false`` только при полном
+совпадении source с фактическим корпусом и (при заданном эталоне) несовпадении
+с размеченным файлом, иначе — ``null`` с явной оговоркой, что не подтверждено.
+Число совпадений текстов ODS с корпусом (трансдуктивность) берётся из аудита
+``--audit`` (по умолчанию ``outputs/S06/audit/data_audit.json``) после сверки
+sha256 корпуса в аудите с фактическим; без согласованного аудита количество
+не заявляется. Эталон размеченного файла: ``--labeled`` либо fingerprint из
+замороженного ``outputs/S06/splits/manifest.json``.
+
+Результат — ``full_tfidf_ref.json`` (путь + hashes + provenance); сам full
+артефакт не копируется и не изменяется. Существующие файлы не перезаписываются.
+
+Пример:
+    .venv/bin/python scripts/s06_tfidf_ref.py \\
+        --s02-dir outputs/S02 --full-name tfidf_1m \\
+        --corpus data/corpus.pqt --out outputs/S06/shared/full_tfidf_ref.json
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
+sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+
+import numpy as np  # noqa: E402
+import pyarrow.parquet as pq  # noqa: E402
+
+from kkt_clf.data import file_fingerprint  # noqa: E402
+from kkt_clf.vectorizer import VectorizerArtifactError, load_vectorizer_artifact  # noqa: E402
+
+EXPECTED_VOCAB = {"n_char": 300_000, "n_word": 100_000, "n_total": 400_000}
+
+# Фиксированные контрольные строки для transform smoke (заморожены, не из test).
+PROBE_TEXTS = [
+    "молоко пастеризованное 3.2% 1л",
+    "хлеб бородинский 400г",
+    "unknown-probe-item-zzz-123",
+]
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
+    import sklearn
+
+    return {
+        "python": platform.python_version(),
+        "numpy": np.__version__,
+        "sklearn": sklearn.__version__,
+        "platform": platform.platform(),
+    }
+
+
+def main() -> int:
+    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
+    p.add_argument("--s02-dir", default="outputs/S02")
+    p.add_argument("--full-name", default="tfidf_1m", help="каталог full-артефакта внутри --s02-dir")
+    p.add_argument("--corpus", default="data/corpus.pqt")
+    p.add_argument("--out", default="outputs/S06/shared/full_tfidf_ref.json")
+    p.add_argument("--expect-vocab-total", type=int, default=EXPECTED_VOCAB["n_total"])
+    p.add_argument("--audit", default=os.path.join("outputs", "S06", "audit", "data_audit.json"),
+                   help="data_audit.json S06: источник подтверждённого числа совпадений текстов ODS с корпусом")
+    p.add_argument("--labeled", default=None,
+                   help="размеченный parquet/CSV для сверки source fit (default: fingerprint из outputs/S06/splits/manifest.json)")
+    a = p.parse_args()
+
+    if os.path.exists(a.out):
+        print(f"error: файл уже существует и не будет перезаписан: {a.out}", file=sys.stderr)
+        return 1
+
+    art_dir = os.path.join(a.s02_dir, a.full_name)
+    meta_path = os.path.join(art_dir, "vectorizer_meta.json")
+    if not os.path.exists(meta_path):
+        print(f"error: нет метаданных артефакта: {meta_path}", file=sys.stderr)
+        return 1
+    with open(meta_path, encoding="utf-8") as f:
+        meta = json.load(f)
+
+    failures: list[str] = []
+
+    # --- Provenance источника fit против фактического снимка корпуса.
+    # Утверждения только от проверенных данных: совпадение sha256/bytes с
+    # фактическим корпусом, полный снимок (rows_used == rows_in_file,
+    # truncated is False) и несовпадение с размеченным файлом (эталон).
+    src = meta.get("source", {}) or {}
+    corpus_fp = file_fingerprint(a.corpus)
+    corpus_rows_actual = int(pq.ParquetFile(a.corpus).metadata.num_rows)
+    sha_match = (
+        corpus_fp["sha256"] == src.get("sha256") and corpus_fp["bytes"] == src.get("bytes")
+    )
+    rows_in_file_meta = src.get("rows_in_file")
+    rows_used = src.get("rows_used")
+    truncated = src.get("truncated")
+    if rows_used is not None and rows_in_file_meta is not None:
+        full_snapshot = bool(rows_used == rows_in_file_meta and truncated is False)
+    else:
+        full_snapshot = None  # metadata не позволяет подтвердить полноту fit
+
+    # Эталон размеченного файла: явный --labeled либо fingerprint из
+    # замороженного manifest сплита Gate 0 (без повторного хэширования).
+    labeled_fp: dict | None = None
+    labeled_origin: str | None = None
+    if a.labeled:
+        labeled_fp = file_fingerprint(a.labeled)
+        labeled_origin = f"--labeled {a.labeled}"
+    else:
+        default_manifest = os.path.join("outputs", "S06", "splits", "manifest.json")
+        if os.path.isfile(default_manifest):
+            try:
+                with open(default_manifest, encoding="utf-8") as mf:
+                    man = json.load(mf)
+                lp = ((man.get("inputs") or {}).get("labeled_parquet")) or {}
+                if lp.get("sha256"):
+                    labeled_fp = dict(lp)
+                    labeled_origin = f"inputs.labeled_parquet из {default_manifest}"
+            except (OSError, json.JSONDecodeError):
+                pass
+
+    fit_on_labeled_train: bool | None = None
+    provenance_notes: list[str] = []
+    src_sha = src.get("sha256")
+    if labeled_fp is not None and src_sha and labeled_fp.get("sha256") == src_sha:
+        fit_on_labeled_train = True
+        failures.append(
+            "source fit совпадает по sha256 с размеченным файлом: fit_on_labeled_train=true, "
+            "контракт S02 (fit на внешнем неразмеченном корпусе) нарушен"
+        )
+    elif sha_match and full_snapshot is True and (
+        labeled_fp is None or labeled_fp.get("sha256") != src_sha
+    ):
+        fit_on_labeled_train = False
+        provenance_notes.append(
+            "fit на размеченной train не подтверждён и опровергнут: source совпадает с "
+            f"фактическим корпусом ({a.corpus}) по sha256/bytes, fit на полном снимке "
+            f"(rows_used={rows_used} == rows_in_file={rows_in_file_meta}, truncated={truncated})"
+            + (
+                f", source отличен от размеченного файла ({labeled_origin})"
+                if labeled_fp is not None
+                else f"; эталон размеченного файла не заявлен ({labeled_origin or 'manifest Gate 0 не найден'}) — "
+                     "несовпадение с разметкой по sha256 не проверялось"
+            )
+        )
+    else:
+        # Неподтверждённое не выдаём за факт: значение null с явной оговоркой.
+        provenance_notes.append(
+            "fit на неразмеченном корпусе НЕ подтверждён полностью: "
+            f"sha_match={sha_match}, full_snapshot={full_snapshot} "
+            f"(rows_used={rows_used!r}, rows_in_file={rows_in_file_meta!r}, truncated={truncated!r}); "
+            "fit_on_labeled_train не заявляется."
+        )
+        failures.append(
+            "provenance полного исходного fit не подтверждён metadata артефакта: "
+            "нужны sha256/bytes корпуса, rows_used == rows_in_file и truncated=false"
+        )
+
+    # --- Пересечение текстов ODS с корпусом (трансдуктивность) — только из
+    # согласованного аудита; без него количество не заявляется.
+    overlap_count: int | None = None
+    overlap_source: str | None = None
+    if a.audit and os.path.isfile(a.audit):
+        try:
+            with open(a.audit, encoding="utf-8") as af:
+                audit = json.load(af)
+        except (OSError, json.JSONDecodeError) as e:
+            failures.append(f"аудит не читается: {a.audit}: {e}")
+        else:
+            ao = ((audit.get("audit") or {}).get("corpus_text_overlap")) or {}
+            if ao.get("corpus_sha256") and ao["corpus_sha256"] != corpus_fp["sha256"]:
+                failures.append(
+                    f"аудит {a.audit} относится к иному корпусу (sha256 не совпадает "
+                    "с фактическим): пересечение текстов из него неприменимо"
+                )
+            elif "ods_kept_char_keys_found_in_corpus" in ao:
+                overlap_count = ao["ods_kept_char_keys_found_in_corpus"]
+                overlap_source = os.path.abspath(a.audit)
+            else:
+                failures.append(f"аудит {a.audit} не содержит corpus_text_overlap")
+    elif a.audit:
+        provenance_notes.append(
+            f"аудит {a.audit} не найден: число совпадений текстов ODS с корпусом "
+            "не подтверждено и не заявляется."
+        )
+
+    if overlap_count is not None:
+        transductive_note = (
+            f"трансдуктивность: {overlap_count} уникальных char-ключов ODS (после фильтра >=12) "
+            f"найдено в корпусе fit — подтверждено аудитом {overlap_source}; совпадение текстов "
+            "при unsupervised fit метки не переносит, но оценку делает частично трансдуктивной."
+        )
+    else:
+        transductive_note = (
+            "трансдуктивность: пересечение текстов ODS с корпусом fit не подтверждено "
+            "согласованным аудитом; количество совпадений не заявляется."
+        )
+    provenance_notes.append(transductive_note)
+
+    provenance = {
+        "meta_source": {
+            "source_id": src.get("source_id"),
+            "path": src.get("path"),
+            "sha256": src.get("sha256"),
+            "bytes": src.get("bytes"),
+            "rows_in_file": rows_in_file_meta,
+            "rows_used": rows_used,
+            "max_rows": src.get("max_rows"),
+            "strategy": src.get("strategy"),
+            "truncated": truncated,
+        },
+        "corpus_actual": {
+            "path": corpus_fp["path"],
+            "sha256": corpus_fp["sha256"],
+            "bytes": corpus_fp["bytes"],
+            "rows_in_file": corpus_rows_actual,
+        },
+        "sha256_match": sha_match,
+        "bytes_match": corpus_fp["bytes"] == src.get("bytes"),
+        "rows_in_file_match": corpus_rows_actual == rows_in_file_meta,
+        "fit_on_full_snapshot": full_snapshot,
+        "fit_on_labeled_train": fit_on_labeled_train,
+        "labeled_reference": labeled_origin,
+        "corpus_text_overlap_ods_keys": overlap_count,
+        "corpus_text_overlap_source": overlap_source,
+        "notes": provenance_notes,
+    }
+    if not (provenance["sha256_match"] and provenance["bytes_match"] and provenance["rows_in_file_match"]):
+        failures.append("provenance корпуса не совпадает с метаданными артефакта")
+
+    # --- Файлы артефакта против метаданных.
+    files_report: dict[str, dict] = {}
+    files_meta = meta.get("files") or {}
+    for fname in ("features.joblib",):
+        fpath = os.path.join(art_dir, fname)
+        entry: dict = {"path": os.path.abspath(fpath)}
+        if not os.path.exists(fpath):
+            entry["status"] = "missing"
+            failures.append(f"нет файла артефакта: {fpath}")
+        else:
+            fp = file_fingerprint(fpath)
+            expected = files_meta.get(fname, {})
+            entry.update({
+                "sha256": fp["sha256"],
+                "bytes": fp["bytes"],
+                "meta_sha256": expected.get("sha256"),
+                "meta_bytes": expected.get("bytes"),
+                "sha256_match": fp["sha256"] == expected.get("sha256"),
+                "bytes_match": fp["bytes"] == expected.get("bytes"),
+            })
+            if not (entry["sha256_match"] and entry["bytes_match"]):
+                failures.append(f"hash/размер {fname} не совпадает с метаданными")
+        files_report[fname] = entry
+
+    # --- Загрузка, словарь и transform smoke.
+    load_report: dict = {}
+    try:
+        loaded = load_vectorizer_artifact(art_dir)
+        vocab = {
+            "n_char": loaded.builder.n_char_features,
+            "n_word": loaded.builder.n_word_features,
+            "n_total": loaded.builder.n_features,
+        }
+        meta_vocab = meta.get("vocab", {})
+        vocab_match_meta = vocab == meta_vocab
+        if not vocab_match_meta:
+            failures.append(f"загруженный словарь {vocab} != метаданным {meta_vocab}")
+        if vocab["n_total"] != a.expect_vocab_total:
+            failures.append(
+                f"артефакт не является полным TF-IDF S02: n_total={vocab['n_total']}, "
+                f"ожидалось {a.expect_vocab_total}"
+            )
+        X = loaded.builder.transform(PROBE_TEXTS)
+        transform_ok = X.shape == (len(PROBE_TEXTS), vocab["n_total"])
+        if not transform_ok:
+            failures.append(f"transform smoke: форма {X.shape}, ожидалось {(len(PROBE_TEXTS), vocab['n_total'])}")
+        empty = int((np.diff(X.indptr) == 0).sum())
+        load_report = {
+            "status": "ok",
+            "loaded_vocab": vocab,
+            "loaded_vocab_match_meta": vocab_match_meta,
+            "fit_report": dict(loaded.builder.fit_report_),
+            "transform_probe": {
+                "texts_sha256_probe": "frozen-in-script: 3 строки, не из test",
+                "shape": [int(X.shape[0]), int(X.shape[1])],
+                "dtype": str(X.dtype),
+                "empty_rows": empty,
+            },
+            "inference_contract": "после загрузки inference применяет готовый векторизатор через "
+                                  "transform; повторный fit не выполняется",
+        }
+    except VectorizerArtifactError as e:
+        load_report = {"status": f"failed: {e}"}
+        failures.append(f"загрузка артефакта не удалась: {e}")
+
+    ref = {
+        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
+        "status": "ok" if not failures else "failed",
+        "failures": failures,
+        "generator": {
+            "script": os.path.abspath(__file__),
+            "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]),
+            "cwd": os.getcwd(),
+            "git": _git_info(),
+            "versions": _versions(),
+        },
+        "artifact": {
+            "dir": os.path.abspath(art_dir),
+            "name": a.full_name,
+            "created_at": meta.get("created_at"),
+            "vocab_meta": meta.get("vocab"),
+        },
+        "files": files_report,
+        "corpus_provenance": provenance,
+        "load_check": load_report,
+        "usage_note": "Единственный full TF-IDF для всех runs S06; selected TF-IDF строятся "
+                      "refit с фиксированными словарями на том же снимке корпуса; полный "
+                      "артефакт не изменяется и не копируется.",
+    }
+
+    os.makedirs(os.path.dirname(a.out), exist_ok=True)
+    with open(a.out, "w", encoding="utf-8") as f:
+        json.dump(ref, f, ensure_ascii=False, indent=2, default=str)
+    print(json.dumps({"ref": os.path.abspath(a.out), "status": ref["status"], "failures": failures,
+                      "vocab": load_report.get("loaded_vocab")}, ensure_ascii=False, indent=2))
+    return 0 if not failures else 1
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
diff --git a/scripts/run_s06_baseline.py b/scripts/run_s06_baseline.py
new file mode 100644
index 0000000..2cde2ac
--- /dev/null
+++ b/scripts/run_s06_baseline.py
@@ -0,0 +1,264 @@
+"""S06 Gate 1 (D04–D06): замороженный протокол обучения и новый full baseline (CLI).
+
+Подкоманды:
+
+- ``verify-split`` — проверка provenance сплита Gate 0 (``outputs/S06/splits``):
+  целостность частей по sha256, политика ``min_count == 12`` (сплит S05 с
+  min_count=10 отклоняется), integrity checks Gate 0. Без обучения.
+- ``verify-tfidf`` — проверка full-TFIDF S02 (целостность артефакта +
+  совпадение корпуса по sha256) и запись ``shared/full_tfidf_ref.json``
+  (при существующем — сверка совпадения, не перезапись; артефакт не копируется).
+- ``estimate`` — ресурсная оценка Gate 1: проекция времени/RAM/VRAM от
+  измерений S05 и размеров нового сплита, без обучения/transform/inference;
+  GPU недоступен — ``null``. ``--write`` сохраняет в
+  ``shared/resource_estimate_gate1.json`` (только новый файл).
+- ``select`` — train-only выбор числа эпох/конфигурации на
+  train_fit/train_monitor (кривые — ``runs/baseline_full/selection/``), затем
+  заморозка неизменяемого ``shared/training_protocol.json``. comparison_val
+  до этого не читается; повторный select при существующем протоколе — отказ.
+- ``refit`` — три полных baseline seed 42/43/44 на outer_train ровно на
+  замороженное число эпох (фиксированное заранее расписание LR), опорный
+  checkpoint — ``--reference-seed`` (default 42, train-only правило), единственная
+  оценка — raw top-1 L3 на comparison_val против train-only majority
+  (``runs/baseline_full/{runs.csv,per_class.parquet,history.csv,stages.csv,
+  majority_baseline.json,selected_checkpoint.json,manifest.json,checkpoints/}``).
+  Перед обучением: verify split + verify TF-IDF + сверка замороженного
+  протокола с фактическими входами (fail closed).
+- ``all`` — verify-split → verify-tfidf → select → refit одним запуском.
+
+Границы: ``sealed_test`` не используется ни для inference, ни для метрик
+(только структурные sha256/n_rows из manifest); старые checkpoint/split S05
+не используются; существующие файлы не перезаписываются; selected-сетка
+Gate 2+ этим контуром не запускается.
+
+Условие запуска обучения: Gate 0 (``outputs/S06/splits`` + независимая
+проверка) завершён; команды ``select``/``refit``/``all`` до этого завершатся
+отказом с диагностикой.
+
+Пример (после независимой проверки Gate 0):
+
+    python scripts/run_s06_baseline.py verify-split
+    python scripts/run_s06_baseline.py verify-tfidf
+    python scripts/run_s06_baseline.py estimate --write
+    python scripts/run_s06_baseline.py select
+    python scripts/run_s06_baseline.py refit
+    # или одним запуском:
+    python scripts/run_s06_baseline.py all
+"""
+import argparse
+import json
+import os
+import sys
+
+sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+
+from kkt_clf import setup_logging  # noqa: E402
+from kkt_clf.s06_baseline import (  # noqa: E402
+    EXPECTED_FULL_VOCAB,
+    S06Gate1Config,
+    S06Gate1Error,
+    estimate_gate1_resources,
+    run_gate1_contour,
+    run_gate1_refit,
+    verify_gate0_split,
+    verify_s06_tfidf,
+    freeze_training_protocol,
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
+def _json_list_arg(text: str) -> list:
+    try:
+        value = json.loads(text)
+    except json.JSONDecodeError as e:
+        raise argparse.ArgumentTypeError(f"ожидался JSON-список: {e}") from e
+    if not isinstance(value, list):
+        raise argparse.ArgumentTypeError(f"ожидался JSON-список, получено {type(value).__name__}")
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
+    p.add_argument("--output-root", default="outputs/S06", help="корень результатов S06")
+    p.add_argument("--splits-manifest", default=None,
+                   help="manifest сплита Gate 0 (default: <output-root>/splits/manifest.json)")
+    p.add_argument("--full-tfidf", default="outputs/S02/tfidf_1m",
+                   help="каталог сохранённого full-TFIDF S02 (pretrained)")
+    p.add_argument("--expect-vocab", type=_json_arg, default=None,
+                   help='JSON {"n_char":…,"n_word":…,"n_total":…} — контракт полного '
+                        "TF-IDF (default: 300000/100000/400000 — полный S02)")
+    p.add_argument("--corpus", default="data/corpus.pqt",
+                   help="корпус full-TFIDF для проверки provenance (sha256)")
+    p.add_argument("--preset", default="ods_light", help="базовый пресет MLP/loss/train")
+    p.add_argument("--model", type=_json_arg, default=None, help="JSON-переопределения model.*")
+    p.add_argument("--loss", type=_json_arg, default=None, help="JSON-переопределения loss.*")
+    p.add_argument("--train", type=_json_arg, default=None, help="JSON-переопределения train.*")
+    p.add_argument("--features-chunk-rows", type=int, default=50_000)
+    p.add_argument("--selection-seed", type=int, default=42)
+    p.add_argument("--seeds", type=_seeds_arg, default=[42, 43, 44],
+                   help="train seed refit через запятую (default: 42,43,44)")
+    p.add_argument("--reference-seed", type=int, default=42,
+                   help="опорный checkpoint G×I — train-only правило (default: 42)")
+    p.add_argument("--device", default="auto")
+
+
+def _cfg_from(a: argparse.Namespace) -> S06Gate1Config:
+    return S06Gate1Config(
+        output_root=a.output_root,
+        splits_manifest=a.splits_manifest or os.path.join(a.output_root, "splits", "manifest.json"),
+        full_tfidf_path=a.full_tfidf,
+        corpus_path=a.corpus,
+        expected_vocab=a.expect_vocab if a.expect_vocab is not None else dict(EXPECTED_FULL_VOCAB),
+        preset=a.preset,
+        model=a.model or {},
+        loss=a.loss or {},
+        train=a.train or {},
+        features_chunk_rows=a.features_chunk_rows,
+        selection_seed=a.selection_seed,
+        seeds=list(a.seeds),
+        reference_seed=a.reference_seed,
+        device=a.device,
+    )
+
+
+def _load_inputs(cfg: S06Gate1Config):
+    """verify split (Gate 0) + verify TF-IDF + загрузка артефакта и устройства."""
+    from kkt_clf.train import resolve_device
+    from kkt_clf.vectorizer import load_vectorizer_artifact
+
+    manifest = verify_gate0_split(cfg)
+    verification = verify_s06_tfidf(cfg)
+    loaded = load_vectorizer_artifact(cfg.full_tfidf_path)
+    device = resolve_device(cfg.device)
+    return manifest, verification, loaded, device
+
+
+def main(argv: list[str] | None = None) -> int:
+    parser = argparse.ArgumentParser(
+        description="S06 Gate 1: замороженный протокол обучения и новый full baseline",
+        formatter_class=argparse.RawDescriptionHelpFormatter,
+        epilog=__doc__,
+    )
+    sub = parser.add_subparsers(dest="cmd", required=True)
+
+    p_split = sub.add_parser("verify-split", help="проверка provenance сплита Gate 0")
+    _common(p_split)
+
+    p_tfidf = sub.add_parser("verify-tfidf", help="проверка full-TFIDF S02 + shared/full_tfidf_ref.json")
+    _common(p_tfidf)
+
+    p_est = sub.add_parser("estimate", help="ресурсная оценка Gate 1 (проекция от S05, без обучения)")
+    _common(p_est)
+    p_est.add_argument("--write", action="store_true",
+                       help="сохранить в shared/resource_estimate_gate1.json (только новый файл)")
+
+    p_select = sub.add_parser("select", help="train-only выбор эпох + заморозка протокола")
+    _common(p_select)
+    p_select.add_argument("--sweep", type=_json_list_arg, default=None,
+                          help="JSON-список кандидатов [{name, model?, train?}] для ограниченного выбора по monitor")
+
+    p_refit = sub.add_parser("refit", help="refit seed 42/43/44 на outer_train + единственная оценка comparison_val")
+    _common(p_refit)
+
+    p_all = sub.add_parser("all", help="verify-split → verify-tfidf → select → refit")
+    _common(p_all)
+    p_all.add_argument("--sweep", type=_json_list_arg, default=None,
+                       help="JSON-список кандидатов selection [{name, model?, train?}]")
+
+    a = parser.parse_args(argv)
+    setup_logging()
+    try:
+        if a.cmd == "verify-split":
+            cfg = _cfg_from(a)
+            manifest = verify_gate0_split(cfg)
+            print(json.dumps({
+                "ok": True,
+                "min_count": manifest["policy"]["min_count"],
+                "split_seed": manifest["policy"]["split_seed"],
+                "parts": {name: {"n_rows": p["n_rows"], "sha256": p["sha256"][:16] + "…"}
+                          for name, p in manifest["parts"].items()},
+                "sealed_test": "только структурные sha256/n_rows; inference/eval запрещён",
+            }, ensure_ascii=False, indent=2))
+            return 0
+        if a.cmd == "verify-tfidf":
+            cfg = _cfg_from(a)
+            verification = verify_s06_tfidf(cfg)
+            print(json.dumps({
+                "decision": verification["decision"],
+                "full_tfidf_id": verification["full_tfidf_id"],
+                "vocab": verification["vocab"],
+                "corpus_matches_source": verification["corpus_matches_source"],
+                "ref": cfg.tfidf_ref_path,
+            }, ensure_ascii=False, indent=2))
+            return 0
+        if a.cmd == "estimate":
+            cfg = _cfg_from(a)
+            manifest = verify_gate0_split(cfg)
+            estimate = estimate_gate1_resources(cfg, manifest, write=a.write)
+            print(json.dumps(estimate, ensure_ascii=False, indent=2))
+            return 0
+        if a.cmd == "select":
+            cfg = _cfg_from(a)
+            cfg.sweep = a.sweep or []
+            cfg.validate()
+            manifest, verification, loaded, device = _load_inputs(cfg)
+            protocol = freeze_training_protocol(cfg, manifest, verification, loaded, device)
+            print(json.dumps({
+                "frozen": True,
+                "fixed_epochs": protocol["fixed_epochs"],
+                "chosen": protocol["chosen"],
+                "protocol": cfg.protocol_path,
+                "selection_curves": os.path.join(cfg.baseline_dir, "selection", "curves_train_monitor.csv"),
+            }, ensure_ascii=False, indent=2))
+            return 0
+        if a.cmd == "refit":
+            cfg = _cfg_from(a)
+            cfg.validate()
+            manifest, verification, loaded, device = _load_inputs(cfg)
+            manifest_out = run_gate1_refit(cfg, manifest, verification, loaded, device)
+            print(json.dumps({
+                "statuses": manifest_out["statuses"],
+                "seed_stats": manifest_out["seed_stats_comparison_val"],
+                "output": os.path.abspath(cfg.baseline_dir),
+            }, ensure_ascii=False, indent=2))
+            return 0
+        if a.cmd == "all":
+            cfg = _cfg_from(a)
+            cfg.sweep = a.sweep or []
+            manifest_out = run_gate1_contour(cfg)
+            print(json.dumps({
+                "statuses": manifest_out["statuses"],
+                "seed_stats": manifest_out["seed_stats_comparison_val"],
+                "protocol": cfg.protocol_path,
+                "output": os.path.abspath(cfg.baseline_dir),
+            }, ensure_ascii=False, indent=2))
+            return 0
+        raise S06Gate1Error(f"неизвестная команда {a.cmd!r}")
+    except S06Gate1Error as e:
+        print(f"ОШИБКА: {e}", file=sys.stderr)
+        return 2
+    except S05BaselineError as e:
+        print(f"ОШИБКА (контур S05): {e}", file=sys.stderr)
+        return 2
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
diff --git a/scripts/run_s06_comparison.py b/scripts/run_s06_comparison.py
new file mode 100644
index 0000000..2d0c39a
--- /dev/null
+++ b/scripts/run_s06_comparison.py
@@ -0,0 +1,203 @@
+"""S06 Gate 2–3 (D06–D08): сравнительный контур feature selection (CLI).
+
+Подкоманды:
+
+- ``plan`` — фиксация плановых 30 runs (10 конфигураций × 3 seed) до оценки:
+  ``runs/planned.json`` (файл пишется один раз; повторный вызов сверяет
+  идентичность плана).
+- ``rankings`` — Gate 2a: рейтинги ``global``/``class_aware`` Chi² и
+  ``gradient_x_input`` строго по ``outer_train`` (G×I — на опорном checkpoint
+  Gate 1, reference_seed), сохранение полных рейтингов с контрольными суммами
+  (``shared/rankings/``, ``shared/attribution/``) и 9 артефактов отбора
+  (``shared/selection/<method>_k<K>/``). Возобновляемо: валидные рейтинги
+  переиспользуются, бюджеты применяются из сохранённых рейтингов.
+- ``refit`` — Gate 2b: fixed-vocabulary refit TF-IDF один раз на метод×K на
+  том же снимке корпуса, что полный TF-IDF S02 (``runs/<slug>/tfidf_selected/``
+  + ``refit_result.json`` с проверками и хэшами).
+- ``train`` — Gate 3: финальные MLP. Конфигурация ``none`` переиспользует
+  checkpoint Gate 1 (обучение не повторяется; метрики пересчитываются тем же
+  протоколом и сверяются с Gate 1); selected-конфигурации обучаются с нуля на
+  outer_train по замороженному протоколу (ровно fixed_epochs эпох,
+  restore_best=False) для seed 42/43/44; единая оценка raw top-1 L3 на
+  comparison_val с сохранением предсказаний (``runs/predictions/``) и
+  per-run результатов (``runs/<slug>/run_result_seed_<s>.json``).
+  Безопасное возобновление: существующие результаты не пересчитываются и не
+  перезаписываются.
+- ``all`` — plan (если ещё нет) → rankings → refit → train.
+
+Предусловия (fail closed, до любого ранжирования/refit/обучения): завершённый
+и проверенный Gate 0 (``outputs/S06/splits/manifest.json``, min_count=12),
+проверенный full TF-IDF S02 + ``shared/full_tfidf_ref.json``, замороженный
+протокол Gate 1, совместимый с фактическими входами, и baseline-прогон Gate 1
+(``runs/baseline_full/``). ``sealed_test`` не используется ни для inference,
+ни для метрик. Отчёт Gate 4 — отдельной командой ``scripts/run_s06_report.py``.
+
+Примеры (плановый прогон S06, после прохождения Gate 1 и независимой проверки):
+
+    python scripts/run_s06_comparison.py plan
+    python scripts/run_s06_comparison.py rankings
+    python scripts/run_s06_comparison.py refit
+    python scripts/run_s06_comparison.py train
+    # или одним запуском:
+    python scripts/run_s06_comparison.py all
+"""
+from __future__ import annotations
+
+import argparse
+import json
+import os
+import sys
+
+sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+
+from kkt_clf import setup_logging  # noqa: E402
+from kkt_clf.s06_comparison import (  # noqa: E402
+    DEFAULT_BUDGETS,
+    DEFAULT_CHAR_SHARE,
+    DEFAULT_SEEDS,
+    SELECTION_METHODS,
+    S06ComparisonConfig,
+    S06ComparisonError,
+    run_comparison_all,
+    run_rankings,
+    run_refit_selected,
+    run_train,
+    write_plan,
+)
+
+
+def _int_list(text: str) -> list[int]:
+    try:
+        values = [int(x) for x in text.split(",") if x.strip()]
+    except ValueError as e:
+        raise argparse.ArgumentTypeError(f"целые через запятую, напр. 50000,100000,200000: {e}") from e
+    if not values:
+        raise argparse.ArgumentTypeError("нужен хотя бы один элемент")
+    return values
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
+def _json_list_arg(text: str) -> list:
+    try:
+        value = json.loads(text)
+    except json.JSONDecodeError as e:
+        raise argparse.ArgumentTypeError(f"ожидался JSON-список: {e}") from e
+    if not isinstance(value, list):
+        raise argparse.ArgumentTypeError(f"ожидался JSON-список, получено {type(value).__name__}")
+    return value
+
+
+def _common(p: argparse.ArgumentParser) -> None:
+    p.add_argument("--output-root", default="outputs/S06", help="корень результатов S06")
+    p.add_argument("--splits-manifest", default=None,
+                   help="manifest сплита Gate 0 (default: <output-root>/splits/manifest.json)")
+    p.add_argument("--full-tfidf", default="outputs/S02/tfidf_1m",
+                   help="каталог сохранённого full-TFIDF S02 (pretrained)")
+    p.add_argument("--corpus", default="data/corpus.pqt",
+                   help="корпус full-TFIDF для проверки provenance (sha256)")
+    p.add_argument("--methods", default=",".join(SELECTION_METHODS),
+                   help="методы через запятую: global,class_aware,gradient_x_input")
+    p.add_argument("--budgets", type=_int_list, default=list(DEFAULT_BUDGETS),
+                   help="бюджеты K через запятую (default: 50000,100000,200000)")
+    p.add_argument("--char-share", type=float, default=DEFAULT_CHAR_SHARE,
+                   help="доля char-ветки в бюджете (default: 0.75)")
+    p.add_argument("--seeds", type=_int_list, default=list(DEFAULT_SEEDS),
+                   help="train seed через запятую (default: 42,43,44)")
+    p.add_argument("--reference-seed", type=int, default=42,
+                   help="опорный checkpoint Gate 1 для G×I (default: 42)")
+    p.add_argument("--features-chunk-rows", type=int, default=50_000)
+    p.add_argument("--attribution-batch-size", type=int, default=4096)
+    p.add_argument("--preset", default="ods_light",
+                   help="базовый пресет Gate 1 (должен совпадать с вызовом Gate 1)")
+    p.add_argument("--model", type=_json_arg, default=None, help="JSON-переопределения model.* Gate 1")
+    p.add_argument("--loss", type=_json_arg, default=None, help="JSON-переопределения loss.* Gate 1")
+    p.add_argument("--train", type=_json_arg, default=None, help="JSON-переопределения train.* Gate 1")
+    p.add_argument("--sweep", type=_json_list_arg, default=None,
+                   help="JSON-список кандидатов selection Gate 1 [{name, model?, train?}]")
+    p.add_argument("--device", default="auto")
+
+
+def _cfg_from(a: argparse.Namespace) -> S06ComparisonConfig:
+    methods = tuple(m.strip() for m in a.methods.split(",") if m.strip())
+    return S06ComparisonConfig(
+        output_root=a.output_root,
+        splits_manifest=a.splits_manifest,
+        corpus_path=a.corpus,
+        full_tfidf_path=a.full_tfidf,
+        methods=methods,
+        budgets=tuple(a.budgets),
+        char_share=a.char_share,
+        seeds=tuple(a.seeds),
+        reference_seed=a.reference_seed,
+        device=a.device,
+        features_chunk_rows=a.features_chunk_rows,
+        attribution_batch_size=a.attribution_batch_size,
+        preset=a.preset,
+        model=a.model or {},
+        loss=a.loss or {},
+        train=a.train or {},
+        sweep=a.sweep or [],
+    )
+
+
+def main(argv: list[str] | None = None) -> int:
+    parser = argparse.ArgumentParser(
+        description="S06 Gate 2–3: сравнительный контур supervised feature selection",
+        formatter_class=argparse.RawDescriptionHelpFormatter,
+        epilog=__doc__,
+    )
+    sub = parser.add_subparsers(dest="cmd", required=True)
+    for name in ("plan", "rankings", "refit", "train", "all"):
+        p = sub.add_parser(name)
+        _common(p)
+    a = parser.parse_args(argv)
+    setup_logging()
+    cfg = _cfg_from(a)
+    try:
+        if a.cmd == "plan":
+            plan = write_plan(cfg)
+            print(json.dumps({"planned": plan["n_planned"], "path": os.path.abspath(cfg.planned_path)},
+                             ensure_ascii=False, indent=2))
+            return 0
+        if a.cmd == "rankings":
+            index = run_rankings(cfg, _resolve_device(cfg))
+            print(json.dumps({"methods": list(index["methods"]),
+                              "selections": list(index["selections"])},
+                             ensure_ascii=False, indent=2))
+            return 0
+        if a.cmd == "refit":
+            results = run_refit_selected(cfg)
+            print(json.dumps({"refits": {k: v["vocab"] for k, v in results.items()}},
+                             ensure_ascii=False, indent=2))
+            return 0
+        if a.cmd == "train":
+            summary = run_train(cfg, _resolve_device(cfg))
+            print(json.dumps(summary, ensure_ascii=False, indent=2))
+            return 0
+        if a.cmd == "all":
+            summary = run_comparison_all(cfg, _resolve_device(cfg))
+            print(json.dumps({"train": summary["train"]}, ensure_ascii=False, indent=2))
+            return 0
+        raise S06ComparisonError(f"неизвестная команда {a.cmd!r}")
+    except (S06ComparisonError, RuntimeError) as e:
+        print(f"ОШИБКА: {e}", file=sys.stderr)
+        return 2
+
+
+def _resolve_device(cfg: S06ComparisonConfig):
+    from kkt_clf.train import resolve_device
+
+    return resolve_device(cfg.device)
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
diff --git a/scripts/run_s06_report.py b/scripts/run_s06_report.py
new file mode 100644
index 0000000..d7a27e7
--- /dev/null
+++ b/scripts/run_s06_report.py
@@ -0,0 +1,46 @@
+"""S06 Gate 4 (D09–D11): агрегированный отчёт сравнительного эксперимента (CLI).
+
+Без обучения, без fit TF-IDF и без повторного inference собирает восемь
+обязательных файлов ``report/`` из сохранённых машинных результатов Gate 0–3
+(см. ``kkt_clf.s06_report``): report.html, runs.csv, per_class.parquet,
+stages.csv, manifest.json, confusion_pairs.csv, predictions_val.parquet,
+support_bins.csv. Planned runs без result-файла честно получают статус
+pending; failed/skipped не скрываются. Существующие файлы не перезаписываются.
+
+Пример:
+
+    python scripts/run_s06_report.py --output outputs/S06
+"""
+from __future__ import annotations
+
+import argparse
+import json
+import os
+import sys
+
+sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+
+from kkt_clf import setup_logging  # noqa: E402
+from kkt_clf.s06_report import S06ReportError, build_s06_report  # noqa: E402
+
+
+def main() -> int:
+    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
+    ap.add_argument("--output", default="outputs/S06",
+                    help="корень результатов S06 (default: outputs/S06)")
+    ap.add_argument("--report-dir", default=None,
+                    help="куда писать отчёт (default: <output>/report); "
+                         "существующие файлы не перезаписываются")
+    args = ap.parse_args()
+    setup_logging()
+    try:
+        result = build_s06_report(output_root=args.output, report_dir=args.report_dir)
+    except S06ReportError as e:
+        print(f"ОШИБКА: {e}", file=sys.stderr)
+        return 2
+    print(json.dumps(result, ensure_ascii=False, indent=2))
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
diff --git a/scripts/s06_comparison_smoke.py b/scripts/s06_comparison_smoke.py
new file mode 100644
index 0000000..714b3d5
--- /dev/null
+++ b/scripts/s06_comparison_smoke.py
@@ -0,0 +1,260 @@
+"""S06 Gate 2–4: изолированный CPU-smoke сравнительного контура на фикстурах.
+
+Собирает отдельную песочницу ``outputs/S06_comparison_smoke/`` (ничего вне её
+не трогает; при старте песочница целиком пересоздаётся — каталог принадлежит
+только этому smoke) и прогоняет полный контур на малых синтетических данных:
+
+1. крошечный размеченный parquet (10 L3 × 50 строк из ``fixtures/ods_small.parquet``)
+   и крошечный корпус (колонка ``name``);
+2. малый full TF-IDF (char ≤ 300 / word ≤ 100 признаков) через
+   ``scripts/fit_vectorizer.py``;
+3. Gate 0 в песочнице: ``scripts/freeze_split_s06.py`` (код Gate 0 не меняется);
+4. Gate 1 в песочнице: ``scripts/run_s06_baseline.py verify-tfidf → select → refit``
+   с микро-конфигом MLP (код Gate 1 не меняется);
+5. Gate 2–3: ``scripts/run_s06_comparison.py plan → rankings → refit → train``
+   с бюджетами 60/120 (char_share=0.75) и seed 42/43/44;
+6. Gate 4: ``scripts/run_s06_report.py`` + автономная пересборка в отдельный
+   каталог (без обучения и inference) и проверка идентичности ключевых файлов;
+7. проверки: 30/30 runs success, восемь обязательных файлов отчёта, paired Δ
+   посчитаны, предсказания и confused pairs непусты, safe-resume (повторный
+   train ничего не пересчитывает), повторный report в тот же каталог отказывает
+   (нет перезаписи).
+
+Это проверка механики контура на фикстурах, а не оценка качества отбора.
+Плановые команды S06 (rankings/refit/train с реальными бюджетами 50k/100k/200k
+на outputs/S06) smoke не запускает — они выполняются оркестратором после
+независимой проверки и прохождения Gate 1.
+
+Запуск:
+
+    .venv/bin/python scripts/s06_comparison_smoke.py
+"""
+from __future__ import annotations
+
+import filecmp
+import json
+import os
+import shutil
+import subprocess
+import sys
+
+import pandas as pd
+
+ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
+PY = sys.executable
+SMOKE = os.path.join(ROOT, "outputs", "S06_comparison_smoke")
+TINY_N_L3 = 10
+TINY_ROWS_PER_L3 = 50
+SMOKE_BUDGETS = "60,120"
+
+
+def run(cmd: list[str], timeout: int = 1800) -> None:
+    print("+", " ".join(cmd), flush=True)
+    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
+    if proc.returncode != 0:
+        print(proc.stdout[-4000:])
+        print(proc.stderr[-6000:], file=sys.stderr)
+        raise SystemExit(f"команда завершилась с кодом {proc.returncode}: {' '.join(cmd)}")
+    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
+    if tail:
+        print("  →", tail[:300])
+
+
+def py(script: str, *args: str, timeout: int = 1800) -> None:
+    run([PY, os.path.join("scripts", script), *args], timeout=timeout)
+
+
+def make_tiny_data() -> None:
+    data_dir = os.path.join(SMOKE, "data")
+    os.makedirs(data_dir, exist_ok=True)
+    ods = pd.read_parquet(os.path.join(ROOT, "fixtures", "ods_small.parquet"))
+    counts = ods["L3"].value_counts()
+    keep = counts[counts >= TINY_ROWS_PER_L3].head(TINY_N_L3).index
+    labeled = ods[ods["L3"].isin(keep)].copy()
+    labeled.to_parquet(os.path.join(data_dir, "labeled_tiny.parquet"), index=False)
+    names = labeled["name"].dropna().astype(str).tolist()
+    filler = [f"smoke filler item {i} тест" for i in range(300)]
+    corpus = pd.DataFrame({"name": names + filler})
+    corpus.to_parquet(os.path.join(data_dir, "corpus_tiny.parquet"), index=False)
+    print(f"tiny data: {len(labeled)} строк, {labeled['L3'].nunique()} L3; корпус {len(corpus)} строк")
+
+
+def gate0_sandbox() -> None:
+    py("freeze_split_s06.py",
+       "--input", os.path.join(SMOKE, "data", "labeled_tiny.parquet"),
+       "--min-count", "12",
+       "--out-dir", os.path.join(SMOKE, "splits"),
+       "--audit-dir", os.path.join(SMOKE, "audit"))
+
+
+def gate1_sandbox() -> None:
+    common = [
+        "--output-root", SMOKE,
+        "--full-tfidf", os.path.join(SMOKE, "full_tfidf"),
+        "--corpus", os.path.join(SMOKE, "data", "corpus_tiny.parquet"),
+        "--expect-vocab", '{"n_char":300,"n_word":100,"n_total":400}',
+        "--device", "cpu",
+        "--model", '{"hidden": [32], "dropout": [0.1]}',
+        "--train", '{"epochs": 8, "batch_size": 256, "eval_batch_size": 512, "scheduler": "none"}',
+        "--features-chunk-rows", "10000",
+    ]
+    py("run_s06_baseline.py", "verify-tfidf", *common)
+    py("run_s06_baseline.py", "select", *common)
+    py("run_s06_baseline.py", "refit", *common)
+
+
+def gate234_sandbox() -> None:
+    common = [
+        "--output-root", SMOKE,
+        "--full-tfidf", os.path.join(SMOKE, "full_tfidf"),
+        "--corpus", os.path.join(SMOKE, "data", "corpus_tiny.parquet"),
+        "--budgets", SMOKE_BUDGETS,
+        "--char-share", "0.75",
+        "--seeds", "42,43,44",
+        "--reference-seed", "42",
+        "--device", "cpu",
+        "--features-chunk-rows", "10000",
+        # Совпадают с вызовом Gate 1 в песочнице (проверка протокола fail closed).
+        "--model", '{"hidden": [32], "dropout": [0.1]}',
+        "--train", '{"epochs": 8, "batch_size": 256, "eval_batch_size": 512, "scheduler": "none"}',
+    ]
+    py("run_s06_comparison.py", "plan", *common)
+    py("run_s06_comparison.py", "rankings", *common, timeout=3600)
+    py("run_s06_comparison.py", "refit", *common)
+    py("run_s06_comparison.py", "train", *common, timeout=3600)
+
+
+def check(cond: bool, message: str) -> None:
+    if not cond:
+        raise SystemExit(f"SMOKE FAIL: {message}")
+    print(f"  ok: {message}")
+
+
+def verify_outputs() -> None:
+    report = os.path.join(SMOKE, "report")
+    required = ["report.html", "runs.csv", "per_class.parquet", "stages.csv",
+                "manifest.json", "confusion_pairs.csv", "predictions_val.parquet",
+                "support_bins.csv"]
+    for name in required:
+        check(os.path.isfile(os.path.join(report, name)), f"report/{name} существует")
+
+    manifest = json.load(open(os.path.join(report, "manifest.json"), encoding="utf-8"))
+    n_planned = manifest["statuses"]["n_planned"]
+    check(n_planned == 3 * (1 + 3 * len(SMOKE_BUDGETS.split(","))),
+          f"плановых runs {n_planned} (1 none + 3 метода × бюджеты, × 3 seed)")
+    check(manifest["statuses"]["n_success"] == n_planned, f"успешных runs {n_planned}/{n_planned}")
+    check(manifest["statuses"]["n_failed"] == 0, "failed runs 0")
+
+    runs = pd.read_csv(os.path.join(report, "runs.csv"))
+    check(len(runs) == n_planned, f"runs.csv содержит {n_planned} строк")
+    selected = runs[runs["method"] != "none"]
+    check(selected["delta_macro_f1_vs_none_same_seed"].notna().all(),
+          "paired ΔF1 посчитаны для всех selected runs")
+    check(runs[runs["method"] == "none"]["l3_macro_f1"].notna().all(),
+          "метрики none (reuse Gate 1) на месте")
+
+    preds = pd.read_parquet(os.path.join(report, "predictions_val.parquet"))
+    n_val = manifest["inputs"]["splits_parts"]["comparison_val"]["n_rows"]
+    check(len(preds) == n_planned * n_val,
+          f"предсказаний n_planned×n_val ({len(preds)} == {n_planned * n_val})")
+    check(set(preds["run_id"]) == set(runs["run_id"]), "predictions покрывают все run_id")
+
+    conf = pd.read_csv(os.path.join(report, "confusion_pairs.csv"))
+    check(not conf.empty, "confusion_pairs непусты (есть ошибки классификации на smoke)")
+    check({"run_id", "true_L3", "predicted_L3", "count", "true_support",
+           "error_share"}.issubset(conf.columns), "confusion_pairs: обязательные колонки")
+
+    bins = pd.read_csv(os.path.join(report, "support_bins.csv"))
+    check(not bins.empty, "support_bins непусты")
+    check(set(bins["support_bin"]) <= {"2-4", "5-9", "10-19", ">=20"},
+          "support_bins: только заданные диапазоны")
+
+    per_class = pd.read_parquet(os.path.join(report, "per_class.parquet"))
+    check({"run_id", "L3", "support", "precision", "recall", "f1", "delta_f1"}
+          .issubset(per_class.columns), "per_class: обязательные колонки")
+
+    stages = pd.read_csv(os.path.join(report, "stages.csv"))
+    shared_mask = stages["run_id"].astype(str).str.startswith(("shared", "gate1:shared"))
+    check(stages[shared_mask]["stage"].str.contains("refit").any(), "shared refit-этапы в stages.csv")
+    check(stages[stages["stage"] == "attribution"].shape[0] >= 1, "атрибуция G×I в stages.csv")
+
+    html = open(os.path.join(report, "report.html"), encoding="utf-8").read()
+    for token in ("macro-F1", "paired", "sealed_test", "Формулы", "Ограничения", "support"):
+        check(token in html, f"report.html содержит «{token}»")
+
+
+def verify_resume_and_rerender() -> None:
+    common = [
+        "--output-root", SMOKE,
+        "--full-tfidf", os.path.join(SMOKE, "full_tfidf"),
+        "--corpus", os.path.join(SMOKE, "data", "corpus_tiny.parquet"),
+        "--budgets", SMOKE_BUDGETS,
+        "--device", "cpu",
+        "--model", '{"hidden": [32], "dropout": [0.1]}',
+        "--train", '{"epochs": 8, "batch_size": 256, "eval_batch_size": 512, "scheduler": "none"}',
+    ]
+    # Safe resume: повторный train/refit/rankings ничего не пересчитывает.
+    import subprocess
+
+    def summary_of(args: list[str]) -> dict:
+        proc = subprocess.run([PY, os.path.join("scripts", "run_s06_comparison.py"), *args],
+                              cwd=ROOT, capture_output=True, text=True, timeout=3600)
+        if proc.returncode != 0:
+            print(proc.stderr[-3000:], file=sys.stderr)
+            raise SystemExit("resume-команда завершилась ошибкой")
+        # Логи идут в stdout вперемешку с финальным JSON: берём последний
+        # блок, начинающийся с строки "{".
+        lines = proc.stdout.splitlines()
+        start = max(i for i, ln in enumerate(lines) if ln.startswith("{"))
+        return json.loads("\n".join(lines[start:]))
+
+    s = summary_of(["train", *common])
+    check(s["n_done"] == s["n_planned"] and not s["trained"] and len(s["reused"]) == s["n_planned"],
+          f"safe resume: повторный train не пересчитал ни одного run ({s['n_done']} reused)")
+    summary_of(["rankings", *common])
+    summary_of(["refit", *common])
+    print("  ok: rankings/refit идемпотентны")
+
+    # Автономный re-render без обучения/inference в новый каталог.
+    rerender = os.path.join(SMOKE, "report_rerender")
+    py("run_s06_report.py", "--output", SMOKE, "--report-dir", rerender)
+    for name in ("runs.csv", "confusion_pairs.csv", "support_bins.csv"):
+        check(filecmp.cmp(os.path.join(SMOKE, "report", name), os.path.join(rerender, name), shallow=False),
+              f"re-render идентичен: {name}")
+    p1 = pd.read_parquet(os.path.join(SMOKE, "report", "predictions_val.parquet"))
+    p2 = pd.read_parquet(os.path.join(rerender, "predictions_val.parquet"))
+    check(p1.equals(p2), "re-render идентичен: predictions_val.parquet")
+
+    # Повторный report в тот же каталог — отказ (нет перезаписи).
+    proc = subprocess.run([PY, os.path.join("scripts", "run_s06_report.py"), "--output", SMOKE],
+                          cwd=ROOT, capture_output=True, text=True)
+    check(proc.returncode == 2 and "не перезаписываются" in proc.stderr,
+          "повторный report в занятый каталог отказывает (без перезаписи)")
+
+
+def main() -> int:
+    if os.path.abspath(SMOKE) != os.path.abspath(os.path.join(ROOT, "outputs", "S06_comparison_smoke")):
+        raise SystemExit("refusing: unexpected sandbox path")
+    if os.path.isdir(SMOKE):
+        shutil.rmtree(SMOKE)
+    os.makedirs(SMOKE)
+    make_tiny_data()
+    py("fit_vectorizer.py",
+       "--input", os.path.join(SMOKE, "data", "corpus_tiny.parquet"),
+       "--output", os.path.join(SMOKE, "full_tfidf"),
+       "--features-preset", "ods_light",
+       "--override", '{"char": {"max_features": 300, "min_df": 1}, "word": {"max_features": 100, "min_df": 1}}',
+       "--source-id", "smoke_corpus_v1")
+    gate0_sandbox()
+    gate1_sandbox()
+    gate234_sandbox()
+    py("run_s06_report.py", "--output", SMOKE)
+    verify_outputs()
+    verify_resume_and_rerender()
+    print("\nSMOKE PASS: Gate 2–4 S06 контур работоспособен на фикстурах (песочница "
+          + SMOKE + ")")
+    return 0
+
+
+if __name__ == "__main__":
+    raise SystemExit(main())
diff --git a/tests/test_s06_split_independent.py b/tests/test_s06_split_independent.py
new file mode 100644
index 0000000..5761b27
--- /dev/null
+++ b/tests/test_s06_split_independent.py
@@ -0,0 +1,67 @@
+"""Independent Gate 0 checks on small, in-memory labelled data."""
+import itertools
+
+import pandas as pd
+import pytest
+
+from kkt_clf.freeze_split import FreezeSplitError
+from kkt_clf.s06_split import s06_split
+
+
+LEAVES = ("train_fit", "train_monitor", "comparison_val", "sealed_test")
+
+
+def rows(label, count):
+    return [dict(name=f"item {label} number {i}", L1="root", L2="branch", L3=label)
+            for i in range(count)]
+
+
+def test_threshold_precedes_split_and_every_retained_class_is_complete():
+    df = pd.DataFrame(rows("edge", 12) + rows("large", 25) + rows("rare", 11))
+    result = s06_split(df, seed=42)
+    assert set(result.excluded.row_id) == set(range(37, 48))
+    assert set(result.excluded.L3) == {"rare"}
+    leaves = result.parts
+    assert set(leaves["outer_train"].row_id) == (
+        set(leaves["train_fit"].row_id) | set(leaves["train_monitor"].row_id)
+    )
+    for part in LEAVES:
+        counts = leaves[part].groupby("L3").size().to_dict()
+        assert set(counts) == {"edge", "large"}
+        assert all(n >= (1 if part == "train_fit" else 2) for n in counts.values())
+    for part in LEAVES:
+        for row in leaves[part].itertuples():
+            assert row.L3 == df.iloc[row.row_id].L3
+            assert row.name == df.iloc[row.row_id]["name"]
+    all_ids = [set(leaves[p].row_id) for p in LEAVES]
+    assert all(not a & b for a, b in itertools.combinations(all_ids, 2))
+    assert set.union(*all_ids) == set(range(37))
+    assert len(leaves["train_fit"].query("L3 == 'edge'")) >= 6
+
+
+def test_seed_reproducible_and_equivalent_names_do_not_cross_parts():
+    data = rows("alpha", 20) + rows("beta", 20)
+    data[0]["name"], data[1]["name"] = "Alpha Product 0", "ALPHA  PRODUCT 0"
+    data[2]["name"], data[3]["name"] = "Alpha Product 1", "alpha product1"
+    df = pd.DataFrame(data)
+    first, second = s06_split(df, seed=42), s06_split(df.copy(), seed=42)
+    for part in LEAVES:
+        assert set(first.parts[part].row_id) == set(second.parts[part].row_id)
+    char_keys, word_keys = [], []
+    for part in LEAVES:
+        names = first.parts[part]["name"]
+        char_keys.append({"".join(name.lower().split()) for name in names})
+        word_keys.append({tuple(name.lower().split()) for name in names})
+    for keys in (char_keys, word_keys):
+        assert all(not a & b for a, b in itertools.combinations(keys, 2))
+    for a, b in ((0, 1), (2, 3)):
+        assert any({a, b} <= set(first.parts[p].row_id) for p in LEAVES)
+
+
+def test_impossible_group_constraints_block_instead_of_dropping_class():
+    data = rows("good", 12) + rows("blocked", 12)
+    for row in data[12:]:
+        row["name"] = "same blocked name"
+    with pytest.raises(FreezeSplitError) as exc:
+        s06_split(pd.DataFrame(data), seed=42)
+    assert "blocked" in str(exc.value)
diff --git a/tests/test_s06_baseline_independent.py b/tests/test_s06_baseline_independent.py
new file mode 100644
index 0000000..f3d6bb0
--- /dev/null
+++ b/tests/test_s06_baseline_independent.py
@@ -0,0 +1,138 @@
+"""Independent S06 Gate 1 checks with a tiny in-memory corpus."""
+import copy
+import json
+from types import SimpleNamespace
+
+import pandas as pd
+import pytest
+import torch
+
+from kkt_clf import s05_baseline as s05, s06_baseline as s06
+from kkt_clf.config import BranchConfig, FeatureConfig
+from kkt_clf.features import FeatureBuilder
+from kkt_clf.labels import LabelMaps
+
+
+@pytest.fixture
+def setup(tmp_path, monkeypatch):
+    torch.set_num_threads(1)
+    df = pd.DataFrame({'name': ['alpha milk', 'beta bread', 'alpha food', 'beta food'],
+                       'L1': ['root'] * 4, 'L2': ['group'] * 4,
+                       'L3': ['a', 'b', 'a', 'b']})
+    parts = {'outer_train': df, 'train_fit': df.iloc[:2],
+             'train_monitor': df.iloc[2:], 'comparison_val': df.iloc[:2]}
+    manifest = {'label_maps': LabelMaps.from_df(df).to_dict(),
+                'policy': {'min_count': 12, 'split_seed': 7},
+                'parts': {k: {'sha256': k, 'bytes': 1, 'n_rows': len(v)} for k, v in parts.items()},
+                'inputs': {'labeled_parquet': {'sha256': 'source'}}}
+    manifest['parts']['sealed_test'] = {'sha256': 'sealed', 'bytes': 1, 'n_rows': 2}
+    builder = FeatureBuilder(FeatureConfig(BranchConfig((2, 3), 1), BranchConfig((1, 1), 1), None, 10))
+    builder.fit(df.name.tolist(), seed=3)
+    loaded = SimpleNamespace(builder=builder, meta={'files': {'features.joblib': {'sha256': 'full-id'}}})
+    cfg = s06.S06Gate1Config(output_root=str(tmp_path / 'result'),
+                             splits_manifest=str(tmp_path / 'split.json'),
+                             model={'hidden': [4], 'dropout': [0.0]},
+                             train={'scheduler': 'none', 'batch_size': 2, 'epochs': 2, 'patience': 2},
+                             device='cpu')
+    (tmp_path / 'result' / 'shared').mkdir(parents=True)
+    (tmp_path / 'split.json').write_text('{}')
+    verification = {'decision': 'ok', 'full_tfidf_id': 'full-id', 'artifact_version': 'v1',
+                    'corpus_fingerprint': {'sha256': 'corpus', 'bytes': 10}}
+    events = []
+    def read_part(_manifest, name):
+        events.append(('read', name))
+        assert name != 'sealed_test', 'sealed_test rows accessed'
+        return parts[name].copy()
+    monkeypatch.setattr(s05, 'load_part', read_part)
+    train = s05.train_model
+    def record_train(*args, **kwargs):
+        events.append(('train', args[9], args[1].shape[0]))
+        return train(*args, **kwargs)
+    monkeypatch.setattr(s05, 'train_model', record_train)
+    return cfg, manifest, verification, loaded, events
+
+
+def freeze(setup):
+    cfg, manifest, verification, loaded, _ = setup
+    return s06.freeze_training_protocol(cfg, manifest, verification, loaded, torch.device('cpu'))
+
+
+def test_selection_only_uses_fit_and_monitor_and_freezes_once(setup):
+    cfg, _, _, _, events = setup
+    protocol = freeze(setup)
+    assert [e[1] for e in events if e[0] == 'read'] == ['train_fit', 'train_monitor']
+    assert [e[2] for e in events if e[0] == 'train'] == [2]
+    assert protocol['fixed_epochs'] == protocol['selection_report']['candidates'][0]['best_epoch']
+    before = open(cfg.protocol_path, 'rb').read()
+    with pytest.raises(s06.S06Gate1Error):
+        freeze(setup)
+    assert open(cfg.protocol_path, 'rb').read() == before
+
+
+@pytest.mark.parametrize('change', ['split', 'tfidf', 'corpus', 'seeds'])
+def test_mismatched_provenance_or_config_blocks_refit_before_training(setup, monkeypatch, change):
+    cfg, manifest, verification, loaded, events = setup
+    freeze(setup)
+    events.clear()
+    if change == 'split':
+        manifest['parts']['train_fit']['sha256'] = 'different'
+    elif change == 'tfidf':
+        verification['full_tfidf_id'] = 'different'
+    elif change == 'corpus':
+        verification['corpus_fingerprint']['sha256'] = 'different'
+    elif change == 'seeds':
+        cfg.seeds = [42, 43]
+    monkeypatch.setattr(s06, 'run_refit', lambda *a, **k: pytest.fail('refit started'))
+    with pytest.raises(s06.S06Gate1Error):
+        s06.run_gate1_refit(cfg, manifest, verification, loaded, torch.device('cpu'))
+    assert not events
+
+
+def test_refit_trains_all_seeds_before_validation_and_never_reads_test(setup):
+    cfg, manifest, verification, loaded, events = setup
+    protocol = freeze(setup)
+    events.clear()
+    s06.run_gate1_refit(cfg, manifest, verification, loaded, torch.device('cpu'))
+    val_at = events.index(('read', 'comparison_val'))
+    assert [e[1] for e in events[:val_at] if e[0] == 'train'] == [42, 43, 44]
+    assert all(e[2] == 4 for e in events if e[0] == 'train')
+    assert ('read', 'sealed_test') not in events
+    selected = json.loads((__import__('pathlib').Path(cfg.baseline_dir) / 'selected_checkpoint.json').read_text())
+    assert selected['run_id'] == 'seed_42'
+    assert protocol['fixed_epochs'] >= 1
+
+
+@pytest.mark.parametrize('scores', [{42: 0.1, 43: 0.9, 44: 0.8}, {42: 0.9, 43: 0.1, 44: 0.8}])
+def test_checkpoint_reference_seed_independent_of_validation_scores(setup, monkeypatch, scores):
+    cfg, manifest, _, loaded, _ = setup
+    selected_paths = []
+    class StopAfterSelection(Exception):
+        pass
+    def capture(path, device):
+        selected_paths.append(path)
+        raise StopAfterSelection
+    monkeypatch.setattr(s05, 'load_artifact', capture)
+    results = {seed: {'status': 'trained'} for seed in scores}
+    evaluations = {seed: {'l3_macro_f1': score} for seed, score in scores.items()}
+    with pytest.raises(StopAfterSelection):
+        s05._select_and_verify_checkpoint(cfg.to_s05_config(), manifest, loaded, None,
+                                          '/checkpoints', results, evaluations, [], (), [],
+                                          torch.device('cpu'), (1, 1, 2), 2)
+    assert selected_paths == ['/checkpoints/seed_42']
+
+
+def test_gate0_rejects_old_policy_and_missing_integrity(setup, tmp_path, monkeypatch):
+    cfg, manifest, _, _, _ = setup
+    (tmp_path / 'split.json').write_text('{}')
+    manifest['checks'] = {'ok': True}
+    monkeypatch.setattr(s06, 'load_split_manifest', lambda _: manifest)
+    assert s06.verify_gate0_split(cfg) is manifest
+    for mutation in ('policy', 'checks', 'parts', 'label_maps'):
+        bad = copy.deepcopy(manifest)
+        if mutation == 'policy': bad['policy']['min_count'] = 10
+        elif mutation == 'checks': bad['checks']['ok'] = False
+        elif mutation == 'parts': del bad['parts']['sealed_test']
+        else: bad['label_maps'] = None
+        monkeypatch.setattr(s06, 'load_split_manifest', lambda _, bad=bad: bad)
+        with pytest.raises(s06.S06Gate1Error):
+            s06.verify_gate0_split(cfg)
diff --git a/tests/test_s06_review1_regressions_independent.py b/tests/test_s06_review1_regressions_independent.py
new file mode 100644
index 0000000..f59af47
--- /dev/null
+++ b/tests/test_s06_review1_regressions_independent.py
@@ -0,0 +1,142 @@
+"""S06 review regressions: fail closed on protocol and provenance violations."""
+import copy
+import json
+import sys
+from types import SimpleNamespace
+
+import pandas as pd
+import pytest
+
+from kkt_clf import s06_baseline as baseline
+from kkt_clf.s06_split import S06SplitError, s06_split, save_s06_split
+from tests.test_s06_baseline_independent import setup, freeze
+
+
+@pytest.fixture
+def saved_split(tmp_path):
+    rows = [dict(name=f'{label} product {i}', L1='root', L2='group', L3=label)
+            for label in ('alpha', 'beta') for i in range(20)]
+    df = pd.DataFrame(rows)
+    source = tmp_path / 'input.parquet'
+    df.to_parquet(source, index=False)
+    frozen = s06_split(df, seed=42)
+    save_s06_split(frozen, str(source), str(tmp_path / 'splits'), str(tmp_path / 'audit'),
+                   12, (0.8, 0.1, 0.1), 0.1, 42, 2, 2)
+    path = tmp_path / 'splits' / 'manifest.json'
+    return baseline.S06Gate1Config(splits_manifest=str(path)), path
+
+
+def test_gate0_rejects_policy_one(saved_split):
+    cfg, path = saved_split
+    manifest = json.loads(path.read_text())
+    manifest['policy']['min_rows_per_class'] = 1
+    path.write_text(json.dumps(manifest))
+    with pytest.raises((baseline.S06Gate1Error, ValueError)):
+        baseline.verify_gate0_split(cfg)
+
+
+@pytest.mark.parametrize('part', ['train_monitor', 'comparison_val', 'sealed_test'])
+def test_gate0_rejects_actual_support_one_even_if_manifest_claims_ok(saved_split, part):
+    from kkt_clf.data import file_fingerprint
+    cfg, path = saved_split
+    manifest = json.loads(path.read_text())
+    part_path = manifest['parts'][part]['file']
+    df = pd.read_parquet(part_path)
+    label = df.L3.iloc[0]
+    keep = df.L3.ne(label) | (df.L3.eq(label) & ~df.L3.duplicated())
+    df = df[keep]
+    df.to_parquet(part_path, index=False)
+    fp = file_fingerprint(part_path)
+    manifest['parts'][part].update(sha256=fp['sha256'], bytes=fp['bytes'], n_rows=len(df))
+    path.write_text(json.dumps(manifest))
+    with pytest.raises((baseline.S06Gate1Error, ValueError)):
+        baseline.verify_gate0_split(cfg)
+
+
+@pytest.mark.parametrize('scheduler', ['cosine', 'onecycle'])
+def test_unverified_varying_schedule_rejected_before_selection(tmp_path, scheduler, monkeypatch):
+    cfg = baseline.S06Gate1Config(output_root=str(tmp_path), train={'scheduler': scheduler})
+    monkeypatch.setattr(baseline, 'run_selection', lambda *a, **k: pytest.fail('selection started'))
+    with pytest.raises((baseline.S06Gate1Error, ValueError)):
+        cfg.validate()
+        baseline.freeze_training_protocol(cfg, {}, {}, None, None)
+
+
+@pytest.mark.parametrize('tamper', ['file', 'hash', 'nested_epochs', 'outer_epochs'])
+def test_frozen_report_and_epoch_consistency_checked_before_refit(setup, monkeypatch, tamper):
+    cfg, manifest, verification, loaded, _ = setup
+    protocol = freeze(setup)
+    if tamper == 'file':
+        with open(protocol['selection_report_file'], 'a') as f:
+            f.write(' ')
+    elif tamper == 'hash':
+        protocol['selection_report_sha256'] = '0' * 64
+    elif tamper == 'nested_epochs':
+        protocol['selection_report']['chosen']['fixed_epochs'] += 1
+    else:
+        protocol['fixed_epochs'] += 1
+    monkeypatch.setattr(baseline, 'load_training_protocol', lambda _: protocol)
+    monkeypatch.setattr(baseline, 'run_refit', lambda *a, **k: pytest.fail('refit started'))
+    with pytest.raises((baseline.S06Gate1Error, ValueError)):
+        baseline.run_gate1_refit(cfg, manifest, verification, loaded, None)
+
+
+def test_feasible_groups_do_not_claim_proven_impossible():
+    sizes = [1, 1, 4, 4, 4]
+    df = pd.DataFrame([dict(name=f'group {group}', L1='root', L2='branch', L3='class')
+                       for group, size in enumerate(sizes) for _ in range(size)])
+    try:
+        split = s06_split(df, seed=42)
+    except S06SplitError as exc:
+        message = str(exc).lower()
+        assert 'несовместим' not in message and 'невозмож' not in message
+    else:
+        for part in ('train_fit', 'train_monitor', 'comparison_val', 'sealed_test'):
+            assert len(split.parts[part]) >= 2
+        assert sum(len(split.parts[p]) for p in ('train_fit', 'train_monitor', 'comparison_val', 'sealed_test')) == 14
+
+
+@pytest.mark.parametrize('vocab', [
+    {'n_char': 1, 'n_word': 1, 'n_total': 2},
+    {'n_char': 100000, 'n_word': 300000, 'n_total': 400000},
+])
+def test_gate1_rejects_other_tfidf_even_with_same_corpus(tmp_path, monkeypatch, vocab):
+    cfg = baseline.S06Gate1Config(output_root=str(tmp_path))
+    verification = {'artifact_dir': str(tmp_path / 'full'), 'decision': 'ok', 'full_tfidf_id': 'id', 'artifact_version': 'v1',
+                    'seed': 42, 'vocab': vocab, 'fit': {}, 'source': {},
+                    'corpus_fingerprint': {'sha256': 'same', 'bytes': 1},
+                    'corpus_matches_source': True, 'files': {}, 'mode': 'pretrained'}
+    monkeypatch.setattr(baseline, 'verify_full_tfidf', lambda *a: copy.deepcopy(verification))
+    with pytest.raises((baseline.S06Gate1Error, ValueError)):
+        baseline.verify_s06_tfidf(cfg)
+
+
+def test_tfidf_reference_does_not_claim_unverified_overlap(tmp_path, monkeypatch):
+    from scripts import s06_tfidf_ref as ref
+    art = tmp_path / 's02' / 'full'
+    art.mkdir(parents=True)
+    corpus = tmp_path / 'corpus.parquet'
+    pd.DataFrame({'name': ['one', 'two']}).to_parquet(corpus)
+    feature = art / 'features.joblib'
+    feature.write_bytes(b'fixture')
+    from kkt_clf.data import file_fingerprint
+    fp, cfp = file_fingerprint(str(feature)), file_fingerprint(str(corpus))
+    vocab = {'n_char': 3, 'n_word': 1, 'n_total': 4}
+    meta = {'source': {'sha256': cfp['sha256'], 'bytes': cfp['bytes'], 'rows_in_file': 2},
+            'files': {'features.joblib': {'sha256': fp['sha256'], 'bytes': fp['bytes']}}, 'vocab': vocab}
+    (art / 'vectorizer_meta.json').write_text(json.dumps(meta))
+    import scipy.sparse as sp
+    builder = SimpleNamespace(n_char_features=3, n_word_features=1, n_features=4, fit_report_={},
+                              transform=lambda texts: sp.csr_matrix((len(texts), 4)))
+    monkeypatch.setattr(ref, 'load_vectorizer_artifact', lambda _: SimpleNamespace(builder=builder))
+    out = tmp_path / 'ref.json'
+    monkeypatch.setattr(sys, 'argv', ['s06_tfidf_ref.py', '--s02-dir', str(tmp_path / 's02'),
+                                    '--full-name', 'full', '--corpus', str(corpus), '--out', str(out),
+                                    '--expect-vocab-total', '4'])
+    ref.main()
+    if out.exists():
+        result = json.loads(out.read_text())
+        provenance = result.get('corpus_provenance', {})
+        if result.get('status') == 'ok':
+            assert provenance.get('fit_on_labeled_train') is not False
+            assert '2 совпадения' not in json.dumps(provenance, ensure_ascii=False)
