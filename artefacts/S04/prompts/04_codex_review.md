# Independent S04 review, round 1

You are the independent reviewer already invoked by the orchestrator; review directly, do not delegate or edit files. Review the full S04 contract and actual diff included below (tracked diff plus new production files), prioritizing correctness, data leakage, checkpoint/split identity, memory, end-to-end and report contract. Return concrete findings with file and line, severity and reproduction; state limitations. The smoke run on CPU with budgets 200/400/800 and 1 epoch reported 10 successes; baseline fast tests before implementation: 126 passed. Independent tests are underway. The smoke manifest says empty rows do not enter N_c, apparently contradicting D05; investigate. Do not assume passing smoke proves correctness. No commits or changes.

## Original requirements and Definition of Done
# S04 — Gradient × Input по L3-margin: отбор признаков и интеграция в экспериментальный пайплайн

**Статус:** постановка к реализации.  
**Проект:** `kkt_mlp_classifier`.  
**Приоритет:** проверенная, воспроизводимая, переносимая между открытым и закрытым контурами **методология**, а не достижение конкретной метрики на ODS.  
**Предусловие:** S03 интегрирован в актуальную основную ветку (`dev` по последнему отчёту; оркестратор обязан проверить). Существующие артефакты S02/S03 и пользовательские файлы не перезаписывать.

## Value

Добавить к двум методам отбора из S03 **нейросетевой class-aware feature selection на основе Gradient × Input**, объясняющий отрыв логита правильной L3 от её сильнейшего конкурента. Реализовать метод так, чтобы он использовал уже построенный полный TF-IDF и обученную на нём MLP, создавал независимый сокращённый TF-IDF посредством **повторного fit на том же корпусе** и включался в **единое сопоставимое сравнение запусков**.

В результате исследователь должен иметь возможность применять **тот же код, конфигурацию и отчётность** на ноутбуке с малой отладочной выборкой и в закрытом контуре с другими данными и бюджетами; рейтинги и выводы PoC в закрытый контур не переносятся.

## Согласованная методика

### 1. Что объясняем

Для обучающего примера `i` с истинным классом `y_i` и логитами головы L3 `z_i`:

- Конкурент `r_i = argmax_{c ≠ y_i} z_{i,c}`: сильнейший **другой** класс L3; при равенстве логитов — детерминированное правило по индексу класса.
- Целевая величина `m_i = z_{i,y_i} - z_{i,r_i}` (**logit margin**, не вероятность и не cross-entropy).
- Для **активного** TF-IDF признака `f`: `a_{i,f} = x_{i,f} · ∂m_i/∂x_{i,f}` (*Gradient × Input*). Индекс `r_i` фиксируется при вычислении градиента для данного примера.

Вычисление выполнять для обученной модели в `eval()` (без dropout, с корректной семантикой BatchNorm), по голове **L3**, с градиентом по **значениям ненулевых входных TF-IDF-признаков**. Изменять веса во время атрибуции нельзя. Численное значение отрыва и выбор конкурента должны соответствовать именно исходному полному TF-IDF и исходному checkpoint модели.

### 2. Какие примеры и вклады используем

- Истинная L3 известна **только из размеченного train**; validation/test не участвуют в вычислении или агрегировании важности. Модель для атрибуции обучается на train (validation может использоваться для обычного early stopping в рамках существующего протокола; это нужно явно указать в метаданных).
- **Ошибочно классифицированные** обучающие примеры сохраняются в расчёте. Конкурент определяется одинаково для правильных и ошибочных предсказаний. Сохранять диагностическое число правильных/ошибочных примеров.
- Для основного отбора использовать **только положительные** `a_{i,f}`: признак увеличивает margin правильного класса против конкурента. Отрицательные значения не ранжировать как полезные для отбора, но учитывать их в диагностических агрегатах (количество, сумма модулей, доля). Нулевые вклады не выдумывать.
- Пустые TF-IDF-векторы обрабатывать явно и учитывать в отчёте; градиенты для отсутствующих признаков не материализовать.

### 3. Class-aware ранжирование

Для каждой присутствующей в train категории `c` определять оценку признака как **средний положительный вклад по всем примерам класса, включая нули для отсутствующих и неположительных вкладов**:

`score(c,f) = (1 / N_c) · Σ_{i: y_i=c} max(0, a_{i,f})`, где `N_c` — число **фактически атрибутированных обучающих примеров** L3=`c` (после воспроизводимого отбора строк, если он включён).

Нормировка на `N_c` устраняет прямое преимущество массовых классов в рейтинге за счёт числа строк; она **не гарантирует статистической устойчивости** редких классов. Рейтинги строятся **отдельно для char и word**. При равных оценках — устойчивый порядок по идентификатору исходного признака. Для класса без положительных кандидатов формировать пустой рейтинг и фиксировать причину, не подменять её вымышленными значениями.

Применить механизм **равноправного циклического объединения категорий**, согласованный и реализованный в S03: в рамках отдельной квоты ветки идти по рейтингу каждой L3, выбирать ещё не выбранный признак, затем повторять. Добор до бюджета при исчерпании уникальных положительных кандидатов — детерминированно из глобального рейтинга тех же положительных атрибуций; если положительных кандидатов не хватает, допустим добор из остальных признаков в стабильном порядке с **обязательной маркировкой такого добора**. Не вводить в S04 альтернативные методы балансировки.

### 4. Бюджет и дальнейший пайплайн

- Метод в конфигурации: `gradient_x_input` (конкретные имена ключей согласовать с текущей схемой S03; не создавать второй независимый оркестратор эксперимента).
- Бюджеты на проверку: **50k / 100k / 200k**, базовая доля **char/word = 75/25**, оба параметра настраиваются. Бюджет — суммарное число **уникальных** признаков; фактические размеры веток отражать в артефактах и отчёте.
- **Полный TF-IDF и исходная MLP для атрибуции** сохраняются отдельно. После ранжирования создать новые char/word TfidfVectorizer с **фиксированными выбранными словарями** и выполнить **новый fit на точно той же фактической подвыборке исходного корпуса**, что использовалась для full TF-IDF. Не заменять это срезом готовой нормализованной матрицы и не переносить IDF вручную.
- **Финальные MLP обучаются с нуля** на каждом `tfidf_selected`. Атрибуционный checkpoint не является финальной MLP и не переносится путём обрезки весов. Инференс использует сохранённый `tfidf_selected` без fit.

## Scope

1. Конфигурируемый путь `full TF-IDF → обученная attribution MLP → Gradient × Input → class-aware Top-K → fixed-vocabulary TF-IDF refit → новая MLP → оценка`.
2. Повторное использование существующих механизмов S03: сплиты/label maps, бюджеты веток, циклическое объединение, фиксация provenance, selected TF-IDF, train/inference, подсчёт ресурсов и агрегированный отчёт. Методы `none / global / class_aware` Chi² и их прежние контракты остаются работоспособными.
3. Сериализуемые рейтинги, выбранные индексы char/word и связь с исходными полным векторизатором, checkpoint MLP, train-сплитом и параметрами расчёта.
4. Два режима запуска на одном коде: **smoke** (математическая корректность, короткий запуск на фикстуре) и **experiment** (осмысленно обученная attribution MLP и сопоставимые запуски); параметры обучения и отбора должны быть конфигурируемыми.
5. Обновление единого агрегированного отчёта — см. следующий раздел.

## Out of scope

- Integrated Gradients, SHAP, permutation/ablation selection, новые способы агрегирования сверх согласованного, изменение определения margin или автоматический подбор оптимального K.
- Выбор промышленно оптимального метода/бюджета на PoC; перенос ODS-рейтингов в закрытый контур.
- Обучение на 300 млн строк, Spark-реализация, распределённая атрибуция, гарантии RAM/VRAM и скорости в закрытом контуре. Алгоритм проектировать для ограниченной памяти, но промышленную применимость проверять отдельно.
- Смена модели/голов/таксономии, переопределение метрик отказа, переработка S03 ради эстетики, изменение `SPEC_v1.0.md`.
- Commit/push/merge, перезапись S02/S03, выгрузка данных и артефактов в Git без прямого разрешения.

## Контракт агрегированного отчёта S04

**Выход:** `outputs/S04/report/` (конфигурируемо, не коммитить), те же обязательные пять файлов и значения полей S03:

```text
report.html           единый автономный сводный отчёт по всем запланированным runs
runs.csv              одна строка на run, включая failed/skipped
per_class.parquet     метрики каждой L3 × успешный run
stages.csv            время/память по этапам и статусам
manifest.json         версия схемы, входы, сплиты, конфигурации, provenance
```

**Сопоставимый эксперимент:** планировать **10 конфигураций**: `none` (full) + `global` Chi² × 3 K + `class_aware` Chi² × 3 K + `gradient_x_input` × 3 K. Все на **одних и тех же train/validation/label maps, конфигурации финальной MLP, seed и протоколе raw top-1 L3 без отказа**. Если S03 outputs переиспользуются вместо повторных прогонов, их совместимость по *фактическим* данным, сплитам, версии TF-IDF, настройкам модели и протоколу оценки должна быть доказана; иначе несовместимые результаты не объединять. Test не использовать для selection/подбора/сравнения.

**Сохранять контракт S03**, не переименовывая и не меняя смысл обязательных полей `runs.csv`, `per_class.parquet`, `stages.csv`, `manifest.json`. Добавление нового метода `gradient_x_input`, новых стадий и диагностик допускается; при несовместимом изменении явно повысить `schema_version` и покрыть миграцию/проверки тестами. Не считать старые агрегаты S03 автоматически эквивалентными данным нового эксперимента. Дельты вычислять относительно `none` **только при сопоставимости**; пропущенные значения — `null`, не ноль.

**Дополнительные обязательные данные для G×I:**

- `manifest.json`: секция `attribution` с определением цели (`true_l3_vs_best_other_l3_logit_margin`), формулой score, checkpoint и fingerprint его весов/конфига, идентификатором full TF-IDF и train split, числом использованных строк по L3, долей correct/incorrect, параметрами/seed и способом отбора строк, вариантом положительного фильтра, политикой ties/пустых строк, этапом `eval`, используемыми ветками и диагностикой ограничений.
- `stages.csv`: отдельный этап `attribution` со временем, доступными замерами пиков RSS/VRAM и пояснением метода измерения; **стоимость предварительного обучения attribution MLP** отражать отдельно (`attribution_model_train` или явно именованный разделяемый этап) и **не прятать в** `mlp_train` финального запуска. Разделяемые затраты считать **один раз**, объяснять в отчёте, как они аллоцированы между тремя бюджетами.
- Рейтинги и selected-индексы сохранять отдельно от каталога `report/`, вместе с class-aware provenance и числом выбранных/добранных признаков по каждой ветке. Сохранять компактные сводки статистик положительных/отрицательных/нулевых атрибуций; не требовать сохранения значений атрибуций для каждой строки корпуса.
- `report.html`: единая таблица и графики `K → macro-F1 L3` и `качество ↔ ресурсы` для трёх методов, линия full baseline; отдельная секция G×I с числом атрибутированных примеров, correct/incorrect, пустыми строками, распределением положительных/отрицательных вкладов, покрытием классов и стоимостью attribution. Таблицы изменений F1/recall по L3 с support, низкие support явно отмечать. Все незавершённые/проваленные run показывать как таковые. Автономный HTML без CDN; генерация **из сохранённых результатов без retrain**.

**Разделение измерений:** стоимость предварительной атрибуционной MLP, атрибуции, selection/объединения, TF-IDF refit, финального train и validation inference не смешивать. Не складывать пиковую RAM различных этапов. Отдельно описывать логически общие одноразовые вычисления и суммарную стоимость сценария без двойного счёта.

**Интерпретация:** низкая F1 на ODS и быстрое обучение не считаются провалом корректной реализации. **Одноэпоховый smoke не доказывает полезность G×I**: для диагностического сравнения нужна достаточно обученная и зафиксированная attribution MLP; качество её обучения и число эпох приводить фактически, без обязательного порога F1. Не объявлять превосходство одного метода на PoC промышленным выводом.

## Definition of Done

- [ ] **D01 — Git и сохранность.** Ветка `sprint/S04-gradient-x-input` создана от актуальной интегрированной `dev` (или фактической основной ветки после проверки). Пользовательские изменения, корпус и исторические outputs S01–S03 не повреждены; нет несанкционированных commit/push/merge.
- [ ] **D02 — Конфиг и данные.** Параметризуются пути `full_tfidf`, размеченного корпуса, checkpoint attribution MLP (обучить/загрузить), бюджеты, квоты char/word, seed, объём/сэмплинг train для атрибуции и выходные каталоги. Зависимости от validation/test меток **в атрибуции/ранжировании отсутствуют**. Все inputs/сэмплы и label maps имеют проверяемые идентификаторы.
- [ ] **D03 — Корректный margin.** Выбор конкурента исключает истинную L3, при ties детерминирован; вычисляются именно `z_true − z_best_other` и `x · ∂m/∂x` с фиксированным competitor. Ошибки классификации не отбрасываются. Unit-тесты на маленькой модели с аналитически известным результатом и сравнение с независимым autograd/конечными разностями вдали от точек недифференцируемости.
- [ ] **D04 — Градиенты и ресурсы.** Атрибуции вычисляются в `model.eval()` и батчах по ненулевым TF-IDF значениям/CSR, без полного dense `N×F`, `B×F` и `C×F` и без обновления весов. Тесты на char/word сдвиг индексов, пустые векторы, ReLU и хотя бы один иной доступный activation, корректность CPU; GPU, если доступна, — дополнительная проверка либо явная оговорка.
- [ ] **D05 — Positive-only и class-aware.** `score(c,f)` реализована как среднее `max(0,a)` по **всем** примерам данного класса; отрицательные/нулевые не становятся положительным рейтингом, но отражены в диагностике. Классы с различной частотой, ошибочные примеры, пустой положительный рейтинг и tie cases покрыты тестами. Равноправное циклическое объединение и детерминированный добор повторно используют согласованный механизм S03.
- [ ] **D06 — Отбор и артефакты.** Для 50k/100k/200k формируются стабильные уникальные наборы признаков в пределах отдельных char/word квот; сохраняются исходные/новые индексы, score/provenance, версия исходного TF-IDF и checkpoint MLP. Не создавать O(C×F) плотные хранилища; последовательная агрегация/промежуточные данные ограничены по памяти с явной оценкой ресурсов на уровне ~1000 L3.
- [ ] **D07 — Selected TF-IDF.** Для каждого K новый fixed-vocabulary TF-IDF переобучен на **том же** фактическом корпусе и подвыборке, что full TF-IDF. Проверены fingerprint/настройки, совпадение IDF выбранных признаков при одинаковом корпусе, раздельная новая L2-нормировка и round-trip сохранённого векторизатора. Full и S03 артефакты нетронуты.
- [ ] **D08 — End-to-end и baseline.** Новые MLP обучаются с нуля на выбранных TF-IDF, сериализуются и работают после независимого `inference`; старые `none/global/class_aware` продолжают работать без регрессий. Проверены train→save→load→inference и воспроизводимость выбора при повторном прогоне с теми же inputs/seed. Нет изменения семантики L1/L2/L3, отказа и обработки пустых строк.
- [ ] **D09 — Эксперимент.** Запланированы все **10** конфигураций на фиксированных splits и eval-labels; выполнен локальный диагностический прогон либо ресурсно ограниченный общий smoke-прогон. Для содержательного G×I-эксперимента сохраняется отдельный checkpoint attribution MLP, обученный не только в режиме одноэпохового smoke; его фактическое качество/эпохи документируются. Неисполненные runs указываются как `failed/skipped` с причиной и не подменяются фиктивными метриками. Минимальный end-to-end для приёмки: `none`, один `gradient_x_input` и оба Chi² метода на **одном протоколе** (выполненные заново либо доказанно сопоставимые).
- [ ] **D10 — Единый отчёт.** Созданы все пять обязательных файлов в `outputs/S04/report/` с полями S03 и дополнениями S04, отражены 10 плановых runs и разделяемые стоимости. Проверены правильность delta, метрики per L3, статусы failed/skipped, support=0, отсутствие смешения результатов с разными splits и воспроизведение HTML без обучения. Нет внешних сетевых зависимостей.
- [ ] **D11 — Производительность и ограничения.** Измерены локальные время/RSS (и VRAM, если возможно) отдельно для attribution checkpoint training, G×I, selection, selected TF-IDF refit, финального train/inference. На синтетических тестах показано отсутствие полной dense аллокации `N×F`/`C×F`, оценены риски большого числа L3 и дороговизны attribution. Масштабируемость до промышленного корпуса **не объявляется проверенной**.
- [ ] **D12 — Тесты и независимое ревью.** Реализация — `kimi -p`, создание/изменение тестов — отдельные `codex exec`, ревью — другой независимый `codex exec`; бюджет **2 раунда**. Все фактически использованные задания сохранены в `artefacts/S04/prompts/` до исполнения. Прошли быстрый и полный pytest (либо причины невозможности документированы), `git diff --check`; незакрытые замечания и отсутствие повторного ревью последних исправлений явно указаны.
- [ ] **D13 — Документация и итог.** `artefacts/S04/result.md` содержит D01–D13 со статусами, команды, версии/коммиты, пути, статусы runs, результаты тестов/ревью, ресурсные замеры, известные ограничения и инструкции для воспроизведения. README/STATUS актуализированы, DECISION_LOG дополнен без переписывания истории, `SPEC_v1.0.md` неизменна.

## Условия приёмки

**Принимаем работоспособную методологию, а не достигнутый F1.** Отсутствие результатов промышленного масштаба или выигрышного Top-K не блокирует S04, если это честно отмечено. Некорректная формула, утечка val/test в атрибуцию, отсутствие refit фиксированного словаря, поломка train/inference, отсутствие проверяемого агрегированного отчёта или только имитация запуска **блокируют** приёмку. Дополнительные сравнения beyond smoke приветствуются, но нельзя объявлять их выполненными без фактических измерений.

## Original orchestration request
# Промпт Codex-оркестратору — S04 Gradient × Input

Ты — **Codex-оркестратор** спринта S04 проекта `kkt_mlp_classifier`. Твоя задача — организовать реализацию, независимые тесты, ревью и проверку методологии **Gradient × Input по margin правильной L3 относительно сильнейшего конкурента** в существующем экспериментальном пайплайне.

## Источник требований и приоритеты

**Главный контракт — `artefacts/S04/value_dod.md` (D01–D13).** Прочитай его целиком до планирования. Затем `AGENTS.md`, `artefacts/S03/value_dod.md`, `artefacts/S03/result.md`, `artefacts/DECISION_LOG.md`, `artefacts/STATUS.md`, README и фактические конфиги, CLI, селекторы, сериализацию и репортер S03. Особое внимание удели реализации round-robin class-aware, строителю selected TF-IDF, схемам `runs.csv`, `stages.csv`, `manifest.json` и метрикам raw top-1 L3.

Не придумывай новую архитектуру, если компоненты S03 переиспользуемы. Если постановка расходится с текущим кодом, документируй противоречие и выбирай минимальное решение, сохраняющее DoD. **Сейчас важна отладка переносимого метода на ноутбуке; реальные промышленные характеристики и качество на ODS не выводим.** `SPEC_v1.0.md` не менять.

## Организация и ограничения

1. Проверь `git status --short`, текущую основную ветку (ранее `dev`), наличие интегрированного S03 и сохранность пользовательских изменений. Создай `sprint/S04-gradient-x-input` от актуальной основной ветки. Не выполняй `reset --hard`, `clean`, force checkout, необратимых изменений. При небезопасном переключении — сообщи блокер/предложи `git worktree`.
2. В `artefacts/S04/prompts/` сохраняй **до запуска** каждый реально отправленный промпт с последовательными именами. Реализация и исправления — только через `kimi -p`; тесты/их изменения — отдельный `codex exec`; независимый code review — другой `codex exec`, с текстом `value_dod.md`, соответствующим diff и результатами проверок. Ты можешь править проектную документацию, но **не код и не тесты**.
3. **Бюджет ревью — 2 независимых раунда.** Исправления после последнего ревью не объявлять независимо принятыми. Если есть нерешённые замечания, показать конкретный остаток; не обходить ограничение самопроверкой.
4. Не делать commit, push, merge и не размещать корпуса, parquet, модели или outputs в Git без отдельного разрешения. Не перезаписывать `data/corpus.pqt`, S02/S03 outputs и результаты пользователя. Не менять исходные Value/DoD и промпт оркестратора.

## Последовательность работ

**A. Обследование и точка расширения.** Найди в актуальном коде S03 механизм формирования общей матрицы, fixed splits, класс-ориентированный merge, интерфейс selection, цикл MLP, разреженный вход `EmbeddingBag`, генерацию сравнения. Предложи минимальную интеграцию нового метода (ключ конфигурации `gradient_x_input`, параметры атрибуции). Убедись, что attribution-модель на full TF-IDF имеет те же `label_maps`/индексы, что ранжирование и оценка. Оцени требования памяти/времени и поддержку CPU/GPU.

**B. Корректный Gradient × Input.** Делегируй Kimi реализацию `r_i = argmax_{c≠y_i} z_ic`, `m_i = z_i,y_i − z_i,r_i`, `a_if = x_if · ∂m_i/∂x_if`, фиксируя `r_i` в градиентном шаге. В `model.eval()` без обновления параметров. Градиент по **активным значениям CSR** обеих веток, с учётом сдвига word-индексов и весов `EmbeddingBag`; запрещены полные dense `N×F`, `B×F`, `C×F`. Сильнейший конкурент берётся по логитам L3, даже при неверном top-1. Детерминированные ties и явные пустые векторы. Никакого использования val/test меток.

**C. Агрегирование и selected TF-IDF.** По каждому классу на train накопить средний `max(0,a_if)` со знаменателем **число атрибутированных примеров N_c в классе**, а отрицательные/нулевые значения вести как диагностику. Рейтинги char и word раздельны; переиспользовать S03 round-robin class-aware и детерминированный добор (причины добора отражать в метаданных). Бюджеты 50k/100k/200k, доли char/word 75/25 — **только дефолты эксперимента**. Рейтинги/индексы/хэши источников сохранить. Для каждого K заново выполнить S03 fixed-vocabulary refit на том же фактическом `corpus.pqt`, что full TF-IDF, и отдельно обучить финальную MLP. Не срезать нормализованную матрицу, не переносить IDF вручную.

**D. Атрибуционная MLP и протокол.** Поддержать два сценария: (1) загрузить проверенный checkpoint MLP на full TF-IDF; (2) обучить такую MLP из конфигурации. Для smoke допускается короткая учебная модель только для проверки корректности; для диагностической кривой надо обучить/зафиксировать MLP достаточно, чтобы рейтинг не был бессмысленной атрибуцией случайной инициализации. Не вводить жёсткий порог качества. Записывать эпохи, metric/accuracy на val, checkpoint, seed, версию данных; validation допускается для обычного early stopping, но **не для самого ранжирования**. Финальные MLP каждого K обучаются с нуля в идентичном протоколе.

**E. Сравнение и отчёт.** На единой размеченной train/validation и полном TF-IDF запланировать **10 запусков**: `none`, `global` × 3, `class_aware` × 3, `gradient_x_input` × 3. Точными остаются схемы 5 файлов отчёта S03; расширить `outputs/S04/report/` по требованиям S04, не делать второй параллельный репортер. При переиспользовании S03 runs **сначала проверить** одинаковые fingerprint корпуса, full TF-IDF, train/val, метки, модели и конфигурации. Несопоставимые runs не объединять, лучше перезапустить все на общем протоколе. Незапущенные результаты marked failed/skipped, не имитировать метрики. Отдельно показать фактическую стоимость обучения attribution MLP и вычисления градиентов; распределить общие затраты без двойного счёта. HTML должен строиться из сохранённых табличных результатов без retrain, автономно, без CDN.

**F. Тесты/ресурсы.** Делегируй тесты через **отдельную сессию** `codex exec`: известный аналитический margin и градиенты, численные конечные разности вдали от ReLU-boundary, ошибочные предсказания, ties, пустые признаки, char/word offset, `train`-only, positive-only class aggregation (с делением на N_c), round-robin, сохранение рейтингов, refit IDF/L2, train→save→load→inference, детерминизм. Отдельно проверь сценарий сломанного run при генерации отчёта и отсутствие некорректного нуля для непомеренного RAM. Измерь RSS/VRAM/время этапов; на синтетике проверь отсутствие запрещённых dense матриц. Не выдавай синтетический стресс-тест за промышленные измерения.

**G. Ревью и завершение.** Запусти быстрые и полные тесты `.venv/bin/python -m pytest -q -m 'not slow'`, `.venv/bin/python -m pytest -q`; `git diff --check` по изменяемым файлам (с учётом неотслеживаемых/пользовательских). Проведи до двух независимых ревью с отдельным `codex exec`. Организуй исправления через Kimi и повторные тесты; не делай самостоятельных кодовых исправлений.

## Обязательный результат и финальное сообщение

Подготовь `artefacts/S04/result.md`: статус каждого D01–D13 (`done/partial/blocked`), выполненные команды, список файлов и контрактных изменений, 10 плановых run и их фактические статусы, пути `outputs/S04/report/`, provenance/split hashes, нагрузку на RAM/CPU/GPU, замеры отдельно для attribution/checkpoint training/selection/refit/final MLP, результаты unit/integration/e2e, оба раунда независимого ревью и незакрытые замечания. Обнови README и `artefacts/STATUS.md`, допиши append-only `DECISION_LOG.md` при появлении решений.

В последнем ответе кратко сообщи: ветку, что реализовано, какие реально прогоны выполнены, состояние 5 файлов сводного отчёта, тесты/ревью, измеренные ресурсы, ограничения и что не проверено. **Не коммитить и не пушить без поручения. Не объявлять победителя по PoC.**

## Tracked diff against dev
diff --git a/kkt_clf/__init__.py b/kkt_clf/__init__.py
index d2a40bf..18bd7d3 100644
--- a/kkt_clf/__init__.py
+++ b/kkt_clf/__init__.py
@@ -1,4 +1,12 @@
 """Классификатор позиций кассовых чеков L1 → L2 → L3 (итерация 1, baseline)."""
+from .attribution import (
+    AttributionConfig,
+    AttributionError,
+    MarginAttributions,
+    build_gxi_ratings,
+    compute_margin_attributions,
+    select_features_gradient_x_input,
+)
 from .config import (
     DataConfig,
     InferenceConfig,
@@ -31,6 +39,8 @@ from .vectorizer import (
 
 __all__ = [
     "DataConfig",
+    "AttributionConfig", "AttributionError", "MarginAttributions",
+    "build_gxi_ratings", "compute_margin_attributions", "select_features_gradient_x_input",
     "InferenceConfig", "RunConfig", "build_config", "build_feature_config",
     "default_config", "ods_light_config",
     "ExperimentConfig", "SelectionPlanConfig", "ExperimentError",
diff --git a/kkt_clf/experiment.py b/kkt_clf/experiment.py
index fc280fc..158b172 100644
--- a/kkt_clf/experiment.py
+++ b/kkt_clf/experiment.py
@@ -55,7 +55,19 @@ import scipy.sparse as sp
 import torch
 from sklearn.metrics import precision_recall_fscore_support
 
-from .artifact import environment_versions, save_artifact
+from .artifact import environment_versions, load_artifact, save_artifact
+from .attribution import (
+    METHOD as GRADIENT_X_INPUT,
+    AttributionConfig,
+    AttributionError,
+    build_gxi_ratings,
+    checkpoint_fingerprint,
+    compute_margin_attributions,
+    save_attribution_report,
+    select_attributed_rows,
+    select_features_gradient_x_input,
+    verify_attribution_checkpoint,
+)
 from .config import (
     DataConfig,
     LossConfig,
@@ -81,11 +93,17 @@ from .vectorizer import LoadedVectorizer, load_vectorizer_artifact, refit_select
 
 log = logging.getLogger(__name__)
 
-EXPERIMENT_SCHEMA_VERSION = "s03-experiment@1"
-REPORT_SCHEMA_VERSION = "s03-report@1"
+EXPERIMENT_SCHEMA_VERSION = "s04-experiment@1"
+REPORT_SCHEMA_VERSION = "s04-report@1"
 RUNS_DIRNAME = "runs"
 REPORT_DIRNAME = "report"
+SHARED_DIRNAME = "shared"
+SHARED_STAGES_FILENAME = "attribution_stages.json"
+SHARED_RUN_ID = "shared_attribution"
 BASELINE_RUN_ID = "none"
+# Методы плановых прогонов: Chi²-отбор S03 + Gradient × Input S04. Режим
+# 'none' — контрольный прогон полного TF-IDF без отбора.
+PLAN_METHODS = (*SELECTION_METHODS, GRADIENT_X_INPUT)
 
 RUNS_COLUMNS = [
     "run_id", "status", "error", "method", "requested_k",
@@ -159,9 +177,9 @@ class SelectionPlanConfig:
     include_full: bool = True
 
     def validate(self) -> None:
-        bad = [m for m in self.methods if m not in SELECTION_METHODS]
+        bad = [m for m in self.methods if m not in PLAN_METHODS]
         if bad:
-            raise ExperimentError(f"selection.methods: одно из {SELECTION_METHODS}, получено {bad}")
+            raise ExperimentError(f"selection.methods: одно из {PLAN_METHODS}, получено {bad}")
         if not self.budgets:
             raise ExperimentError("selection.budgets: нужен хотя бы один бюджет K")
         for k in self.budgets:
@@ -196,6 +214,7 @@ class ExperimentConfig:
     loss: LossConfig = field(default_factory=LossConfig)
     train: TrainConfig = field(default_factory=TrainConfig)
     selection: SelectionPlanConfig = field(default_factory=SelectionPlanConfig)
+    attribution: AttributionConfig = field(default_factory=AttributionConfig)
     noisy_support: int = 2
 
     def validate(self) -> None:
@@ -210,6 +229,10 @@ class ExperimentConfig:
         if self.noisy_support < 0:
             raise ExperimentError("noisy_support: >= 0")
         self.selection.validate()
+        try:
+            self.attribution.validate()
+        except AttributionError as e:
+            raise ExperimentError(str(e)) from e
         # Валидация вложенных разделов через стандартный путь конфига прогона.
         validate_config(RunConfig(
             output_dir=self.output_dir, seed=self.seed, data=self.data,
@@ -529,6 +552,12 @@ class _ExperimentContext:
         }
         self._X_full_train: sp.csr_matrix | None = None
         self._X_full_val: sp.csr_matrix | None = None
+        # S04: разделяемые между gradient_x_input_* результаты атрибуции
+        # (модель, рейтинги, диагностика) либо зафиксированная ошибка подготовки.
+        self.attribution: AttributionBundle | None = None
+        self.attribution_error: str | None = None
+        self.attribution_meta: dict[str, Any] | None = None
+        self.shared_stage_rows: list[dict[str, Any]] = []
 
     @property
     def n_char_full(self) -> int:
@@ -556,6 +585,342 @@ class _ExperimentContext:
         return self._X_full_val, False
 
 
+# ---------------------------------------------------------------------------
+# Gradient × Input: разделяемая attribution MLP, атрибуция и рейтинги (S04)
+# ---------------------------------------------------------------------------
+
+
+@dataclass
+class AttributionBundle:
+    """Разделяемые между прогонами ``gradient_x_input_*`` результаты подготовки.
+
+    Модель атрибуции (checkpoint на полном TF-IDF), каталог полных G×I-рейтингов
+    (S03-формат), сводная диагностика и проверенные идентификаторы источников.
+    Стоимость подготовки измеряется один раз и фиксируется отдельными строками
+    ``stages.csv`` с ``run_id=shared_attribution`` (без двойного счёта между
+    бюджетами).
+    """
+
+    model: SparseMLP
+    checkpoint_dir: str
+    checkpoint: dict[str, Any]
+    ratings_dir: str
+    ratings_manifest: dict
+    report_path: str
+    report: dict
+    sampling: dict[str, Any]
+    verification: dict[str, Any]
+
+    def run_link(self) -> dict[str, Any]:
+        """Компактная аудируемая ссылка прогона на разделяемую атрибуцию."""
+        return {
+            "ratings_dir": os.path.abspath(self.ratings_dir),
+            "report_path": os.path.abspath(self.report_path),
+            "checkpoint_dir": os.path.abspath(self.checkpoint_dir),
+            "checkpoint": self.checkpoint,
+            "sampling": dict(self.sampling),
+            "verification": dict(self.verification),
+        }
+
+
+def _shared_measurement(
+    ctx: _ExperimentContext, stage: str, note: str, fn: Any
+) -> Any:
+    """Замер разделяемого этапа подготовки атрибуции: строка stages.csv с
+    run_id=shared_attribution; ошибка пробрасывается после фиксации строки."""
+    m = _Measurement(ctx.device, note)
+    try:
+        with m:
+            result = fn()
+    except Exception as e:
+        row = m.stage_row(SHARED_RUN_ID, stage, "failed")
+        row["measurement_note"] += f" Этап прерван: {type(e).__name__}: {e}"
+        ctx.shared_stage_rows.append(row)
+        raise
+    ctx.shared_stage_rows.append(m.stage_row(SHARED_RUN_ID, stage, "success"))
+    return result
+
+
+def _reject_info_stub() -> dict[str, Any]:
+    return {
+        "note": "S04: τ не калибровался; сравнение по raw top-1 на validation",
+        "tau_mode": "fixed",
+    }
+
+
+def _train_attribution_model(ctx: _ExperimentContext, out_dir: str) -> dict[str, Any]:
+    """Обучение attribution MLP на полном TF-IDF (train, early stopping по
+    validation — метрики валидации не используются для самого отбора)."""
+    cfg = ctx.cfg
+    checkpoint_dir = os.path.join(out_dir, "attribution_model")
+
+    def work() -> dict[str, Any]:
+        X_train, _ = ctx.ensure_full_train()
+        X_val, _ = ctx.ensure_full_val()
+        set_seed(cfg.seed)
+        model = SparseMLP(
+            ctx.full_builder.n_features,
+            tuple(ctx.maps.n_classes(lvl) for lvl in LEVELS),
+            cfg.model,
+        )
+        history, best, _trace = train_model(
+            model, X_train, ctx.y_train, X_val, ctx.y_val,
+            cfg.loss, cfg.train, ctx.class_weights, ctx.device, cfg.seed,
+        )
+        train_counts = {
+            lvl: ctx.data.splits["train"][lvl].value_counts().sort_index().astype(int).to_dict()
+            for lvl in LEVELS
+        }
+        save_artifact(
+            checkpoint_dir, model, cfg.model, ctx.full_builder, ctx.maps,
+            RejectRule(0.5, False, False), _reject_info_stub(),
+            {
+                "mode": cfg.loss.class_weight_mode,
+                "alpha": cfg.loss.class_weight_alpha,
+                "weights": {h: (None if w is None else w.tolist()) for h, w in ctx.class_weights.items()},
+                "train_counts": train_counts,
+            },
+            {
+                "experiment_run": "attribution",
+                "role": "gradient_x_input_attribution",
+                "config": cfg.to_dict(),
+            },
+            {
+                "seed": cfg.seed,
+                "n_features": int(ctx.full_builder.n_features),
+                "device": str(ctx.device),
+                "train": {
+                    "epochs_run": len(history),
+                    "best_epoch": int(best["epoch"]),
+                    "early_stopping_metric": best["metric"],
+                    "best_score": float(best["score"]),
+                    "final_lr": float(best["final_lr"]),
+                    "n_train_rows_used": int(best["n_train_rows_used"]),
+                },
+            },
+        )
+        last = history[-1] if history else {}
+        return {
+            "model": model,
+            "checkpoint_dir": checkpoint_dir,
+            "train_info": {
+                "epochs_run": len(history),
+                "best_epoch": int(best["epoch"]),
+                "early_stopping_metric": best["metric"],
+                "best_score": float(best["score"]),
+                "final_lr": float(best["final_lr"]),
+                "last_epoch_val_l3_macro_f1": last.get("val_l3_macro_f1"),
+                "last_epoch_val_l3_accuracy": last.get("val_l3_accuracy"),
+            },
+        }
+
+    result = _shared_measurement(
+        ctx, "attribution_model_train",
+        "обучение attribution MLP на полном TF-IDF (train, early stopping по "
+        "validation в рамках существующего протокола); разделяемый этап S04",
+        work,
+    )
+    result["fingerprint"] = checkpoint_fingerprint(checkpoint_dir)
+    return result
+
+
+def _load_attribution_model(ctx: _ExperimentContext) -> dict[str, Any]:
+    """Загрузка проверенного checkpoint'а attribution MLP из конфигурации."""
+    cfg = ctx.cfg
+    art_dir = cfg.attribution.checkpoint_path
+
+    def work() -> dict[str, Any]:
+        art = load_artifact(art_dir, ctx.device)
+        data_cfg = json.loads(
+            json.dumps(dataclasses.asdict(cfg.data), ensure_ascii=False, default=str)
+        )
+        verification = verify_attribution_checkpoint(
+            art_dir, art.model, art.config,
+            n_features_full=ctx.full_builder.n_features,
+            label_map_sha256=ctx.label_map_sha256,
+            data_cfg=data_cfg,
+            seed=cfg.seed,
+        )
+        return {
+            "model": art.model,
+            "checkpoint_dir": os.path.abspath(art_dir),
+            "train_info": (art.meta or {}).get("train"),
+            "verification": verification,
+            "artifact_config": art.config,
+        }
+
+    result = _shared_measurement(
+        ctx, "attribution_model_load",
+        "загрузка checkpoint'а attribution MLP с проверкой идентичности "
+        "n_features полного TF-IDF, label maps и data-конфига/seed (splits)",
+        work,
+    )
+    result["fingerprint"] = checkpoint_fingerprint(art_dir)
+    return result
+
+
+def _prepare_attribution(ctx: _ExperimentContext) -> None:
+    """Подготовка разделяемой атрибуции для прогонов ``gradient_x_input_*``.
+
+    Два сценария модели (S04): загрузить проверенный checkpoint из
+    ``attribution.checkpoint_path`` или обучить из конфигурации эксперимента.
+    Затем Gradient × Input по margin на (возможно, сэмплированном) размеченном
+    train и полные рейтинги на диск. Результаты — в ``output_dir/shared/``;
+    строки этапов — в ``ctx.shared_stage_rows`` (run_id=shared_attribution).
+    При ошибке фиксируется ``ctx.attribution_error``: прогоны G×I получат
+    статус failed с этой причиной, остальные методы не затрагиваются.
+    """
+    cfg = ctx.cfg
+    out_dir = os.path.join(ctx.out, SHARED_DIRNAME)
+    os.makedirs(out_dir, exist_ok=True)
+    ctx.shared_stage_rows = []
+    model = None
+    checkpoint: dict[str, Any] = {}
+    train_info = None
+    verification: dict[str, Any] = {}
+    try:
+        if cfg.attribution.checkpoint_path:
+            loaded = _load_attribution_model(ctx)
+            model = loaded["model"]
+            checkpoint = {
+                "path": loaded["checkpoint_dir"],
+                "source": "loaded",
+                **loaded["fingerprint"],
+            }
+            train_info = loaded.get("train_info")
+            verification = loaded.get("verification") or {}
+        else:
+            trained = _train_attribution_model(ctx, out_dir)
+            model = trained["model"]
+            checkpoint = {
+                "path": os.path.abspath(trained["checkpoint_dir"]),
+                "source": "trained_in_experiment",
+                **trained["fingerprint"],
+            }
+            train_info = trained["train_info"]
+        ctx.attribution_error = None
+    except Exception as e:  # noqa: BLE001 — фиксируем причину; прогоны G×I упадут осмысленно
+        ctx.attribution_error = f"{type(e).__name__}: {e}"[:1000]
+        log.exception("attribution model: failed")
+    if model is None:
+        ctx.attribution_meta = {"error": ctx.attribution_error}
+        _write_json(
+            {"stages": ctx.shared_stage_rows, "error": ctx.attribution_error},
+            os.path.join(out_dir, SHARED_STAGES_FILENAME),
+        )
+        return
+
+    ratings_dir = os.path.join(out_dir, "attribution", "ratings")
+    report_path = os.path.join(out_dir, "attribution", "report.json")
+
+    def attribute() -> dict[str, Any]:
+        X_full, built = ctx.ensure_full_train()
+        n_train = int(X_full.shape[0])
+        rows = select_attributed_rows(n_train, cfg.attribution, cfg.seed)
+        X_att = X_full[rows]
+        y_att = ctx.y_train[2][rows]
+        att = compute_margin_attributions(
+            model, X_att, y_att, ctx.device, batch_size=cfg.attribution.batch_size
+        )
+        manifest, report = build_gxi_ratings(att, ctx.n_char_full, ratings_dir, ctx.l3_labels)
+        report["sampling"] = {
+            "method": cfg.attribution.sampling,
+            "max_rows": cfg.attribution.max_rows,
+            "seed": int(cfg.seed),
+            "n_train_rows_total": n_train,
+            "n_rows_used": int(len(rows)),
+            "resampled_transform": bool(built),
+        }
+        report["checkpoint"] = dict(checkpoint)
+        report["train_info"] = train_info
+        save_attribution_report(report_path, report)
+        return {"manifest": manifest, "report": report, "rows": rows}
+
+    try:
+        done = _shared_measurement(
+            ctx, "attribution",
+            "Gradient × Input по margin истинной L3 против сильнейшего конкурента "
+            "на train-only (sparse CSR value gradients, model.eval(), без обновления "
+            "весов); агрегация class-aware рейтингов на диск; метки val/test не используются",
+            attribute,
+        )
+    except Exception as e:  # noqa: BLE001
+        ctx.attribution_error = f"{type(e).__name__}: {e}"[:1000]
+        log.exception("attribution: failed")
+        ctx.attribution_meta = {"error": ctx.attribution_error}
+        _write_json(
+            {"stages": ctx.shared_stage_rows, "error": ctx.attribution_error},
+            os.path.join(out_dir, SHARED_STAGES_FILENAME),
+        )
+        return
+    sampling = dict(done["report"]["sampling"])
+    ctx.attribution = AttributionBundle(
+        model=model,
+        checkpoint_dir=checkpoint["path"],
+        checkpoint=checkpoint,
+        ratings_dir=ratings_dir,
+        ratings_manifest=done["manifest"],
+        report_path=report_path,
+        report=done["report"],
+        sampling=sampling,
+        verification=verification,
+    )
+    full_fp = ctx.split_fingerprints.get("train", {}) if hasattr(ctx, "split_fingerprints") else {}
+    ctx.attribution_meta = {
+        "target": done["report"]["target"],
+        "formula": done["report"]["formula"],
+        "mode": (
+            "model.eval(); batch gradient по ненулевым CSR-значениям; веса не "
+            "обновлялись; competitor фиксировался до градиентного шага"
+        ),
+        "batch_size": int(cfg.attribution.batch_size),
+        "checkpoint": checkpoint,
+        "train_info": train_info,
+        "verification": verification,
+        "sampling": sampling,
+        "positive_filter": "rankings: max(0, a) > 0; negative/zero — только диагностика",
+        "empty_policy": (
+            "пустые TF-IDF-строки: margin вычисляется через bias-путь, вкладов нет, "
+            "в N_c не входят; их число — в диагностике"
+        ),
+        "ties_policy": (
+            "равные логиты конкурента → наименьший индекс класса; равные оценки "
+            "признака → наименьший индекс признака"
+        ),
+        "early_stopping_note": (
+            "attribution MLP обучалась с обычным early stopping по validation "
+            "(метрика train.*); метки validation/test в ранжирование не входят"
+        ),
+        "full_tfidf": {
+            "path": os.path.abspath(cfg.full_tfidf_path),
+            "artifact_version": ctx.full_meta.get("artifact_version"),
+            "source_sha256": (ctx.full_meta.get("source") or {}).get("sha256"),
+        },
+        "train_split": {
+            "sha256": (full_fp or {}).get("sha256"),
+            "label_map_sha256": getattr(ctx, "label_map_sha256", None),
+            "rows": sampling.get("n_train_rows_total"),
+        },
+        "report_path": os.path.abspath(report_path),
+        "ratings_manifest": done["manifest"],
+        "summary": {
+            "n_attributed_rows": done["report"]["n_attributed_rows"],
+            "n_empty_rows": done["report"]["n_empty_rows"],
+            "n_classes": done["report"]["n_classes"],
+            "n_classes_with_positive": done["report"]["n_classes_with_positive"],
+            "correct": done["report"]["correct"],
+            "incorrect": done["report"]["incorrect"],
+            "margins": done["report"]["margins"],
+            "contributions": done["report"]["contributions"],
+        },
+    }
+    _write_json(
+        {"stages": ctx.shared_stage_rows, "error": None},
+        os.path.join(out_dir, SHARED_STAGES_FILENAME),
+    )
+    log.info("Атрибуция подготовлена: %s", report_path)
+
+
 # ---------------------------------------------------------------------------
 # Выполнение одного прогона
 # ---------------------------------------------------------------------------
@@ -660,11 +1025,70 @@ def _run_none(plan: PlannedRun, ctx: _ExperimentContext, run_result: dict, run_d
     run_result["artifacts"]["vectorizer"] = os.path.abspath(ctx.cfg.full_tfidf_path)
     _train_mlp(ctx, ctx.full_builder.n_features, X_train, X_val, run_result, run_dir,
                ctx.full_builder, tag=plan.run_id)
-    from .artifact import load_artifact
     model = load_artifact(run_result["artifacts"]["model"], ctx.device).model
     _validate_and_store_metrics(ctx, model, X_val, run_result)
 
 
+def _select_for_run(
+    plan: PlannedRun, ctx: _ExperimentContext, run_result: dict, run_dir: str, sel_cfg: SelectionConfig
+) -> tuple[Any, str]:
+    """Этап selection прогона: Chi² (S03) или Gradient × Input (S04).
+
+    Оба варианта — только по размеченному train полного TF-IDF; метки val/test
+    не используются. Для G×I рейтинги построены один раз (shared/): здесь
+    выполняется class-aware Top-K round-robin и сохранение артефакта отбора с
+    полным provenance (источник TF-IDF, размеченные данные, checkpoint
+    атрибуции). Возвращает (SelectionResult, абсолютный путь артефакта отбора).
+    """
+    cfg = ctx.cfg
+    if plan.method == GRADIENT_X_INPUT:
+        if ctx.attribution is None:
+            raise ExperimentError(
+                f"атрибуция Gradient × Input недоступна: {ctx.attribution_error}"
+            )
+        bundle = ctx.attribution
+        note = (
+            "class-aware Top-K round-robin из предварительно вычисленных G×I-рейтингов "
+            "(shared, train-only, см. run_id=shared_attribution); метки val/test не используются"
+        )
+    else:
+        bundle = None
+        note = (
+            "supervised Chi²-отбор только по размеченному train полного TF-IDF; "
+            "метки val/test не используются"
+        )
+    with _stage(ctx.device, note, run_result, "selection") as m:
+        if plan.method == GRADIENT_X_INPUT:
+            result = select_features_gradient_x_input(
+                bundle.ratings_dir, sel_cfg, ctx.n_char_full,
+                int(ctx.full_builder.n_features), l3_labels=ctx.l3_labels,
+            )
+        else:
+            X_full, built = ctx.ensure_full_train()
+            if built:
+                m.note += "; включён первичный transform полного train (результат закэширован)"
+            result = select_features(
+                X_full, ctx.y_train[2], sel_cfg, ctx.n_char_full, l3_labels=ctx.l3_labels
+            )
+        sel_dir = os.path.join(run_dir, "selection")
+        provenance: dict[str, Any] = {
+            "vectorizer": {
+                "path": os.path.abspath(cfg.full_tfidf_path),
+                "source_sha256": (ctx.full_meta.get("source") or {}).get("sha256"),
+                "artifact_version": ctx.full_meta.get("artifact_version"),
+            },
+            "labeled": {"path": os.path.abspath(cfg.data.path), "split": "train"},
+            "experiment": {"run_id": plan.run_id, "seed": cfg.seed},
+        }
+        if bundle is not None:
+            provenance["attribution"] = bundle.run_link()
+        save_selection_artifact(
+            sel_dir, result, provenance=provenance, overwrite=True,
+            ratings_copy=plan.method == GRADIENT_X_INPUT,
+        )
+    return result, os.path.abspath(sel_dir)
+
+
 def _run_selected(plan: PlannedRun, ctx: _ExperimentContext, run_result: dict, run_dir: str) -> None:
     """Прогон с отбором: selection (train) → refit на том же корпусе → MLP → метрики."""
     cfg = ctx.cfg
@@ -672,33 +1096,11 @@ def _run_selected(plan: PlannedRun, ctx: _ExperimentContext, run_result: dict, r
         method=plan.method, k=int(plan.requested_k),
         char_share=cfg.selection.char_share, seed=cfg.seed,
     )
-    with _stage(
-        ctx.device,
-        "supervised Chi²-отбор только по размеченному train полного TF-IDF; метки val/test не используются",
-        run_result, "selection",
-    ) as m:
-        X_full, built = ctx.ensure_full_train()
-        if built:
-            m.note += "; включён первичный transform полного train (результат закэширован)"
-        result = select_features(
-            X_full, ctx.y_train[2], sel_cfg, ctx.n_char_full, l3_labels=ctx.l3_labels
-        )
-        sel_dir = os.path.join(run_dir, "selection")
-        save_selection_artifact(
-            sel_dir, result,
-            provenance={
-                "vectorizer": {
-                    "path": os.path.abspath(cfg.full_tfidf_path),
-                    "source_sha256": (ctx.full_meta.get("source") or {}).get("sha256"),
-                    "artifact_version": ctx.full_meta.get("artifact_version"),
-                },
-                "labeled": {"path": os.path.abspath(cfg.data.path), "split": "train"},
-                "experiment": {"run_id": plan.run_id, "seed": cfg.seed},
-            },
-            overwrite=True,
-        )
+    result, sel_dir = _select_for_run(plan, ctx, run_result, run_dir, sel_cfg)
     run_result["selection_report"] = result.report
-    run_result["artifacts"]["selection"] = os.path.abspath(sel_dir)
+    run_result["artifacts"]["selection"] = sel_dir
+    if plan.method == GRADIENT_X_INPUT and ctx.attribution is not None:
+        run_result["attribution"] = ctx.attribution.run_link()
 
     with _stage(
         ctx.device,
@@ -729,7 +1131,6 @@ def _run_selected(plan: PlannedRun, ctx: _ExperimentContext, run_result: dict, r
 
     _train_mlp(ctx, refit.builder.n_features, X_train, X_val, run_result, run_dir,
                refit.builder, tag=plan.run_id)
-    from .artifact import load_artifact
     model = load_artifact(run_result["artifacts"]["model"], ctx.device).model
     _validate_and_store_metrics(ctx, model, X_val, run_result)
 
@@ -759,6 +1160,7 @@ def _fresh_run_result(plan: PlannedRun, ctx: _ExperimentContext) -> dict[str, An
         "val_inference_seconds": None,
         "selection_report": None,
         "refit_checks": None,
+        "attribution": None,
         "device": str(ctx.device),
         "versions": environment_versions(),
     }
@@ -806,9 +1208,10 @@ def _write_experiment_meta(ctx: _ExperimentContext, planned: list[PlannedRun]) -
     cfg = ctx.cfg
     git_commit, git_dirty = _git_info()
     full_dir = os.path.abspath(cfg.full_tfidf_path)
+    label = "S04" if any(p.method == GRADIENT_X_INPUT for p in planned) else "S03"
     meta = {
         "schema_version": EXPERIMENT_SCHEMA_VERSION,
-        "experiment_id": f"S03-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
+        "experiment_id": f"{label}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
         "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
         "git_commit": git_commit,
         "git_dirty": git_dirty,
@@ -866,7 +1269,15 @@ def _write_experiment_meta(ctx: _ExperimentContext, planned: list[PlannedRun]) -
         },
         "data_report": ctx.data.report,
         "metric_protocol": dict(METRIC_PROTOCOL),
-        "limitations": list(LIMITATION_NOTES),
+        "attribution": ctx.attribution_meta,
+        "limitations": list(LIMITATION_NOTES) + (
+            [
+                "gradient_x_input: интерпретация обученной attribution MLP не доказывает "
+                "полезность признака после переобучения; одноэпоховый smoke не является "
+                "доказательством качества метода."
+            ]
+            if ctx.attribution_meta is not None else []
+        ),
     }
     path = os.path.join(ctx.out, "experiment.json")
     _write_json(meta, path)
@@ -908,6 +1319,14 @@ def run_experiment(
         if unknown:
             raise ExperimentError(f"--only: неизвестные run_id {unknown}; план: {[p.run_id for p in planned]}")
     ctx = _prepare_context(cfg, device)
+    # S04: разделяемая attribution MLP и G×I-рейтинги готовятся один раз до
+    # прогонов, если среди выполняемых есть gradient_x_input_*.
+    gxi_planned = [
+        p for p in planned
+        if p.method == GRADIENT_X_INPUT and (only is None or p.run_id in only)
+    ]
+    if gxi_planned:
+        _prepare_attribution(ctx)
     meta_path = _write_experiment_meta(ctx, planned)
     log.info("Мета эксперимента: %s", meta_path)
 
@@ -1212,13 +1631,54 @@ def _per_class_frame(
     return pd.DataFrame(out, columns=PER_CLASS_COLUMNS)
 
 
-def _stages_frame(planned: list[dict], results: dict[str, dict]) -> pd.DataFrame:
+def _load_shared_stage_rows(output_dir: str) -> list[dict[str, Any]]:
+    """Читает строки разделяемых этапов атрибуции (shared/attribution_stages.json).
+
+    Повреждённый или отсутствующий файл не ломает генерацию отчёта: возвращается
+    пустой список. Содержимое проверяется по минимальной структуре строки.
+    """
+    path = os.path.join(output_dir, SHARED_DIRNAME, SHARED_STAGES_FILENAME)
+    if not os.path.isfile(path):
+        return []
+    try:
+        raw = _read_json(path)
+    except (OSError, json.JSONDecodeError) as e:
+        log.warning("Разделяемые этапы атрибуции не прочитаны (%s): пропускаю", e)
+        return []
+    rows = raw.get("stages") if isinstance(raw, dict) else None
+    if not isinstance(rows, list):
+        return []
+    out: list[dict[str, Any]] = []
+    for row in rows:
+        if not isinstance(row, dict) or not isinstance(row.get("stage"), str):
+            continue
+        row = dict(row)
+        row["run_id"] = SHARED_RUN_ID
+        if row.get("status") not in _RUN_STATUSES:
+            row["status"] = "skipped"
+        for key in ("seconds", "peak_rss_bytes", "peak_vram_bytes"):
+            if key in row and not _is_finite_number(row[key]):
+                row[key] = None
+        if not isinstance(row.get("measurement_note"), str):
+            row["measurement_note"] = (
+                "" if row.get("measurement_note") is None else str(row.get("measurement_note"))
+            )
+        out.append(row)
+    return out
+
+
+def _stages_frame(
+    planned: list[dict], results: dict[str, dict], output_dir: str | None = None
+) -> pd.DataFrame:
     """Строки этапов всех плановых прогонов.
 
     Для прогонов без полного набора этапов (failed/skipped, старые или
     частичные ``run_result.json``) недостигнутые этапы дописываются строками
     ``skipped`` с пояснением: так стоимость неуспешной работы не скрывается и
-    контракт stages.csv выполняется для каждого планового прогона.
+    контракт stages.csv выполняется для каждого планового прогона. При заданном
+    ``output_dir`` добавляются разделяемые этапы атрибуции
+    (``run_id=shared_attribution``): их затраты учтены один раз и не дублируются
+    в прогонах ``gradient_x_input_*``.
     """
     rows = []
     for p in planned:
@@ -1240,6 +1700,8 @@ def _stages_frame(planned: list[dict], results: dict[str, dict]) -> pd.DataFrame
                 reason = "этап не зафиксирован в сохранённых результатах прогона"
             for name in missing:
                 rows.append(_skipped_stage(p["run_id"], name, reason))
+    if output_dir is not None:
+        rows.extend(_load_shared_stage_rows(output_dir))
     return pd.DataFrame(rows, columns=STAGES_COLUMNS)
 
 
@@ -1271,6 +1733,7 @@ def build_manifest(output_dir: str, exp: dict, runs_df: pd.DataFrame) -> dict[st
             "loss": (exp.get("config") or {}).get("loss"),
         },
         "metric_protocol": exp.get("metric_protocol"),
+        "attribution": exp.get("attribution"),
         "planned_runs": [
             {
                 "run_id": row.run_id, "method": row.method, "requested_k": row.requested_k,
@@ -1489,7 +1952,7 @@ def _html_report(
     # Ось K — все запланированные бюджеты (включая failed/skipped): значения
     # незапущенных конфигураций не дорисовываются, на их месте — разрыв линии.
     ks = sorted({float(k) for k in runs_df["requested_k"].dropna()})
-    colors = {"global": "#1f77b4", "class_aware": "#ff7f0e"}
+    colors = {"global": "#1f77b4", "class_aware": "#ff7f0e", GRADIENT_X_INPUT: "#9467bd"}
 
     def series_of(metric: str) -> list[tuple[str, str, dict[float, float]]]:
         out = []
@@ -1569,11 +2032,100 @@ def _html_report(
             ["run_id", "method", "requested_k", "status", "error"],
         )
 
+    # Секция Gradient × Input: разделяемые этапы (attribution MLP, атрибуция),
+    # целевая величина, число атрибутированных примеров, correct/incorrect,
+    # пустые строки, распределение вкладов, покрытие классов и стоимость.
+    attribution = manifest.get("attribution") if isinstance(manifest, dict) else None
+    gxi_html = ""
+    if attribution:
+        shared_stages = stages_df[stages_df["run_id"] == SHARED_RUN_ID]
+        shared_tbl = _table_html(
+            shared_stages,
+            ["stage", "status", "seconds", "peak_rss_bytes", "peak_vram_bytes"],
+            formats={
+                "seconds": lambda v: f"{v:.2f}",
+                "peak_rss_bytes": lambda v: f"{v / (1024 * 1024):.0f} МБ",
+                "peak_vram_bytes": lambda v: f"{v / (1024 * 1024):.0f} МБ",
+            },
+        ) if len(shared_stages) else '<p class="muted">Этапы атрибуции не зафиксированы.</p>'
+        summary = attribution.get("summary") or {}
+        contributions = summary.get("contributions") or {}
+        correct = summary.get("correct") or {}
+        incorrect = summary.get("incorrect") or {}
+        margins = summary.get("margins") or {}
+        sampling = attribution.get("sampling") or {}
+        checkpoint = attribution.get("checkpoint") or {}
+
+        def _c(key: str) -> Any:
+            block = contributions.get(key) or {}
+            if not block:
+                return "—"
+            if key == "negative":
+                return f"{block.get('count')} (|sum|={_fmt(block.get('abs_sum'), 4)}, доля={_fmt(block.get('share'), 4)})"
+            return f"{block.get('count')} (sum={_fmt(block.get('sum'), 4)}, доля={_fmt(block.get('share'), 4)})"
+
+        rows = [
+            ("Цель", attribution.get("target")),
+            ("Формула", attribution.get("formula")),
+            ("Режим", attribution.get("mode")),
+            ("Сэмплинг train", f"{sampling.get('method')} (max_rows={sampling.get('max_rows')}, seed={sampling.get('seed')}), "
+                               f"использовано {sampling.get('n_rows_used')} из {sampling.get('n_train_rows_total')}"),
+            ("Checkpoint", f"{checkpoint.get('path')} (источник: {checkpoint.get('source')}; "
+                           f"sha256 model.pt: {str((checkpoint.get('model.pt') or {}).get('sha256'))[:16]}…)"),
+            ("Атрибутировано примеров", summary.get("n_attributed_rows")),
+            ("Правильных предсказаний", f"{correct.get('count')} (доля {_fmt(correct.get('fraction'), 4)})"),
+            ("Ошибочных предсказаний", f"{incorrect.get('count')} (доля {_fmt(incorrect.get('fraction'), 4)})"),
+            ("Пустых TF-IDF-строк", summary.get("n_empty_rows")),
+            ("Положительные вклады", _c("positive")),
+            ("Отрицательные вклады", _c("negative")),
+            ("Нулевые вклады", f"{(contributions.get('zero') or {}).get('count')} "
+                               f"(доля {_fmt((contributions.get('zero') or {}).get('share'), 4)})"),
+            ("Margin: mean/std", f"{_fmt(margins.get('mean'), 4)} / {_fmt(margins.get('std'), 4)}"),
+            ("Margin: min/max", f"{_fmt(margins.get('min'), 4)} / {_fmt(margins.get('max'), 4)}"),
+            ("Классов L3 (с положительными кандидатами)",
+             f"{summary.get('n_classes')} ({summary.get('n_classes_with_positive')})"),
+        ]
+        summary_tbl = _table_html(
+            pd.DataFrame(rows, columns=["показатель", "значение"]), ["показатель", "значение"]
+        )
+        train_info = attribution.get("train_info") or {}
+        train_note = ""
+        if train_info:
+            train_note = (
+                f"<p class=\"muted\">Attribution MLP: эпох выполнено {train_info.get('epochs_run')}, "
+                f"лучшая {train_info.get('best_epoch')} "
+                f"({train_info.get('early_stopping_metric')}={_fmt(train_info.get('best_score'), 4)}). "
+                "Одноэпоховый smoke не доказывает полезность G×I; жёсткий порог качества не вводился.</p>"
+            )
+        error_note = ""
+        if attribution.get("error"):
+            error_note = f'<p class="warn">Атрибуция завершилась ошибкой: {_esc(attribution["error"])}</p>'
+        gxi_html = f"""
+<h2>Gradient × Input (атрибуция по L3-margin)</h2>
+{error_note}
+{summary_tbl}
+{train_note}
+<h3>Разделяемые этапы атрибуции (run_id={_esc(SHARED_RUN_ID)})</h3>
+{shared_tbl}
+<p class="caption">Затраты обучения attribution MLP и самой атрибуции учтены один раз под
+run_id={_esc(SHARED_RUN_ID)} и разделяются между всеми прогонами gradient_x_input_*; при подсчёте
+суммарной стоимости сценария добавляйте их один раз (без двойного счёта). Метод замера: {RSS_SAMPLING_NOTE}</p>
+"""
+
     inputs = exp.get("inputs") or {}
     splits = exp.get("splits") or {}
     lm = exp.get("label_maps") or {}
     proto = exp.get("metric_protocol") or {}
     env = exp.get("environment") or {}
+    report_label = "S04" if "s04" in str(manifest.get("schema_version", "")).lower() else "S03"
+    gxi_runs = runs_df[runs_df["method"] == GRADIENT_X_INPUT]
+    gxi_note = ""
+    if len(gxi_runs):
+        gxi_note = (
+            " · gradient_x_input: " + ", ".join(
+                f"{r.run_id}={r.status}" for r in gxi_runs.itertuples()
+            )
+        )
 
     runs_tbl = _table_html(
         runs_df,
@@ -1621,7 +2173,7 @@ def _html_report(
 <html lang="ru">
 <head>
 <meta charset="utf-8">
-<title>S03 — агрегированный отчёт по отбору признаков</title>
+<title>{report_label} — агрегированный отчёт по отбору признаков</title>
 <style>
 body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 24px; color: #222; }}
 h1 {{ font-size: 22px; }} h2 {{ font-size: 18px; margin-top: 28px; }} h3 {{ font-size: 15px; }}
@@ -1642,9 +2194,9 @@ code {{ background: #f5f5f5; padding: 1px 4px; }}
 </style>
 </head>
 <body>
-<h1>S03 — supervised feature selection: агрегированный отчёт</h1>
+<h1>{report_label} — supervised feature selection: агрегированный отчёт</h1>
 <p class="muted">experiment_id: {_esc(exp.get('experiment_id'))} · создан {_esc(manifest.get('date'))} ·
-git {_esc(str(exp.get('git_commit'))[:10])} (dirty={_esc(exp.get('git_dirty'))}) · устройство {_esc(env.get('device'))}</p>
+git {_esc(str(exp.get('git_commit'))[:10])} (dirty={_esc(exp.get('git_dirty'))}) · устройство {_esc(env.get('device'))}{gxi_note}</p>
 
 <div class="warn"><strong>Ограничение интерпретации.</strong> ODS/PoC не является оценкой промышленной
 применимости: результаты диагностические, оптимальный метод/бюджет здесь не объявляется промышленным решением.</div>
@@ -1658,6 +2210,7 @@ git {_esc(str(exp.get('git_commit'))[:10])} (dirty={_esc(exp.get('git_dirty'))})
 {time_chart}
 {size_chart}
 {base_size_note}
+{gxi_html}
 
 <h2>Качество ↔ ресурсы</h2>
 {qr_tbl}
@@ -1691,7 +2244,7 @@ seed={_esc(((corpus.get('row_selection') or {}).get('seed')))}, source_id={_esc(
 {_esc((splits.get('train') or {}).get('bytes'))}/{_esc((splits.get('val') or {}).get('bytes'))}/{_esc((splits.get('test') or {}).get('bytes'))} байт
 (sha256 в manifest.json); label maps: {len(lm.get('l3_train_counts') or {})} L3 (хэш состава — в manifest.json).</li>
 <li>Отчёт построен автоматически из runs.csv, per_class.parquet, stages.csv, manifest.json;
-повторная генерация без обучения: <code>python scripts/run_s03_experiment.py report --output {_esc(output_dir)}</code>.</li>
+повторная генерация без обучения: <code>python scripts/run_{_esc(report_label.lower())}_experiment.py report --output {_esc(output_dir)}</code>.</li>
 </ul>
 {_table_html(version_rows, ['компонент', 'версия'])}
 <p class="muted">Ограничения: {'; '.join(_esc(x) for x in (exp.get('limitations') or []))}</p>
@@ -1720,6 +2273,7 @@ def _exp_from_manifest(manifest: dict) -> dict:
         "splits": manifest.get("splits") or {},
         "label_maps": manifest.get("label_maps") or {},
         "metric_protocol": manifest.get("metric_protocol") or {},
+        "attribution": manifest.get("attribution"),
         "limitations": manifest.get("resource_limitations") or [],
         "planned_runs": manifest.get("planned_runs") or [],
     }
@@ -1785,7 +2339,7 @@ def build_report(output_dir: str) -> dict[str, str]:
         baseline = results.get(BASELINE_RUN_ID)
         runs_df = _runs_frame(planned, results, baseline)
         per_class_df = _per_class_frame(planned, results, baseline)
-        stages_df = _stages_frame(planned, results)
+        stages_df = _stages_frame(planned, results, output_dir)
         manifest = build_manifest(output_dir, exp, runs_df)
 
         os.makedirs(report_dir, exist_ok=True)
diff --git a/kkt_clf/selection.py b/kkt_clf/selection.py
index 614c23a..7702aef 100644
--- a/kkt_clf/selection.py
+++ b/kkt_clf/selection.py
@@ -77,6 +77,7 @@ _RATINGS_STREAM_CHUNK = 8192
 # Происхождение отобранного признака (записывается в payload["origin"]).
 ORIGIN_CLASS = "class"           # выбран из ранжированного списка категории
 ORIGIN_GLOBAL_FALLBACK = "global_fallback"  # добор из глобального рейтинга
+ORIGIN_REMAINDER_FILL = "remainder_fill"    # добор остатка в стабильном порядке (G×I)
 
 
 class SelectionError(RuntimeError):
@@ -422,6 +423,128 @@ class _RankStream:
                 return f, s
 
 
+def roundrobin_select_from_rank_files(
+    idx_path: str,
+    score_path: str,
+    counts: np.ndarray,
+    classes: np.ndarray,
+    k: int,
+    n_feat: int,
+    global_scores: np.ndarray,
+    positive_only_fallback: bool = False,
+) -> tuple[BranchSelection, dict]:
+    """Равноправный циклический выбор из ранжированных списков категорий.
+
+    Единый механизм class-aware объединения S03/S04: классы обходятся в
+    детерминированном (отсортированном) порядке, из списка каждой категории за
+    цикл берётся не больше одного ещё не выбранного признака; порядок внутри
+    списка — по убыванию оценки, равные оценки разрываются возрастанием
+    индекса признака. Потоки разделяют два файловых дескриптора
+    (``idx_path``/``score_path``), буфер потока — не более ``chunk`` элементов:
+    точный round-robin без хранения ``C×K`` кандидатов в RAM.
+
+    Добор до бюджета детерминирован: сначала из глобального рейтинга
+    (``positive_only_fallback=False`` — top-k глобального рейтинга целиком,
+    поведение S03; ``True`` — только глобально положительные оценки, G×I), и,
+    если положительных кандидатов не хватает, из остальных признаков в
+    стабильном порядке возрастания индекса с маркировкой
+    ``remainder_fill``. Возвращает (BranchSelection, report) с числом
+    классов, положительными кандидатами на класс, числом доборов
+    (``n_fallback``, ``n_remainder_fill``).
+    """
+    counts = np.asarray(counts, dtype=np.int64)
+    classes = np.asarray(classes, dtype=np.int64)
+    global_scores = np.asarray(global_scores, dtype=np.float64)
+    n_classes = len(classes)
+    if len(counts) != n_classes:
+        raise SelectionError("counts/classes не согласованы по длине")
+    if global_scores.shape[0] != n_feat:
+        raise SelectionError(
+            f"global_scores: длина {global_scores.shape[0]} != n_feat {n_feat}"
+        )
+    used = np.zeros(n_feat, dtype=bool)
+    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]) if n_classes else np.empty(0, dtype=np.int64)
+    streams: list[_RankStream] = []
+    f_i = f_s = None
+    try:
+        f_i = open(idx_path, "rb")
+        f_s = open(score_path, "rb")
+        streams = [
+            _RankStream(f_i, f_s, int(counts[ci]), int(offsets[ci]))
+            for ci in range(n_classes)
+        ]
+        selected: list[int] = []
+        sel_scores: list[float] = []
+        sel_origin: list[str] = []
+        while len(selected) < k:
+            progressed = False
+            for ci in range(n_classes):
+                if len(selected) >= k:
+                    break
+                nxt = streams[ci].next_unused(used)
+                if nxt is None:
+                    continue
+                f, s = nxt
+                used[f] = True
+                selected.append(f)
+                sel_scores.append(s)
+                sel_origin.append(f"{ORIGIN_CLASS}:{int(classes[ci])}")
+                progressed = True
+            if not progressed:
+                break
+    finally:
+        streams = []
+        if f_s is not None:
+            f_s.close()
+        if f_i is not None:
+            f_i.close()
+
+    n_fallback = 0
+    if len(selected) < k:
+        if positive_only_fallback:
+            # Добор из глобального рейтинга тех же положительных атрибуций.
+            pos_idx = np.flatnonzero(global_scores > 0)
+            order_g = np.lexsort((pos_idx, -global_scores[pos_idx]))
+            fallback_pool = pos_idx[order_g]
+        else:
+            fallback_pool = _rank_topk(global_scores, k)
+        for f in fallback_pool:
+            if len(selected) >= k:
+                break
+            if used[f]:
+                continue
+            used[f] = True
+            selected.append(int(f))
+            sel_scores.append(float(global_scores[f]))
+            sel_origin.append(ORIGIN_GLOBAL_FALLBACK)
+            n_fallback += 1
+    n_remainder_fill = 0
+    if positive_only_fallback and len(selected) < k:
+        # Положительных кандидатов не хватает: добор остальных признаков в
+        # стабильном порядке (возрастание индекса) — обязательно с маркировкой.
+        for f in range(n_feat):
+            if len(selected) >= k:
+                break
+            if used[f]:
+                continue
+            used[f] = True
+            selected.append(int(f))
+            sel_scores.append(float(global_scores[f]))
+            sel_origin.append(ORIGIN_REMAINDER_FILL)
+            n_remainder_fill += 1
+    order = np.argsort(np.asarray(selected, dtype=np.int64), kind="stable")
+    idx = np.asarray(selected, dtype=np.int64)[order]
+    scores = np.asarray(sel_scores, dtype=np.float64)[order]
+    origin = [sel_origin[i] for i in order]
+    report = {
+        "n_classes": int(n_classes),
+        "n_positive_per_class": [int(c) for c in counts],
+        "n_fallback": int(n_fallback),
+        "n_remainder_fill": int(n_remainder_fill),
+    }
+    return BranchSelection(indices=idx, scores=scores, origin=origin), report
+
+
 def _select_branch_global(
     X_b: sp.csr_matrix, y: np.ndarray, k: int,
     ratings_dir: str | None = None, branch: str | None = None,
@@ -505,78 +628,21 @@ def _select_branch_class_aware(
     counts.tofile(counts_path)
     _write_global_ratings(ratings_dir, label, global_scores)
 
-    # Равноправный round-robin: классы в детерминированном (отсортированном)
-    # порядке; из списка каждой категории за цикл берётся не больше одного
-    # признака, поэтому потоковое чтение точно воспроизводит полный обход.
-    # Потоки разделяют два файловых дескриптора, поэтому при любом числе
-    # классов дескрипторов не больше двух и закрыть их можно в finally.
-    used = np.zeros(n_feat, dtype=bool)
-    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]) if n_classes else np.empty(0, dtype=np.int64)
-    streams: list[_RankStream] = []
-    f_i = f_s = None
-    try:
-        f_i = open(idx_path, "rb")
-        f_s = open(score_path, "rb")
-        streams = [
-            _RankStream(f_i, f_s, int(counts[ci]), int(offsets[ci]))
-            for ci in range(n_classes)
-        ]
-        selected: list[int] = []
-        sel_scores: list[float] = []
-        sel_origin: list[str] = []
-        while len(selected) < k:
-            progressed = False
-            for ci in range(n_classes):
-                if len(selected) >= k:
-                    break
-                nxt = streams[ci].next_unused(used)
-                if nxt is None:
-                    continue
-                f, s = nxt
-                used[f] = True
-                selected.append(f)
-                sel_scores.append(s)
-                sel_origin.append(f"{ORIGIN_CLASS}:{int(classes[ci])}")
-                progressed = True
-            if not progressed:
-                break
-    finally:
-        streams = []
-        if f_s is not None:
-            f_s.close()
-        if f_i is not None:
-            f_i.close()
+    # Равноправный round-robin и добор из глобального рейтинга — общий
+    # механизм с G×I (roundrobin_select_from_rank_files): классы в
+    # детерминированном порядке, из списка каждой категории за цикл берётся не
+    # больше одного признака, потоки разделяют два файловых дескриптора.
     try:
-        # Детерминированный добор из глобального рейтинга до бюджета. Пул
-        # кандидатов формируется всегда (top-k глобального рейтинга): добор
-        # использует его, только если round-robin не заполнил бюджет.
-        fallback_pool = _rank_topk(global_scores, k)
-        n_fallback = 0
-        if len(selected) < k:
-            for f in fallback_pool:
-                if len(selected) >= k:
-                    break
-                if used[f]:
-                    continue
-                used[f] = True
-                selected.append(int(f))
-                sel_scores.append(float(global_scores[f]))
-                sel_origin.append(ORIGIN_GLOBAL_FALLBACK)
-                n_fallback += 1
-        order = np.argsort(np.asarray(selected, dtype=np.int64), kind="stable")
-        idx = np.asarray(selected, dtype=np.int64)[order]
-        scores = np.asarray(sel_scores, dtype=np.float64)[order]
-        origin = [sel_origin[i] for i in order]
-        report = {
-            "n_classes": int(n_classes),
-            "n_positive_per_class": [int(c) for c in counts],
-            "n_fallback": int(n_fallback),
-        }
-        return BranchSelection(indices=idx, scores=scores, origin=origin, ratings_dir=ratings_dir), report
+        sel, report = roundrobin_select_from_rank_files(
+            idx_path, score_path, counts, classes, k, n_feat, global_scores,
+            positive_only_fallback=False,
+        )
     except Exception:
         if own_dir:
             shutil.rmtree(ratings_dir, ignore_errors=True)
         raise
+    sel.ratings_dir = ratings_dir
+    return sel, report
 
 
 def select_features(
@@ -690,6 +756,7 @@ def save_selection_artifact(
     result: SelectionResult,
     provenance: dict,
     overwrite: bool = False,
+    ratings_copy: bool = False,
 ) -> str:
     """Сохраняет версионированный артефакт отбора: payload (выбранные
     признаки/индексы/происхождение + манифест полных рейтингов), метаданные
@@ -697,15 +764,19 @@ def save_selection_artifact(
     рейтинги) и сами полные рейтинги (каталог ``ratings/``).
 
     Полные рейтинги — самостоятельный результат отбора: глобальный рейтинг
-    обеих веток и, для ``class_aware``, ранжированные списки положительных
-    кандидатов по каждой категории. По ним можно проверить позиции невошедших
-    признаков и применить другой бюджет без пересчёта Chi²; они не
-    загружаются в RAM целиком.
+    обеих веток и ранжированные списки положительных кандидатов по каждой
+    категории (``class_aware`` Chi² и ``gradient_x_input``). По ним можно
+    проверить позиции невошедших признаков и применить другой бюджет без
+    пересчёта оценок; они не загружаются в RAM целиком.
 
     ``provenance`` — сведения об источниках: как минимум ``vectorizer``
     (путь/идентификатор полного артефакта TF-IDF и его отпечаток) и ``labeled``
     (путь/отпечаток размеченных данных) — копируется в метаданные как есть.
 
+    ``ratings_copy=True`` копирует рейтинги в артефакт, не забирая исходный
+    каталог (нужно для G×I: рейтинги в ``shared/`` переиспользуются всеми
+    бюджетами; артефакт каждого прогона остаётся самодостаточным для аудита).
+
     Сохранение **транзакционно** (в том числе при ``overwrite=True``): payload,
     рейтинги и метаданные собираются в соседнем скрытом staging-каталоге, и
     только после успешной сборки staging атомарно переносится на место
@@ -732,7 +803,10 @@ def save_selection_artifact(
         payload_path = os.path.join(staging, "selection.joblib")
         joblib.dump(result.to_payload(), payload_path)
         ratings_dest = os.path.join(staging, RATINGS_DIRNAME)
-        shutil.move(result.ratings_dir, ratings_dest)
+        if ratings_copy:
+            shutil.copytree(result.ratings_dir, ratings_dest)
+        else:
+            shutil.move(result.ratings_dir, ratings_dest)
         files = {
             "selection.joblib": {
                 "sha256": _sha256_file(payload_path),
@@ -755,9 +829,12 @@ def save_selection_artifact(
             "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             "versions": environment_versions(),
             "notes": [
-                "Рейтинги Chi² вычислены по размеченному train; метки val/test не использовались.",
-                "Ожидание: feature_sum * n_class / n_train; global — стандартный многоклассовый "
-                "Chi² (sklearn.feature_selection.chi2).",
+                "Рейтинги вычислены по размеченному train; метки val/test не использовались.",
+                "Chi²: ожидание feature_sum * n_class / n_train; global — стандартный "
+                "многоклассовый Chi² (sklearn.feature_selection.chi2).",
+                "gradient_x_input: score(c,f) = mean_{i∈c} max(0, x·∂m/∂x) по margin "
+                "истинной L3 против сильнейшего конкурента (S04); отрицательные/нулевые "
+                "вклады только в диагностике; добор остатка маркируется remainder_fill.",
                 "Полные рейтинги (каталог ratings/, формат kkt-selection-ratings@2) — "
                 "самостоятельный результат отбора: позиции всех кандидатов и повторное "
                 "применение бюджета без пересчёта; целиком в RAM не загружаются; "

## New production files
"""Gradient × Input по L3-margin (S04): атрибуция, рейтинги и class-aware отбор.

Метод объясняет отрыв логита истинной L3 от сильнейшего конкурента. Для
обучающего примера ``i`` с истинным классом ``y_i`` и логитами головы L3
``z_i``:

- конкурент ``r_i = argmax_{c ≠ y_i} z_{i,c}`` — сильнейший **другой** класс;
  при равенстве логитов побеждает наименьший индекс класса (детерминированно);
- margin ``m_i = z_{i,y_i} − z_{i,r_i}`` (логит, не вероятность/CE);
- для **активного** TF-IDF признака ``f``: ``a_{i,f} = x_{i,f} · ∂m_i/∂x_{i,f}``
  (Gradient × Input). Индекс ``r_i`` фиксируется до градиентного шага и внутри
  него не дифференцируется.

Свойства реализации:

- атрибуция выполняется для обученной модели в ``model.eval()`` (dropout выкл.,
  BatchNorm по running statistics) **без обновления весов**: градиент считается
  ``torch.autograd.grad`` по значениям ненулевых элементов CSR
  (``per_sample_weights`` EmbeddingBag), поэтому не материализуются и не
  требуются плотные матрицы ``N×F`` и ``B×F``;
- градиенты для отсутствующих признаков не создаются: вклад существует только
  в позициях ненулевых значений CSR; пустые TF-IDF-векторы обрабатываются явно
  (margin для них определён через bias-путь, но в N_c не входят и вкладов не
  имеют);
- ошибочно классифицированные примеры сохраняются; конкурент определяется
  одинаково для правильных и ошибочных предсказаний; фиксируются доли
  correct/incorrect и диагностика по margin;
- class-aware ранжирование: ``score(c,f) = (1/N_c)·Σ_{i: y_i=c} max(0, a_{i,f})``
  по **всем** атрибутированным примерам класса (неналичие признака и
  неположительные вклады дают ноль в сумме); отрицательные и нулевые вклады в
  рейтинг не попадают, но полностью отражаются в диагностике (число, сумма
  модулей, доля). Рейтинги char/word раздельны; равные оценки разрываются
  возрастанием индекса исходного признака;
- объединение категорий — тот же равноправный циклический механизм S03
  (:func:`kkt_clf.selection.roundrobin_select_from_rank_files`) по полным
  рейтингам на диске (RAM не хранит ``C×F`` и ``C×K``); при нехватке
  положительных кандидатов — детерминированный добор из глобального рейтинга
  положительных атрибуций, затем из остальных признаков в стабильном порядке
  с обязательной маркировкой ``remainder_fill``;
- метки validation/test в атрибуцию и ранжирование не попадают: вход — только
  размеченный train (возможен воспроизводимый сэмплинг строк, см.
  :class:`AttributionConfig`).

Полные рейтинги сохраняются в том же формате, что и S03
(``kkt-selection-ratings@2``), — по ним можно повторно применить бюджет без
пересчёта атрибуции. Это инструментология PoC: интерпретация обученной модели
не доказывает полезность признака после переобучения, а результаты на ODS не
переносятся в закрытый контур.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import torch

from .artifact import environment_versions
from .model import SparseBatch, csr_to_batch
from .selection import (
    RATINGS_FORMAT_VERSION,
    BranchSelection,
    SelectionConfig,
    SelectionError,
    SelectionResult,
    branch_budgets,
    roundrobin_select_from_rank_files,
)

log = logging.getLogger(__name__)

METHOD = "gradient_x_input"
TARGET_NAME = "true_l3_vs_best_other_l3_logit_margin"
FORMULA = (
    "m_i = z_{i,y_i} - z_{i,r_i}, r_i = argmax_{c != y_i} z_{i,c} "
    "(ties: lowest class index, fixed before the gradient step); "
    "a_{i,f} = x_{i,f} * dm_i/dx_{i,f} over nonzero CSR values; "
    "score(c,f) = (1/N_c) * sum_{i: y_i=c} max(0, a_{i,f})"
)

SAMPLING_HEAD = "head"
SAMPLING_RANDOM = "random"
SAMPLINGS = (SAMPLING_HEAD, SAMPLING_RANDOM)


class AttributionError(RuntimeError):
    """Базовая ошибка операций Gradient × Input-атрибуции."""


@dataclass
class AttributionConfig:
    """Параметры Gradient × Input-атрибуции (D02).

    ``checkpoint_path`` — загрузить проверенный checkpoint MLP на полном
    TF-IDF; ``None`` — обучить attribution MLP из конфигурации эксперимента
    (общие model/loss/train, full TF-IDF, early stopping по validation в рамках
    существующего протокола). ``max_rows``/``sampling``/seed эксперимента —
    воспроизводимый отбор размеченных train-строк для атрибуции (None — все
    строки). ``batch_size`` — батчи ненулезных CSR-значений; плотные ``B×F`` не
    создаются при любом размере.
    """

    checkpoint_path: str | None = None
    max_rows: int | None = None
    sampling: str = SAMPLING_HEAD
    batch_size: int = 4096

    def validate(self) -> None:
        if self.max_rows is not None and (not isinstance(self.max_rows, int) or self.max_rows <= 0):
            raise AttributionError(f"attribution.max_rows: None или целое > 0, получено {self.max_rows!r}")
        if self.sampling not in SAMPLINGS:
            raise AttributionError(f"attribution.sampling: одно из {SAMPLINGS}, получено {self.sampling!r}")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise AttributionError(f"attribution.batch_size: целое > 0, получено {self.batch_size!r}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class MarginAttributions:
    """Результат Gradient × Input по margin на размеченном train.

    ``values`` — CSR float64 той же формы, что и входная матрица: значения
    ``a_{i,f} = x_{i,f}·∂m_i/∂x_{i,f}`` ровно в позициях ненулевых элементов
    (градиенты отсутствующих признаков не материализуются). Для пустых строк
    значений нет. ``margins``/``competitor``/``correct`` определены для всех
    строк (включая пустые: margin через bias-путь), ``empty`` — маска пустых.
    ``rows`` — исходные индексы строк входной матрицы (после сэмплинга).
    """

    values: sp.csr_matrix
    margins: np.ndarray       # float64 [N]
    competitor: np.ndarray    # int64 [N], класс-индекс r_i (всегда != y_i)
    correct: np.ndarray       # bool [N], top-1 == y_i
    empty: np.ndarray         # bool [N], нулевой TF-IDF-вектор
    y: np.ndarray             # int64 [N], истинные id L3 (только train)
    rows: np.ndarray          # int64 [N], исходные индексы строк входа
    n_classes: int

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])


def competitor_indices(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Детерминированный конкурент ``argmax_{c ≠ y_i} z_{i,c}``.

    Истинный класс маскируется ``-inf``; при равенстве логитов выбирается
    наименьший индекс класса (первое вхождение максимума, гарантия numpy).
    Работает одинаково для правильных и ошибочных предсказаний.
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if z.ndim != 2 or z.shape[0] != len(y):
        raise AttributionError(f"logits/y не согласованы: {z.shape} и {len(y)}")
    masked = z.copy()
    masked[np.arange(len(y)), y] = -np.inf
    return np.argmax(masked, axis=1).astype(np.int64)


@torch.no_grad()
def _l3_logits(model: torch.nn.Module, batch: SparseBatch, device: torch.device) -> torch.Tensor:
    return model(batch.to(device))[2]


def compute_margin_attributions(
    model: torch.nn.Module,
    X: sp.csr_matrix,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> MarginAttributions:
    """Gradient × Input по margin для обученной модели (D03, D04).

    Модель переводится в ``eval()``; веса не обновляются (градиент —
    ``torch.autograd.grad`` только по значениям ненулевых CSR-элементов,
    параметры в ``.grad`` не накапливаются). Конкурент фиксируется по логитам
    из отдельного прохода под ``no_grad`` и в градиентном шаге не
    дифференцируется. ``X`` — train-матрица полного TF-IDF (CSR); ``y`` —
    целочисленные id L3 только размеченного train (значения < 0 отклоняются).
    Возвращает разреженные вклады той же структуры, что и ``X``.
    """
    if not sp.isspmatrix_csr(X):
        X = sp.csr_matrix(X)
    y = np.asarray(y, dtype=np.int64)
    if X.shape[0] != len(y):
        raise AttributionError(f"X/y не согласованы: X{X.shape}, len(y)={len(y)}")
    if X.shape[1] == 0:
        raise AttributionError("Пустое пространство признаков: атрибуция невозможна")
    if len(y) == 0:
        raise AttributionError("Пустая выборка: атрибуция невозможна")
    if np.any(y < 0):
        raise AttributionError("В y есть неразмеченные строки (-1): атрибуция выполняется только по размеченному train")
    n_classes = int(model.n_classes[2]) if hasattr(model, "n_classes") else int(y.max()) + 1
    if n_classes < 2:
        raise AttributionError("Для margin нужно не менее двух классов L3")
    if batch_size <= 0:
        raise AttributionError(f"batch_size: целое > 0, получено {batch_size!r}")

    was_training = model.training
    model.eval()
    device = torch.device(device)
    n = X.shape[0]
    att_data = np.empty(X.nnz, dtype=np.float64)
    margins = np.empty(n, dtype=np.float64)
    competitor = np.empty(n, dtype=np.int64)
    correct = np.empty(n, dtype=bool)
    empty = np.diff(X.indptr) == 0
    pos = 0
    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            Xb = X[start:end]
            batch = csr_to_batch(Xb)
            logits = _l3_logits(model, batch, device).float().cpu().numpy()
            yb = y[start:end]
            competitor[start:end] = competitor_indices(logits, yb)
            margins[start:end] = logits[np.arange(len(yb)), yb] - logits[
                np.arange(len(yb)), competitor[start:end]
            ]
            correct[start:end] = np.argmax(logits, axis=1) == yb
            nnz_b = Xb.nnz
            if nnz_b:
                weights = torch.from_numpy(Xb.data.astype(np.float32, copy=True)).to(device)
                weights.requires_grad_(True)
                _, _, lg3 = model(SparseBatch(batch.indices, batch.offsets, weights))
                arange = torch.arange(len(yb), device=device)
                r = torch.from_numpy(competitor[start:end]).to(device)
                yy = torch.from_numpy(yb).to(device)
                margin_sum = (lg3[arange, yy] - lg3[arange, r]).sum()
                grad = torch.autograd.grad(margin_sum, weights)[0]
                att_data[pos : pos + nnz_b] = (weights.detach() * grad).double().cpu().numpy()
                pos += nnz_b
    finally:
        if was_training:
            model.train()
    if pos != X.nnz:
        raise AttributionError(f"Внутренняя ошибка атрибуции: собрано {pos} вкладов при nnz={X.nnz}")
    values = sp.csr_matrix(
        (att_data, X.indices.astype(np.int64), X.indptr.astype(np.int64)), shape=X.shape
    )
    return MarginAttributions(
        values=values,
        margins=margins,
        competitor=competitor,
        correct=correct,
        empty=empty,
        y=y,
        rows=np.arange(n, dtype=np.int64),
        n_classes=n_classes,
    )


def select_attributed_rows(n_rows: int, cfg: AttributionConfig, seed: int) -> np.ndarray:
    """Воспроизводимый отбор строк train для атрибуции (D02).

    ``max_rows=None`` (или >= n_rows) — все строки в исходном порядке;
    ``sampling="head"`` — первые ``max_rows``; ``"random"`` — детерминированный
    выбор без возвращения через ``np.random.default_rng(seed)``, отсортированный
    по возрастанию индекса (порядок батчей стабилен).
    """
    if n_rows <= 0:
        raise AttributionError("Пустой train: отбирать нечего")
    if cfg.max_rows is None or cfg.max_rows >= n_rows:
        return np.arange(n_rows, dtype=np.int64)
    if cfg.sampling == SAMPLING_HEAD:
        return np.arange(cfg.max_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=cfg.max_rows, replace=False)).astype(np.int64)


def _positive_part(values: sp.csr_matrix) -> sp.csr_matrix:
    """Разреженная положительная часть вкладов: max(0, a), нули удалены."""
    out = values.copy()
    out.data = np.maximum(out.data, 0.0)
    out.eliminate_zeros()
    return out


def _class_segments(y_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Границы сегментов классов в отсортированном y: (classes, start, end)."""
    classes, starts, counts = np.unique(y_sorted, return_index=True, return_counts=True)
    return classes.astype(np.int64), starts.astype(np.int64), (starts + counts).astype(np.int64)


def build_gxi_ratings(
    att: MarginAttributions,
    n_char: int,
    ratings_dir: str,
    l3_labels: list[str] | None = None,
) -> tuple[dict, dict]:
    """Class-aware рейтинги Gradient × Input на диске (D05, D06).

    Считает ``score(c,f) = mean_{i∈c} max(0, a_{i,f})`` по атрибутированным
    (непустым) строкам train; пишет полные рейтинги в ``ratings_dir`` в формате
    S03 (``kkt-selection-ratings@2``): глобальный рейтинг ветки
    (средний положительный вклад по всем атрибутированным строкам), списки
    положительных кандидатов категорий (``-score`` ↓, равные — по индексу
    признака), смещения и длины. Формирует диагностику: число/сумма/доля
    положительных, отрицательных (сумма модулей) и нулевых вкладов, доли
    correct/incorrect, пустые строки, статистика margin, число использованных
    строк по L3 и покрытие классов.

    Память: последовательная агрегация — плотный вектор одной ветки одного
    класса за раз (``O(F_branch)``, при ~1000 L3 и 400k признаках ≈ 3 МБ на
    класс, не ``C×F``); исходные вклады не сохраняются построчно.
    """
    if not 0 < n_char < att.values.shape[1]:
        raise AttributionError(f"n_char: граница веток в (0, {att.values.shape[1]}), получено {n_char!r}")
    if att.n_rows == 0:
        raise AttributionError("Нет строк для агрегации рейтингов")
    os.makedirs(ratings_dir, exist_ok=True)

    attributed = ~att.empty
    n_empty = int(att.empty.sum())
    A_pos = _positive_part(att.values[attributed])
    y_att = att.y[attributed]
    margins_att = att.margins[attributed]
    correct_att = att.correct[attributed]
    n_attributed = int(attributed.sum())
    n_total_values = int(att.values.nnz)

    order = np.argsort(y_att, kind="stable")
    A_s = A_pos[order]
    y_s = y_att[order]
    classes, starts, ends = _class_segments(y_s)
    n_classes = len(classes)
    n_branch = {"char": int(n_char), "word": int(att.values.shape[1] - n_char)}

    # Диагностика по знакам вкладов (все значения, включая пустые строки — там
    # вкладов нет): положительные идут в рейтинг, отрицательные/нулевые — только
    # в диагностику, нулевые не выдумываются.
    data = att.values.data
    pos_mask = data > 0
    neg_mask = data < 0
    n_positive = int(pos_mask.sum())
    n_negative = int(neg_mask.sum())
    n_zero = int(n_total_values - n_positive - n_negative)
    contributions = {
        "n_values_total": n_total_values,
        "positive": {
            "count": n_positive,
            "share": n_positive / n_total_values if n_total_values else None,
            "sum": float(data[pos_mask].sum()) if n_positive else 0.0,
        },
        "negative": {
            "count": n_negative,
            "share": n_negative / n_total_values if n_total_values else None,
            "abs_sum": float(np.abs(data[neg_mask]).sum()) if n_negative else 0.0,
        },
        "zero": {
            "count": n_zero,
            "share": n_zero / n_total_values if n_total_values else None,
        },
    }

    manifest: dict[str, Any] = {
        "format": RATINGS_FORMAT_VERSION,
        "method": METHOD,
        "branches": {},
        "attribution": {
            "target": TARGET_NAME,
            "formula": FORMULA,
            "n_attributed_rows": n_attributed,
            "n_empty_rows": n_empty,
            "n_classes": n_classes,
        },
    }
    per_class: list[dict[str, Any]] = []
    branch_diag: dict[str, Any] = {}
    for branch, offset in (("char", 0), ("word", n_char)):
        width = n_branch[branch]
        g_global = np.zeros(width, dtype=np.float64)
        counts = np.zeros(n_classes, dtype=np.int64)
        idx_parts: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []
        for ci in range(n_classes):
            seg = A_s[starts[ci] : ends[ci]]
            n_c = int(seg.shape[0])
            counts[ci] = n_c
            if n_c == 0:
                continue
            sums = np.asarray(seg.sum(axis=0), dtype=np.float64).ravel()[offset : offset + width]
            scores = sums / n_c
            g_global += sums
            pos_idx = np.flatnonzero(scores > 0)
            order_c = np.lexsort((pos_idx, -scores[pos_idx]))
            ranked = pos_idx[order_c]
            idx_parts.append(ranked.astype(np.int64))
            score_parts.append(scores[ranked].astype(np.float64))
        g_global /= max(n_attributed, 1)
        idx_arr = np.concatenate(idx_parts) if idx_parts else np.empty(0, dtype=np.int64)
        score_arr = np.concatenate(score_parts) if score_parts else np.empty(0, dtype=np.float64)
        idx_path = os.path.join(ratings_dir, f"{branch}_rank_idx.npy")
        score_path = os.path.join(ratings_dir, f"{branch}_rank_score.npy")
        counts_path = os.path.join(ratings_dir, f"{branch}_rank_counts.npy")
        classes_path = os.path.join(ratings_dir, f"{branch}_classes.npy")
        global_path = os.path.join(ratings_dir, f"{branch}_global.npy")
        idx_arr.tofile(idx_path)
        score_arr.tofile(score_path)
        counts.tofile(counts_path)
        np.asarray(classes, dtype=np.int64).tofile(classes_path)
        g_global.tofile(global_path)
        manifest["branches"][branch] = {
            "n_features": width,
            "global_scores": f"{branch}_global.npy",
            "classes": f"{branch}_classes.npy",
            "rank_indices": f"{branch}_rank_idx.npy",
            "rank_scores": f"{branch}_rank_score.npy",
            "rank_counts": f"{branch}_rank_counts.npy",
        }
        branch_diag[branch] = {"n_positive_candidates": int(len(idx_arr))}

    # Статистика по классам (по сегментам отсортированной матрицы вкладов).
    att_vals_sorted = att.values[attributed][order]
    correct_sorted = correct_att[order]
    margins_sorted = margins_att[order]
    cls_stats: list[dict[str, Any]] = []
    for ci in range(n_classes):
        cls = int(classes[ci])
        n_c = int(ends[ci] - starts[ci])
        seg_vals = att_vals_sorted[starts[ci] : ends[ci]]
        d = seg_vals.data
        pmask = d > 0
        nmask = d < 0
        row_correct = correct_sorted[starts[ci] : ends[ci]]
        row_margins = margins_sorted[starts[ci] : ends[ci]]
        entry: dict[str, Any] = {
            "class_id": cls,
            "label": str(l3_labels[cls]) if l3_labels is not None and cls < len(l3_labels) else None,
            "n_attributed": n_c,
            "n_correct": int(row_correct.sum()),
            "correct_fraction": float(row_correct.mean()) if n_c else None,
            "margin_mean": float(row_margins.mean()) if n_c else None,
            "positive": {
                "count": int(pmask.sum()),
                "sum": float(d[pmask].sum()) if pmask.any() else 0.0,
            },
            "negative": {
                "count": int(nmask.sum()),
                "abs_sum": float(np.abs(d[nmask]).sum()) if nmask.any() else 0.0,
            },
            "zero": {"count": int((~pmask & ~nmask).sum())},
        }
        n_pos_c = entry["positive"]["count"]
        if n_pos_c == 0:
            entry["reason"] = "no positive Gradient×Input contributions for the class"
        cls_stats.append(entry)

    with open(os.path.join(ratings_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    report: dict[str, Any] = {
        "target": TARGET_NAME,
        "formula": FORMULA,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "versions": environment_versions(),
        "n_rows_total": int(att.n_rows),
        "n_attributed_rows": n_attributed,
        "n_empty_rows": n_empty,
        "n_classes": n_classes,
        "n_classes_with_positive": int(sum(1 for e in cls_stats if e["positive"]["count"] > 0)),
        "n_classes_without_positive": int(sum(1 for e in cls_stats if e["positive"]["count"] == 0)),
        "correct": {
            "count": int(correct_att.sum()),
            "fraction": float(correct_att.mean()) if n_attributed else None,
        },
        "incorrect": {
            "count": int((~correct_att).sum()),
            "fraction": float((~correct_att).mean()) if n_attributed else None,
        },
        "margins": {
            "mean": float(margins_att.mean()) if n_attributed else None,
            "std": float(margins_att.std()) if n_attributed else None,
            "min": float(margins_att.min()) if n_attributed else None,
            "max": float(margins_att.max()) if n_attributed else None,
        },
        "contributions": contributions,
        "branches": branch_diag,
        "per_class": cls_stats,
        "notes": [
            "Атрибуция и агрегирование выполнены только по размеченному train; метки val/test не использовались.",
            "N_c — число фактически атрибутированных (непустых) train-примеров класса; пустые TF-IDF-векторы "
            "в знаменатель не входят, их число зафиксировано отдельно.",
            "Положительные вклады: max(0, a) > 0; отрицательные/нулевые в рейтинги не входят и здесь учтены "
            "как диагностика (число, сумма модулей, доля). Нулевые вклады не выдумываются.",
            "Полные рейтинги — формат kkt-selection-ratings@2 (те же соглашения, что S03): повторное "
            "применение бюджета без пересчёта атрибуции; оценки float64, ties по индексу признака.",
            "Значения вкладов для каждой строки не сохраняются: только рейтинги и сводная диагностика.",
        ],
    }
    log.info(
        "G×I-рейтинги: строк %d (пустых %d), классов %d (с положительными %d), "
        "вкладов +%d/−%d/0:%d",
        n_attributed, n_empty, n_classes, report["n_classes_with_positive"],
        n_positive, n_negative, n_zero,
    )
    return manifest, report


def select_features_gradient_x_input(
    ratings_dir: str,
    cfg: SelectionConfig,
    n_char: int,
    n_total: int,
    l3_labels: list[str] | None = None,
) -> SelectionResult:
    """Class-aware Top-K отбор из сохранённых G×I-рейтингов (D05, D06).

    Рейтинги строятся один раз (:func:`build_gxi_ratings`) и повторно
    используются для каждого бюджета K: здесь выполняется только равноправный
    циклический round-robin категорий с дедупликацией в пределах квот веток
    char/word, детерминированный добор из глобального рейтинга положительных
    атрибуций и (при нехватке положительных кандидатов) добор остальных
    признаков в стабильном порядке с маркировкой ``remainder_fill``. Индексы
    результата — полно-пространные (word со смещением ``n_char``).
    """
    if cfg.k is None or cfg.k <= 0:
        raise SelectionError(f"selection.k: обязательное целое > 0, получено {cfg.k!r}")
    if not os.path.isdir(ratings_dir):
        raise SelectionError(f"Каталог G×I-рейтингов {ratings_dir!r} не найден")
    meta_path = os.path.join(ratings_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise SelectionError(f"В каталоге рейтингов {ratings_dir!r} нет meta.json")
    with open(meta_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("method") != METHOD:
        raise SelectionError(
            f"Рейтинги в {ratings_dir!r} относятся к методу {manifest.get('method')!r}, "
            f"ожидается {METHOD!r}"
        )
    k_char, k_word = branch_budgets(cfg.k, cfg.char_share, n_char, n_total - n_char)
    report: dict[str, Any] = {
        "method": METHOD,
        "k_requested": int(cfg.k),
        "char_share": float(cfg.char_share),
        "seed": int(cfg.seed),
        "target": TARGET_NAME,
        "formula": FORMULA,
        "ratings_dir": os.path.abspath(ratings_dir),
        "tie_policy": "competitor ties -> lowest class index; score ties -> lowest feature index; no RNG",
        "budgets": {"k_char": int(k_char), "k_word": int(k_word)},
        "n_features_total": int(n_total),
        "positive_filter": "max(0, a) > 0 only; negative/zero contribute to diagnostics only",
    }
    branches: dict[str, BranchSelection] = {}
    for branch, k_b, offset in (("char", k_char, 0), ("word", k_word, n_char)):
        width = n_char if branch == "char" else n_total - n_char
        entry = (manifest.get("branches") or {}).get(branch) or {}
        classes = np.fromfile(os.path.join(ratings_dir, entry["classes"]), dtype=np.int64)
        counts = np.fromfile(os.path.join(ratings_dir, entry["rank_counts"]), dtype=np.int64)
        global_scores = np.fromfile(os.path.join(ratings_dir, entry["global_scores"]), dtype=np.float64)
        sel, rep = roundrobin_select_from_rank_files(
            os.path.join(ratings_dir, entry["rank_indices"]),
            os.path.join(ratings_dir, entry["rank_scores"]),
            counts, classes, k_b, width, global_scores,
            positive_only_fallback=True,
        )
        branches[branch] = sel
        report[branch] = {
            "n_selected": len(sel.indices),
            "n_classes": rep["n_classes"],
            "n_positive_per_class": rep["n_positive_per_class"],
            "n_fallback": rep["n_fallback"],
            "n_remainder_fill": rep["n_remainder_fill"],
        }
    sel_w = branches["word"]
    word = BranchSelection(
        indices=sel_w.indices + n_char if len(sel_w.indices) else sel_w.indices,
        scores=sel_w.scores,
        origin=sel_w.origin,
    )
    classes_labels = None
    if l3_labels is not None:
        present = np.fromfile(
            os.path.join(ratings_dir, manifest["branches"]["char"]["classes"]), dtype=np.int64
        )
        classes_labels = [str(l3_labels[int(i)]) for i in present]
    log.info(
        "G×I-отбор: k=%d → char %d/%d (fallback %d, remainder %d), word %d/%d (fallback %d, remainder %d)",
        cfg.k, len(branches["char"].indices), k_char,
        report["char"]["n_fallback"], report["char"]["n_remainder_fill"],
        len(word.indices), k_word,
        report["word"]["n_fallback"], report["word"]["n_remainder_fill"],
    )
    return SelectionResult(
        method=METHOD,
        k_requested=int(cfg.k),
        char_share=float(cfg.char_share),
        seed=int(cfg.seed),
        char=branches["char"],
        word=word,
        classes=classes_labels,
        report=report,
        ratings_dir=ratings_dir,
        ratings_manifest=manifest,
    )


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def checkpoint_fingerprint(art_dir: str) -> dict[str, Any]:
    """Отпечаток checkpoint MLP: веса (model.pt) и конфиг (config.json)."""
    fp: dict[str, Any] = {"artifact_dir": os.path.abspath(art_dir)}
    for name in ("model.pt", "config.json"):
        path = os.path.join(art_dir, name)
        if os.path.isfile(path):
            fp[name] = {"sha256": _sha256_file(path), "bytes": os.path.getsize(path)}
    maps_path = os.path.join(art_dir, "label_maps.json")
    if os.path.isfile(maps_path):
        fp["label_maps_sha256"] = _sha256_file(maps_path)
    return fp


def verify_attribution_checkpoint(
    art_dir: str,
    model: torch.nn.Module,
    artifact_config: dict,
    n_features_full: int,
    label_map_sha256: str | None,
    data_cfg: dict,
    seed: int,
) -> dict[str, Any]:
    """Проверяет идентичность checkpoint'а атрибуции и эксперимента (D02, D08).

    Контролируются: входная размерность модели равна полному TF-IDF;
    fingerprint label maps checkpoint'а совпадает с экспериментным
    ``label_map_sha256`` (одинаковый состав L3 ⇒ согласованное кодирование);
    data-конфиг (путь/сплит/min_count/max_rows/seed) checkpoint'а совпадает с
    экспериментным — вместе с общим seed это фиксирует те же splits. Любое
    несовпадение — ``AttributionError`` (явный блокер, молчаливая подмена
    источника недопустима). Возвращает словарь выполненных проверок.
    """
    checks: dict[str, Any] = {}
    n_feat_model = int(getattr(model, "n_features", -1))
    checks["n_features_matches_full_tfidf"] = {
        "expected": int(n_features_full), "actual": n_feat_model,
        "ok": n_feat_model == int(n_features_full),
    }
    # config.json артефакта модели оборачивает RunConfig в {"experiment_run",
    # "config": <RunConfig>}: принимаем оба расположения.
    inner = artifact_config.get("config") if isinstance(artifact_config, dict) else None
    if not isinstance(inner, dict):
        inner = artifact_config if isinstance(artifact_config, dict) else {}
    cfg_data = inner.get("data")
    checks["data_config_matches_experiment"] = {
        "expected": data_cfg, "actual": cfg_data, "ok": cfg_data == data_cfg,
    }
    cfg_seed = inner.get("seed")
    checks["seed_matches_experiment"] = {
        "expected": int(seed), "actual": cfg_seed, "ok": cfg_seed == int(seed),
    }
    maps_sha = checkpoint_fingerprint(art_dir).get("label_maps_sha256")
    checks["label_maps_match_experiment"] = {
        "expected": label_map_sha256, "actual": maps_sha,
        "ok": bool(label_map_sha256) and maps_sha == label_map_sha256,
    }
    failed = {k: v for k, v in checks.items() if not v["ok"]}
    if failed:
        details = "; ".join(
            f"{k}: ожидалось {str(v['expected'])[:80]!r}, получено {str(v['actual'])[:80]!r}"
            for k, v in failed.items()
        )
        raise AttributionError(
            f"Checkpoint атрибуции {art_dir!r} не совместим с экспериментом ({details}): "
            "идентичность full TF-IDF / label maps / splits не подтверждена, атрибуция остановлена"
        )
    return checks


def save_attribution_report(path: str, report: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    return path
"""Единый эксперимент S04 (Gradient × Input) и агрегированный отчёт (CLI).

Тот же оркестратор, что и S03 (``kkt_clf.experiment.run_experiment``), с
планом S04: контрольный ``none`` (полный TF-IDF), ``global`` Chi² × K,
``class_aware`` Chi² × K и ``gradient_x_input`` × K — 10 конфигураций на общих
splits, label maps, seed и протоколе raw top-1 L3 без отказа. Для G×I один раз
готовятся разделяемые attribution MLP (обучить из конфигурации или загрузить
проверенный checkpoint) и полные рейтинги; стоимость этих этапов учитывается
один раз (run_id=shared_attribution). Selection — только по размеченному
train; метки test не используются. Smoke- и обычный режим — один код, разные
параметры.

Пример полного локального прогона на ODS (при наличии ресурсов):

    python scripts/run_s04_experiment.py run \\
        --output outputs/S04 \\
        --full-tfidf outputs/S02/tfidf_1m \\
        --corpus data/corpus.pqt \\
        --data data/ods_alpha.pqt \\
        --budgets 50000,100000,200000 --char-share 0.75 \\
        --preset ods_light --force

    python scripts/run_s04_experiment.py report --output outputs/S04

Повторное использование обученной attribution MLP в новом эксперименте:

    python scripts/run_s04_experiment.py run ... \\
        --attribution-checkpoint outputs/S04/shared/attribution_model

Регенерация отчёта из сохранённых результатов без повторного обучения:

    python scripts/run_s04_experiment.py report --output outputs/S04
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kkt_clf import (  # noqa: E402
    AttributionConfig,
    DataConfig,
    ExperimentConfig,
    SelectionPlanConfig,
    build_config,
    build_report,
    planned_runs,
    run_experiment,
    setup_logging,
)


def _split(value: str) -> tuple[float, float, float]:
    parts = [float(x) for x in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("split: три доли через запятую, напр. 0.8,0.1,0.1")
    return tuple(parts)  # type: ignore[return-value]


def _csv_ints(value: str) -> list[int]:
    try:
        return [int(x) for x in value.split(",") if x.strip()]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"список целых через запятую: {e}")


def _csv_strs(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _build_cfg(a: argparse.Namespace) -> ExperimentConfig:
    rc = build_config(a.preset, json.loads(a.override))
    return ExperimentConfig(
        output_dir=a.output,
        seed=a.seed,
        full_tfidf_path=a.full_tfidf,
        corpus_path=a.corpus,
        data=DataConfig(
            path=a.data,
            max_rows=a.data_max_rows,
            min_count=a.min_count,
            split=a.split,
            n_duplicate_examples=rc.data.n_duplicate_examples,
        ),
        features_chunk_rows=a.features_chunk_rows,
        model=rc.model,
        loss=rc.loss,
        train=rc.train,
        selection=SelectionPlanConfig(
            methods=a.methods,
            budgets=a.budgets,
            char_share=a.char_share,
            include_full=not a.no_full,
        ),
        attribution=AttributionConfig(
            checkpoint_path=a.attribution_checkpoint,
            max_rows=a.attribution_max_rows,
            sampling=a.attribution_sampling,
            batch_size=a.attribution_batch_size,
        ),
        noisy_support=a.noisy_support,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="выполнить плановые прогоны и построить отчёт")
    pr.add_argument("--output", required=True, help="корневой каталог результатов эксперимента")
    pr.add_argument("--full-tfidf", required=True, help="каталог автономного артефакта полного TF-IDF (S01/S02)")
    pr.add_argument("--corpus", required=True, help="тот же корпус, что в метаданных полного TF-IDF (для повторного fit)")
    pr.add_argument("--data", required=True, help="размеченный Parquet (name, L1, L2, L3)")
    pr.add_argument("--data-max-rows", type=int, default=None, help="ограничение размеченных строк (страт. сэмпл)")
    pr.add_argument("--min-count", type=int, default=10, help="data.min_count (по умолчанию 10)")
    pr.add_argument("--split", type=_split, default=(0.8, 0.1, 0.1), help="доли train,val,test (по умолчанию 0.8,0.1,0.1)")
    pr.add_argument("--methods", type=_csv_strs, default=["global", "class_aware", "gradient_x_input"],
                    help="методы отбора через запятую (по умолчанию global,class_aware,gradient_x_input)")
    pr.add_argument("--budgets", type=_csv_ints, default=[50_000, 100_000, 200_000],
                    help="бюджеты K через запятую (по умолчанию 50000,100000,200000)")
    pr.add_argument("--char-share", type=float, default=0.75, help="квота char в бюджете (по умолчанию 0.75)")
    pr.add_argument("--no-full", action="store_true", help="не включать контрольный прогон none (полный TF-IDF)")
    pr.add_argument("--attribution-checkpoint", default=None,
                    help="загрузить checkpoint attribution MLP вместо обучения (проверяется идентичность)")
    pr.add_argument("--attribution-max-rows", type=int, default=None,
                    help="ограничение числа train-строк для атрибуции (None — все)")
    pr.add_argument("--attribution-sampling", choices=["head", "random"], default="head",
                    help="способ отбора строк train для атрибуции (по умолчанию head)")
    pr.add_argument("--attribution-batch-size", type=int, default=4096,
                    help="батч ненулевых CSR-значений для Gradient × Input")
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--features-chunk-rows", type=int, default=50_000,
                    help="чанкование transform внутри эксперимента")
    pr.add_argument("--noisy-support", type=int, default=2,
                    help="порог support на validation для пометки неустойчивых L3 в HTML")
    pr.add_argument("--preset", default="ods_light", help="источник MLP/loss/train-конфига: default | ods_light")
    pr.add_argument("--override", default="{}",
                    help='JSON с переопределениями model/train/loss, напр. \'{"train": {"epochs": 5}}\'')
    pr.add_argument("--only", type=_csv_strs, default=None,
                    help="выполнить только эти run_id (остальные плановые — skipped)")
    pr.add_argument("--force", action="store_true", help="разрешить запись в непустой каталог --output")

    prep = sub.add_parser("report", help="перестроить отчёт из сохранённых результатов без обучения")
    prep.add_argument("--output", required=True, help="корневой каталог результатов эксперимента")

    a = p.parse_args()
    setup_logging()
    if a.cmd == "run":
        cfg = _build_cfg(a)
        plan = planned_runs(cfg)
        print("Плановые прогоны:", ", ".join(p.run_id for p in plan))
        res = run_experiment(cfg, only=a.only, force=a.force)
        paths = build_report(cfg.output_dir)
        print(json.dumps({
            "output_dir": os.path.abspath(cfg.output_dir),
            "statuses": res["statuses"],
            "report": {k: os.path.abspath(v) if k != "report_dir" else os.path.abspath(v)
                       for k, v in paths.items()},
        }, ensure_ascii=False, indent=2))
    else:
        paths = build_report(a.output)
        print(json.dumps({"report": {k: os.path.abspath(v) for k, v in paths.items()}},
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
