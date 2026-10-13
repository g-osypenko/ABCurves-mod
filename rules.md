# ABCurves Refactoring & Architectural Rules

> **Назначение**: Технический регламент для рефакторинга **ABCurves** под замкнутый контур генерации траекторий (Closed-Loop Trajectory Generation / Receding Horizon Control).
> 
> **Главный принцип**: Мы **НЕ** переписываем математику с нуля! Мы проводим **хирургическую модификацию и интеграцию** уже существующих модулей библиотеки, сохраняя проверенные биомеханические модели траекторий и отсекая нестабильный FSM. Задача сфокусирована исключительно на генерации траекторий без привязки к компонентам обнаружения или ввода.
>
> ⚠️ **ПОПРАВКИ ПОСЛЕ СВЕРКИ С КОДОМ** (см. [`REFACTORING_DOCS/DESIGN_NOTE.md`](REFACTORING_DOCS/DESIGN_NOTE.md)): несколько имён файлов/моделей в исходной редакции ресёрча оказались неточными. Парадигма (Подход Б) верна, но перед кодингом учитывай: (1) `FastPlanner` живёт в `abcurves/pipeline.py`, а `abcurves/fast_planner.py` — это `NumbaPlannerForward`; (2) `get_analytical_state_at` реализуется в `abcurves/prodmp.py` (+ контроллер), не в `fast_planner.py`; (3) `motor.onnx` грузится только из `_continuous`; решение **B2′** — он KEEP как офлайн bake-ассет (запекает библиотеку forcing-весов), а `_continuous` удаляется из рантайма. Пункты ниже помечены маркером `[corr]` там, где текст скорректирован.

---

## 1. Что УЖЕ работает из коробки (НЕ ПЕРЕПИСЫВАТЬ, А СОХРАНЯТЬ)

Многие биомеханические свойства уже заложены в ядро ABCurves. Задача — **не сломать их**:

1. **Закон двух третей ($v \propto \kappa^{-1/3}$) и профили скорости (minimum-jerk)**:
   - *Где реализовано*: В `abcurves/prodmp.py`. `[corr]` Привязка к `motor.onnx` условна — она зависит от решения по источнику forcing-весов `w` ([DESIGN_NOTE §3](REFACTORING_DOCS/DESIGN_NOTE.md)); minimum-jerk-профиль даёт сам ProDMP и без `motor.onnx`.
   - *Директива*: `[corr]` Макроплан строить на уровне `ProDMP.generate_deltas(..., goal_mode='target', goal=target)` (параметр `goal_mode='target'` живёт в `prodmp.py`/`planner.py`, а **не** в `FastPlanner.plan()`). Никаких самодельных генераторов траекторий писать не нужно.
2. **Физиологический тремор (8–12 Гц) и логнормальная микроструктура**:
   - *Где реализовано*: В высокочастотной рекуррентной GRU-сети `models/renderer_global_h80.bin` и C-runtime (`abcurves/portable_renderer.py`).
   - *Директива*: Не сбрасывать скрытое состояние GRU ($h_{80}$) между шагами планирования! Держать единый персистентный объект `PortableRendererStream`.
3. **Нативный C-runtime (1000 Гц)**:
   - *Где реализовано*: `runtime/c/` и `PortableRendererStream.step()`.
   - *Директива*: Использовать напрямую. Время шага уже оптимизировано в C.

---

## 2. В чём РЕАЛЬНО состоит задача рефакторинга (Хирургические правки)

Существующий `ContinuousPlanner` нестабилен (флаттер переключения головок TCN + зависания на 100–200 мс в `hazard.onnx`/`brake.onnx`). Мы заменяем его на прозрачную парадигму **Receding Horizon Control (RHC)**:

### А. Прунинг (удаление лишнего):
- `[corr]` **При выбранном B2′ `abcurves/_continuous/` НЕ удаляется физически** — bake-скрипт (`tools/bake_forcing_library.py`) зависит от него, а integrity-гейт бандла требует на диске весь фрозен-набор (включая `hazard/brake/choice`). Смысл прунинга реализован иначе: **рантайм замкнутого контура изолирован от `_continuous`** — новый `ClosedLoopProDMPController` импортирует только `prodmp` + `portable_renderer` + `forcing_library` и ни разу не трогает `_continuous`/onnxruntime (проверено). `_continuous` и классификаторы остаются исключительно офлайн bake-ассетами.
- Исключить использование классификаторов `hazard.onnx`, `brake.onnx`, `choice.onnx` **в рантайме** (выполнено: рантайм их не грузит).
- Оставить `motor.onnx` и `renderer_global_h80.bin`.

### Б. Модификация `abcurves/prodmp.py` + новый контроллер (~30-50 строк кода): `[corr]`
- `[corr]` **Было**: «модификация `abcurves/fast_planner.py`». **Стало**: `fast_planner.py` — это `NumbaPlannerForward` (компиляция TCN), его не трогаем. Аналитическая позиция уже есть в `ProDMP.evaluate(...)`; добавляем аналитическую **скорость** `ProDMP.velocity_at(...)` (производная `evaluate` через существующие `_critically_damped`/`_basis_at`).
- Метод `get_analytical_state_at(delta_t)` — обёртка **на контроллере**, которая хранит текущий план `(w_g, ydot_b, tau, y_b)` и возвращает аналитическую позицию и скорость кривой в момент времени $\Delta t$ через `evaluate`/`velocity_at`. Точная формула — в [DESIGN_NOTE §4](REFACTORING_DOCS/DESIGN_NOTE.md).
- **Зачем**: Это обеспечивает $C^1$-сплайсинг (бесшовную склейку планов). Скорость на стыке сегментов траектории берется из аналитической производной ProDMP, а **НЕ** из шумной разницы внешних координат ($\Delta x / \Delta t$).

### В. Создание класса `ClosedLoopProDMPController`:
Один компактный класс-оркестратор траектории, связывающий макропланирование и высокочастотный рендеринг:
1. **Динамическое обновление цели `on_target_update(target_offset, dt, apply_deadzone=True)`**:
   - **Демпфирование в deadzone (динамически управляемо из Python)**: НЕ прибито намертво в коде! Доступно по требованию: через параметры метода (`apply_deadzone: bool = True/False`), методы настройки (`set_deadzone(radius, lambda_param)`), либо отдельный метод демпфирования. При вызове плавно масштабирует ошибку логистической сигмоидой $\omega(r) = \frac{1}{1 + e^{-\lambda (r - r_{inner} - \epsilon)}}$, предотвращая резкие скачки ускорения на границе зоны.
   - Извлекает текущую скорость из предыдущего плана через `get_analytical_state_at(dt)`.
   - Генерирует новый макроплан через `FastPlanner.plan(goal_mode='target', ...)` на ближайшие 10–15 мс.
   - Складывает полученные дельты траектории в буфер для рендерера.
2. **Высокочастотный шаг генерации `step()` (1000 Гц)**:
   - Забирает дельту из буфера и делает `renderer_stream.step(delta)`.
   - Если буфер пуст (цель в deadzone или пауза в обновлениях) — делает **холостой ход** `renderer_stream.step((0, 0))`. Рендерер сам гасит остаточный импульс (Sigma-Delta модулятор) и продолжает генерировать микротремор.
   - Возвращает результирующие смещения траектории $(\Delta x, \Delta y)$.
3. **Метод `reset()`**:
   - Очищает буфер очереди, плавно обнуляет аналитическую скорость, генерирует 10–15 мс синтетического микрошума покоя ($\sigma \approx 0.1$) для исключения деления на ноль при старте из неподвижности ($v=0$).

---

## 3. Таблица файлов (Audit & Pruning)

> `[corr]` Таблица пересобрана после сверки с кодом. Полный разбор — [DESIGN_NOTE §1–2](REFACTORING_DOCS/DESIGN_NOTE.md).

| Файл / Каталог | Действие | Пояснение |
| :--- | :--- | :--- |
| `abcurves/prodmp.py` | **MODIFY** `[corr]` | Было KEEP. Добавить аналитический `velocity_at(...)` (носитель $C^1$). Остальное не трогать. |
| `abcurves/fast_planner.py` | **KEEP** `[corr]` | Было MODIFY. Это `NumbaPlannerForward` (компиляция TCN), ничего не добавляем. |
| `abcurves/pipeline.py` (`FastPlanner`/`StaticPlanner`) | **KEEP** `[corr]` | Настоящий `FastPlanner`. Prefix-based static planner; в RHC напрямую не вызывается. |
| `abcurves/portable_renderer.py` | **KEEP** | Обвязка C-runtime, не трогать. |
| `runtime/c/*` | **KEEP** | 1000 Гц движок, не трогать. |
| `abcurves/<new>_closed_loop.py` | **CREATE** `[corr]` | Новый `ClosedLoopProDMPController` + обёртка `get_analytical_state_at`. |
| `abcurves/_continuous/*` | **RETAIN (offline-only)** `[corr]` | Было DELETE. При B2′ нужен bake-скрипту. В рантайме не используется (контроллер от него изолирован). |
| `models/continuous/hazard.onnx, brake.onnx, choice.onnx` | **RETAIN (offline-only)** `[corr]` | Было DELETE. В рантайме не грузятся; нужны integrity-гейту бандла при офлайн-бейке. |
| `models/continuous/choice_split.onnx, events.onnx` | **RETAIN (offline-only)** `[corr]` | Грузятся только из `_continuous`; в рантайме не используются. |
| `models/continuous/forcing_lib.npz` | **NEW (bake output)** `[corr]` | Запечённая библиотека forcing-весов B2′, грузится рантаймом через `ForcingLibrary`. |
| `models/continuous/motor.onnx` | **KEEP (bake-ассет)** `[corr]` | Решение B2′ ([DESIGN_NOTE §3](REFACTORING_DOCS/DESIGN_NOTE.md)): мотор прогоняется **офлайн один раз** для запекания библиотеки forcing-весов `models/continuous/forcing_lib.npz`. В рантайме не грузится; `_continuous` удаляется. Это сохраняет обученную макро-кривизну без конечно-разностного шума живого контура. |
| `models/renderer_global_h80.bin` | **KEEP** | Веса GRU микрорендерера. |
| `models/renderer_adapter.bin` | **KEEP** `[corr]` | Provenance/export-source; кодом не грузится. Не трогать. |

---

## 4. Порядок реализации (Изолированные шаги)

1. **Шаг 1: Скелет контроллера и прунинг**: объявить сигнатуры `ClosedLoopProDMPController`, отключить `_continuous`.
2. **Шаг 2: Аналитическое состояние в FastPlanner**: реализовать `get_analytical_state_at` в `fast_planner.py` и протестировать непрерывность производной.
3. **Шаг 3: Логика `on_target_update`**: демпфирование deadzone сигмоидой, передача аналитической скорости, пуш в буфер.
4. **Шаг 4: Логика `step`**: такт 1000 Гц, холостые шаги (idle step), возврат рассчитанных смещений.
