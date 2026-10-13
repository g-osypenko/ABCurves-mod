# Дизайн-нота: сверка ресёрча с кодом + нерешённые решения

> **Назначение**: Этот документ выверяет архитектурный ресёрч (`abcurves refactoring part 1.md`,
> `rules.md`) с **фактическим кодом** репозитория. Ресёрч делался deep-research моделью по
> снимку GitHub и в нескольких местах угадал API неправильно. Здесь зафиксированы расхождения
> (со ссылками `файл:строка`), исправлена таблица прунинга и выделены **решения, требующие
> подписи владельца** до написания кода.
>
> **Статус вывода ресёрча**: Подход Б (Receding Horizon на ProDMP + persistent GRU-рендерер,
> удаление нестабильного `_continuous`) — **архитектурно верен и подтверждён**. Исправления
> ниже касаются конкретных имён файлов/классов и одного замолчанного вопроса дизайна, а не
> самой парадигмы.

---

## 1. Сверка фактов (что реально есть в коде)

| Утверждение ресёрча / rules.md | Факт в коде | Вывод |
| :--- | :--- | :--- |
| `FastPlanner` живёт в `abcurves/fast_planner.py` | `fast_planner.py` содержит `NumbaPlannerForward` — Numba-компиляцию forward-прохода TCN ([fast_planner.py:104](../abcurves/fast_planner.py#L104)). `FastPlanner` — в [pipeline.py:115](../abcurves/pipeline.py#L115), экспортируется как `StaticPlanner` ([__init__.py:19](../abcurves/__init__.py#L19)). | **Имя файла неверно.** |
| «Добавить `get_analytical_state_at` в `fast_planner.py`» | Аналитическая позиция уже есть: `ProDMP.evaluate(t, w_g, tau, ydot_b, ...)` для произвольного `t` ([prodmp.py:240](../abcurves/prodmp.py#L240)). Производные компонент доступны из `_critically_damped` (`yd1,yd2`) и `_basis_at` (`dphi`). | **Метод принадлежит `prodmp.py` + контроллеру, не `fast_planner.py`.** ~30-50 строк. |
| «KEEP `motor.onnx` — FastPlanner берёт из него ковариации `w`» | `FastPlanner` грузит `planner_seedN.pt` (PyTorch TCN-чекпойнт), НЕ `motor.onnx` ([pipeline.py:119-124](../abcurves/pipeline.py#L119), [model_store.py:140](../abcurves/model_store.py#L140)). `motor.onnx` грузится **только** из `_continuous/backend.py:16`. | **Противоречие.** Удаляя `_continuous`, мы оставляем `motor.onnx` без единого загрузчика. См. §3 (решение). |
| `goal_mode='target'` динамически меняет цель | Подтверждено: `generate`/`evaluate` переопределяют последнюю строку `w_g` целью ([prodmp.py:278](../abcurves/prodmp.py#L278)). | **Верно.** |
| `PortableRendererStream.step(delta)` + persistent GRU `h80` | Подтверждено: [portable_renderer.py:356-383](../abcurves/portable_renderer.py#L356); поток создаётся через `begin_stream(event_seed=...)` ([:337](../abcurves/portable_renderer.py#L337)). | **Верно.** |
| (не упомянуто) | `renderer_adapter.bin` не используется кодом — только в [models/README.md](../models/README.md) как export-source. | Provenance-артефакт, вне рефакторинга. |
| (не упомянуто) | В `models/continuous/` есть ещё `events.onnx`, `choice_split.onnx` — тоже грузятся только из `_continuous/backend.py`. | Удаляются вместе с `_continuous`. |
| (не упомянуто) | `FastPlanner.plan()` жёстко prefix-based: требует A→B-префикс 160 сэмплов, horizon 1000, seam-contract `edge@0.8`, и внутри зовёт `generate_deltas(..., goal_mode="learned")` ([planner.py:214](../abcurves/planner.py#L214), [:564](../abcurves/planner.py#L564)). Он **не пробрасывает** `goal_mode='target'` наружу. | Для RHC `FastPlanner.plan()` **не подходит как есть** — макроплан строим на уровне `ProDMP`. См. §3. |

---

## 2. Исправленная таблица прунинга

| Локация | Статус | Исправление к rules.md |
| :--- | :--- | :--- |
| `abcurves/prodmp.py` | **MODIFY** | Было KEEP. Добавить аналитический `velocity_at(t, ...)` (производная `evaluate`) — носитель $C^1$. |
| `abcurves/fast_planner.py` | **KEEP** | Было «MODIFY — add get_analytical_state_at». Это `NumbaPlannerForward`; ничего сюда не добавляем. |
| `abcurves/pipeline.py` (`FastPlanner`) | **KEEP** | Prefix-based static planner. В RHC напрямую не используется (решение §3 определит, нужен ли как источник весов). |
| `abcurves/portable_renderer.py` | **KEEP** | Без изменений. |
| `runtime/c/*` | **KEEP** | Без изменений. |
| `abcurves/<новый>_closed_loop.py` | **CREATE** | Новый `ClosedLoopProDMPController` (оркестратор + `get_analytical_state_at` как обёртка над `prodmp.velocity_at`). |
| `abcurves/_continuous/*` | **DELETE** | Весь нестабильный FSM-стек. |
| `models/continuous/hazard.onnx, brake.onnx, choice.onnx, choice_split.onnx, events.onnx` | **DELETE** | Все грузятся только из `_continuous`. |
| `models/continuous/motor.onnx` | **DECISION PENDING** | Было KEEP «для FastPlanner» (неверно). Судьба зависит от §3. |
| `models/renderer_global_h80.bin` | **KEEP** | Веса GRU-рендерера. Используется и continuous-, и static-пайплайном. |
| `models/renderer_adapter.bin` | **KEEP** | Provenance; кодом не грузится. Не трогаем. |

---

## 3. ГЛАВНОЕ НЕРЕШЁННОЕ РЕШЕНИЕ: откуда брать forcing-веса `w` на каждом сегменте?

Ресёрч пишет «FastPlanner использует motor.onnx для `w`» — это неверно (см. §1). Поэтому
источник 20-базисных forcing-весов `w` для каждого шага перепланирования **не определён**.
ProDMP строит макроплан как:

```
y(t) = xi1(t)·y_b + xi2(t)·ẏ_b  +  H(t) · w
       └──── граничный отклик ────┘   └─ forcing ─┘
```

При `goal_mode='target'` последняя строка `w` подменяется целью. Остаётся вопрос: чем
заполнить **остальные** строки `w` (форму ускорения)?

### Варианты

- **B1 — Чистый homing (нулевой/фиксированный forcing).**
  `w = 0` (кроме goal-строки) → критически-демпфированный minimum-jerk-ход к цели.
  *Плюсы*: детерминизм, максимальная робастность, `_continuous` и `motor.onnx` удаляются
  полностью. *Минусы*: макро-путь почти прямой (низкая кривизна) → two-thirds power law
  работает слабо на макро-масштабе (микро-текстуру всё равно даёт рендерер).

- **B2 (буквальный) — живой `motor.onnx` в цикле.**
  Мотор крутится онлайн каждые 32 мс. *Минусы (критичные)*: `motor.onnx` **не изолирован** —
  `Assets` ([backend.py:19-44](../abcurves/_continuous/backend.py#L19)) требует фрозен-бандл
  (включая hazard/brake/choice/events + `weights.npz`), входы `coarse/fine/dynamics` собирает
  `kernels.motor_features` (конечно-разностные фичи = **патология #3**), а декод идёт через
  собственный 16-головый стохастический prodmp-базис (риск вернуть **патологию #1**). Тащим
  backend+kernels+neural_native+native. Противоречит цели «стабильность/без нервозности».

- **B2′ — запечённая офлайн библиотека forcing-весов. ✅ ВЫБРАНО.**
  Прогнать `motor.onnx` офлайн **один раз** по репрезентативным условиям → снять библиотеку
  наборов `w`, заморозить в небольшой `.npz`. Рантайм сэмплит/блендит из неё (**sticky на
  время одного захвата**, не каждые 32 мс), применяет `goal_mode='target'` + аналитическую
  $C^1$-скорость. *Плюсы*: сохраняет обученную макро-кривизну (цель владельца «рендерера
  мало»), но **без** kernels/FSM/конечных разностей в живом контуре; `_continuous` удаляется
  из рантайма; `motor.onnx` — одноразовый инструмент запекания. Обратимо: откат к B1 (нулевой
  forcing) или расширение библиотеки.

- **B1 — чистый homing (нулевой forcing).** Запасной вариант/фолбэк внутри B2′.

### Решение (подписано владельцем, 2026-10-11)

**B2′.** Владелец задал цели (стабильность, убрать нервозность, точность, убрать патологии
исходных доков) и делегировал технический выбор. Буквальный B2 возвращает патологии #1 и #3 —
отвергнут. B2′ — синтез: обученная макро-текстура мотора без живого шумного контура.

> **Следствие для прунинга (РЕАЛИЗОВАНО):** `motor.onnx` = **KEEP** (офлайн bake-ассет);
> `_continuous/*` и классификаторы = **RETAIN, но offline-only** — физически НЕ удаляются
> (bake-скрипту они нужны, а integrity-гейт бандла требует весь набор на диске), однако рантайм
> замкнутого контура от них **изолирован** (проверено: `ClosedLoopProDMPController` не импортирует
> `_continuous`/onnxruntime). Библиотека весов запечена в `models/continuous/forcing_lib.npz`
> (512 форм). Калибровка: `forcing_gain≈0.45` даёт заметную кривизну при точной сходимости;
> `>~0.6` начинает бороться с homing. Sticky-форма на захват — рычаг стабильности.

---

## 4. Спецификация `get_analytical_state_at` (носитель $C^1$)

Склейка $C^1$ требует: позиция и скорость нового сегмента в точке стыка берутся как
**аналитическое состояние предыдущего плана** в момент $\Delta t$, а не из шумной разницы
координат.

**4.1. Новый метод `ProDMP.velocity_at(t, w_g, *, tau, ydot_b, y_b=None, t_b=0.0)`** —
зеркало `evaluate`, но для $\dot y$:

```
ẏ(t) = xi1'(t)·y_b + xi2'(t)·ẏ_b + H'(t) · w_g
```

Все слагаемые выводятся из уже существующих примитивов (новой математики нет):
- `xi1'(t) = (yd2b·ẏ1 − yd1b·ẏ2) / det`, `xi2'(t) = (y1b·ẏ2 − y2b·ẏ1) / det`,
  где `ẏ1, ẏ2` — это `yd1, yd2` из `_critically_damped` ([prodmp.py:465](../abcurves/prodmp.py#L465));
- `H'(t) = dphi(t)/tau + xi3'(t)·phi_b + xi4'(t)·phidot_b`, `xi3'=−xi1'`, `xi4'=−xi2'`;
  `dphi` — второй выход `_basis_at` ([prodmp.py:162](../abcurves/prodmp.py#L162)).

**4.2. Обёртка на контроллере.** `ClosedLoopProDMPController` хранит текущий план
`(w_g, ydot_b, tau, y_b)` и `time_since_plan`. Тогда:

```
get_analytical_state_at(Δt):
    pos = ProDMP.evaluate([Δt], w_g, tau=tau, ydot_b=ydot_b, y_b=y_b)
    vel = ProDMP.velocity_at([Δt], w_g, tau=tau, ydot_b=ydot_b, y_b=y_b)
    return pos[0], vel[0]
```

На каждом `on_target_update`: `ẏ_b^{(k+1)} = get_analytical_state_at(time_since_plan).vel` →
строгая непрерывность скорости на стыке. Пространственные невязки гасит Sigma-Delta рендерера.

---

## 5. Открытые решения — статус

1. ~~**§3 — источник forcing-весов `w`**~~ → **РЕШЕНО/РЕАЛИЗОВАНО: B2′**. `motor.onnx` = KEEP (bake-ассет), `_continuous` retained offline-only, рантайм изолирован.
2. ~~**Стенд**~~ → **РЕАЛИЗОВАН** как zero-dependency stdlib-сервер + canvas (`tools/live_harness/`), а не websocket. Живой B1↔B2′ A/B через ползунок `gain`. Baseline старого `ContinuousPlanner` при желании снимается отдельно (он требует Python 3.11+ из-за `hashlib.file_digest`).
3. ~~**Единицы координат**~~ → **РЕШЕНО**: контроллер и стенд работают в **native mouse counts** (x-right/y-up, как у рендерера) — без лишнего преобразования. Приложение само мапит свои цели в эту систему.
4. ~~**Deadzone-параметры**~~ → **ЗАДАНЫ дефолты** в `ClosedLoopConfig`: `r_inner=2.0`, `deadzone_lambda=1.0`, `deadzone_eps=0.0`. Это плейсхолдеры для тюнинга на живом стенде (сигмоида мягкая — при необходимости поднять `lambda`).

## 6. Что реализовано (2026-10-11)

| Компонент | Файл | Статус |
| :--- | :--- | :--- |
| Аналитическая скорость $C^1$ | `abcurves/prodmp.py::velocity_at` | ✅ + тест |
| Контроллер RHC | `abcurves/closed_loop.py::ClosedLoopProDMPController` | ✅ + тесты (сходимость, гладкость, deadzone, idle) |
| Библиотека forcing-весов | `abcurves/forcing_library.py` + `models/continuous/forcing_lib.npz` | ✅ + тест |
| Офлайн-бейк (B2′) | `tools/bake_forcing_library.py` | ✅ (512 форм из 140 прогонов) |
| Живой стенд | `tools/live_harness/{server.py,index.html}` | ✅ |

Запуск стенда: `python tools/live_harness/server.py` → открыть напечатанный URL.
Тесты: `python -m pytest tests/test_prodmp_velocity.py tests/test_forcing_library.py tests/test_closed_loop.py --noconftest` (11 passed; `--noconftest`, т.к. общий conftest тянет torch).

---

*Сверено с кодом на коммите `main` (см. git log). Все ссылки `файл:строка` указывают на
состояние на момент написания — перед реализацией перепроверить, что сигнатуры не сдвинулись.*
