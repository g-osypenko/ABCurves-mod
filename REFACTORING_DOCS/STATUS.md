# ABCurves Closed-Loop Refactor — Status

> Живой статус рефакторинга: что сделано, чем доказано, что осталось. Полный
> архитектурный разбор и решения — в [DESIGN_NOTE.md](DESIGN_NOTE.md); регламент —
> в [../rules.md](../rules.md). Обновлено: 2026-10-11.

## Цель (кратко)
Превратить ABCurves в замкнутый Receding-Horizon генератор траекторий для слежения
за движущейся целью, убрав нестабильный `_continuous` FSM и сохранив человечность
движения. Цели владельца: **стабильность, без нервозности, точность, убрать патологии**
исходных доков.

## Принятые решения
- **Подход Б (RHC + ProDMP + persistent GRU-рендерер)** — вместо эвристик вокруг TCN.
- **B2′** — forcing-веса (макро-кривизна) запекаются из `motor.onnx` **офлайн** в
  библиотеку, а не гоняются живьём. `motor.onnx` сохранён как bake-ассет; `_continuous`
  оставлен offline-only, рантайм от него изолирован. (см. [DESIGN_NOTE §3](DESIGN_NOTE.md))
- Координаты: **native mouse counts** (как у рендерера).

---

## ✅ Сделано

| Компонент | Файл | Проверено |
| :--- | :--- | :--- |
| Аналитическая C¹-скорость | [abcurves/prodmp.py](../abcurves/prodmp.py) `velocity_at` | совпадает с конечной разностью; при t→0 = `ydot_b` |
| RHC-контроллер | [abcurves/closed_loop.py](../abcurves/closed_loop.py) `ClosedLoopProDMPController` | сходимость, гладкость, deadzone, idle, zero-start |
| Библиотека forcing-весов | [abcurves/forcing_library.py](../abcurves/forcing_library.py) + `models/continuous/forcing_lib.npz` (512 форм) | orient round-trip; добавляет кривизну, сохраняя сходимость |
| Офлайн-бейк B2′ | [tools/bake_forcing_library.py](../tools/bake_forcing_library.py) | 512 форм из 140 прогонов |
| Метрики траектории | [abcurves/trajectory_metrics.py](../abcurves/trajectory_metrics.py) | восстанавливает β≈1/3 на синтетике |
| Headless-оценщик | [tools/evaluate_trajectory.py](../tools/evaluate_trajectory.py) | B1 vs B2′ по сценариям |
| Texture19-оценщик | [tools/texture_eval.py](../tools/texture_eval.py) | метод репозитория (floors) |
| Живой стенд | [tools/live_harness/](../tools/live_harness/) | авто-цель, 8-мс такт, pause, slow-mo, zoom, живой счёт |
| Прунинг (изоляция) | — | рантайм не импортирует `_continuous`/onnxruntime (проверено) |
| **P1: адверсариальный сюит** | [tests/test_adversarial.py](../tests/test_adversarial.py) | см. ниже |
| **P2: безопасность forcing + адаптивный рычаг** | `closed_loop.py` `_effective_gain` | см. ниже |

### Доказательства (измерено)

**Симптомы мертвы (P1):**
| Патология | Триггер | Старое поведение | Измерено сейчас |
| :--- | :--- | :--- | :--- |
| #3 усиление шума | джиттер цели σ=1.5 @8мс | Δx/Δt → ±188 c/s фантома | пик скорости ratio **1.00**, excess rms ~22 c/s |
| #1 флаттер/звон | резкий разворот +X→−X | осцилляция ~31 Гц | overshoot **0.0**, разворотов скорости **0** |
| #2 паралич+саккада | пауза→дальний перезахват | фриз 100–200 мс + overshoot | латентность **5 мс**, overshoot **0.0**, пик ограничен |
| v=0 сингулярность | старт из покоя | взрыв кривизны | старт мягче пика, не глобальный максимум jerk |

**Человечность (измерено):**
- Закон двух третей (step): β≈**0.39–0.45**, R²≈**0.72–0.88** (человек ~0.33).
- Texture19 (метод floors, 256 строк): within-human **0.17**, between-human **0.50**,
  **B1 = 0.88 (OUTSIDE)**, **B2′ = 0.42 (INSIDE человеческого диапазона)**. Макро-кривизна
  улучшает и текстуру. Контроль метода: наш within-floor 0.17 ≈ опубликованный same-session 0.15.

**P2 — forcing безопасен:**
- **Safety-потолок (всегда вкл):** `forcing_gain_max=0.6`. Любой запрошенный gain (до 2.5)
  сходится (resid **0.00**, 20 seeds). Корень нестабильности — положительная обратная связь по
  скорости (forcing раздувает переносимую `ydot_b`), а не разовый большой изгиб; потолок её гасит.
- **Adaptive back-off (дистанционный) реализован, но `forcing_adaptive=False` по умолчанию:**
  измерено, что он снижает круговой лаг (67→~50), но регрессирует текстуру reach
  (0.42→0.54), т.к. срезает кривизну у цели. Круговой лаг — не патология, а reach-текстура —
  главная метрика; поэтому рычаг припаркован до P4.
- Текстура при дефолте сохранена: **B2′ = 0.421 INSIDE**.

**Тесты:** 23 passing — `test_adversarial`, `test_prodmp_velocity`, `test_forcing_library`,
`test_trajectory_metrics`, `test_closed_loop`.

---

## ⏳ Осталось (приоритет)

- **P3 — Закрыть топ-отличия текстуры.** Крупнейшие разрывы Texture19: `bp_30_80`,
  `sign_flip_rate`, `bp_80_200`, `hf_power`, `iri_p95`. Рычаги: rest-микрошум, update-такт,
  профиль доводки. Успех: дистанция → к 0.17–0.25.
- **P4 — Тюнинг рабочей точки.** `horizon_ms, commit_ms, deadzone(r_inner,λ,ε), update_ms,
  gain` сейчас плейсхолдеры. Свип против метрик. Сюда же — тюнинг **adaptive back-off**
  (`forcing_adaptive`, реализован в P2, по умолчанию off): найти точку, где он улучшает
  pursuit, не ломая reach-текстуру. Успех: чистая доводка в deadzone без «подползания»/удара о стену.
- **P5 — Решающий тест человечности (нужны данные).** Texture19 — только текстура. Для
  cold/unknown-person судьи на Trajectory14/Full49 нужен человеческий **трекинг-датасет**
  ([ABCurves-Capture](https://github.com/optima-manent/ABCurves-Capture)). Единственный пункт
  с внешней зависимостью.

---

## Как запускать
```bash
# тесты (общий conftest тянет torch, поэтому --noconftest)
python -m pytest tests/test_adversarial.py tests/test_prodmp_velocity.py \
  tests/test_forcing_library.py tests/test_trajectory_metrics.py tests/test_closed_loop.py \
  --noconftest

python tools/live_harness/server.py                 # живой стенд → печатает URL
python tools/evaluate_trajectory.py --scenario all  # headless метрики B1 vs B2′
python tools/texture_eval.py --rows 256             # Texture19 vs человек
python tools/bake_forcing_library.py --runs 140     # пере-запечь библиотеку B2′
```

## Окружение
- Рабочий Python — 3.9 (Anaconda: numpy/scipy/numba есть). Контроллер/стенд/метрики
  работают на 3.9; bake ставит шим `hashlib.file_digest` (API 3.11+). Py3.11 есть, но без
  scipy/numba. onnxruntime не нужен (нативный numba-бэкенд).
