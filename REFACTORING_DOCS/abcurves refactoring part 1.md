
!!INITIAL PROMPT!!

# Deep Research Task: Architectural Audit & Refactoring of ABCurves for Closed-Loop Trajectory Generation & Dynamic Target Tracking

!!!! researchable github repository link - https://github.com/optima-manent/ABCurves !!!!

## CONTEXT & PRIMARY OBJECTIVE

You are provided with an attached GitHub repository containing the **ABCurves** codebase (a continuous and static movement generation library based on ProDMP and a high-frequency GRU C-runtime renderer).

We are developing a **closed-loop trajectory generation controller** (Receding Horizon Control) for dynamic target tracking:

- **Task Definition:** The system tracks dynamic target displacements $(\Delta x, \Delta y)$ in a closed loop. The controller smoothly generates trajectories towards moving targets with dynamic target updates.
- **Dynamic Updates:** Dynamic target displacement error $(\Delta x, \Delta y)$ is provided at runtime with dynamic time delta $\Delta t$.
- **Trajectory Generation:** High-frequency 1000 Hz generation step producing continuous trajectory displacements.
- **Core Dilemma:** The library's `ContinuousPlanner` (`ContinuousPipeline`) currently exhibits severe behavioral instability under dynamic closed-loop tracking ("spastic, jerky, ataxia-like motion"). On the other hand, the vanilla `StaticPlanner` was designed exclusively for open-loop movement completion and cannot natively initialize from a standstill ($v=0$) or perform closed-loop target tracking.

### STRICT INSTRUCTION ON OUTPUT FORMAT:

- **DO NOT** generate monolithic, copy-paste raw Python code blocks. Deep research models write poor immediate production code.
- **DO** write like an elite systems architect and biomechanics engineer: provide clear technical explanations, architectural diagrams/dataflows, algorithmic pseudocode for method interfaces, mathematical justifications, a concrete file-by-file refactoring checklist, and an authoritative engineering verdict.
- **DO NOT** invent speculative external math from scratch. The solution MUST primarily leverage, refactor, and glue together existing modules and weights already present in this ABCurves repository.

---

## 1. IN-DEPTH AUDIT OF THE ATTACHED REPOSITORY

Inspect the attached ABCurves repository structure, specifically:

- `abcurves/prodmp.py` (The analytical Probabilistic Dynamic Movement Primitives formulation).
- `abcurves/fast_planner.py` & `abcurves/planner.py` (The Numba-compiled 16-head TCN static planner).
- `abcurves/portable_renderer.py` & `runtime/c/` (The C-runtime GRU model `renderer_global_h80.bin` running at 1000 Hz with persistent `PortableRendererStream`).
- `abcurves/_continuous/` (`planner.py`, `kernels.py`, `runtime.py`, and ONNX models `motor.onnx`, `choice.onnx`, `hazard.onnx`, `brake.onnx`).

Explain the exact mathematical and state-machine root causes of the "spastic motion" in `ContinuousPlanner`:

1. Why does sampling `motor_rng.categorical(probabilities)` across the 16 heads every 32 ms produce high-frequency lateral oscillation (31 Hz curvature flipping)?
2. How does `hazard.onnx` trigger premature Poisson braking (`Mode 1`), lock the system into `Mode 2 (Hold)` for 100–200 ms, and cause explosive overshoot when resuming with `previous_valid = False`?
3. How does finite-difference velocity calculation in `kernels.py` amplify $\pm 1-2$ px target coordinate jitter into false lead compensations?

---

## 2. THE CORE COMPARATIVE STUDY: OPTION 1 VS OPTION 2

Evaluate and compare two concrete refactoring strategies that utilize existing repository modules:

### Approach A: Surgical Stabilization of `ContinuousPlanner`

How to minimally prune and stabilize the existing `ContinuousPipeline` without throwing away the trained continuous models:

- How to eliminate 32 ms head flapping (e.g., sticky head, argmax lock per tracking engagement).
- How to surgically bypass or neuter the Poisson `hazard.onnx` / `brake.onnx` FSM to prevent freezing while maintaining natural terminal deceleration.
- How to filter target velocity inputs to insulate the model from coordinate noise.
- What are the residual risks and architectural limitations of this approach?

### Approach B: Closed-Loop Tracking via `StaticPlanner` / `ProDMP` + Global Renderer (The "Fork" Approach)

How to adapt `StaticPlanner` (`FastPlanner` / raw ProDMP) and the persistent 1000 Hz C-stream into a closed-loop receding-horizon controller with dynamic target updates:

- How to initialize ProDMP from zero velocity ($v=0$) using a synthetic stationary prefix without math singularities ($\kappa \to \infty$ or heading blow-up).
- How `goal_mode = 'target'` in `prodmp.py` can be used to ensure mathematical convergence to target error while retaining the learned 20-basis acceleration profile.
- How to feed the resulting smooth trajectory into the persistent C-runtime `PortableRendererStream` without resetting internal GRU hidden states between replanning steps.

### Verdict:

Deliver a definitive engineering verdict comparing Approach A vs Approach B in terms of implementation friction, runtime latency, robustness against dynamic target maneuvers, and kinematic cleanliness.

---

## 3. FOUR ESSENTIAL KINEMATIC & INTEGRATION CHALLENGES

Provide concrete mathematical and algorithmic answers for:

### 1. Zero-Start Kinematics ($v=0$)

In `StaticPlanner` / `ProDMP`, standard inference expects a moving prefix.

- Exactly what prefix tensor (shape, values, micro-noise vs zeros) should be passed to `FastPlanner` when starting from rest?
- How do ProDMP boundary conditions ($\xi_1 y_0 + \xi_2 \dot{y}_0$) mathematically guarantee smooth, zero-jerk acceleration from $\dot{y}_0 = [0, 0]$?

### 2. Inter-Segment Velocity Continuity ($C^1$ Splicing across Replanning Steps)

When target updates arrive with new coordinates:

- How should boundary velocity $\dot{y}_b$ be assigned at step $k+1$ to prevent step velocity discontinuities (infinite jerk spikes) at the seam between trajectory segments?
- Should $\dot{y}_b$ be derived from recent trajectory output, the prior ProDMP plan derivative, or the W5 filtered state?

### 3. Biomechanically Realistic Deadzone Deceleration

We have an inner deadzone ($r \le r_{inner}$) where the trajectory must settle without jitter or abrupt stopping:

- How should deceleration be governed inside and near the deadzone?
- Compare two mechanisms: (a) decaying/dampening the target error vector fed into ProDMP vs (b) zeroing deltas sent to `PortableRendererStream` and letting the GRU/Sigma-Delta accumulator naturally absorb residual momentum.
- How to prevent "wall slamming" (hitting the deadzone edge like a rigid mechanical barrier)?

### 4. Biomechanical Invariant Preservation & Kinematic Classification

To ensure the generated motion is statistically consistent with human motor control:

- Detail how this refactored pipeline preserves core human motor invariants:
  - The Two-Thirds Power Law ($v(t) = \alpha \cdot \kappa(t)^{-\beta}$ connecting speed and curvature).
  - Bell-shaped, log-normal velocity and jerk envelopes.
  - 8–12 Hz physiological micro-tremor and physical sensor quantization.
- Specifically analyze the risk of synthetic/robotic artifacts appearing at:
  - Inter-segment splicing seams.
  - Deadzone boundary transitions.
- How does routing through `PortableRendererStream` (`renderer_global_h80.bin`) provide natural micro-tremor and authentic high-frequency movement dynamics?

---

## 4. ACTIONABLE REFACTORING CHECKLIST & ARCHITECTURAL SPECIFICATION

Provide a step-by-step developer's guide to refactoring the ABCurves repo:

1. **Repository Pruning Table:** A strict file-by-file audit listing:
   - What files/models to keep unchanged.
   - What files to modify (and what specific functions to alter).
   - What files/models can be safely removed or bypassed.
2. **Controller Class Interface & Algorithmic Flow:**
   - Define the class interface (e.g. `ClosedLoopProDMPController` or refactored `ContinuousPipeline`).
   - Detail the internal state variables (persistent renderer stream, sticky head index, current boundary velocity, anchor position).
   - Provide algorithmic step-by-step logic for:
     - `on_target_update(target_offset, dt)`: handling error, head locking, ProDMP generation, and pushing to the 1000 Hz buffer.
     - `step()`: high-frequency 1000 Hz step of the trajectory generator.
     - `reset()`: controller reset logic.

final output and answer should be in russian



!!ANSWER!!

Архитектурный аудит и рефакторинг библиотеки ABCurves для систем замкнутого контура генерации траекторий и динамического сопровождения целей

Кодовая база ABCurves представляет собой программный комплекс, предназначенный для генерации человекоподобных траекторий движения в режиме реального времени. Базируясь на математическом аппарате вероятностных динамических двигательных примитивов (Probabilistic Dynamic Movement Primitives, ProDMP) и высокочастотном рекуррентном рендерере (GRU C-runtime), эта библиотека продемонстрировала высокую эффективность в задачах выполнения движений в разомкнутом контуре (open-loop). Тем не менее, текущая инженерная задача требует создания контроллера динамического слежения за целями в замкнутом контуре (Receding Horizon Control).

Задача сфокусирована исключительно на генерации траекторий без привязки к компонентам обнаружения или ввода. При динамическом поступлении вектора целевого смещения $(\Delta x, \Delta y)$ контроллер должен плавно перестраивать траекторию. В таких условиях существующий в библиотеке модуль ContinuousPlanner демонстрирует критическую поведенческую нестабильность, выражающуюся в спастическом, атактическом характере движения. С другой стороны, базовый StaticPlanner не способен инициализировать движение из состояния покоя ($v=0$) и не поддерживает динамическое обновление целеуказания в замкнутом контуре.

Настоящий отчет представляет собой архитектурный и математический аудит репозитория, сравнительный анализ стратегий рефакторинга и спецификацию для построения робастного контроллера замкнутого цикла, сохраняющего фундаментальные биомеханические инварианты движения.

1. Глубокий аудит архитектуры репозитория ABCurves
Инспекция структуры репозитория выявила четкое концептуальное и физическое разделение между аналитическим макропланировщиком (ProDMP, реализованным в abcurves/fast_planner.py и abcurves/prodmp.py), микрорендерером текстуры движения (abcurves/portable_renderer.py и директория runtime/c/) и экспериментальным контуром непрерывного планирования (abcurves/_continuous/). Для понимания причин кинематической деградации системы в условиях замкнутого контура необходимо детально разобрать математическую природу дефектов модуля ContinuousPlanner.

1.1. Математическая природа высокочастотной латеральной осцилляции (31 Гц флаттер)
В архитектуре ContinuousPlanner используется 16-головая сеть временных сверток (Temporal Convolutional Network, TCN), которая на каждом шаге логического вывода предсказывает распределение вероятностей по 16 топологическим кластерам траекторий (головкам). Выбор конкретного двигательного примитива осуществляется посредством стохастического сэмплирования из категориального распределения вероятностей. Эта операция вызывается с фиксированным интервалом в 32 мс.

В условиях открытого контура, на которых обучалась сеть, распределение вероятностей плавно эволюционирует на протяжении сотен миллисекунд. Однако в замкнутом контуре, где координаты цели динамически обновляются, вектор целевой ошибки ($\Delta \vec{e}$) подвергается постоянным высокочастотным возмущениям из-за маневров динамической цели. Стохастическое сэмплирование каждые 32 мс (что соответствует частоте обновления $\approx 31.25$ Гц) в таких условиях приводит к явлению, известному в теории нелинейного управления как переключение мод (mode chattering).

Если распределение вероятностей между двумя соседними топологическими головками, отвечающими за противоположные фазовые состояния (например, плавная дуга влево и резкий компенсаторный рывок вправо), сближается до значений порядка $P(A) \approx 0.45$ и $P(B) \approx 0.55$, стохастическое сэмплирование начинает чередовать эти базисы. Каждая головка TCN задает собственное уникальное пространство параметров: индивидуальную матрицу ковариации, базовую кривизну и профиль ускорения. Смена базиса с частотой 31.25 Гц вызывает мгновенную инверсию знака второй производной (ускорения) в латеральной плоскости. В траектории это проявляется как высокочастотный тремор и кинематическая атаксия. Это фундаментально разрушает базовые биомеханические принципы, в частности закон двух третей (Two-Thirds Power Law), который постулирует плавную взаимосвязь между кривизной траектории и скоростью движения.

1.2. Автомат состояний Пуассоновского торможения и кинематический паралич
Архитектура непрерывного контура жестко опирается на конечный автомат (Finite State Machine, FSM), управляемый двумя нейросетевыми классификаторами: hazard.onnx (классификатор аномалий и резких смен курса) и brake.onnx (контроллер профиля замедления). Эта подсистема была спроектирована для имитации времени реакции на внезапные возмущения.

В замкнутом динамическом контуре эта имитация приводит к отказам управления. Модель hazard.onnx была обучена на наборах данных, где резкие изменения градиента цели являются редкими аномалиями. При отслеживании динамически маневрирующей цели изменение траектории является нормой, но hazard.onnx классифицирует эти изменения как кинематические аномалии. Это мгновенно переводит конечный автомат в режим экстренного торможения (Mode 1), а затем в режим удержания (Mode 2).

Для обеспечения биологической правдоподобности выход из режима удержания смоделирован как пуассоновский процесс, что вводит стохастическую задержку длительностью 100–200 мс. В течение этой «слепой зоны» система игнорирует обновления координат цели, а вектор ошибки ($\Delta x, \Delta y$) продолжает накапливаться по мере движения цели. Когда FSM наконец разрешает возобновление движения (путем сброса флага внутренней валидации), планировщик получает колоссальную ошибку. Пытаясь компенсировать ее, ContinuousPlanner генерирует баллистическую саккаду с максимальным ускорением. Эта саккада неизбежно перерегулирует (overshoot) цель, генерирует новую аномалию градиента, и цикл ложного торможения запускается вновь, создавая эффект непрерывных рывков и зависаний.

1.3. Усиление шума координат при конечно-разностном дифференцировании
Третьей критической уязвимостью текущей архитектуры является метод вычисления скорости цели в модуле abcurves/_continuous/kernels.py. В текущей реализации используется наивное дифференцирование методом конечных разностей: $v = \frac{\Delta x}{\Delta t}$.

При динамическом обновлении координат цели входной сигнал может содержать координатный шум с амплитудой порядка $\pm 1-2$ единиц. При дискретном поступлении данных с малым шагом по времени $\Delta t$ изменение координаты из-за шума транслируется в значительный ложный вектор скорости:

$$
v_{noise} = \frac{\Delta x_{noise}}{\Delta t} \gg 0
$$

Поскольку ContinuousPlanner использует вычисленную скорость для предиктивного упреждения (lead compensation), этот высокочастотный шум напрямую подается в латентное пространство планировщика. Незначительный координатный шум превращается в фантомные скачки скорости сотен единиц в секунду. Полное отсутствие низкочастотных фильтров или фильтров состояния делает контур обратной связи математически нестабильным и подверженным резонансному усилению шума.

2. Сравнительное исследование архитектурных парадигм
Для создания робастного генератора траекторий необходимо проанализировать две диаметрально противоположные стратегии рефакторинга, опирающиеся на уже существующие компоненты репозитория ABCurves.

Подход А: Хирургическая стабилизация непрерывного планировщика
Данная стратегия предполагает сохранение текущего модуля ContinuousPipeline и всех связанных с ним ONNX-моделей (motor, choice, hazard, brake), но с внедрением алгоритмических ограничений для подавления выявленных артефактов.

Устранение 31 Гц флаттера головок потребует перехода от стохастического сэмплирования к детерминированному выбору через операцию нахождения максимума (argmax) по распределению вероятностей. Для предотвращения переключения топологий в процессе одного непрерывного движения потребуется внедрить логику «липкого индекса» (sticky head). Этот механизм будет фиксировать выбранную головку TCN при инициализации движения и удерживать ее до тех пор, пока скалярное произведение текущего вектора цели и исходного вектора на момент захвата не станет отрицательным (что свидетельствует о смене полусферы движения).

Для нейтрализации эффекта пуассоновского паралича потребуется вмешательство в Python-обвязку для принудительного обхода (bypassing) автомата состояний hazard.onnx. Придется искусственно удерживать систему в состоянии активного движения, перехватывать сигналы торможения и транслировать их в плавное экспоненциальное затухание целевого вектора. Кроме того, для kernels.py потребуется развернуть альфа-бета фильтр или фильтр Калмана с моделью постоянного ускорения (Constant Acceleration Model) для изоляции дифференциатора от высокочастотного координатного шума.

Фундаментальным ограничением и критическим риском этого подхода является проблема сдвига распределения (distribution shift). Нейросетевые веса motor.onnx были обучены на конкретных наборах данных в условиях открытого контура. Любая попытка искусственно зафиксировать индекс головки, отбросить стохастичность или обойти автомат состояний приведет к тому, что тензоры, подаваемые на рекуррентные слои, выйдут за пределы обучающего многообразия. Генератор начнет выдавать траектории из неисследованных областей латентного пространства, что приведет к математическим сингулярностям: зависаниям векторов, бесконечным значениям кривизны или генерации NaN в выходных потоках.

Подход Б: Контроль с удаляющимся горизонтом через StaticPlanner и ProDMP (Подход «Форк»)
Вторая парадигма требует радикального отказа от черного ящика TCN и перехода к математически доказуемой модели управления с удаляющимся горизонтом (Receding Horizon Control). В основе этого подхода лежит использование исходных классов ProDMP (FastPlanner) в качестве макро-генератора траекторий и персистентного PortableRendererStream (renderer_global_h80.bin) для высокочастотной микротекстуризации.

В теории Receding Horizon Control генератор на каждом такте вычисляет оптимальную траекторию на некоторый горизонт вперед, но применяет только начальный сегмент, после чего пересчитывает план с учетом новых целевых координат. Модель вероятностных динамических двигательных примитивов (ProDMP) идеально подходит для этого, так как она представляет траекторию как гладкую линейную комбинацию базисных функций:

$$
y(t) = c_1(y_0, \dot{y}_0)\psi_1(t) + c_2(y_0, \dot{y}_0)\psi_2(t) + \Phi(t)^\top w
$$

Где $\psi_1(t)$ и $\psi_2(t)$ — гомогенные решения базовой динамической системы второго порядка, а $\Phi(t)$ — матрица базисных функций. Использование внутреннего параметра библиотеки goal_mode = 'target' позволяет динамически изменять целевую точку при каждом обновлении цели. Дифференциальные уравнения ProDMP гарантируют аналитическую гладкость и достижение цели без риска нейросетевых галлюцинаций.

Для инициализации системы из состояния покоя ($v=0$), чего не умеет делать ванильный StaticPlanner, генерируется синтетический стационарный префикс, имитирующий физиологический микротремор. Макротраектория, сгенерированная на каждом шаге планирования, передается в PortableRendererStream. Поскольку рекуррентная сеть (GRU) внутри C-рендерера сохраняет свои внутренние скрытые состояния ($h_{80}$) между вызовами, она обеспечивает бесшовную генерацию микроструктуры движения на частоте 1000 Гц, сглаживая стыки макроплана.

Инженерный вердикт и сравнительная оценка
Сравнение двух стратегий демонстрирует превосходство Подхода Б. В таблице ниже представлена детальная оценка параметров обеих архитектур.

| Критерий оценки | Подход А (Стабилизация Continuous) | Подход Б (Receding Horizon + ProDMP) |
| :--- | :--- | :--- |
| **Теоретическая обоснованность** | Эвристическая борьба с нейросетевыми артефактами. Высокий риск непредсказуемого поведения. | Строгая математическая доказуемость гладкости и сходимости (ProDMP). |
| **Задержка вычислений (Latency)** | Высокая (проход через 4 ONNX модели на Python). | Минимальная (NumPy матричные операции + легковесный C-runtime GRU). |
| **Робастность к смене курса цели** | Низкая (FSM блокируется при смене курса). | Высокая (перестроение горизонта планирования без задержек). |
| **Биомеханическая чистота** | Разрушается принудительной фиксацией головок TCN. | Гарантируется базисными функциями ProDMP и Sigma-Delta модуляцией рендерера. |
| **Сложность интеграции** | Требует настройки десятка эвристических порогов. | Требует математической настройки граничных условий ($C^1$ сшивки). |

Подход Б является технически оптимальным решением для создания стабильного генератора траекторий. Он использует математическую строгость ProDMP в качестве макро-планировщика, а нейронную сеть renderer_global_h80.bin — исключительно как микро-генератор логнормальной текстуры и тремора. Вся папка _continuous должна быть объявлена устаревшей и удалена.

3. Решение кинематических и интеграционных вызовов (Подход Б)
Реализация парадигмы Receding Horizon Control на базе ProDMP требует преодоления ключевых кинематических вызовов на стыке макропланирования и высокочастотной генерации.

3.1. Кинематика нулевого старта ($v=0$) и устранение сингулярностей
Классический пайплайн FastPlanner в ABCurves ожидает наличия предыдущего вектора движения (префикса) для обеспечения бесшовного старта. Граничные условия ProDMP инициализируются начальной позицией $y_0$ и начальной скоростью $\dot{y}_0$. Если система находится в состоянии покоя, передача строго нулевого массива может привести к математическим сингулярностям при вычислении кривизны.

Для корректной инициализации формируется синтетический префиксный массив размерности (prefix_length, 2), где prefix_length покрывает интервал в 10–15 мс. Этот массив заполняется репрезентацией естественного микрошума: генерируется гауссовский шум с субпиксельной амплитудой ($\sigma \approx 0.1$), который пропускается через полосовой фильтр с полосой пропускания 8–12 Гц, имитируя микротремор покоя.

При подаче этого префикса граничные условия скорости устанавливаются в ноль: $\dot{y}_0 = [0, 0]$. Согласно уравнению динамической системы ProDMP:

$$
\tau^2 \ddot{y} = \alpha_z (\beta_z (g - y) - \tau \dot{y}) + f(x)
$$

Где $\alpha_z$ и $\beta_z$ — константы системы, $g$ — цель, а $f(x)$ — нелинейная функция формы, параметризованная весами $w$. При $\dot{y}_0 = 0$ демпфирующий член $-\tau \dot{y}$ равен нулю, и начальное ускорение $\ddot{y}$ определяется дистанцией до цели $(g - y_0)$ и функцией формы. Поскольку базисные функции Гаусса в ProDMP плавно нарастают от нуля, аналитически обеспечивается гладкое нарастание ускорения и старт с нулевым рывком (zero-jerk).

3.2. Непрерывность скорости между сегментами ($C^1$ сплайсинг)
При динамическом обновлении координат цели генератор должен строить новую траекторию $\mathcal{P}_{k+1}(t)$, которая бесшовно продолжает предыдущую траекторию $\mathcal{P}_k(t)$. Если начальная скорость нового плана $\dot{y}_0^{(k+1)}$ не совпадает с конечной скоростью предыдущего плана, возникает разрыв первого рода в профиле скорости, генерирующий мгновенный скачок рывка (jerk spike).

Критическим решением является выбор источника граничной скорости $\dot{y}_b$. Использование дискретных накопленных шагов смещения не рекомендуется из-за шума квантования, а фильтры внешних сигналов вносят фазовую задержку.

Скорость $\dot{y}_b$ извлекается исключительно как аналитическая производная предыдущего макроплана ProDMP на момент перепланирования. Если на шаге $k$ был сгенерирован макроплан, и до шага $k+1$ прошло время $\Delta t$, граничные условия для нового плана задаются аналитически:

$$
y_0^{(k+1)} = \mathcal{P}_k(\Delta t), \quad \dot{y}_0^{(k+1)} = \frac{d\mathcal{P}_k}{dt}(\Delta t)
$$

Этот механизм гарантирует строгую непрерывность первого порядка ($C^1$) в пространстве макропланирования. Пространственные невязки на стыке мягко поглощаются Sigma-Delta модулятором C-рендерера, который распределяет остаточный импульс и сохраняет гладкость движения.

3.3. Биомеханически реалистичное замедление в мертвой зоне
Внутренняя мертвая зона (inner deadzone, $r \le r_{inner}$) представляет собой область вблизи цели, где траектория должна стабилизироваться без микроколебаний и без резкой остановки.

Для предотвращения эффекта «удара о стену» (wall slamming), когда траектория резко обрывается на границе зоны, переход реализуется через нелинейную функцию сглаживания. Вклад целевой ошибки $\omega(r)$ масштабируется с использованием логистической сигмоиды:

$$
\omega(r) = \frac{1}{1 + e^{-\lambda (r - r_{inner} - \epsilon)}}
$$

Где $\lambda$ контролирует крутизну перехода, а $\epsilon$ задает смещение. Когда цель попадает в мертвую зону, дельты, отправляемые от макропланировщика в буфер рендерера, плавно обнуляются. При этом высокочастотный шаг генератора (1000 Гц) не останавливается, а переходит в холостой режим (idle run). Рекуррентные ячейки GRU содержат накопленный кинетический потенциал во внутренних состояниях $h_{80}$. При нулевом управляющем сигнале рендерер генерирует естественную кривую выбега, сбрасывая остаточный импульс через Sigma-Delta модулятор. Это формирует гладкий логнормальный хвост замедления.

3.4. Сохранение биомеханических инвариантов моторного контроля
Система обеспечивает строгое соблюдение фундаментальных биомеханических характеристик движения:

1. **Закон двух третей (Two-Thirds Power Law)** постулирует зависимость между линейной скоростью движения $v(t)$ и кривизной траектории $\kappa(t)$ в форме $v(t) = \alpha \cdot \kappa(t)^{-\beta}$, где $\beta \approx 1/3$. В подходе B эта зависимость гарантируется имплицитно: базисные функции ProDMP извлекают ковариационные веса $w$ из матриц FastPlanner (через модель motor.onnx), проецируя движение на биомеханическое многообразие.
2. **Колоколообразные, логнормальные профили скорости и рывка**. Линейная динамическая система ProDMP аналитически минимизирует интеграл рывка (minimum-jerk model), формируя плавные профили ускорения и замедления.
3. **Физиологический микротремор (8–12 Гц)**. Непрерывный поток макроплана поступает в PortableRendererStream (renderer_global_h80.bin). Рекуррентная нейросеть C-рендерера работает с тактом 1000 Гц и никогда не сбрасывает свое скрытое состояние между шагами обновления макроплана. Это позволяет GRU непрерывно накладывать физиологический тремор (в диапазоне 8–12 Гц), обеспечивая аутентичные биомеханические свойства траектории.

4. Архитектурная спецификация и пошаговый чек-лист рефакторинга
Для внедрения генератора на основе Receding Horizon и ProDMP предоставляется спецификация модификации кодовой базы ABCurves.

4.1. Таблица аудита и прунинга репозитория

| Локация в репозитории | Имя файла или компонента | Статус | Необходимые модификации и директивы |
| :--- | :--- | :--- | :--- |
| `abcurves/` | `prodmp.py` | **KEEP** | Оставить без изменений. Ядро вычислений базисных функций ProDMP. |
| `abcurves/` | `fast_planner.py` | **MODIFY** | Добавить метод `get_analytical_state_at(delta_t)` для аналитического вычисления скорости и позиции в момент $\Delta t$. |
| `abcurves/` | `portable_renderer.py` | **KEEP** | Оставить без изменений. Python-интерфейс к C-runtime движку. |
| `runtime/c/` | `*` (C-runtime GRU) | **KEEP** | Высокочастотный 1000 Гц движок рендеринга, без изменений. |
| `abcurves/_continuous/` | `planner.py` | **DELETE** | Полностью удалить. Устаревший FSM с артефактами переключения головок. |
| `abcurves/_continuous/` | `kernels.py` | **DELETE** | Удалить. Заменяется аналитическим извлечением скорости из ProDMP. |
| `abcurves/_continuous/` | `runtime.py` | **DELETE** | Удалить. Упраздняется в пользу архитектуры Receding Horizon. |
| `models/continuous/` | `hazard.onnx, brake.onnx, choice.onnx` | **DELETE** | Удалить весовые файлы классификаторов. |
| `models/continuous/` | `motor.onnx` | **KEEP** | Сохранить. Используется FastPlanner для матриц ковариаций $w$. |
| `models/` | `renderer_global_h80.bin` | **KEEP** | Сохранить. Веса GRU для генерации тремора и микротекстуры. |

4.2. Интерфейс контроллера и алгоритмический поток данных
Архитектурным центром системы становится класс `ClosedLoopProDMPController`, который инкапсулирует логику сшивки планов и связывает динамическое обновление цели с высокочастотным тактом генерации (1000 Гц).

Внутреннее состояние контроллера поддерживается набором персистентных переменных:
- `self.renderer_stream` (`PortableRendererStream`): персистентный объект, хранящий скрытые состояния GRU $h_{80}$.
- `self.current_plan`: полиномиальное представление текущего сгенерированного макроплана ProDMP.
- `self.time_since_plan_ms`: счетчик миллисекунд с момента генерации текущего плана.
- `self.last_analytical_vel`: текущая аналитическая скорость $\dot{y}_b$.

Алгоритмическая логика динамического обновления цели реализуется в методе `on_target_update(target_offset, dt, apply_deadzone=True)`:
1. **Нелинейное демпфирование ошибки**: если активировано демпфирование, вектор `target_offset` масштабируется сигмоидой затухания мертвой зоны $\omega(r)$.
2. **Извлечение граничных условий ($C^1$ сплайсинг)**: вычисляется аналитическое состояние $\dot{y}_b$ через `get_analytical_state_at(self.time_since_plan_ms)`. Значение записывается в `self.last_analytical_vel`.
3. **Генерация макроплана (Receding Horizon)**: вызывается `FastPlanner.plan(goal_mode='target', ...)` с начальной позицией $y_0 = [0, 0]$, скоростью $\dot{y}_0 = \text{self.last\_analytical\_vel}$ и целевым вектором $g$.
4. **Сброс таймера**: счетчик `self.time_since_plan_ms` обнуляется.
5. **Трансляция в рендерер**: сгенерированные дельты траектории помещаются в буфер `self.renderer_stream` на горизонт 10–15 мс.

Параллельно функционирует высокочастотный шаг генератора `step()` (1000 Гц):
1. Проверяется доступность дельт в буфере `renderer_stream`.
2. Если буфер содержит данные, выполняется шаг рекуррентной сети: `renderer_stream.step(delta)`.
3. Если буфер пуст (цель в мертвой зоне или пауза в обновлениях), выполняется холостой прогон `renderer_stream.step((0, 0))`. Sigma-Delta модулятор плавно гасит кинетическую энергию, а GRU генерирует физиологический микротремор.
4. Счетчик времени инкрементируется: `self.time_since_plan_ms += 1`.
5. Метод возвращает рассчитанные смещения траектории $(\Delta x, \Delta y)$.

При сбросе контроллера вызывается метод `reset()`:
Очищает буфер очереди дельт, плавно обнуляет аналитическую скорость `self.last_analytical_vel = [0, 0]` и синтезирует 10–15 мс стартового Гауссовского микрошума ($\sigma \approx 0.1$) для обеспечения бесшовного старта (zero-start kinematics) при следующем обновлении цели.

Данная архитектура устраняет нестабильные элементы, перенося фокус на математически строгий Receding Horizon Control на базе ProDMP. Она гарантирует непрерывность $C^1$, биомеханическую чистоту выходной траектории и высокую устойчивость при сопровождении динамических целей.
