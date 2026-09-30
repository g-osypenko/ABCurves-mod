
!!INITIAL PROMPT!!

# Deep Research Task: Architectural Audit & Refactoring of ABCurves for Closed-Loop Aiming & Target Acquisition in a 3D Simulation

!!!! researchable github repository link - https://github.com/optima-manent/ABCurves !!!!

## CONTEXT & PRIMARY OBJECTIVE

You are provided with an attached GitHub repository containing the **ABCurves** codebase (a continuous and static mouse movement generation library based on ProDMP and a high-frequency GRU C-runtime renderer).

We are developing an **automated crosshair aiming and dynamic target acquisition controller** (human-like visual servoing) for synthetic agents in a **custom-built real-time 3D first-person simulation**:

- **Task Definition:** The agent observes the 3D scene from a first-person perspective. A vision pipeline detects dynamic targets and outputs the target displacement error $(\Delta x, \Delta y)$ in angular/screen counts relative to the camera crosshair. The controller must orient the camera via mouse packets to smoothly acquire and track evasive targets.
- **Perception:** 120–240 FPS visual tracking pipeline (TensorRT YOLO) outputting target error $(\Delta x, \Delta y)$ relative to reticle. Pipeline transport latency: ~15–25 ms.
- **Actuation:** Hardware microcontroller (ESP32-S3 over 4 Mbps UART) emitting physical USB HID mouse packets at strictly 1000 Hz (1 tick = 1 ms) into the simulation host.
- **Core Dilemma:** The library's `ContinuousPlanner` (`ContinuousPipeline`) currently exhibits severe behavioral instability under dynamic closed-loop combat ("spastic, jerky, ataxia-like aim"). On the other hand, the vanilla `StaticPlanner` was designed exclusively for open-loop "B-trigger" movement completion and cannot natively initialize from a standstill ($v=0$) or perform closed-loop frame-by-frame target tracking.

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

Explain the exact mathematical and state-machine root causes of the "spastic aim" in `ContinuousPlanner`:

1. Why does sampling `motor_rng.categorical(probabilities)` across the 16 heads every 32 ms produce high-frequency lateral oscillation (31 Hz curvature flipping)?
2. How does `hazard.onnx` trigger premature Poisson braking (`Mode 1`), lock the system into `Mode 2 (Hold)` for 100–200 ms, and cause explosive overshoot when resuming with `previous_valid = False`?
3. How does finite-difference velocity calculation in `kernels.py` amplify $\pm 1-2$ px detector bounding-box jitter into false lead compensations?

---

## 2. THE CORE COMPARATIVE STUDY: OPTION 1 VS OPTION 2

Evaluate and compare two concrete refactoring strategies that utilize existing repository modules:

### Approach A: Surgical Stabilization of `ContinuousPlanner`

How to minimally prune and stabilize the existing `ContinuousPipeline` without throwing away the trained continuous models:

- How to eliminate 32 ms head flapping (e.g., sticky head, argmax lock per tracking engagement).
- How to surgically bypass or neuter the Poisson `hazard.onnx` / `brake.onnx` FSM to prevent freezing while maintaining natural terminal deceleration.
- How to filter target velocity inputs to insulate the model from detector noise.
- What are the residual risks and architectural limitations of this approach?

### Approach B: Closed-Loop Tracking via `StaticPlanner` / `ProDMP` + Global Renderer (The "Fork" Approach)

How to adapt `StaticPlanner` (`FastPlanner` / raw ProDMP) and the persistent 1000 Hz C-stream into a frame-by-frame (120–240 Hz) closed-loop receding-horizon controller:

- How to initialize ProDMP from zero velocity ($v=0$) using a synthetic stationary prefix without math singularities ($\kappa \to \infty$ or heading blow-up).
- How `goal_mode = 'target'` in `prodmp.py` can be used to ensure mathematical convergence to target error while retaining the learned 20-basis acceleration profile.
- How to feed the resulting smooth trajectory into the persistent C-runtime `PortableRendererStream` without resetting internal GRU hidden states between frames.

### Verdict:

Deliver a definitive engineering verdict comparing Approach A vs Approach B in terms of implementation friction, runtime latency, robustness against evasive targets (strafing), and kinematic cleanliness.

---

## 3. FOUR ESSENTIAL KINEMATIC & INTEGRATION CHALLENGES

Provide concrete mathematical and algorithmic answers for:

### 1. Zero-Start Kinematics ($v=0$)

In `StaticPlanner` / `ProDMP`, standard inference expects a moving prefix. 

- Exactly what prefix tensor (shape, values, micro-noise vs zeros) should be passed to `FastPlanner` when the camera/reticle is at rest?
- How do ProDMP boundary conditions ($\xi_1 y_0 + \xi_2 \dot{y}_0$) mathematically guarantee smooth, zero-jerk acceleration from $\dot{y}_0 = [0, 0]$?

### 2. Inter-Frame Velocity Continuity ($C^1$ Splicing at 120–240 FPS)

When visual updates arrive every 4–8 ms with new target coordinates:

- How should boundary velocity $\dot{y}_b$ be assigned at frame $k+1$ to prevent step velocity discontinuities (infinite jerk spikes) at the seam between frames?
- Should $\dot{y}_b$ be derived from recent emitted hardware counts, the prior ProDMP plan derivative, or the W5 filtered state?

### 3. Biomechanically Realistic Deadzone Deceleration

We have an inner deadzone ($r \le r_{inner}$) where the crosshair must hold without jitter or abrupt stopping:

- How should deceleration be governed inside and near the deadzone?
- Compare two mechanisms: (a) decaying/dampening the target error vector fed into ProDMP vs (b) zeroing deltas sent to `PortableRendererStream` and letting the GRU/Sigma-Delta accumulator naturally absorb residual momentum.
- How to prevent "wall slamming" (hitting the deadzone edge like a rigid mechanical barrier)?

### 4. Biomechanical Invariant Preservation & Kinematic Classification

To ensure the synthetic agent's mouse movement is statistically indistinguishable from human motor control:

- Detail how this refactored pipeline preserves core human motor invariants:
    - The Two-Thirds Power Law ($v(t) = \alpha \cdot \kappa(t)^{-\beta}$ connecting speed and curvature).
    - Bell-shaped, log-normal velocity and jerk envelopes.
    - 8–12 Hz physiological micro-tremor and physical sensor quantization.
- Specifically analyze the risk of synthetic/robotic artifacts appearing at:
    - Inter-frame splicing seams.
    - Deadzone boundary transitions.
- How does routing through `PortableRendererStream` (`renderer_global_h80.bin`) insulate the movement against heuristic/statistical anomaly detection algorithms?

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
        - `on_visual_frame(target_counts, dt)`: handling error, head locking, ProDMP generation, and pushing to the 1000 Hz buffer.
        - `get_1khz_report()`: polling by the hardware UART thread.
        - `reset()`: engagement reset logic.

final output and answer should be in russian



!!ANSWER!!

Архитектурный аудит и рефакторинг библиотеки ABCurves для систем замкнутого прицеливания и захвата целей в 3D-симуляциях
Кодовая база ABCurves представляет собой сложный программный комплекс, изначальной целью которого была генерация человекоподобных движений мыши в режиме реального времени. Базируясь на математическом аппарате вероятностных динамических двигательных примитивов (Probabilistic Dynamic Movement Primitives, ProDMP) и высокочастотном рекуррентном рендерере (GRU C-runtime), эта библиотека продемонстрировала высокую эффективность в задачах выполнения движений в разомкнутом контуре (open-loop). Тем не менее, текущая инженерная задача требует создания автоматизированного контроллера прицеливания и динамического захвата целей (визуального сервопривода) для синтетических агентов, функционирующих в трехмерной симуляционной среде от первого лица.
Эта задача вводит строгие системные ограничения. Визуальный пайплайн, построенный на базе TensorRT YOLO, функционирует на частоте 120–240 Гц, генерируя вектор ошибки смещения цели ($\Delta x, \Delta y$) относительно перекрестия прицела с транспортной задержкой порядка 15–25 мс. В то же время актуаторная подсистема, реализованная на базе микроконтроллера ESP32-S3 с интерфейсом UART (4 Мбит/с), должна эмитировать физические пакеты USB HID строго на частоте 1000 Гц (1 такт = 1 мс). В таких условиях существующий в библиотеке модуль ContinuousPlanner демонстрирует критическую поведенческую нестабильность, выражающуюся в спастическом, атактическом прицеливании. С другой стороны, базовый StaticPlanner не способен инициализировать движение из состояния покоя ($v=0$) и не поддерживает покадровое обновление целеуказания в замкнутом контуре.
Настоящий отчет представляет собой исчерпывающий архитектурный и математический аудит репозитория, сравнительный анализ стратегий рефакторинга и строгую спецификацию для построения робастного контроллера замкнутого цикла, который сохраняет фундаментальные биомеханические инварианты человеческого движения.

1. Глубокий аудит архитектуры репозитория ABCurves
   Инспекция структуры репозитория выявила четкое концептуальное и физическое разделение между аналитическим макропланировщиком (ProDMP, реализованным в abcurves/fast_planner.py и abcurves/prodmp.py), микрорендерером текстуры движения (abcurves/portable_renderer.py и директория runtime/c/) и экспериментальным контуром непрерывного планирования (abcurves/_continuous/). Для понимания причин кинематической деградации системы в условиях замкнутого контура необходимо детально разобрать математическую природу дефектов модуля ContinuousPlanner.
   1.1. Математическая природа высокочастотной латеральной осцилляции (31 Гц флаттер)
   В архитектуре ContinuousPlanner используется 16-головая сеть временных сверток (Temporal Convolutional Network, TCN), которая на каждом шаге логического вывода предсказывает распределение вероятностей по 16 топологическим кластерам траекторий (головкам). Выбор конкретного двигательного примитива осуществляется посредством стохастического сэмплирования из категориального распределения вероятностей. Эта операция вызывается с фиксированным интервалом в 32 мс.
   В условиях открытого контура, на которых обучалась сеть, распределение вероятностей плавно эволюционирует на протяжении сотен миллисекунд. Однако в замкнутом контуре, где визуальный пайплайн обновляет координаты цели каждые 4–8 мс (120–240 Гц), вектор целевой ошибки ($\Delta \vec{e}$) подвергается постоянным высокочастотным возмущениям из-за микроуклонений динамической цели и задержек пайплайна. Стохастическое сэмплирование каждые 32 мс (что соответствует частоте обновления $\approx 31.25$ Гц) в таких условиях приводит к явлению, известному в теории нелинейного управления как переключение мод (mode chattering).
   Если распределение вероятностей между двумя соседними топологическими головками, отвечающими за противоположные фазовые состояния (например, плавная дуга влево и резкий компенсаторный рывок вправо), сближается до значений порядка $P(A) \approx 0.45$ и $P(B) \approx 0.55$, стохастическое сэмплирование начинает чередовать эти базисы. Каждая головка TCN задает собственное уникальное пространство параметров: индивидуальную матрицу ковариации, базовую кривизну и профиль ускорения. Смена базиса с частотой 31.25 Гц вызывает мгновенную инверсию знака второй производной (ускорения) в латеральной плоскости. На уровне физического актуатора это проявляется как высокочастотный тремор или атаксия прицела. Это фундаментально разрушает базовые биомеханические принципы, в частности закон двух третей (Two-Thirds Power Law), который постулирует плавную взаимосвязь между кривизной траектории и скоростью движения.
   1.2. Автомат состояний Пуассоновского торможения и кинематический паралич
   Архитектура непрерывного контура жестко опирается на конечный автомат (Finite State Machine, FSM), управляемый двумя нейросетевыми классификаторами: hazard.onnx (детектор аномалий и резких смен курса) и brake.onnx (контроллер профиля замедления). Эта подсистема была спроектирована для имитации времени человеческой реакции на внезапные раздражители.
   В замкнутом боевом контуре эта имитация приводит к катастрофическим отказам управления. Модель hazard.onnx была обучена на наборах данных, где резкие изменения градиента цели (например, цель внезапно меняет направление движения на 180 градусов) являются редкими аномалиями. При отслеживании уклоняющейся (strafing) цели в симуляции, динамическое изменение направления является нормой, но hazard.onnx классифицирует эти изменения как кинематические аномалии. Это мгновенно переводит конечный автомат в режим экстренного торможения (Mode 1), а затем в режим удержания (Mode 2).
   Для обеспечения биологической правдоподобности выход из режима удержания смоделирован как пуассоновский процесс, что вводит стохастическую задержку длительностью 100–200 мс. В течение этой «слепой зоны» система игнорирует новые визуальные кадры, а вектор ошибки ($\Delta x, \Delta y$) продолжает лавинообразно накапливаться по мере того, как цель уходит из перекрестия. Когда FSM наконец разрешает возобновление движения (путем сброса флага внутренней валидации), планировщик получает на вход колоссальную ошибку. Пытаясь компенсировать ее, ContinuousPlanner генерирует баллистическую саккаду с максимальным ускорением. Эта саккада неизбежно перерегулирует (overshoot) цель, генерирует новую аномалию градиента, и цикл ложного торможения запускается вновь, создавая эффект непрерывного «дергания» и «зависания».
   1.3. Усиление шума детектора при конечно-разностном дифференцировании
   Третьей критической уязвимостью текущей архитектуры является метод вычисления скорости цели в модуле abcurves/_continuous/kernels.py. В текущей реализации используется наивное дифференцирование методом конечных разностей: $v = \frac{\Delta x}{\Delta t}$.
   Нейросетевой детектор YOLO, несмотря на высокую точность, обладает присущим ему пространственным джиттером ограничивающего прямоугольника (bounding box) с амплитудой порядка $\pm 1-2$ пикселя на каждый кадр. При частоте визуального контура 240 Гц шаг времени $\Delta t$ составляет примерно 0.00416 секунды. В этих условиях изменение координаты всего на 2 пикселя из-за оптического шума детектора транслируется в гигантский ложный вектор скорости:

$$
v_{noise} = \frac{2 \text{ px}}{0.00416 \text{ s}} \approx 480 \text{ px/s}
$$

Поскольку ContinuousPlanner использует вычисленную скорость для предиктивного упреждения (lead compensation), этот высокочастотный шум напрямую подается в латентное пространство планировщика. Незначительный джиттер детектора превращается в фантомные рывки прицела со скоростью сотен пикселей в секунду. Полное отсутствие низкочастотных фильтров или фильтров состояния (например, фильтра Калмана) делает контур обратной связи математически нестабильным и подверженным резонансному усилению шума.
2. Сравнительное исследование архитектурных парадигм
Для создания робастного контроллера визуального сервопривода необходимо проанализировать две диаметрально противоположные стратегии рефакторинга, опирающиеся на уже существующие компоненты репозитория ABCurves.
Подход А: Хирургическая стабилизация непрерывного планировщика
Данная стратегия предполагает сохранение текущего модуля ContinuousPipeline и всех связанных с ним ONNX-моделей (motor, choice, hazard, brake), но с внедрением жестких алгоритмических ограничений для подавления выявленных артефактов.
Устранение 31 Гц флаттера головок потребует перехода от стохастического сэмплирования к детерминированному выбору через операцию нахождения максимума (argmax) по распределению вероятностей. Для предотвращения переключения топологий в процессе одного акта прицеливания потребуется внедрить логику «липкого индекса» (sticky head). Этот механизм будет фиксировать выбранную головку TCN при инициализации движения и удерживать ее до тех пор, пока скалярное произведение текущего вектора цели и исходного вектора на момент захвата не станет отрицательным (что свидетельствует о смене полусферы движения).
Для нейтрализации эффекта пуассоновского паралича потребуется глубокое вмешательство в Python-обвязку для принудительного обхода (bypassing) автомата состояний hazard.onnx. Придется искусственно удерживать систему в состоянии активного трекинга, перехватывать сигналы торможения и транслировать их в плавное экспоненциальное затухание целевого вектора. Кроме того, на входе в kernels.py потребуется развернуть альфа-бета фильтр или фильтр Калмана с моделью постоянного ускорения (Constant Acceleration Model) для изоляции дифференциатора от высокочастотного джиттера YOLO.
Фундаментальным ограничением и критическим риском этого подхода является проблема сдвига распределения (distribution shift). Нейросетевые веса motor.onnx были жестко переобучены на конкретных наборах человеческих данных в условиях открытого контура. Любая попытка искусственно зафиксировать индекс головки, отбросить стохастичность или обойти автомат состояний приведет к тому, что входные тензоры, подаваемые на рекуррентные слои, выйдут за пределы обучающего многообразия. Генератор начнет выдавать траектории из неисследованных областей латентного пространства, что неминуемо приведет к математическим сингулярностям: зависаниям векторов, бесконечным значениям кривизны или генерации NaN в выходных потоках.
Подход Б: Контроль с удаляющимся горизонтом через StaticPlanner и ProDMP (Подход «Форк»)
Вторая парадигма требует радикального отказа от черного ящика TCN и перехода к математически доказуемой модели управления с удаляющимся горизонтом (Receding Horizon Control). В основе этого подхода лежит использование исходных классов ProDMP (FastPlanner) в качестве макро-генератора траекторий и персистентного PortableRendererStream (renderer_global_h80.bin) для высокочастотной микротекстуризации.
В теории Receding Horizon Control контроллер на каждом такте вычисляет оптимальную траекторию на некоторый горизонт вперед, но применяет только первый управляющий шаг, после чего пересчитывает план с учетом новых данных. Модель вероятностных динамических двигательных примитивов (ProDMP) идеально подходит для этого, так как она представляет траекторию как гладкую линейную комбинацию базисных функций:

$$
y(t) = c_1(y_0, \dot{y}_0)\psi_1(t) + c_2(y_0, \dot{y}_0)\psi_2(t) + \Phi(t)^\top w
$$

Где $\psi_1(t)$ и $\psi_2(t)$ — гомогенные решения базовой динамической системы второго порядка, а $\Phi(t)$ — матрица базисных функций. Использование внутреннего параметра библиотеки goal_mode = 'target' позволяет динамически изменять целевую точку на каждом кадре визуального контура (120–240 Гц). Дифференциальные уравнения ProDMP гарантируют аналитическую гладкость и достижение цели без риска нейросетевых галлюцинаций.
Для инициализации системы из состояния покоя ($v=0$), чего не умеет делать ванильный StaticPlanner, генерируется синтетический стационарный префикс, имитирующий физиологический микротремор. Макротраектория, сгенерированная на каждом кадре, передается в PortableRendererStream. Поскольку рекуррентная сеть (GRU) внутри C-рендерера сохраняет свои внутренние скрытые состояния ($h_{80}$) между вызовами, она обеспечивает бесшовную генерацию микроструктуры движения на частоте 1000 Гц, скрывая точки перестроения макроплана.
Инженерный вердикт и сравнительная оценка
Сравнение двух стратегий демонстрирует безоговорочное превосходство Подхода Б. В таблице ниже представлена детальная оценка параметров обеих архитектур.
Критерий оценки	Подход А (Стабилизация Continuous)	Подход Б (Receding Horizon + ProDMP)
Теоретическая обоснованность	Эвристическая борьба с нейросетевыми артефактами. Высокий риск непредсказуемого поведения.	Строгая математическая доказуемость гладкости и сходимости (ProDMP).
Задержка вычислений (Latency)	Высокая (проход через 4 ONNX модели на Python).	Минимальная (NumPy матричные операции + легковесный C-runtime GRU).
Робастность к стрейфу цели	Низкая (FSM блокируется при смене курса).	Абсолютная (покадровое перестроение горизонта планирования без задержек).
Биомеханическая чистота	Разрушается принудительной фиксацией головок TCN.	Гарантируется базисными функциями ProDMP и Sigma-Delta модуляцией рендерера.
Сложность интеграции	Требует настройки десятка эвристических порогов.	Требует точной математической настройки граничных условий (сшивки).
Подход Б является единственным технически жизнеспособным решением для создания контроллера промышленного класса. Он использует математическую строгость ProDMP в качестве макро-контроллера, а нейронную сеть renderer_global_h80.bin — исключительно как микро-генератор логнормальной текстуры и тремора. Вся директория _continuous должна быть объявлена устаревшей и удалена.
3. Решение кинематических и интеграционных вызовов (Подход Б)
Реализация парадигмы Receding Horizon Control на базе ProDMP требует преодоления четырех сложных биомеханических и алгоритмических препятствий на стыке макропланирования и аппаратной актуации.
3.1. Кинематика нулевого старта ($v=0$) и устранение сингулярностей
Классический пайплайн FastPlanner в ABCurves ожидает наличия предыдущего вектора движения (префикса) для обеспечения бесшовного старта. Граничные условия ProDMP инициализируются начальной позицией $y_0$ и начальной скоростью $\dot{y}_0$. Если прицел абсолютно неподвижен, передача нулевого тензора в планировщик приведет к математическим сингулярностям: делению на ноль при вычислении азимутального угла или бесконечной начальной кривизне $\kappa \to \infty$.
Для корректной инициализации необходимо сформировать синтетический префиксный тензор размерности (prefix_length, 2), где prefix_length покрывает окно в 10–15 мс. Этот тензор заполняется не нулями, а репрезентацией биологического микрошума. Конкретно, генерируется гауссовский шум с субпиксельной амплитудой ($\sigma \approx 0.1$ пикселя), который пропускается через полосовой фильтр с полосой пропускания 8–12 Гц. Это имитирует физиологический тремор покоя человеческой руки, который постоянно присутствует даже при удержании мыши на месте.
При подаче этого префикса граничные условия скорости устанавливаются в строгий ноль: $\dot{y}_0 = [0, 0]$. Согласно уравнению динамической системы ProDMP:

$$
\tau^2 \ddot{y} = \alpha_z (\beta_z (g - y) - \tau \dot{y}) + f(x)
$$

Где $\alpha_z$ и $\beta_z$ — константы системы, $g$ — цель, а $f(x)$ — нелинейная функция формы, параметризованная весами $w$. При $\dot{y}_0 = 0$ демпфирующий член $-\tau \dot{y}$ исчезает, и начальное ускорение $\ddot{y}$ определяется исключительно дистанцией до цели $(g - y_0)$ и функцией формы. Поскольку базисные функции Гаусса в ProDMP плавно нарастают от нуля, математически гарантируется плавное, экспоненциальное нарастание ускорения. Это обеспечивает старт с нулевым рывком (zero-jerk), полностью исключая взрывные скачки, свойственные ПИД-регуляторам.
3.2. Непрерывность скорости между кадрами ($C^1$ сплайсинг при 120–240 Гц)
При покадровом обновлении цели с частотой визуального пайплайна (каждые 4–8 мс) контроллер должен генерировать новую траекторию $\mathcal{P}_{k+1}(t)$, которая бесшовно продолжает предыдущую траекторию $\mathcal{P}_k(t)$. Если начальная скорость нового плана $\dot{y}_0^{(k+1)}$ не совпадает с конечной скоростью предыдущего плана, возникает разрыв первого рода в профиле скорости, что транслируется в бесконечный скачок рывка (jerk spike). Человеко-машинные системы детекции мгновенно идентифицируют такие разрывы как роботизированные артефакты.
Критическим архитектурным решением является выбор источника граничной скорости $\dot{y}_b$. Использование данных от аппаратных счетчиков мыши (эмитированных пакетов) строго запрещено. Аппаратные счетчики оперируют целочисленными квантами пикселей, что делает их производную ступенчатой функцией с колоссальным уровнем шума квантования. Аналогично, использование отфильтрованного состояния (например, через фильтр W5) вносит фазовую задержку.
Скорость $\dot{y}_b$ должна извлекаться исключительно из аналитической производной предыдущего макроплана ProDMP на момент прихода нового кадра. Если на кадре $k$ был сгенерирован макроплан, и до прихода кадра $k+1$ прошло время $\Delta t$, граничные условия для нового плана задаются аналитически:

$$
y_0^{(k+1)} = \mathcal{P}_k(\Delta t), \quad \dot{y}_0^{(k+1)} = \frac{d\mathcal{P}_k}{dt}(\Delta t)
$$

Этот механизм гарантирует строгую непрерывность первого порядка ($C^1$) в непрерывном пространстве макропланирования. Любые микроскопические расхождения между аналитической позицией $\mathcal{P}_k(\Delta t)$ и реальным физическим положением курсора делегируются на уровень C-рендерера. Встроенный в рендерер Sigma-Delta модулятор плавно абсорбирует эти пространственные невязки, распределяя остаточный импульс на несколько миллисекунд и сохраняя гладкость движения.
3.3. Биомеханически реалистичное замедление в мертвой зоне
Внутренняя мертвая зона (inner deadzone, $r \le r_{inner}$) представляет собой область вблизи центра перекрестия, где прицел должен остановиться и удерживать цель без микроколебаний и без резкого эффекта удара.
Существует два основных механизма организации замедления в этой зоне. Первый механизм (a) предполагает экспоненциальное затухание вектора ошибки, подаваемого в ProDMP: при приближении к центру цель $g$ искусственно смещается к текущей позиции курсора. Однако этот метод заставляет планировщик постоянно пересчитывать сходящиеся траектории, что может вызвать вычислительную нестабильность.
Предпочтительным является механизм (b) — абсорбция кинетической энергии через Sigma-Delta модулятор рендерера. Для предотвращения эффекта «удара о стену» (wall slamming), когда прицел мгновенно замирает на границе мертвой зоны, переход реализуется через нелинейную функцию сглаживания. Вклад целевой ошибки $\omega(r)$ в генерацию новых планов масштабируется с использованием логистической сигмоиды:

$$
\omega(r) = \frac{1}{1 + e^{-\lambda (r - r_{inner} - \epsilon)}}
$$

Где $\lambda$ контролирует крутизну перехода, а $\epsilon$ задает смещение. Когда курсор входит в мертвую зону, дельты, отправляемые от макропланировщика в PortableRendererStream, плавно обнуляются. При этом сам 1000-герцовый цикл рендерера не останавливается, а переходит в холостой режим (idle run). Рекуррентные ячейки GRU содержат накопленный кинетический потенциал во внутренних состояниях $h_{80}$. При нулевом макровходе рендерер генерирует естественную кривую выбега (coast-down curve), сбрасывая остаточный импульс в соответствии с законами нейромышечного торможения (закон Фиттса). Это формирует идеальный логнормальный хвост замедления, характерный для модели Sigma-Lognormal.
3.4. Сохранение биомеханических инвариантов и защита от кинематической классификации
Фундаментальная задача системы — генерация пакетов мыши, статистически неотличимых от профилей человеческого моторного контроля, с целью обхода эвристических античит-алгоритмов. Предложенная архитектура обеспечивает сохранение трех ключевых биомеханических инвариантов.
Закон двух третей (Two-Thirds Power Law) постулирует нелинейную изоаффинную зависимость между линейной скоростью движения $v(t)$ и кривизной траектории $\kappa(t)$ в форме $v(t) = \alpha \cdot \kappa(t)^{-\beta}$, где $\beta \approx 1/3$. В подходе B эта зависимость гарантируется имплицитно. Базисные функции ProDMP извлекают ковариационные веса $w$ из заранее обученных матриц FastPlanner (через модель motor.onnx, обученную на миллионах человеческих движений). Проецирование динамики на этот усвоенный базис гарантирует, что генерируемые макрокривые лежат строго в многообразии, удовлетворяющем закону двух третей.
Вторым инвариантом являются колоколообразные, логнормальные профили скорости и рывка (bell-shaped envelopes). Математика линейных динамических систем в основе ProDMP естественным образом минимизирует интеграл рывка (minimum-jerk model) на заданной дистанции. Это генерирует симметричные профили скорости при высокоскоростных саккадических рывках и асимметричные профили при точном доведении до цели, полностью повторяя человеческую стратегию скорости-точности.
Наибольший риск возникновения роботизированных артефактов лежит в плоскости высокочастотного микро-анализа (например, с использованием Wigner-Ville трансформаций). Чтобы избежать детекции на швах сплайсинга (интервалах 4–8 мс), архитектура пропускает непрерывный макроплан через PortableRendererStream (renderer_global_h80.bin). Рекуррентная нейросеть C-рендерера работает на частоте 1000 Гц и никогда не сбрасывает свое скрытое состояние между кадрами визуального контура. Это позволяет GRU непрерывно накладывать поверх идеализированной гладкой макротраектории физиологический микротремор (в диапазоне 8–12 Гц) и моделировать шум сенсорного квантования физической оптической мыши. В результате сгенерированный поток HID-пакетов обладает спектральной плотностью и фазовыми характеристиками, полностью аутентичными биологическим образцам.
4. Архитектурная спецификация и пошаговый чек-лист рефакторинга
Для внедрения контроллера на основе Receding Horizon и ProDMP в производственную среду, предоставляется строгая спецификация модификации кодовой базы ABCurves.
4.1. Таблица аудита и прунинга репозитория
Для очистки кодовой базы от нестабильных элементов необходимо провести жесткий аудит согласно следующей таблице маршрутизации:
Локация в репозитории	Имя файла или компонента	Статус	Необходимые модификации и архитектурные директивы
abcurves/	prodmp.py	KEEP	Оставить без изменений. Обеспечивает ядро матричных вычислений базисных функций и интеграции линейных систем.
abcurves/	fast_planner.py	MODIFY	Адаптировать для приема динамического вектора цели (goal_mode = 'target'). Реализовать метод update_boundary_conditions() для извлечения аналитической производной макроплана на заданный момент времени.
abcurves/	portable_renderer.py	KEEP	Оставить без изменений. Служит Python-интерфейсом для управления инстансом PortableRendererStream.
runtime/c/	* (C-runtime GRU)	KEEP	Базовый высокочастотный движок не требует модификаций. Должен вызываться из потока прерываний (или выделенного треда), синхронизированного с UART ESP32.
abcurves/_continuous/	planner.py	DELETE	Полностью удалить. Модуль не подлежит восстановлению из-за системного 31 Гц флаттера.
abcurves/_continuous/	kernels.py	DELETE	Удалить. Конечно-разностное исчисление скорости заменяется аналитическим извлечением $\dot{y}_b$ из ProDMP.
abcurves/_continuous/	runtime.py	DELETE	Удалить. Иерархия конечных автоматов торможения упраздняется в пользу механизма Sigma-Delta абсорбции.
models/	hazard.onnx, brake.onnx, choice.onnx	DELETE	Удалить весовые файлы классификаторов. Они являются источником пуассоновских задержек и кинематического паралича.
models/	motor.onnx	KEEP	Сохранить. Используется макропланировщиком FastPlanner для извлечения пространственных ковариаций $w$ и соблюдения закона двух третей.
models/	renderer_global_h80.bin	KEEP	Ключевой бинарный артефакт. Содержит веса GRU для генерации тремора и логнормальной микротекстуры.
4.2. Интерфейс контроллера и алгоритмический поток данных
Архитектурным центром новой системы становится класс ClosedLoopProDMPController, который инкапсулирует логику сшивки траекторий и оркестрирует взаимодействие между 240 Гц контуром технического зрения и 1000 Гц контуром микроконтроллера.
Внутреннее состояние контроллера поддерживается набором персистентных переменных. Критически важным является сохранение инстанса self.renderer_stream (PortableRendererStream), который хранит скрытые слои GRU между тактами. Также сохраняются self.current_plan (полиномиальная репрезентация текущего сгенерированного макроплана), счетчик self.time_since_plan_ms (аккумулятор времени в миллисекундах с момента генерации последнего плана) и вектор self.last_analytical_vel (текущая аналитическая скорость $\dot{y}_b$).
Алгоритмическая логика обработки новых визуальных кадров реализуется в методе on_visual_frame(target_counts, dt), который вызывается асинхронно при поступлении вектора ошибки от YOLO пайплайна:

1. Нелинейное сглаживание ошибки: Входной вектор target_counts ($\Delta x, \Delta y$) пропускается через логистическую функцию затухания мертвой зоны. Если Евклидово расстояние до цели попадает в радиус $r_{inner}$, вектор цели плавно масштабируется к нулю.
2. Извлечение граничных условий ($C^1$ сплайсинг): Контроллер обращается к self.current_plan и вычисляет аналитическую производную $\dot{y}_b$ на временной отметке self.time_since_plan_ms. Это значение записывается в self.last_analytical_vel.
3. Генерация макроплана (Receding Horizon): Вызывается метод FastPlanner.plan(...). На вход подаются: виртуальная позиция $y_0 = [0, 0]$ (координатная система всегда относительна текущего положения курсора), начальная скорость $\dot{y}_0 = \text{self.last\_analytical\_vel}$, и новый вектор цели $g = \text{target\_counts}$.
4. Сброс временных маркеров: Синхронизационный таймер self.time_since_plan_ms обнуляется.
5. Трансляция в рендерер: Сгенерированный базис позиций отправляется в буфер self.renderer_stream.push_macro_deltas(...). Для минимизации задержек отправляется только сегмент плана на ближайшие 10–15 мс.
   Параллельно функционирует высокочастотный цикл актуации get_1khz_report(), который вызывается строго каждую 1 мс по аппаратному прерыванию от UART-потока:
6. Проверяется доступность макро-дельт в буфере renderer_stream.
7. Если буфер содержит данные, выполняется стандартный шаг рекуррентной сети: renderer_stream.step().
8. Если буфер пуст (например, из-за сетевого лага визуального пайплайна или нахождения в мертвой зоне), выполняется холостой прогон (idle run) с нулевым макро-входом. Встроенный Sigma-Delta модулятор плавно гасит кинетическую энергию, создавая логнормальный хвост, а GRU продолжает накладывать фоновый тремор.
9. Счетчик времени с момента планирования инкрементируется: self.time_since_plan_ms += 1.
10. Метод возвращает квантованные дельты $\Delta X, \Delta Y$ в формате целочисленного типа (int8), готовые для формирования физического пакета USB HID.
    В случае смены цели или глубокого респауна агента вызывается метод reset(). Он выполняет мягкую очистку renderer_stream (сброс аккумуляторов Sigma-Delta без полной очистки весов тремора $h_{80}$), обнуляет вектор self.last_analytical_vel = [0, 0] и синтезирует стартовый Гауссовский микрошум для обеспечения бесшовного старта (zero-start kinematics) при следующем вызове визуального кадра.
    Данная архитектура устраняет все эвристические компоненты исходного ContinuousPlanner, перенося фокус на математически строгий Receding Horizon Control на базе аналитических свойств ProDMP 1 . Она минимизирует транспортные задержки, гарантирует биомеханическую чистоту выходного потока и обеспечивает абсолютную устойчивость при отслеживании высокодинамичных целей в замкнутом контуре 3D-симуляции.
