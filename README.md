# SUSNN:Self-Unifying Spiking Neural Network
# 自统一脉冲神经网络

[论文 DOI](https://doi.org/10.5281/zenodo.22195346)
[Paper DOI badge](https://doi.org/10.5281/zenodo.22195346)

[文档](https://gmunitx.com/index.php/2026/08/06/%e8%87%aa%e7%bb%9f%e4%b8%80%e8%84%89%e5%86%b2%e7%a5%9e%e7%bb%8f%e7%bd%91%e7%bb%9c%e8%ae%be%e8%ae%a1%e6%96%87%e6%a1%a3/)
[Document](https://gmunitx.com/index.php/2026/08/06/%e8%87%aa%e7%bb%9f%e4%b8%80%e8%84%89%e5%86%b2%e7%a5%9e%e7%bb%8f%e7%bd%91%e7%bb%9c%e8%ae%be%e8%ae%a1%e6%96%87%e6%a1%a3/)

# 中文

一个基于三维空间坐标的脉冲神经网络，通过最小化预测误差驱动学习，具备天然的具身交互需求。

---

## 设计哲学

### 智能即误差最小化

本系统假设智能是一个**最小化输入与预测之间误差**的系统。在该设计下，系统会产生一种"天然好奇心"——为了降低误差，它必须主动获取足够的信息来更好地预测外部世界。

### 天然的具身需求

设计中并不直接对动作神经元传递误差信号，因此系统必须依赖自己的动作对外部世界的影响来学习操作，形成间接的闭环控制回路。

### 语言无特殊优先级

不同于传统方式从符号出发转向具身，本系统并未预留专门的文字接口。语言并不存在特殊优先级，而是从图像、声音等多种方式融合学习，即使用类人方式。

---

## 核心脉冲引擎

### 网络空间结构

引擎是一个具有三维空间坐标的脉冲神经网络，分为三个功能区域：

- **第一面**：接收外部输入
- **第二面**：对外输出预测，并接收误差和放置动作神经元
- **中间神经网络**：连接第一面与第二面

第一面的接收输入神经元和第二面的输出预测神经元是对应的，但动作神经元并不与第一面对应。

#### 初始化方式

中间神经网络的神经元初始空间位置可来源于**宇宙大尺度星图（星系团分布）的坐标缩放**。这种初始化方式使网络天然同时具备：

- **局部聚集性**（有序柱状结构）
- **全局散落性**（无序背景）

避免了随机初始化的空洞和规则网格的过度对称，是天然的类人脑结构。

#### 连接约束

每个神经元拥有固定的连接半径，超出该半径的神经元不具备建立突触连接的条件。

---

### 神经元模型

每个神经元采用积分放电模型的变体，核心规则如下：

- **积分**：膜电位持续累积来自外部输入和其他神经元的脉冲输入。
- **放电与减法重置**：当膜电位达到当前放电阈值时，神经元发放脉冲，随后膜电位直接减去当前阈值，剩余差值保留为下一轮积分的起点。这种重置方式保留了神经元的惯性，同时天然形成放电后的类似不应期的状态，防止不降低膜电位导致的过度放电。
- **动态阈值**：每个神经元的放电阈值并非固定常数，而是根据该神经元过去一段时间内膜电位的滑动窗口平均值实时动态调整。神经元活跃时阈值自适应抬高，沉寂时阈值降低，从而维持网络整体的发放率稳态。

---

### 时间步的定义

一个时间步 = 完成一次对所有神经元的完整遍历（即一轮全空间扫描）。

每一轮计算中，每个神经元恰好被访问一次，执行一次状态更新。该时间步是引擎内部最基本的时间单位，所有可塑性规则（STDP、动态阈值、睡眠期的结构统计）均基于这个时间步进行计数和计算。

---

### 基础运行机制

引擎内部维持一个不停止的扫描过程，按照一定的顺序访问所有神经元并进行计算。这是一种应对硬件资源不足的方式，资源足够时也可以直接一次性计算，但由于可以理解为被同时扫描，下文中仍用"扫描过程"代指该过程。

这个扫描过程是引擎最底层的运行机制，在清醒态和睡眠态中持续运行，不受状态切换的影响。

每当访问到某个神经元时，该神经元执行以下操作：

1. 将外部注入该位置的模拟信号（如有）直接叠加到膜电位上；
2. 将从其他神经元向该神经元的连接发来的、已在缓存中等待的脉冲电荷累加到膜电位上；
3. 判断膜电位是否达到当前阈值：
   - 若达到，发放脉冲，将脉冲按连接权重放入缓存队列（等待下一轮扫描到目标神经元时传递），随后执行减法重置；
   - 若未达到，仅做积分，不动作；
4. 根据更新后的膜电位，刷新该神经元的滑动窗口平均值，并调整其动态阈值。

脉冲在当前神经元发放后，不立刻作用于目标，而是暂存于缓存，等待在下一轮扫描过程访问目标神经元时再施加。该过程为完整的一个时间步。

---

### 清醒态

在此状态下，引擎执行以下操作：

- **基础扫描与脉冲传导**：扫描器持续运行，所有神经元按上述操作正常更新。
- **STDP（脉冲时间依赖可塑性）**：每对前后发放的神经元，根据它们放电的时间差，实时调整两者之间已有突触连接的权重。
- **动态阈值更新**：每个神经元的放电阈值随其膜电位滑动窗口持续自适应调节。

> 清醒态不涉及连接的生成和剪枝。已有连接的权重可以变化，但不会生成新连接，也不会删除旧连接。

---

### 睡眠态

进入睡眠后，引擎的基础扫描与脉冲传导机制完全不变。

睡眠态与清醒态的唯一区别在于增加了结构可塑性操作：

- **生长新连接**：对于满足连接半径条件且符合共同放电规律（Hebb痕迹）的神经元对，生成新的突触连接。新连接的初始权重设为一个较小的正值。
- **剪枝冗余连接**：删除那些权重已衰减至接近于零的冗余连接。

总结如下：

| 状态 | 操作 |
|------|------|
| 清醒态 | 基础扫描 + STDP（权重调节）+ 动态阈值 |
| 睡眠态 | 基础扫描 + STDP（权重调节）+ 动态阈值 + 结构生长与剪枝 |

> 睡眠态期间并不额外统计共同放电历史，而是依赖系统自身不停息的运行和外部输入。两者的底层完全相同，睡眠仅是在其之上开启了额外的结构重塑性操作。

---

### 睡眠态的触发与退出

睡眠态的进入和退出均由外部程序手动控制，暂不内置自动触发策略。睡眠时长的设定（即睡眠阶段持续多少个时间步）亦留待实验研究确定。接口层面仅提供"进入睡眠"和"退出睡眠"的切换控制，不预设具体参数。

---

### 引擎的标准化接口

引擎对外提供以下标准交互方式：

- **向第一面传入信号**：外部系统将一整面模拟强度图（每个值在 -1 到 1 之间）一次性注入第一面的所有神经元。每个位置的强度值直接叠加到对应神经元的当前膜电位上。
- **从第二面读取输出**：外部系统一次性读取第二面所有神经元的当前实时膜电位值（连续标量），作为网络对当前输入的综合预测输出。
- **向第二面注入误差**：外部系统将计算得到的误差信号（同样为 -1 到 1 的整面强度图）一次性注入第二面除动作神经元外的神经元，直接叠加到其膜电位上。
> 引擎内部不设"收敛判定"或"步长等待"。外部系统按照自身的采样频率随时进行整面读写，引擎始终在后台持续运行其扫描过程。

---

## 外围交互回路（引擎外部应用层）

引擎本身不关心信号的具体物理含义。所有感官编码、误差计算和执行器驱动均在引擎外部实现，通过上述接口与引擎交互。

### 感官编码

各类物理信号在外部被编码为 -1~1 的强度图：

- **视觉**：将图像拆分为 RGB 三个通道，每个像素对应第一面的一个神经元位置，传入该像素对应颜色通道的归一化强度。
- **听觉**：对音频信号进行频谱分解，将各频段的能量强度映射至 -1~1，传入第一面对应位置的神经元。
- **触觉**：将各触觉压力感受器的物理读数线性映射至 -1~1，传入第一面相应位置的神经元。
- 气味、温度等都可以用类似的方式编码。

---

### 预测误差闭环

外部主控程序按照固定的采样周期运行以下循环：

1. 从第二面读取当前的预测输出膜电位；
2. 获取输入，将输入和预测逐位置相减（误差值 = 预测值 - 真实值），得到误差图（值域仍在 -1~1 内，图仅代表了一种二维结构，不代表是视觉信号）；
3. 将该误差图一次性注入第二面；
4. 将当前时刻的输入信号注入第一面，正误差会向导致该神经元更容易放电从而触发反向的STDP从而削减连接权重降低预测值，反之则会导致连接增强提升预测值；
5. 进入下一轮循环。

> 误差直接回传，无需进行脉冲编码，因为引擎的接口原生接受 -1~1 的连续模拟值。

---

### 动作执行扩展

在网络的第二面，可新建若干不参与误差回传的神经元作为动作神经元。外部系统读取这些神经元的放电，将其映射为控制指令，例如：

- 自动驾驶场景下某个神经元放电则控制车轮左转1度
- 机器人场景下控制扬声器（作为"声带"）发出某个频率的声音

这些动作神经元本身不直接接收误差信号，但它们输出的动作会改变外部环境，进而影响下一时刻传入第一面的输入，因此通过外部世界间接形成了完整的闭环控制回路。



# English

A spiking neural network structured by 3D spatial coordinates, driven by prediction-error minimization, with intrinsic embodied interaction needs.

---

## Design Philosophy

### Intelligence as Error Minimization

This system assumes that intelligence is a mechanism that **minimizes the error between input and prediction**. Under this design, the system naturally exhibits a form of "curiosity"—to reduce error, it must actively acquire sufficient information to better predict the external world.

### Intrinsic Embodiment Requirement

The design does not directly transmit error signals to motor neurons. Therefore, the system must rely on its own actions and their effects on the external world to learn manipulation, forming an indirect closed-loop control circuit.

### No Special Priority for Language

Unlike conventional approaches that start from symbols and then move toward embodiment, this system does not reserve a dedicated textual interface. Language holds no special priority; instead, it learns through the fusion of multiple modalities—such as vision and sound—in a human-like manner.

---

## Core Spiking Engine

### Network Spatial Structure

The engine is a spiking neural network with 3D spatial coordinates, divided into three functional regions:

- **First face**: Receives external input.
- **Second face**: Outputs predictions, receives errors, and hosts motor neurons.
- **Intermediate neural network**: Connects the first face to the second face.

The input neurons on the first face and the prediction-output neurons on the second face are correspondingly paired, but motor neurons do not correspond to the first face.

#### Initialization Method

The initial spatial positions of neurons in the intermediate network can be derived from **scaled coordinates of large-scale cosmic galaxy distributions**. This initialization endows the network with two inherent properties simultaneously:

- **Local clustering** (ordered column-like structures)
- **Global scattering** (disordered background)

This avoids the hollow regions of random initialization and the excessive symmetry of regular grids, providing a naturally brain-like structure.

#### Connection Constraint

Each neuron has a fixed connection radius. Neurons beyond this radius cannot form synaptic connections.

---

### Neuron Model

Each neuron uses a variant of the integrate-and-fire model, with the following core rules:

- **Integration**: Membrane potential continuously accumulates spike inputs from external sources and other neurons.
- **Fire and subtractive reset**: When membrane potential reaches the current firing threshold, the neuron fires a spike, and then the threshold is directly subtracted from the membrane potential. The remaining surplus is retained as the starting point for the next integration cycle. This reset method preserves neuronal inertia and naturally creates a refractory-like state after firing, preventing excessive discharges that would occur if the membrane potential were not reduced.
- **Dynamic threshold**: The firing threshold of each neuron is not fixed. Instead, it is dynamically adjusted in real time based on the sliding-window average of that neuron's membrane potential over the recent past. The threshold adaptively rises when the neuron is active and falls when it is silent, thus maintaining a stable overall firing rate across the network.

---

### Definition of a Time Step

One time step = completing one full traversal of all neurons (i.e., one round of full-space scanning).

In each round, every neuron is visited exactly once and performs one state update. This time step is the engine's most fundamental temporal unit. All plasticity rules (STDP, dynamic thresholds, and structural statistics during sleep) are counted and computed based on this time step.

---

### Basic Operating Mechanism

The engine maintains a non‑stop scanning process that visits and computes neurons in a fixed order. This is a way to cope with limited hardware resources; with sufficient resources, computation could be done all at once. However, since the process can be conceptually treated as simultaneous scanning, it is referred to as "scanning" throughout.

This scanning mechanism is the engine's most fundamental operating layer. It runs continuously during both wake and sleep states, unaffected by state transitions.

Whenever a neuron is visited, it performs the following operations:

1. Any external analog signal injected at its location (if present) is directly added to its membrane potential.
2. Pending spike charges from other neurons' connections (already in the buffer) are accumulated into the membrane potential.
3. It checks whether the membrane potential has reached the current threshold:
   - If yes, it fires a spike, places the spike into the outgoing buffer queue according to connection weights (to be delivered when the target neuron is visited in the next round), and then performs the subtractive reset.
   - If not, it only integrates and takes no firing action.
4. Based on the updated membrane potential, it refreshes the sliding-window average and adjusts its dynamic threshold.

Spikes fired by a neuron are not immediately applied to targets; they are temporarily stored in a buffer and applied only when the target neuron is visited in the next scanning round. This entire process constitutes one complete time step.

---

### Wake State

In this state, the engine performs:

- **Basic scanning and spike conduction**: The scanner runs continuously, and all neurons update normally as described above.
- **STDP (Spike-Timing-Dependent Plasticity)**: For each pair of pre‑ and post‑synaptic neurons, the weight of their existing synaptic connection is adjusted in real time based on the temporal difference between their spikes.
- **Dynamic threshold updates**: Each neuron's firing threshold continuously adapts according to its membrane‑potential sliding window.

> In the wake state, no connection generation or pruning occurs. Existing connection weights may change, but no new connections are created and no old ones are deleted.

---

### Sleep State

Upon entering sleep, the engine's basic scanning and spike conduction mechanisms remain completely unchanged.

The sole difference between sleep and wake states is the addition of structural plasticity operations:

- **Growth of new connections**: For neuron pairs that satisfy the connection‑radius condition and exhibit co‑activation patterns (Hebbian traces), new synaptic connections are formed. Initial weights are set to a small positive value.
- **Pruning of redundant connections**: Connections whose weights have decayed to near zero are removed.

Summary:

| State | Operations |
|-------|------------|
| Wake  | Basic scanning + STDP (weight adjustment) + Dynamic thresholds |
| Sleep | Basic scanning + STDP (weight adjustment) + Dynamic thresholds + Structural growth and pruning |

> During sleep, co‑activation histories are not separately tallied. Instead, the system relies on its own uninterrupted operation and external inputs. The underlying mechanisms are identical in both states; sleep merely enables additional structural‑plasticity operations on top.

---

### Sleep State Trigger and Exit

The entry to and exit from the sleep state are manually controlled by external programs; no automatic triggering strategy is built in. The duration of sleep (i.e., how many time steps the sleep phase lasts) is left for experimental determination. At the interface level, only "enter sleep" and "exit sleep" switching controls are provided, without preset parameters.

---

### Standardized Engine Interface

The engine provides the following standard interaction methods:

- **Signal input to the first face**: The external system injects a full‑face analog intensity map (each value between -1 and 1) into all neurons on the first face at once. Each positional intensity value is directly added to the corresponding neuron's current membrane potential.
- **Output reading from the second face**: The external system reads, all at once, the current real‑time membrane potentials (continuous scalars) of all neurons on the second face, as the network's comprehensive prediction output for the current input.
- **Error injection to the second face**: The external system injects a computed error signal (also a full‑face intensity map, values in -1 to 1) into all neurons on the second face except the motor neurons, directly adding it to their membrane potentials.

> The engine does not include internal "convergence detection" or "step‑waiting" mechanisms. The external system performs full‑face reads and writes at its own sampling frequency, while the engine continuously runs its scanning process in the background.

---

## Peripheral Interaction Loop (External Application Layer)

The engine itself does not care about the specific physical meaning of signals. All sensory encoding, error computation, and actuator driving are implemented outside the engine and interact with it through the above interfaces.

### Sensory Encoding

Various physical signals are encoded externally as intensity maps in the range -1 to 1:

- **Vision**: An image is split into RGB channels. Each pixel corresponds to a neuron position on the first face, and the normalized intensity of the corresponding color channel is fed there.
- **Audition**: An audio signal is decomposed into its frequency spectrum. The energy intensity of each frequency band is mapped to -1 to 1 and fed to the corresponding positions on the first face.
- **Touch**: Readings from tactile pressure sensors are linearly mapped to -1 to 1 and fed to corresponding first‑face neurons.
- Odor, temperature, and other modalities can be encoded similarly.

---

### Prediction Error Closed Loop

The external main program runs the following loop at a fixed sampling period:

1. Read the current prediction output membrane potentials from the second face.
2. Acquire the current input, compute the error map element‑wise as (prediction - ground truth), obtaining an error map (values still within -1 to 1; note that the map is a 2D structure but does not necessarily represent a visual signal).
3. Inject this error map into the second face.
4. Inject the current input signal into the first face. Positive errors make the corresponding neurons more likely to fire, which triggers reversed STDP, reducing connection weights and thus lowering future predictions; negative errors have the opposite effect, strengthening connections and raising predictions.
5. Proceed to the next loop iteration.

> Error is fed back directly without spike encoding, because the engine's interface natively accepts continuous analog values in the range -1 to 1.

---

### Motor Execution Extension

On the second face of the network, a number of new neurons can be designated as motor neurons that do not participate in error feedback. The external system reads the spike activity of these neurons and maps it to control commands, for example:

- In autonomous driving, a neuron firing might command a 1‑degree left turn of the wheels.
- In a robot, it might command a speaker (as a "vocal apparatus") to emit a sound at a certain frequency.

These motor neurons do not directly receive error signals. However, their output actions alter the external environment, which in turn affects the inputs fed to the first face at the next moment. Thus, a complete closed‑loop control circuit is formed indirectly through the external world.
