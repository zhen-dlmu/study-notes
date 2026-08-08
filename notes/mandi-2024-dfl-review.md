---
title: Decision-Focused Learning 系统阅读笔记
date: 2026-08-08
tags: [DFL, 论文阅读, 运筹优化, 机器学习]
ai_generated: true
---

Exit code: 0
Wall time: 0.9 seconds
Output:
# Decision-Focused Learning 系统阅读笔记

> 阅读对象：Mandi et al. (2024), *Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities* [1]  
> 适用读者：刚进入 Decision-Focused Learning（DFL）领域的研究人员  
> 笔记目标：建立问题定义、方法分类、实验认识和后续文献阅读路线

## 1. 论文定位

这篇论文从机器学习视角系统梳理 Decision-Focused Learning，并完成了三项主要工作：

1. 给出 Predict-Then-Optimize 与 DFL 的统一问题定义；
2. 将梯度型 DFL 方法划分为四类，同时总结无梯度方法；
3. 在 7 个优化任务上比较 11 种方法，并总结其适用条件、计算成本和未来方向。

全文的核心问题是：

> 当机器学习模型的预测不是最终目的，而是后续优化决策的输入时，应当如何训练预测模型，使最终决策质量尽可能高？

## 2. 从 Predict-Then-Optimize 到 DFL

### 2.1 问题结构

设：

- $z$：决策前可观察到的特征；
- $c$：优化问题中真实但未知的参数；
- $m_\omega$：参数为 $\omega$ 的预测模型；
- $\hat c=m_\omega(z)$：模型预测的优化参数；
- $\mathcal F$：可行域；
- $f(x,c)$：在参数 $c$ 下，决策 $x$ 的目标函数。

参数化优化问题为：

$$
x^*(c)=\arg\min_{x\in\mathcal F} f(x,c).
$$

实际决策流程是：

$$
z \longrightarrow \hat c=m_\omega(z)
\longrightarrow x^*(\hat c)
\longrightarrow f(x^*(\hat c),c).
$$

其中 $x^*(c)$ 是已知真实参数时的全信息最优决策，$x^*(\hat c)$ 是根据预测参数得到的处方决策。

### 2.2 Prediction-Focused Learning

传统 Prediction-Focused Learning（PFL）通常最小化预测误差，例如：

$$
\min_\omega \frac{1}{N}\sum_{i=1}^N
\left\|m_\omega(z_i)-c_i\right\|_2^2.
$$

该范式隐含假设：参数预测越准确，最终决策就越好。但这一假设通常不完全成立，因为不同参数误差对决策的影响并不相同。

例如，在最短路径问题中：

- 一条永远不会进入最优路径的边即使预测误差很大，也可能不影响决策；
- 一条关键边上的微小预测误差，却可能改变整条最短路径。

### 2.3 Decision-Focused Learning

DFL 直接根据最终决策质量训练预测模型。最常见的任务损失是 regret：

$$
R(\hat c,c)
=f(x^*(\hat c),c)-f(x^*(c),c).
$$

因此 DFL 的经验风险最小化问题可以写成双层优化：

$$
\begin{aligned}
\min_\omega \quad &
\frac{1}{N}\sum_{i=1}^N
\left[f(x^*(\hat c_i),c_i)-f(x^*(c_i),c_i)\right]\\
\text{s.t.}\quad &\hat c_i=m_\omega(z_i),\\
&x^*(\hat c_i)=\arg\min_{x\in\mathcal F}f(x,\hat c_i).
\end{aligned}
$$

DFL 与 PFL 的根本区别是：

| 范式 | 直接优化的对象 | 模型关注点 |
|---|---|---|
| PFL | 参数预测误差 | 是否准确恢复 $c$ |
| DFL | 决策损失或 regret | $\hat c$ 是否产生高质量决策 |

DFL 不一定要求 $\hat c$ 是对 $c$ 的无偏或逐项准确估计。只要预测产生的决策好，它就可能是有效的决策型预测。

## 3. DFL 的核心技术困难

神经网络训练需要计算：

$$
\frac{dL}{d\omega}
=\frac{dL}{dx^*(\hat c)}
\frac{dx^*(\hat c)}{d\hat c}
\frac{d\hat c}{d\omega}.
$$

其中：

- $dL/dx^*$ 通常容易获得；
- $d\hat c/d\omega$ 可以由自动微分计算；
- $dx^*/d\hat c$ 是 DFL 的核心困难。

### 3.1 优化映射是隐式定义的

$x^*(c)$ 来自一个优化问题，不一定存在显式闭式表达，因此不能像普通神经网络层一样直接求导。

### 3.2 离散决策导致零梯度和不连续

对于 LP、ILP 和许多组合优化问题，成本向量到最优解的映射通常是分段常数的：

- 在同一决策区域内改变成本，最优解不变，梯度为零；
- 跨越决策边界时，最优解跳变，梯度不存在。

因此，即使能计算“精确梯度”，它也可能无法提供有效的学习方向。

### 3.3 训练计算成本高

训练过程中可能要为每个样本、每个 epoch 重复求解优化问题。对于 NP-hard 的组合优化任务，求解成本可能成为实际瓶颈。

由此可以把 DFL 的研究目标概括为：

> 构造一个与真实决策损失足够一致、能够提供有效训练信号、同时计算成本可接受的替代梯度或替代损失。

## 4. 方法分类总览

论文首先将 DFL 分为梯度型与无梯度型方法。梯度型方法进一步分为四类。

| 方法类别 | 核心思想 | 代表方法 | 主要优势 | 主要局限 |
|---|---|---|---|---|
| 优化映射的解析微分 | 对最优性条件或求解算法进行隐式/显式微分 | OptNet、cvxpylayers、solver unrolling | 梯度具有明确数学来源；部分方法可学习约束参数 | 原始离散映射的精确梯度仍可能无用 |
| 优化映射的解析平滑 | 给目标或约束加入平滑正则，再微分平滑问题 | QPTL、HSD、entropy、log-barrier | 适合 LP；可利用成熟凸优化微分技术 | 存在平滑偏差；ILP 松弛质量很关键 |
| 随机扰动平滑 | 对成本向量加噪声，用附近最优解构造平滑映射 | DBB、DPO、FY、I-MLE | 可调用黑箱组合优化器 | 梯度方差、温度和扰动尺度敏感 |
| 可微代理损失 | 用容易优化的任务损失替代真实 regret | SPO+、NCE、MAP、LTR losses | 避免直接计算 $dx^*/d\hat c$；部分方法有理论保证 | 常依赖特定损失、标签和问题结构 |

## 5. 四类梯度型 DFL 方法

### 5.1 优化映射的解析微分

#### OptNet 与 KKT 微分

OptNet 将凸二次规划作为神经网络层，并对 KKT 最优性条件进行隐式微分 [2]。对于满足适当正则条件的光滑凸优化问题，可以通过求解由 KKT 条件导出的线性系统获得梯度。

#### 锥规划与 cvxpylayers

Agrawal 等将广泛的凸优化问题规范化为锥规划，并对锥规划求解映射进行微分 [3,4]。cvxpylayers 因此能够把符合 disciplined convex programming 规则的凸优化问题嵌入 PyTorch 等框架。

#### Solver unrolling

另一种方式是展开迭代求解器的若干步，并通过整个迭代计算图反向传播。它实现直接，但需要保存中间状态，可能带来显著内存与时间开销。

#### 适用性判断

这类方法适合：

- 连续、光滑、凸的优化问题；
- 需要对目标和约束参数同时求导；
- 优化层位于深度模型中间，而非只在流水线末端。

它不直接解决离散映射“几乎处处零梯度”的问题。

### 5.2 优化映射的解析平滑

#### QPTL

QPTL 将线性规划目标

$$
\min_{x\in\mathcal F}c^\top x
$$

替换为平滑的二次目标：

$$
\min_{x\in\mathcal F}
c^\top x+\mu\|x\|_2^2.
$$

加入严格凸正则后，解映射变得连续，可以使用 QP 微分技术训练；测试时再求解原优化问题 [5]。

平滑系数 $\mu$ 控制一个重要权衡：

- $\mu$ 较大：梯度更平滑，但对原问题的近似偏差更大；
- $\mu$ 较小：更接近原问题，但梯度可能不稳定。

#### HSD 与 log-barrier

Mandi 与 Guns 通过 homogeneous self-dual embedding 和内点法产生可微的 LP 近似 [6]。该方法利用提前停止的 barrier 迭代获得平滑代理。

#### ILP 的连续松弛

对于 ILP，可以先删除整数约束，得到 LP relaxation，再进行平滑微分。其有效性取决于松弛是否能准确代表整数问题。需要重点检查：

$$
\text{integrality gap}
\quad\text{以及}\quad
\|x_{\mathrm{LP}}-x_{\mathrm{ILP}}\|.
$$

### 5.3 随机扰动平滑

#### DBB

Differentiation of Blackbox Combinatorial Solvers（DBB）比较扰动前后的最优解，形成替代梯度 [7]：

$$
x^*\!\left(
\hat c+\delta\frac{dL}{dx^*}
\right)-x^*(\hat c).
$$

DBB 只需要求解器返回最优解，不需要访问求解器内部。因此可以使用 ILP、CP、SAT、Dijkstra 或商业求解器。

#### DPO 与 Fenchel-Young

Differentiable Perturbed Optimizers（DPO）对成本加入随机噪声，对多个扰动最优解求平均，由此得到平滑近似 [8]。Fenchel-Young loss 进一步构造了与这种扰动优化器相适配的损失。

#### I-MLE

I-MLE 将 perturb-and-MAP 与隐式最大似然估计结合，通过比较两组带噪优化解形成梯度估计 [9]。

#### 关键超参数

随机扰动方法通常需要调节：

- 扰动步长 $\delta$；
- 温度或噪声强度 $\epsilon$；
- Monte Carlo 样本数。

扰动过小可能使最优解不变；扰动过大又会偏离原问题。噪声减小时，估计偏差可能下降，但方差可能上升。

### 5.4 可微代理损失

#### SPO+

Smart Predict-Then-Optimize（SPO）是 DFL 的代表性工作 [10]。它使用真实 regret 的凸代理上界 SPO+，常用次梯度为：

$$
x^*(c)-x^*(2\hat c-c).
$$

SPO+ 的突出特点包括：

- 不直接求 $dx^*/d\hat c$；
- 适用于预测参数线性进入目标函数的问题；
- 在一定分布假设下具有 Fisher consistency；
- 已有风险界和泛化界研究 [11,12]。

#### Contrastive loss 与 solution cache

Mulamba 等通过 NCE、MAP 类对比损失和 solution cache 降低训练时反复调用求解器的成本 [13]。

缓存方法以有限候选集 $\mathcal S\subset\mathcal F$ 代替完整可行域，在缓存中查找预测成本下的最佳解。其性能取决于缓存是否覆盖决策边界附近和高质量区域中的关键解。

#### Learning to Rank

Mandi 等将 DFL 解释为对可行解进行排序的问题，提出 Pairwise、Pairwise Difference 和 Listwise 损失 [14]。模型的目标不再是精确恢复成本，而是让 $\hat c$ 与 $c$ 对候选解产生相近的优劣排序。

#### 与可微优化层的区别

- 可微优化层近似的是优化映射或其导数，可放在神经网络中间，并原则上配合不同下游损失；
- SPO+ 等代理损失直接针对特定任务损失，通常要求优化位于 Predict-Then-Optimize 流水线末端。

## 6. 无梯度 DFL

无梯度方法不依赖神经网络的反向传播，代表路线包括：

- 直接按 regret 训练决策树或树集成 [15]；
- 把线性预测模型的训练问题转化为 MILP；
- coordinate descent；
- dynamic programming、divide-and-conquer 与 branch-and-learn。

这类方法可以直接处理不连续 regret，部分小规模问题还能得到全局最优训练解。但它们通常难以扩展到深度网络和大规模复杂优化问题。

## 7. 基准实验

### 7.1 七个任务

论文使用以下任务：

1. 合成 $5\times5$ 网格最短路径；
2. 投资组合优化；
3. Warcraft 图像最短路径；
4. 能源成本感知调度；
5. 背包问题；
6. 带多样性约束的二部图匹配；
7. 子集选择。

### 7.2 十一种比较方法

- Prediction-Focused（PF）；
- SPO；
- DBB；
- I-MLE；
- Fenchel-Young（FY）；
- HSD；
- QPTL；
- Listwise；
- Pairwise；
- Pairwise Difference；
- MAP。

### 7.3 评价指标

主要指标为平均 relative regret：

$$
\frac{1}{N_{\mathrm{test}}}
\sum_{i=1}^{N_{\mathrm{test}}}
\frac{
c_i^\top\left(x^*(\hat c_i)-x^*(c_i)\right)
}{c_i^\top x^*(c_i)}.
$$

实践中，如果最优目标值接近零或可能为负，relative regret 可能不稳定。因此建议同时报告：

- absolute regret；
- relative regret；
- 原始任务目标值；
- 参数预测误差；
- 训练和推理时间。

## 8. 实验结论

### 8.1 不存在全面最优的方法

没有一种 DFL 方法在所有任务上表现最好。方法表现取决于目标形式、约束结构、连续或离散变量、松弛质量、求解成本和模型设定。

### 8.2 SPO 是稳健的默认基线

SPO 虽然不总是第一，但跨任务表现稳定。因此，新 DFL 研究至少应比较：

- PF/MSE；
- SPO+；
- 一种与问题结构匹配的可微优化或黑箱方法。

仅证明新方法优于 MSE，通常不足以证明其相对于已有 DFL 方法的优势。

### 8.3 QPTL 依赖 LP 松弛质量

当 LP relaxation 是原 ILP 的良好近似时，QPTL 可能非常有效；论文中其在二部图匹配上表现突出。调度问题中 LP 解与整数解偏差较大，QPTL 表现较差。

### 8.4 MAP 与排序损失具有计算优势

当求解概率 $p_{\mathrm{solve}}$ 较低时，MAP、Listwise、Pairwise 和 Pairwise Difference 的训练明显更快。MAP 在多数任务上仍能取得接近 SPO 的 regret，适合单次求解昂贵且可维护候选解缓存的场景。

### 8.5 黑箱求解器兼容性影响可扩展性

QPTL 和 HSD 因计算资源限制未能在 Warcraft 任务上运行。只依赖最优解的黑箱型方法可以使用专用求解器，往往更容易扩展。

### 8.6 DFL 不保证优于 PFL

在带二次约束的投资组合问题中，I-MLE、FY、DBB 和 QPTL 都可能差于 PF。直接以决策为训练目标，并不自动保证更低的测试 regret；代理梯度的偏差、方差和结构不匹配都可能损害性能。

## 9. 面对新问题时如何选择方法

可以按以下顺序判断：

1. **预测参数出现在哪里？** 若在约束中，需要额外考虑不可行性和 recourse；本文多数方法主要处理目标参数。
2. **目标关于预测参数是否线性？** SPO、DBB 和很多扰动方法主要围绕线性目标建立。
3. **决策是否连续且问题是否凸？** 若是，可优先考虑 cvxpylayers 或 KKT 微分。
4. **若决策离散，LP 松弛是否紧？** 若紧，可尝试 QPTL/HSD；若不紧，应谨慎使用松弛梯度。
5. **是否只有黑箱求解器？** 可优先考虑 SPO、DBB、I-MLE 或其他 perturbation 方法。
6. **单次求解是否昂贵？** 可考虑 solution cache、MAP、LTR losses 或代理求解器。
7. **训练数据是否包含真实成本 $c$？** 如果只有最优决策标签，依赖真实成本的 regret 代理可能无法直接使用。
8. **是否需要把优化层放在网络中间？** 若是，应采用真正的可微优化层，而非只适用于最终 regret 的代理损失。

## 10. 论文的适用边界

这篇综述主要关注：

- 监督数据 $(z,c)$；
- 不确定参数出现在目标函数中；
- 可行域通常已知且固定；
- 单阶段 Predict-Then-Optimize；
- LP、ILP 和线性成本组合优化问题。

以下问题仍未被充分覆盖：

- 不确定约束及预测诱发的不可行性；
- 非线性离散目标；
- 风险敏感和分布鲁棒决策；
- 多阶段决策；
- 分布偏移与跨优化实例泛化；
- 多模态输入；
- 大规模训练中的系统与求解成本。

## 11. 未来研究方向

论文提出的主要方向包括：

1. 跨相关优化任务泛化；
2. 非线性目标与离散变量；
3. 鲁棒、风险敏感 DFL；
4. 零阶或 score-function 梯度及方差缩减；
5. 从双层优化角度发展可扩展算法；
6. solution cache 和代理求解器的质量-效率权衡；
7. 更广泛的统计和优化理论保证；
8. 不确定约束；
9. 多阶段决策；
10. 多模态真实数据。

一个基础而重要的研究问题是：

> 在什么条件下，代理梯度或代理损失的下降方向与真实 regret 的下降方向一致？

## 12. 推荐阅读顺序

### 第一轮：理解问题

阅读 Abstract、Section 1、Sections 2.1-2.3，以及 Figures 1 和 3。目标是能够解释 PFL、DFL、regret 和双层优化结构。

### 第二轮：建立方法地图

阅读 Section 3 开头、Figure 4、各个 3.1.x 小节的首段与 Discussion，以及 Table 1。目标是能把新方法归入解析微分、解析平滑、随机扰动或代理损失。

### 第三轮：精读三个代表方法

- QPTL：理解平滑优化映射；
- DBB：理解黑箱替代梯度；
- SPO+：理解 regret 代理损失。

### 第四轮：理解实验规律

阅读 Section 5.2，重点是 Section 5.2.3，而不是记忆每幅实验图中的排名。

### 第五轮：形成研究问题

阅读 Section 6，把自己感兴趣的问题写成以下模板：

> 对于具有 ______ 结构的优化问题，现有 ______ 方法由于 ______ 存在局限。计划通过 ______ 改善，并在 ______ 指标上与 PF、SPO+ 和 ______ 比较。

## 13. 阅读自测

1. 为什么更低的 MSE 不一定产生更低的 regret？
2. 为什么 LP 的成本到最优解映射几乎处处梯度为零？
3. QPTL 中 $\mu$ 控制什么权衡？
4. DBB 为什么能够使用黑箱求解器？
5. 可微优化层与 regret 代理损失有什么区别？
6. SPO+ 的次梯度为什么需要求解 $x^*(2\hat c-c)$？
7. integrality gap 为什么会影响基于 LP relaxation 的训练？
8. solution cache 如何降低计算成本，又会引入什么近似误差？
9. 为什么 DFL 实验必须保留 PF 基线？
10. 为什么不能从这篇论文得出“DFL 总是优于 PFL”？

## 14. 核心结论

DFL 的本质不是简单地把求解器接在神经网络后面，而是在优化映射不光滑、真实 regret 难以求导且求解成本高昂的情况下，设计一个同时满足以下要求的学习信号：

- 与真实决策质量具有足够一致性；
- 能够提供稳定而有效的优化方向；
- 在目标问题规模上计算可承受。

方法之间最关键的差异，正是它们在**忠实度、梯度质量、通用性和计算成本**四者之间采取了不同折中。

## 参考文献

[1] Mandi, J., Kotary, J., Berden, S., Mulamba, M., Bucarey, V., Guns, T., & Fioretto, F. (2024). Decision-focused learning: Foundations, state of the art, benchmark and future opportunities. *Journal of Artificial Intelligence Research, 81*, 1623-1701.

[2] Amos, B., & Kolter, J. Z. (2017). OptNet: Differentiable optimization as a layer in neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (pp. 136-145). PMLR.

[3] Agrawal, A., Amos, B., Barratt, S., Boyd, S., Diamond, S., & Kolter, J. Z. (2019). Differentiable convex optimization layers. In *Advances in Neural Information Processing Systems 32*.

[4] Agrawal, A., Barratt, S., Boyd, S., Busseti, E., & Moursi, W. M. (2019). Differentiating through a cone program. *Journal of Applied and Numerical Optimization, 1*(2), 107-115.

[5] Wilder, B., Dilkina, B., & Tambe, M. (2019). Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization. In *Proceedings of the AAAI Conference on Artificial Intelligence* (pp. 1658-1665).

[6] Mandi, J., & Guns, T. (2020). Interior point solving for LP-based prediction+optimisation. In *Advances in Neural Information Processing Systems 33* (pp. 7272-7282).

[7] Pogančić, M. V., Paulus, A., Musil, V., Martius, G., & Rolinek, M. (2020). Differentiation of blackbox combinatorial solvers. In *International Conference on Learning Representations*.

[8] Berthet, Q., Blondel, M., Teboul, O., Cuturi, M., Vert, J.-P., & Bach, F. (2020). Learning with differentiable perturbed optimizers. In *Advances in Neural Information Processing Systems 33* (pp. 9508-9519).

[9] Niepert, M., Minervini, P., & Franceschi, L. (2021). Implicit MLE: Backpropagating through discrete exponential family distributions. In *Advances in Neural Information Processing Systems 34* (pp. 14567-14579).

[10] Elmachtoub, A. N., & Grigas, P. (2022). Smart “Predict, then Optimize”. *Management Science, 68*(1), 9-26.

[11] Liu, H., & Grigas, P. (2021). Risk bounds and calibration for a smart predict-then-optimize method. In *Advances in Neural Information Processing Systems 34* (pp. 22083-22094).

[12] El Balghiti, O., Elmachtoub, A. N., Grigas, P., & Tewari, A. (2019). Generalization bounds in the predict-then-optimize framework. In *Advances in Neural Information Processing Systems 32*.

[13] Mulamba, M., Mandi, J., Diligenti, M., Lombardi, M., Bucarey, V., & Guns, T. (2021). Contrastive losses and solution caching for predict-and-optimize. In *Proceedings of the 30th International Joint Conference on Artificial Intelligence* (pp. 2833-2840).

[14] Mandi, J., Bucarey, V., Tchomba, M. M. K., & Guns, T. (2022). Decision-focused learning: Through the lens of learning to rank. In *Proceedings of the 39th International Conference on Machine Learning* (pp. 14935-14947). PMLR.

[15] Elmachtoub, A., Liang, J. C. N., & McNellis, R. (2020). Decision trees for decision-making under the predict-then-optimize framework. In *Proceedings of the 37th International Conference on Machine Learning* (pp. 2858-2867). PMLR.

[16] Donti, P. L., Kolter, J. Z., & Amos, B. (2017). Task-based end-to-end model learning in stochastic optimization. In *Advances in Neural Information Processing Systems 30* (pp. 5484-5494).

[17] Kotary, J., Fioretto, F., Van Hentenryck, P., & Wilder, B. (2021). End-to-end constrained optimization learning: A survey. In *Proceedings of the 30th International Joint Conference on Artificial Intelligence* (pp. 4475-4482).

[18] Sadana, U., Chenreddy, A., Delage, E., Forel, A., Frejinger, E., & Vidal, T. (2024). A survey of contextual optimization methods for decision-making under uncertainty. *European Journal of Operational Research*.

[19] Tang, B., & Khalil, E. B. (2023). PyEPO: A PyTorch-based end-to-end predict-then-optimize library for linear and integer programming. arXiv preprint.

[20] Boyd, S. P., & Vandenberghe, L. (2014). *Convex Optimization*. Cambridge University Press.

