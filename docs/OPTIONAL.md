# c4tts 性能优化备选方案（RTF 进一步压缩）

> 目标：在已落地的优化之上，分析针对 Apple Silicon 的、可能更激进的 RTF 优化方案，
> 评估能否再提升 2–3×。本文档是分析与路线图，不是已实现清单。

## 已落地的优化（基线）

- 融合注意力 `mx::fast::scaled_dot_product_attention`（保持数值一致）
- `gelu_new` 用乘法代替 `mx::power`
- **T2S 投影权重 int8 量化（默认开启，`C4TTS_QUANT=8`）—— T2S 约 2×**
- 可选项：fp16 投影（`C4TTS_FP16`）、int4（`C4TTS_QUANT=4`，质量较差不推荐）、
  量化分组 `C4TTS_QUANT_GROUP`
- 服务端：音色 sqlite 持久化（默认音色库锚定到可执行文件目录，不随 CWD 变化）、
  音色条件特征 LRU 缓存（`--lrucache`，避免每次 TTS 重新提取）、
  MLX Metal 缓冲缓存上限 + 每请求 `clear_cache()`（内存不再随请求增长）

## 当前基线 Profile（int8 默认，9.6s 音频，RTF ≈ 0.57）

| 阶段 | 耗时 | 占比 | 性质 |
|---|---|---|---|
| **t2s**（GPT-2 24 层 AR 解码） | 2981 ms | **54%** | 自回归 ~330 步 × 24 层，~9 ms/步，**调度/延迟受限** |
| **s2a**（DiT flow-matching） | 1218 ms | 22% | Euler 一阶求解，25 步 × CFG(2×) = **50 次前向**，fp32 |
| **bigvgan**（声码器） | 1081 ms | 20% | 单次卷积栈前向，fp32 |
| prompt | 208 ms | 4% | 重复同音色时已被 LRU 缓存 |

> int8 已经把分布"摊平"——T2S 不再一家独大。

## 关键约束（先把数学讲清楚）

要整体再快 **2×**，必须大幅砍 T2S：即使 S2A + BigVGAN **归零**，上限也只有
`1 / (0.54 + 0.04) ≈ 1.7×`。所以 **T2S 是闸门**，2–3× 必须以 T2S 的算法级提速为核心，
再叠加另外两个阶段。

而 T2S 是 int8 化的自回归解码，已经带宽优化过，纯靠"再量化/再压"基本到顶。
**纯工程手段（无重训）的现实上限约 1.5–1.8×。真正的 2–3× 需要算法改变。**

---

## Tier A — 无需重训，约 1.4–1.7×（RTF 0.57 → ~0.35）

### A1. S2A 换更高阶 ODE 求解器 —— ✅ 已落地（AB2，opt-in）
当前默认仍是一阶 Euler @ 25 步。已加入二阶 Adams-Bashforth（`C4TTS_SOLVER=ab2`，
1 次/步、复用上一步速度场），**同步数下严格优于 Euler**，故可降步数保质量。

**实测（log-mel 距离，相位鲁棒；euler-25 为当前质量基准）：**
| 配置 | S2A 耗时 | vs euler-25 (RMSE) | 提速 |
|---|---|---|---|
| euler-25（默认） | 964 ms | 0（基准） | 1.0× |
| ab2-16 | 630 ms | 0.167 | ~1.5× |
| ab2-14 | 552 ms | 0.253 | ~1.75× |
| ab2-12 | 474 ms | 0.363 | ~2.0×（略降质量）|

结论：比文档最初设想的"12 步达 euler-25 质量"更保守——AB2-16 ≈ euler-25（~1.5×），
AB2-12 ~2× 但有小幅质量差。默认保持 euler-25（忠实原模型），opt-in 提速。
**度量教训：波形 cosine 不可靠（BigVGAN 把微小 mel 差异放大成相位差），必须用 mel 谱距离。**

### A2. ~~BigVGAN int8/fp16~~ —— ❌ 实测 fp16 无效，改走 int8（仅 S2A）
**实测结论（2026-06）：fp16 在本机 MLX/Metal 上不能加速 conv/这些阶段。**
做法：先修了 `nn.cpp` 的标量 dtype 串扰（`mx::array(eps, x.dtype())` + норм 在 fp32 统计、
返回输入 dtype），消除 churn 后再给 BigVGAN 开 fp16——结果 **BigVGAN fp16 反而更慢**
（890ms vs fp32 664ms），且波形 max_abs 0.108。原因：MLX 的 Metal **卷积**没有
fp16 加速路径（甚至内部上转），cast 开销反而拖累。这与早先 T2S fp16 仅 ~1.1×、S2A fp16
无收益一致。**该路已放弃并回退**（含 nn dtype 改动，因无 fp16 消费者）。

**真正有效的量化原语是 int8 `quantized_matmul`**（T2S 实测 2×），只适用于 **matmul 重**
的阶段：
- **S2A DiT**：其 "conv1/conv2/proj" 其实是 Linear，且按 T 帧是 M>1 的 GEMM —— 用
  int8 `quantized_matmul`（同 T2S proj 方案）有望 ~1.5–2×/前向，且与 A1 求解器（减前向数）
  正交叠加。**这是 S2A 计算侧的推荐路线。**
- **BigVGAN**：纯卷积，MLX 无 int8 conv —— 短期维持 fp32，要提速得上自定义 Metal kernel（D1）。

### A3. T2S `mx::compile`（定长 KV cache）+ GPU 端采样 + semantic_head int8
- **compile**：需要预分配定长 KV cache（之前用 shapeless 被 MLX 形状推断挡住：
  cache 的 `concatenate` 让维度变 symbolic，下游 `split`/`slice` 的 `output_shapes`
  推断失败）。正确做法是固定形状 + 位置掩码，标准（非 shapeless）compile 只 trace 一次。
  把每步 ~240 次 kernel dispatch 的逐元素胶水（layernorm/gelu/残差/bias）融合 → ~1.2–1.4×。
- **semantic_head**：是 `(8194, 1280)` fp32，每步读 ~42 MB 算全词表 logits；int8 化省一半带宽（小但近免费）。
- **GPU 端采样**：只同步选中的 token id（而非整词表回传）——收益偏小（实测 greedy 与
  sampling 的 t2s 耗时接近，说明 CPU 采样不是瓶颈，瓶颈是 GPU 计算 + dispatch）。

> Tier A 合计大约 **1.5×，到不了 2×**——因为 T2S 只动了 ~1.3×。

---

## Tier B — 轻量训练，可达 ~2–2.5×（这是 2–3× 的钥匙）

### B1. T2S 投机 / 多 token 解码（单点最高杠杆，已评估）
T2S 占 44–54%，profiling 证实其每步成本几乎全在 24 层前向（5.7ms/步），head 0.37ms、
采样 0.019ms 可忽略——所以唯一出路是减少/摊薄前向次数。

**已做评估（2026-06）：semantic token 流的可预测性。** 实测多条生成序列：
**token 100% 唯一、零 n-gram 重复、相邻重复率 0%**（高熵 VQ 码，每帧一个不同码）。结论：
- **免训练的 n-gram / prompt-lookup 投机：不可行**（草稿接受率 0%，无任何收益）。
- 只剩需训练的两条：
  - **Medusa 多头**：训练 t+2/t+3 轻量头（冻结主干，PyTorch 侧，~数 GPU 时）+ C++ 解码加
    树状验证。但 token 高熵 → t+2 可预测性存疑，接受率/加速**不确定**（可能远低于 LLM 上的 2×）。
  - **草稿模型投机**：训练一个更小的 T2S 草稿，同样受高熵限制。
- 工程量大（跨 PyTorch 训练 + C++ 树状/投机验证），收益因高熵而不确定。**性价比待定**，
  上之前应先小规模训练 Medusa 头、量一下 t+2 接受率再决定。

### B2. S2A CFG 间隔 —— ✅ 已落地（opt-in，无需训练）
profiling 证实：S2A 每步跑 2 次 DiT 前向（cond + uncond），第 2 次 ~占 S2A 一半。
不必蒸馏——直接用 **CFG 间隔**（`C4TTS_CFG_LO/HI`）：只在 `lo<=t<=hi` 施加 guidance，
其余步只跑 cond。实测：`CFG_HI=0.5` → S2A ~1.35×、log-mel RMSE 0.10（`0.7` 更稳 RMSE 0.06）。
**guidance 在早期关键 → 降 HI（跳过晚期）安全，抬 LO（跳过早期）质量崩**（mid 区间 RMSE 0.72）。
与 ① AB2 正交：**ab2-16 + cfg_hi=0.6 ≈ S2A 2×**。默认全 CFG 不变。
（guidance distillation 能更彻底地省掉 uncond 前向，但需训练；CFG 间隔是免训练的近似。）

> Tier A（①求解器 + B2 CFG间隔）已让 S2A 达 ~2×（免训练）。BigVGAN fp16 实测无效（见上）。
> T2S 仍是闸门——整体要冲 2–3× 仍需 B1 投机解码（④，需训练）。

---

## Tier C — 全蒸馏，3–5×+，但要重训（偏离"忠实原模型"）

- **S2A 一致性 / rectified-flow 蒸馏** → 1–4 步生成（vs 现在 25 步），S2A 5×+。
- **T2S 层蒸馏**（24→12 层）或非自回归 / 并行 semantic 解码器。

这些改动最大、收益最大，但与项目原则"尽可能忠实原模型"冲突，需要训练管线。

---

## Apple Silicon 原生的横切手段（与上面正交）

### D1. 自定义融合 Metal kernel（PLAN.md 本就规划的方向）
- **T2S 解码步**：把 24 层的逐元素胶水手写成融合 kernel，KV 留在 threadgroup memory，
  利用统一内存零拷贝，绕开 MLX 的逐 op dispatch → ~1.5–2× 且**无质量损失**。
  工作量大但最"Apple 原生"。
- **BigVGAN 卷积栈 / snakebeta** 融合 kernel 同理。

### D2. ANE（Apple Neural Engine）卸载
把 GPT-2（或 BigVGAN）经 CoreML 跑在 ANE 上——ANE 对 transformer 解码极其高效且能与
GPU 并行，可能是单点最大的 Apple 专属收益。但架构破坏性大：静态形状要求、算子受限、
CoreML 转换、与 MLX 混跑的协调，落地风险高、不确定性大。

---

## 度量注意

- **波形级 cosine 不可靠**：S2A 用种子化噪声 `z`，步数不同会把 ODE 积到不同终点，产生
  相位 / 时移；波形 cosine 对一个采样点的位移都极敏感（cosine 暴跌不等于音质差）。
  评估减步数 / 换求解器 / 量化时，应用 **mel 谱距离或感知指标 / 人耳试听**，而非原始波形 cosine。
- **量化保真**：int8 logit cosine 0.9999、token 一致；int4 cosine ~0.97、token 流漂移、
  时长 ±20%（不推荐）。
- **warmup 混淆**：服务端首个请求慢主要是 MLX kernel 首次编译（warmup），不要把它误算成
  某项优化的收益；用第 2、3 个（已 warm）请求做对比。
- **fp16 在本机不是加速手段**：实测 T2S fp16 仅 ~1.1×、S2A/BigVGAN fp16 无收益甚至更慢
  （BigVGAN conv fp16 890ms vs fp32 664ms）。MLX/Metal 对这些 op 的 fp16 路径不加速。
  **加速靠 int8 `quantized_matmul`（matmul 重的阶段），不是 fp16。**
- **批处理（⑥）不是纯 dispatch 收益**：原以为 M=1→M=N 几乎免费（纯 dispatch 受限），
  实测 int8 解码每步有真实 compute，batch-2 步 ≈ 1.44× 单步，故净收益仅 ~1.2–1.3×（非 N×），
  且段长不均会浪费并行（跑到最长段）。结论正确无损但收益有限、随**均衡段数**增长，已设为 opt-in。

---

## 结论与建议落地顺序

纯量化 / 工程到此收益有限（再 ~1.5–1.8× 封顶）。**稳健的 2–3× 必须靠算法级减"步数"**
——T2S 的投机 / 多 token 解码 + S2A 的高阶求解器 / 蒸馏，最好再叠加自定义 Metal kernel。

ROI 从高到低（均可独立验证）：

| 选项 | 收益 | 工作量 | 重训? | 状态 |
|---|---|---|---|---|
| ① S2A 换 DPM-Solver++ 求解器 | S2A ~1.5×（ab2-16） | 中 | 否 | ✅ 已落地（opt-in）|
| ⑥ 长文本多段批处理（batched decode） | T2S ~1.2–1.3×（实测，非 2.5×）| 中大 | 否 | ✅ 已落地（opt-in，默认关）|
| ②' S2A DiT int8 `quantized_matmul` | S2A/前向 ~1.5–2× | 中大 | 否 | 计划中（替代原 fp16 方案）|
| ③ T2S `mx::compile` + 定长 KV cache | T2S ~1.3× | 中大 | 否 | — |
| ④ **T2S Medusa / 投机解码** | T2S ~2× | 大 | 轻量 | 冲 2–3× 必经 |
| ⑤ 自定义融合 Metal kernel（解码步 / 声码器） | ~1.5–2× | 很大 | 否 | BigVGAN 提速唯一路 |
| ~~② BigVGAN/S2A fp16~~ | — | — | — | ❌ 实测无效，已放弃 |

- ① + ⑥ + ②' 一轮做完，长文本可达 ~1.8–2.2×（求解器 + 批处理 + S2A int8），全部无重训。
- 要让**短句**也到 2–3×，④（投机解码）是必经之路，或上 ⑤（Metal kernel）。

**推荐首步：① S2A 高阶求解器** —— 单点最干净、无重训、可立刻 A/B 验证质量的 ~2× 收益。
**关键教训：本机加速靠 int8 `quantized_matmul`，不靠 fp16。**
