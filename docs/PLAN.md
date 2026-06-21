# Confucius4-TTS → `c4tts`：C++ / Apple Silicon 原生推理引擎移植计划

> 目标：参照 [antirez/ds4 (DwarfStar)](https://github.com/antirez/ds4) 的工程思路，把本项目重写为一个
> **自包含、不依赖 Python 运行时、为 Apple Silicon 深度优化的 C++ 推理引擎**，在尽量不损失 TTS 精度的
> 前提下把 RTF（Real-Time Factor）压到最低。产物全部位于本仓库 `c4tts/` 目录。
>
> 本文件只输出**计划**。每一步都被拆成「可独立验证」的小任务，先稳后快：**先对齐数值精度，再做性能优化**。

---

## 0. 借鉴 ds4 的核心理念

ds4（DwarfStar，DeepSeek V4 的专用 C 推理引擎）的几条原则，直接套用到本项目：

1. **窄聚焦（narrow focus）**：不做通用框架，只服务 Confucius4-TTS 这一个模型族。不追求支持任意
   GGUF / 任意算子，只实现本模型实际用到的算子。
2. **官方对齐校验（logits 对照）**：以 PyTorch 原实现为「黄金参考」，逐模块用张量级数值对照保证正确性，
   而不是「看起来能跑 / 听起来还行」。
3. **端到端自包含**：从文本输入到 wav 输出，整条链路（分词、文本归一化、特征提取、三段模型、声码器）
   都在 C++ 内完成，无 Python、无 PyTorch、无 transformers 运行时依赖。
4. **硬件优先（Apple Metal first）**：主目标是 Apple Silicon（M 系列）统一内存 + Metal/MLX；其它平台暂不考虑。
5. **权重为一等公民**：用一套自定义的、对引擎和硬件友好的权重打包格式；量化是可选增强而非前提。

---

## 1. 现状分析：推理链路与组件清单

### 1.1 推理流水线（`confuciustts/cli/inference.py`）

```
                       参考音频 prompt.wav
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
   resample 16k          resample 22.05k        resample 16k
        │                     │                      │
        ▼                     ▼                      ▼
  ┌───────────┐         ┌───────────┐          ┌───────────┐
  │ W2V-BERT  │         │ ref mel   │          │ Kaldi     │
  │ layer-17  │         │ (STFT)    │          │ fbank     │
  │ semantic  │         └─────┬─────┘          └─────┬─────┘
  │ features  │               │                      ▼
  └─────┬─────┘               │                ┌───────────┐
        │ (归一化 mean/var)    │                │ CAMPPlus  │ (192-d 说话人风格)
        │                     │                └─────┬─────┘
文本 ───┼── normalize ── tokenizer(SentencePiece) ─┐ │
        │                                          │ │
        ▼                                          ▼ │
  ┌──────────────────────────────────────────────────────┐
  │ ① T2S  Text2Semantic（GPT-2 backbone, 24L/1280d, AR） │
  │    in: text tokens + semantic-condition(说话人编码器)  │
  │    out: semantic_codes (离散) + lm_latent (1280-d)     │
  └───────────────────────────┬──────────────────────────┘
                              │ semantic_codes, lm_latent, ref_mel, spk_emb
                              ▼
  ┌──────────────────────────────────────────────────────┐
  │ ② S2A  MaskedDiffWithXvec（Flow-Matching / DiT）       │
  │    token emb + lm_latent → length regulator(上采样)    │
  │    → ConditionalCFM Euler ODE(25 步 × CFG×2) → mel     │
  └───────────────────────────┬──────────────────────────┘
                              ▼ mel (80-band, 22.05kHz)
  ┌──────────────────────────────────────────────────────┐
  │ ③ Vocoder  BigVGAN v2 (22kHz/80band/256x)             │
  │    mel → waveform                                      │
  └───────────────────────────┬──────────────────────────┘
                              ▼
                  分段 cross-fade 拼接 → output.wav
```

长文本会按标点切分成多段（`segment_text`），逐段合成后 cross-fade 拼接。

### 1.2 组件清单与移植难度评估

| # | 组件 | 源文件 | 角色 | 权重来源 | 运行成本 | 移植难度 |
|---|------|--------|------|----------|----------|----------|
| F1 | 文本归一化 | `frontend/text_normalizer.py` | 数字/标点/多语种规整 | — | 低（一次） | **中**（依赖 `wetext`/`inflect`/`pykakasi`） |
| F2 | 分词器 | HF `AutoTokenizer` + `tokenizer.model` | SentencePiece(LLaMA 式, 32000) | 本地 `checkpoints/` | 低 | 中 |
| F3 | 音频 IO / 重采样 | `torchaudio` | wav 读写 + 16k/22.05k 重采样 | — | 低 | 低 |
| F4 | STFT mel | `utils/audio_features.py:mel_spectrogram` | ref mel（S2A 条件 + 声码器口径） | — | 低 | **中**（需逐位对齐 librosa slaney） |
| F5 | Kaldi fbank | `torchaudio.compliance.kaldi.fbank` | CAMPPlus 输入 | — | 低 | **高**（Povey 窗等细节多） |
| F6 | Seamless 特征 | `SeamlessM4TFeatureExtractor` | W2V-BERT 输入特征 | HF | 低 | **高**（归一化/分帧细节） |
| C1 | W2V-BERT | `transformers.Wav2Vec2BertModel` | 语义特征（取第 17 层） | HF `facebook/w2v-bert-2.0` | 中（一次，~600M 参数, 仅需前 17 层） | **高** |
| C2 | CAMPPlus | `external/campplus/DTDNN.py` | 说话人风格 192-d | HF `funasr/campplus` | 低 | 中 |
| M1 | T2S | `llm/llm.py` (`Text2Semantic`) | 文本→语义 token（自回归） | HF `t2s_model.safetensors` | **高（AR 串行，是延迟大头）** | **高** |
| M1a | 文本嵌入投影 | `llm/text_encoder.py` | 冻结 `Embedding(32000,4096)`+MLP | 同上 | 低 | 低 |
| M1b | 内置说话人编码器 | `llm/speaker_encoder.py` | ECAPA-TDNN（语义特征→条件 token） | 同上 | 低 | 中 |
| M1c | GPT-2 backbone | HF `GPT2Model` | 24 层 Transformer + KV cache | 同上 | 高 | 中 |
| M2 | S2A | `flow/flow.py` 等 | 语义→mel（flow matching） | HF `s2a_model.pt` | **高（25×2 次 DiT）** | **高** |
| M2a | length regulator | `flow/length_regulator.py` | 最近邻上采样 + Conv1d 栈 | 同上 | 低 | 低 |
| M2b | DiT estimator | `flow/DiT/dit.py` + `modules.py` | 13 层 DiT + WaveNet 末层 | 同上 | 高 | **高** |
| M2c | ConditionalCFM | `flow/flow_matching.py` | Euler ODE + CFG | 同上 | 高 | 中 |
| V1 | BigVGAN | `external/bigvgan/bigvgan.py` | mel→波形（256x 上采样） | HF `nvidia/bigvgan_v2_22khz_80band_256x` | **高（卷积密集）** | **高**（snakebeta + 抗混叠滤波） |

**RTF 关键路径**：M1（自回归、串行、延迟敏感）、M2（25×2 次 DiT 前向）、V1（密集卷积）。
C1/C2/前端只在每条 utterance 跑一次（针对 prompt），是固定开销而非每帧开销。

### 1.3 重要细节（移植时必须逐位复现，否则精度漂移）

- **STFT mel（F4）**：`center=False`，手动 `reflect` pad `(n_fft-hop)//2`；幅度谱 `sqrt(power+1e-9)`；
  librosa **slaney** norm 的 mel 滤波器组；最后 `log(clamp(·, 1e-5))`。
- **GPT-2 位置编码**：HF 的 `wpe` 被替换成全零（`DummyPositionEmbedding`），所有位置信息来自
  **自定义可学习绝对位置编码**（text / semantic 各一套）。KV-cache 解码时单 token 的位置
  = `attention_mask.shape[1] - prefix_len - 1`。
- **前缀缓存**：`condition_emb`（1 个 token）+ `text_emb` 在生成前由 `store_conditioning` 缓存一次。
- **T2S 采样**：默认 `num_beams=3` + `do_sample` + `top_p=0.8` + `top_k=30` + `temperature=0.8`
  + `repetition_penalty=10` + `early_stopping`。`return_latent=True` 时会用**固定 token 序列再跑一次
  整序列前向**取 latent（这一步是确定性的，适合做严格数值对齐）。
- **mel 长度启发式**：`target_mel_len = round(semantic_len * 1.72)`。
- **CFM ODE（M2c）**：config 用 **linear** t-schedule（不是 cosine）；CFG 通过 batch 翻倍实现；
  每个 Euler 步都把 prompt 区域置零；prompt 段从输出里裁掉。
- **DiT（M2b）**：RoPE 用 float32 计算；AdaLN 用 RMSNorm；FFN 是 SwiGLU；U-Net 长跳连接
  （前半层 emit、后半层 receive）；末层是 WaveNet（门控卷积 + FinalLayer adaLN）。
- **BigVGAN（V1）**：`snakebeta`（logscale）激活；抗混叠 up/down 采样（Kaiser 窗 FIR）；MRF
  多感受野残差块；加载时 `remove_weight_norm`（导出权重时应**预先折叠** weight_norm）。
- **CAMPPlus fbank（F5）**：`torchaudio` Kaldi fbank（Povey 窗、pre-emphasis 0.97、dither=0），
  之后逐帧减均值。这是已知的高对齐风险点。

---

## 2. 关键技术决策

### 2.1 计算后端：MLX C++ 为主，Metal 自定义 kernel 补位

**决策**：以 **MLX 的 C++ API** 作为主张量/算子后端，热点算子用**自定义 Metal kernel** 补位。

理由：
- 本项目算子谱很广（Conv1d/转置卷积、STFT、ECAPA-TDNN、Conformer、GPT-2、DiT、ODE、抗混叠滤波），
  MLX 已原生提供绝大多数算子且基于 Metal、统一内存、惰性求值 + 图融合，能快速达到「正确且不慢」。
- 纯手写 Metal 控制力最强但工作量巨大；先用 MLX 跑通对齐，再针对 profiler 找出的少数热点
  （如 snakebeta 抗混叠、STFT、采样、Conv 融合）写专用 kernel，是性价比最高的路线，也符合 ds4
  「先正确、后局部极致优化」的节奏。
- **抽象层**：定义薄薄一层 `c4::Tensor` / op 接口，把 MLX 调用封装其后。这样即便将来某模块换成纯
  Metal/手写 kernel，也不影响上层模型代码。**这是贯穿全程的架构约束。**

> 开放决策点：若后续 profiling 表明 MLX 在自回归小 batch、KV-cache 解码场景开销过大，可对 M1
> 单独改为手写 Metal/Accelerate 实现。计划中的模块边界已为此预留。

### 2.2 权重打包格式

- 提供一个 **Python 导出脚本**（`c4tts/tools/export_weights.py`），把所有 HF 权重
  （t2s safetensors、s2a `.pt`、bigvgan、campplus、w2v-bert 前 17 层、文本嵌入、stats）转换并
  **合并成单个 `c4tts` 自定义权重文件**（建议直接复用 `safetensors` 二进制布局 + 一个 JSON manifest，
  便于 C++ 端零依赖 mmap 读取；后续可演进为 GGUF 式分块/量化）。
- 导出时完成所有「推理期静态变换」：折叠 BigVGAN 的 `weight_norm`、预计算 mel 滤波器组 / RoPE /
  抗混叠 FIR 系数、合并 LayerNorm 仿射等，让 C++ 端只做纯前向。
- 精度策略：**先全 fp32 对齐**，再逐模块切 fp16/bf16 并复测精度；量化（int8/int4）列为后续可选项。

### 2.3 分词器与文本归一化

- F2：移植 SentencePiece。可选 (a) 链接官方 `sentencepiece` C++ 库（仍是自包含的本地 C++ 依赖，
  非 Python），或 (b) 自己实现一份 unigram/BPE 解码器读取 `tokenizer.model`。**默认选 (a)**，更稳。
- F1：文本归一化依赖较重（`wetext` 的中文 TN、`inflect` 英文数字、`pykakasi` 日文）。策略：
  - 第一阶段**只实现英文/中文最常用规则**的 C++ 版，复杂 TN 暂以「直通 + 基础规则」近似；
  - 把 TN 做成可插拔模块，必要时单独评估其对最终音频的影响（多数 TN 差异只影响个别读法，不影响管线正确性验证）。

### 2.4 数值对齐方法论（贯穿所有移植步骤）

- 在 PyTorch 端写一个**黄金向量采集脚本**（`c4tts/tools/dump_golden.py`），用固定随机种子 + 固定输入，
  导出每个模块的**输入/输出张量**为 `.npz`（含中间激活，如 W2V-BERT 第 17 层、T2S latent、S2A mel、声码器波形）。
- C++ 端每个模块配一个 **gtest/Catch2 单测**：喂相同输入，与黄金向量比对。
- 阈值约定：
  - 线性/卷积/归一化等逐元素模块：`max_abs_err < 1e-4`（fp32）。
  - 深层堆叠模块（W2V-BERT、GPT-2、DiT 单次前向）：`cosine_sim > 0.9999` 且 `rel_L2 < 1e-3`。
  - 含采样的 T2S 生成：**不做 token 级逐一对齐**（RNG 不同）；改为
    (a) 贪心模式下 logits/argmax 对齐；(b) 固定 token 序列下 latent 对齐；(c) 端到端用客观指标（见 §5）。
  - 端到端音频：mel 距离 / PESQ / 说话人余弦相似度 + 人工试听。

---

## 3. 目标目录结构 `c4tts/`

```
c4tts/
├── CMakeLists.txt
├── README.md
├── third_party/            # mlx, sentencepiece 等（submodule 或 fetch）
├── include/c4tts/          # 公共头文件
│   ├── tensor.h            # c4::Tensor / op 抽象（封装 MLX）
│   ├── weights.h           # 权重文件 mmap 加载
│   └── pipeline.h          # 顶层 API: synth(text, lang, prompt_wav) -> wav
├── src/
│   ├── core/               # tensor 抽象、op 封装、Metal kernel 注册
│   ├── audio/              # wav io、resample、STFT-mel、kaldi-fbank、seamless 特征
│   ├── frontend/           # 文本归一化、SentencePiece 封装、分段
│   ├── cond/               # W2V-BERT、CAMPPlus、ref-mel 条件提取
│   ├── t2s/                # 文本嵌入投影、说话人编码器、GPT-2、采样、KV-cache
│   ├── s2a/                # token emb、length regulator、DiT、CFM ODE
│   ├── vocoder/            # BigVGAN
│   └── pipeline.cpp        # 串联 + cross-fade
├── kernels/                # 自定义 .metal 着色器
├── tools/
│   ├── export_weights.py   # HF 权重 → c4tts 权重包
│   └── dump_golden.py      # 黄金向量采集
├── tests/                  # 各模块数值对齐单测 + 端到端测试
└── apps/
    └── c4tts_cli.cpp       # 等价 example.py 的命令行入口
```

---

## 4. 分阶段实施计划（每步可独立验证）

> 约定：每个步骤都给出 **目标 / 交付物 / 验证方法 / 依赖**。验证全部以 §2.4 的黄金向量为准。
> 顺序总体是：**基础设施 → 前端 → prompt 条件 → T2S → S2A → 声码器 → 集成 → 优化**。
> 一个隐含的好处：声码器（V1）和 S2A（M2）可以脱离 T2S 用「真实中间张量」单独验证，便于并行推进。

### Phase A — 基础设施与对齐脚手架

- **A1. 构建系统与依赖**
  - 目标：CMake 工程能编译，链接 MLX、SentencePiece、gtest；产出空的 `c4tts_cli`。
  - 验证：`cmake --build` 成功；`c4tts_cli --version` 可运行。
- **A2. `c4::Tensor` / op 抽象层**
  - 目标：封装 MLX 的张量与基础算子（matmul、conv1d、layernorm/rmsnorm、gelu/silu/softmax、
    interpolate、stft 等），定义统一 dtype/shape/设备语义。
  - 验证：对每个封装算子写微基准 + 与 NumPy/PyTorch 单算子黄金向量对齐（`max_abs_err<1e-5`）。
- **A3. 权重加载器**
  - 目标：mmap 读取 §2.2 的权重包 + manifest，按名字取张量。
  - 验证：加载导出包，逐张量校验 shape/dtype 与 md5 与 PyTorch 端一致。
- **A4. 黄金向量与导出工具**
  - 目标：`dump_golden.py`（采集各模块 I/O）+ `export_weights.py`（合并权重）跑通。
  - 验证：生成的 `.npz` 与权重包可被 A3 读取；固定输入下可复现。

### Phase B — 前端（文本 → token；音频 → 特征）

- **B1. 音频 IO + 重采样（F3）**
  - 验证：读 `prompt.wav`，重采样到 16k/22.05k，与 `torchaudio` 输出 `max_abs_err<1e-4`。
- **B2. STFT-mel（F4）**
  - 验证：对参考 wav 计算 ref mel，与 `mel_spectrogram` 逐位对齐（`max_abs_err<1e-3`，注意 log 域）。
- **B3. Kaldi fbank（F5）**【高风险，单列】
  - 验证：与 `torchaudio.compliance.kaldi.fbank`（含减均值）对齐；先保 `cosine>0.999`，
    若 Povey 窗/pre-emphasis 难逐位对齐，记录偏差并评估其对 CAMPPlus 输出的影响（见 C2）。
- **B4. SentencePiece 分词（F2）**
  - 验证：对一批中/英/日句子，token id 序列与 HF `AutoTokenizer` **完全一致**；含
    `"You are a helpful assistant. {lang_token}:{text}"` 模板拼接与 BOS/EOS 规则。
- **B5. 文本归一化 + 分段（F1）**
  - 验证：英文/中文常见用例与 Python `TextNormalizer.normalize` + `segment_text` 对齐；
    差异用例登记到「已知 TN 偏差」清单。**此步允许近似**，不阻塞主链路。

### Phase C — Prompt 条件提取

- **C1. SeamlessM4T 特征（F6）**【高风险】
  - 验证：输入特征张量与 `SeamlessM4TFeatureExtractor` 输出对齐（含分帧、log-mel、归一化、padding/mask）。
- **C2. W2V-BERT（C1）**【大模块，分子步】
  - 子步：feature projection → Conformer block（FFN/MHSA-相对位置/Conv module/FFN + LN）×N → 取第 17 层。
  - 关键：只需前 17 层，可在导出时裁掉其余层省内存/算力。
  - 验证：逐 block 累积对齐；最终第 17 层隐藏态（经 stats 归一化）与 Python `_extract_semantic`
    输出 `cosine>0.9999`。
- **C3. CAMPPlus（C2）**
  - 验证：fbank → 192-d 风格向量，与 `style_encoder` 输出对齐（`cosine>0.999`）；
    若受 B3 偏差影响，量化其对最终说话人相似度的影响。
- **C4. 条件聚合**
  - 目标：产出 `(semantic_features, style_embedding, reference_mel)` 三元组。
  - 验证：与 `inference.py` 中三者数值对齐，作为后续阶段的「真实输入」。

### Phase D — T2S（文本 → 语义 token，自回归）

- **D1. 文本嵌入投影（M1a）**
  - 验证：冻结 `Embedding(32000,4096)` + fc1/silu/fc2 → 1280-d，与 `TextEmbeddingProjector` 对齐。
- **D2. 内置说话人编码器（M1b, ECAPA-TDNN）**
  - 验证：语义特征 → 1 个 condition token（1280-d），与 `Qwen3TTSSpeakerEncoder` 对齐。
- **D3. 位置编码 + 前缀缓存（M1c 一部分）**
  - 验证：text/semantic 两套可学习绝对位置编码、`store_conditioning` 前缀、单 token 位置公式
    与 Python 完全一致。
- **D4. GPT-2 backbone + KV-cache（M1c）**
  - 子步：QKV/causal attention、MLP(gelu)、LayerNorm、24 层、`final_norm`、`semantic_head`。
  - 验证：(a) **整序列前向**（固定输入）logits 与 HF `GPT2Model` 路径对齐（`cosine>0.9999`）；
    (b) 单步 KV-cache 前向与整序列对应位置 logits 一致（`max_abs_err<1e-4`）。
- **D5. 采样 / beam search / 重复惩罚（M1）**
  - 目标：实现 temperature/top-k/top-p/repetition_penalty + beam(`num_beams`) + early stopping + EOS。
  - 验证：贪心模式下与 HF `generate(do_sample=False)` token 序列**完全一致**；采样模式做分布级
    校验（同种子统计量接近）。**注意**：默认 beam=3 是性能负担，作为优化期可调项（见 §6）。
- **D6. latent 提取**
  - 验证：给定 token 序列，按 `return_latent` 的「整序列再前向」路径取 `lm_latent`，与 Python 对齐
    （这是确定性步骤，必须 `max_abs_err<1e-4`）。
- **D7. T2S 端到端**
  - 验证：贪心模式下 `semantic_codes` 与 `lm_latent` 与 Python 一致；喂给后续阶段。

### Phase E — S2A（语义 token → mel，flow matching）

- **E1. 语义 token 嵌入（M2 入口）**
  - 验证：`SemanticTokenEmbedding`（Embedding(8192,8)+Conv1d 1x1→1024）与 Python 对齐。
- **E2. encoder_proj + Length Regulator（M2a）**
  - 验证：`cat(lm_latent, semantic_emb)`→Linear→最近邻 `interpolate` 到目标帧数 + Conv1d/GroupNorm/Mish
    栈，与 `InterpolateRegulator` 对齐（注意 mask 与 `target_feat_len` 计算）。
- **E3. DiT 单次前向（M2b）**【大模块，分子步】
  - 子步：InputEmbedding、TimestepEmbedding（正弦+MLP）、RoPE、AdaLN(RMSNorm)、SwiGLU、
    U-Net 跳连、WaveNet 末层（门控卷积 + FinalLayer）。
  - 验证：固定 `(x, mask, mu, t, spks, cond)` 输入，输出速度场与 `DiT.forward` 对齐（`cosine>0.9999`）。
- **E4. ConditionalCFM Euler ODE + CFG（M2c）**
  - 验证：固定初始噪声 `z`（同种子），linear t-schedule、25 步、CFG batch 翻倍、prompt 区域置零，
    最终 mel 与 `solve_euler` 对齐（`max_abs_err<1e-3`）。
- **E5. S2A 端到端**
  - 验证：用 Phase C/D 的真实中间量 + 固定噪声，`inference()` 输出 mel 与 Python 对齐；裁掉 prompt 段。

### Phase F — 声码器 BigVGAN（mel → 波形）

- **F-1. 卷积/转置卷积 + MRF 残差块**
  - 验证：单个 AMPBlock / 上采样块与 PyTorch 对齐。
- **F-2. snakebeta 激活 + 抗混叠 up/down（V1 核心）**
  - 验证：Activation1d（Kaiser FIR 上采样 → snakebeta → 下采样）与 torch 版对齐；
    导出时折叠 weight_norm，C++ 端直接用折叠后权重。
- **F-3. BigVGAN 端到端**
  - 验证：给定真实 mel，输出波形与 Python `bigvgan(mel)` 对齐（`max_abs_err<1e-3`，并听感一致）。
  - 可独立验证：本阶段不依赖 T2S/S2A，可用黄金 mel 直接驱动，**适合早期并行启动**。

### Phase G — 端到端集成

- **G1. 单段合成**：串起 C→D→E→F，复现 `_synth_segment`。
  - 验证：同 prompt + 同文本 + 贪心 + 固定噪声下，输出 wav 与 Python（同设定）逐样本接近；客观指标见 §5。
- **G2. 多段 + cross-fade**：复现 `segment_text` 分段与 `cross_fade_concat` 拼接。
  - 验证：长文本输出与 Python 对齐；段边界无爆音。
- **G3. CLI**：`c4tts_cli --prompt_wav --text --lang --out`，对齐 `example.py` 参数。
  - 验证：命令行端到端跑通，产出可播放 wav。

### Phase H — 性能优化（压 RTF）

见 §6。**进入本阶段的前提：G1/G2 已通过数值/客观指标验证**。每次优化后都要回归 §5 的客观指标，
确保「不降精度」。

---

## 5. 端到端验收指标（"不降精度" 的定义）

以 Python 原实现在 **相同设定**（同 prompt、同文本、贪心或固定种子、相同 ODE 步数）下的输出为基准：

- **mel L1/L2 距离**：C++ 与 Python mel 的平均逐帧距离低于设定阈值。
- **说话人相似度**：两份输出经 CAMPPlus 提取的 embedding 余弦相似度 > 0.99（音色保持）。
- **可懂度**：用一个第三方 ASR 对两份音频转写，WER/CER 差异在噪声范围内。
- **（可选）PESQ/UTMOS**：客观音质评分不低于基准。
- **人工试听**：A/B 盲听无可感差异。

> 采样路径（beam/top-p）天然不可逐位复现，因此「精度不降」以上述**分布级 + 感知级**指标为准；
> 确定性子模块（latent、DiT 单次前向、声码器）仍用严格张量阈值。

---

## 6. 性能优化路线（RTF）

按 §1.2 的关键路径，优化优先级：

1. **T2S 自回归（延迟大头）**
   - KV-cache 复用 + 预分配，避免每步重算前缀（前缀已缓存）。
   - **默认 `num_beams=3` → 评估降到 1（贪心/采样）** 的质量损失；beam 是 3× 成本。
   - fp16/bf16 权重 + 融合 QKV/MLP；必要时为单 token 解码写专用 Metal kernel（小 batch 场景 MLX 可能偏慢）。
   - 探索投机解码 / 早停阈值调优。
2. **S2A flow matching（25×2 次 DiT）**
   - 减少 ODE 步数（25 → 10/8）并评估质量；CFG batch 翻倍可在单次 kernel 内并行。
   - DiT 算子融合（AdaLN+attention、SwiGLU）、RoPE 预计算、Conv 融合。
   - fp16/bf16。
3. **BigVGAN（密集卷积）**
   - 转置卷积 / Conv1d 用 Metal 优化或 MLX 融合；snakebeta 抗混叠写专用 kernel（点态 + 短 FIR，融合收益大）。
4. **prompt 条件（固定开销）**
   - W2V-BERT 只跑前 17 层；prompt 特征/风格向量可缓存复用（同一说话人多次合成时）。
5. **全局**
   - 统一内存零拷贝、惰性图融合、算子常驻、避免 host/device 往返。
   - 权重量化（int8/int4，先在 T2S/S2A 大权重上试）作为后续增强，需复测 §5 指标。

每项优化都遵循：**先 profiler 定位热点 → 改 → 回归精度与 RTF**，记录到性能跟踪表。

---

## 7. 风险与开放问题

- **R1 高风险对齐点**：F5（Kaldi fbank）、F6（Seamless 特征）、C1（W2V-BERT）。这些前端/编码器细节
  最容易引入偏差。缓解：单列子步、先 cosine 后逐位、量化其对最终音频的影响。
- **R2 文本归一化**：`wetext`/`inflect`/`pykakasi` 难以完全复刻。缓解：做成可插拔，分阶段补齐，登记偏差。
- **R3 采样不可复现**：beam/top-p RNG 与 HF 不一致。缓解：用确定性子模块做严格对齐 + 端到端用感知指标。
- **R4 MLX 在自回归小 batch 的开销**：可能需要为 T2S 单步解码改手写 kernel。模块边界已预留。
- **R5 权重许可与体积**：W2V-BERT/BigVGAN/CAMPPlus 来自不同上游，需核对 license 与分发方式；导出包体积较大。
- **开放决策**：
  - 后端是否在某些模块用纯 Metal 替代 MLX（待 profiling）。
  - 权重格式是否升级为 GGUF 式分块/量化。
  - 文本归一化的目标覆盖度（哪些语言/规则必须 C++ 化）。

---

## 8. 里程碑（建议顺序，可并行处）

| 里程碑 | 内容 | 完成判据 |
|--------|------|----------|
| **M0** | Phase A 脚手架 + 导出/黄金工具 | 能编译、能加载权重、能跑单算子对齐 |
| **M1** | Phase B + C 前端与条件提取 | 三元组 `(sem_feat, spk_emb, ref_mel)` 对齐 Python |
| **M2** | Phase F 声码器（可与 M1 并行） | 黄金 mel → 波形对齐 |
| **M3** | Phase E S2A | 真实中间量 → mel 对齐 |
| **M4** | Phase D T2S | 贪心 token + latent 对齐 |
| **M5** | Phase G 端到端集成 + CLI | 端到端 wav 通过 §5 客观指标 |
| **M6** | Phase H 性能优化 | RTF 达标且 §5 指标不退化 |

> 关键并行机会：**M2（声码器）和 M1（前端/条件）可同时开工**；S2A（M3）一旦有黄金中间量也可先于
> T2S 完成验证。T2S（M4）是最复杂、最影响延迟的模块，建议安排充足时间并最先做性能预研。

---

## 附录 A：关键超参（来自 `config/inference_config.yaml`）

- 音频：`sr=22050, n_fft=1024, hop=256, win=1024, n_mels=80, fmin=0, fmax=None`，prompt sr=16000。
- T2S：`24 层, dim=1280, heads=20, text_max=520, sem_max=1520, vocab=32000, sem_vocab=8194,`
  `text_embed=4096, spk_embed=1024, BOS=8192, EOS=8193`。
- S2A：`input=512, output(mel)=80, spk=192, sem_embed=1024, lm_latent=1280, DiT depth=13, heads=8,`
  `hidden=512, mlp_ratio=3.0, wavenet 8 层/kernel 5, CFM linear schedule, cfg_rate=0.7, n_steps=25`。
- length regulator：`channels=512, in=1024, out=512, sampling_ratios=(1,1,1,1)`。
- BigVGAN：`upsample_rates=[4,4,2,2,2,2], init_channel=1536, resblock kernels=[3,7,11],`
  `snakebeta(logscale), 22.05kHz/80band/256x`。
- 生成默认：`temperature=0.8, top_p=0.8, top_k=30, num_beams=3, repetition_penalty=10,`
  `max_length=1520, n_timesteps=25, cfg_rate=0.7, max_text_tokens_per_segment=80, cross_fade=0.3s`。
- mel 长度启发式：`target_mel_len = round(semantic_len * 1.72)`。
