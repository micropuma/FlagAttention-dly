# FlagAttention 仓库结构与开发指引

本文档面向在本仓库中工作的智能体或开发者，目标是快速解释项目结构、模块职责、典型工作流，以及修改代码时应该优先关注的文件。

## 1. 项目定位

FlagAttention 是一个基于 Triton 的 memory-efficient attention 算子仓库，核心价值不是做一个大而全的训练框架，而是提供可读、可改、可验证的注意力算子实现与基准测试。

当前仓库的代码重心集中在 `src/flag_attn/`，已经公开的主要算子包括：

- `flash_attention`
- `piecewise_attention`
- `flash_attention_split_kv`
- `paged_attention`

其中 README 主要强调 `flash_attention` 和 `piecewise_attention`，但实际导出的公共 API 已经覆盖 `paged_attention` 和 `split_kv` 路径，后续如果继续维护文档，需要注意 README 与真实导出保持一致。

## 2. 顶层目录结构

仓库根目录的主要结构如下：

```text
FlagAttention-dly/
├── README.md
├── README_cn.md
├── LICENSE
├── pyproject.toml
├── agent.md
├── .github/workflows/
├── assets/
├── benchmark/
├── examples/
├── src/flag_attn/
└── tests/flag_attn/
```

各目录职责：

- `README.md` / `README_cn.md`
  对外说明项目背景、安装方式、公开接口和性能图。适合先建立全局认知，但不能替代源码。
- `pyproject.toml`
  Python 打包入口，定义项目名、依赖、`pytest` 路径和 `ruff` 配置。
- `.github/workflows/code-check.yml`
  CI 配置，目前主要执行 editable install 和 `pytest tests`。
- `assets/`
  项目图像资源、logo、性能图，不参与核心逻辑。
- `examples/`
  最小可运行示例，适合确认 API 调用方式、输入张量形状与返回值。
- `benchmark/`
  性能基准脚本，和 `examples/` 不同，这里更关注吞吐、延迟和外部实现对比。
- `src/flag_attn/`
  真实实现所在目录，包括 Triton kernel、autograd Function、reference/testing 辅助模块。
- `tests/flag_attn/`
  正确性测试，核心策略是把 Triton 实现与 `flag_attn.testing` 中的 PyTorch reference implementation 做数值比较。

## 3. 源码主干

### 3.1 `src/flag_attn/__init__.py`

这个文件决定包的公开接口。当前导出：

- `piecewise_attention`
- `flash_attention`
- `flash_attention_split_kv`
- `paged_attention`
- `testing`

如果新增算子、重命名接口或调整模块边界，这里必须同步更新，否则外部调用方无法直接访问。

### 3.2 `src/flag_attn/flash.py`

这是标准 flash attention 的核心实现，也是仓库里最关键的文件之一。结构上可以按四层理解：

1. 公共 Python 包装层
   暴露 `attention(...)`，参数包括 `causal`、`sm_scale`、`dropout_p`、是否返回 `log_normalizer` / `total_attention` / `seed_offset` 等。
2. `FlashAttention(torch.autograd.Function)`
   用自定义 `forward` / `backward` 将 Python 接口接到 Triton kernel。
3. kernel 调度与配置选择
   包括是否走普通前向、是否切换到 split-kv、如何选择 block size、warp、stage 等。
4. Triton kernels
   真正执行前向、反向、预处理与 softmax 相关逻辑。

这个文件还体现了几个重要设计点：

- 支持 MQA/GQA：`q` 的头数可以是 `k/v` 的整数倍。
- 支持辅助输出：`log_normalizer`、`total_attention`。
- 支持 dropout：通过 `dropout.py` 中的 Philox seed/offset 配合重算 mask。
- 当前支持的 head dim 明确限制在 `{16, 32, 64, 128}`。
- 当 `S > 1` 时会切到 `split_kv` 路径，以提升低并行度场景下的利用率。

### 3.3 `src/flag_attn/split_kv.py`

这个模块实现 flash decoding / split-kv 相关逻辑，本质上是给 flash attention 暴露额外的并行维度，适用于 `B * H * blocks_along_M` 不足以吃满 SM 的场景。

主要职责：

- `_fwd_split_kv_kernel`
  对 K/V 沿序列维做分片计算局部输出和局部 log-normalizer。
- `_fwd_combine_kv_splits`
  把不同 split 的部分结果做在线 logsumexp 归并。
- `num_splits_herustic`
  根据 `B/H/M/N`、block 配置和 SM 数量决定是否值得切分以及切几份。

`flash.py` 会直接依赖这里的 kernel 和 heuristic，因此改动 split-kv 时必须联动检查 `flash.py` 的调度逻辑和测试覆盖。

### 3.4 `src/flag_attn/total.py`

这个模块负责计算 total attention，也就是对 attention 权重沿 query 维求和后的结果。它不是一个独立公开主算子，而是 `flash_attention(..., return_total_attention=True)` 的辅助能力。

如果只改主 attention 输出却忽略这里，很容易造成辅助输出与主路径不一致。

### 3.5 `src/flag_attn/dropout.py`

这里提供 dropout 所需的随机数种子和偏移量管理：

- `philox_cuda_seed_offset(increment, device=None)`

`flash.py` 的 dropout 路径会依赖这里生成 `(seed, offset)`，测试侧再通过 `flag_attn.testing.recompute_mask` 重构 dropout mask。也就是说，这个模块本身代码不多，但它连接了前向随机性和测试可复现性。

### 3.6 `src/flag_attn/piecewise.py`

这是项目的定制 attention 核心扩展，主要服务于 piecewise score 计算。其特点是：

- 输入不是单组 `(q, k)`，而是两组 `(q1, k1)` 和 `(q2, k2)`。
- 根据 `dist_threshold` 选择当前 token 对应使用哪一组 score。
- 前向阶段拼接两类 score，反向阶段再把梯度拆回两条路径。

从结构上看，它和 `flash.py` 类似，也分为：

- Python 接口 `attention(...)`
- `PiecewiseAttention(torch.autograd.Function)`
- kernel config 选择
- Triton 前向 / 反向 kernels

但它没有 `flash.py` 那么多辅助能力，也没有 split-kv / total attention / dropout 的复杂分支。

### 3.7 `src/flag_attn/paged.py`

这个模块实现 paged attention，更偏向推理场景 KV cache 访问。

输入形式已经不再是完整的 `(B, H, T, D)`，而是：

- `query`
- `key_cache`
- `value_cache`
- `context_lens`
- `block_tables`
- `attn_scale`
- `max_context_len`
- `num_splits`

可以把它理解为“围绕 block table 访问 KV cache 的 paged kernel”。这个模块的典型上下文是 vLLM 风格的数据布局，而不是训练态普通 attention。

内部关键点：

- 支持 MQA/GQA 场景。
- 根据 query group size 做 padding。
- 支持单分片和多分片 reduce 两种路径。
- 默认 heuristic 和实现明显更偏 A100 / Ampere。

如果改这个模块，优先检查 `tests/flag_attn/test_paged_attention.py` 和 `benchmark/paged_benchmark.py`。

## 4. `flag_attn.testing` 的作用

`src/flag_attn/testing/` 是这个仓库里非常重要的一层，不是“边角料”，而是正确性验证基线。

包含：

- `testing/flash.py`
  PyTorch 版 flash attention reference。
- `testing/piecewise.py`
  PyTorch 版 piecewise reference 和梯度检查辅助逻辑。
- `testing/paged.py`
  直接按 block table 取 KV，再用 PyTorch 计算 paged attention 的 reference。
- `testing/dropout.py`
  Triton kernel 重算 dropout mask，用于验证随机路径。

测试策略的核心思想是：

- Triton 实现不直接和“数学绝对真值”对比。
- 而是和一个更稳定、可读的 PyTorch 参考实现对比。
- 对低精度路径，既比较 upcast reference，也比较与 PyTorch 同精度模拟实现之间的误差比例。

如果你在仓库里新增算子或新增返回值，应该优先补 `flag_attn.testing`，再写正式测试。否则测试会失去统一基线。

## 5. 测试目录结构

`tests/flag_attn/` 当前主要包括：

- `test_flash_attention.py`
- `test_piecewise_attention.py`
- `test_paged_attention.py`
- `test_dropout.py`

各自职责：

- `test_flash_attention.py`
  覆盖 flash attention 的前向、反向、aux outputs、不同 shape、不同 dtype、不同 stride order，以及 MQA/GQA 场景。
- `test_piecewise_attention.py`
  覆盖 piecewise attention 前向和反向，比较 Triton 与 reference 的误差上界。
- `test_paged_attention.py`
  用 block table 和 context length 组织输入，验证 paged attention 在不同 block size、head size、split 配置下的正确性。
- `test_dropout.py`
  验证重算出来的 dropout mask 在统计意义上符合给定概率。

这个仓库的测试风格有几个鲜明特征：

- 高度参数化，主要靠 `pytest.mark.parametrize` 做 shape 扫描。
- 默认在所有可见 GPU 上跑：测试用例里经常用 `list(range(torch.cuda.device_count()))`。
- 很多测试直接假设 CUDA 环境可用，不是 CPU 兼容测试。

因此如果你在非 GPU 环境里工作，不能把“本地跑不动测试”理解成仓库坏了。

## 6. `examples/` 目录

`examples/` 更像“调用样板”而不是正式测试：

- `flash_attention_example.py`
  标准 flash attention 前后向示例。
- `flash_attention_with_aux_outputs.py`
  展示如何请求 `log_normalizer` 和 `total_attention`。
- `piecewise_example.py`
  展示 piecewise attention 的输入组织方式。
- `paged_example.py`
  展示 paged attention 的 KV cache、block table、context_lens 用法。
- `use_cutom_config_func.py`
  展示如何 monkey patch `flash.get_fwd_config` 来替换默认配置函数。

如果要快速理解接口，应先看 `examples/`；如果要改实现，则必须回到 `src/flag_attn/` 和 `tests/`。

## 7. `benchmark/` 目录

`benchmark/` 主要负责性能验证，而不是数值正确性。当前脚本包括：

- `flash_benchmark.py`
  对比 `flag_attn`、PyTorch reference、可选的 flash-attn 实现。
- `flash_decoding_benchmark.py`
  面向 decode / split-kv 场景的性能测试。
- `piecewise_benchmark.py`
  对 piecewise attention 做吞吐对比。
- `paged_benchmark.py`
  对 paged attention 与可选 vLLM 实现做对比。

这些 benchmark 的共同特征：

- 用 `triton.testing.Benchmark` 组织测试矩阵。
- 关注 TFLOP/s 或毫秒级延迟。
- 遇到 PyTorch OOM 会把对应结果视为不可用，而不是直接失败。
- 某些脚本会自动创建结果目录，例如 `results_flash_attention_YYYYMMDD`。

如果你的修改涉及 kernel 调度、block size heuristic、split 策略或访存行为，benchmark 往往比单纯 pytest 更能暴露真实问题。

## 8. 配置与工程化约束

### 8.1 `pyproject.toml`

可以从这里读出几个重要约束：

- 包名：`flag_attn`
- Python 要求：`>=3.7`
- 核心依赖：`triton>=2.2.0`
- 测试额外依赖：`pytest>=7.1.0`
- `pytest` 默认从 `tests/` 收集，并把 `src`、`tests/flag_attn` 加到 `pythonpath`
- `ruff` 只做非常轻量的配置，忽略 `E741`

这说明仓库工程化相对轻量，主要依靠源码约束和测试正确性，而不是复杂的 lint/type system。

### 8.2 CI

`.github/workflows/code-check.yml` 当前主要流程是：

1. checkout
2. 激活预置 virtualenv
3. `pip install --no-dependencies -e .`
4. `pytest tests`

也就是说，CI 假设运行环境已经有 CUDA / Triton 等先决条件，不会在 workflow 里完整装依赖。

## 9. 推荐的阅读顺序

面对不同任务，建议从以下入口开始：

### 9.1 想理解公共 API

按这个顺序读：

1. `README.md` 或 `README_cn.md`
2. `src/flag_attn/__init__.py`
3. 对应 `examples/*.py`

### 9.2 想修改 flash attention

按这个顺序读：

1. `src/flag_attn/flash.py`
2. `src/flag_attn/split_kv.py`
3. `src/flag_attn/total.py`
4. `src/flag_attn/dropout.py`
5. `src/flag_attn/testing/flash.py`
6. `tests/flag_attn/test_flash_attention.py`
7. `benchmark/flash_benchmark.py` / `benchmark/flash_decoding_benchmark.py`

### 9.3 想修改 piecewise attention

按这个顺序读：

1. `src/flag_attn/piecewise.py`
2. `src/flag_attn/testing/piecewise.py`
3. `tests/flag_attn/test_piecewise_attention.py`
4. `benchmark/piecewise_benchmark.py`

### 9.4 想修改 paged attention

按这个顺序读：

1. `src/flag_attn/paged.py`
2. `src/flag_attn/testing/paged.py`
3. `tests/flag_attn/test_paged_attention.py`
4. `examples/paged_example.py`
5. `benchmark/paged_benchmark.py`

## 10. 在这个仓库里改代码时的工作约定

1. 先判断你改的是公共接口、kernel 实现，还是配置 heuristic。
2. 任何公共接口变化，都要同步检查 `__init__.py`、`README.md`、`README_cn.md`、`examples/`。
3. 任何 kernel 逻辑变化，都至少要检查对应 `flag_attn.testing` reference 和 `tests/`。
4. 如果涉及性能优化，不要只看 pytest，通过 benchmark 看吞吐变化。
5. 如果涉及随机路径，比如 dropout，必须保留或更新重算 mask 的验证逻辑。
6. 如果只改了一个路径，但另一路共享辅助输出或公共调度逻辑，例如 `total_attention`、split-kv，不能假设它们天然正确。

## 11. 典型验证命令

安装：

```bash
pip install -e .
```

全量测试：

```bash
pytest tests
```

按模块测试：

```bash
pytest tests/flag_attn/test_flash_attention.py
pytest tests/flag_attn/test_piecewise_attention.py
pytest tests/flag_attn/test_paged_attention.py
pytest tests/flag_attn/test_dropout.py
```

运行示例：

```bash
python examples/flash_attention_example.py
python examples/piecewise_example.py
python examples/paged_example.py
```

运行 benchmark：

```bash
cd benchmark
python flash_benchmark.py
python flash_decoding_benchmark.py
python piecewise_benchmark.py
python paged_benchmark.py
```

## 12. 一句话总结

这个仓库不是“通用训练框架”，而是一个以 Triton attention kernel 为中心的小而专注的算子仓库。理解它的最佳方式不是从外层脚手架入手，而是抓住一条主线：

`公开 API -> autograd Function -> Triton kernel -> testing reference -> pytest -> benchmark`

后续如果要扩展新算子或继续维护现有算子，建议始终围绕这条主线组织修改。
