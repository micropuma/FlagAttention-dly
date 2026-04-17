---
name: flagattention-repo
description: 用于在 FlagAttention 仓库中进行中文分析、实现、调试和文档维护。适合需要理解 Triton attention 算子结构、修改 flash/piecewise/paged attention、补测试、跑 benchmark、同步 README 与示例时使用。
---

# FlagAttention 仓库技能

这个 skill 面向 `FlagAttention` 仓库本身，而不是通用 Triton 项目。触发后，默认按仓库当前结构工作，避免把它误判成训练框架或普通 Python 工具库。

## 先做什么

1. 先读仓库根目录的 `agent.md`，建立整体结构认知。
2. 再根据任务类型定位到对应实现、reference、测试和 benchmark。
3. 如果任务涉及公开 API、返回值或用法变化，最后同步检查 `README.md`、`README_cn.md` 和 `examples/`。

## 仓库地图

- `src/flag_attn/__init__.py`
  公共导出入口，确认外部用户能调用哪些 API。
- `src/flag_attn/flash.py`
  标准 flash attention 主实现，含 autograd、kernel 调度、dropout、aux outputs。
- `src/flag_attn/split_kv.py`
  split-kv / flash decoding 路径和分片 heuristic。
- `src/flag_attn/total.py`
  `return_total_attention=True` 的辅助输出实现。
- `src/flag_attn/dropout.py`
  dropout 的 Philox seed/offset 管理。
- `src/flag_attn/piecewise.py`
  piecewise attention 主实现。
- `src/flag_attn/paged.py`
  paged attention 与 block table / KV cache 访问逻辑。
- `src/flag_attn/testing/`
  所有 PyTorch reference implementation，是测试基线，不是可有可无的辅助目录。
- `tests/flag_attn/`
  参数化正确性测试。
- `examples/`
  API 使用样例。
- `benchmark/`
  性能脚本与外部实现对比。

## 按任务分流

### 修改 flash attention

优先读取：

1. `src/flag_attn/flash.py`
2. `src/flag_attn/split_kv.py`
3. `src/flag_attn/total.py`
4. `src/flag_attn/dropout.py`
5. `src/flag_attn/testing/flash.py`
6. `tests/flag_attn/test_flash_attention.py`

额外要求：

- 留意 MQA/GQA 路径。
- 留意 `return_log_normalizer` / `return_total_attention`。
- 留意 split-kv 是否被一起影响。
- 若改动性能路径，补跑 `benchmark/flash_benchmark.py` 或 `benchmark/flash_decoding_benchmark.py`。

### 修改 piecewise attention

优先读取：

1. `src/flag_attn/piecewise.py`
2. `src/flag_attn/testing/piecewise.py`
3. `tests/flag_attn/test_piecewise_attention.py`
4. `benchmark/piecewise_benchmark.py`

额外要求：

- 保持 `dist_threshold` 语义稳定。
- 前向 score 拼接逻辑与反向梯度拆分逻辑必须成对检查。

### 修改 paged attention

优先读取：

1. `src/flag_attn/paged.py`
2. `src/flag_attn/testing/paged.py`
3. `tests/flag_attn/test_paged_attention.py`
4. `examples/paged_example.py`
5. `benchmark/paged_benchmark.py`

额外要求：

- 明确输入布局不是 `(B, H, T, D)`，而是 query + KV cache + block tables。
- 检查 `query_group_size`、`num_kv_heads`、`block_size`、`num_splits`。

## 这个仓库里的默认判断

- 这是一个“算子仓库”，不是训练框架。
- 正确性验证默认依赖 `flag_attn.testing` 中的 reference implementation。
- 大多数测试默认需要 CUDA。
- 公开接口变化必须联动文档和示例。
- README 当前不一定完整覆盖所有已导出的 API，最终以 `src/flag_attn/__init__.py` 为准。

## 推荐验证步骤

优先级从高到低：

1. `pytest tests/flag_attn/test_xxx.py`
2. `pytest tests`
3. 相关 `examples/*.py`
4. 相关 `benchmark/*.py`

常用命令：

```bash
pip install -e .
pytest tests/flag_attn/test_flash_attention.py
pytest tests/flag_attn/test_piecewise_attention.py
pytest tests/flag_attn/test_paged_attention.py
cd benchmark && python flash_benchmark.py
```

## 文档同步规则

发生以下情况时，顺手同步文档：

- 公共函数签名变化
- 新增导出 API
- 返回值变化
- 示例脚本已过时
- benchmark 入口或运行方式变化

优先检查：

- `README.md`
- `README_cn.md`
- `examples/`
- `agent.md`

## 输出风格

- 默认用中文总结仓库结构、模块边界和改动影响。
- 优先解释“这个文件为什么存在”以及“改它还要看哪里”。
- 对性能优化类改动，不只描述代码差异，还要说明它影响的是调度、访存、并行度还是辅助输出路径。
