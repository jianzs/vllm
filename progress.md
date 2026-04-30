# DyCP Progress

## 当前状态：P1 实现阶段（完成）

### 已完成 (P0) ✓

1. **`vllm/config/parallel.py`** — `cp_size_thresholds` 字段 + 全量校验
   - 新增 `dycp_enabled`, `dycp_sorted_thresholds`, `dycp_all_cp_sizes`, `dycp_max_cp_size` 属性
   - 校验逻辑：PCP/DCP 互斥、cp_size 必须是 2 的幂且是 dp_per_domain 的因子

2. **`vllm/engine/arg_utils.py`** — `--cp-size-thresholds` CLI 参数 + 解析
   - `_parse_cp_size_thresholds()` 使用 `ast.literal_eval` 解析 JSON-like 字符串

3. **`vllm/v1/core/sched/request_queue.py`** — `get_cp_size_for_request()` 阈值映射函数

4. **`vllm/v1/core/sched/cross_dp_scheduler.py`** — DyCP 调度逻辑
   - `RequestManager.select_dp()` 新增 `cp_size` 参数，支持对齐子组选择
   - `_schedule()` 中集成 DyCP 阈值查表、per-req cp_size 跟踪、batch 级 actual_cp_size 计算

5. **`vllm/v1/core/sched/output.py`** — SchedulerOutput 新增 `actual_cp_size`, `per_req_cp_sizes`

6. **`vllm/distributed/parallel_state.py`** — NCCL 对齐子组创建 + `get_dycp_subgroup(cp_size)`

7. **`vllm/v1/core/cross_dp_kv_cache_manager.py`** — cp_ranks 长度约束改为 power-of-2 + factor 校验

### 已完成 (P1) ✓

1. **`vllm/v1/worker/block_table.py`** — per-request interleave (`compute_domain_slot_mapping`)
   - 新增 `per_req_cp_sizes` 参数，支持每个请求独立的 cp_size/cp_rank 计算
   - 向量化 interleave：`token_cp_sizes`, `token_cp_ranks` 替代标量
   - 保持向后兼容：None 时退化为固定 total_cp_world_size 路径

2. **`vllm/v1/worker/cp_utils.py`** — PCPManager 参数化
   - `update_tokens_for_pcp()` 新增 `effective_pcp_world_size` 参数
   - 方法内使用局部变量 `pcp_world_size`/`pcp_rank`，支持 DyCP 传入 actual_cp_size
   - `pcp_rank` 运行时从 `dycp_rank % effective_world_size` 推导

3. **`vllm/v1/worker/gpu_model_runner.py`** — 传递 DyCP 参数
   - 构建 `per_req_cp_sizes_np` 数组传递给 block_table
   - DyCP prefill 路径传递 `actual_cp_size` 给 PCPManager

- `vllm/v1/worker/block_table.py` — per-request interleave (`compute_domain_slot_mapping`)
- `vllm/v1/worker/cp_utils.py` — PCPManager 参数化
- `vllm/v1/worker/gpu_model_runner.py` — 传递 `actual_cp_size`，构造 per-req arrays

### 已完成 (P2) ✓

1. **`vllm/v1/attention/backends/mla/common.py`** — decode 用 `get_dycp_subgroup(cp_size)`
   - MLACommonMetadata 新增 `actual_cp_size` 字段
   - decode all-reduce 和 prefill allgather 根据 actual_cp_size 选择子组
   - `cp_size < dycp_world_size` 时用 subgroup，等于时用 full group

2. **`vllm/v1/attention/backends/utils.py`** — CommonAttentionMetadata 加 `actual_cp_size`

3. **`vllm/forward_context.py`** — `BatchDescriptor` 加 `cp_size` 字段（第 3 维 graph key）

4. **`vllm/v1/cudagraph_dispatcher.py`** — graph key 加 `cp_size` 维度 + 剪枝
   - `_create_padded_batch_descriptor` 接受 cp_size
   - `initialize_cudagraph_keys` 使用 DyCP 剪枝：cp_tokens=0→cp_size=1, cp_tokens>0→cp_size>1
   - `dispatch` 接受 cp_size 参数

5. **`vllm/v1/worker/gpu_model_runner.py`** — 传递 actual_cp_size 到 dispatcher 和 attention metadata
   - `_determine_batch_execution_and_padding` 新增 actual_cp_size 参数
   - `_dummy_run` 新增 actual_cp_size 参数
   - `_capture_cudagraphs` 使用 4-tuple (num_tokens, lora, cp_tokens, cp_size)
   - 使用 `_build_cp_cases` 帮助函数生成带剪枝的 graph capture 组合

### 未完成 (P2 剩余)

- `vllm/attention/ops/common.py` — `CPTritonContext` 字典化（按 cp_size 索引 triton context）
- `cp_local_seq_lens` per-request 化（当前使用固定 dycp_world_size，decode 中不同 cp_size 请求共存时需要 per-request 值）

## 下一步

1. **P2 剩余**: `vllm/attention/ops/common.py` — `CPTritonContext` 字典化
2. **P2 剩余**: `cp_local_seq_lens` per-request 化
3. **测试**: 同步到远程机器，运行端到端测试验证 DyCP 功能

## 发现的问题

- 暂无

## 会话记录

### 2026-04-30 Session 1
- 恢复上下文，审计当前代码变更
- 确认 P0 层（config、scheduler、parallel_state）已基本完成
- 识别出剩余 P0 工作：cross_dp_kv_cache_manager 硬约束

### 2026-04-30 Session 2
- 确认所有 P0 工作已完成（7 个文件的修改均已到位）
- cross_dp_kv_cache_manager.py 硬约束已替换为 power-of-2 + factor 校验
- 提交 P0 commit
- 完成 P1 工作：
  - block_table.py: per-request interleave，支持 per_req_cp_sizes 向量化计算
  - cp_utils.py: PCPManager 新增 effective_pcp_world_size 参数，运行时推导 pcp_rank
  - gpu_model_runner.py: 构建 per_req_cp_sizes_np 传给 block_table，actual_cp_size 传给 PCPManager
- 提交 P1 commit
- 完成 P2 主要工作：
  - MLA attention: decode/prefill 路径使用 get_dycp_subgroup(cp_size) 替代 get_dycp_group()
  - CommonAttentionMetadata + MLACommonMetadata: 新增 actual_cp_size 字段，从 scheduler 传递到 attention
  - BatchDescriptor: 新增 cp_size 字段（CUDA graph 第 3 维 key）
  - cudagraph_dispatcher: 3D graph key 生成 + 剪枝（cp_tokens=0→cp_size=1, cp_tokens>0→cp_size>1）
  - gpu_model_runner: 传递 actual_cp_size 到 dispatcher 和 attention，capture loop 使用 4-tuple
- 提交 P2 commit
