# DyCP Progress

## 当前状态：P2 实现阶段（完成）→ 测试与修复阶段

### 已完成 (P0) ✓

1. **`vllm/config/parallel.py`** — `cp_size_thresholds` 字段 + 全量校验
   - 新增 `dycp_enabled`, `dycp_sorted_thresholds`, `dycp_all_cp_sizes`, `dycp_max_cp_size` 属性
   - 校验逻辑：PCP/DCP 互斥、cp_size 必须是 2 的幂且是 dp_per_domain 的因子

2. **`vllm/engine/arg_utils.py`** — `--cp-size-thresholds` CLI 参数 + 解析

3. **`vllm/v1/core/sched/request_queue.py`** — `get_cp_size_for_request()` 阈值映射函数

4. **`vllm/v1/core/sched/cross_dp_scheduler.py`** — DyCP 调度逻辑
   - `RequestManager.select_dp()` 新增 `cp_size` 参数，支持对齐子组选择
   - `_schedule()` 中集成 DyCP 阈值查表、per-req cp_size 跟踪、batch 级 actual_cp_size 计算

5. **`vllm/v1/core/sched/output.py`** — SchedulerOutput 新增 `actual_cp_size`, `per_req_cp_sizes`

6. **`vllm/distributed/parallel_state.py`** — NCCL 对齐子组创建 + `get_dycp_subgroup(cp_size)`

7. **`vllm/v1/core/cross_dp_kv_cache_manager.py`** — cp_ranks 长度约束改为 power-of-2 + factor 校验

### 已完成 (P1) ✓

1. **`vllm/v1/worker/block_table.py`** — per-request interleave (`compute_domain_slot_mapping`)

2. **`vllm/v1/worker/cp_utils.py`** — PCPManager 参数化

3. **`vllm/v1/worker/gpu_model_runner.py`** — 传递 DyCP 参数

### 已完成 (P2) ✓

1. **`vllm/v1/attention/backends/mla/common.py`** — decode 用 `get_dycp_subgroup(cp_size)`

2. **`vllm/v1/attention/backends/utils.py`** — CommonAttentionMetadata 加 `actual_cp_size`

3. **`vllm/forward_context.py`** — `BatchDescriptor` 加 `cp_size` 字段

4. **`vllm/v1/cudagraph_dispatcher.py`** — graph key 加 `cp_size` 维度 + 剪枝

5. **`vllm/v1/worker/gpu_model_runner.py`** — 传递 actual_cp_size 到 dispatcher 和 attention metadata

6. **`vllm/attention/ops/common.py`** — CPTritonContext 字典化

7. **`vllm/v1/attention/backends/utils.py`** — `get_cp_local_seq_lens` 支持 per-request cp_world_size tensor

8. **`vllm/v1/attention/backends/flash_attn.py`** — decode 路径使用 `get_dycp_subgroup(cp_size)`

9. **`vllm/v1/attention/backends/flashinfer.py`** — decode 路径使用 `get_dycp_subgroup(cp_size)`

10. **`vllm/v1/worker/cp_utils.py`** — PCPManager allgather 使用 `get_dycp_subgroup(actual_cp_size)`

### 测试与修复阶段（进行中）

#### 端到端测试结果

- **CP=1（短请求 <4K tokens）**: ✓ 通过
- **CP=4（17K+ tokens）**: ✗ CUDA index out of bounds 崩溃

#### 发现的问题与修复

1. **MLA workspace 分配不足** ✓ 已修复
   - 问题：workspace 使用 `dycp_world_size=8` 分配，但 CP=4 时每个 rank 有 2x 的本地数据
   - 修复：使用 `dycp_min_cp_size`（阈值中最小的 cp_size > 1）分配 workspace
   - 文件：`vllm/v1/attention/backends/mla/common.py`

2. **`_context_parallel_compute_prefill_context` 使用错误的 cp_world_size** ✓ 已修复
   - 问题：传入 `self.dycp_world_size=8` 但实际 allgather 使用 `actual_cp_size=4`
   - 修复：传入 `attn_metadata.actual_cp_size`
   - 文件：`vllm/v1/attention/backends/mla/common.py`

3. **MLA chunked context metadata 使用错误的 virtual block size** ✓ 已修复
   - 问题：使用 `self.cp_virtual_block_size`（基于 dycp_world_size=8）但 CP=4 时 virtual block size 应为 `block_size * 4`
   - 修复：在 `_build_mixed_dycp_dp_prefill` 和标准路径中使用 `actual_cp_size` 计算 virtual block size
   - 文件：`vllm/v1/attention/backends/mla/common.py`

4. **`max_num_blocks_per_req` 不足** ✓ 已修复
   - 问题：使用 `total_cp_world_size=8` 计算，但 CP=4 时需要更多 blocks
   - 修复：DyCP 启用时使用 `min(dycp_all_cp_sizes)` 计算
   - 文件：`vllm/v1/worker/block_table.py`

5. **CP=4 请求仍然崩溃** — 系统性修复 `dycp_world_size` → `actual_cp_size`
   - 上述修复已应用但 CP=4 请求仍然导致 CUDA index out of bounds
   - 系统性审计所有 `self.dycp_world_size` 引用，发现并修复以下问题：

6. **`get_cp_local_seq_lens` 使用错误的 cp_world_size** ✓ 已修复
   - 问题：`mla/common.py` 和 `flash_attn.py`/`flashinfer.py` 的 decode 路径使用 `self.dycp_world_size` 而非 `actual_cp_size`
   - 修复：改用 `actual_cp_size` 和 `self.dycp_rank % actual_cp_size`
   - 文件：`mla/common.py`, `flash_attn.py`, `flashinfer.py`, `gpu_model_runner.py`

7. **PCP KV indices 使用错误的 cp_size 和 rank** ✓ 已修复
   - 问题：`get_pcp_kv_indices` 和 `get_pcp_query_indices` 使用 `self.dycp_world_size` 和 `self.dycp_rank`
   - 修复：改用 `actual_cp_size` 和 `self.dycp_rank % actual_cp_size`
   - 文件：`mla/common.py` (lines 1114-1120, 1620-1628)

8. **`local_context_lens_allranks` tensor 形状和 rank 索引错误** ✓ 已修复
   - 问题：tensor 使用 `(num_prefills, self.dycp_world_size)` 形状，rank 使用 `self.dycp_rank`
   - 修复：改用 `(num_prefills, actual_cp_size)` 和 `self.dycp_rank % actual_cp_size`
   - 文件：`mla/common.py` (lines 1488-1496)

9. **DualChunkSwap prefill 路径使用错误的 cp_size 和 rank** ✓ 已修复
   - 问题：kv_head_seq_lens/kv_tail_seq_lens 使用 `self.dycp_world_size` 和 `self.dycp_rank`
   - 修复：改用 `prefill.actual_cp_size` 和 `self.dycp_rank % prefill.actual_cp_size`
   - 文件：`mla/common.py` (lines 2371-2379)

10. **workspace 分区在 DyCP 模式下不兼容** ✓ 已修复
    - 问题：`workspace.shape[0] // (cp_world_size + 1)` 在 `actual_cp_size != dycp_min_cp_size` 时不能整除
    - 修复：DyCP 模式使用固定分区 `chunked_prefill_workspace_size // dycp_min_cp_size`
    - 文件：`mla/common.py` (lines 2849-2870)

11. **`prefill_k_start` 使用错误的 cp_size** ✓ 已修复
    - 问题：`num_decode_tokens * self.dycp_world_size` 应为 `num_decode_tokens * actual_cp_size`
    - 文件：`mla/common.py` (line 3288)

12. **`MLACommonPrefillMetadata` 缺少 `actual_cp_size`** ✓ 已修复
    - 添加 `actual_cp_size` 字段到 prefill metadata，并在构建时设置
    - 文件：`mla/common.py`

13. **`dycp_allgathered_size` 和 `dycp_allgather_size` 使用错误的 cp_size** ✓ 已修复
    - 问题：`gpu_model_runner.py` 中使用 `self.dycp_world_size` 而非 `actual_cp_size`
    - 文件：`gpu_model_runner.py` (lines 1744, 1991)

14. **`actual_cp_size` 变量作用域错误** ✓ 已修复
    - 问题：非 mixed DyCP 路径中 `actual_cp_size` 未定义
    - 修复：在 `is_dycp_mixed` 判断后立即定义 `actual_cp_size`
    - 文件：`mla/common.py`

#### 当前状态

- **CP=1（短请求 <4K tokens）**: ✓ 通过
- **CP=2（4K-16K tokens）**: ✓ 通过
- **CP=4（16K-32K tokens）**: ✓ 通过
- **CP=8（32K+ tokens）**: ✓ 通过
- **Mixed batch（CP=1/2/4/8 并发）**: ✓ 通过
- **Decode path（CP=2/CP=4 + 50 output tokens）**: ✓ 通过

15. **`get_logits_indices` 使用 `pcp_world_size` 而非 `actual_cp_size`** ✓ 已修复
    - 问题：DyCP 模式下 logit 采样索引计算错误，导致 CUDA OOB
    - 修复：添加 `effective_world_size` 参数，DyCP 时传入 `actual_cp_size`
    - 文件：`cp_utils.py`

16. **`get_discard_request_mask` 使用 `pcp_world_size` 而非 `actual_cp_size`** ✓ 已修复
    - 问题：DyCP 模式下请求完成判断错误
    - 修复：添加 `effective_world_size` 参数，DyCP 时传入 `actual_cp_size`
    - 文件：`cp_utils.py`

17. **`MLACommonImpl` 缺少 `dycp_min_cp_size` 等属性** ✓ 已修复
    - 问题：`dycp_min_cp_size`、`cp_world_size`、`cp_local_block_size`、`cp_virtual_block_size`、`chunked_prefill_workspace_size`、`cp_kv_cache_interleave_size` 只在 `MLACommonMetadataBuilder.__init__` 中设置，但 `_context_parallel_compute_prefill_context` 在 `MLACommonBaseImpl`/`MLACommonImpl` 实例上调用
    - 修复：在 `MLACommonImpl.__init__` 中添加这些属性
    - 文件：`mla/common.py`

18. **`local_pd_connector.py` IPC KV load rank 映射错误** ✓ 已修复
    - 问题：`_start_load_kv_ipc` 中 `owning_ranks` 是 CP 相对 rank（0..cp_world_size-1），但 `my_rank` 是全局 DyCP group rank（0..dp_per_domain-1），导致 local/remote copy 判断和 IPC handle 查找错误
    - 修复：计算 `my_cp_rank` 和 `subgroup_start`，使用 `subgroup_start + src_rank` 查找 IPC handle，使用 `my_cp_rank` 判断 local rank
    - 同时修正 `save_kv_layer` 中 `cp_size` → `num_dycp_reqs` 变量名
    - 文件：`local_pd_connector.py`

#### 待完成

- 运行 benchmark 对比 DyCP 性能与 DP 基线
- PD 分离端到端测试（需要 proxy 路由 prefill/decode 请求）

## 会话记录

### 2026-05-01 Session 1
- 恢复上下文，审计当前代码变更
- 提交 P2 扩展改动（flash_attn, flashinfer, cp_utils, gpu_model_runner 中的 DyCP subgroup 选择）
- 端到端测试：CP=1 通过，CP=4 崩溃
- 根因分析：MLA workspace 和 prefill context 使用 dycp_world_size 而非 actual_cp_size
- 修复4个关键问题：workspace 分配、prefill context cp_world_size、chunked context virtual block size、max_num_blocks_per_req
- CP=4 请求仍然崩溃，需要进一步调查

### 2026-05-01 Session 2
- 系统性审计所有 `self.dycp_world_size` 引用（mla/common.py, flash_attn.py, flashinfer.py, gpu_model_runner.py）
- 修复14个关键问题：get_cp_local_seq_lens、PCP KV indices、local_context_lens_allranks、DualChunkSwap、workspace 分区、prefill_k_start、prefill metadata、dycp_allgather_size、actual_cp_size 作用域
- CP=1 测试通过，CP=2 和 CP=4 仍然崩溃（CUDA index out of bounds）
- 崩溃点在 `hidden_states[logits_indices]`，需要进一步调试

### 2026-05-01 Session 3
- 修复 `get_logits_indices` 和 `get_discard_request_mask`：添加 `effective_world_size` 参数，DyCP 时传入 `actual_cp_size`
- CP=2 测试通过！CP=4 报 `AttributeError: 'FlashMLAImpl' object has no attribute 'dycp_min_cp_size'`
- 根因：`dycp_min_cp_size` 等属性只在 `MLACommonMetadataBuilder.__init__` 中设置，但 `_context_parallel_compute_prefill_context` 在 `MLACommonBaseImpl`/`MLACommonImpl` 实例上调用
- 修复：在 `MLACommonImpl.__init__` 中添加 `dycp_min_cp_size`、`cp_world_size`、`cp_local_block_size`、`cp_virtual_block_size`、`chunked_prefill_workspace_size`、`cp_kv_cache_interleave_size`
- CP=1/2/4/8 全部通过！Mixed batch 和 decode path 也通过
- 移除 debug sync/checks

### 2026-04-30 Session 3
- 完成 P2 剩余工作
- P2 全部完成
- 下一步：同步到远程机器，运行端到端测试验证 DyCP 功能

### 2026-05-01 Session 4
- 修复 `local_pd_connector.py` IPC KV load 的 DyCP subgroup 兼容性问题
  - `_start_load_kv_ipc` 中 `owning_ranks` 是 CP 相对 rank，但 `my_rank` 是全局 rank
  - 修复：`my_global_rank` → `my_cp_rank = my_global_rank % cp_world_size`，`subgroup_start + src_rank` 用于 IPC handle 查找
  - 同时修正 `save_kv_layer` 中的误导变量名 `cp_size` → `num_dycp_reqs`
- 端到端测试验证（DyCP + CrossDPExampleConnector）：
  - CP=1（短请求 <4K）：✓ 通过
  - CP=2（8K tokens）：✓ 通过
  - CP=4（20K tokens）：✓ 通过
  - CP=8（40K tokens）：✓ 通过
  - Mixed batch（CP=1/2/4/8 并发）：✓ 通过
  - Decode path（CP=2/CP=4 + 50 output tokens）：✓ 通过
- DyCP+LocalPDConnector 联合测试：
  - 服务器启动正常，IPC handles 交换成功（27 layers × 8 ranks）
  - 短请求和长请求（CP=1/2/4/8）均正常返回
  - PD 的 IPC KV load 路径需要 proxy 才能触发，当前测试验证了初始化和基本功能
- 下一步：调查 CP=4/8 decode 性能回归根因，优化 NCCL 通信开销

### Benchmark 结果（2026-05-01）

**测试环境**: DeepSeek-V2-Lite, 8×GPU, dp_per_domain=8, FLASHMLA

**DP Baseline（无 DyCP）**:
| 场景 | Input | Output | TTFT P50 | TPOT P50 |
|------|-------|--------|-----------|-----------|
| Prefill | 4K | 1 | 33.70ms | - |
| Decode | 4K | 1024 | 43.15ms | 9.58ms |

**DyCP 结果**:
| 场景 | Input | Output | CP Size | TTFT P50 | TPOT P50 | 备注 |
|------|-------|--------|---------|-----------|-----------|------|
| Prefill | 4K | 1 | CP=1 | 34.00ms | - | 与基线对齐 |
| Decode | 4K | 1024 | CP=1 | 42.54ms | 9.38ms | 与基线对齐 |
| Prefill | 8K | 1 | CP=2 | 52.59ms | - | 正常 |
| Decode | 8K | 1024 | CP=2 | 61.36ms | 9.70ms | 与基线接近 |
| Prefill | 20K | 1 | CP=4 | 119.97ms | - | 正常 |
| Decode | 20K | 1024 | CP=4 | 148.56ms | 80.61ms | 性能回归！ |
| Prefill | 40K | 1 | CP=8 | 272.70ms | - | 正常 |

**关键发现**:
- CP=1/2 的 prefill 和 decode 性能与 DP 基线对齐
- CP=4/8 的 prefill 性能正常（TTFT 按预期增长）
- CP=4 decode 性能严重回归（TPOT P50 80.61ms vs 基线 9.58ms，~8.4x 慢）
- 可能原因：CP=4/8 decode 路径的 NCCL allgather/allreduce 通信开销过大
- 需要进一步 profiling 确认瓶颈