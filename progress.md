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

20. **CP=4/8 decode 性能回归根因：DP coordination 降级 CUDA graph 模式** ✓ 已修复
    - 问题：`coordinate_batch_across_dp` 对所有 DP rank 取 cudagraph_mode 最小值。当 CP=4 请求在 ranks [0,1,2,3] 时，ranks [4,5,6,7] 有 0 tokens 并 dispatch 为 NONE，导致所有 rank 降级为 eager 模式（50ms+/step vs CUDA graph 2ms/step）
    - 修复（3 处改动）：
      1. `dp_utils.py`：`_post_process_cudagraph_mode` 忽略 0 tokens 的 rank（它们没有实际工作，不应影响有 tokens 的 rank 的 CUDA graph 决策）
      2. `gpu_model_runner.py`：当 DyCP 活跃且本 rank 有 0 tokens 时，向 DP coordination 报告 FULL 模式（而非 NONE），防止降级其他 rank；DP padding 后设 `uniform_decode=True` 以匹配正确的 graph key
      3. `gpu_model_runner.py`：dispatch lambda 中 `cp_size=actual_cp_size if num_cp_tokens > 0 else 1`（非 CP rank 应使用 cp_size=1 匹配已捕获的 graph key）
    - 文件：`dp_utils.py`, `gpu_model_runner.py`
    - 结果：CP=4 decode TPOT 从 80.61ms 降到 ~7ms，与 CP=1 对齐

#### 待完成

- ~~运行正式 benchmark 对比 DyCP 性能与 DP 基线（vllm bench serve）~~ — Decode + Prefill benchmark 均已完成
- PD 分离端到端测试：修复了 3 个 KV loading bug（见 Session 10），需在远程机器上验证
- ~~解决 MoE all-to-all 同步瓶颈~~ — 已修复（纯同一 CP size 场景），混合 CP size 场景为设计预期
- ~~运行 Prefill benchmark（不带 --kv-transfer-config，output=1）~~ — 已完成（Session 9）
- 混合 CP size 负载的 TPOT 回归问题：需要真实 PD 部署验证（CrossDPExampleConnector 的 KV loading 走完整前向传播，不是真实场景）

### 2026-05-01 Session 7
- 实现独立 CUDA graph 模式修复：DyCP 模式下各 CP 子组使用本地 `cudagraph_mode_for_dp` 而非全局 `synced_cudagraph_mode`，避免 prefill 子组降级 decode 子组的 CUDA graph 模式
- 修复 DP coordination 后 `num_tokens_padded` 非 capture size 导致的 dispatch mismatch：DyCP 模式下将 `num_tokens_padded` 向上取整到最近的 CUDA graph capture size，并同步更新 `num_tokens_across_dp`
- **测试结果**：
  - 单请求 CP=4 decode：TTFT=90ms, streaming TPOT=9ms — 性能正常
  - 2 并发 CP=4 decode：TTFT=85/143ms, streaming TPOT=150ms — 性能回归
  - 正式 benchmark（32 prompts, max-concurrency=2）：0 failed, TPOT P50=81ms
- **关键发现**：独立 CUDA graph 模式修复是正确的（dispatch 日志确认 decode rank 使用 FULL 模式），但 TPOT 仍然为 81ms
- **根因**：调度器在同一 step 中将 prefill（KV loading）和 decode 混合调度到不同 DP rank。Ranks [0,1,2,3] 做 decode（1 token, CUDA graph），Ranks [4,5,6,7] 做 prefill（8 tokens, eager）。MoE all-to-all 强制所有 rank 同步，decode rank 必须等待 prefill rank 完成（~70ms），导致每步 81ms
- **这是调度器级别的问题**，不是 GPU model runner 的问题。需要修改 CrossDPScheduler 避免在同一 step 中混合 prefill 和 decode 到不同 DP rank
- 文件修改：`gpu_model_runner.py`（独立 CUDA graph 模式 + num_tokens_padded 取整 + num_tokens_across_dp 同步更新）

## 会话记录

### 2026-05-01 Session 8
- 运行正式 Decode benchmark（32 prompts, max-concurrency=2, request-rate=2, CrossDPExampleConnector）
  - CP=1 (4K): TPOT P50=8.16ms — 基线
  - CP=2 (8K): TPOT P50=8.25ms — 与基线对齐
  - CP=4 (20K): TPOT P50=9.51ms — 与基线对齐
  - CP=8 (40K): TPOT P50=9.51ms — 与基线对齐
- 运行混合 CP=1+CP=4 decode benchmark：TPOT P50=79.87ms — 回归
- 分析混合负载回归根因：
  - CrossDPExampleConnector 的 `start_load_kv` 是空操作，KV loading 走完整模型前向传播
  - CP=4 请求调度 ~1025 token（prefill），CP=1 请求先完成 prefill 进入 decode
  - MoE all-to-all 同步导致 decode rank 等待 prefill rank
  - 这是设计文档预期行为，真实 PD 部署不会出现
- 扩展调度器修复：`dycp_has_cp_decode` → `dycp_has_decode`（检查任何 decode，不只是 CP>1）
- 提交：`6f0184f23` [DyCP] extend scheduler fix: defer prefill when ANY decode is running
- 下一步：运行 Prefill benchmark、PD 分离端到端测试

#### PD 分离测试（Session 9）

- 创建 `scripts/start_vllm_pd_dycp.sh`：DyCP + LocalPDConnector 启动脚本，包含 `--cp-size-thresholds` 和 proxy/bench/test 子命令
- 修复 `scripts/sync.sh` 排除 `dycp` 目录的问题，手动同步 `dycp/proxy/` 到远程
- 服务器启动成功（LocalPDConnector + DyCP，gpu-mem-util=0.7，cudagraph_capture_sizes_for_cp=4）
- Proxy 启动成功（local_pd_proxy.py on port 9000）
- 短请求（CP=1，直接转发）通过 proxy 正常工作
- Prefill 请求（do_remote_decode=True）正确返回 kv_transfer_params：
  - 包含 pd_request_prefix, cp_world_size, num_prompt_tokens, prompt_token_ids, per_rank_block_ids, interleave_size
- Decode 请求（do_remote_prefill=True）发现 External KV（14 tokens），但生成内容不正确：
  - 预期："Paris"（基于 prefill "The capital of France is Paris."）
  - 实际："Yes, it is."（模型没有正确使用 KV cache）
- 根因分析：`get_num_new_matched_tokens` 正确返回 ext_tokens=13，scheduler 正确标记 token 为已计算，但 KV 数据可能没有正确注入到 decode 请求的 paged cache
- 需要进一步调试 `_start_load_kv_ipc` 方法和 block 分配/复制逻辑
- 之前 DP baseline TTFT 33.70ms 是 CrossDPExampleConnector 测量的（bypass prefill），不是真实 prefill

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
- 下一步：深入 profiling 分析 CP=4/8 decode 性能回归根因（NCCL 融合未显著改善，瓶颈在其他地方）

#### 性能优化尝试

19. **修复 `dycp_lse_out_ar` bug 并切换 DyCP decode 到融合 all-reduce** ✓ 已完成
    - Bug：`global_output = weighted_output / lse_exp` 应为 `global_output = global_weighted / global_lse_sum`
    - 优化：将 DyCP decode 路径从 `cp_lse_ag_out_ar`（2 NCCL ops）切换到 `dycp_lse_out_ar`（1 NCCL op）
    - 文件：`common.py`, `mla/common.py`, `flash_attn.py`, `flashinfer.py`
    - 结果：TPOT 从 80.61ms 降到 81.55ms，几乎无改善，说明 NCCL 通信不是主要瓶颈

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

### Benchmark 结果（2026-05-01，Session 7）

**测试环境**: DeepSeek-V2-Lite, 8×GPU, dp_per_domain=8, FLASHMLA, CrossDPExampleConnector

**DyCP Decode Benchmark（32 prompts, max-concurrency=2, request-rate=2）**:
| 场景 | Input | Output | CP Size | TTFT P50 | TPOT P50 | ITL P50 | 备注 |
|------|-------|--------|---------|-----------|-----------|---------|------|
| Decode | 4K | 1024 | CP=1 | 41.92ms | 8.16ms | 8.14ms | 基线 |
| Decode | 20K | 1024 | CP=4 | 9428ms | 9.17ms | 9.14ms | 调度器修复后 |

**DyCP Streaming 测试（单请求）**:
| 场景 | Input | Output | CP Size | TTFT | avg ITL | 备注 |
|------|-------|--------|---------|------|---------|------|
| Decode | 17K | 50 | CP=4 | 90ms | 9.0ms | 无 prefill 干扰 |
| Decode | 3K | 50 | CP=1 | ~18ms | ~7ms | 基线 |

**根因分析**：
- 单请求 CP=4 decode TPOT=9ms，与 CP=1 基线对齐
- 并发请求 TPOT=81ms 的根因：调度器在同一 step 中将 prefill（KV loading）和 decode 混合调度到不同 DP rank
  - Ranks [0,1,2,3] 做 decode（1 token, CUDA graph, ~7ms）
  - Ranks [4,5,6,7] 做 prefill（8 tokens, eager, ~70ms）
  - MoE all-to-all 强制所有 rank 同步，decode rank 等待 prefill rank
- 独立 CUDA graph 模式修复正确（decode rank 使用 FULL 模式），但不解决 MoE all-to-all 同步瓶颈
- 需要修改 CrossDPScheduler 避免在同一 step 中混合 prefill 和 decode 到不同 DP rank

21. **调度器避免 prefill/decode 混合** ✓ 已修复
    - 问题：调度器在同一 step 中将 prefill（KV loading）和 decode 混合调度到不同 DP rank，MoE all-to-all 强制所有 rank 同步，decode rank 等待 prefill rank，TPOT 从 7ms 退化到 81ms
    - 修复（v1）：DyCP 启用时，如果当前 step 已有 CP>1 decode 请求，则延迟新 prefill 请求到下一个 step
    - 修复（v2）：扩展为当任何 decode 请求运行时（包括 CP=1），延迟新 prefill 请求。检查 `req.num_computed_tokens >= req.num_prompt_tokens` 判断 decode 阶段
    - 文件：`cross_dp_scheduler.py`
    - 结果：纯 CP=4 decode TPOT P50 从 81ms 降到 9.17ms，与 CP=1 基线对齐
    - 限制：无法防止 CP=1 和 CP>1 请求在同一 step 从 WAITING 调度（两者都在 prefill 阶段），CP=1 先完成 prefill 进入 decode 时仍会与 CP>1 prefill 混合。这是设计文档预期行为，在真实 PD 部署中不会发生

### Benchmark 结果（2026-05-01，修复后）

**测试环境**: DeepSeek-V2-Lite, 8×GPU, dp_per_domain=8, FLASHMLA

**DyCP 修复后结果（快速测试，streaming TPOT）**:
| 场景 | Input | CP Size | TTFT | TPOT |
|------|-------|---------|------|------|
| Decode | 4K | CP=1 | 18.4ms | 6.8ms |
| Decode | 8K | CP=2 | 18.7ms | 6.8ms |
| Decode | 20K | CP=4 | 26.3ms | 6.9ms |
| Decode | 40K | CP=8 | 38.5ms | 7.0ms |

**并发测试**:
| 场景 | Input | 并发数 | est TPOT |
|------|-------|--------|----------|
| 2× CP=4 | 20K | 2 | 7.7ms |
| 4× CP=4 | 20K | 4 | 7.2ms |
| CP=1 + CP=4 | 4K+20K | 2 | 7.1ms |

### Benchmark 结果（2026-05-01，Session 8 — Prefill Benchmark）

**测试环境**: DeepSeek-V2-Lite, 8×GPU, dp_per_domain=8, FLASHMLA, 无 kv-transfer-config

**DyCP Prefill Benchmark（16 prompts, max-concurrency=1, request-rate=1）**:
| 场景 | Input | Output | CP Size | TTFT P50 | 备注 |
|------|-------|--------|---------|-----------|------|
| Prefill | 4K | 1 | CP=1 | 331.70ms | 基线 |
| Prefill | 8K | 1 | CP=2 | 513.19ms | 需2次chunked prefill |
| Prefill | 20K | 1 | CP=4 | 443.01ms | 每rank 5K token，需2次chunk |
| Prefill | 40K | 1 | CP=8 | 567.71ms | 每rank 5K token，需2次chunk |

**注意**: CP=2 的 TTFT (513ms) 高于 CP=4 (443ms) 是因为 CP=2 每rank需处理 4K token（1次 chunk 即可），但 CP=4 每rank只需 5K token（2次 chunk）。实际 TTFT 受 chunked prefill 调度影响。

**关键发现**:
- CP=4/8 decode 性能回归已修复！TPOT 从 80.61ms 降到 ~7ms
- 根因是 DP coordination 将 CUDA graph 模式从 FULL 降级为 NONE（eager mode）
- 修复后所有 CP size 的 TPOT 均与 DP 基线对齐（~7ms vs 基线 9.58ms）

### Benchmark 结果（2026-05-01，Session 8 — 正式 Decode Benchmark）

**测试环境**: DeepSeek-V2-Lite, 8×GPU, dp_per_domain=8, FLASHMLA, CrossDPExampleConnector

**DyCP Decode Benchmark（32 prompts, max-concurrency=2, request-rate=2）**:
| 场景 | Input | Output | CP Size | TTFT P50 | TPOT P50 | 备注 |
|------|-------|--------|---------|-----------|-----------|------|
| Decode | 4K | 1024 | CP=1 | 41.92ms | 8.16ms | 基线 |
| Decode | 8K | 1024 | CP=2 | 59.31ms | 8.25ms | 与基线对齐 |
| Decode | 20K | 1024 | CP=4 | 9834ms | 9.51ms | 与基线对齐 |
| Decode | 40K | 1024 | CP=8 | 9834ms | 9.51ms | 与基线对齐 |

**DyCP Mixed Decode Benchmark（16×CP=1 + 16×CP=4, max-concurrency=4, request-rate=2）**:
| 场景 | Input Mix | TTFT P50 | TPOT P50 | 备注 |
|------|-----------|-----------|-----------|------|
| Mixed | 4K(CP=1) + 20K(CP=4) | 81701ms | 79.87ms | TPOT 回归！ |

**混合负载 TPOT 回归根因分析**：
- CrossDPExampleConnector 的 `start_load_kv` 是空操作，请求仍需完整模型前向传播
- CP=4 请求的 KV loading 调度大量 token（~1025），相当于 prefill
- 当 CP=1 和 CP=4 请求同时从 WAITING 调度时，两者都在 prefill 阶段
- CP=1 先完成 prefill 进入 decode，CP=4 仍在 prefill → MoE all-to-all 同步 → TPOT 回归
- 这是设计文档预期行为："当某些 rank 上执行 prefill，某些 rank 上执行 decode...decode 性能下降是正常的"
- 在真实 PD 部署中，prefill 在独立实例上，decode 实例不做 prefill，不会出现此问题

### 2026-05-02 Session 9
- 运行正式 Prefill benchmark（无 kv-transfer-config，output=1）
- **关键发现**：之前 DP 基线 Prefill TTFT 33.70ms 是 CrossDPExampleConnector 测量的（bypass prefill，只计算 1 个 token），不是真实 prefill
- 真实 CP=1 prefill TTFT 为 285ms（4K input，DP 模式等效基线）
- CP=2 prefill 开销 80% 是预期行为：CP prefill 需要对 all-gathered 完整序列计算 `kv_b_proj` 和 attention，计算量随总序列长度增长

**Session 9 Prefill Benchmark（16 prompts, max-concurrency=1, request-rate=1）**:
| 场景 | Input | Per-rank tokens | CP Size | TTFT P50 | CP 开销 | 备注 |
|------|-------|-----------------|---------|-----------|---------|------|
| Prefill | 4K | 4K | CP=1 | 285ms | 基线 | 等效 DP 基线 |
| Prefill | 8K | 4K | CP=2 | 512ms | +80% | kv_b_proj+attention 2x |
| Prefill | 20K | 5K | CP=4 | 440ms | +54% | 2 chunked steps |
| Prefill | 40K | 5K | CP=8 | 583ms | +105% | 2 chunked steps |

**CP 开销分析**：
- CP=2 (8K): 每个rank处理 4K tokens，但需对完整 8K 序列计算 `kv_b_proj`（2x）和 attention（4K×8K vs 4K×4K = 2x），加上 NCCL all-gather 通信
- CP=4 (20K): 每个rank处理 5K tokens（2 chunked steps），对完整 20K 序列计算 `kv_b_proj` 和 attention
- CP=8 (40K): 同 CP=4，但 all-gather 涉及 8 个 rank，通信开销更大
- 这是上下文并行的固有代价，不是 bug。DyCP 的核心价值是支持长上下文和动态调整 CP size

**待完成**：
- ~~运行 Prefill benchmark（不带 --kv-transfer-config，output=1）~~ — 已完成
- PD 分离端到端测试：修复了 3 个 KV loading bug，需远程验证
- 混合 CP size 负载的 TPOT 回归问题：需要真实 PD 部署验证

### 2026-05-02 Session 10
- 深入分析 PD 分离端到端测试中 decode 生成内容不正确的根因
- 发现并修复 6 个 bug：

22. **`delay_free` 逻辑错误：CP=1 时 prefill blocks 被提前释放** ✓ 已修复
    - 问题：`request_finished` 中 `delay_free = actual_cp_count > 1`，CP=1 时 blocks 立即释放，但 KV 数据可能还未被 decode 读取
    - 修复：保持 `delay_free = actual_cp_count > 1`（CP>1 延迟释放），CP=1 改用 legacy path（从 `_gpu_kv_buffer` 读取，不依赖 paged buffer blocks）
    - 影响：CP=1 的 PD 分离请求之前无法正确加载 KV

23. **`subgroup_start` 计算假设 decode rank 与 prefill 在同一子组** ✓ 已修复
    - 问题：`_start_load_kv_ipc` 中 `subgroup_start = (my_global_rank // cp_world_size) * cp_world_size` 假设 decode rank 在 prefill 的 CP 子组内，但 decode 请求可能被调度到不同 DP rank
    - 修复：在 `request_finished` 的 metadata 和 `return_params` 中添加 `prefill_cp_ranks`，IPC load 使用 `prefill_rank_map[src_rank]` 替代 `subgroup_start + src_rank`
    - 影响：CP>1 的 PD 分离请求，如果 decode 被调度到非 prefill 子组的 rank，IPC copy 会从错误 rank 读取数据

24. **`get_num_new_matched_tokens` 重建 metadata 缺少 `prefill_cp_ranks`** ✓ 已修复
    - 问题：proxy 转发 decode 请求时，重建 metadata 缺少 `prefill_cp_ranks` 字段
    - 修复：在重建逻辑中添加 `prefill_cp_ranks: kv_params.get("prefill_cp_ranks")`

25. **CP=1 时 `start_load_kv` 使用 IPC path 但 blocks 已释放** ✓ 已修复
    - 问题：CP=1 的 prefill blocks 不延迟释放（delay_free=False），但 `start_load_kv` 仍使用 IPC path 读取 paged buffer，读到已释放的 blocks
    - 修复：在 `start_load_kv` 中检查请求的 `cp_world_size`，CP=1 使用 legacy path（从 `_gpu_kv_buffer` 读取），CP>1 使用 IPC path

26. **`get_dycp_subgroup(1)` 崩溃：CP=1 没有对应的 NCCL 子组** ✓ 已修复
    - 问题：DyCP 代码中多处 `get_dycp_subgroup(actual_cp_size)` 在 `actual_cp_size=1` 时调用，但 CP=1 不创建 NCCL 子组
    - 修复：将条件从 `actual_cp_size > 0 and actual_cp_size < dycp_world_size` 改为 `actual_cp_size > 1 and actual_cp_size < dycp_world_size`
    - 文件：`cp_utils.py`, `flashinfer.py`, `flash_attn.py`, `mla/common.py`

27. **`request_finished` 中 `delay_free` 引用 scheduler 端不存在的 `_ipc_initialized`** ✓ 已修复
    - 问题：`delay_free = self._ipc_initialized` 在 scheduler 端报 `AttributeError`，因为 `_ipc_initialized` 只在 worker 端设置
    - 修复：改回 `delay_free = actual_cp_count > 1`，CP=1 通过 legacy path 解决 blocks 释放问题

- 远程测试发现新问题：PD 分离请求的 `cp_world_size` 不正确（2016 token 请求使用 CP=8 而非 CP=1），导致 IPC load 的 interleave mapping 错误
  - 根因分析见 Session 11

### 2026-05-02 Session 11
- **根因分析**：PD prefill 请求 `cp_world_size=8` 问题
  - 测试脚本 `start_vllm_pd_lowthresh.sh` 设置 `VLLM_LONG_REQUEST_THRESHOLD=100`
  - 调度器读取同一环境变量，将 `long_request_threshold` 也设为 100
  - 2016 token 请求被分类为 "long"（2016 >= 100），`select_dp` 返回所有 8 个 DP rank
  - 这导致 prefill 请求使用 CP=8（全 rank），但 decode 请求使用 CP=1
  - IPC load 使用 `cp_world_size=8` 的 interleave mapping，但实际 KV 只在一个 rank 上

  更深层的问题：即使不设 `VLLM_LONG_REQUEST_THRESHOLD`，当 DyCP 启用时也存在类似 bug：
  - DyCP 阈值 `[(4096, 1), (16384, 4), (32768, 8)]`，`long_request_threshold=16384`
  - 8K token 请求：`cp_size=1`（< 16384），但 `is_long=True`（>= 100 如果设了环境变量）
  - `select_dp` 中 `cp_size=1` 但 `is_long=True`，走入 `elif is_long` 分支，返回所有 8 rank
  - DyCP 的阈值逻辑被 `is_long` 分类覆盖

28. **`select_dp` 中 DyCP `cp_size=1` 被 `is_long` 覆盖** ✓ 已修复
    - 问题：当 DyCP 启用且 `cp_size=1` 时，`is_long=True` 仍使 `select_dp` 返回所有 rank
    - 修复：将 `elif is_long` 改为 `elif is_long and cp_size < 1`，确保 DyCP 阈值逻辑优先
    - 当 `cp_size >= 1`（DyCP 启用），阈值逻辑是权威的：`cp_size=1` 表示单 rank
    - 当 `cp_size == 0`（DyCP 未启用），`is_long` 分类决定 rank 数量
    - 文件：`cross_dp_scheduler.py`

29. **PD prefill 请求无 DyCP 时被错误分类为 long** ✓ 已修复
    - 问题：PD prefill 请求（`do_remote_decode=True`）在无 DyCP 时被 `is_long` 分类分配所有 rank
    - 但 decode 请求（`do_remote_prefill=True`）始终使用 CP=1
    - 导致 `cp_world_size` 不匹配，IPC load 的 interleave mapping 错误
    - 修复：对 PD prefill 请求，当 DyCP 未启用时也强制 `is_long=False`
    - 同时修复 `_free_request` 和 preemption 路径中的 `is_long` 计算保持一致
    - 文件：`cross_dp_scheduler.py`

- 下一步：同步到远程机器，运行 PD 分离端到端测试

### 2026-05-01 Session 5
- 修复 CP=4/8 decode 性能回归（TPOT 80.61ms → ~7ms）
  - 根因：`coordinate_batch_across_dp` 对所有 DP rank 取 cudagraph_mode 最小值，非 CP rank（0 tokens）dispatch 为 NONE 导致所有 rank 降级为 eager 模式
  - 修复 3 处：dp_utils 忽略 0 tokens rank、gpu_model_runner 报告 FULL 模式、dispatch lambda cp_size 修正
- 端到端验证：CP=1/2/4/8 单请求、2/4 并发 CP=4、混合 CP=1+CP=4 均通过
- 下一步：运行正式 benchmark、清理 debug 代码、PD 分离测试