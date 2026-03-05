# vLLM 多节点部署的非均匀拓扑支持

本文档描述了 vLLM Multiproc 后端非均匀拓扑支持的设计与实现，使得不同节点可以向推理集群贡献不同数量的 GPU。

## 引言

### 动机

vLLM 的 Multiproc 后端历史上假设多节点部署中的所有节点都具有相同数量的 GPU（均匀分布）。这一假设带来了几个限制：

1. **扩展不灵活**：无法优雅地处理非均匀 GPU 分布（例如，将 16 个 GPU 迁移到跨节点的 12 个 GPU）
2. **未定义行为**：当节点具有不同的 GPU 数量时，系统崩溃或出现不可预测的行为
3. **运维约束**：要求所有节点的硬件完全匹配

### 问题陈述

考虑如下部署场景：
- 节点 0：4 个 GPU（CUDA_VISIBLE_DEVICES=0,1,2,3）
- 节点 1：2 个 GPU（CUDA_VISIBLE_DEVICES=0,1）

在此增强之前，vLLM 会崩溃或产生错误的 rank 分配，导致：
- 全局 rank 计算错误
- 节点间通信失败
- Worker 连接失败

### 目标

1. 支持跨节点的非均匀 GPU 分布
2. 保持与均匀分布的向后兼容性
3. 启用运行时拓扑发现
4. 正确处理所有并行模式（TP、PP、DP、EP）

### 支持的分布约束

**重要说明：** 非均匀拓扑支持是**受限的**——并非所有任意 GPU 分布都受支持。分布必须满足以下约束，由 `RankTopology.validate_distribution()` 验证：

**约束 1（单节点 DP 副本）：**
如果节点拥有的 GPU 数量 >= world_size（其中 world_size = TP * PP），则 GPU 数量必须能被 world_size **整除**。这确保该节点上的每个 DP 副本恰好有 world_size 个 GPU。

**约束 2（跨节点 DP 副本）：**
如果节点拥有的 GPU 数量 < world_size，则必须纳入后续节点，直到累积的 GPU 数量**恰好等于 world_size**。这确保跨节点 DP 副本总共有恰好 world_size 个 GPU。

**有效分布示例：**

| 分布 | world_size | 有效? | 原因 |
|------|------------|-------|------|
| [4, 2] | 3 (TP=3) | ✅ 是 | 节点 0: 4 ≥ 3, 4 % 3 ≠ 0 → 跨节点: 1+2=3 ✅ |
| [8, 4] | 4 (TP=4) | ✅ 是 | 节点 0: 8 % 4 = 0 ✅; 节点 1: 4 % 4 = 0 ✅ |
| [3, 3] | 3 (TP=3) | ✅ 是 | 节点 0: 3 % 3 = 0 ✅; 节点 1: 3 % 3 = 0 ✅ |
| [1, 7] | 4 (TP=4) | ❌ 否 | 节点 0: 1 < 4 → 累积: 1+7=8 ≠ 4 |
| [2, 2] | 3 (TP=3) | ❌ 否 | 跨节点: 2+2=4 ≠ 3 |
| [5, 1] | 3 (TP=3) | ❌ 否 | 节点 0: 5 ≥ 3, 但 5 % 3 ≠ 0 |

**原理：** 每个 DP 副本需要恰好 world_size 个 GPU 以进行正确的 TP/PP 通信。这些约束确保 GPU 边界与 DP 副本边界对齐。

## 架构

### 核心组件

#### RankTopology 数据结构

新的数据结构用于捕获跨物理节点的 rank 分布：

```
RankTopology
├── rank_to_local_rank: dict[int, int]
│   └── 将全局 rank 映射到节点内的本地 rank
├── node_device_counts: list[int]
│   └── 每个节点的 GPU 数量列表（索引 = node_rank）
└── 用于 rank/DP 计算的方法
    ├── get_device_count_for_node_rank()
    ├── get_global_start_rank_for_node_rank()
    ├── get_dp_start_rank_for_node_rank()
    ├── get_node_rank_within_dp()
    ├── get_nnodes_within_dp()
    ├── get_local_world_size_for_dp_rank()
    ├── _get_dp_group_node_membership()
    └── validate_distribution()
```

**非均匀分布场景下的关键方法：**

- `get_global_start_rank_for_node_rank(node_rank)`：返回节点的起始全局 rank，计算方式为所有前置节点的 GPU 数量之和。

- `get_nnodes_within_dp(data_parallel_size, dp_rank)`：返回特定 DP 副本跨越的节点数量。在非均匀分布中，不同 DP 副本可能跨越不同数量的节点。

- `get_node_rank_within_dp(node_rank, dp_rank, data_parallel_size)`：返回节点在其所属 DP 副本内的位置（0 表示领导节点）。

- `_get_dp_group_node_membership(dp_rank, world_size_within_dp)`：内部方法，确定哪些节点为特定 DP 组贡献 GPU。

#### NodeInfo 数据结构

在发现过程中描述单个节点的 GPU 配置：

```
NodeInfo
└── device_count: int
    └── 该节点可见的 GPU 数量（通过 CUDA_VISIBLE_DEVICES）
```

### 拓扑发现流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      引擎参数创建                                │
│  (vllm/engine/arg_utils.py - create_engine_config)             │
│                                                                 │
│  条件: data_parallel_backend == "mp" AND                        │
│        NOT is_api_server_child                                  │
│                                                                 │
│  is_api_server_child = (_api_process_count > 1) AND             │
│                        (_api_process_rank >= 0)                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────────┐
│     主进程          │       │   API Server 子进程     │
│                     │       │                         │
│  discover_rank_     │       │  从 _rank_topology_     │
│  topology()         │       │  data 恢复              │
│         │           │       │                         │
│         ▼           │       │  RankTopology.from_     │
│  validate_          │       │  dict(data)             │
│  distribution()     │       │                         │
└─────────┬───────────┘       └───────────┬─────────────┘
          │                               │
          └───────────────┬───────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│            RankTopology 附加到 ParallelConfig                   │
│                                                                 │
│  parallel_config._rank_topology = topology                     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│           MultiprocExecutor 初始化                               │
│  (vllm/v1/executor/multiproc_executor.py)                       │
│                                                                 │
│  - 验证拓扑不为 None（assert）                                  │
│  - 存储本地节点的全局 rank                                      │
│  - 计算每个节点的 rank 分配                                     │
│  - 确定每个 DP 副本的领导节点                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 统一拓扑初始化

**关键设计决策：** 拓扑在 `create_engine_config()` 中初始化，适用于使用 MP 后端的**单节点和多节点**部署。这统一了初始化路径，并实现了更早的错误检测。

| 场景 | 初始化时机 | 位置 |
|------|-----------|------|
| 单节点 + MP 后端 | `create_engine_config()` | `arg_utils.py:1576` |
| 多节点 + MP 后端 | `create_engine_config()` | `arg_utils.py:1576` |
| Ray 后端 | 不初始化（不适用） | - |
| API server 子进程 | 使用父进程的拓扑 | - |

拓扑发现的条件：
```python
if self.data_parallel_backend == "mp" and not is_api_server_child:
    topology = discover_rank_topology(...)
```

其中 `is_api_server_child` 在 `_api_process_count > 1 and _api_process_rank >= 0` 时为 True。

### 关键架构变更

#### 变更 1：world_size_within_dp 定义

**之前（错误）：**

```
world_size_within_dp = world_size // data_parallel_size
```

**修正后：**

```
world_size_within_dp = TP * PP
```

这是每个 DP 副本的 GPU 数量，与存在多少个 DP 副本无关。

#### 变更 2：领导节点确定

在多节点部署中，每个跨越多个节点的 DP 副本需要一个领导节点。领导节点由 `ParallelConfig.node_rank_within_dp` 属性确定：

- `node_rank_within_dp == 0`：领导节点（协调 DP 副本）
- `node_rank_within_dp > 0`：非领导节点（运行 headless worker）

**serve.py 中的使用方式：**
```python
if parallel_config.node_rank_within_dp > 0:
    # 非领导节点：运行 headless worker
    executor = MultiprocExecutor(vllm_config, monitor_workers=False)
    executor.start_worker_monitor(inline=True)
    return

# 领导节点：继续执行主逻辑
```

`node_rank_within_dp` 属性内部通过 `RankTopology.get_node_rank_within_dp()` 计算得出。

#### 变更 3：MultiprocExecutor 中的 Rank 分配

**之前：** 假设均匀分布

```
global_rank = local_world_size * node_rank_within_dp + local_rank
```

**新方案：** 拓扑感知分配

```python
# 获取此节点的拓扑感知 rank 分配
node_rank = self.parallel_config.node_rank
global_start_rank = (
    topology.get_global_start_rank_for_node_rank(node_rank)
    % self.world_size
)
for local_rank in range(self.local_world_size):
    global_rank = global_start_rank + local_rank
    # 使用正确的 rank 分配创建 worker
```

## 组件修改

### vllm/config/parallel.py

**新增组件：**
- `NodeInfo` dataclass 用于节点 GPU 信息
- `RankTopology` dataclass 用于 rank 分布
- `discover_rank_topology()` 函数用于运行时发现

**RankTopology 关键方法：**
- `get_device_count_for_node_rank()`：节点上的 GPU 总数
- `get_global_ranks_for_node_rank()`：节点的所有全局 rank
- `get_local_ranks_for_node_rank()`：(global_rank, local_rank) 对
- `get_dp_start_rank_for_node_rank()`：节点的起始 DP rank
- `get_node_rank_within_dp()`：节点在 DP 副本内的位置
- `get_nnodes_within_dp()`：DP 副本内的节点数量
- `get_local_world_size_for_dp_rank()`：每个 DP rank 在每个节点上的 GPU 数量
- `validate_distribution()`：验证分布约束

**修改的组件：**
- 添加了 `ParallelConfig._rank_topology` 字段（从 hash 计算中排除）
- `ParallelConfig.node_rank_within_dp` 属性使用拓扑
- `ParallelConfig.nnodes_within_dp` 属性使用拓扑
- `ParallelConfig.local_world_size` 属性使用拓扑

### vllm/engine/arg_utils.py

**变更：**
- 为单节点和多节点 MP 后端进行拓扑发现
- 使用 `topology.validate_distribution()` 进行分布验证
- 拓扑感知的多节点 DP rank 推断
- 多节点场景缺少拓扑时的明确错误信息
- 添加了 `_rank_topology_data` 字段用于 API server 子进程拓扑恢复

**核心逻辑：**
```python
# 为 MP 后端发现拓扑（除非是 API server 子进程）
topology = None
is_api_server_child = (
    self._api_process_count > 1 and self._api_process_rank >= 0
)
if self.data_parallel_backend == "mp" and not is_api_server_child:
    topology = discover_rank_topology(...)
elif is_api_server_child:
    # API server 子进程：从父进程恢复拓扑
    assert self._rank_topology_data is not None
    topology = RankTopology.from_dict(self._rank_topology_data)

# 验证分布
if topology is not None:
    topology.validate_distribution(world_size_within_dp)
elif self.nnodes > 1 and self.data_parallel_backend == "mp":
    raise RuntimeError("RankTopology 未为多节点初始化...")

# 多节点：从拓扑推断 DP rank
if self.nnodes > 1:
    assert topology is not None, "RankTopology 必需..."
    inferred_data_parallel_rank = topology.get_dp_start_rank_for_node_rank(...)
```

### vllm/entrypoints/cli/serve.py

**修改的组件：**
- `run_headless()` 使用 `parallel_config.node_rank_within_dp` 属性判断领导节点
- 非领导节点（`node_rank_within_dp > 0`）运行 headless worker 并提前返回
- 领导节点（`node_rank_within_dp == 0`）继续执行 CoreEngineProcManager 初始化

**领导节点判断逻辑：**
```python
if parallel_config.node_rank_within_dp > 0:
    # 非领导节点：运行 headless worker
    executor = MultiprocExecutor(vllm_config, monitor_workers=False)
    executor.start_worker_monitor(inline=True)
    return

# 领导节点：继续创建 engine
engine_manager = CoreEngineProcManager(...)
```

### vllm/v1/executor/multiproc_executor.py

**关键变更：**

拓扑验证现在**内联**在 `_init_executor()` 中：
```python
# 拓扑应该在引擎配置创建期间已经发现
topology = self.parallel_config._rank_topology
assert topology is not None, (
    "RankTopology 未初始化。"
    "这应该在引擎配置创建期间完成。"
)
```

Rank 分配使用拓扑感知计算：
```python
# 获取此节点的拓扑感知 rank 分配
node_rank = self.parallel_config.node_rank
global_start_rank = (
    topology.get_global_start_rank_for_node_rank(node_rank)
    % self.world_size
)
for local_rank in range(self.local_world_size):
    global_rank = global_start_rank + local_rank
```

**消息队列初始化顺序：** `_init_message_queues()` 现在在 `init_device()` 之后调用，以确保跨节点 DP 组的并行组已初始化。

**移除的约束：** 断言 `world_size % nnodes_within_dp == 0` 已移除，因为它假设均匀分布。

### vllm/v1/utils.py

**变更：**
- `APIServerProcessManager.__init__()` 接受可选的 `rank_topology` 参数
- 如果提供了拓扑，它被序列化为 dict 并通过 `client_config` 传递给子进程

```python
def __init__(
    self,
    ...
    rank_topology: "RankTopology | None" = None,
):
    ...
    if rank_topology is not None:
        client_config["rank_topology"] = rank_topology.to_dict()
```

### vllm/entrypoints/openai/api_server.py

**变更：**
- 子进程从父进程传递的 `client_config` 恢复拓扑
- 设置 `engine_args._rank_topology_data`，然后在 `create_engine_config()` 期间使用

```python
if client_config:
    ...
    rank_topology_data = client_config.get("rank_topology")
    if rank_topology_data is not None:
        engine_args._rank_topology_data = rank_topology_data
```

## 并行模式处理

### TP/PP 多节点

对于跨节点的传统张量/流水线并行：
- DP 副本跨越多个节点
- 领导节点（node_rank_within_dp == 0）协调副本
- Worker 根据其节点位置分配 rank

示例：节点 0 上 4 个 GPU，节点 1 上 2 个 GPU，TP=3，DP=2
- DP 0：ranks 0,1,2（都在节点 0 上）→ nnodes_within_dp = 1
- DP 1：ranks 3,4,5（rank 3 在节点 0 上，ranks 4,5 在节点 1 上）→ nnodes_within_dp = 2

### MoE 数据并行（专家并行）

对于具有数据并行的 MoE 模型：
- 每个 GPU 是自己的 DP rank（world_size_within_dp = TP * PP = 1）
- 每个节点为每个本地 GPU 运行独立的 EngineCoreProc
- 所有节点都被视为"领导节点"，因为每个 GPU 都是独立的

### API Server 横向扩展

对于 API server 横向扩展（`--api-server-count > 1`）：
- **支持单节点和多节点部署**
- 父进程通过 `discover_rank_topology()` 发现拓扑
- 拓扑数据通过 `APIServerProcessManager` 传递给子进程：
  - 父进程：`client_config["rank_topology"] = topology.to_dict()`
  - 子进程：`RankTopology.from_dict(client_config["rank_topology"])`
- 子进程从 `_rank_topology_data` 恢复拓扑而非重新发现，避免端口冲突

此机制确保所有 API server 进程具有一致的拓扑信息。

## 设计决策

### Per-DP-Rank 计算

**洞察：** 在非均匀分布中，不同的 DP 副本可能跨越不同数量的节点。

**示例：** 节点 0 上 4 个 GPU + 节点 1 上 2 个 GPU，TP=3，DP=2：
- DP rank 0：使用 GPU 0,1,2 → 都在节点 0 上 → `nnodes_within_dp = 1`
- DP rank 1：使用 GPU 3,4,5 → GPU 3 在节点 0 上，GPU 4,5 在节点 1 上 → `nnodes_within_dp = 2`

**设计决策：** `RankTopology` 中所有与 DP 相关的方法都接受 `dp_rank` 参数以返回 per-DP-rank 的值：
- `get_nnodes_within_dp(data_parallel_size, dp_rank)`
- `get_node_rank_within_dp(node_rank, dp_rank, data_parallel_size)`
- `get_local_world_size_for_dp_rank(dp_rank, pp_size, tp_size)`

内部实现使用 `_get_dp_group_node_membership()` 确定哪些节点为特定 DP 组贡献 GPU。

### 消息队列初始化顺序

**设计决策：** `WorkerProc` 中的消息队列初始化发生在 `init_device()` 之后而非之前。

**理由：** 对于跨节点 DP 组（`nnodes_within_dp > 1`），`_init_message_queues()` 内部调用 `get_inner_dp_world_group()`，这要求并行组已初始化。`init_device()` 调用会初始化这些并行组，因此 MQ 初始化必须在其之后。

**代码位置：** `vllm/v1/executor/multiproc_executor.py:606-608`

### 拓扑存储位置

**决策：** 将拓扑存储在 `ParallelConfig._rank_topology` 作为私有字段。

**理由：**
- 拓扑是运行时状态，而非用户配置
- 多个组件需要访问（Executor、serve.py、ParallelConfig 属性）
- 在引擎配置创建时设置一次
- 从配置 hash 计算中排除

### 统一初始化时机

**决策：** 在 `EngineArgs.create_engine_config()` 期间发现拓扑，适用于单节点和多节点。

**理由：**
- 统一单节点和多节点的初始化路径
- 更早的错误检测（配置时而非执行器时）
- 简化执行器代码（无需 fallback 拓扑创建）
- ParallelConfig 属性需要立即访问拓扑

### 错误处理

**决策：** 在执行器中使用 `assert` 进行拓扑验证，在 arg_utils 中使用明确错误。

**理由：**
- 在 arg_utils 中：为配置问题提供清晰的错误信息
- 在执行器中：使用 assert 进行不变量检查（拓扑必须存在）

### 向后兼容性

**决策：** 均匀分布作为非均匀分布的特例工作。

**理由：**
- 不需要代码路径分支
- 拓扑发现对两种情况都能无缝工作
- 现有部署保持不变

## 测试

### 单元测试

位于 `tests/v1/executor/test_rank_topology.py`：
- RankTopology 构造和属性
- 按 node_rank 查找 rank
- DP 组计算
- 分布验证
- 边缘情况（单节点、空拓扑）

### 端到端测试

验证的 E2E 场景（参见 `claude/test_topology_e2e.py`）：
- E2E-01：单 GPU
- E2E-02：单节点，4 GPU（DP=4）
- E2E-02A：单节点，4 GPU，API server 横向扩展（api_server_count=2）
- E2E-03 至 E2E-11：各种多节点配置
- E2E-12 至 E2E-19：非均匀分布的 TP/PP 场景
- E2E-ERR-*：无效分布场景（预期验证错误）

每个测试验证：
1. 正确的进程生成
2. 准确的 rank 分配
3. 成功的推理请求

## 修改的文件

| 文件 | 变更 |
|------|------|
| `vllm/config/parallel.py` | 添加了 RankTopology、NodeInfo、discover_rank_topology()；更新了属性以支持拓扑 |
| `vllm/engine/arg_utils.py` | 统一拓扑发现、DP rank 推断、添加 `_rank_topology_data` 字段 |
| `vllm/entrypoints/cli/serve.py` | 使用 `node_rank_within_dp` 属性进行领导节点判断 |
| `vllm/entrypoints/openai/api_server.py` | 子进程从父进程恢复拓扑 |
| `vllm/v1/executor/multiproc_executor.py` | 内联拓扑验证、拓扑感知 rank 分配、消息队列初始化顺序 |
| `vllm/v1/utils.py` | 向 APIServerProcessManager 添加 `rank_topology` 参数 |
| `tests/v1/executor/test_rank_topology.py` | RankTopology 单元测试（构造、DP 计算、验证、序列化） |
| `tests/distributed/test_multiproc_executor.py` | 适配拓扑要求 |

## 未来工作

1. **弹性扩展**：支持节点的动态添加/移除
2. **拓扑感知调度**：利用拓扑进行最优请求分发
3. **异构硬件**：支持具有不同 GPU 型号的节点