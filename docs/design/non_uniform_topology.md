# Non-Uniform Topology Support for Multi-Node vLLM Deployments

This document describes the design and implementation of non-uniform topology support for vLLM's Multiproc Backend, enabling deployments where different nodes contribute different numbers of GPUs to the inference cluster.

## Introduction

### Motivation

vLLM's Multiproc Backend historically assumed that all nodes in a multi-node deployment have the same number of GPUs (uniform distribution). This assumption created several limitations:

1. **Inflexible scaling**: Cannot gracefully handle non-uniform GPU distributions (e.g., migrating from 16 GPUs to 12 GPUs across nodes)
2. **Undefined behavior**: System crashes or unpredictable behavior when nodes have different GPU counts
3. **Operational constraints**: Requires exact hardware matching across all nodes

### Problem Statement

Consider a deployment with:
- Node 0: 4 GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3)
- Node 1: 2 GPUs (CUDA_VISIBLE_DEVICES=0,1)

Before this enhancement, vLLM would either crash or produce incorrect rank assignments, causing:
- Global rank miscalculation
- Failed inter-node communication
- Worker connection failures

### Goals

1. Support non-uniform GPU distributions across nodes
2. Maintain backward compatibility with uniform distributions
3. Enable runtime topology discovery
4. Correctly handle all parallelism modes (TP, PP, DP, EP)

### Supported Distribution Constraints

**Important:** The non-uniform topology support is **constrained**—not all arbitrary GPU distributions are supported. The distribution must satisfy the following constraints validated by `RankTopology.validate_distribution()`:

**Constraint 1 (Single-node DP replicas):**
If a node has GPUs >= world_size (where world_size = TP * PP), the GPU count must be **divisible by world_size**. This ensures each DP replica on this node has exactly world_size GPUs.

**Constraint 2 (Cross-node DP replicas):**
If a node has GPUs < world_size, subsequent nodes must be included until the accumulated GPU count **exactly equals world_size**. This ensures cross-node DP replicas have exactly world_size GPUs total.

**Examples of valid distributions:**

| Distribution | world_size | Valid? | Reason |
|--------------|------------|--------|--------|
| [4, 2] | 3 (TP=3) | ✅ Yes | Node 0: 4 ≥ 3, 4 % 3 ≠ 0 → cross-node: 1+2=3 ✅ |
| [8, 4] | 4 (TP=4) | ✅ Yes | Node 0: 8 % 4 = 0 ✅; Node 1: 4 % 4 = 0 ✅ |
| [3, 3] | 3 (TP=3) | ✅ Yes | Node 0: 3 % 3 = 0 ✅; Node 1: 3 % 3 = 0 ✅ |
| [1, 7] | 4 (TP=4) | ❌ No | Node 0: 1 < 4 → accumulate: 1+7=8 ≠ 4 |
| [2, 2] | 3 (TP=3) | ❌ No | Cross-node: 2+2=4 ≠ 3 |
| [5, 1] | 3 (TP=3) | ❌ No | Node 0: 5 ≥ 3, but 5 % 3 ≠ 0 |

**Rationale:** Each DP replica requires exactly world_size GPUs for proper TP/PP communication. These constraints ensure GPU boundaries align with DP replica boundaries.

## Architecture

### Core Components

#### RankTopology Data Structure

A new data structure captures the distribution of ranks across physical nodes:

```
RankTopology
├── rank_to_local_rank: dict[int, int]
│   └── Maps global rank to local rank within node
├── node_device_counts: list[int]
│   └── List of GPU counts per node (index = node_rank)
└── Methods for rank/DP calculations
    ├── get_device_count_for_node_rank()
    ├── get_global_start_rank_for_node_rank()
    ├── get_dp_start_rank_for_node_rank()
    ├── get_node_rank_within_dp()
    ├── get_nnodes_within_dp()
    ├── get_local_world_size_for_dp_rank()
    ├── _get_dp_group_node_membership()
    └── validate_distribution()
```

**Key Methods for Non-Uniform Distribution:**

- `get_global_start_rank_for_node_rank(node_rank)`: Returns the starting global rank for a node, computed as the sum of GPU counts for all preceding nodes.

- `get_nnodes_within_dp(data_parallel_size, dp_rank)`: Returns the number of nodes a specific DP replica spans. Different DP replicas may span different numbers of nodes in non-uniform distributions.

- `get_node_rank_within_dp(node_rank, dp_rank, data_parallel_size)`: Returns a node's position within its DP replica (0 = leader node).

- `_get_dp_group_node_membership(dp_rank, world_size_within_dp)`: Internal method that determines which nodes contribute GPUs to a specific DP group.

#### NodeInfo Data Structure

Describes a single node's GPU configuration during discovery:

```
NodeInfo
└── device_count: int
    └── Number of GPUs visible to this node (via CUDA_VISIBLE_DEVICES)
```

### Topology Discovery Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Engine Args Creation                         │
│  (vllm/engine/arg_utils.py - create_engine_config)             │
│                                                                  │
│  Condition: data_parallel_backend == "mp" AND                   │
│             NOT is_api_server_child                              │
│                                                                  │
│  is_api_server_child = (_api_process_count > 1) AND             │
│                        (_api_process_rank >= 0)                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────────┐
│  Main Process       │       │  API Server Child       │
│                     │       │                         │
│  discover_rank_     │       │  Restore from           │
│  topology()         │       │  _rank_topology_data    │
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
│            RankTopology attached to ParallelConfig              │
│                                                                  │
│  parallel_config._rank_topology = topology                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│           MultiprocExecutor Initialization                       │
│  (vllm/v1/executor/multiproc_executor.py)                       │
│                                                                  │
│  - Validates topology is not None (assert)                      │
│  - Stores local node's global ranks                             │
│  - Computes rank assignments per node                           │
│  - Determines leader nodes per DP replica                       │
└─────────────────────────────────────────────────────────────────┘
```

### Unified Topology Initialization

**Key Design Decision:** Topology is initialized during `create_engine_config()` for **both single-node and multi-node** deployments with the MP backend. This unifies the initialization path and enables earlier error detection.

| Scenario | Initialization Timing | Location |
|----------|----------------------|----------|
| Single-node + MP backend | `create_engine_config()` | `arg_utils.py:1576` |
| Multi-node + MP backend | `create_engine_config()` | `arg_utils.py:1576` |
| Ray backend | Not initialized (N/A) | - |
| API server child process | Uses parent's topology | - |

The condition for topology discovery:
```python
if self.data_parallel_backend == "mp" and not is_api_server_child:
    topology = discover_rank_topology(...)
```

Where `is_api_server_child` is True when `_api_process_count > 1 and _api_process_rank >= 0`.

### Key Architectural Changes

#### Change 1: world_size_within_dp Definition

**Previous (incorrect):**

```
world_size_within_dp = world_size // data_parallel_size
```

**Corrected:**

```
world_size_within_dp = TP * PP
```

This is the number of GPUs per DP replica, independent of how many DP replicas exist.

#### Change 2: Leader Node Determination

In multi-node deployments, each DP replica that spans multiple nodes needs a leader. The leader is determined by `ParallelConfig.node_rank_within_dp` property:

- `node_rank_within_dp == 0`: Leader node (coordinates the DP replica)
- `node_rank_within_dp > 0`: Non-leader node (runs headless workers)

**Usage in serve.py:**
```python
if parallel_config.node_rank_within_dp > 0:
    # Run headless workers for non-leader nodes
    executor = MultiprocExecutor(vllm_config, monitor_workers=False)
    executor.start_worker_monitor(inline=True)
    return

# Leader node continues with main logic
```

The `node_rank_within_dp` property is computed by `RankTopology.get_node_rank_within_dp()` internally.

#### Change 3: Rank Assignment in MultiprocExecutor

**Previous:** Assumed uniform distribution

```
global_rank = local_world_size * node_rank_within_dp + local_rank
```

**New:** Topology-aware assignment

```python
# Get topology-aware rank assignments for this node
node_rank = self.parallel_config.node_rank
global_start_rank = (
    topology.get_global_start_rank_for_node_rank(node_rank)
    % self.world_size
)
for local_rank in range(self.local_world_size):
    global_rank = global_start_rank + local_rank
    # Create worker with correct rank assignment
```

## Component Modifications

### vllm/config/parallel.py

**New components:**
- `NodeInfo` dataclass for node GPU information
- `RankTopology` dataclass for rank distribution
- `discover_rank_topology()` function for runtime discovery

**Key methods in RankTopology:**
- `get_device_count_for_node_rank()`: Total GPUs on a node
- `get_global_ranks_for_node_rank()`: All global ranks for a node
- `get_local_ranks_for_node_rank()`: (global_rank, local_rank) pairs
- `get_dp_start_rank_for_node_rank()`: Starting DP rank for a node
- `get_node_rank_within_dp()`: Node's position within DP replica
- `get_nnodes_within_dp()`: Number of nodes in a DP replica
- `get_local_world_size_for_dp_rank()`: GPUs per DP rank per node
- `validate_distribution()`: Validate distribution constraints

**Modified components:**
- `ParallelConfig._rank_topology` field added (excluded from hash)
- `ParallelConfig.node_rank_within_dp` property uses topology
- `ParallelConfig.nnodes_within_dp` property uses topology
- `ParallelConfig.local_world_size` property uses topology

### vllm/engine/arg_utils.py

**Changes:**
- Topology discovery for both single-node and multi-node MP backend
- Distribution validation with `topology.validate_distribution()`
- Topology-aware DP rank inference for multi-node
- Clear error messages for missing topology in multi-node scenarios
- Added `_rank_topology_data` field for API server child process topology restoration

**Key logic:**
```python
# Discover topology for MP backend (unless API server child)
topology = None
is_api_server_child = (
    self._api_process_count > 1 and self._api_process_rank >= 0
)
if self.data_parallel_backend == "mp" and not is_api_server_child:
    topology = discover_rank_topology(...)
elif is_api_server_child:
    # API server child: restore topology from parent process
    assert self._rank_topology_data is not None
    topology = RankTopology.from_dict(self._rank_topology_data)

# Validate distribution
if topology is not None:
    topology.validate_distribution(world_size_within_dp)
elif self.nnodes > 1 and self.data_parallel_backend == "mp":
    raise RuntimeError("RankTopology not initialized for multi-node...")

# Multi-node: infer DP rank from topology
if self.nnodes > 1:
    assert topology is not None, "RankTopology is required..."
    inferred_data_parallel_rank = topology.get_dp_start_rank_for_node_rank(...)
```

### vllm/entrypoints/cli/serve.py

**Modified components:**
- `run_headless()` uses `parallel_config.node_rank_within_dp` property for leader determination
- Non-leader nodes (`node_rank_within_dp > 0`) run headless workers and return early
- Leader nodes (`node_rank_within_dp == 0`) continue with CoreEngineProcManager initialization

**Leader determination logic:**
```python
if parallel_config.node_rank_within_dp > 0:
    # Non-leader node: run headless worker
    executor = MultiprocExecutor(vllm_config, monitor_workers=False)
    executor.start_worker_monitor(inline=True)
    return

# Leader node: continue with engine creation
engine_manager = CoreEngineProcManager(...)
```

### vllm/v1/executor/multiproc_executor.py

**Key changes:**

Topology validation is now **inlined** in `_init_executor()`:
```python
# Topology should already be discovered during engine config creation
topology = self.parallel_config._rank_topology
assert topology is not None, (
    "RankTopology not initialized. "
    "This should have been done during engine config creation."
)
```

Rank assignment uses topology-aware calculation:
```python
# Get topology-aware rank assignments for this node
node_rank = self.parallel_config.node_rank
global_start_rank = (
    topology.get_global_start_rank_for_node_rank(node_rank)
    % self.world_size
)
for local_rank in range(self.local_world_size):
    global_rank = global_start_rank + local_rank
```

**Message queue initialization order:** `_init_message_queues()` is now called after `init_device()` to ensure parallel groups are initialized for cross-node DP groups.

**Removed constraint:** The assertion `world_size % nnodes_within_dp == 0` was removed since it assumes uniform distribution.

### vllm/v1/utils.py

**Changes:**
- `APIServerProcessManager.__init__()` accepts optional `rank_topology` parameter
- If topology is provided, it's serialized to dict and passed to child processes via `client_config`

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

**Changes:**
- Child processes restore topology from `client_config` passed by parent
- Sets `engine_args._rank_topology_data` which is then used during `create_engine_config()`

```python
if client_config:
    ...
    rank_topology_data = client_config.get("rank_topology")
    if rank_topology_data is not None:
        engine_args._rank_topology_data = rank_topology_data
```

## Parallelism Mode Handling

### TP/PP Multi-Node

For traditional tensor/pipeline parallelism across nodes:
- DP replicas span multiple nodes
- Leader node (node_rank_within_dp == 0) coordinates the replica
- Workers are assigned ranks based on their node's position

Example: 4 GPUs on node 0, 2 GPUs on node 1, with TP=3, DP=2
- DP 0: ranks 0,1,2 (all on node 0) → nnodes_within_dp = 1
- DP 1: ranks 3,4,5 (rank 3 on node 0, ranks 4,5 on node 1) → nnodes_within_dp = 2

### MoE Data Parallelism (Expert Parallelism)

For MoE models with data parallelism:
- Each GPU is its own DP rank (world_size_within_dp = TP * PP = 1)
- Each node runs independent EngineCoreProc for each local GPU
- All nodes are treated as "leaders" since each GPU is independent

### API Server Scale-Out

For API server scale-out (`--api-server-count > 1`):
- **Supports both single-node and multi-node deployments**
- Parent process discovers topology via `discover_rank_topology()`
- Topology data is passed to child processes through `APIServerProcessManager`:
  - Parent: `client_config["rank_topology"] = topology.to_dict()`
  - Child: `RankTopology.from_dict(client_config["rank_topology"])`
- Child processes restore topology from `_rank_topology_data` instead of re-discovering, avoiding port conflicts

This mechanism ensures consistent topology information across all API server processes.

## Design Decisions

### Per-DP-Rank Calculations

**Insight:** In non-uniform distributions, different DP replicas may span different numbers of nodes.

**Example:** 4 GPUs on node 0 + 2 GPUs on node 1, with TP=3, DP=2:
- DP rank 0: Uses GPUs 0,1,2 → All on node 0 → `nnodes_within_dp = 1`
- DP rank 1: Uses GPUs 3,4,5 → GPU 3 on node 0, GPUs 4,5 on node 1 → `nnodes_within_dp = 2`

**Design Decision:** All DP-related methods in `RankTopology` accept `dp_rank` parameter to return per-DP-rank values:
- `get_nnodes_within_dp(data_parallel_size, dp_rank)`
- `get_node_rank_within_dp(node_rank, dp_rank, data_parallel_size)`
- `get_local_world_size_for_dp_rank(dp_rank, pp_size, tp_size)`

Internal implementation uses `_get_dp_group_node_membership()` to determine which nodes contribute GPUs to a specific DP group.

### Message Queue Initialization Order

**Design Decision:** Message queue initialization in `WorkerProc` occurs after `init_device()` rather than before.

**Rationale:** For cross-node DP groups (`nnodes_within_dp > 1`), `_init_message_queues()` internally calls `get_inner_dp_world_group()`, which requires parallel groups to be initialized. The `init_device()` call initializes these parallel groups, so MQ initialization must follow.

**Code location:** `vllm/v1/executor/multiproc_executor.py:606-608`

### Topology Storage Location

**Decision:** Store topology in `ParallelConfig._rank_topology` as a private field.

**Rationale:**
- Topology is runtime state, not user configuration
- Needed by multiple components (Executor, serve.py, ParallelConfig properties)
- Set once during engine config creation
- Excluded from config hash computation

### Unified Initialization Timing

**Decision:** Discover topology during `EngineArgs.create_engine_config()` for both single-node and multi-node.

**Rationale:**
- Unifies single-node and multi-node initialization paths
- Earlier error detection (at config time, not executor time)
- Simplifies executor code (no fallback topology creation needed)
- ParallelConfig properties need topology immediately

### Error Handling

**Decision:** Use `assert` for topology validation in executor, explicit errors in arg_utils.

**Rationale:**
- In arg_utils: Clear error messages for configuration issues
- In executor: Assert for invariant checking (topology must exist)

### Backward Compatibility

**Decision:** Uniform distribution works as a special case of non-uniform.

**Rationale:**
- No code path forking needed
- Topology discovery works seamlessly for both cases
- Existing deployments unchanged

## Testing

### Unit Tests

Located in `tests/v1/executor/test_rank_topology.py`:
- RankTopology construction and properties
- Rank lookups by node_rank
- DP group calculations
- Distribution validation
- Edge cases (single node, empty topology)

### End-to-End Tests

E2E scenarios validated (see `claude/test_topology_e2e.py`):
- E2E-01: Single GPU
- E2E-02: Single node, 4 GPUs (DP=4)
- E2E-02A: Single node, 4 GPUs, API server scale-out (api_server_count=2)
- E2E-03 to E2E-11: Various multi-node configurations
- E2E-12 to E2E-19: TP/PP scenarios with non-uniform distribution
- E2E-ERR-*: Invalid distribution scenarios (validation error expected)

Each test validates:
1. Correct process spawning
2. Accurate rank assignment
3. Successful inference requests

## Files Modified

| File | Changes |
|------|---------|
| `vllm/config/parallel.py` | Added RankTopology, NodeInfo, discover_rank_topology(); updated properties with topology support |
| `vllm/engine/arg_utils.py` | Unified topology discovery, DP rank inference, added `_rank_topology_data` field |
| `vllm/entrypoints/cli/serve.py` | Use `node_rank_within_dp` property for leader determination |
| `vllm/entrypoints/openai/api_server.py` | Restore topology from parent process in child |
| `vllm/v1/executor/multiproc_executor.py` | Inlined topology validation, topology-aware rank assignment, message queue init order |
| `vllm/v1/utils.py` | Added `rank_topology` parameter to APIServerProcessManager |
| `tests/v1/executor/test_rank_topology.py` | Unit tests for RankTopology (construction, DP calculations, validation, serialization) |
| `tests/distributed/test_multiproc_executor.py` | Adapted for topology requirement |

## Future Work

1. **Elastic scaling**: Support dynamic addition/removal of nodes
2. **Topology-aware scheduling**: Leverage topology for optimal request distribution
3. **Heterogeneous hardware**: Support nodes with different GPU models