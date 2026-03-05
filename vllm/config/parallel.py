# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import Field, field_validator, model_validator
from torch.distributed import ProcessGroup, ReduceOp
from typing_extensions import Self

import vllm.envs as envs
from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_ports_list
from vllm.utils.torch_utils import cuda_device_count_stateless

if TYPE_CHECKING:
    from ray.runtime_env import RuntimeEnv
    from ray.util.placement_group import PlacementGroup

    from vllm.v1.executor import Executor
else:
    RuntimeEnv = Any
    PlacementGroup = Any
    Executor = Any

logger = init_logger(__name__)

ExpertPlacementStrategy = Literal["linear", "round_robin"]
DistributedExecutorBackend = Literal["ray", "mp", "uni", "external_launcher"]
DataParallelBackend = Literal["ray", "mp"]
EPLBPolicyOption = Literal["default"]
All2AllBackend = Literal[
    "naive",
    "pplx",
    "deepep_high_throughput",
    "deepep_low_latency",
    "mori",
    "allgather_reducescatter",
    "flashinfer_all2allv",
]


@dataclass
class NodeInfo:
    """Node information for topology discovery.

    Describes the GPU configuration of a single physical node.
    Used during topology discovery to build the complete cluster map.
    """

    device_count: int
    """Number of GPUs visible to this node (via CUDA_VISIBLE_DEVICES).

    This represents the total number of physical GPUs on this node.
    """


@dataclass
class RankTopology:
    """Describes the distribution of ranks across physical nodes.

    This data structure eliminates the assumption that each node has the same
    number of GPUs, enabling non-uniform distributions like:
    - Node 1: 4 GPUs (ranks 0-3)
    - Node 2: 2 GPUs (ranks 4-5)

    The topology is discovered at runtime using StatelessProcessGroup to
    coordinate between nodes.

    Key concepts:
    - device_count: Total GPUs on a node (physical hardware count)
    - local_world_size: GPUs assigned to a specific DP rank on a node
    """

    rank_to_local_rank: dict[int, int]
    """Mapping from global rank to local rank within the node."""

    node_device_counts: list[int] = field(default_factory=list)
    """List of device counts for each node, indexed by node_rank.

    This represents the total number of GPUs on each physical node.
    For example, [4, 2] means node 0 has 4 GPUs and node 1 has 2 GPUs.
    """

    @property
    def world_size(self) -> int:
        """Total number of GPUs across all nodes."""
        return sum(self.node_device_counts)

    def get_device_count_for_node_rank(self, node_rank: int) -> int:
        """Get the total number of GPUs on a node.

        This returns the physical GPU count on the node, regardless of
        how they are distributed among DP replicas.

        Args:
            node_rank: The rank of the node (0-indexed).

        Returns:
            Total number of GPUs on the node.

        Raises:
            ValueError: If node_rank is invalid.
        """
        if node_rank < 0 or node_rank >= len(self.node_device_counts):
            raise ValueError(
                f"node_rank {node_rank} is out of range "
                f"[0, {len(self.node_device_counts)})"
            )
        return self.node_device_counts[node_rank]

    def get_global_start_rank_for_node_rank(self, node_rank: int) -> int:
        """Get the starting global rank for a node identified by node_rank.

        Args:
            node_rank: The rank of the node (0-indexed).

        Returns:
            The starting global rank for the node.

        Raises:
            ValueError: If node_rank is invalid.
        """
        # Calculate the starting rank for this node_rank
        start_rank = sum(self.node_device_counts[:node_rank])
        return start_rank

    def get_dp_start_rank_for_node_rank(
        self, node_rank: int, pp_size: int = 1, tp_size: int = 1
    ) -> int:
        """Get the starting DP rank for a node.

        This calculates the starting DP rank based on the actual rank distribution
        in the topology, rather than assuming uniform distribution across nodes.

        Args:
            node_rank: The rank of the node (0-indexed).
            pp_size: Pipeline parallel size. Defaults to 1.
            tp_size: Tensor parallel size. Defaults to 1.

        Returns:
            The starting DP rank for the specified node.

        Raises:
            ValueError: If node_rank is invalid.

        Example:
            For a 2-node setup with 4 GPUs on node 0 and 2 GPUs on node 1:
            >>> topology.get_dp_start_rank_for_node_rank(0, pp_size=1, tp_size=1)
            0  # Node 0 contains global ranks 0-3, so DP ranks start at 0
            >>> topology.get_dp_start_rank_for_node_rank(1, pp_size=1, tp_size=1)
            4  # Node 1 contains global ranks 4-5, so DP ranks start at 4
        """
        # Validate node_rank
        if node_rank < 0 or node_rank >= len(self.node_device_counts):
            raise ValueError(
                f"node_rank {node_rank} is out of range "
                f"[0, {len(self.node_device_counts)})"
            )

        # Calculate the starting global rank for this node
        start_global_rank = sum(self.node_device_counts[:node_rank])

        # Each DP replica requires pp_size * tp_size ranks
        world_size_within_dp = pp_size * tp_size

        # The starting DP rank is the number of complete DP groups before this node
        return start_global_rank // world_size_within_dp

    def _get_dp_group_node_membership(
        self, dp_rank: int, world_size_within_dp: int
    ) -> list[int]:
        """Get the list of node_ranks that contribute GPUs to a DP group.

        A DP group consists of world_size_within_dp GPUs. This method determines
        which nodes contribute GPUs to a specific DP group identified by dp_rank.

        Args:
            dp_rank: The DP rank to query.
            world_size_within_dp: Number of GPUs per DP group (TP * PP).

        Returns:
            List of node_ranks that contribute GPUs to this DP group.
        """
        dp_start_gpu = dp_rank * world_size_within_dp
        dp_end_gpu = dp_start_gpu + world_size_within_dp

        member_nodes: list[int] = []
        accumulated_gpus = 0

        for node_rank, device_count in enumerate(self.node_device_counts):
            node_start = accumulated_gpus
            node_end = accumulated_gpus + device_count

            # Check if this node overlaps with the DP group's GPU range
            if node_end > dp_start_gpu and node_start < dp_end_gpu:
                member_nodes.append(node_rank)

            accumulated_gpus += device_count
            if accumulated_gpus >= dp_end_gpu:
                break

        return member_nodes

    def get_nnodes_within_dp(
        self,
        data_parallel_size: int,
        dp_rank: int,
    ) -> int:
        """Get the number of nodes within a single DP replica.

        For non-uniform distribution, different DP replicas may span different
        numbers of nodes. This method calculates how many nodes a specific
        DP replica spans.

        Args:
            data_parallel_size: Total number of DP replicas.
            dp_rank: The DP rank to query.

        Returns:
            Number of nodes that the specified DP replica spans.

        Raises:
            ValueError: If data_parallel_size is invalid or dp_rank is out of range.

        Example:
            For 4+2 GPUs with TP=3, DP=2:
            - DP 0: ranks 0,1,2 (all in node 0) → nnodes=1
            - DP 1: ranks 3,4,5 (rank 3 in node 0, ranks 4,5 in node 1) → nnodes=2
        """
        if data_parallel_size <= 0:
            raise ValueError(
                f"data_parallel_size must be positive, got {data_parallel_size}"
            )

        if dp_rank < 0 or dp_rank >= data_parallel_size:
            raise ValueError(
                f"dp_rank {dp_rank} is out of range [0, {data_parallel_size})"
            )

        if len(self.node_device_counts) <= 1:
            return 1

        total_gpus = self.world_size
        world_size_within_dp = total_gpus // data_parallel_size

        if world_size_within_dp <= 0:
            raise ValueError(
                f"Invalid configuration: world_size ({total_gpus}) is smaller than "
                f"data_parallel_size ({data_parallel_size})"
            )

        members = self._get_dp_group_node_membership(dp_rank, world_size_within_dp)
        return max(1, len(members))

    def get_node_rank_within_dp(
        self,
        node_rank: int,
        dp_rank: int,
        data_parallel_size: int,
    ) -> int:
        """Get the node's position within its DP replica.

        This determines a node's position (0-indexed) within a DP group.
        Position 0 means the node is the leader of that DP group.

        Args:
            node_rank: Global node rank (0-indexed).
            dp_rank: The DP rank to query.
            data_parallel_size: Total number of DP replicas.

        Returns:
            Node's position within its DP group (0 = leader).

        Raises:
            ValueError: If node_rank, dp_rank, or data_parallel_size is invalid.
            ValueError: If the node does not belong to the specified DP group.

        Example:
            For 4+2 GPUs with TP=3, DP=2:
            - Node 0: contains ranks 0-3
                      DP 0 members: [node 0] -> node 0 position 0
                      DP 1 members: [node 0, node 1] -> node 0 position 0
            - Node 1: contains ranks 4-5
                      DP 1 members: [node 0, node 1] -> node 1 position 1
        """
        # Calculate world_size_within_dp
        world_size_within_dp = self.world_size // data_parallel_size
        # Get the member nodes of this DP group
        members = self._get_dp_group_node_membership(dp_rank, world_size_within_dp)
        # Return the position of this node in the member list
        assert node_rank in members, (
            f"node_rank {node_rank} does not belong to DP group {dp_rank}. "
            f"DP group {dp_rank} members: {members}"
        )
        return members.index(node_rank)

    def get_local_world_size_for_dp_rank(
        self,
        node_rank: int,
        dp_rank: int,
        world_size_within_dp: int,
    ) -> int:
        """Get the number of GPUs this DP rank uses on this node.

        This calculates how many GPUs a specific DP replica uses on a specific node.
        The value is determined by the distribution constraints:

        - Constraint 1: If device_count >= world_size_within_dp, each local DP replica
          uses world_size_within_dp GPUs (the node hosts multiple complete DP replicas).
        - Constraint 2: If device_count < world_size_within_dp, the node contributes
          all its GPUs to a cross-node DP replica.

        Args:
            node_rank: Global node rank (0-indexed).
            dp_rank: The DP rank to query.
            world_size_within_dp: GPUs per DP replica (TP * PP).

        Returns:
            Number of GPUs this DP rank uses on this node.

        Example:
            For 4+2 GPUs with TP=1, DP=6 (MoE DP):
            - world_size_within_dp = 1, each DP replica uses 1 GPU
            - Node 0 (4 GPUs): DP 0-3 each use 1 GPU on this node
            - Node 1 (2 GPUs): DP 4-5 each use 1 GPU on this node
            - get_local_world_size_for_dp_rank(0, 0, 1) = 1
            - get_local_world_size_for_dp_rank(0, 3, 1) = 1
            - get_local_world_size_for_dp_rank(1, 4, 1) = 1

            For 4+2 GPUs with TP=3, DP=2:
            - world_size_within_dp = 3
            - DP 0: uses ranks 0,1,2 (all in node 0) -> node 0 contributes 3 GPUs
            - DP 1: uses ranks 3,4,5 (rank 3 in node 0, ranks 4,5 in node 1)
              -> node 0 contributes 1 GPU, node 1 contributes 2 GPUs
            - get_local_world_size_for_dp_rank(0, 0, 3) = 3
            - get_local_world_size_for_dp_rank(0, 1, 3) = 1
            - get_local_world_size_for_dp_rank(1, 1, 3) = 2
        """
        if node_rank < 0 or node_rank >= len(self.node_device_counts):
            return 0

        if world_size_within_dp <= 0:
            return 0

        # Calculate global GPU range for this DP replica
        dp_start_gpu = dp_rank * world_size_within_dp
        dp_end_gpu = dp_start_gpu + world_size_within_dp

        # Calculate this node's GPU range
        node_start_gpu = sum(self.node_device_counts[:node_rank])
        node_end_gpu = node_start_gpu + self.node_device_counts[node_rank]

        # Calculate overlap
        overlap_start = max(dp_start_gpu, node_start_gpu)
        overlap_end = min(dp_end_gpu, node_end_gpu)

        return max(0, overlap_end - overlap_start)

    def validate_distribution(self, world_size: int) -> None:
        """Validate GPU distribution for a given world_size (TP * PP).

        This validates that the GPU distribution across nodes satisfies the
        constraints for non-uniform topology:

        Constraint 1 (no cross-node DP replicas):
            If a node has GPUs >= world_size, the GPU count must be divisible
            by world_size. This ensures each DP replica on this node has
            exactly world_size GPUs (TP * PP).

        Constraint 2 (cross-node DP replicas):
            If a node has GPUs < world_size, subsequent nodes must be included
            until the accumulated GPU count exactly equals world_size. This
            ensures cross-node DP replicas have exactly world_size GPUs total.

        Args:
            world_size: The size of a single DP replica (TP * PP).

        Raises:
            ValueError: If the distribution violates the constraints.
        """
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")

        device_counts = self.node_device_counts
        node_idx = 0
        nnodes = len(device_counts)

        # Edge case: empty distribution
        if nnodes == 0:
            raise ValueError("Empty GPU distribution. At least one node is required.")

        while node_idx < nnodes:
            gpus = device_counts[node_idx]

            if gpus >= world_size:
                # Constraint 1: GPU count must divide world_size
                if gpus % world_size != 0:
                    raise ValueError(
                        f"Node {node_idx} has {gpus} GPUs (>= world_size "
                        f"{world_size}), but {gpus} is not divisible by {world_size}. "
                        "Each node with GPUs >= world_size must have GPUs "
                        "divisible by world_size to host complete DP replicas."
                    )
                node_idx += 1
            else:
                # Constraint 2: Accumulate GPUs until exactly world_size
                accumulated = 0
                start_idx = node_idx

                while node_idx < nnodes and accumulated < world_size:
                    accumulated += device_counts[node_idx]
                    node_idx += 1

                if accumulated != world_size:
                    remaining = device_counts[start_idx:]
                    raise ValueError(
                        f"Starting from node {start_idx}, accumulated "
                        f"{accumulated} GPUs but world_size is {world_size}. "
                        f"Cross-node DP replica requires exactly world_size "
                        f"GPUs. GPU distribution: {device_counts}, "
                        f"nodes: {remaining}"
                    )

    @classmethod
    def from_node_infos(cls, node_infos: list[NodeInfo]) -> "RankTopology":
        """Build topology from a list of node information.

        Args:
            node_infos: List of NodeInfo objects, one per node.
                       The order in the list determines global rank assignment
                       and node_rank (index 0 = node_rank 0, etc.).

        Returns:
            A RankTopology instance describing the complete distribution.

        Example:
            For 2 nodes with 4 and 2 GPUs respectively:
            >>> node_infos = [
            ...     NodeInfo(device_count=4),
            ...     NodeInfo(device_count=2),
            ... ]
            >>> topology = RankTopology.from_node_infos(node_infos)
            >>> topology.world_size
            6
            >>> topology.get_global_start_rank_for_node_rank(0)
            0
            >>> topology.get_global_start_rank_for_node_rank(1)
            4
        """
        rank_to_local_rank: dict[int, int] = {}
        node_device_counts: list[int] = []

        current_global_rank = 0
        for node_info in node_infos:
            node_device_counts.append(node_info.device_count)

            for local_rank in range(node_info.device_count):
                global_rank = current_global_rank + local_rank
                rank_to_local_rank[global_rank] = local_rank

            current_global_rank += node_info.device_count

        return cls(
            rank_to_local_rank=rank_to_local_rank,
            node_device_counts=node_device_counts,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize RankTopology to a dictionary for inter-process communication."""
        return {
            "rank_to_local_rank": self.rank_to_local_rank,
            "node_device_counts": self.node_device_counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RankTopology":
        """Deserialize RankTopology from a dictionary."""
        return cls(
            rank_to_local_rank=data["rank_to_local_rank"],
            node_device_counts=data["node_device_counts"],
        )


def discover_rank_topology(
    nnodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
    coord_port_offset: int = 1000,
) -> RankTopology:
    """Discover rank topology across nodes before ParallelConfig is constructed.

    This function gathers GPU count information from all nodes and builds a
    RankTopology. It can be called during engine config creation to enable
    accurate configuration inference for non-uniform GPU distributions.

    For single-node deployments, builds topology from local GPU count only.
    For multi-node deployments, uses StatelessProcessGroup to coordinate.

    Args:
        nnodes: Number of nodes in the cluster.
        node_rank: Rank of this node (0-indexed).
        master_addr: Address of the master node for coordination.
        master_port: Port of the master node.
        coord_port_offset: Port offset for topology coordination.
            The actual coordination port is master_port + coord_port_offset.

    Returns:
        RankTopology describing the distribution of ranks across nodes.

    Raises:
        AssertionError: If the number of node infos gathered doesn't match nnodes.
    """
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.utils.network_utils import get_ip

    local_ip = get_ip()
    local_gpu_count = current_platform.device_count()

    if nnodes <= 1:
        # Single node: build topology from local info only
        node_infos = [NodeInfo(device_count=local_gpu_count)]
        logger.debug(
            "Single node deployment, topology built with %d GPUs",
            local_gpu_count,
        )
        return RankTopology.from_node_infos(node_infos)

    # Multi-node: coordinate via StatelessProcessGroup
    # Use fixed port offset (not dependent on dp_rank, which is not yet known)
    coord_port = master_port + coord_port_offset
    logger.info(
        "Discovering topology with %d nodes, node_rank=%d, coord_port=%d",
        nnodes,
        node_rank,
        coord_port,
    )

    coord_group = StatelessProcessGroup.create(
        host=master_addr,
        port=coord_port,
        rank=node_rank,
        world_size=nnodes,
    )

    local_info = NodeInfo(device_count=local_gpu_count)

    logger.debug(
        "Node %d (%s) has %d visible GPUs",
        node_rank,
        local_ip,
        local_gpu_count,
    )

    # All-gather node infos from all nodes
    all_infos = coord_group.all_gather_obj(local_info)

    # Type narrow - all_infos should be fully populated after all_gather
    node_infos = [info for info in all_infos if info is not None]
    assert len(node_infos) == nnodes, (
        f"Expected {nnodes} node infos, got {len(node_infos)}"
    )

    # Build topology
    topology = RankTopology.from_node_infos(node_infos)

    logger.info(
        "Topology discovery complete. World size: %d, Node rank %d has %d GPUs",
        topology.world_size,
        node_rank,
        topology.get_device_count_for_node_rank(node_rank),
    )

    return topology


@config
class EPLBConfig:
    """Configuration for Expert Parallel Load Balancing (EP)."""

    window_size: int = 1000
    """Window size for expert load recording."""
    step_interval: int = 3000
    """
    Interval for rearranging experts in expert parallelism.

    Note that if this is greater than the EPLB window size, only the metrics
    of the last `lb_window_size` steps will be used for rearranging experts.
    """

    num_redundant_experts: int = Field(default=0, ge=0)
    """Number of redundant experts to use for expert parallelism."""

    log_balancedness: bool = False
    """
    Log the balancedness each step of expert parallelism.
    This is turned off by default since it will cause communication overhead.
    """
    log_balancedness_interval: int = 1
    """
    Interval for logging the balancedness.
    """
    use_async: bool = False
    """
    Whether to use non-blocking EPLB.
    """

    policy: EPLBPolicyOption = "default"
    """The policy type for expert parallel load balancing (EPLB)."""

    @model_validator(mode="after")
    def _validate_eplb_config(self) -> Self:
        if self.use_async and self.policy != "default":
            raise ValueError("Async EPLB is only supported with the default policy.")
        if self.log_balancedness and self.log_balancedness_interval <= 0:
            raise ValueError("log_balancedness_interval must be greater than 0.")
        return self


@config
class ParallelConfig:
    """Configuration for the distributed execution."""

    pipeline_parallel_size: int = 1
    """Number of pipeline parallel groups."""
    tensor_parallel_size: int = 1
    """Number of tensor parallel groups."""
    prefill_context_parallel_size: int = 1
    """Number of prefill context parallel groups."""
    data_parallel_size: int = 1
    """Number of data parallel groups. MoE layers will be sharded according to
    the product of the tensor parallel size and data parallel size."""
    data_parallel_size_local: int = 1
    """Number of local data parallel groups."""
    data_parallel_rank: int = 0
    """Rank of the data parallel group."""
    data_parallel_rank_local: int | None = None
    """Local rank of the data parallel group,
    set only in SPMD mode."""
    data_parallel_master_ip: str = "127.0.0.1"
    """IP of the data parallel master."""
    data_parallel_rpc_port: int = 29550
    """Port for data parallel messaging."""
    data_parallel_master_port: int = 29500
    """Port of the data parallel master."""
    data_parallel_backend: DataParallelBackend = "mp"
    """Backend to use for data parallel, either "mp" or "ray"."""
    data_parallel_external_lb: bool = False
    """Whether to use "external" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. This is useful for a "one-pod-per-rank"
    wide-EP setup in Kubernetes. Set implicitly when --data-parallel-rank
    is provided explicitly to vllm serve."""
    data_parallel_hybrid_lb: bool = False
    """Whether to use "hybrid" DP LB mode. Applies only to online serving
    and when data_parallel_size > 0. Enables running an AsyncLLM
    and API server on a "per-node" basis where vLLM load balances
    between local data parallel ranks, but an external LB balances
    between vLLM nodes/replicas. Set explicitly in conjunction with
    --data-parallel-start-rank."""
    is_moe_model: bool | None = None
    """Whether the deployed model is MoE (if known)."""
    enable_expert_parallel: bool = False
    """Use expert parallelism instead of tensor parallelism for MoE layers."""
    enable_eplb: bool = False
    """Enable expert parallelism load balancing for MoE layers."""
    eplb_config: EPLBConfig = Field(default_factory=EPLBConfig)
    """Expert parallelism configuration."""
    expert_placement_strategy: ExpertPlacementStrategy = "linear"
    """The expert placement strategy for MoE layers:\n
    - "linear": Experts are placed in a contiguous manner. For example, with 4
      experts and 2 ranks, rank 0 will have experts [0, 1] and rank 1 will have
      experts [2, 3].\n
    - "round_robin": Experts are placed in a round-robin manner. For example,
      with 4 experts and 2 ranks, rank 0 will have experts [0, 2] and rank 1
      will have experts [1, 3]. This strategy can help improve load balancing
      for grouped expert models with no redundant experts."""
    all2all_backend: All2AllBackend = "allgather_reducescatter"
    """All2All backend for MoE expert parallel communication. Available options:

    - "naive": Naive all2all implementation using broadcasts\n
    - "allgather_reducescatter": All2all based on allgather and reducescatter\n
    - "deepep_high_throughput": Use deepep high-throughput kernels\n
    - "deepep_low_latency": Use deepep low-latency kernels\n
    - "mori": Use mori kernels\n
    - "flashinfer_all2allv": Use flashinfer alltoallv kernels for mnnvl"""

    max_parallel_loading_workers: int | None = None
    """Maximum number of parallel loading workers when loading model
    sequentially in multiple batches. To avoid RAM OOM when using tensor
    parallel and large models."""

    disable_custom_all_reduce: bool = False
    """Disable the custom all-reduce kernel and fall back to NCCL."""

    enable_elastic_ep: bool = False
    """Enable elastic expert parallelism with stateless NCCL groups for DP/EP."""

    enable_dbo: bool = False
    """Enable dual batch overlap for the model executor."""
    ubatch_size: int = 0
    """Number of ubatch size."""

    dbo_decode_token_threshold: int = 32
    """The threshold for dual batch overlap for batches only containing decodes.
    If the number of tokens in the request is greater than this threshold,
    microbatching will be used. Otherwise, the request will be processed in a
    single batch."""
    dbo_prefill_token_threshold: int = 512  # TODO(lucas): tune
    """The threshold for dual batch overlap for batches that contain one or more
    prefills. If the number of tokens in the request is greater than this
    threshold, microbatching will be used. Otherwise, the request will be
    processed in a single batch."""

    disable_nccl_for_dp_synchronization: bool | None = Field(default=None)
    """Forces the dp synchronization logic in vllm/v1/worker/dp_utils.py 
    to use Gloo instead of NCCL for its all reduce.

    Defaults to True when async scheduling is enabled, False otherwise.
    """

    ray_workers_use_nsight: bool = False
    """Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler."""

    ray_runtime_env: RuntimeEnv | None = None
    """Ray runtime environment to pass to distributed workers."""

    placement_group: PlacementGroup | None = None
    """ray distributed model workers placement group."""

    distributed_executor_backend: (
        str | DistributedExecutorBackend | type[Executor] | None
    ) = None
    """Backend to use for distributed model workers, either "ray" or "mp"
    (multiprocessing). If the product of pipeline_parallel_size and tensor_parallel_size
    is less than or equal to the number of GPUs available, "mp" will be used to
    keep processing on a single host. Otherwise, an error will be raised. To use "mp"
    you must also set nnodes, and to use "ray" you must manually set
    distributed_executor_backend to "ray".

    Note that tpu only support Ray for distributed inference."""

    worker_cls: str = "auto"
    """The full name of the worker class to use. If "auto", the worker class
    will be determined based on the platform."""
    sd_worker_cls: str = "auto"
    """The full name of the worker class to use for speculative decoding.
    If "auto", the worker class will be determined based on the platform."""
    worker_extension_cls: str = ""
    """The full name of the worker extension class to use. The worker extension
    class is dynamically inherited by the worker class. This is used to inject
    new attributes and methods to the worker class for use in collective_rpc
    calls."""
    master_addr: str = "127.0.0.1"
    """distributed master address for multi-node distributed 
    inference when distributed_executor_backend is mp."""
    master_port: int = 29501
    """distributed master port for multi-node distributed 
    inference when distributed_executor_backend is mp."""
    node_rank: int = 0
    """distributed node rank for multi-node distributed 
    inference when distributed_executor_backend is mp."""
    nnodes: int = 1
    """num of nodes for multi-node distributed 
    inference when distributed_executor_backend is mp."""

    world_size: int = Field(init=False)
    """world_size is TPxPP, it affects the number of workers we create."""

    rank: int = 0
    """Global rank in distributed setup."""

    _data_parallel_master_port_list: list[int] = Field(default_factory=list)
    """List of open port auto-queried for data parallel messaging.
    Set to be private as it's not intended to be configured by users.
    """

    _stateless_dp_group_port_list: list[list[int]] = Field(default_factory=list)
    """List of open ports for stateless DP groups when enable_elastic_ep is True.
    Set to be private as it's not intended to be configured by users.
    It is a list of list[int], with each inner list contains a set of 3 ports
    to be used for setting up the stateless CPU/device/TCPStore groups
    in StatelessGroupCoordinator. The number of inner lists is equal to
    the number of DP groups, 
    i.e., len(self._stateless_dp_group_port_list) == world_size_across_dp // dp_size,
    and len(self._stateless_dp_group_port_list[i]) == 3 for all i.
    """

    _stateless_ep_group_port_list: list[list[int]] = Field(default_factory=list)
    """List of open ports for stateless EP groups when enable_elastic_ep is True.
    Set to be private as it's not intended to be configured by users.
    len(self._stateless_ep_group_port_list) == world_size_across_dp // ep_size,
    """

    _stateless_eplb_group_port_list: list[list[int]] = Field(default_factory=list)
    """List of open ports for stateless EPLB groups when enable_elastic_ep is True.
    Same topology as EP but separate NCCL communicator to avoid deadlocks.
    """

    _stateless_world_group_port_list: list[list[int]] = Field(default_factory=list)
    """List of open ports for stateless world group when enable_elastic_ep is True.
    Set to be private as it's not intended to be configured by users.
    len(self._stateless_world_group_port_list) == 1,
    """

    decode_context_parallel_size: int = 1
    """Number of decode context parallel groups, because the world size does
    not change by dcp, it simply reuse the GPUs of TP group, and tp_size
    needs to be divisible by dcp_size."""

    dcp_kv_cache_interleave_size: int = 1
    """
    Interleave size of kv_cache storage while using DCP.
    dcp_kv_cache_interleave_size has been replaced by cp_kv_cache_interleave_size,
    and will be deprecated when PCP is fully supported.

    """
    cp_kv_cache_interleave_size: int = 1
    """Interleave size of kv_cache storage while using DCP or PCP.
    For `total_cp_rank = pcp_rank * dcp_world_size + dcp_rank`,
        and `total_cp_world_size = pcp_world_size * dcp_world_size`.
    store interleave_size tokens on total_cp_rank i,
    then store next interleave_size tokens on total_cp_rank i+1.
    Interleave_size=1: token-level alignment, where token `i` is stored on
        total_cp_rank `i % total_cp_world_size`.
    Interleave_size=block_size: block-level alignment, where tokens are
        first populated to the preceding ranks. Tokens are then stored
        in (rank i+1, block j) only after (rank i, block j) is fully occupied.
    Block_size should be greater than or equal to cp_kv_cache_interleave_size.
    Block_size should be divisible by cp_kv_cache_interleave_size.
    """

    data_parallel_index: int = Field(init=False)
    """Equal to the data parallel rank but not used for torch process groups
    and not overridden for dense models."""

    _api_process_count: int = Field(default=1, gt=0)
    """
    The number of API processes initialized.

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
    """

    _api_process_rank: int = Field(default=0, ge=-1)
    """
    The rank of this API process, or `-1` for engine core processes
    under API server scale-out.

    Note:
        This is an internal config that is only valid for and
        should only be set by API server scale-out.
    """

    _rank_topology: RankTopology | None = Field(default=None)
    """
    Runtime topology describing rank distribution across physical nodes.

    This is set during engine config creation (create_engine_config) for both
    single-node and multi-node deployments with MP backend, enabling non-uniform
    GPU distributions across nodes (e.g., node 1 with 4 GPUs, node 2 with 2 GPUs).

    Note:
        This is an internal config set at runtime, not by user configuration.
    """

    @field_validator("disable_nccl_for_dp_synchronization", mode="wrap")
    @classmethod
    def _skip_none_validation(cls, value: Any, handler: Callable) -> Any:
        """Skip validation if the value is `None` when initialisation is delayed."""
        return None if value is None else handler(value)

    @model_validator(mode="after")
    def _validate_parallel_config(self) -> Self:
        if self._api_process_rank >= self._api_process_count:
            raise ValueError(
                "Invalid value of `_api_process_rank`. "
                f"Expected to be `-1` or `[0, {self._api_process_count})`, "
                f"but found: {self._api_process_rank}"
            )

        if self.all2all_backend == "pplx":
            logger.warning(
                "The 'pplx' all2all backend has been removed. "
                "Falling back to 'allgather_reducescatter'."
            )
            self.all2all_backend = "allgather_reducescatter"

        if self.data_parallel_size_local > self.data_parallel_size:
            raise ValueError(
                f"data_parallel_size_local ({self.data_parallel_size_local}) "
                f"must be <= data_parallel_size ({self.data_parallel_size})"
            )

        if self.data_parallel_size <= 1 and self.data_parallel_external_lb:
            raise ValueError(
                "data_parallel_external_lb can only be set when data_parallel_size > 1"
            )

        if self.enable_eplb:
            if not current_platform.is_cuda_alike():
                raise ValueError(
                    "Expert parallelism load balancing is only supported on "
                    "CUDA devices or ROCm devices now."
                )
            if not self.enable_expert_parallel:
                raise ValueError("enable_expert_parallel must be True to use EPLB.")
            if self.tensor_parallel_size * self.data_parallel_size <= 1:
                raise ValueError(
                    "EPLB requires tensor_parallel_size or data_parallel_size "
                    f"to be greater than 1, but got "
                    f"TP={self.tensor_parallel_size},DP={self.data_parallel_size}."
                )
        else:
            if self.eplb_config.num_redundant_experts != 0:
                raise ValueError(
                    "num_redundant_experts is set to "
                    f"{self.eplb_config.num_redundant_experts} but EPLB is not "
                    "enabled. Either enable EPLB or unset "
                    "num_redundant_experts."
                )

        # Note(hc): In the current implementation of decode context
        # parallel(DCP), tp_size needs to be divisible by dcp_size,
        # because the world size does not change by dcp, it simply
        # reuses the GPUs of TP group, and split one TP group into
        # tp_size//dcp_size DCP groups.
        if self.tensor_parallel_size % self.decode_context_parallel_size != 0:
            raise ValueError(
                f"tp_size={self.tensor_parallel_size} must be divisible by"
                f"dcp_size={self.decode_context_parallel_size}."
            )

        return self

    @property
    def world_size_across_dp(self) -> int:
        """world_size_across_dp is TPxPPxDP, it is the size of the world
        including data parallelism."""
        return self.world_size * self.data_parallel_size

    @property
    def use_ubatching(self) -> bool:
        return self.enable_dbo or self.ubatch_size > 1

    @property
    def num_ubatches(self) -> int:
        return 2 if self.enable_dbo else self.ubatch_size

    @property
    def local_engines_only(self) -> bool:
        """
        Client manages local+remote EngineCores in pure internal LB case.
        Client manages local EngineCores in hybrid and external LB case.
        """
        return self.data_parallel_external_lb or self.data_parallel_hybrid_lb

    def get_next_dp_init_port(self) -> int:
        """
        We might need to initialize process groups in multiple
        processes that is related to data parallelism,
        e.g. both in the worker and in the engine, which
        can live in different processes. To avoid port conflicts, we
        pop a new port from the prepared port list each time we need to
        initialize a new process group related to data parallelism.
        """
        if self._data_parallel_master_port_list:
            answer = self._data_parallel_master_port_list.pop()
        else:
            answer = self.data_parallel_master_port
            self.data_parallel_master_port += 1

        return answer

    def allocate_elastic_ep_ports(self) -> None:
        """Allocate all ports for elastic EP (stateless groups + DP master).

        Must be called AFTER ray.init() so that ports claimed by Ray's
        idle worker pool are already in use and won't be returned by
        get_open_ports_list().
        """
        if not self.enable_elastic_ep:
            return
        if self._stateless_world_group_port_list:
            return

        num_world_groups = 1
        dp_size = self.data_parallel_size
        ep_size = self.data_parallel_size * self.world_size_across_dp
        num_dp_groups = max(1, self.world_size_across_dp // dp_size)
        num_ep_groups = max(1, self.world_size_across_dp // ep_size)
        num_eplb_groups = num_ep_groups
        total_stateless_ports = (
            num_world_groups + num_dp_groups + num_ep_groups + num_eplb_groups
        ) * 3
        num_dp_master_ports = 5

        all_ports = get_open_ports_list(total_stateless_ports + num_dp_master_ports)

        self._data_parallel_master_port_list = all_ports[-num_dp_master_ports:]
        self.data_parallel_master_port = self._data_parallel_master_port_list.pop()
        all_ports = all_ports[:-num_dp_master_ports]

        self._stateless_world_group_port_list = [
            all_ports[i : i + 3] for i in range(0, num_world_groups * 3, 3)
        ]
        start_idx = num_world_groups * 3
        self._stateless_dp_group_port_list = [
            all_ports[i : i + 3]
            for i in range(start_idx, start_idx + num_dp_groups * 3, 3)
        ]
        start_idx += num_dp_groups * 3
        self._stateless_ep_group_port_list = [
            all_ports[i : i + 3]
            for i in range(start_idx, start_idx + num_ep_groups * 3, 3)
        ]
        start_idx += num_ep_groups * 3
        self._stateless_eplb_group_port_list = [
            all_ports[i : i + 3]
            for i in range(start_idx, start_idx + num_eplb_groups * 3, 3)
        ]

    def get_next_stateless_world_group_port(self) -> list[int]:
        return self._stateless_world_group_port_list.pop()

    def get_next_stateless_dp_group_port(self) -> list[int]:
        return self._stateless_dp_group_port_list.pop()

    def get_next_stateless_ep_group_port(self) -> list[int]:
        return self._stateless_ep_group_port_list.pop()

    def get_next_stateless_eplb_group_port(self) -> list[int]:
        return self._stateless_eplb_group_port_list.pop()

    def stateless_init_dp_group(self, return_store: bool = False) -> ProcessGroup:
        # NOTE: In high-concurrency scenarios multiple processes
        # can pick the same (currently free) port through a race
        # condition when calling `get_open_port()`. When the first
        # process binds the port the others will subsequently fail
        # with `torch.distributed.DistNetworkError: EADDRINUSE`.
        # To make the initialization more robust we retry a few times
        # with a fresh port whenever this specific error is observed.
        from torch.distributed import DistNetworkError

        from vllm.distributed.utils import (
            stateless_init_torch_distributed_process_group,
        )

        max_retries = 5
        last_exc: Exception | None = None
        for _ in range(max_retries):
            try:
                # use gloo since the engine process might not have cuda device
                return stateless_init_torch_distributed_process_group(
                    self.data_parallel_master_ip,
                    self.get_next_dp_init_port(),
                    self.data_parallel_rank,
                    self.data_parallel_size,
                    backend="gloo",
                    return_store=return_store,
                )
            except DistNetworkError as e:
                # We only want to retry when the root cause is EADDRINUSE.
                if "EADDRINUSE" in str(e):
                    logger.warning("Address already in use. Retrying with a new port.")
                    last_exc = e
                    continue  # try again with a new port
                raise e

        # If we get here all retries have failed.
        assert last_exc is not None
        raise last_exc

    # The all_reduce at the end of attention (during o_proj) means that
    # inputs are replicated across each rank of the tensor parallel group.
    # If using expert-parallelism with DeepEP All2All ops, replicated
    # tokens results in useless duplicate computation and communication.
    #
    # In this case, ensure the input to the experts is sequence parallel
    # to avoid the excess work.
    #
    @property
    def use_sequence_parallel_moe(self) -> bool:
        return (
            self.all2all_backend
            in (
                "allgather_reducescatter",
                "naive",
                "deepep_high_throughput",
                "deepep_low_latency",
                "mori",
            )
            and self.enable_expert_parallel
            and self.tensor_parallel_size > 1
            and self.data_parallel_size > 1
        )

    @property
    def node_rank_within_dp(self) -> int:
        """Get the node rank within its DP replica.

        Requires topology information to be available for accurate results,
        especially in non-uniform distribution scenarios.

        For single-node deployments, returns 0.
        For multi-node deployments, requires _rank_topology to be set during
        engine config creation.

        Returns:
            Node rank within the DP replica (0-indexed).

        Raises:
            RuntimeError: If topology is required but not initialized.
        """
        if self.nnodes == 1:
            return 0

        assert self._rank_topology is not None, (
            "RankTopology not initialized for multi-node deployment. "
            "This property requires topology information which should be "
            "discovered during engine config creation."
        )

        # Calculate dp_rank for this node
        dp_rank = self._rank_topology.get_dp_start_rank_for_node_rank(
            self.node_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
        )
        return self._rank_topology.get_node_rank_within_dp(
            self.node_rank,
            dp_rank,
            self.data_parallel_size,
        )

    @property
    def nnodes_within_dp(self) -> int:
        """Get the number of nodes within a single DP replica.

        Requires topology information for accurate results in non-uniform
        distribution scenarios.

        For single-node deployments, returns 1.
        For multi-node deployments, requires _rank_topology to be set.

        Returns:
            Number of nodes within a single DP replica.

        Raises:
            RuntimeError: If topology is required but not initialized.
        """
        if self.nnodes == 1:
            return 1

        assert self._rank_topology is not None, (
            "RankTopology not initialized for multi-node deployment. "
            "This property requires topology information which should be "
            "discovered during engine config creation."
        )

        # Calculate dp_rank for this node
        dp_rank = self._rank_topology.get_dp_start_rank_for_node_rank(
            self.node_rank,
            self.pipeline_parallel_size,
            self.tensor_parallel_size,
        )
        return self._rank_topology.get_nnodes_within_dp(
            self.data_parallel_size, dp_rank
        )

    @property
    def local_world_size(self) -> int:
        """Get the number of GPUs for this DP rank on this node.

        This is the number of GPUs that this specific DP replica uses on the
        current node. In MoE DP scenarios (world_size=1), each DP replica
        typically uses 1 GPU per node it spans.

        For TP/PP scenarios without DP, this equals the node's total GPU count.

        Returns:
            Number of GPUs for this DP rank on this node.

        Raises:
            RuntimeError: If topology is required but not initialized for
                multi-node deployments.
        """
        assert self._rank_topology is not None, (
            "RankTopology not initialized for multi-node deployment. "
            "This property requires topology information which should be "
            "discovered during engine config creation."
        )
        # Use DP-aware local world size calculation
        # world_size_within_dp = TP * PP (GPUs per DP replica)
        world_size_within_dp = self.tensor_parallel_size * self.pipeline_parallel_size
        return self._rank_topology.get_local_world_size_for_dp_rank(
            self.node_rank,
            self.data_parallel_rank,
            world_size_within_dp,
        )

    @property
    def data_parallel_start_rank(self) -> int:
        """The starting DP rank for this node.

        For multi-node deployments with non-uniform topology, this returns
        the actual starting DP rank based on RankTopology. For single-node
        or uniform deployments, falls back to 0.
        """
        assert self._rank_topology is not None, (
            "RankTopology not initialized for multi-node deployment. "
            "This property requires topology information which should be "
            "discovered during engine config creation."
        )
        return self._rank_topology.get_dp_start_rank_for_node_rank(
            self.node_rank,
            pp_size=self.pipeline_parallel_size,
            tp_size=self.tensor_parallel_size,
        )

    @staticmethod
    def has_unfinished_dp(dp_group: ProcessGroup, has_unfinished: bool) -> bool:
        tensor = torch.tensor([has_unfinished], dtype=torch.int32, device="cpu")
        # dp rank 0: has_unfinished_seqs=True
        # dp rank 1: has_unfinished_seqs=False
        # aggregated: has_unfinished_seqs=True
        # so this is an OR operation, i.e. MAX in integers
        torch.distributed.all_reduce(tensor, op=ReduceOp.MAX, group=dp_group)
        aggregated_has_unfinished = bool(tensor.item())
        return aggregated_has_unfinished

    @staticmethod
    def sync_kv_cache_memory_size(dp_group: ProcessGroup, kv_cache_memory: int) -> int:
        if kv_cache_memory == -1:
            kv_cache_memory = torch.iinfo(torch.int64).max
        tensor = torch.tensor([kv_cache_memory], dtype=torch.int64, device="cpu")
        # we cannot use broadcast for stateless dp group since it depends
        # on global rank
        torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=dp_group)
        return tensor.item()

    def compute_hash(self):
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.

        This hash is also used for DP worker configuration validation
        to prevent hangs from mismatched collective communication patterns.
        """
        ignored_factors = {
            # Derived/runtime topology, networking, or launch details
            "data_parallel_rank",
            "data_parallel_rank_local",
            "data_parallel_size_local",
            "data_parallel_index",
            "data_parallel_backend",
            "data_parallel_external_lb",
            "data_parallel_hybrid_lb",
            "data_parallel_master_ip",
            "data_parallel_master_port",
            "_data_parallel_master_port_list",
            "data_parallel_rpc_port",
            "rank",
            "master_addr",
            "master_port",
            "node_rank",
            "nnodes",
            "max_parallel_loading_workers",
            "disable_custom_all_reduce",
            "ray_workers_use_nsight",
            "ray_runtime_env",
            "placement_group",
            "distributed_executor_backend",
            "worker_cls",
            "sd_worker_cls",
            "worker_extension_cls",
            "_api_process_count",
            "_api_process_rank",
            "_rank_topology",
        }

        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    def __post_init__(self) -> None:
        # Continue with the rest of the initialization
        self.world_size = (
            self.pipeline_parallel_size
            * self.tensor_parallel_size
            * self.prefill_context_parallel_size
        )

        if self.distributed_executor_backend == "external_launcher":
            logger.info("Using external launcher for distributed inference.")
            self.world_size *= self.data_parallel_size

        if self.enable_elastic_ep:
            if not self.enable_eplb:
                raise ValueError("Elastic EP is only supported with enable_eplb=True.")
            if self.pipeline_parallel_size > 1:
                raise ValueError(
                    "Elastic EP is not supported with pipeline parallelism "
                    f"(pipeline_parallel_size={self.pipeline_parallel_size})."
                )
            if self.data_parallel_external_lb or self.data_parallel_hybrid_lb:
                raise NotImplementedError(
                    "Elastic EP is not compatible with data_parallel_external_lb "
                    "or data_parallel_hybrid_lb. Elastic EP relies on a single API "
                    "server and core client to coordinate scale up/down."
                )

        if self.data_parallel_size > 1 or self.data_parallel_size_local == 0:
            # Data parallel was specified in the engine args.
            if self.distributed_executor_backend == "external_launcher":
                # For external launcher,
                # we need to set the data parallel rank automatically
                self.data_parallel_rank = int(os.environ["RANK"]) // (
                    self.world_size // self.data_parallel_size
                )
                logger.info(
                    "Set data_parallel_rank to %d automatically.",
                    self.data_parallel_rank,
                )
            if not self.enable_elastic_ep:
                if not self._data_parallel_master_port_list:
                    self._data_parallel_master_port_list = get_open_ports_list(5)
                self.data_parallel_master_port = (
                    self._data_parallel_master_port_list.pop()
                )

            if not (0 <= self.data_parallel_rank < self.data_parallel_size):
                raise ValueError(
                    f"data_parallel_rank ({self.data_parallel_rank})"
                    f" must be in the range [0, {self.data_parallel_size})"
                )
        else:
            # Otherwise fall back to env vars (e.g. for offline SPMD case).
            self.data_parallel_size = envs.VLLM_DP_SIZE
            self.data_parallel_rank = envs.VLLM_DP_RANK
            self.data_parallel_rank_local = envs.VLLM_DP_RANK_LOCAL
            self.data_parallel_master_ip = envs.VLLM_DP_MASTER_IP
            self.data_parallel_master_port = envs.VLLM_DP_MASTER_PORT

            if self.data_parallel_size > 1 and self.is_moe_model is False:
                raise ValueError(
                    "Offline data parallel mode is not supported/useful"
                    " for dense models."
                )

        self.data_parallel_index = self.data_parallel_rank

        if self.distributed_executor_backend == "external_launcher":
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
            logger.info("Disabling V1 multiprocessing for external launcher.")

        if self.distributed_executor_backend is None and self.world_size_across_dp > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from vllm.v1.executor import ray_utils

            backend: DistributedExecutorBackend = "mp"
            ray_found = ray_utils.ray_is_available()
            if current_platform.is_tpu() and envs.VLLM_XLA_USE_SPMD:
                backend = "uni"
            elif current_platform.is_cuda() and self.nnodes > 1:
                backend = "mp"
            elif (
                current_platform.is_cuda()
                and cuda_device_count_stateless() < self.world_size
            ):
                gpu_count = cuda_device_count_stateless()
                raise ValueError(
                    f"World size ({self.world_size}) is larger than the number of "
                    f"available GPUs ({gpu_count}) in this node. If this is "
                    "intentional and you are using:\n"
                    "- ray, set '--distributed-executor-backend ray'.\n"
                    "- multiprocessing, set '--nnodes' appropriately."
                )
            elif self.data_parallel_backend == "ray":
                logger.info(
                    "Using ray distributed inference because "
                    "data_parallel_backend is ray"
                )
                backend = "ray"
            elif ray_found:
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized

                    if ray_is_initialized():
                        from ray.util import get_current_placement_group

                        if get_current_placement_group():
                            backend = "ray"
            self.distributed_executor_backend = backend
            logger.debug("Defaulting to use %s for distributed inference", backend)

        if self.distributed_executor_backend is None and self.world_size == 1:
            self.distributed_executor_backend = "uni"

        if self.max_parallel_loading_workers is not None:
            logger.warning(
                "max_parallel_loading_workers is currently "
                "not supported and will be ignored."
            )
        allowed_backends = ("mp", "uni", "external_launcher")
        if (
            self.distributed_executor_backend not in allowed_backends
            and self.nnodes > 1
        ):
            raise ValueError(
                "nnodes > 1 can only be set when distributed executor "
                "backend is mp, uni or external_launcher."
            )

        if (
            self.all2all_backend in ("allgather_reducescatter", "naive")
            and self.eplb_config.use_async
        ):
            logger.warning(
                "Async EPLB causes hangs with the '%s' all2all backend. "
                "Forcing synchronous EPLB.",
                self.all2all_backend,
            )
            self.eplb_config.use_async = False

    @property
    def use_ray(self) -> bool:
        return self.distributed_executor_backend == "ray" or (
            isinstance(self.distributed_executor_backend, type)
            and getattr(self.distributed_executor_backend, "uses_ray", False)
        )

    @model_validator(mode="after")
    def _verify_args(self) -> Self:
        # Lazy import to avoid circular import
        from vllm.v1.executor import Executor

        # Enable batch invariance settings if requested
        if vllm_is_batch_invariant():
            self.disable_custom_all_reduce = True

        if (
            self.distributed_executor_backend is not None
            and not isinstance(self.distributed_executor_backend, str)
            and not (
                isinstance(self.distributed_executor_backend, type)
                and issubclass(self.distributed_executor_backend, Executor)
            )
        ):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' 'uni', 'external_launcher', "
                " custom Executor subclass or its import path."
            )
        if self.use_ray:
            from vllm.v1.executor import ray_utils

            ray_utils.assert_ray_available()

        if not current_platform.use_custom_allreduce():
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on current platform."
            )
        if self.nnodes > 1:
            self.disable_custom_all_reduce = True
            logger.debug(
                "Disabled the custom all-reduce since we are running on multi-node."
            )
        if self.ray_workers_use_nsight and not self.use_ray:
            raise ValueError(
                "Unable to use nsight profiling unless workers run with Ray."
            )

        return self
