# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test RankTopology data structure."""

import pytest

from vllm.config.parallel import NodeInfo, RankTopology


def _get_global_ranks_for_node(topology: RankTopology, node_rank: int) -> list[int]:
    """Helper function to get global ranks for a node.

    This replaces the non-existent get_global_ranks_for_node_rank() method.
    """
    start_rank = topology.get_global_start_rank_for_node_rank(node_rank)
    device_count = topology.get_device_count_for_node_rank(node_rank)
    return list(range(start_rank, start_rank + device_count))


def _get_local_ranks_for_node(
    topology: RankTopology, node_rank: int
) -> list[tuple[int, int]]:
    """Helper function to get (global_rank, local_rank) pairs for a node.

    This replaces the non-existent get_local_ranks_for_node_rank() method.
    """
    start_rank = topology.get_global_start_rank_for_node_rank(node_rank)
    device_count = topology.get_device_count_for_node_rank(node_rank)
    return [(start_rank + i, i) for i in range(device_count)]


class TestRankTopology:
    """RankTopology unit tests."""

    def test_single_node(self):
        """Test single node topology (uniform case as a special case)."""
        node_infos = [
            NodeInfo(device_count=8),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        assert topology.world_size == 8
        assert topology.get_device_count_for_node_rank(0) == 8
        assert _get_global_ranks_for_node(topology, 0) == [0, 1, 2, 3, 4, 5, 6, 7]
        assert _get_local_ranks_for_node(topology, 0) == [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            (5, 5),
            (6, 6),
            (7, 7),
        ]

    def test_multi_node_uniform(self):
        """Test multi-node with uniform distribution (same GPU count per node)."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=4),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        assert topology.world_size == 8
        assert topology.get_device_count_for_node_rank(0) == 4
        assert topology.get_device_count_for_node_rank(1) == 4
        assert _get_global_ranks_for_node(topology, 0) == [0, 1, 2, 3]
        assert _get_global_ranks_for_node(topology, 1) == [4, 5, 6, 7]
        assert _get_local_ranks_for_node(topology, 0) == [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
        ]
        assert _get_local_ranks_for_node(topology, 1) == [
            (4, 0),
            (5, 1),
            (6, 2),
            (7, 3),
        ]

    def test_non_uniform_4_2(self):
        """Test non-uniform distribution: 4 GPUs on node 0, 2 GPUs on node 1."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=2),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        assert topology.world_size == 6
        assert topology.get_device_count_for_node_rank(0) == 4
        assert topology.get_device_count_for_node_rank(1) == 2
        assert _get_global_ranks_for_node(topology, 0) == [0, 1, 2, 3]
        assert _get_global_ranks_for_node(topology, 1) == [4, 5]
        assert _get_local_ranks_for_node(topology, 0) == [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
        ]
        assert _get_local_ranks_for_node(topology, 1) == [(4, 0), (5, 1)]

    def test_non_uniform_4_2_3(self):
        """Test non-uniform distribution with 3 nodes: 4, 2, 3 GPUs."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=2),
            NodeInfo(device_count=3),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        assert topology.world_size == 9
        assert topology.get_device_count_for_node_rank(0) == 4
        assert topology.get_device_count_for_node_rank(1) == 2
        assert topology.get_device_count_for_node_rank(2) == 3
        assert _get_global_ranks_for_node(topology, 0) == [0, 1, 2, 3]
        assert _get_global_ranks_for_node(topology, 1) == [4, 5]
        assert _get_global_ranks_for_node(topology, 2) == [6, 7, 8]

        assert _get_local_ranks_for_node(topology, 0) == [
            (0, 0),
            (1, 1),
            (2, 2),
            (3, 3),
        ]
        assert _get_local_ranks_for_node(topology, 1) == [(4, 0), (5, 1)]
        assert _get_local_ranks_for_node(topology, 2) == [(6, 0), (7, 1), (8, 2)]

    def test_get_device_count_errors(self):
        """Test error handling for invalid node_rank in get_device_count."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=2),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        # Test negative node_rank
        with pytest.raises(ValueError, match="out of range"):
            topology.get_device_count_for_node_rank(-1)

        # Test node_rank beyond available nodes
        with pytest.raises(ValueError, match="out of range"):
            topology.get_device_count_for_node_rank(2)

    def test_get_global_start_rank(self):
        """Test get_global_start_rank_for_node_rank method."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=2),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        assert topology.get_global_start_rank_for_node_rank(0) == 0
        assert topology.get_global_start_rank_for_node_rank(1) == 4

    def test_get_global_start_rank_behavior(self):
        """Test behavior of get_global_start_rank_for_node_rank."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=2),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        # Valid node_ranks
        assert topology.get_global_start_rank_for_node_rank(0) == 0
        assert topology.get_global_start_rank_for_node_rank(1) == 4

        # Note: The method doesn't validate invalid node_rank values.
        # It uses slice semantics, so:
        # - Negative index returns 0 (sum of empty slice)
        # - Out-of-bounds positive index returns sum of all elements

    def test_rank_to_local_rank_mapping(self):
        """Test the rank_to_local_rank mapping."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=2),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        # Verify rank_to_local_rank mapping
        assert topology.rank_to_local_rank[0] == 0
        assert topology.rank_to_local_rank[1] == 1
        assert topology.rank_to_local_rank[2] == 2
        assert topology.rank_to_local_rank[3] == 3
        assert topology.rank_to_local_rank[4] == 0  # local rank resets on new node
        assert topology.rank_to_local_rank[5] == 1

    def test_single_gpu_nodes(self):
        """Test distribution with single GPU nodes."""
        node_infos = [
            NodeInfo(device_count=1),
            NodeInfo(device_count=7),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        assert topology.world_size == 8
        assert topology.get_global_start_rank_for_node_rank(0) == 0
        assert topology.get_global_start_rank_for_node_rank(1) == 1
        assert _get_global_ranks_for_node(topology, 0) == [0]
        assert _get_global_ranks_for_node(topology, 1) == [1, 2, 3, 4, 5, 6, 7]

    def test_extreme_non_uniform(self):
        """Test extreme non-uniform distribution: 3, 2, 2, 1 GPUs."""
        node_infos = [
            NodeInfo(device_count=3),
            NodeInfo(device_count=2),
            NodeInfo(device_count=2),
            NodeInfo(device_count=1),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        assert topology.world_size == 8
        assert _get_global_ranks_for_node(topology, 0) == [0, 1, 2]
        assert _get_global_ranks_for_node(topology, 1) == [3, 4]
        assert _get_global_ranks_for_node(topology, 2) == [5, 6]
        assert _get_global_ranks_for_node(topology, 3) == [7]

        # Verify rank_to_local_rank mapping
        assert topology.rank_to_local_rank[0] == 0
        assert topology.rank_to_local_rank[1] == 1
        assert topology.rank_to_local_rank[2] == 2
        assert topology.rank_to_local_rank[3] == 0  # node 1
        assert topology.rank_to_local_rank[4] == 1
        assert topology.rank_to_local_rank[5] == 0  # node 2
        assert topology.rank_to_local_rank[6] == 1
        assert topology.rank_to_local_rank[7] == 0  # node 3


class TestRankTopologyDP:
    """Test RankTopology DP-related methods."""

    def test_get_dp_start_rank_basic(self):
        """Test get_dp_start_rank_for_node_rank with basic configuration."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=2),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        # With TP=1, PP=1, each GPU is a DP rank
        assert topology.get_dp_start_rank_for_node_rank(0, pp_size=1, tp_size=1) == 0
        assert topology.get_dp_start_rank_for_node_rank(1, pp_size=1, tp_size=1) == 4

    def test_get_dp_start_rank_with_tp(self):
        """Test get_dp_start_rank_for_node_rank with tensor parallelism."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=4),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        # With TP=2, PP=1, world_size_within_dp=2
        # Node 0 starts at global rank 0, DP rank 0
        # Node 1 starts at global rank 4, DP rank 2
        assert topology.get_dp_start_rank_for_node_rank(0, pp_size=1, tp_size=2) == 0
        assert topology.get_dp_start_rank_for_node_rank(1, pp_size=1, tp_size=2) == 2

    def test_get_nnodes_within_dp_single_node(self):
        """Test get_nnodes_within_dp with single-node DP groups."""
        node_infos = [
            NodeInfo(device_count=8),
            NodeInfo(device_count=8),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        # With DP=2, TP=4, each node hosts one DP replica
        assert topology.get_nnodes_within_dp(data_parallel_size=2, dp_rank=0) == 1
        assert topology.get_nnodes_within_dp(data_parallel_size=2, dp_rank=1) == 1

    def test_get_nnodes_within_dp_cross_node(self):
        """Test get_nnodes_within_dp with cross-node DP groups."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=2),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        # With TP=3, DP=2, world_size_within_dp=3
        # DP 0: ranks 0,1,2 (all in node 0) -> nnodes=1
        # DP 1: ranks 3,4,5 (rank 3 in node 0, ranks 4,5 in node 1) -> nnodes=2
        assert topology.get_nnodes_within_dp(data_parallel_size=2, dp_rank=0) == 1
        assert topology.get_nnodes_within_dp(data_parallel_size=2, dp_rank=1) == 2

    def test_get_node_rank_within_dp(self):
        """Test get_node_rank_within_dp for position within DP group."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=2),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        # With DP=2, TP=3 (world_size_within_dp=3)
        # DP 0: node 0 only -> position 0
        # DP 1: nodes 0 and 1 -> node 0 position 0, node 1 position 1
        assert (
            topology.get_node_rank_within_dp(
                node_rank=0, dp_rank=0, data_parallel_size=2
            )
            == 0
        )

        assert (
            topology.get_node_rank_within_dp(
                node_rank=0, dp_rank=1, data_parallel_size=2
            )
            == 0
        )
        assert (
            topology.get_node_rank_within_dp(
                node_rank=1, dp_rank=1, data_parallel_size=2
            )
            == 1
        )

    def test_get_local_world_size_for_dp_rank(self):
        """Test get_local_world_size_for_dp_rank."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=2),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        # With TP=3 (world_size_within_dp=3), DP=2
        # DP 0: ranks 0,1,2 -> node 0 contributes 3 GPUs
        # DP 1: ranks 3,4,5 -> node 0 contributes 1 GPU, node 1 contributes 2 GPUs
        assert (
            topology.get_local_world_size_for_dp_rank(
                node_rank=0, dp_rank=0, world_size_within_dp=3
            )
            == 3
        )
        assert (
            topology.get_local_world_size_for_dp_rank(
                node_rank=0, dp_rank=1, world_size_within_dp=3
            )
            == 1
        )
        assert (
            topology.get_local_world_size_for_dp_rank(
                node_rank=1, dp_rank=1, world_size_within_dp=3
            )
            == 2
        )


class TestRankTopologyValidation:
    """Test RankTopology validation methods."""

    def test_validate_uniform_distribution(self):
        """Test validation passes for uniform distribution."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=4),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        # Should pass for world_size=4 (TP*PP)
        topology.validate_distribution(world_size=4)

        # Should pass for world_size=1 (each GPU is a DP replica)
        topology.validate_distribution(world_size=1)

    def test_validate_non_uniform_divisible(self):
        """Test validation passes for non-uniform but valid distribution."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=2),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        # With world_size=3 (TP=3), distributed as:
        # DP 0: 0,1,2 (node 0)
        # DP 1: 3,4,5 (node 0 + node 1)
        # But node 0 has 4 GPUs which is not divisible by 3
        # This should FAIL because node 0 has 4 GPUs >= world_size=3, but 4 % 3 != 0
        with pytest.raises(ValueError, match="not divisible"):
            topology.validate_distribution(world_size=3)

    def test_validate_cross_node_dp(self):
        """Test validation for cross-node DP replicas."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=2),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        # With world_size=6 (TP=6), all 6 GPUs form one DP replica
        # Node 0: 4 GPUs (4 < 6)
        # Node 1: 2 GPUs (4 + 2 = 6 exactly)
        # This should pass
        topology.validate_distribution(world_size=6)

    def test_validate_cross_node_dp_invalid(self):
        """Test validation fails if cross-node DP doesn't sum correctly."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=3),  # 4 + 3 = 7, not divisible by 6
        ]
        topology = RankTopology.from_node_infos(node_infos)

        # With world_size=6, accumulated GPUs from node 0 is 4 (< 6)
        # Then from node 1: 4 + 3 = 7 != 6
        # Should fail
        with pytest.raises(ValueError, match="accumulated.*but world_size"):
            topology.validate_distribution(world_size=6)

    def test_validate_not_divisible(self):
        """Test validation fails when node GPU count is not divisible by world_size."""
        node_infos = [
            NodeInfo(device_count=5),  # 5 is not divisible by 4
        ]
        topology = RankTopology.from_node_infos(node_infos)

        with pytest.raises(ValueError, match="not divisible"):
            topology.validate_distribution(world_size=4)

    def test_validate_empty_distribution(self):
        """Test validation fails for empty GPU distribution."""
        topology = RankTopology(rank_to_local_rank={}, node_device_counts=[])

        with pytest.raises(ValueError, match="Empty GPU distribution"):
            topology.validate_distribution(world_size=1)

    def test_validate_single_node(self):
        """Test validation passes for single node with divisible GPU count."""
        node_infos = [
            NodeInfo(device_count=8),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        # Should pass for world_size=4
        topology.validate_distribution(world_size=4)

        # Should pass for world_size=2
        topology.validate_distribution(world_size=2)

        # Should pass for world_size=8
        topology.validate_distribution(world_size=8)


class TestRankTopologySerialization:
    """Test RankTopology serialization and deserialization."""

    def test_to_dict(self):
        """Test to_dict serialization."""
        node_infos = [
            NodeInfo(device_count=4),
            NodeInfo(device_count=2),
        ]
        topology = RankTopology.from_node_infos(node_infos)

        data = topology.to_dict()
        assert data["node_device_counts"] == [4, 2]
        assert data["rank_to_local_rank"] == {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 0,
            5: 1,
        }

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "node_device_counts": [4, 2],
            "rank_to_local_rank": {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 0,
                5: 1,
            },
        }
        topology = RankTopology.from_dict(data)

        assert topology.world_size == 6
        assert topology.get_device_count_for_node_rank(0) == 4
        assert topology.get_device_count_for_node_rank(1) == 2

    def test_round_trip(self):
        """Test serialization round-trip."""
        node_infos = [
            NodeInfo(device_count=3),
            NodeInfo(device_count=2),
            NodeInfo(device_count=1),
        ]
        topology1 = RankTopology.from_node_infos(node_infos)

        # Serialize and deserialize
        data = topology1.to_dict()
        topology2 = RankTopology.from_dict(data)

        # Verify they are equivalent
        assert topology1.world_size == topology2.world_size
        assert topology1.node_device_counts == topology2.node_device_counts
        assert topology1.rank_to_local_rank == topology2.rank_to_local_rank
