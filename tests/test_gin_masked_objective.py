"""Tests for the masked node-type objective helper (P2.3)."""

import torch

from s4_encoding.graph_gnn_encoder import N_NODE_TYPES, _mask_node_types


def test_mask_node_types_masks_at_least_one_per_graph():
    # Two graphs of 3 and 10 nodes, batched.
    n1, n2 = 3, 10
    x = torch.rand(n1 + n2, N_NODE_TYPES + 384)
    ptr = torch.tensor([0, n1, n1 + n2])

    x_masked, mask = _mask_node_types(x, ptr, mask_ratio=0.15)

    assert mask.shape == (n1 + n2,)
    # At least 1 node masked per graph (15% of 3 rounds to 0 -> floor of 1).
    assert mask[:n1].sum().item() >= 1
    assert mask[n1:].sum().item() >= 1
    # Masked nodes have zeroed type one-hot; label embedding (cols 4+) untouched.
    assert torch.all(x_masked[mask, :N_NODE_TYPES] == 0.0)
    assert torch.equal(x_masked[:, N_NODE_TYPES:], x[:, N_NODE_TYPES:])
    # Unmasked nodes are unchanged.
    assert torch.equal(x_masked[~mask], x[~mask])
