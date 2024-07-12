import sys
import os
import tensorlayerx as tlx
from gammagl.datasets.actor import Actor

root = "./data"

def test_actor_dataset():
    dataset = Actor(root)
    assert len(dataset) == 1, f"Expected 1 graph, but got {len(dataset)}"
    data = dataset[0]
    edge_num = tlx.get_tensor_shape(data.edge_index)[1]
    node_num = tlx.get_tensor_shape(data.x)[0]
    assert node_num == 7600
    assert edge_num == 30019
    assert tlx.get_tensor_shape(data.x)[1] == 932
    assert data.y.max().item() + 1 == 5

if __name__ == "__main__":
    test_actor_dataset()

