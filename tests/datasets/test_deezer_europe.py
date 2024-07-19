import sys
import os
import tensorlayerx as tlx
from deezer_europe import DeezerEurope

root = "./data"

def test_deezer_europe_dataset():
    dataset = DeezerEurope(root)
    assert len(dataset) == 1, f"Expected 1 graph, but got {len(dataset)}"
    data = dataset[0]
    edge_num = tlx.get_tensor_shape(data.edge_index)[1]
    node_num = tlx.get_tensor_shape(data.x)[0]
    num_features = tlx.get_tensor_shape(data.x)[1]
    assert node_num == 28281, f"Expected 28281 nodes, but got {node_num}"
    assert edge_num == 185504, f"Expected 185504 edges, but got {edge_num}"
    assert num_features == 128, f"Expected 128 features, but got {num_features}"
    assert data.y.max().item() + 1 == 2, f"Expected 2 classes, but got {data.y.max().item() + 1}"


if __name__ == "__main__":
    test_deezer_europe_dataset()
