import sys
import os
import tensorlayerx as tlx
from gammagl.datasets.actor import Actor

root = "./data"

def test_actor_dataset():
    dataset = Actor(root)

    # 检查数据集的长度
    assert len(dataset) == 1, f"Expected 1 graph, but got {len(dataset)}"

    # 获取第一个图对象
    data = dataset[0]

    # 使用 tlx 进行张量形状检查
    edge_num = tlx.get_tensor_shape(data.edge_index)[1]
    node_num = tlx.get_tensor_shape(data.x)[0]

    # 检查图的指标
    assert node_num == 7600, f"Expected 7600 nodes, but got {node_num}"
    assert edge_num == 30019, f"Expected 30019 edges, but got {edge_num}"
    assert tlx.get_tensor_shape(data.x)[1] == 932, f"Expected 932 features, but got {tlx.get_tensor_shape(data.x)[1]}"
    assert data.y.max().item() + 1 == 5, f"Expected 5 classes, but got {data.y.max().item() + 1}"


if __name__ == "__main__":
    test_actor_dataset()

