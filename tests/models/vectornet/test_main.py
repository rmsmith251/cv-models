import torch

from models.vision.vectornet import VectorNet


def test_VectorNet():
    model = VectorNet()
    original_loc = torch.tensor((1111, 1111, 1111))
    data = [
        [(0, 0, 0), (1113, 1113, 1111), (1,), (0,)],
        [(1113, 1113, 1111), (1115, 1115, 1111), (1,), (1,)],
        [(1115, 1115, 1111), (1117, 1117, 1111), (1,), (2,)],
    ]
    breakpoint()
    inputs = torch.tensor(data)
    inputs[:, 0] = inputs[:, 0] / original_loc[0]
    inputs[:, 1] = inputs[:, 1] / original_loc[1]
    inputs[:, 2] = inputs[:, 2] / original_loc[2]
    breakpoint()
    _ = model(inputs)


if __name__ == "__main__":
    test_VectorNet()
