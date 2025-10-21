import paddle

from paddle_geometric.testing import withDevice
from paddle_geometric.utils import index_sort


@withDevice
def test_index_sort_stable(device):
    for _ in range(100):
        inputs = paddle.randint(0, 4, shape=[
            10,
        ])
        inputs = inputs.to(device=device)

        out = index_sort(inputs, stable=True)
        expected = paddle.sort(inputs, stable=True), paddle.argsort(
            stable=True, x=inputs)

        assert paddle.equal_all(out[0], expected[0]).item()
        assert paddle.equal_all(out[1], expected[1]).item()
