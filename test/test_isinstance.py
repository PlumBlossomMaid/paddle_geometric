import paddle
from paddle_geometric import is_paddle_instance
from paddle_geometric.testing import onlyLinux, withPackage


def test_basic():
    assert is_paddle_instance(
        paddle.nn.Linear(in_features=1, out_features=1), paddle.nn.Linear
    )


@onlyLinux
@withPackage("paddle>=3.0.0")
def test_compile():
    model = paddle.jit.to_static(paddle.nn.Linear(1, 1), backend="CINN")
    assert isinstance(model, paddle.nn.Linear)
    assert is_paddle_instance(model, paddle.nn.Linear)
