from typing import Optional, Tuple

import paddle

import paddle_geometric.typing
from paddle_geometric import is_compiling
from paddle_geometric.typing import pyg_lib


def index_sort(
    inputs: paddle.Tensor,
    max_value: Optional[int] = None,
    stable: bool = False,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    r"""Sorts the elements of the :obj:`inputs` tensor in ascending order.
    It is expected that :obj:`inputs` is one-dimensional and that it only
    contains positive integer values. If :obj:`max_value` is given, it can
    be used by the underlying algorithm for better performance.

    Args:
        inputs (Tensor): A vector with positive integer values.
        max_value (int, optional): The maximum value stored inside
            :obj:`inputs`. This value can be an estimation, but needs to be
            greater than or equal to the real maximum.
            (default: :obj:`None`)
        stable (bool, optional): Makes the sorting routine stable, which
            guarantees that the order of equivalent elements is preserved.
            (default: :obj:`False`)
    """
    if stable or not paddle_geometric.typing.WITH_INDEX_SORT or is_compiling():
        return paddle.sort(stable=stable,
                           x=inputs), paddle.argsort(stable=stable, x=inputs)

    return pyg_lib.ops.index_sort(inputs, max_value=max_value)
