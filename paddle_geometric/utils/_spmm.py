import warnings

import paddle
from paddle import Tensor

from paddle_geometric import EdgeIndex
from paddle_geometric.typing import Adj, SparseTensor
from paddle_geometric.utils import is_paddle_sparse_tensor


def spmm(
    src: Adj,
    other: Tensor,
    reduce: str = 'sum',
) -> Tensor:
    r"""Matrix product of sparse matrix with dense matrix.

    Args:
        src (paddle.Tensor or paddle_sparse.SparseTensor or EdgeIndex):
            The input sparse matrix which can be a
            :pyg:`paddle_geometric` :class:`paddle_sparse.SparseTensor`,
            a :paddle:`Paddle` :class:`paddle.sparse.Tensor` or
            a :pyg:`paddle_geometric` :class:`EdgeIndex`.
        other (paddle.Tensor): The input dense matrix.
        reduce (str, optional): The reduce operation to use
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`
    """
    reduce = "sum" if reduce == "add" else reduce
    if reduce not in ["sum", "mean", "min", "max"]:
        raise ValueError(f"`reduce` argument '{reduce}' not supported")
    warnings.warn(f"Dot not support reduce:{reduce}")
    if isinstance(src, EdgeIndex):
        return src.matmul(other=other)
    if isinstance(src, SparseTensor):
        if src.nnz() == 0:
            return paddle.zeros(shape=[src.shape[0], other.shape[1]],
                                dtype=other.dtype)
        if (other.dim() == 2 and not src.is_cuda()
                and not src.requires_grad()):
            csr = src.to_paddle_sparse_csr_tensor().to(other.dtype)
            return paddle.sparse.matmul(csr, other)
        # return paddle_sparse.matmul(src, other, reduce)
        raise NotImplementedError()
    if not is_paddle_sparse_tensor(src):
        raise ValueError(
            "'src' must be a 'torch_sparse.SparseTensor' or a 'torch.sparse.Tensor'"
        )
    if src.place.is_gpu_place() and (reduce == "min" or reduce == "max"):
        raise NotImplementedError(
            f"`{reduce}` reduction is not yet supported for 'torch.sparse.Tensor' on device '{src.place}'"
        )
    if src.is_sparse_coo():
        warnings.warn(
            "Converting sparse tensor to CSR format for more efficient "
            "processing. Consider converting your sparse tensor to CSR format "
            f"beforehand to avoid repeated conversion (got '{src.layout}')")
        src = src.to_sparse_csr()

    if reduce == "sum":
        return paddle.sparse.matmul(x=src, y=other)
    if src.is_sparse_csr() and not src.place.is_gpu_place():
        return paddle.sparse.matmul(src, other)
    if reduce == "mean":
        if src.is_sparse_csr():
            ptr = src.crows()
            deg = ptr[1:] - ptr[:-1]
        else:
            raise NotImplementedError()
        return paddle.sparse.matmul(x=src, y=other.cast(src.dtype)) / deg.view(
            -1, 1).cast(src.dtype).clip_(min=1)

    return paddle.sparse.matmul(src, other)
