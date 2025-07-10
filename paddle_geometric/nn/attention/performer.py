import math
from typing import Callable, Optional
import paddle
from paddle import Tensor
import paddle.nn.functional as F

def _orthogonal_matrix(dim: int) -> Tensor:
    r"""Get an orthogonal matrix by applying QR decomposition."""
    mat = paddle.randn((dim, dim))
    q, _ = paddle.linalg.qr(mat, mode='reduced')
    return q.t()


def orthogonal_matrix(num_rows: int, num_cols: int) -> paddle.Tensor:
    """Generate an orthogonal matrix with `num_rows` rows
    and `num_cols` columns.
    """
    num_full_blocks = int(num_rows / num_cols)
    blocks = []
    for _ in range(num_full_blocks):
        q = _orthogonal_matrix(num_cols)
        blocks.append(q)
    remain_rows = num_rows - num_full_blocks * num_cols
    if remain_rows > 0:
        q = _orthogonal_matrix(num_cols)
        blocks.append(q[:remain_rows])
    mat = paddle.concat(x=blocks)
    return mat

def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm

def linear_attention(
    q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor
) -> paddle.Tensor:
    """Efficient attention mechanism from the
    `"Rethinking Attention with Performers"
    <https://arxiv.org/abs/2009.14794>`_ paper.

    .. math::
        \\mathbf{\\hat{D}}^{-1}(\\mathbf{Q}'((\\mathbf{K}')^{\\top} \\mathbf{V}))

    """
    D_inv = 1.0 / (q @ k.sum(axis=-2).unsqueeze(axis=-1))
    kv = k.transpose(perm=dim2perm(k.ndim, -2, -1)) @ v
    qkv = q @ kv
    out = paddle.einsum("...L,...Ld->...Ld", D_inv.squeeze(axis=-1), qkv)
    return out


def generalized_kernel(
    x: Tensor,
    mat: Tensor,
    kernel: Callable = F.relu,
    epsilon: float = 0.001,
) -> Tensor:
    batch_size, num_heads = x.shape[:2]
    projection = mat.t().expand([batch_size, num_heads, -1, -1])
    x = x @ projection
    out = kernel(x) + epsilon
    return out


class PerformerProjection(paddle.nn.Layer):
    r"""The fast attention that uses a projection matrix from the `"Rethinking Attention with Performers" <https://arxiv.org/abs/2009.14794>`_ paper.
    """
    def __init__(self, num_cols: int, kernel: Callable = F.relu):
        super().__init__()
        num_rows = int(num_cols * math.log(num_cols))
        self.num_rows = num_rows
        self.num_cols = num_cols
        projection_matrix = orthogonal_matrix(self.num_rows, self.num_cols)
        self.register_buffer('projection_matrix', projection_matrix)
        assert kernel is not None
        self.kernel = kernel

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = generalized_kernel(q, self.projection_matrix, self.kernel)
        k = generalized_kernel(k, self.projection_matrix, self.kernel)
        out = linear_attention(q, k, v)
        return out

# @finshed
class PerformerAttention(paddle.nn.Layer):
    """The linear scaled attention mechanism from the
    `"Rethinking Attention with Performers"
    <https://arxiv.org/abs/2009.14794>`_ paper.

    Args:
        channels (int): Size of each input sample.
        heads (int, optional): Number of parallel attention heads.
        head_channels (int, optional): Size of each attention head.
            (default: :obj:`64.`)
        kernel (Callable, optional): Kernels for generalized attention.
            If not specified, `ReLU` kernel will be used.
            (default: :obj:`torch.nn.ReLU()`)
        qkv_bias (bool, optional): If specified, add bias to query, key
            and value in the self attention. (default: :obj:`False`)
        attn_out_bias (bool, optional): If specified, add bias to the
            attention output. (default: :obj:`True`)
        dropout (float, optional): Dropout probability of the final
            attention output. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        channels: int,
        heads: int,
        head_channels: int = 64,
        kernel: Callable = F.relu,
        qkv_bias: bool = False,
        attn_out_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert channels % heads == 0
        if head_channels is None:
            head_channels = channels // heads

        self.heads = heads
        self.head_channels = head_channels
        self.kernel = kernel
        self.fast_attn = PerformerProjection(head_channels, kernel)

        inner_channels = head_channels * heads
        self.q = paddle.nn.Linear(channels, inner_channels, bias_attr=qkv_bias)
        self.k = paddle.nn.Linear(channels, inner_channels, bias_attr=qkv_bias)
        self.v = paddle.nn.Linear(channels, inner_channels, bias_attr=qkv_bias)
        self.attn_out = paddle.nn.Linear(inner_channels, channels, bias_attr=attn_out_bias)
        self.dropout = paddle.nn.Dropout(dropout)

    def forward(
        self, x: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Node feature tensor
                :math:`\\mathbf{X} \\in \\mathbb{R}^{B \\times N \\times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            mask (torch.Tensor, optional): Mask matrix
                :math:`\\mathbf{M} \\in {\\{ 0, 1 \\}}^{B \\times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
        """
        B, N, *_ = tuple(x.shape)
        q, k, v = self.q(x), self.k(x), self.v(x)
        q, k, v = map(
            lambda t: t.reshape(B, N, self.heads, self.head_channels).transpose(
                perm=[0, 2, 1, 3]
            ),
            (q, k, v),
        )
        if mask is not None:
            mask = mask[:, None, :, None]
            v.masked_fill_(mask=~mask, value=0.0)
        out = self.fast_attn(q, k, v)
        out = out.transpose(perm=[0, 2, 1, 3]).reshape([B, N, -1])
        out = self.attn_out(out)
        out = self.dropout(out)
        return out

    def redraw_projection_matrix(self):
        r"""As described in the paper, periodically redraw examples to improve overall approximation of attention."""
        num_rows = self.fast_attn.num_rows
        num_cols = self.fast_attn.num_cols
        projection_matrix = orthogonal_matrix(num_rows, num_cols)
        paddle.assign(projection_matrix, output=self.fast_attn.projection_matrix)
        del projection_matrix

    def _reset_parameters(self):
        # pass
        # self.q.weight.set_value(paddle.nn.initializer.KaimingUniform())
        # self.k.weight.set_value(paddle.nn.initializer.KaimingUniform())
        # self.v.weight.set_value(paddle.nn.initializer.KaimingUniform())
        # self.attn_out.weight.set_value(paddle.nn.initializer.KaimingUniform())
        self.redraw_projection_matrix()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(heads={self.heads}, head_channels={self.head_channels}, kernel={self.kernel})'
