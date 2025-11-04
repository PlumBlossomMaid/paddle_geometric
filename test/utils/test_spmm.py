import paddle
import pytest

from paddle_geometric import EdgeIndex
# from paddle_geometric.profile import benchmark
from paddle_geometric.testing import onlyCUDA
from paddle_geometric.utils import spmm


# Paddle: Not support CPU kernel of 'sparse.matmul' now.
@onlyCUDA
@pytest.mark.parametrize('reduce', ['sum', 'mean'])
def test_spmm_basic(reduce):
    device = 'gpu'
    src = paddle.randn(5, 4, device=device)
    other = paddle.randn(4, 8, device=device)

    out1 = (src @ other) / (src.size(1) if reduce == 'mean' else 1)
    out2 = spmm(src.to_sparse_csr(), other, reduce=reduce)
    assert out1.size() == (5, 8)
    assert paddle.allclose(out1, out2, atol=1e-6)
    # if paddle_geometric.typing.WITH_PADDLE_SPARSE:
    #     out3 = spmm(SparseTensor.from_dense(src), other, reduce=reduce)
    #     assert paddle.allclose(out2, out3, atol=1e-6)

    # Test `mean` reduction with isolated nodes:
    src[0] = 0.
    out1 = (src @ other) / (4. if reduce == 'mean' else 1.)
    out2 = spmm(src.to_sparse_csr(), other, reduce=reduce)
    assert out1.size() == (5, 8)
    assert paddle.allclose(out1, out2, atol=1e-6)
    # if paddle_geometric.typing.WITH_PADDLE_SPARSE:
    #     out3 = spmm(SparseTensor.from_dense(src), other, reduce=reduce)
    #     assert paddle.allclose(out2, out3, atol=1e-6)


@onlyCUDA
@pytest.mark.parametrize('reduce', ['min', 'max'])
def test_spmm_reduce(reduce):
    device = 'gpu'

    src = paddle.randn(5, 4, device=device)
    other = paddle.randn(4, 8, device=device)

    if src.is_cuda:
        with pytest.raises(NotImplementedError, match="not yet supported"):
            spmm(src.to_sparse_csr(), other, reduce)
    else:
        out1 = spmm(src.to_sparse_csr(), other, reduce)
        assert out1.size() == (5, 8)
        # if paddle_geometric.typing.WITH_PADDLE_SPARSE:
        #     out2 = spmm(SparseTensor.from_dense(src), other, reduce=reduce)
        #     assert paddle.allclose(out1, out2)


@onlyCUDA
@pytest.mark.parametrize('layout', ["coo", "csr"])
@pytest.mark.parametrize('reduce', ['sum', 'mean', 'min', 'max'])
def test_spmm_layout(layout, reduce):
    device = "gpu"
    src = paddle.randn(5, 4, device=device)
    if layout == "coo":
        src = src.to_sparse_coo(src.ndim)
    elif layout == "csr":
        src = src.to_sparse_csr()
    else:
        assert layout == "csr"
        src = src.to_sparse_csc()
    other = paddle.randn(4, 8, device=device)

    if src.is_cuda and reduce in {'min', 'max'}:
        with pytest.raises(NotImplementedError, match="not yet supported"):
            spmm(src, other, reduce=reduce)
    elif layout != "csr":
        with pytest.warns(UserWarning, match="Converting sparse tensor"):
            spmm(src, other, reduce=reduce)
    else:
        spmm(src, other, reduce=reduce)


# @pytest.mark.parametrize('reduce', ['sum', 'mean'])
# def test_spmm_jit(reduce):
#     @paddle.jit.script
#     def jit_paddle_sparse(src: SparseTensor, other: Tensor,
#                          reduce: str) -> Tensor:
#         return spmm(src, other, reduce=reduce)

#     @paddle.jit.script
#     def jit_paddle(src: Tensor, other: Tensor, reduce: str) -> Tensor:
#         return spmm(src, other, reduce=reduce)

#     src = paddle.randn(5, 4)
#     other = paddle.randn(4, 8)

#     out1 = src @ other
#     out2 = jit_paddle(src.to_sparse_csr(), other, reduce)
#     assert out1.size() == (5, 8)
#     if reduce == 'sum':
#         assert paddle.allclose(out1, out2, atol=1e-6)
#     if paddle_geometric.typing.WITH_PADDLE_SPARSE:
#         out3 = jit_paddle_sparse(SparseTensor.from_dense(src), other, reduce)
#         assert paddle.allclose(out2, out3, atol=1e-6)


@onlyCUDA
@pytest.mark.parametrize('reduce', ['sum', 'mean', 'min', 'max'])
def test_spmm_edge_index(reduce):
    device = "gpu"
    src = EdgeIndex(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        sparse_size=(4, 3),
        sort_order='row',
        device=device,
    )
    other = paddle.rand(3, 4, device=device)
    out = spmm(src, other, reduce=reduce)
    assert out.size() == (4, 4)

    # if not other.is_cuda or reduce not in ['min', 'max']:
    #     out2 = spmm(src.to_sparse_csr(), other, reduce=reduce)
    #     assert paddle.allclose(out, out2)


# if __name__ == '__main__':
#     import argparse

#     warnings.filterwarnings('ignore', ".*Sparse CSR tensor support.*")
#     warnings.filterwarnings('ignore', ".*Converting sparse tensor to CSR.*")

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--device', type=str, default='cuda')
#     parser.add_argument('--backward', action='store_true')
#     args = parser.parse_args()

#     num_nodes, num_edges = 10_000, 200_000
#     x = paddle.randn(num_nodes, 64, device=args.device)
#     edge_index = paddle.randint(num_nodes, (2, num_edges), device=args.device)

#     reductions = ['sum', 'mean']
#     if not x.is_cuda:
#         reductions.extend(['min', 'max'])
#     layouts = [paddle.sparse_coo, paddle.sparse_csr, paddle.sparse_csc]

#     for reduce, layout in itertools.product(reductions, layouts):
#         print(f'Aggregator: {reduce}, Layout: {layout}')

#         adj = to_paddle_coo_tensor(edge_index, size=num_nodes)
#         adj = adj.to_sparse(layout=layout)

#         benchmark(
#             funcs=[spmm],
#             func_names=['spmm'],
#             args=(adj, x, reduce),
#             num_steps=50 if args.device == 'cpu' else 500,
#             num_warmups=10 if args.device == 'cpu' else 100,
#             backward=args.backward,
#         )
