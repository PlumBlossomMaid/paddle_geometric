import os
from typing import List

import paddle
import pytest

from paddle_geometric import EdgeIndex, Index, place2devicestr
from paddle_geometric.edge_index import SortReturnType, set_tuple_item
from paddle_geometric.io import fs
from paddle_geometric.paddle_utils import *  # noqa
from paddle_geometric.testing import onlyCUDA, withCUDA, withPackage
from paddle_geometric.typing import INDEX_DTYPES, SparseTensor

DTYPES = [pytest.param(dtype, id=str(dtype)[6:]) for dtype in INDEX_DTYPES]
IS_UNDIRECTED = [
    pytest.param(False, id="directed"),
    pytest.param(True, id="undirected"),
]
TRANSPOSE = [pytest.param(False, id=""), pytest.param(True, id="transpose")]


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_basic(dtype, device):
    kwargs = dict(dtype=dtype, device=device, sparse_size=(3, 3))
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs)
    adj.validate()
    assert isinstance(adj, EdgeIndex)
    assert str(adj).startswith("""EdgeIndex([[0, 1, 1, 2],\n [1, 0, 2, 1]],""")
    assert "sparse_size=(3, 3), nnz=4" in str(adj)
    assert (f"device='{device}'" in str(adj)) == (device
                                                  != paddle.get_device())
    assert (f"dtype={dtype}" in str(adj)) == (dtype != paddle.int64)

    assert adj.dtype == dtype
    assert adj.device == device
    assert adj.sparse_size() == (3, 3)
    assert adj.sparse_size(0) == 3
    assert adj.sparse_size(-1) == 3
    assert adj.sort_order is None
    assert not adj.is_sorted
    assert not adj.is_sorted_by_row
    assert not adj.is_sorted_by_col
    assert not adj.is_undirected

    out = adj.as_tensor()
    assert not isinstance(out, EdgeIndex)
    assert out.dtype == dtype
    assert place2devicestr(out.place) == device
    out = adj * 1
    assert not isinstance(out, EdgeIndex)
    assert out.dtype == dtype
    assert place2devicestr(out.place) == device


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
def test_identity(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sparse_size=(3, 3), **kwargs)

    out = EdgeIndex(adj)
    assert not isinstance(out.as_tensor(), EdgeIndex)
    assert out.data_ptr() == adj.data_ptr()
    assert out.dtype == adj.dtype
    assert out.place == adj.place
    assert out.sparse_size() == adj.sparse_size()
    assert out.sort_order == adj.sort_order
    assert out.is_undirected == adj.is_undirected

    out = EdgeIndex(adj, sparse_size=(4, 4), sort_order="row")
    assert out.sparse_size() == (4, 4)
    assert out.sort_order == "row"


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_tensor(dtype, device):
    kwargs = dict(dtype=dtype, device=device, is_undirected=True)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)

    out = EdgeIndex(adj.to_sparse_coo())
    assert out.equal_all(y=adj).item()
    assert out.sort_order == "row"
    assert out.sparse_size() == (3, 3)
    assert out._indptr is None

    out = EdgeIndex(adj.to_sparse_csr())
    assert out.equal_all(y=adj).item()
    assert out.sort_order == "row"
    assert out.sparse_size() == (3, 3)
    assert out._indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4], place=device, dtype=dtype)).item()

    # Paddle don't suppport csc format
    # out = EdgeIndex(adj.to_sparse_csc())
    # assert out.equal_all(y=adj.sort_by("col")[0]).item()
    # assert out.sort_order == "col"
    # assert out.sparse_size() == (3, 3)
    # assert out._indptr.equal(paddle.to_tensor(data=[0, 1, 3, 4]))


def test_set_tuple_item():
    tmp = 0, 1, 2
    assert set_tuple_item(tmp, 0, 3) == (3, 1, 2)
    assert set_tuple_item(tmp, 1, 3) == (0, 3, 2)
    assert set_tuple_item(tmp, 2, 3) == (0, 1, 3)
    with pytest.raises(IndexError, match="tuple index out of range"):
        set_tuple_item(tmp, 3, 3)
    assert set_tuple_item(tmp, -1, 3) == (0, 1, 3)
    assert set_tuple_item(tmp, -2, 3) == (0, 3, 2)
    assert set_tuple_item(tmp, -3, 3) == (3, 1, 2)
    with pytest.raises(IndexError, match="tuple index out of range"):
        set_tuple_item(tmp, -4, 3)


def test_validate():
    with pytest.raises(ValueError, match="unsupported data type"):
        EdgeIndex([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="needs to be two-dimensional"):
        EdgeIndex([[[0], [1]], [[1], [0]]])
    with pytest.raises(ValueError, match="needs to have a shape of"):
        EdgeIndex([[0, 1], [1, 0], [1, 1]])
    with pytest.raises(ValueError, match="received a non-symmetric size"):
        EdgeIndex([[0, 1], [1, 0]], is_undirected=True, sparse_size=(2, 3))
    with pytest.raises(TypeError, match="invalid combination of arguments"):
        EdgeIndex(paddle.to_tensor(data=[[0, 1], [1, 0]]), "int64")
    with pytest.raises(TypeError, match="invalid keyword arguments"):
        EdgeIndex(paddle.to_tensor(data=[[0, 1], [1, 0]]), dtype="int64")
    with pytest.raises(ValueError, match="contains negative indices"):
        EdgeIndex([[-1, 0], [0, 1]]).validate()
    with pytest.raises(ValueError, match="than its number of rows"):
        EdgeIndex([[0, 10], [1, 0]], sparse_size=(2, 2)).validate()
    with pytest.raises(ValueError, match="than its number of columns"):
        EdgeIndex([[0, 1], [10, 0]], sparse_size=(2, 2)).validate()
    with pytest.raises(ValueError, match="not sorted by row indices"):
        EdgeIndex([[1, 0], [0, 1]], sort_order="row").validate()
    with pytest.raises(ValueError, match="not sorted by column indices"):
        EdgeIndex([[0, 1], [1, 0]], sort_order="col").validate()


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_undirected(dtype, device):
    kwargs = dict(dtype=dtype, device=device, is_undirected=True)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs)
    assert isinstance(adj, EdgeIndex)
    assert adj.is_undirected

    assert adj.sparse_size() == (None, None)
    adj.get_num_rows()
    assert adj.sparse_size() == (3, 3)
    adj.validate()

    adj = EdgeIndex([[0, 1], [1, 0]], sparse_size=(3, None), **kwargs)
    assert adj.sparse_size() == (3, 3)
    adj.validate()

    adj = EdgeIndex([[0, 1], [1, 0]], sparse_size=(None, 3), **kwargs)
    assert adj.sparse_size() == (3, 3)
    adj.validate()

    with pytest.raises(ValueError, match="'EdgeIndex' is not undirected"):
        EdgeIndex([[0, 1, 1, 2], [0, 0, 1, 1]], **kwargs).validate()


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
def test_fill_cache_(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)
    adj.validate().fill_cache_()
    assert adj.sparse_size() == (3, 3)
    assert adj._indptr.dtype == dtype
    assert adj._indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4], dtype=dtype, place=device)).item()
    assert adj._T_perm.dtype == dtype
    assert adj._T_perm.equal_all(
        paddle.to_tensor(data=[1, 0, 3, 2], dtype=dtype,
                         place=device)).item() or adj._T_perm.equal_all(
                             paddle.to_tensor(data=[1, 3, 0, 2], place=device,
                                              dtype=dtype)).item()
    assert adj._T_index[0].dtype == dtype
    assert (adj._T_index[0].equal_all(y=paddle.to_tensor(
        data=[1, 0, 2, 1], dtype=dtype, place=device)).item()
            or adj._T_index[0].equal_all(y=paddle.to_tensor(
                data=[1, 2, 0, 1], dtype=dtype, place=device)).item())
    assert adj._T_index[1].dtype == dtype
    assert (adj._T_index[1].equal_all(y=paddle.to_tensor(
        data=[0, 1, 1, 2], dtype=dtype, place=device)).item())
    if is_undirected:
        assert adj._T_indptr is None
    else:
        assert adj._T_indptr.dtype == dtype
        assert adj._T_indptr.equal_all(
            paddle.to_tensor(data=[0, 1, 3, 4], dtype=dtype, place=device))
    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], sort_order="col", **kwargs)
    adj.validate().fill_cache_()
    assert adj.sparse_size() == (3, 3)
    assert adj._indptr.dtype == dtype
    assert adj._indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4], dtype=dtype, place=device))
    assert adj._T_perm.equal_all(
        paddle.to_tensor(data=[1, 0, 3, 2], dtype=dtype,
                         place=device)).item() or adj._T_perm.equal_all(
                             paddle.to_tensor(data=[1, 3, 0, 2], dtype=dtype,
                                              place=device)).item()
    assert adj._T_index[0].dtype == dtype
    assert (adj._T_index[0].equal_all(y=paddle.to_tensor(
        data=[0, 1, 1, 2], place=device, dtype=dtype)).item())
    assert adj._T_index[1].dtype == dtype
    assert (adj._T_index[1].equal_all(y=paddle.to_tensor(
        data=[1, 0, 2, 1], place=device, dtype=dtype)).item()
            or adj._T_index[1].equal_all(y=paddle.to_tensor(
                data=[1, 2, 0, 1], place=device, dtype=dtype)).item())
    if is_undirected:
        assert adj._T_indptr is None
    else:
        assert adj._T_indptr.dtype == dtype
        assert adj._T_indptr.equal_all(
            paddle.to_tensor(data=[0, 1, 3, 4], place=device,
                             dtype=dtype)).item()


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
def test_clone(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)

    out = adj.clone()
    assert isinstance(out, EdgeIndex)
    assert out.dtype == dtype
    assert out.device == device
    assert out.is_sorted_by_row
    assert out.is_undirected == is_undirected

    out = paddle.clone(x=adj)
    assert isinstance(out, EdgeIndex)
    assert out.dtype == dtype
    assert out.device == device
    assert out.is_sorted_by_row
    assert out.is_undirected == is_undirected


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
def test_to_function(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)
    adj.fill_cache_()

    adj = adj.to(device)
    assert isinstance(adj, EdgeIndex)
    assert adj.device == device
    assert adj._indptr.dtype == dtype
    assert place2devicestr(adj._indptr.place) == device
    assert adj._T_perm.dtype == dtype
    assert place2devicestr(adj._T_perm.place) == device

    out = adj.cpu()
    assert isinstance(out, EdgeIndex)
    assert out.device == "cpu"

    out = adj.to("int32")
    assert out.dtype == paddle.int32

    assert isinstance(out, EdgeIndex)
    assert out._indptr.dtype == paddle.int32
    assert out._T_perm.dtype == paddle.int32

    out = adj.to("float32")
    assert not isinstance(out, EdgeIndex)
    assert out.dtype == paddle.float32

    out = adj.astype(dtype="int64")
    assert isinstance(out, EdgeIndex)
    assert out.dtype == paddle.int64

    out = adj.astype(dtype="int32")
    assert out.dtype == paddle.int32
    assert isinstance(out, EdgeIndex)


@onlyCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_cpu_cuda(dtype):
    kwargs = dict(dtype=dtype)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs)
    assert adj.is_cuda

    out = adj.cpu()
    assert isinstance(out, EdgeIndex)
    assert out.is_cpu

    out = out.cuda()
    assert isinstance(out, EdgeIndex)
    assert out.is_cuda


@pytest.mark.skip(reason='Paddle not support')
@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_share_memory(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)
    adj.fill_cache_()
    out = adj.share_memory_()
    assert isinstance(out, EdgeIndex)
    assert out.is_shared()
    assert out._data.is_shared()
    assert out._indptr.is_shared()
    assert out.data_ptr() == adj.data_ptr()


@pytest.mark.skip(reason='Paddle not support')
@onlyCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_pin_memory(dtype):
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=dtype)
    assert "pinned" not in str(adj.place)
    out = adj.pin_memory()
    assert "pinned" in str(out.place)


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_contiguous(dtype, device):
    kwargs = dict(dtype=dtype, place=device)
    data = paddle.to_tensor([[0, 1], [1, 0], [1, 2], [2, 1]], **kwargs).t()
    with pytest.raises(ValueError, match="needs to be contiguous"):
        EdgeIndex(data)
    adj = EdgeIndex(data.contiguous()).contiguous()
    assert isinstance(adj, EdgeIndex)
    assert True


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
def test_sort_by(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)
    out = adj.sort_by("row")
    assert isinstance(out, SortReturnType)
    assert isinstance(out.values, EdgeIndex)
    assert not isinstance(out.indices, EdgeIndex)
    assert out.values.equal_all(adj).item()
    assert out.indices is None

    adj = EdgeIndex([[0, 1, 2, 1], [1, 0, 1, 2]], **kwargs)
    out = adj.sort_by("row")
    assert isinstance(out, SortReturnType)
    assert isinstance(out.values, EdgeIndex)
    assert not isinstance(out.indices, EdgeIndex)
    assert (out.values[0].equal_all(y=paddle.to_tensor(
        data=[0, 1, 1, 2], place=device, dtype=dtype)).item())
    assert (out.values[1].equal_all(y=paddle.to_tensor(
        data=[1, 0, 2, 1], place=device, dtype=dtype)).item()
            or out.values[1].equal_all(y=paddle.to_tensor(
                data=[1, 2, 0, 1], place=device, dtype=dtype)).item())
    assert out.indices.equal_all(
        paddle.to_tensor(
            data=[0, 1, 3, 2], place=device)).item() or out.indices.equal_all(
                paddle.to_tensor(data=[0, 3, 1, 2], place=device)).item()

    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)
    out, perm = adj.sort_by("col")
    assert adj._T_perm is not None
    assert adj._T_index[0] is not None and adj._T_index[1] is not None
    assert (out[0].equal_all(y=paddle.to_tensor(
        data=[1, 0, 2, 1], place=device, dtype=dtype)).item()
            or out[0].equal_all(y=paddle.to_tensor(
                data=[1, 2, 0, 1], place=device, dtype=dtype)).item())
    assert out[1].equal_all(y=paddle.to_tensor(data=[0, 1, 1, 2], place=device,
                                               dtype=dtype)).item()
    assert (perm.equal_all(y=paddle.to_tensor(data=[1, 0, 3, 2], place=device,
                                              dtype=dtype)).item()
            or perm.equal_all(y=paddle.to_tensor(
                data=[1, 3, 0, 2], place=device, dtype=dtype)).item())
    assert out._T_perm is None
    assert out._T_index[0] is None and out._T_index[1] is None

    out, perm = out.sort_by("row")
    assert out[0].equal_all(y=paddle.to_tensor(data=[0, 1, 1, 2], place=device,
                                               dtype=dtype)).item()
    assert (out[1].equal_all(y=paddle.to_tensor(
        data=[1, 0, 2, 1], place=device, dtype=dtype)).item()
            or out[1].equal_all(y=paddle.to_tensor(
                data=[1, 2, 0, 1], place=device, dtype=dtype)).item())
    assert (perm.equal_all(y=paddle.to_tensor(data=[1, 0, 3, 2], place=device,
                                              dtype=dtype)).item()
            or perm.equal_all(y=paddle.to_tensor(
                data=[2, 3, 0, 1], place=device, dtype=dtype)).item())
    assert out._T_perm is None
    assert out._T_index[0] is None and out._T_index[1] is None


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
def test_cat(dtype, device, is_undirected):
    args = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj1 = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sparse_size=(3, 3), **args)
    adj2 = EdgeIndex([[1, 2, 2, 3], [2, 1, 3, 2]], sparse_size=(4, 4), **args)
    adj3 = EdgeIndex([[1, 2, 2, 3], [2, 1, 3, 2]], dtype=dtype, device=device)
    out = paddle.concat(x=[adj1, adj2], axis=1)
    assert tuple(out.shape) == (2, 8)
    assert isinstance(out, EdgeIndex)
    assert out.sparse_size() == (4, 4)
    assert not out.is_sorted
    assert out.is_undirected == is_undirected
    assert out._cat_metadata.nnz == [4, 4]
    assert out._cat_metadata.sparse_size == [(3, 3), (4, 4)]
    assert out._cat_metadata.sort_order == [None, None]
    assert out._cat_metadata.is_undirected == [is_undirected, is_undirected]
    out = paddle.concat(x=[adj1, adj2, adj3], axis=1)
    assert tuple(out.shape) == (2, 12)
    assert isinstance(out, EdgeIndex)
    assert out.sparse_size() == (None, None)
    assert not out.is_sorted
    assert not out.is_undirected
    out = paddle.concat(x=[adj1, adj2], axis=0)
    assert tuple(out.shape) == (4, 4)
    assert not isinstance(out, EdgeIndex)
    inplace = paddle.empty(shape=[2, 8], dtype=dtype)
    out = paddle.assign(paddle.concat(x=[adj1, adj2], axis=1), output=inplace)
    assert out.data_ptr() == inplace.data_ptr()
    assert not isinstance(out, EdgeIndex)
    assert not isinstance(inplace, EdgeIndex)


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
def test_flip(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)
    adj.fill_cache_()

    out = adj.flip(0)
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[1, 0, 2, 1], [0, 1, 1, 2]],
                                            place=device, dtype=dtype)).item()
    assert out.is_sorted_by_col
    assert out.is_undirected == is_undirected
    assert out._T_indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4], place=device, dtype=dtype)).item()

    out = adj.flip([0, 1])
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[1, 2, 0, 1], [2, 1, 1, 0]],
                                            place=device, dtype=dtype)).item()
    assert not out.is_sorted
    assert out.is_undirected == is_undirected
    assert out._T_indptr is None

    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], sort_order="col", **kwargs)
    out = adj.flip(0)
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[0, 1, 1, 2], [1, 0, 2, 1]],
                                            place=device, dtype=dtype)).item()
    assert out.is_sorted_by_row
    assert out.is_undirected == is_undirected


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
def test_index_select(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)

    index = paddle.to_tensor(data=[1, 3], place=device)
    out = adj.index_select(index, 1)
    assert out.equal_all(y=paddle.to_tensor(data=[[1, 2], [0, 1]],
                                            place=device, dtype=dtype)).item()
    assert isinstance(out, EdgeIndex)
    assert not out.is_sorted
    assert not out.is_undirected

    index = paddle.to_tensor(data=[0], place=device)
    out = adj.index_select(index, 0)
    assert out.equal_all(y=paddle.to_tensor(data=[[0, 1, 1, 2]], place=device,
                                            dtype=dtype)).item()
    assert not isinstance(out, EdgeIndex)

    index = paddle.to_tensor(data=[1, 3], place=device)
    inplace = paddle.empty(shape=[2, 2], dtype=dtype)
    out = paddle.index_select(adj, axis=1, index=index, out=inplace)

    assert out.data_ptr() == inplace.data_ptr()
    assert not isinstance(out, EdgeIndex)
    assert not isinstance(inplace, EdgeIndex)


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
def test_narrow(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)

    out = adj.narrow(dim=1, start=1, length=2)
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[1, 1], [0, 2]],
                                            place=device, dtype=dtype)).item()
    assert out.is_sorted_by_row
    assert not out.is_undirected

    out = adj.narrow(dim=0, start=0, length=1)
    assert not isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[0, 1, 1, 2]], place=device,
                                            dtype=dtype)).item()


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
def test_getitem(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)

    out = adj[:,
              paddle.to_tensor(data=[False, True, False, True], place=device)]
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[1, 2], [0, 1]],
                                            place=device, dtype=dtype)).item()
    assert out.is_sorted_by_row
    assert not out.is_undirected

    out = adj[..., paddle.to_tensor(data=[1, 3], place=device)]
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[1, 2], [0, 1]],
                                            place=device, dtype=dtype)).item()
    assert not out.is_sorted
    assert not out.is_undirected

    out = adj[..., 1::2]
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[1, 2], [0, 1]],
                                            place=device, dtype=dtype)).item()
    assert out.is_sorted_by_row
    assert not out.is_undirected

    out = adj[...]
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[0, 1, 1, 2], [1, 0, 2, 1]],
                                            place=device, dtype=dtype)).item()
    assert out.is_sorted_by_row
    assert out.is_undirected == is_undirected

    out = adj[None]
    assert not isinstance(out, EdgeIndex)
    assert out.equal_all(
        y=paddle.to_tensor(data=[[[0, 1, 1, 2], [1, 0, 2, 1]]], place=device,
                           dtype=dtype)).item()

    out = adj[0, 0]
    assert not isinstance(out, EdgeIndex)
    assert out.equal_all(
        y=paddle.to_tensor(data=0, place=device, dtype=dtype)).item()

    out = adj[:, 0]
    assert not isinstance(out, EdgeIndex)
    out = adj[paddle.to_tensor(data=[0], place=device, dtype=dtype)]
    assert not isinstance(out, EdgeIndex)

    out = adj[
        paddle.to_tensor(data=[0], place=device),
        paddle.to_tensor(data=[0], place=device),
    ]
    assert not isinstance(out, EdgeIndex)


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_select(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row",
                    sparse_size=(4, 5), **kwargs).fill_cache_()

    out = adj[0]
    assert isinstance(out, Index)
    assert out.equal_all(y=paddle.to_tensor(data=[0, 1, 1, 2], place=device,
                                            dtype=dtype)).item()
    assert out.dim_size == 4
    assert out.is_sorted
    assert out._indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4, 4], place=device,
                         dtype=dtype)).item()

    out = adj[-1]
    assert isinstance(out, Index)
    assert out.equal_all(y=paddle.to_tensor(data=[1, 0, 2, 1], place=device,
                                            dtype=dtype)).item()
    assert out.dim_size == 5
    assert not out.is_sorted
    assert out._indptr is None

    out = adj[-2, 2:4]
    assert isinstance(out, Index)
    assert out.equal_all(
        y=paddle.to_tensor(data=[1, 2], place=device, dtype=dtype)).item()
    assert out.dim_size == 4
    assert out.is_sorted
    assert out._indptr is None

    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], sort_order="col",
                    sparse_size=(5, 4), **kwargs).fill_cache_()

    out = adj[1]
    assert isinstance(out, Index)
    assert out.equal_all(y=paddle.to_tensor(data=[0, 1, 1, 2], place=device,
                                            dtype=dtype)).item()
    assert out.dim_size == 4
    assert out.is_sorted
    assert out._indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4, 4], place=device,
                         dtype=dtype)).item()

    out = adj[-2]
    assert isinstance(out, Index)
    assert out.equal_all(y=paddle.to_tensor(data=[1, 0, 2, 1], place=device,
                                            dtype=dtype)).item()
    assert out.dim_size == 5
    assert not out.is_sorted
    assert out._indptr is None

    out = adj[-1, 2:4]
    assert isinstance(out, Index)
    assert out.equal_all(
        y=paddle.to_tensor(data=[1, 2], place=device, dtype=dtype)).item()
    assert out.dim_size == 4
    assert out.is_sorted
    assert out._indptr is None


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_unbind(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row",
                    sparse_size=(4, 5), **kwargs).fill_cache_()

    row, col = adj
    assert isinstance(row, Index)
    assert row.equal_all(y=paddle.to_tensor(data=[0, 1, 1, 2], place=device,
                                            dtype=dtype)).item()
    assert row.dim_size == 4
    assert row.is_sorted
    assert row._indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4, 4], place=device,
                         dtype=dtype)).item()
    assert isinstance(col, Index)
    assert col.equal_all(y=paddle.to_tensor(data=[1, 0, 2, 1], place=device,
                                            dtype=dtype)).item()
    assert col.dim_size == 5
    assert not col.is_sorted
    assert col._indptr is None


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("value_dtype", [None, "float64"])
def test_to_dense(dtype, device, value_dtype):
    kwargs = dict(dtype=dtype, device=device)
    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], **kwargs)

    out = adj.to_dense(dtype=value_dtype)
    assert isinstance(out, paddle.Tensor)
    assert tuple(out.shape) == (3, 3)

    expected = [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    assert out.equal_all(y=paddle.to_tensor(data=expected, dtype=value_dtype,
                                            place=device)).item()

    value = paddle.arange(start=1, end=5, dtype=value_dtype or "float32",
                          device=device)
    out = adj.to_dense(value)
    assert isinstance(out, paddle.Tensor)
    assert tuple(out.shape) == (3, 3)

    expected = [[0.0, 2.0, 0.0], [1.0, 0.0, 4.0], [0.0, 3.0, 0.0]]
    assert out.equal_all(y=paddle.to_tensor(data=expected, dtype=value_dtype,
                                            place=device)).item()

    value = paddle.arange(start=1, end=5, dtype=value_dtype or "float32",
                          device=device)
    out = adj.to_dense(value.view(-1, 1))
    assert isinstance(out, paddle.Tensor)
    assert tuple(out.shape) == (3, 3, 1)

    expected = [[[0.0], [2.0], [0.0]], [[1.0], [0.0], [4.0]],
                [[0.0], [3.0], [0.0]]]
    assert out.equal_all(y=paddle.to_tensor(data=expected, dtype=value_dtype,
                                            place=device)).item()


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_to_sparse_coo(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], **kwargs)

    out = adj.to_sparse(layout='coo')
    assert isinstance(out, paddle.Tensor)
    assert out.dtype == paddle.float32
    assert place2devicestr(out.place) == device
    assert out.is_sparse_coo()
    assert tuple(out.shape) == (3, 3)
    assert adj.equal_all(y=out.indices()).item()
    assert not out.is_coalesced()

    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], **kwargs)
    out = adj.to_sparse_coo()
    assert isinstance(out, paddle.Tensor)
    assert out.dtype == paddle.float32
    assert place2devicestr(out.place) == device
    assert out.is_sparse_coo()
    assert tuple(out.shape) == (3, 3)
    assert adj.equal_all(y=out.indices()).item()
    assert not out.is_coalesced()

    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)
    out = adj.to_sparse_coo()
    assert isinstance(out, paddle.Tensor)
    assert out.dtype == paddle.float32
    assert place2devicestr(out.place) == device
    assert out.is_sparse_coo()
    assert tuple(out.shape) == (3, 3)
    assert adj.equal_all(y=out.indices()).item()
    assert out.is_coalesced()


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_to_sparse_csr(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    with pytest.raises(ValueError, match="not sorted"):
        EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs).to_sparse_csr()
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)

    out = adj.to_sparse_csr()
    assert isinstance(out, paddle.Tensor)
    assert out.dtype == paddle.float32
    assert place2devicestr(out.place) == device
    assert out.is_sparse_csr()
    assert tuple(out.shape) == (3, 3)
    assert adj._indptr.equal_all(out.crows()).item()
    assert adj[1].equal_all(y=out.cols()).item()


@pytest.mark.skip(reason="Paddle not support sparse_csc layout")
@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_to_sparse_csc(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    with pytest.raises(ValueError, match="not sorted"):
        EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs).to_sparse_csc()
    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], sort_order="col", **kwargs)

    out = adj.to_sparse_csc()
    assert isinstance(out, paddle.Tensor)
    assert out.dtype == "float32"
    assert out.place == device
    assert out.is_sparse_csc()
    assert tuple(out.shape) == (3, 3)
    assert adj._indptr.equal(out.ccol_indices())
    assert adj[0].equal_all(y=out.row_indices()).item()


@withCUDA
@withPackage("paddle_sparse")
def test_to_sparse_tensor(device):
    kwargs = dict(device=device)

    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs)
    out = adj.to_sparse_tensor()
    assert isinstance(out, SparseTensor)
    assert out.sizes() == [3, 3]
    row, col, _ = out.coo()
    assert row.equal_all(y=adj[0]).item()
    assert col.equal_all(y=adj[1]).item()


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
def test_add(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sparse_size=(3, 3), **kwargs)

    out = paddle.add(adj, 2, alpha=2)
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[4, 5, 5, 6], [5, 4, 6, 5]],
                                            dtype=dtype, place=device)).item()
    assert out.is_undirected == is_undirected
    assert out.sparse_size() == (7, 7)

    out = adj + paddle.to_tensor(data=[2], dtype=dtype, place=device)
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[2, 3, 3, 4], [3, 2, 4, 3]],
                                            place=device, dtype=dtype)).item()
    assert out.is_undirected == is_undirected
    assert out.sparse_size() == (5, 5)

    out = adj + paddle.to_tensor(data=[[2], [1]], dtype=dtype, place=device)
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[2, 3, 3, 4], [2, 1, 3, 2]],
                                            place=device, dtype=dtype)).item()
    assert not out.is_undirected
    assert out.sparse_size() == (5, 4)

    out = adj + paddle.to_tensor(data=[[2], [2]], dtype=dtype, place=device)
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[2, 3, 3, 4], [3, 2, 4, 3]],
                                            place=device, dtype=dtype)).item()
    assert out.is_undirected == is_undirected
    assert out.sparse_size() == (5, 5)

    out = adj.add(adj)
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[0, 2, 2, 4], [2, 0, 4, 2]],
                                            place=device, dtype=dtype)).item()
    assert not out.is_undirected
    assert out.sparse_size() == (6, 6)

    adj += 2
    assert isinstance(adj, EdgeIndex)
    assert adj.equal_all(y=paddle.to_tensor(data=[[2, 3, 3, 4], [3, 2, 4, 3]],
                                            place=device, dtype=dtype)).item()
    assert adj.is_undirected == is_undirected
    assert adj.sparse_size() == (5, 5)

    # with pytest.raises(RuntimeError, match="can't be cast"):
    #     adj += 2.5


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
def test_sub(dtype, device, is_undirected):
    kwargs = dict(dtype=dtype, device=device, is_undirected=is_undirected)
    adj = EdgeIndex([[4, 5, 5, 6], [5, 4, 6, 5]], sparse_size=(7, 7), **kwargs)

    out = paddle.subtract(adj, 2, alpha=2)
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[0, 1, 1, 2], [1, 0, 2, 1]],
                                            place=device, dtype=dtype)).item()
    assert out.is_undirected == is_undirected
    assert out.sparse_size() == (3, 3)

    out = adj - paddle.to_tensor(data=[2], dtype=dtype, place=device)
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[2, 3, 3, 4], [3, 2, 4, 3]],
                                            place=device, dtype=dtype)).item()
    assert out.is_undirected == is_undirected
    assert out.sparse_size() == (5, 5)

    out = adj - paddle.to_tensor(data=[[2], [1]], dtype=dtype, place=device)
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[2, 3, 3, 4], [4, 3, 5, 4]],
                                            place=device, dtype=dtype)).item()
    assert not out.is_undirected
    assert out.sparse_size() == (5, 6)

    out = adj - paddle.to_tensor(data=[[2], [2]], dtype=dtype, place=device)
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[2, 3, 3, 4], [3, 2, 4, 3]],
                                            place=device, dtype=dtype)).item()
    assert out.is_undirected == is_undirected
    assert out.sparse_size() == (5, 5)

    out = adj.sub(adj)
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[[0, 0, 0, 0], [0, 0, 0, 0]],
                                            place=device, dtype=dtype)).item()
    assert not out.is_undirected
    assert out.sparse_size() == (None, None)

    adj -= 2
    assert isinstance(adj, EdgeIndex)
    assert adj.equal_all(y=paddle.to_tensor(data=[[2, 3, 3, 4], [3, 2, 4, 3]],
                                            place=device, dtype=dtype)).item()
    assert adj.is_undirected == is_undirected
    assert adj.sparse_size() == (5, 5)

    # with pytest.raises(RuntimeError, match="can't be cast"):
    #     adj -= 2.5


# @pytest.mark.skip(reason='PaddlePaddle doesn\'t support sparse.spmm')
# @withCUDA
# @withPackage("paddle_sparse")
# @pytest.mark.parametrize("reduce", ReduceType.__args__)
# @pytest.mark.parametrize("transpose", TRANSPOSE)
# @pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
# def test_torch_sparse_spmm(device, reduce, transpose, is_undirected):
#     if is_undirected:
#         kwargs = dict(is_undirected=True)
#         adj = EdgeIndex(
#             [[0, 1, 1, 2], [1, 0, 2, 1]],
#             device=device,
#             **kwargs)
#     else:
#         adj = EdgeIndex([[0, 1, 1, 2], [2, 0, 1, 2]], device=device)
#     adj = adj.sort_by("col" if transpose else "row").values
#     x = paddle.randn(shape=[3, 1])
#     out = _paddle_sparse_spmm(adj, x, None, reduce, transpose)
#     exp = _scatter_spmm(adj, x, None, reduce, transpose)
#     assert out.allclose(y=exp, atol=1e-06).item()
#     x = paddle.randn(shape=[3, 1])
#     value = paddle.rand(shape=adj.shape[1])
#     out = _paddle_sparse_spmm(adj, x, value, reduce, transpose)
#     exp = _scatter_spmm(adj, x, value, reduce, transpose)
#     assert out.allclose(y=exp, atol=1e-06).item()
#     out_4 = paddle.randn(shape=[3, 1])
#     out_4.stop_gradient = not True
#     x1 = out_4
#     out_5 = x1.detach()
#     out_5.stop_gradient = not True
#     x2 = out_5
#     grad = paddle.randn(shape=x1.shape, dtype=x1.dtype)
#     out = _torch_sparse_spmm(adj, x1, None, reduce, transpose)
#     out.backward(grad_tensor=grad)
#     exp = _scatter_spmm(adj, x2, None, reduce, transpose)
#     exp.backward(grad_tensor=grad)
#     assert x1.grad.allclose(x2.grad, atol=1e-06)
#     x = paddle.randn(shape=[3, 1])
#     out_6 = paddle.rand(shape=adj.shape[1])
#     out_6.stop_gradient = not True
#     value1 = out_6
#     out_7 = value1.detach()
#     out_7.stop_gradient = not True
#     value2 = out_7
#     grad = paddle.randn(shape=x.shape, dtype=x.dtype)
#     out = _torch_sparse_spmm(adj, x, value1, reduce, transpose)
#     out.backward(grad_tensor=grad)
#     exp = _scatter_spmm(adj, x, value2, reduce, transpose)
#     exp.backward(grad_tensor=grad)
#     assert value1.grad.allclose(value2.grad, atol=1e-06)

# @pytest.mark.skip(reason='PaddlePaddle doesn\'t support sparse.spmm')
# @withCUDA
# @pytest.mark.parametrize("reduce", ReduceType.__args__)
# @pytest.mark.parametrize("transpose", TRANSPOSE)
# @pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
# def test_torch_spmm(device, reduce, transpose, is_undirected):
#     if is_undirected:
#         kwargs = dict(is_undirected=True)
# adj = EdgeIndex(
#     [[0, 1, 1, 2], [1, 0, 2, 1]],
#     device=device,
#     **kwargs)
#     else:
#         adj = EdgeIndex([[0, 1, 1, 2], [2, 0, 1, 2]], device=device)
#     adj, perm = adj.sort_by("col" if transpose else "row")
#     x = paddle.randn(shape=[3, 2])
#     if (not x.place.is_gpu_place() and paddle_geometric.typing.WITH_PT20
#             or reduce in ["sum", "add"]):
#         out = _TorchSPMM.apply(adj, x, None, reduce, transpose)
#         exp = _scatter_spmm(adj, x, None, reduce, transpose)
#         assert out.allclose(y=exp).item()
#     else:
#         with pytest.raises(AssertionError):
#             _TorchSPMM.apply(adj, x, None, reduce, transpose)
#     x = paddle.randn(shape=[3, 1])
#     value = paddle.rand(shape=adj.shape[1])
#     if (not x.place.is_gpu_place() and paddle_geometric.typing.WITH_PT20
#             or reduce in ["sum", "add"]):
#         out = _TorchSPMM.apply(adj, x, value, reduce, transpose)
#         exp = _scatter_spmm(adj, x, value, reduce, transpose)
#         assert out.allclose(y=exp).item()
#     else:
#         with pytest.raises(AssertionError):
#             _TorchSPMM.apply(adj, x, value, reduce, transpose)
#     out_8 = paddle.randn(shape=[3, 1])
#     out_8.stop_gradient = not True
#     x1 = out_8
#     out_9 = x1.detach()
#     out_9.stop_gradient = not True
#     x2 = out_9
#     grad = paddle.randn(shape=x1.shape, dtype=x1.dtype)
#     if reduce in ["sum", "add"]:
#         out = _TorchSPMM.apply(adj, x1, None, reduce, transpose)
#         out.backward(grad_tensor=grad)
#         exp = _scatter_spmm(adj, x2, None, reduce, transpose)
#         exp.backward(grad_tensor=grad)
#         assert x1.grad.allclose(x2.grad)
#     else:
#         with pytest.raises(AssertionError):
#             out = _TorchSPMM.apply(adj, x1, None, reduce, transpose)
#             out.backward(grad_tensor=grad)
#     x = paddle.randn(shape=[3, 1])
#     out_10 = paddle.rand(shape=adj.shape[1])
#     out_10.stop_gradient = not True
#     value1 = out_10
#     grad = paddle.randn(shape=x.shape, dtype=x.dtype)
#     with pytest.raises((AssertionError, NotImplementedError)):
#         out = _TorchSPMM.apply(adj, x, value1, reduce, transpose)
#         out.backward(grad_tensor=grad)

# @pytest.mark.skip(reason='PaddlePaddle doesn\'t support sparse.spmm')
# @withCUDA
# @withoutExtensions
# @pytest.mark.parametrize("reduce", ReduceType.__args__)
# @pytest.mark.parametrize("transpose", TRANSPOSE)
# @pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
# def test_spmm(without_extensions, device, reduce, transpose, is_undirected):
#     warnings.filterwarnings("ignore", ".*can be accelerated via.*")
#     if is_undirected:
#         kwargs = dict(is_undirected=True)
# adj = EdgeIndex(
#     [[0, 1, 1, 2], [1, 0, 2, 1]],
#     device=device,
#     **kwargs)
#     else:
#         adj = EdgeIndex([[0, 1, 1, 2], [2, 0, 1, 2]], device=device)
#     adj = adj.sort_by("col" if transpose else "row").values
#     x = paddle.randn(shape=[3, 1])
#     with pytest.raises(ValueError, match="to be sorted by"):
#         adj.matmul(x, reduce=reduce, transpose=not transpose)
#     out = adj.matmul(x, reduce=reduce, transpose=transpose)
#     exp = _scatter_spmm(adj, x, None, reduce, transpose)
#     assert out.allclose(y=exp).item()
#     x = paddle.randn(shape=[3, 1])
#     value = paddle.rand(shape=adj.shape[1])
#     with pytest.raises(ValueError, match="'other_value' not supported"):
#         adj.matmul(x, reduce=reduce, other_value=value, transpose=transpose)
#     out = adj.matmul(x, value, reduce=reduce, transpose=transpose)
#     exp = _scatter_spmm(adj, x, value, reduce, transpose)
#     assert out.allclose(y=exp).item()
#     out_11 = paddle.randn(shape=[3, 1])
#     out_11.stop_gradient = not True
#     x1 = out_11
#     out_12 = x1.detach()
#     out_12.stop_gradient = not True
#     x2 = out_12
#     grad = paddle.randn(shape=x1.shape, dtype=x1.dtype)
#     out = adj.matmul(x1, reduce=reduce, transpose=transpose)
#     out.backward(grad_tensor=grad)
#     exp = _scatter_spmm(adj, x2, None, reduce, transpose)
#     exp.backward(grad_tensor=grad)
#     assert x1.grad.allclose(x2.grad)
#     x = paddle.randn(shape=[3, 1])
#     out_13 = paddle.rand(shape=adj.shape[1])
#     out_13.stop_gradient = not True
#     value1 = out_13
#     out_14 = value1.detach()
#     out_14.stop_gradient = not True
#     value2 = out_14
#     grad = paddle.randn(shape=x.shape, dtype=x.dtype)
#     out = adj.matmul(x, value1, reduce=reduce, transpose=transpose)
#     out.backward(grad_tensor=grad)
#     exp = _scatter_spmm(adj, x, value2, reduce, transpose)
#     exp.backward(grad_tensor=grad)
#     assert value1.grad.allclose(value2.grad)

# @pytest.mark.skip(reason='PaddlePaddle doesn\'t support spspmm')
# @withCUDA
# @pytest.mark.parametrize("reduce", ReduceType.__args__)
# @pytest.mark.parametrize("transpose", TRANSPOSE)
# @pytest.mark.parametrize("is_undirected", IS_UNDIRECTED)
# def test_spspmm(device, reduce, transpose, is_undirected):
#     if is_undirected:
#         kwargs = dict(device=device, sort_order="row", is_undirected=True)
#         adj1 = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs)
#     else:
#         kwargs = dict(device=device, sort_order="row")
#         adj1 = EdgeIndex([[0, 1, 1, 2], [2, 0, 1, 2]], **kwargs)
#     adj1_dense = adj1.to_dense().t() if transpose else adj1.to_dense()
#     adj2 = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], sort_order="col",
#                      device=device)
#     adj2_dense = adj2.to_dense()
#     if reduce in ["sum", "add"]:
#         out, value = adj1.matmul(adj2, reduce=reduce, transpose=transpose)
#         assert isinstance(out, EdgeIndex)
#         assert out.is_sorted_by_row
#         assert out._sparse_size == (3, 3)
#         if not paddle_geometric.typing.NO_MKL:
#             assert out._indptr is not None
#         assert paddle.allclose(x=out.to_dense(value),
#                                y=adj1_dense @ adj2_dense).item()
#     else:
#         with pytest.raises(NotImplementedError, match="not yet supported"):
#             adj1.matmul(adj2, reduce=reduce, transpose=transpose)


@withCUDA
def test_matmul(device):
    kwargs = dict(sort_order="row", device=device)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], **kwargs)
    x = paddle.randn(shape=[3, 1])
    expected = adj.to_dense() @ x

    out = adj @ x
    assert paddle.allclose(x=out, y=expected).item()

    out = adj.matmul(x)
    assert paddle.allclose(x=out, y=expected).item()

    out = paddle.mm(adj, x)
    assert paddle.allclose(x=out, y=expected).item()

    out = paddle.matmul(x=adj, y=x)
    assert paddle.allclose(x=out, y=expected).item()


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_row_narrow(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)

    out = adj.sparse_narrow(dim=0, start=1, length=1)
    assert out.equal_all(y=paddle.to_tensor(data=[[0, 0], [0, 2]],
                                            place=device, dtype=dtype)).item()
    assert out.sparse_size() == (1, None)
    assert out.sort_order == "row"
    assert out._indptr.equal_all(
        paddle.to_tensor(data=[0, 2], place=device, dtype=dtype))

    out = adj.sparse_narrow(dim=0, start=2, length=0)
    assert out.equal_all(
        y=paddle.to_tensor(data=[[], []], place=device, dtype=dtype)).item()
    assert out.sparse_size() == (0, None)
    assert out.sort_order == "row"
    assert out._indptr is None

    out = adj.sparse_narrow(dim=1, start=1, length=1)
    assert out.equal_all(
        y=paddle.to_tensor(data=[[0, 2], [0, 0]], place=device)).item()
    assert out.sparse_size() == (3, 1)
    assert out.sort_order == "row"
    assert out._indptr is None

    out = adj.sparse_narrow(dim=1, start=2, length=0)
    assert out.equal_all(
        y=paddle.to_tensor(data=[[], []], place=device, dtype=dtype)).item()
    assert out.sparse_size() == (3, 0)
    assert out.sort_order == "row"
    assert out._indptr is None


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_col_narrow(dtype, device):
    kwargs = dict(dtype=dtype, device=device)
    adj = EdgeIndex([[1, 0, 2, 1], [0, 1, 1, 2]], sort_order="col", **kwargs)

    out = adj.sparse_narrow(dim=1, start=1, length=1)
    assert out.equal_all(y=paddle.to_tensor(data=[[0, 2], [0, 0]],
                                            place=device, dtype=dtype)).item()
    assert out.sparse_size() == (None, 1)
    assert out.sort_order == "col"
    assert out._indptr.equal_all(
        paddle.to_tensor(data=[0, 2], place=device, dtype=dtype))

    out = adj.sparse_narrow(dim=1, start=2, length=0)
    assert out.equal_all(
        y=paddle.to_tensor(data=[[], []], place=device, dtype=dtype)).item()
    assert out.sparse_size() == (None, 0)
    assert out.sort_order == "col"
    assert out._indptr is None

    out = adj.sparse_narrow(dim=0, start=1, length=1)
    assert out.equal_all(
        y=paddle.to_tensor(data=[[0, 0], [0, 2]], place=device)).item()
    assert out.sparse_size() == (1, 3)
    assert out.sort_order == "col"
    assert out._indptr is None

    out = adj.sparse_narrow(dim=0, start=2, length=0)
    assert out.equal_all(
        y=paddle.to_tensor(data=[[], []], place=device, dtype=dtype)).item()
    assert out.sparse_size() == (0, 3)
    assert out.sort_order == "col"
    assert out._indptr is None


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_sparse_resize(dtype, device):
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=dtype, device=device)
    out = adj.sort_by("row")[0].fill_cache_()
    assert out.sparse_size() == (3, 3)
    assert out._indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4], place=device, dtype=dtype))
    assert out._T_indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4], place=device, dtype=dtype))

    out = out.sparse_resize_(4, 5)
    assert out.sparse_size() == (4, 5)
    assert out._indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4, 4], place=device, dtype=dtype))
    assert out._T_indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4, 4, 4], place=device, dtype=dtype))

    out = out.sparse_resize_(3, 3)
    assert out.sparse_size() == (3, 3)
    assert out._indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4], place=device, dtype=dtype))
    assert out._T_indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4], place=device, dtype=dtype))

    out = out.sparse_resize_(None, None)
    assert out.sparse_size() == (None, None)
    assert out._indptr is None
    assert out._T_indptr is None

    out = adj.sort_by("col")[0].fill_cache_()
    assert out.sparse_size() == (3, 3)
    assert out._indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4], place=device, dtype=dtype))
    assert out._T_indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4], place=device, dtype=dtype))

    out = out.sparse_resize_(4, 5)
    assert out.sparse_size() == (4, 5)
    assert out._indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4, 4, 4], place=device, dtype=dtype))
    assert out._T_indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4, 4], place=device, dtype=dtype))

    out = out.sparse_resize_(3, 3)
    assert out.sparse_size() == (3, 3)
    assert out._indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4], place=device, dtype=dtype))
    assert out._T_indptr.equal_all(
        paddle.to_tensor(data=[0, 1, 3, 4], place=device, dtype=dtype))

    out = out.sparse_resize_(None, None)
    assert out.sparse_size() == (None, None)
    assert out._indptr is None
    assert out._T_indptr is None


@pytest.mark.skip(reason='Not supported yet')
def test_to_list():
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]])
    with pytest.raises(RuntimeError, match="supported for tensor subclasses"):
        adj.tolist()


@pytest.mark.skip(reason='Not supported yet')
def test_numpy():
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]])
    with pytest.raises(RuntimeError, match="supported for tensor subclasses"):
        adj.numpy()


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_global_mapping(device, dtype):
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], device=device, dtype=dtype)
    n_id = paddle.to_tensor(data=[10, 20, 30], dtype=dtype, place=device)
    expected = paddle.to_tensor(data=[[10, 20, 20, 30], [20, 10, 30, 20]],
                                place=device, dtype=dtype)
    out = n_id[adj]
    assert not isinstance(out, EdgeIndex)
    assert out.equal_all(expected).item()


@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_to_vector(device, dtype):
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], device=device, dtype=dtype)
    out = adj.to_vector()
    assert not isinstance(out, EdgeIndex)
    assert out.equal_all(y=paddle.to_tensor(data=[1, 3, 5, 7], place=device,
                                            dtype=dtype)).item()


@pytest.mark.skip(reason="EdgeIndex.save() not support")
@withCUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_save_and_load(dtype, device, tmp_path):
    kwargs = dict(dtype=dtype, device=device)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)
    adj.fill_cache_()
    assert adj.sort_order == "row"
    assert adj._indptr is not None
    path = os.path.join(tmp_path, "edge_index.pt")
    paddle.save(obj=adj, path=path)
    out = fs.paddle_load(path)
    assert isinstance(out, EdgeIndex)
    assert out.equal_all(y=adj).item()
    assert out.sort_order == "row"
    assert out._indptr.equal(adj._indptr)


def _collate_fn(edge_indices: List[EdgeIndex]) -> List[EdgeIndex]:
    return edge_indices


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_workers", [
    0,
])
@pytest.mark.parametrize("pin_memory", [False, True])
def test_data_loader(dtype, num_workers, pin_memory):
    kwargs = dict(dtype=dtype)
    adj = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]], sort_order="row", **kwargs)
    adj.fill_cache_()
    adj = adj.to_dict()

    loader = paddle.io.DataLoader(
        dataset=[adj] * 4,
        batch_size=2,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        drop_last=True,
    )
    assert len(loader) == 2
    for batch in loader:
        assert isinstance(batch, list)
        assert len(batch) == 2
        for adj in batch:
            adj = EdgeIndex.from_dict(adj)
            assert isinstance(adj, EdgeIndex)
            assert adj.dtype == dtype


# @pytest.mark.skip(reason="Not support")
# def test_torch_script():
#     class Model(paddle.nn.Layer):
#         def forward(self,
#                     x: paddle.Tensor,
#                     edge_index: EdgeIndex
#             ) -> paddle.Tensor:
#             row, col = edge_index[0], edge_index[1]
#             x_j = x[row]
#             out = scatter(x_j, col, dim_size=edge_index.num_cols)
#             return out

#     x = paddle.randn(shape=[3, 8])
#     edge_index = EdgeIndex([[0, 1, 1, 2], [1, 0, 0, 1]], sparse_size=(3, 3))
#     model = Model()
#     expected = model(x, edge_index)
#     assert tuple(expected.shape) == (3, 8)
#     with pytest.raises(RuntimeError, match="attribute or method 'num_cols'"):
#         paddle.jit.to_static(function=model)

# class ScriptableModel(paddle.nn.Layer):
#     def forward(
#         self, x: paddle.Tensor,
#         edge_index: EdgeIndex) -> paddle.Tensor:
#         row, col = edge_index[0], edge_index[1]
#         x_j = x[row]
#             dim_size: Optional[int] = None
#             if not paddle.jit.is_scripting() \
#                 and isinstance(edge_index, EdgeIndex):
#                 dim_size = edge_index.num_cols
#             out = scatter(x_j, col, dim_size=dim_size)
#             return out

#     script_model = paddle.jit.to_static(function=ScriptableModel())
#     out = script_model(x, edge_index)
#     assert tuple(out.shape) == (2, 8)
#     assert paddle.allclose(x=out, y=expected[:2]).item()

# @pytest.mark.skip(reason="Not support")
# @onlyLinux
# @withPackage("torch==2.3")
# def test_compile_basic():
#     class Model(paddle.nn.Layer):
#         def forward(
#                 self,
#                 x: paddle.Tensor,
#                 edge_index: EdgeIndex) -> paddle.Tensor:
#             x_j = x[edge_index[0]]
#             out = scatter(x_j, edge_index[1], dim_size=edge_index.num_cols)
#             return out

#     x = paddle.randn(shape=[3, 8])
#     edge_index = EdgeIndex(
#         [[0, 1, 1, 2], [1, 0, 0, 1]], sparse_size=(3, 3), sort_order="row"
#     ).fill_cache_()
#     model = Model()
#     expected = model(x, edge_index)
#     assert tuple(expected.shape) == (3, 8)
#     explanation = torch._dynamo.explain(model)(x, edge_index)
#     assert explanation.graph_break_count == 0
#     compiled_model = torch.compile(model, fullgraph=True)
#     out = compiled_model(x, edge_index)
#     assert paddle.allclose(x=out, y=expected).item()

# @pytest.mark.skip(reason="Not support")
# @onlyLinux
# @withPackage("torch==2.3")
# @pytest.mark.skip(reason="Does not work currently")
# def test_compile_create_edge_index():
#     class Model(paddle.nn.Layer):
#         def forward(self) -> EdgeIndex:
#             edge_index = EdgeIndex([[0, 1, 1, 2], [1, 0, 2, 1]])
#             return edge_index

#     model = Model()
#     explanation = torch._dynamo.explain(model)()
#     assert explanation.graph_break_count == 0
#     compiled_model = torch.compile(model, fullgraph=True)
#     assert compiled_model() is None

# if __name__ == "__main__":
#     import argparse

#     warnings.filterwarnings("ignore", ".*Sparse CSR tensor support.*")
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--device", type=str, default="cuda")
#     parser.add_argument("--backward", action="store_true")
#     args = parser.parse_args()
#     channels = 128
#     num_nodes = 20000
#     num_edges = 200000
#     x = paddle.randn(shape=[num_nodes, channels])
#     edge_index = EdgeIndex(
#         paddle.randint(low=0, high=num_nodes, shape=(2, num_edges)),
#         sparse_size=(num_nodes, num_nodes),
#     ).sort_by("row")[0]
#     edge_index.fill_cache_()
#     adj1 = edge_index.to_sparse_csr()
# adj2 = SparseTensor(
#     row=edge_index[0],
#     col=edge_index[1],
#     sparse_sizes=(num_nodes, num_nodes)
# )

#     def edge_index_mm(edge_index, x, reduce):
#         return edge_index.matmul(x, reduce=reduce)

#     def torch_sparse_mm(adj, x):
#         return adj @ x

#     def sparse_tensor_mm(adj, x, reduce):
#         return adj.matmul(x, reduce=reduce)

#     def scatter_mm(edge_index, x, reduce):
#         return _scatter_spmm(edge_index, x, reduce=reduce)

#     funcs = [edge_index_mm, torch_sparse_mm, sparse_tensor_mm, scatter_mm]
#     func_names = ["edge_index", "torch.sparse", "SparseTensor", "scatter"]
#     for reduce in ["sum", "mean", "amin", "amax"]:
#         func_args = [
#             (edge_index, x, reduce),
#             (adj1, x),
#             (adj2, x, reduce),
#             (edge_index, x, reduce),
#         ]
#         print(f"reduce='{reduce}':")
#         benchmark(
#             funcs=funcs,
#             func_names=func_names,
#             args=func_args,
#             num_steps=100 if args.device == "cpu" else 1000,
#             num_warmups=50 if args.device == "cpu" else 500,
#             backward=args.backward,
#         )
