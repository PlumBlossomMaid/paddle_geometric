import functools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
)
import warnings
import paddle
import paddle_geometric.typing as pyg_typing
from paddle import Tensor


HANDLED_FUNCTIONS: Dict[Callable, Callable] = {}

def __array_function__(self, func, types, args, kwargs):
    if func not in HANDLED_FUNCTIONS:
        return NotImplemented
    if not all(issubclass(t, Tensor) for t in types):
        return NotImplemented
    return HANDLED_FUNCTIONS[func](*args, **kwargs)

Tensor.__array_function__ = __array_function__



def ptr2index(ptr: paddle.Tensor, output_size: Optional[int] = None) -> paddle.Tensor:
    index = paddle.arange(dtype=ptr.dtype, end=ptr.size - 1)
    return index.repeat_interleave(repeats=ptr.diff())

def index2ptr(index: Tensor, size: Optional[int] = None) -> Tensor:

    raise NotImplementedError
    # if size is None:
    #     size = int(index.max()) + 1 if index.shape[0] > 0 else 0

    # return paddle.incubate.sparse.convert_indices_from_coo_to_csr(
    #     index, size, out_int32=index.dtype != paddle.int64)



def index2ptr(index: Tensor, size: Optional[int] = None) -> Tensor:
    """
    Convert a 1D index tensor (row indices in COO) to CSR indptr.
    Args:
        index (Tensor): 1D integer tensor of indices.
        size (Optional[int]): number of rows (dim_size). If None, inferred as max(index)+1 or 0.
    Returns:
        Tensor: indptr of shape [size + 1], dtype int32 if index.dtype != int64 else int64.
    """
    # validations (reuse the helpers you have)
    assert_valid_dtype(index)
    assert_one_dimensional(index)
    assert_contiguous(index)

    # infer size if needed
    if size is None:
        size = int(index.max()) + 1 if index.numel() > 0 else 0

    out_dtype = paddle.int64 if index.dtype == paddle.int64 else paddle.int32

    # special cases
    if size == 0:
        # empty dimension -> single zero entry
        return paddle.zeros(shape=(1,), dtype=out_dtype)
    if index.numel() == 0:
        # no entries -> zeros of length size+1
        return paddle.zeros(shape=(size + 1,), dtype=out_dtype)

    # ensure no negative indices
    if (index < 0).any():
        raise ValueError("'index' contains negative values")

    # bincount to get counts per row; ensure we use int64 for bincount input
    counts = paddle.bincount(index.astype('int64'), minlength=size)
    counts = counts.astype(out_dtype)
    # prefix-sum: indptr[0] = 0, indptr[i+1] = sum(counts[:i+1])
    indptr = paddle.concat([paddle.zeros((1,), dtype=out_dtype),
                            counts.cumsum(axis=0, dtype=out_dtype)], axis=0)

    return indptr

class CatMetadata(NamedTuple):
    nnz: List[int]
    dim_size: List[Optional[int]]
    is_sorted: List[bool]


def implements(paddle_function: Callable) -> Callable:
    r"""Registers a PaddlePaddle function override."""
    @functools.wraps(paddle_function)
    def decorator(my_function: Callable) -> Callable:
        HANDLED_FUNCTIONS[paddle_function] = my_function
        return my_function

    return decorator


def assert_valid_dtype(tensor: Tensor) -> None:
    if tensor.dtype not in pyg_typing.INDEX_DTYPES:
        raise ValueError(f"'Index' holds an unsupported data type "
                         f"(got '{tensor.dtype}', but expected one of "
                         f"{pyg_typing.INDEX_DTYPES})")


def assert_one_dimensional(tensor: Tensor) -> None:
    if tensor.dim() != 1:
        raise ValueError(f"'Index' needs to be one-dimensional "
                         f"(got {len(tensor.shape)} dimensions)")


def assert_contiguous(tensor: Tensor) -> None:
    if not tensor.is_contiguous():
        raise ValueError("'Index' needs to be contiguous. Please call "
                         "`index.contiguous()` before proceeding.")


def assert_sorted(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self: 'Index', *args: Any, **kwargs: Any) -> Any:
        if not self.is_sorted:
            cls_name = self.__class__.__name__
            raise ValueError(
                f"Cannot call '{func.__name__}' since '{cls_name}' is not "
                f"sorted. Please call `{cls_name}.sort()` first.")
        return func(self, *args, **kwargs)

    return wrapper

# HANDLED_FUNCTIONS = {}

class Index(Tensor):
    r"""A one-dimensional `index` tensor with additional (meta)data attached.

    :class:`Index` is a subclass of :class:`paddle.Tensor` that holds
    indices of shape `[num_indices]`.

    It includes:
    - `dim_size`: The size of the underlying sparse vector size.
    - `is_sorted`: Whether indices are sorted in ascending order.
    """
    _data: Tensor
    _dim_size: Optional[int] = None
    _is_sorted: bool = False
    _indptr: Optional[Tensor] = None
    _cat_metadata: Optional[CatMetadata] = None

    def __init__(self, 
        data: Any,
        *args: Any,
        dim_size: Optional[int] = None,
        is_sorted: bool = False,
        **kwargs: Any
        ):
        if 'device' in kwargs:
            kwargs['place'] = kwargs['device']
            del kwargs['device']
        if not isinstance(data, Tensor):
            data = paddle.to_tensor(data, *args, **kwargs)
        elif len(args) > 0:
            raise TypeError(
                f"new() received an invalid combination of arguments - got (Tensor, {', '.join(str(type(arg)) for arg in args)})"
            )
        elif len(kwargs) > 0:
            raise TypeError(
                f"new() received invalid keyword arguments - got {set(kwargs.keys())})"
            )
        assert isinstance(data, paddle.Tensor)

        indptr: Optional[paddle.Tensor] = None
        if isinstance(data, Index):
            indptr = data._indptr
            dim_size = dim_size or data.dim_size
            is_sorted = is_sorted or data.is_sorted

        assert_valid_dtype(data)
        assert_one_dimensional(data)
        assert_contiguous(data)

        self.data = data.data
        # self._data = data  # Tensor subclassing is handled by wrapping logic

        # Attach metadata:
        self._dim_size = dim_size
        self._is_sorted = is_sorted
        self._indptr = indptr
    @property
    def _data(self) -> Tensor:
        return self.data

    # Validation ##############################################################
    def validate(self) -> 'Index':
        r"""Validates the `Index` representation."""
        assert_valid_dtype(self._data)
        assert_one_dimensional(self._data)
        assert_contiguous(self._data)

        if self.numel() > 0 and self._data.min() < 0:
            raise ValueError(f"'{self.__class__.__name__}' contains negative "
                             f"indices (got {int(self.min())})")
        if (self.numel() > 0 and self.dim_size is not None
                and self._data.max() >= self.dim_size):
            raise ValueError(f"'{self.__class__.__name__}' contains larger "
                             f"indices than its registered size "
                             f"(got {int(self._data.max())}, but expected "
                             f"values smaller than {self.dim_size})")

        if self.is_sorted and (self._data.diff() < 0).any():
            raise ValueError(f"'{self.__class__.__name__}' is not sorted")

        return self

    # Properties ##############################################################

    @property
    def dim_size(self) -> Optional[int]:
        return self._dim_size

    @property
    def is_sorted(self) -> bool:
        return self._is_sorted
    @property
    def dtype(self) -> paddle.dtype:
        return self._data.dtype

    # Cache Interface #########################################################

    def get_dim_size(self) -> int:
        if self._dim_size is None:
            self._dim_size = int(self._data.max()) + 1 if self.numel() > 0 else 0
        assert isinstance(self._dim_size, int)
        return self._dim_size

    def dim_resize_(self, dim_size: Optional[int]) -> 'Index':
        r"""Assigns or re-assigns the size of the underlying sparse vector."""
        if self.is_sorted and self._indptr is not None:
            if dim_size is None:
                self._indptr = None

            elif self._indptr.size - 1 >= dim_size:
                self._indptr = self._indptr[:dim_size + 1]

            else:
                fill_value = paddle.full(
                    shape=(dim_size - self._indptr.size + 1,),
                    fill_value=self._indptr[-1],
                    dtype=self._indptr.dtype,
                )
                self._indptr = paddle.concat([self._indptr, fill_value], axis=0)

        self._dim_size = dim_size

        return self

    @assert_sorted
    def get_indptr(self) -> paddle.Tensor:
        """Returns the compressed index representation in case :class:`Index`
        is sorted.
        """
        if self._indptr is None:
            self._indptr = index2ptr(self._data, self.get_dim_size())
        assert isinstance(self._indptr, paddle.Tensor)
        return self._indptr

    def fill_cache_(self) -> "Index":
        """Fills the cache with (meta)data information."""
        self.get_dim_size()
        if self.is_sorted:
            self.get_indptr()
        return self

    # Methods #################################################################

    def to_dict(self):
        return {
            'data': self._data,
            'dim_size': self.dim_size,
            'is_sorted': self.is_sorted,
            '_indptr': self._indptr
        }
    @classmethod
    def from_dict(cls, data_dict) -> 'Index':
        r"""Creates a new :class:`Index` instance from a dictionary."""
        out = cls(data_dict['data'],
                  dim_size=data_dict['dim_size'],
                  is_sorted=data_dict['is_sorted']
        )
        out._indptr = data_dict.get('_indptr', None)
        return out

    def share_memory_(self) -> 'Index':
        """"""  # noqa: D419
        warnings.warn("share_memory_ is not supported for Index, ignore it")
        return self

    def is_shared(self) -> bool:
        """"""  # noqa: D419
        # return self._data.is_shared()
        warnings.warn("is_shared is not supported for Index, ignore it")
        return False

    def as_tensor(self) -> Tensor:
        return self._data
    
    @property
    def device(self,):
        place = self.place
        if  place.is_cpu_place():
            device = 'cpu'
        elif place.is_gpu_place():
            device_id = place.gpu_device_id()
            device = 'gpu:' + str(device_id)
        elif place.is_xpu_place():
            raise ValueError("xpu is not supported")
        else:
            raise ValueError(f"The device specification {place} is invalid")

        return device

    def __repr__(self) -> str:
        prefix = f'{self.__class__.__name__}('
        tensor_str = paddle.tensor.to_string._format_tensor(self._data, '')

        suffixes = []
        if self.dim_size is not None:
            suffixes.append(f'dim_size={self.dim_size}')
        if self.device != paddle.device.get_device():
            suffixes.append(f"device='{self.device}'")
        if self.dtype != paddle.int64:
            suffixes.append(f'dtype={self.dtype}')
        if self.is_sorted:
            suffixes.append('is_sorted=True')
        return f"{prefix}{tensor_str}, {', '.join(suffixes)})"
    def __str__(self) -> str:
        return self.__repr__()
    
    def _shallow_copy(self) -> 'Index':
        out = Index(self._data)
        out._dim_size = self._dim_size
        out._is_sorted = self._is_sorted
        out._indptr = self._indptr
        out._cat_metadata = self._cat_metadata
        return out

    def _clear_metadata(self) -> 'Index':
        self._dim_size = None
        self._is_sorted = False
        self._indptr = None
        self._cat_metadata = None
        return self

    def clone(self):
        index = Index(self._data.clone(), dim_size=self.dim_size, is_sorted=self.is_sorted)
        if self._indptr is not None:
            index._indptr = self._indptr.clone()
        return index
    def cuda(self,):
        if self.place.is_gpu_place():
            return self
        index = self.clone()
        index.data = index.data.cuda()
        if self._indptr is not None:
            index._indptr =index._indptr.cuda()
        return index
    def cpu(self,):
        if self.place.is_cpu_place():
            return self
        index = self.clone()
        index.data = index.data.cpu()
        if self._indptr is not None:
            index._indptr =index._indptr.cpu()
        return index

    def flip(self, axis):
        data = self._data
        data = data.flip(axis=axis)
        out = Index(data)
        out._dim_size = self.dim_size

        return out

    def index_select(self, axis, index):

        out = self.data.index_select(
            axis=axis,
            index = index._data if isinstance(index, Index) else index,
        )

        if isinstance(self, Index):
            out = Index(out)
            out._dim_size = self.dim_size
        return out

    def to(self, *args, **kwargs):
        data = self.data.to(*args, **kwargs)
        if data.dtype not in pyg_typing.INDEX_DTYPES:
            return data
        
        if self.data.data_ptr() != data.data_ptr():
            out = Index(data)
        else:  # In-place:
            self.data = data
            out = self
        
        # Copy metadata:
        out._dim_size = self._dim_size
        out._is_sorted = self._is_sorted
        out._cat_metadata = self._cat_metadata

        # Convert cache:
        if self._indptr is not None:
            out._indptr = self._indptr.to(*args, **kwargs)

        return out
    
    def long(self):
        return self.to(dtype=paddle.int64)
    def int(self):
        return self.to(dtype=paddle.int32)
    
    @paddle.utils.decorator_utils.param_one_alias(['axis', 'dim'])
    def sort(self, axis=-1, descending=False, stable=False):
        if self.is_sorted and not descending:
            return self, paddle.arange(self.data.numel(), device=self.data.place)
        
        data = paddle.sort(x=self._data, axis=axis, descending=descending, stable=stable)
        perm = paddle.argsort(x=self._data, axis=axis, descending=descending, stable=stable)

        out = Index(data)
        out._dim_size = self._dim_size

        if not descending:
            out._is_sorted = True
        return out, perm
    
    def narrow(self, dim: int, start, length):

        end = start + length

        if start <= 0 and end > self.data.shape[dim]:
            return self._shallow_copy()
        
        data = paddle.slice(self.data, axes=[dim], starts=[start], ends=[end])
        out = Index(data)
        out._dim_size = self._dim_size
        out._is_sorted = self._is_sorted
        return out
    
    def __getitem__(self, indices):
        if isinstance(indices, slice):
            data = self.data[indices]
            if indices.step is not None and indices.step != 1:
                data = data.contiguous()

            out = Index(data)
            out._dim_size = self.dim_size
            if indices.step is None or indices.step >= 0:
                out._is_sorted = self.is_sorted
            
            return out
        elif indices is Ellipsis:
            return self._shallow_copy()
        elif isinstance(indices, paddle.Tensor):
            data = self.data[indices]
            if data.dim() != 1:
                return data
            out = Index(data)
            if indices.dtype in (paddle.bool, paddle.uint8):  # 1. `index[mask]`.
                out._dim_size = self.dim_size
                out._is_sorted = self.is_sorted

            else:  # 2. `index[index]`.
                out._dim_size = self.dim_size

            return out
        elif isinstance(indices, tuple) and Ellipsis in indices:
            data = self.data[indices]
            if data.dim() != 1:
                return data
            slice_indice = None
            for indice in indices:
                if indice is Ellipsis:
                    continue
                slice_indice = indice
                break
            assert slice_indice is not None

            if slice_indice.step is not None and slice_indice.step != 1:
                data = data.contiguous()

            out = Index(data)
            out._dim_size = self.dim_size
            if slice_indice.step is None or slice_indice.step >= 0:
                out._is_sorted = self.is_sorted
            
            return out
        elif isinstance(indices, tuple) and None in indices:
            data = self.data[indices]
            if data.dim() != 1:
                return data
            slice_indice = None
            for indice in indices:
                if indice is None:
                    continue
                slice_indice = indice
                break
            assert slice_indice is not None

            if slice_indice.step is not None and slice_indice.step != 1:
                data = data.contiguous()

            out = Index(data)
            out._dim_size = self.dim_size
            if slice_indice.step is None or slice_indice.step >= 0:
                out._is_sorted = self.is_sorted
            
            return out
        elif isinstance(indices, int):
            return self._data[indices]
        else:
            raise TypeError(f"Invalid indexing method {type(indices)}")

    def __add__(self, other):
        return paddle.add(self, other)

    def __radd__(self, other):
        return paddle.add(other, self)

    def __sub__(self, other): 
        return paddle.subtract(self, other)
    def __rsub__(self, other): 
        return paddle.subtract(other, self)


    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        
        if func in [paddle.concat]:
            if not all(issubclass(t, Tensor) for t in types):
                return NotImplemented

        if not any(issubclass(t, Tensor) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def get_overloaded_types_and_args(relevant_args):
    """Returns a list of arguments on which to call __array_function__.
    
    __array_function__ implementations should be called in order on the return
    values from this function.
    """
    # Runtime is O(num_arguments * num_unique_types)
    overloaded_types = []
    overloaded_args = []
    for arg in relevant_args:
        arg_type = type(arg)
        if arg_type not in overloaded_types:
            try:
                array_function = arg_type.__array_function__
            except AttributeError:
                continue

            overloaded_types.append(arg_type)

            if array_function is not Tensor.__array_function__:
                index = len(overloaded_args)
                for i, old_arg in enumerate(overloaded_args):
                    if issubclass(arg_type, type(old_arg)):
                        index = i
                        break
                overloaded_args.insert(index, arg)

    return overloaded_types, overloaded_args


def full_name(obj):
    return f'{obj.__module__}.{obj.__qualname__}'
  

def attempt_augmented_error_message(error, append_message):
    """Attempt to recreate an error with an appended message."""
    try:
        return type(error)(error.args[0] + append_message, *error.args[1:])
    except Exception:
        return error
  

def try_array_function_override(func, relevant_arguments, args, kwargs):
    # TODO: consider simplifying the interface, to only require either `types`
    # (by calling __array_function__ a classmethod) or `overloaded_args` (by
    # dropping `types` from the signature of __array_function__)
    types, overloaded_args = get_overloaded_types_and_args(relevant_arguments)
    if not overloaded_args:
        return False, None

    for overloaded_arg in overloaded_args:
        # Note that we're only calling __array_function__ on the *first*
        # occurence of each argument type. This is necessary for reasonable
        # performance with a possibly long list of overloaded arguments, for
        # which each __array_function__ implementation might reasonably need to
        # check all argument types.
        try:
            result = overloaded_arg.__array_function__(
                func, types, args, kwargs)
        except Exception as error:
            # Ensure the type of the overloaded argument ends up in the
            # traceback
            message = (" [while calling {!r} implementation of {!r}]"
                       .format(full_name(type(overloaded_arg)),
                               full_name(func)))
            new_error = attempt_augmented_error_message(error, message)
            # Would probably need to use six to do this sanely on Python 2:
            # https://stackoverflow.com/questions/9157210/
            raise new_error.with_traceback(error.__traceback__) from None

        if result is not NotImplemented:
            return True, result

    raise TypeError('no implementation found for {} on types that implement '
                    '__array_function__: {}'
                    .format(func, list(map(type, overloaded_args))))


def array_function_dispatch(dispatcher):
    """Wrap a function for dispatch with the __array_function__ protocol."""
    def decorator(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            relevant_arguments = dispatcher(*args, **kwargs)
            success, value = try_array_function_override(
                new_func, relevant_arguments, args, kwargs)
            if success:
                return value
            return func(*args, **kwargs)
        return new_func
    return decorator



def _concatenate_dispatcher(x, axis=None, name=None):
    for array in x:
        yield array
setattr(paddle, 'concat', array_function_dispatch(_concatenate_dispatcher)(paddle.concat))

def _binary_dispatcher(x, y, **kwargs):
    return (x, y)
setattr(paddle, 'add', array_function_dispatch(_binary_dispatcher)(paddle.add))


setattr(paddle, 'subtract', array_function_dispatch(_binary_dispatcher)(paddle.subtract))

# def _slice_dispatcher(input, axes, starts, ends):
#     return (input, )
# setattr(paddle, 'slice', array_function_dispatch(_slice_dispatcher)(paddle.slice))



def implements(numpy_function):
    """Register an __array_function__ implementation for MyArray objects."""
    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func
    return decorator

@implements(paddle.concat)
def concat(x, axis=0, name=None):

    data_list = [x_i.data for x_i in x]
    concat_paddle = paddle.concat(data_list, axis=axis)
    if any([not isinstance(tensor, Index) for tensor in x]):
        return concat_paddle

    out =  Index(concat_paddle)

    nnz_list = [t.numel() for t in x]
    dim_size_list = [t.dim_size for t in x]  # type: ignore
    is_sorted_list = [t.is_sorted for t in x]  # type: ignore

    # Post-process `dim_size`:
    total_dim_size: Optional[int] = 0
    for dim_size in dim_size_list:
        if dim_size is None:
            total_dim_size = None
            break
        assert isinstance(total_dim_size, int)
        total_dim_size = max(dim_size, total_dim_size)

    out._dim_size = total_dim_size

    out._cat_metadata = CatMetadata(
        nnz=nnz_list,
        dim_size=dim_size_list,
        is_sorted=is_sorted_list,
    )
    return out


@implements(paddle.add)
def add(x, y, **kwargs):
    # 提取底层 Tensor
    x_data = x.data if isinstance(x, Index) else x
    y_data = y.data if isinstance(y, Index) else y

    alpha = kwargs.get('alpha', 1)
    y_data = y_data * alpha

    # 调用 Paddle 的原始加法
    out_data = x_data + y_data
    if out_data.dtype not in pyg_typing.INDEX_DTYPES:
        return out_data
    
    if out_data.dim() != 1:
        return out_data
    
    out = Index(out_data)
    if isinstance(x, Tensor) and x.numel() <= 1:
        x = int(x)

    if isinstance(y, Tensor) and y.numel() <= 1:
        y = int(y)

    if isinstance(y, int):
        assert isinstance(x, Index)
        if x.dim_size is not None:
            out._dim_size = x.dim_size + alpha * y
        out._is_sorted = x.is_sorted
    elif isinstance(x, int):
        assert isinstance(y, Index)
        if y.dim_size is not None:
            out._dim_size = x + alpha * y.dim_size
        out._is_sorted = y.is_sorted

    elif isinstance(x, Index) and isinstance(y, Index):
        if x.dim_size is not None and y.dim_size is not None:
            out._dim_size = x.dim_size + alpha * y.dim_size
    return out




@implements(paddle.subtract)
def subtract(x, y, **kwargs):
    # 提取底层 Tensor
    x_data = x.data if isinstance(x, Index) else x
    y_data = y.data if isinstance(y, Index) else y

    alpha = kwargs.get('alpha', 1)
    y_data = y_data * alpha

    # 调用 Paddle 的原始加法
    out_data = x_data - y_data
    if out_data.dtype not in pyg_typing.INDEX_DTYPES:
        return out_data
    
    if out_data.dim() != 1:
        return out_data
    
    out = Index(out_data)
    if not isinstance(x, Tensor):
        return out

    if isinstance(y, Tensor) and y.numel() <= 1:
        y = int(y)

    if isinstance(y, int):
        assert isinstance(x, Index)
        if x.dim_size is not None:
            out._dim_size = x.dim_size - alpha * y
        out._is_sorted = x.is_sorted
    return out

    
# @implements(paddle.slice)
# def slice(input, axes, starts, ends):

#     data = input.data
#     data = paddle.slice(data, axes, starts, ends)
#     data = data.contiguous()

#     out = Index(data)
#     out._dim_size = input.dim_size
#     # NOTE We could potentially maintain the `indptr` attribute here,
#     # but it is not really clear if this is worth it. The most important
#     # information `is_sorted` needs to be maintained though:
#     # if step >= 0:
#     out._is_sorted = input.is_sorted

#     return out
