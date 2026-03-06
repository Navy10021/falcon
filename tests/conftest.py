"""Shared pytest configuration and fixtures for the FALCON test suite."""
import sys
import os
import pickle
import contextlib
import types

import numpy as np

# Ensure the project root is importable from every test file.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ══════════════════════════════════════════════════════════════════════════════
# Mock `networkx` if not installed
# ══════════════════════════════════════════════════════════════════════════════

def _install_networkx_mock():
    class _MockMultiDiGraph:
        def __init__(self):
            self._nodes = {}
            self._edges = []
            self._adj = {}

        def add_node(self, node_id, **attrs):
            self._nodes[node_id] = attrs
            if node_id not in self._adj:
                self._adj[node_id] = {}

        def add_edge(self, src, dst, key=None, **attrs):
            self._edges.append((src, dst, key, attrs))
            if src not in self._adj:
                self._adj[src] = {}
            if dst not in self._adj[src]:
                self._adj[src][dst] = {}
            self._adj[src][dst][key] = attrs

        @property
        def nodes(self):
            return self._nodes

        @property
        def edges(self):
            return self._edges

        def __len__(self):
            return len(self._nodes)

        def successors(self, node):
            return list(self._adj.get(node, {}).keys())

        def predecessors(self, node):
            return [s for s, targets in self._adj.items() if node in targets]

        def out_edges(self, node, data=False, keys=False):
            result = []
            for (s, d, k, attrs) in self._edges:
                if s == node:
                    if data and keys:
                        result.append((s, d, k, attrs))
                    elif data:
                        result.append((s, d, attrs))
                    elif keys:
                        result.append((s, d, k))
                    else:
                        result.append((s, d))
            return result

        def in_edges(self, node, data=False):
            return [(s, d) for (s, d, k, attrs) in self._edges if d == node]

        def number_of_nodes(self):
            return len(self._nodes)

        def number_of_edges(self):
            return len(self._edges)

    class _MockNX:
        MultiDiGraph = _MockMultiDiGraph
        DiGraph = _MockMultiDiGraph
        Graph = _MockMultiDiGraph

        @staticmethod
        def is_directed(g):
            return True

    mock = _MockNX()
    mock.__spec__ = None
    mock.__name__ = "networkx"
    mock.__package__ = "networkx"
    mock.__path__ = []
    sys.modules["networkx"] = mock


try:
    import importlib as _il
    _il.import_module("networkx")
except ModuleNotFoundError:
    _install_networkx_mock()


# ══════════════════════════════════════════════════════════════════════════════
# Mock `torch` if not installed
# ══════════════════════════════════════════════════════════════════════════════

def _install_torch_mock():

    # ── Type constants ───────────────────────────────────────────────────────
    class _DType:
        def __init__(self, name):
            self.name = name
        def __repr__(self):
            return f"torch.{self.name}"

    _float32 = _DType("float32")
    _float16 = _DType("float16")
    _long    = _DType("long")
    _int64   = _DType("int64")
    _bool    = _DType("bool")

    # ── MockTensor ───────────────────────────────────────────────────────────
    class MockTensor:
        def __init__(self, data):
            if isinstance(data, MockTensor):
                self._data = data._data.copy()
            elif isinstance(data, np.ndarray):
                self._data = data.astype(np.float32)
            elif isinstance(data, (list, tuple)):
                self._data = np.array(data, dtype=np.float32)
            elif isinstance(data, (int, float, np.integer, np.floating)):
                self._data = np.array(float(data), dtype=np.float32)
            else:
                try:
                    self._data = np.array(data, dtype=np.float32)
                except Exception:
                    self._data = np.zeros(1, dtype=np.float32)

        @property
        def shape(self):
            return self._data.shape

        @property
        def ndim(self):
            return self._data.ndim

        @property
        def dtype(self):
            return _float32

        def size(self, dim=None):
            if dim is None:
                return self._data.shape
            return self._data.shape[dim]

        def item(self):
            return float(self._data.flat[0])

        def numpy(self):
            return self._data.astype(np.float32)

        def detach(self):
            return MockTensor(self._data.copy())

        def cpu(self):
            return self

        def to(self, *args, **kwargs):
            return self

        def float(self):
            return MockTensor(self._data.astype(np.float32))

        def long(self):
            return MockTensor(self._data.astype(np.float32))

        def squeeze(self, dim=None):
            return MockTensor(self._data.squeeze())

        def unsqueeze(self, dim):
            return MockTensor(np.expand_dims(self._data, axis=dim))

        def view(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = shape[0]
            return MockTensor(self._data.reshape(shape))

        def reshape(self, *shape):
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = shape[0]
            return MockTensor(self._data.reshape(shape))

        def __getitem__(self, idx):
            if isinstance(idx, MockTensor):
                idx = idx._data.astype(np.int64)
            result = self._data[idx]
            if isinstance(result, np.ndarray):
                return MockTensor(result)
            return MockTensor(float(result))

        def __setitem__(self, idx, value):
            if isinstance(value, MockTensor):
                self._data[idx] = value._data
            else:
                self._data[idx] = value

        def __len__(self):
            return len(self._data)

        def __add__(self, other):
            o = other._data if isinstance(other, MockTensor) else other
            return MockTensor(self._data + o)

        def __radd__(self, other):
            return MockTensor(other + self._data)

        def __sub__(self, other):
            o = other._data if isinstance(other, MockTensor) else other
            return MockTensor(self._data - o)

        def __rsub__(self, other):
            return MockTensor(other - self._data)

        def __mul__(self, other):
            o = other._data if isinstance(other, MockTensor) else other
            return MockTensor(self._data * o)

        def __rmul__(self, other):
            return MockTensor(other * self._data)

        def __truediv__(self, other):
            o = other._data if isinstance(other, MockTensor) else other
            return MockTensor(self._data / (np.where(o == 0, 1e-12, o)))

        def __neg__(self):
            return MockTensor(-self._data)

        def __gt__(self, other):
            o = other._data if isinstance(other, MockTensor) else other
            return MockTensor((self._data > o).astype(np.float32))

        def __lt__(self, other):
            o = other._data if isinstance(other, MockTensor) else other
            return MockTensor((self._data < o).astype(np.float32))

        def __ge__(self, other):
            o = other._data if isinstance(other, MockTensor) else other
            return MockTensor((self._data >= o).astype(np.float32))

        def __le__(self, other):
            o = other._data if isinstance(other, MockTensor) else other
            return MockTensor((self._data <= o).astype(np.float32))

        def __repr__(self):
            return f"MockTensor({self._data})"

        def backward(self, *args, **kwargs):
            pass

        def mean(self, dim=None, keepdim=False):
            if dim is None:
                return MockTensor(np.mean(self._data))
            return MockTensor(np.mean(self._data, axis=dim, keepdims=keepdim))

        def sum(self, dim=None):
            if dim is None:
                return MockTensor(np.sum(self._data))
            return MockTensor(np.sum(self._data, axis=dim))

        def argmax(self, dim=None):
            if dim is None:
                return MockTensor(float(np.argmax(self._data)))
            return MockTensor(np.argmax(self._data, axis=dim).astype(np.float32))

        def max(self, dim=None):
            if dim is None:
                return MockTensor(np.max(self._data))
            vals = np.max(self._data, axis=dim)
            idxs = np.argmax(self._data, axis=dim)
            return MockTensor(vals), MockTensor(idxs.astype(np.float32))

        def min(self, dim=None):
            if dim is None:
                return MockTensor(np.min(self._data))
            vals = np.min(self._data, axis=dim)
            idxs = np.argmin(self._data, axis=dim)
            return MockTensor(vals), MockTensor(idxs.astype(np.float32))

        def clamp(self, min=None, max=None):
            return MockTensor(np.clip(self._data, min, max))

        def exp(self):
            return MockTensor(np.exp(np.clip(self._data, -50, 50)))

        def log(self):
            return MockTensor(np.log(np.clip(self._data, 1e-10, None)))

        def pow(self, exp):
            return MockTensor(self._data ** exp)

        def sqrt(self):
            return MockTensor(np.sqrt(np.clip(self._data, 0, None)))

        def abs(self):
            return MockTensor(np.abs(self._data))

        def clone(self):
            return MockTensor(self._data.copy())

        def requires_grad_(self, requires_grad=True):
            return self

        @property
        def requires_grad(self):
            return False

        @property
        def grad(self):
            return None

        def __float__(self):
            return float(self._data.flat[0])

        def __int__(self):
            return int(self._data.flat[0])

        def __bool__(self):
            return bool(self._data.flat[0])

        def __iter__(self):
            for x in self._data:
                if isinstance(x, np.ndarray):
                    yield MockTensor(x)
                else:
                    yield MockTensor(float(x))

    # ── torch functions ──────────────────────────────────────────────────────

    def _tensor(data, dtype=None, device=None, requires_grad=False):
        return MockTensor(data)

    def _zeros(*args, dtype=None, device=None, requires_grad=False):
        shape = args[0] if (len(args) == 1 and isinstance(args[0], (tuple, list))) else args
        return MockTensor(np.zeros(shape, dtype=np.float32))

    def _ones(*args, dtype=None, device=None, requires_grad=False):
        shape = args[0] if (len(args) == 1 and isinstance(args[0], (tuple, list))) else args
        return MockTensor(np.ones(shape, dtype=np.float32))

    def _zeros_like(t, dtype=None):
        d = t._data if isinstance(t, MockTensor) else np.array(t)
        return MockTensor(np.zeros_like(d))

    def _ones_like(t, dtype=None):
        d = t._data if isinstance(t, MockTensor) else np.array(t)
        return MockTensor(np.ones_like(d))

    def _full(size, fill_value, dtype=None, device=None):
        return MockTensor(np.full(size, fill_value, dtype=np.float32))

    def _arange(*args, dtype=None, device=None):
        return MockTensor(np.arange(*args, dtype=np.float32))

    def _randperm(n, device=None):
        return MockTensor(np.random.permutation(n).astype(np.float32))

    def _cat(tensors, dim=0):
        arrays = [t._data if isinstance(t, MockTensor) else np.array(t, dtype=np.float32) for t in tensors]
        return MockTensor(np.concatenate(arrays, axis=dim))

    def _stack(tensors, dim=0):
        arrays = [t._data if isinstance(t, MockTensor) else np.array(t, dtype=np.float32) for t in tensors]
        return MockTensor(np.stack(arrays, axis=dim))

    def _clamp(x, min=None, max=None):
        d = x._data if isinstance(x, MockTensor) else np.array(x)
        return MockTensor(np.clip(d, min, max))

    def _sum_fn(x, dim=None):
        d = x._data if isinstance(x, MockTensor) else np.array(x)
        return MockTensor(np.sum(d) if dim is None else np.sum(d, axis=dim))

    def _mean_fn(x, dim=None):
        d = x._data if isinstance(x, MockTensor) else np.array(x)
        return MockTensor(np.mean(d) if dim is None else np.mean(d, axis=dim))

    def _max_fn(x, dim=None):
        d = x._data if isinstance(x, MockTensor) else np.array(x)
        if dim is None:
            return MockTensor(np.max(d))
        return MockTensor(np.max(d, axis=dim)), MockTensor(np.argmax(d, axis=dim).astype(np.float32))

    def _min_fn(x, other_or_dim=None, dim=None):
        d = x._data if isinstance(x, MockTensor) else np.array(x)
        # torch.min(a, b) → element-wise min when second arg is also a Tensor
        if isinstance(other_or_dim, MockTensor):
            o = other_or_dim._data
            return MockTensor(np.minimum(d, o))
        # torch.min(a, dim=k) or torch.min(a)
        actual_dim = other_or_dim if other_or_dim is not None else dim
        if actual_dim is None:
            return MockTensor(np.min(d))
        return MockTensor(np.min(d, axis=actual_dim)), MockTensor(np.argmin(d, axis=actual_dim).astype(np.float32))

    def _exp_fn(x):
        d = x._data if isinstance(x, MockTensor) else np.array(x)
        return MockTensor(np.exp(np.clip(d, -50, 50)))

    def _log_fn(x):
        d = x._data if isinstance(x, MockTensor) else np.array(x)
        return MockTensor(np.log(np.clip(d, 1e-10, None)))

    def _sqrt_fn(x):
        d = x._data if isinstance(x, MockTensor) else np.array(x)
        return MockTensor(np.sqrt(np.clip(d, 0, None)))

    def _abs_fn(x):
        d = x._data if isinstance(x, MockTensor) else np.array(x)
        return MockTensor(np.abs(d))

    def _nan_to_num(x, nan=0.0, posinf=None, neginf=None):
        d = x._data.copy() if isinstance(x, MockTensor) else np.array(x, dtype=np.float32)
        d = np.where(np.isnan(d), nan, d)
        if posinf is not None:
            d = np.where(np.isposinf(d), posinf, d)
        if neginf is not None:
            d = np.where(np.isneginf(d), neginf, d)
        return MockTensor(d)

    def _save(obj, path, **kwargs):
        os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
        with open(str(path), "wb") as f:
            pickle.dump(obj, f)

    def _load(path, map_location=None, **kwargs):
        with open(str(path), "rb") as f:
            return pickle.load(f)

    @contextlib.contextmanager
    def _no_grad():
        yield

    def _device(dev_str):
        return str(dev_str)

    class _FloatTensorCallable:
        def __call__(self, data):
            return MockTensor(data)
        def __instancecheck__(self, obj):
            return isinstance(obj, MockTensor)

    # ── nn.functional ────────────────────────────────────────────────────────

    def _softmax(x, dim=-1):
        d = x._data if isinstance(x, MockTensor) else np.array(x, dtype=np.float32)
        d = d - d.max(axis=dim, keepdims=True)
        e = np.exp(d)
        return MockTensor(e / (e.sum(axis=dim, keepdims=True) + 1e-10))

    def _log_softmax(x, dim=-1):
        sm = _softmax(x, dim=dim)
        return MockTensor(np.log(sm._data + 1e-10))

    def _relu(x):
        d = x._data if isinstance(x, MockTensor) else np.array(x)
        return MockTensor(np.maximum(d, 0))

    def _gelu(x):
        d = x._data if isinstance(x, MockTensor) else np.array(x)
        return MockTensor(d * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (d + 0.044715 * d ** 3))))

    def _tanh_fn(x):
        d = x._data if isinstance(x, MockTensor) else np.array(x)
        return MockTensor(np.tanh(d))

    def _cross_entropy(input_, target, reduction="mean", ignore_index=-100, weight=None):
        d = input_._data if isinstance(input_, MockTensor) else np.array(input_)
        t = (target._data.astype(int) if isinstance(target, MockTensor) else np.array(target, dtype=int))
        ls = _log_softmax(MockTensor(d), dim=-1)._data
        if ls.ndim == 1:
            loss = -ls[int(np.clip(t.flat[0], 0, ls.shape[-1] - 1))]
        else:
            loss = -np.mean([ls[i, int(np.clip(t[i], 0, ls.shape[-1] - 1))] for i in range(len(t))])
        return MockTensor(float(loss))

    def _mse_loss(input_, target, reduction="mean"):
        d = input_._data if isinstance(input_, MockTensor) else np.array(input_)
        t = target._data if isinstance(target, MockTensor) else np.array(target)
        diff = (d - t) ** 2
        return MockTensor(np.mean(diff) if reduction == "mean" else diff)

    class _Functional:
        @staticmethod
        def softmax(x, dim=-1):
            return _softmax(x, dim=dim)

        @staticmethod
        def log_softmax(x, dim=-1):
            return _log_softmax(x, dim=dim)

        @staticmethod
        def relu(x):
            return _relu(x)

        @staticmethod
        def gelu(x, approximate="none"):
            return _gelu(x)

        @staticmethod
        def tanh(x):
            return _tanh_fn(x)

        @staticmethod
        def cross_entropy(input_, target, weight=None, reduction="mean", ignore_index=-100):
            return _cross_entropy(input_, target, reduction, ignore_index)

        @staticmethod
        def mse_loss(input_, target, reduction="mean"):
            return _mse_loss(input_, target, reduction)

        @staticmethod
        def kl_div(input_, target, reduction="batchmean"):
            d = input_._data if isinstance(input_, MockTensor) else np.array(input_)
            t = target._data if isinstance(target, MockTensor) else np.array(target)
            return MockTensor(np.sum(t * (np.log(t + 1e-10) - d)))

        @staticmethod
        def normalize(x, p=2, dim=1, eps=1e-12):
            d = x._data if isinstance(x, MockTensor) else np.array(x)
            norm = np.linalg.norm(d, ord=p, axis=dim, keepdims=True)
            return MockTensor(d / (norm + eps))

        @staticmethod
        def softplus(x, beta=1, threshold=20):
            d = x._data if isinstance(x, MockTensor) else np.array(x)
            return MockTensor(np.log1p(np.exp(d * beta)) / beta)

        @staticmethod
        def sigmoid(x):
            d = x._data if isinstance(x, MockTensor) else np.array(x)
            return MockTensor(1 / (1 + np.exp(-np.clip(d, -50, 50))))

    # ── nn.Module and layers ─────────────────────────────────────────────────

    class Module:
        def __init__(self):
            object.__setattr__(self, "_modules", {})
            object.__setattr__(self, "_parameters", {})
            object.__setattr__(self, "training", True)

        def __setattr__(self, name, value):
            if isinstance(value, Module):
                mods = object.__getattribute__(self, "_modules")
                mods[name] = value
            elif isinstance(value, Parameter):
                params = object.__getattribute__(self, "_parameters")
                params[name] = value
            object.__setattr__(self, name, value)

        def to(self, *args, **kwargs):
            return self

        def eval(self):
            object.__setattr__(self, "training", False)
            return self

        def train(self, mode=True):
            object.__setattr__(self, "training", mode)
            return self

        def parameters(self):
            params = []
            try:
                params.extend(object.__getattribute__(self, "_parameters").values())
            except AttributeError:
                pass
            try:
                for mod in object.__getattribute__(self, "_modules").values():
                    params.extend(mod.parameters())
            except AttributeError:
                pass
            return iter(params)

        def modules(self):
            yield self
            try:
                for mod in object.__getattribute__(self, "_modules").values():
                    yield from mod.modules()
            except AttributeError:
                pass

        def named_modules(self, memo=None, prefix=""):
            yield prefix, self
            try:
                for name, mod in object.__getattribute__(self, "_modules").items():
                    subprefix = f"{prefix}.{name}" if prefix else name
                    yield from mod.named_modules(prefix=subprefix)
            except AttributeError:
                pass

        def state_dict(self, prefix="", destination=None):
            return {}

        def load_state_dict(self, state_dict, strict=True):
            pass

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, *args, **kwargs):
            raise NotImplementedError

    class Parameter(MockTensor):
        def __init__(self, data, requires_grad=True):
            if isinstance(data, np.ndarray):
                super().__init__(data)
            else:
                super().__init__(data)

    class Linear(Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            scale = np.sqrt(2.0 / max(in_features, 1))
            self.weight = Parameter(
                np.random.randn(out_features, in_features).astype(np.float32) * scale
            )
            if bias:
                self.bias = Parameter(np.zeros(out_features, dtype=np.float32))
            else:
                self.bias = None

        def forward(self, x):
            d = x._data if isinstance(x, MockTensor) else np.array(x, dtype=np.float32)
            out = d @ self.weight._data.T
            if self.bias is not None:
                out = out + self.bias._data
            return MockTensor(out)

    class LayerNorm(Module):
        def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = normalized_shape
            self.eps = eps
            if elementwise_affine:
                self.weight = Parameter(np.ones(normalized_shape, dtype=np.float32))
                self.bias = Parameter(np.zeros(normalized_shape, dtype=np.float32))

        def forward(self, x):
            d = x._data if isinstance(x, MockTensor) else np.array(x, dtype=np.float32)
            mean = d.mean(axis=-1, keepdims=True)
            var = ((d - mean) ** 2).mean(axis=-1, keepdims=True)
            normed = (d - mean) / np.sqrt(var + self.eps)
            if hasattr(self, "weight"):
                normed = normed * self.weight._data + self.bias._data
            return MockTensor(normed)

    class Tanh(Module):
        def forward(self, x):
            return _tanh_fn(x)

    class ReLU(Module):
        def __init__(self, inplace=False):
            super().__init__()
        def forward(self, x):
            return _relu(x)

    class GELU(Module):
        def __init__(self, approximate="none"):
            super().__init__()
        def forward(self, x):
            return _gelu(x)

    class Sigmoid(Module):
        def forward(self, x):
            d = x._data if isinstance(x, MockTensor) else np.array(x)
            return MockTensor(1 / (1 + np.exp(-np.clip(d, -50, 50))))

    class Softmax(Module):
        def __init__(self, dim=-1):
            super().__init__()
            self.dim = dim
        def forward(self, x):
            return _softmax(x, dim=self.dim)

    class Dropout(Module):
        def __init__(self, p=0.5, inplace=False):
            super().__init__()
            self.p = p
        def forward(self, x):
            return x

    class BatchNorm1d(Module):
        def __init__(self, num_features, eps=1e-5, momentum=0.1):
            super().__init__()
            self.num_features = num_features
            self.weight = Parameter(np.ones(num_features, dtype=np.float32))
            self.bias = Parameter(np.zeros(num_features, dtype=np.float32))

        def forward(self, x):
            d = x._data if isinstance(x, MockTensor) else np.array(x, dtype=np.float32)
            mean = d.mean(axis=0, keepdims=True)
            std = d.std(axis=0, keepdims=True) + 1e-5
            normed = (d - mean) / std
            return MockTensor(normed * self.weight._data + self.bias._data)

    class Sequential(Module):
        def __init__(self, *layers):
            super().__init__()
            self._layer_list = list(layers)
            mods = object.__getattribute__(self, "_modules")
            for i, layer in enumerate(layers):
                mods[str(i)] = layer

        def __getitem__(self, idx):
            return self._layer_list[idx]

        def __len__(self):
            return len(self._layer_list)

        def forward(self, x):
            for layer in self._layer_list:
                x = layer(x)
            return x

    class MSELoss(Module):
        def __init__(self, reduction="mean"):
            super().__init__()
            self.reduction = reduction
        def forward(self, input_, target):
            return _mse_loss(input_, target, self.reduction)

    class CrossEntropyLoss(Module):
        def __init__(self, reduction="mean", ignore_index=-100):
            super().__init__()
            self.reduction = reduction
            self.ignore_index = ignore_index
        def forward(self, input_, target):
            return _cross_entropy(input_, target, self.reduction, self.ignore_index)

    class _Init:
        @staticmethod
        def orthogonal_(tensor, gain=1.0):
            if not isinstance(tensor, (Parameter, MockTensor)):
                return tensor
            d = tensor._data
            if d.ndim < 2:
                return tensor
            rows = d.shape[0]
            cols = int(np.prod(d.shape[1:]))
            if rows >= cols:
                flat = np.random.randn(rows, cols).astype(np.float32)
                q, _ = np.linalg.qr(flat)   # q: (rows, cols)
                result = q * gain
            else:
                flat = np.random.randn(cols, rows).astype(np.float32)
                q, _ = np.linalg.qr(flat)   # q: (cols, rows)
                result = q.T * gain          # (rows, cols)
            tensor._data[:] = result.reshape(d.shape)
            return tensor

        @staticmethod
        def constant_(tensor, val):
            if isinstance(tensor, (Parameter, MockTensor)):
                tensor._data[:] = val
            return tensor

        @staticmethod
        def xavier_uniform_(tensor, gain=1.0):
            if isinstance(tensor, (Parameter, MockTensor)):
                d = tensor._data
                fan_in = d.shape[-1] if d.ndim > 1 else d.shape[0]
                fan_out = d.shape[0]
                std = gain * np.sqrt(2.0 / (fan_in + fan_out))
                tensor._data[:] = np.random.uniform(
                    -std * np.sqrt(3), std * np.sqrt(3), d.shape
                ).astype(np.float32)
            return tensor

        @staticmethod
        def normal_(tensor, mean=0.0, std=1.0):
            if isinstance(tensor, (Parameter, MockTensor)):
                tensor._data[:] = np.random.normal(mean, std, tensor._data.shape).astype(np.float32)
            return tensor

        @staticmethod
        def zeros_(tensor):
            if isinstance(tensor, (Parameter, MockTensor)):
                tensor._data[:] = 0
            return tensor

    class _NN:
        pass

    _NN.Module = Module
    _NN.Parameter = Parameter
    _NN.Linear = Linear
    _NN.LayerNorm = LayerNorm
    _NN.Tanh = Tanh
    _NN.ReLU = ReLU
    _NN.GELU = GELU
    _NN.Sigmoid = Sigmoid
    _NN.Softmax = Softmax
    _NN.Dropout = Dropout
    _NN.BatchNorm1d = BatchNorm1d
    _NN.Sequential = Sequential
    _NN.MSELoss = MSELoss
    _NN.CrossEntropyLoss = CrossEntropyLoss
    _NN.functional = _Functional()
    _NN.init = _Init()

    # ── distributions ────────────────────────────────────────────────────────

    class Categorical:
        def __init__(self, probs=None, logits=None):
            if logits is not None:
                d = logits._data if isinstance(logits, MockTensor) else np.array(logits, dtype=np.float32)
                d = d - d.max(axis=-1, keepdims=True)
                e = np.exp(d)
                self._probs = e / (e.sum(axis=-1, keepdims=True) + 1e-10)
            elif probs is not None:
                self._probs = probs._data if isinstance(probs, MockTensor) else np.array(probs, dtype=np.float32)
            else:
                self._probs = np.ones(6, dtype=np.float32) / 6
            self._probs = np.clip(self._probs, 1e-10, None)
            self._probs = self._probs / self._probs.sum(axis=-1, keepdims=True)

        def sample(self):
            probs = self._probs.flatten()
            idx = np.random.choice(len(probs), p=probs / probs.sum())
            return MockTensor(float(idx))

        def log_prob(self, action):
            a = int(action._data.flat[0]) if isinstance(action, MockTensor) else int(action)
            probs = self._probs.flatten()
            a = int(np.clip(a, 0, len(probs) - 1))
            return MockTensor(float(np.log(probs[a] + 1e-10)))

        def entropy(self):
            probs = self._probs.flatten()
            return MockTensor(float(-np.sum(probs * np.log(probs + 1e-10))))

    def _kl_divergence(p, q):
        pp = p._probs.flatten()
        qp = q._probs.flatten()
        kl = float(np.sum(pp * np.log((pp + 1e-10) / (qp + 1e-10))))
        return MockTensor(kl)

    class _Distributions:
        pass

    _Distributions.Categorical = Categorical
    _Distributions.kl_divergence = _kl_divergence

    # ── optim ────────────────────────────────────────────────────────────────

    class Adam:
        def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
            self._params = list(params)
            self.lr = lr
            self._step = 0

        def zero_grad(self):
            pass

        def step(self):
            self._step += 1
            for p in self._params:
                if isinstance(p, (Parameter, MockTensor)):
                    noise = np.random.randn(*p._data.shape).astype(np.float32)
                    p._data += noise * self.lr * 0.001

        def state_dict(self):
            return {"lr": self.lr, "step": self._step}

        def load_state_dict(self, sd):
            pass

    class RMSprop:
        def __init__(self, params, lr=1e-2, **kwargs):
            self._params = list(params)
            self.lr = lr
        def zero_grad(self):
            pass
        def step(self):
            pass
        def state_dict(self):
            return {}
        def load_state_dict(self, sd):
            pass

    class _Optim:
        pass

    _Optim.Adam = Adam
    _Optim.RMSprop = RMSprop

    # ── Assemble mock torch as a types.ModuleType ────────────────────────────
    # Using ModuleType avoids Python descriptor protocol binding functions as methods.

    _nn_obj    = _NN()
    _optim_obj = _Optim()
    _dist_obj  = _Distributions()
    # Set functions on instances to avoid Python descriptor binding (self injection)
    _dist_obj.__dict__["kl_divergence"] = _kl_divergence

    mock_torch = types.ModuleType("torch")
    mock_torch.__spec__    = None
    mock_torch.__package__ = "torch"
    mock_torch.__path__    = []
    mock_torch.__version__ = "0.0.0+mock"

    mock_torch.Tensor        = MockTensor
    mock_torch.FloatTensor   = _FloatTensorCallable()
    mock_torch.float32       = _float32
    mock_torch.float16       = _float16
    mock_torch.long          = _long
    mock_torch.int64         = _int64
    mock_torch.bool          = _bool
    mock_torch.nn            = _nn_obj
    mock_torch.optim         = _optim_obj
    mock_torch.distributions = _dist_obj
    mock_torch.tensor        = _tensor
    mock_torch.zeros         = _zeros
    mock_torch.ones          = _ones
    mock_torch.zeros_like    = _zeros_like
    mock_torch.ones_like     = _ones_like
    mock_torch.full          = _full
    mock_torch.arange        = _arange
    mock_torch.randperm      = _randperm
    mock_torch.cat           = _cat
    mock_torch.stack         = _stack
    mock_torch.clamp         = _clamp
    mock_torch.sum           = _sum_fn
    mock_torch.mean          = _mean_fn
    mock_torch.max           = _max_fn
    mock_torch.min           = _min_fn
    mock_torch.exp           = _exp_fn
    mock_torch.log           = _log_fn
    mock_torch.sqrt          = _sqrt_fn
    mock_torch.abs           = _abs_fn
    mock_torch.nan_to_num    = _nan_to_num
    mock_torch.save          = _save
    mock_torch.load          = _load
    mock_torch.no_grad       = _no_grad
    mock_torch.device        = _device

    def _randn(*shape, dtype=None, device=None, requires_grad=False):
        return MockTensor(np.random.randn(*shape).astype(np.float32))

    def _rand(*shape, dtype=None, device=None):
        return MockTensor(np.random.rand(*shape).astype(np.float32))

    def _from_numpy(arr):
        return MockTensor(arr)

    def _is_tensor(obj):
        return isinstance(obj, MockTensor)

    def _sigmoid_fn(x):
        d = x._data if isinstance(x, MockTensor) else np.array(x)
        return MockTensor(1 / (1 + np.exp(-np.clip(d, -50, 50))))

    def _tanh_global(x):
        return _tanh_fn(x)

    def _where_fn(cond, x, y):
        c = (cond._data.astype(bool) if isinstance(cond, MockTensor) else np.array(cond, dtype=bool))
        xd = x._data if isinstance(x, MockTensor) else np.full(c.shape, float(x))
        yd = y._data if isinstance(y, MockTensor) else np.full(c.shape, float(y))
        return MockTensor(np.where(c, xd, yd))

    def _norm_fn(x, p=2, dim=None):
        d = x._data if isinstance(x, MockTensor) else np.array(x)
        if dim is None:
            return MockTensor(np.linalg.norm(d.flatten(), ord=p))
        return MockTensor(np.linalg.norm(d, ord=p, axis=dim))

    mock_torch.randn      = _randn
    mock_torch.rand       = _rand
    mock_torch.from_numpy = _from_numpy
    mock_torch.is_tensor  = _is_tensor
    mock_torch.sigmoid    = _sigmoid_fn
    mock_torch.tanh       = _tanh_global
    mock_torch.where      = _where_fn
    mock_torch.norm       = _norm_fn

    # Register sub-modules so `import torch.nn as nn` etc. works
    def _make_submod(src, name, pkg):
        mod = types.ModuleType(name)
        mod.__spec__    = None
        mod.__package__ = pkg
        mod.__path__    = []
        src_cls = type(src)
        for _k in dir(src_cls):
            if _k.startswith("__"):
                continue
            raw = vars(src_cls).get(_k)
            if isinstance(raw, staticmethod):
                setattr(mod, _k, raw.__func__)
            elif callable(raw) and not isinstance(raw, type):
                # Plain function stored as class attr — use directly (unbound)
                setattr(mod, _k, raw)
            else:
                setattr(mod, _k, getattr(src, _k))
        return mod

    mock_nn_f  = _Functional()
    mock_nn    = _make_submod(_nn_obj,    "torch.nn",            "torch")
    mock_f     = _make_submod(mock_nn_f,  "torch.nn.functional", "torch.nn")
    mock_optim = _make_submod(_optim_obj, "torch.optim",         "torch")
    mock_dist  = _make_submod(_dist_obj,  "torch.distributions", "torch")
    # Explicit raw-function overrides to avoid bound-method wrapping
    mock_dist.kl_divergence = _kl_divergence
    mock_dist.Categorical   = Categorical
    # Attach functional directly on nn sub-module too
    mock_nn.functional = mock_f
    # nn.utils.clip_grad_norm_ — no-op stub
    mock_nn_utils = types.ModuleType("torch.nn.utils")
    mock_nn_utils.__spec__ = None
    mock_nn_utils.clip_grad_norm_ = lambda params, max_norm, norm_type=2.0: None
    mock_nn_utils.clip_grad_value_ = lambda params, clip_value: None
    mock_nn.utils = mock_nn_utils
    sys.modules["torch.nn.utils"] = mock_nn_utils
    # Use ModuleType versions in mock_torch to avoid descriptor binding
    mock_torch.nn            = mock_nn
    mock_torch.optim         = mock_optim
    mock_torch.distributions = mock_dist

    sys.modules["torch"]                  = mock_torch
    sys.modules["torch.nn"]               = mock_nn
    sys.modules["torch.nn.functional"]    = mock_f
    sys.modules["torch.optim"]            = mock_optim
    sys.modules["torch.distributions"]   = mock_dist
    sys.modules["torch.distributions.kl"]= mock_dist

    # Stub-out sub-modules that may be imported but aren't needed
    for _sub in ("torch.cuda", "torch.autograd", "torch.utils", "torch.utils.data",
                 "torch.jit", "torch.onnx", "torch.backends", "torch.backends.cudnn"):
        _mod = type(sys)(_sub)
        _mod.__spec__ = None
        sys.modules.setdefault(_sub, _mod)


try:
    import importlib as _il2
    _il2.import_module("torch")
except ModuleNotFoundError:
    _install_torch_mock()


# ── Mock `tqdm` if not installed ─────────────────────────────────────────────
try:
    import importlib as _il3
    _il3.import_module("tqdm")
except ModuleNotFoundError:
    _tqdm_mod = types.ModuleType("tqdm")
    _tqdm_mod.__spec__ = None

    class _tqdm:
        def __init__(self, iterable=None, desc=None, total=None, **kwargs):
            self._iterable = iterable or []
        def __iter__(self):
            return iter(self._iterable)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            pass
        def set_postfix(self, **kwargs):
            pass
        def close(self):
            pass

    _tqdm_mod.tqdm = _tqdm
    _tqdm_mod.auto = types.ModuleType("tqdm.auto")
    _tqdm_mod.auto.tqdm = _tqdm
    sys.modules["tqdm"] = _tqdm_mod
    sys.modules["tqdm.auto"] = _tqdm_mod.auto
