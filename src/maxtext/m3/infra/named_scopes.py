# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Name scopes for profiles and HLO dumps.

Flax Linen runs every module method under `jax.named_scope(module.name)`. That
is why a legacy MaxText profile reads `decoder/layers_3/mlp/wi_0/dot_general`
where an m3 profile reads `fusion.512`: NNX has no equivalent. An NNX module is
a plain Python object that does not know the attribute it hangs from, and
nothing in `nnx.Module` opens a scope.

`add_named_scopes` supplies the missing piece after construction. It walks the
model and, for every submodule, swaps in a subclass whose `__call__` runs under
a `jax.named_scope` of the attribute-path segment the module sits at (other
public methods use their own name). Nesting does the rest: the ops of
`model.decoder.layers.layers[3].mlp.wi_0` come out as
`decoder/layers/3/mlp/wi_0/...`. Containers that are iterated rather than
called, like the `nnx.List` inside `nnx.Sequential`, add no segment. Code a
module writes inline, such as the attention scores, lands in the scope of the
module that wrote it. Under scan the layer scope reads
`decoder/layers/while/body/.../mlp/wi_0`.

Scopes are HLO metadata. The computation, parameter tree and parameter paths
do not change, and `isinstance` checks against the original classes still
hold. Apply this after scan: scan requires identical layer graphdefs, and the
scoped class of layer 3 is not the scoped class of layer 4.
"""

import functools
import inspect

from flax import nnx
import jax

_SCOPE_ATTR = "_m3_named_scope"


def _scoped(fn, scope: str):
  """`fn` run under `jax.named_scope(scope)`."""

  @functools.wraps(fn)
  def wrapper(self, *args, **kwargs):
    with jax.named_scope(scope):
      return fn(self, *args, **kwargs)

  return wrapper


def _methods_to_scope(cls: type) -> dict[str, object]:
  """`__call__` and the public methods defined by `cls` below `nnx.Module`, by name."""
  methods: dict[str, object] = {}
  for klass in cls.__mro__:
    if klass in (nnx.Module, nnx.Object, object):
      break
    for name, value in vars(klass).items():
      if name in methods or not inspect.isfunction(value):
        continue
      if name == "__call__" or not name.startswith("_"):
        methods[name] = value
  return methods


@functools.cache
def _scoped_class(cls: type, scope: str) -> type:
  """Subclass of unscoped `cls` whose methods open a named scope. Cached so graphdefs stay stable."""
  namespace: dict[str, object] = {_SCOPE_ATTR: scope}
  for name, fn in _methods_to_scope(cls).items():
    namespace[name] = _scoped(fn, scope if name == "__call__" else name)
  return type(cls.__name__, (cls,), namespace)


def _unscoped(cls: type) -> type:
  """The class `add_named_scopes` started from."""
  while _SCOPE_ATTR in vars(cls):
    cls = cls.__bases__[0]
  return cls


def scope_of(module: nnx.Module) -> str | None:
  """The named scope `add_named_scopes` gave `module`, or None."""
  return getattr(type(module), _SCOPE_ATTR, None)


def add_named_scopes(model: nnx.Module) -> nnx.Module:
  """Gives every submodule of `model` a `jax.named_scope` named after its attribute path segment.

  In place and idempotent. The root keeps its class: it has no attribute name. A
  module reachable from several paths is visited once, under the first.
  """
  for path, node in nnx.iter_graph(model):
    if not path or not isinstance(node, nnx.Module):
      continue
    scope = str(path[-1])
    if scope_of(node) == scope:
      continue
    # Bypass `nnx.Object.__setattr__`, which records attributes as graph state.
    object.__setattr__(node, "__class__", _scoped_class(_unscoped(type(node)), scope))
  return model
