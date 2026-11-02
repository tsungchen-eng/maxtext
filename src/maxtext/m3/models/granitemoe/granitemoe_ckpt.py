# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bidirectional HuggingFace `safetensors` <-> MaxText `m3` Orbax checkpoint converter for GraniteMoe."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
import shutil
from typing import Any, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.granitemoe import modeling_granitemoe
import ml_dtypes
import numpy as np
import orbax.checkpoint as ocp
import safetensors
from safetensors import flax as safetensors_flax


class CheckpointAuditReport(NamedTuple):
  """Completeness and bit-exactness report for HF <-> M3 Orbax conversion."""

  num_hf_keys: int
  num_m3_leaves: int
  max_abs_diff: float
  unmapped_hf_keys: tuple[str, ...]
  missing_m3_leaves: tuple[str, ...]


def _tensor_to_numpy(val: Any, target_dtype: np.dtype | None = None) -> np.ndarray:
  """Converts a `jax.Array`, `np.ndarray`, or duck-typed `torch.Tensor` (including `bf16`) to `np.ndarray`."""
  if hasattr(val, "detach") and hasattr(val, "cpu"):
    t_cpu = val.detach().cpu().contiguous()
    dtype_str = str(t_cpu.dtype)
    if "bfloat16" in dtype_str:
      import torch  # pylint: disable=g-import-not-at-top,import-outside-toplevel

      arr = t_cpu.view(torch.uint16).numpy().view(ml_dtypes.bfloat16)
    else:
      arr = t_cpu.numpy()
  else:
    arr = np.asarray(val)

  if target_dtype is not None and arr.dtype != target_dtype:
    return arr.astype(target_dtype)
  return np.array(arr, copy=True)


def load_hf_safetensors(hf_path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
  """Loads all tensors from a single `.safetensors` file or an HF checkpoint directory (without requiring `torch`)."""
  path = os.fspath(hf_path)
  shard_files: list[str] = []
  if os.path.isfile(path):
    shard_files = [path]
  elif os.path.isdir(path):
    index_path = os.path.join(path, "model.safetensors.index.json")
    single_path = os.path.join(path, "model.safetensors")
    if os.path.isfile(index_path):
      with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f).get("weight_map", {})
      shard_files = [os.path.join(path, fname) for fname in sorted(set(weight_map.values()))]
    elif os.path.isfile(single_path):
      shard_files = [single_path]
    else:
      shard_files = sorted(os.path.join(path, fn) for fn in os.listdir(path) if fn.endswith(".safetensors"))
  if not shard_files:
    raise FileNotFoundError(f"No .safetensors files found at {path!r}.")

  tensors: dict[str, np.ndarray] = {}
  for shard in shard_files:
    with safetensors.safe_open(shard, framework="flax") as f:
      for key in f.keys():
        if key in tensors:
          raise ValueError(f"Duplicate safetensors key {key!r} in {shard!r}.")
        tensors[key] = np.asarray(f.get_tensor(key))
  return tensors


def save_hf_safetensors(tensors: Mapping[str, Any], out_file: str | os.PathLike[str]) -> None:
  """Saves an HF state dictionary to a `.safetensors` file without requiring `torch`."""
  out_path = os.fspath(out_file)
  parent = os.path.dirname(out_path)
  if parent:
    os.makedirs(parent, exist_ok=True)
  jax_dict = {k: jnp.asarray(_tensor_to_numpy(v)) for k, v in tensors.items()}
  safetensors_flax.save_file(jax_dict, out_path)


def _pop_first(pool: dict[str, Any], candidates: tuple[str, ...]) -> np.ndarray:
  """Pops and returns the first matching key from `pool`, raising `KeyError` if none exist."""
  for key in candidates:
    if key in pool:
      return pool.pop(key)
  raise KeyError(f"Missing required HF checkpoint key; tried {candidates}.")


def hf_state_dict_to_m3_pure_dict(
    hf_tensors: Mapping[str, Any],
    config: modeling_granitemoe.GraniteMoeConfig,
    *,
    target_dtype: Any = None,
) -> dict[str, Any]:
  """Converts an HF GraniteMoe tensor mapping into a pure nested dict for `nnx.state(model, nnx.Param)`.

  Enforces 100% key consumption: raises `ValueError` if any HF tensor is unmapped or has unexpected shape.
  Supports both Hub safetensors keys (`router.layer.weight`, `input_linear.weight`, `output_linear.weight`)
  and `transformers` v5 in-memory aliases (`router.weight`, `experts.gate_up_proj`, `experts.down_proj`).
  """
  tgt_dt = (
      np.dtype(modeling_granitemoe._resolve_jnp_dtype(target_dtype))  # pylint: disable=protected-access
      if target_dtype is not None
      else None
  )
  pool: dict[str, np.ndarray] = {k: _tensor_to_numpy(v, tgt_dt) for k, v in hf_tensors.items()}

  v_sz = config.vocab_size
  h_sz = config.hidden_size
  i_sz = config.intermediate_size
  n_layers = config.num_hidden_layers
  q_dim = config.num_attention_heads * config.head_dim
  kv_dim = config.num_key_value_heads * config.head_dim
  n_exp = config.num_local_experts

  def _expect(arr: np.ndarray, expected: tuple[int, ...], label: str) -> np.ndarray:
    if arr.shape != expected:
      raise ValueError(f"HF tensor {label!r} shape mismatch: got {arr.shape}, expected {expected}.")
    return arr

  embed_w = _expect(_pop_first(pool, ("model.embed_tokens.weight",)), (v_sz, h_sz), "model.embed_tokens.weight")
  norm_w = _expect(_pop_first(pool, ("model.norm.weight",)), (h_sz,), "model.norm.weight")

  lm_head_kernel: np.ndarray | None = None
  if config.tie_word_embeddings:
    if "lm_head.weight" in pool:
      tied_lm = _expect(pool.pop("lm_head.weight"), (v_sz, h_sz), "lm_head.weight")
      if not np.array_equal(tied_lm.astype(np.float32), embed_w.astype(np.float32)):
        raise ValueError("tie_word_embeddings=True but HF lm_head.weight differs from model.embed_tokens.weight.")
  else:
    lm_w = _expect(_pop_first(pool, ("lm_head.weight",)), (v_sz, h_sz), "lm_head.weight")
    lm_head_kernel = np.ascontiguousarray(lm_w.T)

  in_ln_list: list[np.ndarray] = []
  post_ln_list: list[np.ndarray] = []
  q_list: list[np.ndarray] = []
  k_list: list[np.ndarray] = []
  v_list: list[np.ndarray] = []
  o_list: list[np.ndarray] = []
  qb_list: list[np.ndarray] = []
  kb_list: list[np.ndarray] = []
  vb_list: list[np.ndarray] = []
  ob_list: list[np.ndarray] = []
  router_list: list[np.ndarray] = []
  gu_list: list[np.ndarray] = []
  down_list: list[np.ndarray] = []

  for layer_idx in range(n_layers):
    pfx = f"model.layers.{layer_idx}"
    in_ln_list.append(_expect(_pop_first(pool, (f"{pfx}.input_layernorm.weight",)), (h_sz,), f"{pfx}.input_layernorm"))
    post_ln_list.append(
        _expect(
            _pop_first(pool, (f"{pfx}.post_attention_layernorm.weight",)),
            (h_sz,),
            f"{pfx}.post_attention_layernorm",
        )
    )

    q_w = _expect(_pop_first(pool, (f"{pfx}.self_attn.q_proj.weight",)), (q_dim, h_sz), f"{pfx}.q_proj.weight")
    k_w = _expect(_pop_first(pool, (f"{pfx}.self_attn.k_proj.weight",)), (kv_dim, h_sz), f"{pfx}.k_proj.weight")
    v_w = _expect(_pop_first(pool, (f"{pfx}.self_attn.v_proj.weight",)), (kv_dim, h_sz), f"{pfx}.v_proj.weight")
    o_w = _expect(_pop_first(pool, (f"{pfx}.self_attn.o_proj.weight",)), (h_sz, q_dim), f"{pfx}.o_proj.weight")
    q_list.append(q_w.T)
    k_list.append(k_w.T)
    v_list.append(v_w.T)
    o_list.append(o_w.T)

    if config.attention_bias:
      qb_list.append(_expect(_pop_first(pool, (f"{pfx}.self_attn.q_proj.bias",)), (q_dim,), f"{pfx}.q_proj.bias"))
      kb_list.append(_expect(_pop_first(pool, (f"{pfx}.self_attn.k_proj.bias",)), (kv_dim,), f"{pfx}.k_proj.bias"))
      vb_list.append(_expect(_pop_first(pool, (f"{pfx}.self_attn.v_proj.bias",)), (kv_dim,), f"{pfx}.v_proj.bias"))
      ob_list.append(_expect(_pop_first(pool, (f"{pfx}.self_attn.o_proj.bias",)), (h_sz,), f"{pfx}.o_proj.bias"))

    r_w = _expect(
        _pop_first(
            pool,
            (
                f"{pfx}.block_sparse_moe.router.layer.weight",
                f"{pfx}.block_sparse_moe.router.weight",
            ),
        ),
        (n_exp, h_sz),
        f"{pfx}.router",
    )
    router_list.append(r_w.T)

    gu_hf = _expect(
        _pop_first(
            pool,
            (
                f"{pfx}.block_sparse_moe.input_linear.weight",
                f"{pfx}.block_sparse_moe.experts.gate_up_proj",
            ),
        ),
        (n_exp, 2 * i_sz, h_sz),
        f"{pfx}.gate_up_proj",
    )
    # HF [E, 2*I, H] -> transpose(0, 2, 1) [E, H, 2*I] -> reshape [E, H, 2, I] (0=gate, 1=up).
    gu_list.append(gu_hf.transpose(0, 2, 1).reshape(n_exp, h_sz, 2, i_sz))

    down_hf = _expect(
        _pop_first(
            pool,
            (
                f"{pfx}.block_sparse_moe.output_linear.weight",
                f"{pfx}.block_sparse_moe.experts.down_proj",
            ),
        ),
        (n_exp, h_sz, i_sz),
        f"{pfx}.down_proj",
    )
    # HF [E, H, I] -> transpose(0, 2, 1) [E, I, H].
    down_list.append(down_hf.transpose(0, 2, 1))

  if pool:
    unmapped = sorted(pool.keys())
    raise ValueError(f"Unmapped HF checkpoint keys ({len(unmapped)}): {unmapped}")

  self_attn_dict: dict[str, Any] = {
      "q_proj": {"kernel": np.stack(q_list, axis=0)},
      "k_proj": {"kernel": np.stack(k_list, axis=0)},
      "v_proj": {"kernel": np.stack(v_list, axis=0)},
      "o_proj": {"kernel": np.stack(o_list, axis=0)},
  }
  if config.attention_bias:
    self_attn_dict["q_proj"]["bias"] = np.stack(qb_list, axis=0)
    self_attn_dict["k_proj"]["bias"] = np.stack(kb_list, axis=0)
    self_attn_dict["v_proj"]["bias"] = np.stack(vb_list, axis=0)
    self_attn_dict["o_proj"]["bias"] = np.stack(ob_list, axis=0)

  pure_dict: dict[str, Any] = {
      "embed_tokens": {"embedding": embed_w},
      "layers": {
          "input_layernorm": {"weight": np.stack(in_ln_list, axis=0)},
          "post_attention_layernorm": {"weight": np.stack(post_ln_list, axis=0)},
          "self_attn": self_attn_dict,
          "block_sparse_moe": {
              "router": {"weight": np.stack(router_list, axis=0)},
              "experts": {
                  "gate_up_proj": np.stack(gu_list, axis=0),
                  "down_proj": np.stack(down_list, axis=0),
              },
          },
      },
      "norm": {"weight": norm_w},
  }
  if not config.tie_word_embeddings:
    assert lm_head_kernel is not None
    pure_dict["lm_head"] = {"kernel": lm_head_kernel}
  return pure_dict


def _flatten_pure_dict(tree: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
  """Flattens a nested pure dict into `'a.b.c': array` entries."""
  flat: dict[str, Any] = {}
  for k, v in tree.items():
    path = f"{prefix}.{k}" if prefix else str(k)
    if isinstance(v, Mapping):
      flat.update(_flatten_pure_dict(v, path))
    else:
      flat[path] = v
  return flat


def load_pure_dict_into_model(model: modeling_granitemoe.GraniteMoeModel, pure_dict: Mapping[str, Any]) -> None:
  """Populates every `nnx.Param` leaf of `model` from `pure_dict`, verifying 100% leaf completeness & sharding."""
  param_state = nnx.state(model, nnx.Param)
  model_pure = nnx.to_pure_dict(param_state)
  expected_flat = _flatten_pure_dict(model_pure)
  provided_flat = _flatten_pure_dict(pure_dict)

  missing = sorted(set(expected_flat) - set(provided_flat))
  extra = sorted(set(provided_flat) - set(expected_flat))
  if missing or extra:
    raise ValueError(f"M3 Param leaf mismatch: missing={missing}, extra={extra}.")

  def _assign_leaf(path: tuple[Any, ...], var: Any) -> Any:
    if not isinstance(var, nnx.Variable):
      return var
    key = ".".join(str(getattr(p, "key", p)) for p in path)
    src_np = provided_flat[key]
    cur_arr = var.get_raw_value()
    if src_np.shape != cur_arr.shape:
      raise ValueError(f"Leaf {key!r} shape mismatch: got {src_np.shape}, expected {cur_arr.shape}.")
    placed = jnp.asarray(src_np, dtype=cur_arr.dtype)
    if isinstance(cur_arr, jax.Array) and isinstance(cur_arr.sharding, jax.sharding.NamedSharding):
      placed = jax.device_put(placed, cur_arr.sharding)
    return var.replace(placed)

  updated = jax.tree_util.tree_map_with_path(
      _assign_leaf,
      param_state,
      is_leaf=lambda x: isinstance(x, nnx.Variable),
  )
  nnx.update(model, updated)


def load_hf_weights_into_model(
    model: modeling_granitemoe.GraniteMoeModel,
    hf_dir_or_tensors: str | os.PathLike[str] | Mapping[str, Any],
) -> int:
  """Loads HF safetensors path or state_dict directly into `model` and returns the count of consumed HF keys."""
  hf_tensors = (
      dict(hf_dir_or_tensors) if isinstance(hf_dir_or_tensors, Mapping) else load_hf_safetensors(hf_dir_or_tensors)
  )
  pure_dict = hf_state_dict_to_m3_pure_dict(hf_tensors, model.config, target_dtype=model.config.weight_dtype)
  load_pure_dict_into_model(model, pure_dict)
  return len(hf_tensors)


def m3_pure_dict_to_hf_state_dict(
    pure_dict: Mapping[str, Any],
    config: modeling_granitemoe.GraniteMoeConfig,
    *,
    naming: str = "hub",
) -> dict[str, np.ndarray]:
  """Converts an M3 GraniteMoe pure parameter dict back to HuggingFace safetensors layout (`'hub'` or `'v5'`)."""
  if naming not in ("hub", "v5"):
    raise ValueError(f"naming must be 'hub' or 'v5', got {naming!r}.")

  hf_out: dict[str, np.ndarray] = {}
  hf_out["model.embed_tokens.weight"] = _tensor_to_numpy(pure_dict["embed_tokens"]["embedding"])
  hf_out["model.norm.weight"] = _tensor_to_numpy(pure_dict["norm"]["weight"])
  if not config.tie_word_embeddings:
    lm_k = _tensor_to_numpy(pure_dict["lm_head"]["kernel"])
    hf_out["lm_head.weight"] = np.ascontiguousarray(lm_k.T)

  layers = pure_dict["layers"]
  in_ln = _tensor_to_numpy(layers["input_layernorm"]["weight"])
  post_ln = _tensor_to_numpy(layers["post_attention_layernorm"]["weight"])
  attn = layers["self_attn"]
  q_k = _tensor_to_numpy(attn["q_proj"]["kernel"])
  k_k = _tensor_to_numpy(attn["k_proj"]["kernel"])
  v_k = _tensor_to_numpy(attn["v_proj"]["kernel"])
  o_k = _tensor_to_numpy(attn["o_proj"]["kernel"])
  q_b = _tensor_to_numpy(attn["q_proj"]["bias"]) if config.attention_bias else None
  k_b = _tensor_to_numpy(attn["k_proj"]["bias"]) if config.attention_bias else None
  v_b = _tensor_to_numpy(attn["v_proj"]["bias"]) if config.attention_bias else None
  o_b = _tensor_to_numpy(attn["o_proj"]["bias"]) if config.attention_bias else None

  moe = layers["block_sparse_moe"]
  router_w = _tensor_to_numpy(moe["router"]["weight"])
  gu_w = _tensor_to_numpy(moe["experts"]["gate_up_proj"])
  down_w = _tensor_to_numpy(moe["experts"]["down_proj"])

  n_exp = config.num_local_experts
  h_sz = config.hidden_size
  i_sz = config.intermediate_size

  for i in range(config.num_hidden_layers):
    pfx = f"model.layers.{i}"
    hf_out[f"{pfx}.input_layernorm.weight"] = np.ascontiguousarray(in_ln[i])
    hf_out[f"{pfx}.post_attention_layernorm.weight"] = np.ascontiguousarray(post_ln[i])
    hf_out[f"{pfx}.self_attn.q_proj.weight"] = np.ascontiguousarray(q_k[i].T)
    hf_out[f"{pfx}.self_attn.k_proj.weight"] = np.ascontiguousarray(k_k[i].T)
    hf_out[f"{pfx}.self_attn.v_proj.weight"] = np.ascontiguousarray(v_k[i].T)
    hf_out[f"{pfx}.self_attn.o_proj.weight"] = np.ascontiguousarray(o_k[i].T)
    if config.attention_bias:
      assert q_b is not None and k_b is not None and v_b is not None and o_b is not None
      hf_out[f"{pfx}.self_attn.q_proj.bias"] = np.ascontiguousarray(q_b[i])
      hf_out[f"{pfx}.self_attn.k_proj.bias"] = np.ascontiguousarray(k_b[i])
      hf_out[f"{pfx}.self_attn.v_proj.bias"] = np.ascontiguousarray(v_b[i])
      hf_out[f"{pfx}.self_attn.o_proj.bias"] = np.ascontiguousarray(o_b[i])

    r_hf = np.ascontiguousarray(router_w[i].T)
    gu_hf = np.ascontiguousarray(gu_w[i].reshape(n_exp, h_sz, 2 * i_sz).transpose(0, 2, 1))
    down_hf = np.ascontiguousarray(down_w[i].transpose(0, 2, 1))

    if naming == "hub":
      hf_out[f"{pfx}.block_sparse_moe.router.layer.weight"] = r_hf
      hf_out[f"{pfx}.block_sparse_moe.input_linear.weight"] = gu_hf
      hf_out[f"{pfx}.block_sparse_moe.output_linear.weight"] = down_hf
    else:
      hf_out[f"{pfx}.block_sparse_moe.router.weight"] = r_hf
      hf_out[f"{pfx}.block_sparse_moe.experts.gate_up_proj"] = gu_hf
      hf_out[f"{pfx}.block_sparse_moe.experts.down_proj"] = down_hf

  return hf_out


def m3_model_to_hf_state_dict(
    model: modeling_granitemoe.GraniteMoeModel,
    *,
    naming: str = "hub",
) -> dict[str, np.ndarray]:
  """Exports `model` parameters to an HF state dictionary."""
  pure_dict = nnx.to_pure_dict(nnx.state(model, nnx.Param))
  return m3_pure_dict_to_hf_state_dict(pure_dict, model.config, naming=naming)


def save_orbax_checkpoint(
    pure_or_model: modeling_granitemoe.GraniteMoeModel | Mapping[str, Any],
    orbax_dir: str | os.PathLike[str],
    *,
    overwrite: bool = True,
) -> None:
  """Saves an M3 `GraniteMoeModel` or pure parameter dict to an Orbax `StandardCheckpointer` directory."""
  out_dir = os.path.abspath(os.fspath(orbax_dir))
  if overwrite and os.path.exists(out_dir):
    shutil.rmtree(out_dir)
  os.makedirs(os.path.dirname(out_dir), exist_ok=True)

  if isinstance(pure_or_model, modeling_granitemoe.GraniteMoeModel):
    pure_state = nnx.to_pure_dict(nnx.state(pure_or_model, nnx.Param))
  else:
    pure_state = jax.tree.map(jnp.asarray, dict(pure_or_model))

  ckptr = ocp.StandardCheckpointer()
  ckptr.save(out_dir, pure_state)
  ckptr.wait_until_finished()
  ckptr.close()


def restore_orbax_into_model(
    model: modeling_granitemoe.GraniteMoeModel,
    orbax_dir: str | os.PathLike[str],
) -> None:
  """Restores an Orbax checkpoint directly onto `model`'s sharded `nnx.Param` leaves."""
  in_dir = os.path.abspath(os.fspath(orbax_dir))
  abstract_target = jax.tree.map(
      lambda arr: jax.ShapeDtypeStruct(arr.shape, arr.dtype, sharding=arr.sharding),
      nnx.to_pure_dict(nnx.state(model, nnx.Param)),
  )
  ckptr = ocp.StandardCheckpointer()
  restored_pure = ckptr.restore(in_dir, abstract_target)
  ckptr.close()
  load_pure_dict_into_model(model, restored_pure)


def convert_hf_to_orbax(
    hf_dir: str | os.PathLike[str],
    orbax_dir: str | os.PathLike[str],
    *,
    config: modeling_granitemoe.GraniteMoeConfig | None = None,
    target_dtype: Any = None,
    overwrite: bool = True,
) -> CheckpointAuditReport:
  """Converts an HF checkpoint directory to Orbax and runs a full completeness + roundtrip audit."""
  cfg = config if config is not None else modeling_granitemoe.GraniteMoeConfig.from_hf_config_json(hf_dir)
  hf_tensors = load_hf_safetensors(hf_dir)
  eff_dtype = cfg.weight_dtype if target_dtype is None else target_dtype
  pure_dict = hf_state_dict_to_m3_pure_dict(hf_tensors, cfg, target_dtype=eff_dtype)
  save_orbax_checkpoint(pure_dict, orbax_dir, overwrite=overwrite)
  return audit_hf_orbax_roundtrip(hf_tensors, orbax_dir, cfg, target_dtype=eff_dtype)


def audit_hf_orbax_roundtrip(
    hf_tensors: Mapping[str, Any],
    orbax_dir: str | os.PathLike[str],
    config: modeling_granitemoe.GraniteMoeConfig,
    *,
    target_dtype: Any = None,
    mesh: jax.sharding.Mesh | None = None,
) -> CheckpointAuditReport:
  """Restores `orbax_dir` into a fresh `GraniteMoeModel`, exports back to HF dict, and verifies 0 diff & 100% coverage."""
  eff_dtype = config.weight_dtype if target_dtype is None else target_dtype
  cfg_audit = modeling_granitemoe.GraniteMoeConfig.from_config(config, weight_dtype=eff_dtype, dtype=eff_dtype)
  model = modeling_granitemoe.create_granitemoe_model(cfg_audit, mesh=mesh, rngs=nnx.Rngs(0))
  restore_orbax_into_model(model, orbax_dir)

  naming = "v5" if any(".experts.gate_up_proj" in k for k in hf_tensors) else "hub"
  exported_hf = m3_model_to_hf_state_dict(model, naming=naming)

  tgt_np_dt = np.dtype(modeling_granitemoe._resolve_jnp_dtype(eff_dtype))  # pylint: disable=protected-access
  orig_keys = set(hf_tensors.keys())
  if cfg_audit.tie_word_embeddings:
    orig_keys_compare = orig_keys - {"lm_head.weight"}
  else:
    orig_keys_compare = orig_keys

  unmapped = tuple(sorted(orig_keys_compare - set(exported_hf.keys())))
  missing = tuple(sorted(set(exported_hf.keys()) - orig_keys_compare))
  if unmapped or missing:
    raise ValueError(f"Roundtrip key mismatch: unmapped={unmapped}, missing={missing}.")

  max_diff = 0.0
  for k in sorted(orig_keys_compare):
    ref_arr = _tensor_to_numpy(hf_tensors[k], tgt_np_dt).astype(np.float32)
    got_arr = exported_hf[k].astype(np.float32)
    if ref_arr.shape != got_arr.shape:
      raise ValueError(f"Roundtrip shape mismatch at {k!r}: {got_arr.shape} vs {ref_arr.shape}.")
    diff = float(np.max(np.abs(ref_arr - got_arr)))
    max_diff = max(max_diff, diff)

  num_m3_leaves = len(_flatten_pure_dict(nnx.to_pure_dict(nnx.state(model, nnx.Param))))
  return CheckpointAuditReport(
      num_hf_keys=len(hf_tensors),
      num_m3_leaves=num_m3_leaves,
      max_abs_diff=max_diff,
      unmapped_hf_keys=unmapped,
      missing_m3_leaves=missing,
  )


def main() -> None:
  parser = argparse.ArgumentParser(description="Convert HuggingFace GraniteMoe safetensors <-> M3 Orbax.")
  parser.add_argument("--hf-dir", required=True, help="Input HF checkpoint directory or .safetensors path.")
  parser.add_argument("--orbax-dir", required=True, help="Output Orbax checkpoint directory.")
  parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"), help="Target weight dtype.")
  args = parser.parse_args()

  cfg = modeling_granitemoe.GraniteMoeConfig.from_hf_config_json(args.hf_dir, weight_dtype=args.dtype, dtype=args.dtype)
  report = convert_hf_to_orbax(args.hf_dir, args.orbax_dir, config=cfg, target_dtype=args.dtype)
  print(
      f"Converted {report.num_hf_keys} HF tensors -> {report.num_m3_leaves} M3 leaves at {args.orbax_dir} "
      f"(roundtrip max_abs_diff={report.max_abs_diff:.3e})."
  )


if __name__ == "__main__":
  main()
