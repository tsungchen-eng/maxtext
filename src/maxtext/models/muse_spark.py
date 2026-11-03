# Copyright 2023–2026 Google LLC
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

"""Meta Muse-Spark (`MuseSparkForConditionalGeneration` text backbone) decoder layers."""

from typing import Any

from flax import nnx
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding

from maxtext.common.common_types import (
    Array,
    AttentionType,
    Config,
    DECODE_BATCH,
    DECODE_LENGTH,
    DType,
    D_KV,
    HEAD,
    KV_HEAD,
    MODEL_MODE_AUTOREGRESSIVE,
    MODEL_MODE_PREFILL,
    MODEL_MODE_TRAIN,
    ShardMode,
)
from maxtext.layers import attentions
from maxtext.layers import initializers
from maxtext.layers import linears
from maxtext.layers import moe
from maxtext.layers import quantizations
from maxtext.utils import max_utils
from maxtext.utils import sharding
from maxtext.utils.sharding import create_sharding


# ------------------------------------------------------------------------------
# Layer Schedule & Scan Layout Helpers
# ------------------------------------------------------------------------------


def is_full_attention_layer(layer_idx: int, config: Config) -> bool:
  """Returns True when `layer_idx` uses global NoPE attention instead of sliding-window RoPE attention.

  Matches `MuseSparkTextConfig`:
    `layer_types[i] = "full_attention" if (num_hidden_layers - 1 - i) % 4 == 0 else "sliding_attention"`
    `layer_rope_theta[i] = 0 if (num_hidden_layers - 1 - i) % 4 == 0 else rope_theta`
  """
  cycle = max(1, int(getattr(config, "inhomogeneous_layer_cycle_interval", 4) or 4))
  num_layers = int(config.num_decoder_layers)
  return ((num_layers - 1 - layer_idx) % cycle) == 0


def is_moe_layer(layer_idx: int, config: Config) -> bool:
  """Returns True when `layer_idx` uses the bottlenecked Sparse MoE block."""
  return config.num_experts > 1 and layer_idx >= config.first_num_dense_layers


def scan_layout(config: Config) -> tuple[int, int, int]:
  """Returns `(num_prefix_layers, num_blocks, remainder_start)` for `scan_layers=True`.

  Because `(num_layers - 1 - i) % cycle == 0` places `full_attention` at the last sub-layer
  of every `cycle`-layer block aligned with the end of the stack (`i % cycle == num_layers % cycle`),
  unrolling the leading `num_prefix = num_layers % cycle` layers leaves an exact integer number
  of homogeneous `[sliding, ..., sliding, full]` MoE cycles of length `cycle`.
  If `first_num_dense_layers > num_layers % cycle`, whole cycles overlapping the dense prefix
  are also unrolled into `num_prefix`.
  """
  cycle = int(getattr(config, "inhomogeneous_layer_cycle_interval", 4) or 4)
  n_layers = int(config.num_decoder_layers)
  if config.num_experts <= 1 or cycle <= 0:
    return n_layers, 0, n_layers
  align = n_layers % cycle
  if config.first_num_dense_layers <= align:
    num_prefix = align
  else:
    extra = config.first_num_dense_layers - align
    num_prefix = align + ((extra + cycle - 1) // cycle) * cycle
  num_prefix = min(n_layers, num_prefix)
  num_blocks = (n_layers - num_prefix) // cycle
  remainder_start = num_prefix + num_blocks * cycle
  return num_prefix, num_blocks, remainder_start


# ------------------------------------------------------------------------------
# Normalization & Residual Gate Modules
# ------------------------------------------------------------------------------


class MuseSparkRMSNorm(nnx.Module):
  """Muse-Spark RMSNorm (`MuseSparkRMSNorm`), supporting optional learnable scale in fp32."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-5,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = (),
      scale_init: initializers.Initializer = nnx.initializers.ones,
      with_scale: bool = True,
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.shard_mode = shard_mode
    self.kernel_axes = kernel_axes
    self.scale_init = scale_init
    self.with_scale = with_scale
    self.parameter_memory_host_offload = parameter_memory_host_offload
    if with_scale:
      self.scale = nnx.Param(
          scale_init(rngs.params(), (num_features,), weight_dtype),
          sharding=kernel_axes,
      )
    else:
      self.scale = None

  def __call__(self, x: jnp.ndarray, out_sharding: NamedSharding | None = None) -> jnp.ndarray:
    x_f32 = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * jax.lax.rsqrt(mean2 + self.epsilon)
    if self.with_scale and self.scale is not None:
      scale = self.scale[...]
      if self.parameter_memory_host_offload:
        scale = jax.device_put(scale, max_utils.device_space())
      normed = normed * jnp.asarray(scale, jnp.float32)
    out = jnp.asarray(normed, self.dtype)
    if out_sharding is not None and self.shard_mode == ShardMode.EXPLICIT:
      out = jax.lax.with_sharding_constraint(out, out_sharding)
    return out


class MuseSparkCenteredRMSNorm(nnx.Module):
  """Muse-Spark zero-centered RMSNorm (`MuseSparkCenteredRMSNorm`): `normed * (1.0 + weight)` in fp32."""

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-5,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = ("norm",),
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.shard_mode = shard_mode
    self.kernel_axes = kernel_axes
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.scale = nnx.Param(
        jnp.zeros((num_features,), dtype=weight_dtype),
        sharding=kernel_axes,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    x_f32 = jnp.asarray(x, jnp.float32)
    mean2 = jnp.mean(jax.lax.square(x_f32), axis=-1, keepdims=True)
    normed = x_f32 * jax.lax.rsqrt(mean2 + self.epsilon)
    scale = self.scale[...]
    if self.parameter_memory_host_offload:
      scale = jax.device_put(scale, max_utils.device_space())
    output = normed * (1.0 + jnp.asarray(scale, jnp.float32))
    return jnp.asarray(output, x.dtype)


class MuseSparkInputScaledRMSNorm(nnx.Module):
  """Muse-Spark input-scaled RMSNorm (`MuseSparkInputScaledRMSNorm`).

  Scales the input by `(1.0 + weight)` in fp32, casts back to input dtype, then normalizes in fp32:
    `scaled = (x.float() * (1.0 + weight.float())).type_as(x).float()`
    `normed = (scaled * rsqrt(mean(scaled^2) + eps)).type_as(x)`
  """

  def __init__(
      self,
      num_features: int,
      epsilon: float = 1e-8,
      dtype: Any = jnp.float32,
      weight_dtype: Any = jnp.float32,
      shard_mode: ShardMode = ShardMode.AUTO,
      kernel_axes: tuple[None | str, ...] = ("norm",),
      parameter_memory_host_offload: bool = False,
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.epsilon = epsilon
    self.dtype = dtype
    self.weight_dtype = weight_dtype
    self.shard_mode = shard_mode
    self.kernel_axes = kernel_axes
    self.parameter_memory_host_offload = parameter_memory_host_offload
    self.scale = nnx.Param(
        jnp.zeros((num_features,), dtype=weight_dtype),
        sharding=kernel_axes,
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    scale = self.scale[...]
    if self.parameter_memory_host_offload:
      scale = jax.device_put(scale, max_utils.device_space())
    scaled = (jnp.asarray(x, jnp.float32) * (1.0 + jnp.asarray(scale, jnp.float32))).astype(x.dtype).astype(jnp.float32)
    mean2 = jnp.mean(jax.lax.square(scaled), axis=-1, keepdims=True)
    normed = scaled * jax.lax.rsqrt(mean2 + self.epsilon)
    return jnp.asarray(normed, x.dtype)


class MuseSparkResidualGate(nnx.Module):
  """Learnable temperature-scaled residual gate (`MuseSparkResidualGate`)."""

  def __init__(
      self,
      num_features: int,
      temperature: float = 0.3,
      eps: float = 1e-3,
      weight_dtype: Any = jnp.float32,
      kernel_axes: tuple[None | str, ...] = ("norm",),
      *,
      rngs: nnx.Rngs,
  ):
    self.num_features = num_features
    self.temperature = float(temperature)
    self.eps = float(eps)
    self.gate = nnx.Param(
        jnp.zeros((num_features,), dtype=weight_dtype),
        sharding=kernel_axes,
    )

  def __call__(self, residual: jax.Array, hidden_states: jax.Array) -> jax.Array:
    w = jnp.asarray(self.gate[...], jnp.float32) / self.temperature
    beta = jax.nn.sigmoid(w)
    alpha = jnp.sqrt(jnp.maximum(jax.nn.sigmoid(-w) * (1.0 + beta), self.eps))
    out = alpha * jnp.asarray(residual, jnp.float32) + beta * jnp.asarray(hidden_states, jnp.float32)
    return out.astype(residual.dtype)


# ------------------------------------------------------------------------------
# Rotary Position Embedding & Gated GQA Attention
# ------------------------------------------------------------------------------


def _apply_half_rotate_rope(x: jax.Array, positions: jax.Array, rope_theta: float) -> jax.Array:
  """Applies HF non-interleaved (`rotate_half`) RoPE matching `MuseSparkTextRotaryEmbedding`."""
  dim = x.shape[-1]
  inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
  freqs = jnp.einsum("bs,d->bsd", positions.astype(jnp.float32), inv_freq, precision=jax.lax.Precision.HIGHEST)
  emb = jnp.concatenate([freqs, freqs], axis=-1)[..., None, :]
  cos = jnp.cos(emb).astype(x.dtype)
  sin = jnp.sin(emb).astype(x.dtype)
  x1, x2 = jnp.split(x, 2, axis=-1)
  rotated = jnp.concatenate([-x2, x1], axis=-1)
  return (x * cos + rotated * sin).astype(x.dtype)


class MuseSparkAttention(attentions.Attention):
  """Muse-Spark Gated GQA Attention (`MuseSparkTextAttention`).

  Features:
  - Scaleless RMSNorm (`qk_norm`) on Q and K, followed by `query = (qk_norm(query) * qk_scale_factor).to(dtype)`.
  - Non-interleaved RoPE on sliding-window layers (`is_full_attention=False`), NoPE on full-attention layers.
  - Per-head scaleless RMSNorm (`o_norm`) on attention output before multiplying by `sigmoid(gate_proj(inputs_q))`.
  """

  def __init__(
      self,
      config: Config,
      num_query_heads: int,
      num_kv_heads: int,
      head_dim: int,
      max_target_length: int,
      mesh: Mesh,
      attention_kernel: str,
      inputs_q_shape: tuple[int, ...],
      inputs_kv_shape: tuple[int, ...],
      dtype: DType = jnp.float32,
      weight_dtype: DType = jnp.float32,
      max_prefill_predict_length: int = -1,
      dropout_rate: float = 0.0,
      quant: quantizations.AqtQuantization | None = None,
      kv_quant: quantizations.KVQuant | None = None,
      is_full_attention: bool = False,
      model_mode: str = MODEL_MODE_TRAIN,
      *,
      rngs: nnx.Rngs,
  ):
    attention_type = AttentionType.GLOBAL if is_full_attention else AttentionType.LOCAL_SLIDING
    sliding_window_size = 0 if is_full_attention else int(config.sliding_window_size)
    super().__init__(
        config=config,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=max_target_length,
        mesh=mesh,
        attention_kernel=attention_kernel,
        inputs_q_shape=inputs_q_shape,
        inputs_kv_shape=inputs_kv_shape,
        dtype=dtype,
        weight_dtype=weight_dtype,
        max_prefill_predict_length=max_prefill_predict_length,
        dropout_rate=dropout_rate,
        quant=quant,
        kv_quant=kv_quant,
        attention_type=attention_type,
        sliding_window_size=sliding_window_size,
        query_pre_attn_scalar=float(head_dim**-0.5),
        is_nope_layer=is_full_attention,
        use_bias_in_projections=False,
        model_mode=model_mode,
        rngs=rngs,
    )
    self.is_full_attention = is_full_attention
    self.qk_scale_factor = float(config.muse_spark_qk_scale_factor)
    self.gate_proj = self.init_query_w(inputs_q_shape=inputs_q_shape)
    self.qk_norm = MuseSparkRMSNorm(
        num_features=head_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=dtype,
        weight_dtype=weight_dtype,
        shard_mode=config.shard_mode,
        with_scale=False,
        rngs=rngs,
    )
    self.o_norm = MuseSparkRMSNorm(
        num_features=head_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=dtype,
        weight_dtype=weight_dtype,
        shard_mode=config.shard_mode,
        with_scale=False,
        rngs=rngs,
    )

  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array | None = None,
      decoder_segment_ids: Array | None = None,
      out_sharding: NamedSharding | None = None,
      *,
      model_mode: str = MODEL_MODE_TRAIN,
      deterministic: bool = False,
      previous_chunk: Any = None,
      slot: int | None = None,
      bidirectional_mask: Any = None,
      rope_kwargs: dict | None = None,
      kv_cache: Array | None = None,
      attention_metadata: dict[str, Any] | None = None,
      **kwargs,
  ):
    del slot, rope_kwargs, kwargs
    if model_mode == MODEL_MODE_PREFILL:
      input_axis_names = self.prefill_input_axis_names
    elif model_mode == MODEL_MODE_TRAIN:
      input_axis_names = self.input_axis_names
    else:
      input_axis_names = self.decode_input_axis_names

    inputs_q = self._maybe_shard_with_logical(inputs_q, input_axis_names)
    inputs_kv = self._maybe_shard_with_logical(inputs_kv, input_axis_names)
    qkv_sharding = create_sharding(self.mesh, input_axis_names)

    query = self.query_projection(inputs_q, out_sharding=qkv_sharding)
    key = self.kv_projection(inputs_kv, proj_name="key", out_sharding=qkv_sharding)
    value = self.kv_projection(inputs_kv, proj_name="value", out_sharding=qkv_sharding)

    query = (self.qk_norm(query) * self.qk_scale_factor).astype(query.dtype)
    key = self.qk_norm(key)

    if not self.is_full_attention:
      if inputs_positions is None:
        inputs_positions = jnp.arange(inputs_q.shape[1], dtype=jnp.int32)[None, :]
      query = _apply_half_rotate_rope(query, inputs_positions, float(self.config.rope_max_timescale))
      key = _apply_half_rotate_rope(key, inputs_positions, float(self.config.rope_max_timescale))

    if self.query_pre_attn_scalar and self.query_pre_attn_scalar != 1.0:
      query = (query * self.query_pre_attn_scalar).astype(query.dtype)

    if model_mode == MODEL_MODE_PREFILL:
      query = self._maybe_shard_with_logical(query, self.prefill_query_axis_names)
      key = self._maybe_shard_with_logical(key, self.prefill_key_axis_names)
      value = self._maybe_shard_with_logical(value, self.prefill_value_axis_names)
    elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
      query = self._maybe_shard_with_logical(query, (DECODE_BATCH, DECODE_LENGTH, HEAD, D_KV))
      key = self._maybe_shard_with_logical(key, (DECODE_BATCH, DECODE_LENGTH, KV_HEAD, D_KV))
      value = self._maybe_shard_with_logical(value, (DECODE_BATCH, DECODE_LENGTH, KV_HEAD, D_KV))
    else:
      query = self._maybe_shard_with_logical(query, self.query_axis_names)
      key = self._maybe_shard_with_logical(key, self.key_axis_names)
      value = self._maybe_shard_with_logical(value, self.value_axis_names)

    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    if self.config.attention in ("vllm_rpa", "vllm_batched_rpa") and model_mode != MODEL_MODE_TRAIN:
      batch, seq_len, num_heads, head_dim = query.shape
      attn_out, updated_kv = self.forward_serve_vllm(
          query, key, value, rpa_kv_cache=kv_cache, rpa_metadata=attention_metadata
      )
      out = attn_out.reshape(batch, seq_len, num_heads, head_dim)
      kv_cache = updated_kv
    else:
      cached_values = [None, None]
      if model_mode != MODEL_MODE_TRAIN:
        cached_values = self.update_kv_caches(key, value, decoder_segment_ids, model_mode, previous_chunk)
      out = self.attention_op(
          query,
          key,
          value,
          decoder_segment_ids,
          inputs_positions,
          model_mode,
          cached_values,
          previous_chunk,
          bidirectional_mask,
          self.sinks,
      )

    out = checkpoint_name(out, "attention_out")
    if model_mode == MODEL_MODE_PREFILL:
      out = self._maybe_shard_with_logical(out, self.prefill_out_axis_names)
    elif model_mode == MODEL_MODE_TRAIN:
      out = self._maybe_shard_with_logical(out, self.out_axis_names)
    else:
      out = self._maybe_shard_with_logical(out, self.decode_out_axis_names)

    out = self.o_norm(out)
    gate = jax.nn.sigmoid(self.gate_proj(inputs_q, out_sharding=qkv_sharding).astype(jnp.float32)).astype(out.dtype)
    out = out * gate

    out = self.out_projection(out, out_sharding=out_sharding)
    out = checkpoint_name(out, "out_proj")
    return out, kv_cache


# ------------------------------------------------------------------------------
# Bottlenecked Sparse MoE (`MuseSparkTextSparseMoeBlock`)
# ------------------------------------------------------------------------------


class MuseSparkRouterGate(nnx.Module):
  """Router gate holding fp32 `kernel` and `MoEBiasVar` `bias`."""

  def __init__(
      self,
      emb_dim: int,
      num_experts: int,
      *,
      rngs: nnx.Rngs,
  ):
    self.kernel = nnx.Param(
        initializers.nd_dense_init(1.0, "fan_in", "truncated_normal")(
            rngs.params(), (emb_dim, num_experts), jnp.float32, 0, 1
        ),
        sharding=("embed_router", None),
    )
    self.bias = moe.MoEBiasVar(
        jnp.zeros((num_experts,), dtype=jnp.float32),
        sharding=(None,),
    )


class MuseSparkMoE(nnx.Module):
  """Muse-Spark bottlenecked Sparse MoE block (`MuseSparkTextSparseMoeBlock`).

  Structure:
  1. Router (`MuseSparkRouterGate`) operates on the full-width `emb_dim` (8192) hidden states in fp32,
     scales router logits by `muse_spark_output_multiplier`, applies sigmoid + `e_score_correction_bias`
     for top-k selection, and normalizes the un-biased sigmoid weights over the selected top-k experts.
  2. Shared width-bottleneck down-projection `pre_expert_proj` (`emb_dim -> moe_expert_input_dim`)
     followed by `pre_expert_norm` (`MuseSparkCenteredRMSNorm`).
  3. 256 routed SwiGLU experts (`wi_0`, `wi_1`, `wo` of shape `[num_experts, moe_expert_input_dim, moe_mlp_dim]`)
     followed by per-expert-token `post_expert_norm` (`MuseSparkInputScaledRMSNorm`) before weighting by
     `top_k_weights` and summing over the `k` selected experts.
  4. Shared width-bottleneck up-projection `post_expert_proj` (`moe_expert_input_dim -> emb_dim`).
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      quant: quantizations.AqtQuantization | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.num_experts = int(config.num_experts)
    self.num_experts_per_tok = int(config.num_experts_per_tok)
    self.emb_dim = int(config.emb_dim)
    self.expert_input_dim = int(config.moe_expert_input_dim if config.moe_expert_input_dim > 0 else config.emb_dim)
    self.moe_mlp_dim = int(config.moe_mlp_dim)
    self.output_multiplier = float(config.muse_spark_output_multiplier)
    self.routed_scaling_factor = float(config.routed_scaling_factor)
    self.norm_topk_prob = bool(config.norm_topk_prob)

    self.gate = MuseSparkRouterGate(
        emb_dim=self.emb_dim,
        num_experts=self.num_experts,
        rngs=rngs,
    )
    self.pre_expert_proj = linears.DenseGeneral(
        in_features_shape=self.emb_dim,
        out_features_shape=self.expert_input_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("embed", "mlp"),
        quant=quant,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        use_bias=False,
        rngs=rngs,
    )
    self.pre_expert_norm = MuseSparkCenteredRMSNorm(
        num_features=self.expert_input_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    expert_init = initializers.nd_dense_init(config.dense_init_scale, "fan_in", "truncated_normal")
    self.wi_0 = nnx.Param(
        expert_init(
            rngs.params(), (self.num_experts, self.expert_input_dim, self.moe_mlp_dim), config.weight_dtype, 1, 2
        ),
        sharding=("exp", "embed_moe", "mlp_moe"),
    )
    self.wi_1 = nnx.Param(
        expert_init(
            rngs.params(), (self.num_experts, self.expert_input_dim, self.moe_mlp_dim), config.weight_dtype, 1, 2
        ),
        sharding=("exp", "embed_moe", "mlp_moe"),
    )
    self.wo = nnx.Param(
        expert_init(
            rngs.params(), (self.num_experts, self.moe_mlp_dim, self.expert_input_dim), config.weight_dtype, 1, 2
        ),
        sharding=("exp", "mlp_moe", "embed_moe"),
    )
    self.post_expert_norm = MuseSparkInputScaledRMSNorm(
        num_features=self.expert_input_dim,
        epsilon=config.muse_spark_post_norm_eps,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.post_expert_proj = linears.DenseGeneral(
        in_features_shape=self.expert_input_dim,
        out_features_shape=self.emb_dim,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        kernel_axes=("mlp", "embed"),
        quant=quant,
        shard_mode=config.shard_mode,
        matmul_precision=config.matmul_precision,
        use_bias=False,
        rngs=rngs,
    )

  def __call__(self, inputs: jax.Array, forced_routed_experts: jax.Array | None = None) -> jax.Array:
    batch_size, seq_len, _ = inputs.shape
    num_tokens = batch_size * seq_len
    k = self.num_experts_per_tok
    d_in = self.expert_input_dim

    # 1. Router on full-width hidden_states (in float32)
    router_logits = jnp.einsum(
        "bsd,de->bse",
        inputs.astype(jnp.float32),
        self.gate.kernel[...].astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST,
    )
    router_logits = router_logits * self.output_multiplier
    routing_weights = jax.nn.sigmoid(router_logits)
    scores_for_choice = routing_weights + self.gate.bias[...].astype(jnp.float32)
    _, top_k_index = jax.lax.top_k(scores_for_choice, k=k)
    if forced_routed_experts is not None:
      top_k_index = forced_routed_experts.astype(top_k_index.dtype)
    if self.config.return_routed_experts:
      self.selected_experts = nnx.Intermediate(top_k_index)

    top_k_weights = jnp.take_along_axis(routing_weights, top_k_index, axis=-1)
    if self.norm_topk_prob:
      top_k_weights = top_k_weights / (jnp.sum(top_k_weights, axis=-1, keepdims=True) + 1e-15)
    top_k_weights = (top_k_weights * self.routed_scaling_factor).astype(inputs.dtype)

    # 2. Width bottleneck before experts
    x = self.pre_expert_proj(inputs)
    x = self.pre_expert_norm(x)
    x_flat = jnp.reshape(x, (num_tokens, d_in))

    # 3. Sort token-expert pairs by expert index and evaluate routed experts via ragged_dot
    flat_expert_ids = jnp.reshape(top_k_index, (num_tokens * k,))
    flat_token_ids = jnp.repeat(jnp.arange(num_tokens, dtype=jnp.int32), k)
    flat_weights = jnp.reshape(top_k_weights, (num_tokens * k,))

    perm = jnp.argsort(flat_expert_ids, stable=True)
    sorted_expert_ids = flat_expert_ids[perm]
    sorted_token_ids = flat_token_ids[perm]
    sorted_weights = flat_weights[perm]
    x_sorted = x_flat[sorted_token_ids]
    group_sizes = jnp.bincount(sorted_expert_ids, length=self.num_experts).astype(jnp.int32)

    wi_0 = self.wi_0[...].astype(x.dtype)
    wi_1 = self.wi_1[...].astype(x.dtype)
    wo = self.wo[...].astype(x.dtype)

    gate_out = jax.lax.ragged_dot(x_sorted, wi_0, group_sizes, precision=self.config.matmul_precision)
    up_out = jax.lax.ragged_dot(x_sorted, wi_1, group_sizes, precision=self.config.matmul_precision)
    act = jax.nn.silu(gate_out.astype(jnp.float32)).astype(x.dtype) * up_out
    expert_out = jax.lax.ragged_dot(act, wo, group_sizes, precision=self.config.matmul_precision)

    # Per-expert-output input-scaled RMSNorm before weighting by top_k_weights
    expert_out = self.post_expert_norm(expert_out)
    weighted_sorted = (expert_out * sorted_weights[:, None]).astype(x.dtype)

    # Group back by token_id while preserving ascending expert_idx order (matches HF loop order)
    token_group_perm = jnp.argsort(sorted_token_ids, stable=True)
    by_token = jnp.reshape(weighted_sorted[token_group_perm], (batch_size, seq_len, k, d_in))
    combined = jnp.sum(by_token, axis=2)

    # 4. Width bottleneck up-projection back to emb_dim
    return self.post_expert_proj(combined)


# ------------------------------------------------------------------------------
# Decoder Layer & Scannable Block
# ------------------------------------------------------------------------------


class MuseSparkDecoderLayer(nnx.Module):
  """Single Muse-Spark transformer decoder layer (`MuseSparkTextDecoderLayer`)."""

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str = MODEL_MODE_TRAIN,
      quant: quantizations.AqtQuantization | None = None,
      layer_idx: int = 0,
      is_full_attention_override: bool | None = None,
      is_moe_override: bool | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.quant = quant
    self.model_mode = model_mode
    self.layer_idx = layer_idx
    self.is_full_attention = (
        is_full_attention_layer(layer_idx, config)
        if is_full_attention_override is None
        else bool(is_full_attention_override)
    )
    self.is_moe = is_moe_layer(layer_idx, config) if is_moe_override is None else bool(is_moe_override)

    batch_size, seq_len = max_utils.get_batch_seq_len_for_mode(config, model_mode)
    dummy_inputs_shape = (batch_size, seq_len, config.emb_dim)

    self.input_layernorm = MuseSparkCenteredRMSNorm(
        num_features=config.emb_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.self_attention = MuseSparkAttention(
        config=config,
        num_query_heads=config.num_query_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        max_target_length=config.max_target_length,
        mesh=mesh,
        attention_kernel=config.attention,
        inputs_q_shape=dummy_inputs_shape,
        inputs_kv_shape=dummy_inputs_shape,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        max_prefill_predict_length=config.max_prefill_predict_length,
        dropout_rate=config.dropout_rate,
        quant=quant,
        kv_quant=quantizations.configure_kv_quant(config),
        is_full_attention=self.is_full_attention,
        model_mode=model_mode,
        rngs=rngs,
    )
    self.post_attention_layernorm = MuseSparkRMSNorm(
        num_features=config.emb_dim,
        epsilon=config.muse_spark_post_norm_eps,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        with_scale=False,
        rngs=rngs,
    )
    self.post_attention_residual_gate = MuseSparkResidualGate(
        num_features=config.emb_dim,
        temperature=config.muse_spark_residual_gate_temperature,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    self.pre_feedforward_layernorm = MuseSparkCenteredRMSNorm(
        num_features=config.emb_dim,
        epsilon=config.normalization_layer_epsilon,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    if self.is_moe:
      self.moe_block = MuseSparkMoE(
          config=config,
          mesh=mesh,
          quant=quant,
          rngs=rngs,
      )
    else:
      self.mlp = linears.MlpBlock(
          in_features=config.emb_dim,
          intermediate_dim=config.mlp_dim,
          activations=config.mlp_activations,
          intermediate_dropout_rate=config.dropout_rate,
          dtype=config.dtype,
          weight_dtype=config.weight_dtype,
          config=config,
          mesh=mesh,
          quant=quant,
          model_mode=model_mode,
          rngs=rngs,
      )
    self.post_feedforward_layernorm = MuseSparkInputScaledRMSNorm(
        num_features=config.emb_dim,
        epsilon=config.muse_spark_post_norm_eps,
        dtype=config.dtype,
        weight_dtype=config.weight_dtype,
        shard_mode=config.shard_mode,
        kernel_axes=("norm",),
        rngs=rngs,
    )
    self.post_feedforward_residual_gate = MuseSparkResidualGate(
        num_features=config.emb_dim,
        temperature=config.muse_spark_residual_gate_temperature,
        weight_dtype=config.weight_dtype,
        kernel_axes=("norm",),
        rngs=rngs,
    )

    if model_mode == MODEL_MODE_PREFILL:
      self.activation_axis_names = ("activation_batch", "prefill_activation_norm_length", "activation_embed")
    else:
      self.activation_axis_names = ("activation_batch", "activation_norm_length", "activation_embed")

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids: jax.Array | None,
      decoder_positions: jax.Array | None,
      deterministic: bool,
      model_mode: str,
      previous_chunk: Any = None,
      slot: int | None = None,
      kv_cache: jax.Array | None = None,
      attention_metadata: dict[str, Any] | None = None,
      forced_routed_experts: jax.Array | None = None,
      **kwargs,
  ):
    del kwargs
    inputs = sharding.maybe_shard_with_logical(
        inputs,
        self.activation_axis_names,
        self.mesh,
        self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
    )
    inputs = checkpoint_name(inputs, "decoder_layer_input")

    # 1. Self-Attention sub-block
    residual = inputs
    hidden_states = self.input_layernorm(inputs)
    attn_out, kv_cache = self.self_attention(
        hidden_states,
        hidden_states,
        decoder_positions,
        decoder_segment_ids,
        model_mode=model_mode,
        deterministic=deterministic,
        previous_chunk=previous_chunk,
        slot=slot,
        kv_cache=kv_cache,
        attention_metadata=attention_metadata,
    )
    attn_out = self.post_attention_layernorm(attn_out)
    hidden_states = self.post_attention_residual_gate(residual, attn_out)

    # 2. Feedforward / MoE sub-block
    residual = hidden_states
    ffn_in = self.pre_feedforward_layernorm(hidden_states)
    if self.is_moe:
      ffn_out = self.moe_block(ffn_in, forced_routed_experts=forced_routed_experts)
    else:
      ffn_out = self.mlp(ffn_in, deterministic=deterministic)
    ffn_out = self.post_feedforward_layernorm(ffn_out)
    output = self.post_feedforward_residual_gate(residual, ffn_out)

    output = sharding.maybe_shard_with_logical(
        output,
        self.activation_axis_names,
        self.mesh,
        self.config.shard_mode,
        debug_sharding=self.config.debug_sharding,
    )
    if self.config.scan_layers:
      return output, None
    return output, kv_cache


class MuseSparkScannableBlock(nnx.Module):
  """Scannable inhomogeneous cycle of `inhomogeneous_layer_cycle_interval` (4) Muse-Spark MoE layers.

  Within each cycle:
  - Sub-layers `0 .. cycle - 2` use sliding-window attention (`sliding_window_size=2048`) + RoPE (`is_full_attention=False`).
  - Sub-layer `cycle - 1` uses global causal attention + NoPE (`is_full_attention=True`).
  """

  def __init__(
      self,
      config: Config,
      mesh: Mesh,
      model_mode: str = MODEL_MODE_TRAIN,
      quant: quantizations.AqtQuantization | None = None,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self.model_mode = model_mode
    self.quant = quant
    self.num_of_layers = int(config.inhomogeneous_layer_cycle_interval)
    for sub_id in range(self.num_of_layers):
      layer = MuseSparkDecoderLayer(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          quant=quant,
          layer_idx=sub_id,
          is_full_attention_override=(sub_id == self.num_of_layers - 1),
          is_moe_override=True,
          rngs=rngs,
      )
      setattr(self, f"layers_{sub_id}", layer)

  def __call__(
      self,
      inputs: jax.Array,
      decoder_segment_ids: jax.Array | None,
      decoder_positions: jax.Array | None,
      deterministic: bool,
      model_mode: str,
      previous_chunk: Any = None,
      slot: int | None = None,
      kv_cache: jax.Array | None = None,
      attention_metadata: dict[str, Any] | None = None,
      forced_routed_experts: jax.Array | None = None,
      **kwargs,
  ):
    del kv_cache, kwargs
    y = inputs
    for sub_id in range(self.num_of_layers):
      sub_forced = forced_routed_experts[sub_id] if forced_routed_experts is not None else None
      layer = getattr(self, f"layers_{sub_id}")
      y, _ = layer(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          previous_chunk=previous_chunk,
          slot=slot,
          attention_metadata=attention_metadata,
          forced_routed_experts=sub_forced,
      )
    return y, None
