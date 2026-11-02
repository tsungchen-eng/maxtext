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

"""Training engine hook and sharded step helpers for GraniteMoe (`m3`)."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
from typing import Any, TYPE_CHECKING

from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3.models.granitemoe import modeling_granitemoe
import optax

if TYPE_CHECKING:
  from maxtext.training_engine import abstract_engine

_GRANITEMOE_LOSS_BATCH_KEYS = (
    "inputs",
    "targets",
    "targets_segmentation",
    "targets_position",
    "inputs_segmentation",
    "inputs_position",
    "attention_mask",
)


@dataclasses.dataclass(frozen=True)
class _FallbackWeightedMetric:
  """Fallback WeightedMetric container used only when google-tunix post-train bundle is absent."""

  unreduced_sum: jax.Array
  denominator: jax.Array


def _resolve_weighted_metric_cls() -> Any:
  """Resolves canonical Tunix/MaxText `WeightedMetric` class (falling back cleanly in base-only installs)."""
  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  try:
    from maxtext.training_engine import abstract_engine

    return abstract_engine.WeightedMetric
  except ImportError:
    try:
      from tunix.sft import utils as tunix_sft_utils

      return tunix_sft_utils.WeightedMetric
    except ImportError:
      return _FallbackWeightedMetric


def granitemoe_engine_loss_fn(
    model: modeling_granitemoe.GraniteMoeModel,
    config: Any,
    data: Mapping[str, Any],
    dropout_rng: Any = None,
    params: Any = None,
    *,
    is_train: bool,
) -> tuple[abstract_engine.WeightedMetric, dict[str, jax.Array]]:
  """MaxTextTrainingEngine loss adapter folding Switch `moe_lb_loss` into primary scalar loss."""
  metric_cls = _resolve_weighted_metric_cls()

  del dropout_rng, params
  filtered_batch = {k: data[k] for k in _GRANITEMOE_LOSS_BATCH_KEYS if k in data}
  coef_override: float | None = None
  if config is not None:
    for key in ("router_aux_loss_coef", "load_balance_loss_weight"):
      if hasattr(config, key) and getattr(config, key) is not None:
        coef_override = float(getattr(config, key))
        break

  total_loss, aux = modeling_granitemoe.compute_granitemoe_loss(
      model=model,
      tokens=filtered_batch["inputs"],
      decoder_target_tokens=filtered_batch["targets"],
      decoder_target_mask=filtered_batch.get("targets_segmentation"),
      decoder_segment_ids=filtered_batch.get("inputs_segmentation"),
      decoder_positions=filtered_batch.get("inputs_position"),
      attention_mask=filtered_batch.get("attention_mask"),
      router_aux_loss_coef=coef_override,
      is_train=is_train,
  )
  return (
      metric_cls(
          unreduced_sum=(total_loss * aux["total_weights"]).astype(jnp.float32),
          denominator=aux["total_weights"],
      ),
      aux,
  )


def configure_granitemoe_engine(engine: Any) -> Any:
  """Registers `granitemoe_engine_loss_fn` on `MaxTextTrainingEngine` via `with_loss_fn(..., has_aux=True)`."""
  engine.with_loss_fn(granitemoe_engine_loss_fn, has_aux=True)
  engine.with_gen_model_input_fn(None)
  return engine


def train_step(
    model: modeling_granitemoe.GraniteMoeModel,
    optimizer: nnx.Optimizer,
    tokens: jax.Array,
    decoder_target_tokens: jax.Array,
    decoder_target_mask: jax.Array | None = None,
    *,
    decoder_segment_ids: jax.Array | None = None,
    decoder_positions: jax.Array | None = None,
    router_aux_loss_coef: float | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
  """Runs one fused forward + backward + optimizer update step returning `(total_loss, aux_metrics)`."""

  def _loss_fn(mod: modeling_granitemoe.GraniteMoeModel) -> tuple[jax.Array, dict[str, jax.Array]]:
    return modeling_granitemoe.compute_granitemoe_loss(
        model=mod,
        tokens=tokens,
        decoder_target_tokens=decoder_target_tokens,
        decoder_target_mask=decoder_target_mask,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        router_aux_loss_coef=router_aux_loss_coef,
        is_train=True,
    )

  (loss, aux), grads = nnx.value_and_grad(_loss_fn, has_aux=True)(model)
  raw_grad_leaves = [
      v.get_raw_value() if isinstance(v, nnx.Variable) else v
      for v in jax.tree_util.tree_leaves(grads, is_leaf=lambda x: isinstance(x, nnx.Variable))
  ]
  grad_norm = optax.global_norm(raw_grad_leaves)
  optimizer.update(model, grads)
  return loss, {**aux, "grad_norm": grad_norm}
