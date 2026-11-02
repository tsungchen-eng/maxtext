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

"""Real-weight `ibm-granite/granite-3.0-1b-a400m-instruct` TPU `v5p-8` parity & sharded training verification."""

from __future__ import annotations

from collections.abc import Mapping
import math
import os
import re
from typing import Any
import unittest

POC_ROOT = os.environ.setdefault("GRANITEMOE_POC_DIR", "/dev/shm/jfacevedo_m3_poc")
os.environ.setdefault("HF_HOME", os.path.join(POC_ROOT, "hf_cache"))

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from maxtext.m3 import models as m3_models
from maxtext.m3.configs import config as m3_config
from maxtext.m3.core import sharding as m3_sharding
from maxtext.m3.models.granitemoe import granitemoe_ckpt
from maxtext.m3.models.granitemoe import granitemoe_train
from maxtext.m3.models.granitemoe import modeling_granitemoe
import numpy as np
import optax

try:
  import pytest

  pytestmark = pytest.mark.tpu_only
except ImportError:
  pytest = None  # type: ignore[assignment]

jax.config.update("jax_default_matmul_precision", "highest")

try:
  import huggingface_hub
  import torch
  import transformers

  HAS_TORCH = True
except ImportError:
  huggingface_hub = None  # type: ignore[assignment]
  torch = None  # type: ignore[assignment]
  transformers = None  # type: ignore[assignment]
  HAS_TORCH = False

HF_REPO_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
_ALL_GATHER_RE = re.compile(r"\ball-gather(?:-start|-done)?\b")
_HLO_SHAPE_RE = re.compile(r"[a-z0-9]+\[([0-9,]+)\]")


def _is_tpu_or_opted_in() -> bool:
  if os.environ.get("GRANITEMOE_RUN_REAL_WEIGHT_ON_CPU") == "1":
    return True
  try:
    return jax.default_backend() == "tpu"
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def _flatten_nnx_vars(state: Any) -> list[tuple[str, Any]]:
  flat, _ = jax.tree_util.tree_flatten_with_path(state, is_leaf=lambda x: isinstance(x, nnx.Variable))
  return [(".".join(str(getattr(p, "key", p)) for p in path), leaf) for path, leaf in flat]


def _rel_l2_error(got: np.ndarray, ref: np.ndarray) -> float:
  got_f64 = np.asarray(got, dtype=np.float64)
  ref_f64 = np.asarray(ref, dtype=np.float64)
  denom = max(float(np.linalg.norm(ref_f64)), 1e-12)
  return float(np.linalg.norm(got_f64 - ref_f64) / denom)


def _kl_divergence_fp64(hf_logits: np.ndarray, m3_logits: np.ndarray) -> tuple[float, float]:
  """Returns `(mean_kl, max_token_kl)` for `KL(P_hf || P_m3)` computed in `float64`."""
  hf_f64 = np.asarray(hf_logits, dtype=np.float64)
  m3_f64 = np.asarray(m3_logits, dtype=np.float64)
  hf_log_p = hf_f64 - np.max(hf_f64, axis=-1, keepdims=True)
  hf_log_p = hf_log_p - np.log(np.sum(np.exp(hf_log_p), axis=-1, keepdims=True))
  m3_log_p = m3_f64 - np.max(m3_f64, axis=-1, keepdims=True)
  m3_log_p = m3_log_p - np.log(np.sum(np.exp(m3_log_p), axis=-1, keepdims=True))
  hf_p = np.exp(hf_log_p)
  token_kl = np.sum(hf_p * (hf_log_p - m3_log_p), axis=-1)
  token_kl = np.maximum(token_kl, 0.0)
  return float(np.mean(token_kl)), float(np.max(token_kl))


def _forbidden_expert_shapes(
    model: modeling_granitemoe.GraniteMoeModel,
    tp_size: int = 1,
) -> dict[tuple[int, ...], str]:
  """Collects unsharded/TP-only expert weight shapes that must never appear in HLO all-gathers."""
  forbidden: dict[tuple[int, ...], str] = {}
  n_layers = model.config.num_hidden_layers
  n_exp = model.config.num_local_experts
  h_dim = model.config.hidden_size
  i_dim = model.config.intermediate_size

  def _register(unstacked: tuple[int, ...], label: str) -> None:
    forbidden[(n_layers, *unstacked)] = label
    forbidden[unstacked] = label
    forbidden[(1, *unstacked)] = label
    if len(unstacked) == 4:
      d0, d1, d2, d3 = unstacked
      for collapsed in ((d0, d1, d2 * d3), (d0, d1 * d2, d3)):
        forbidden[collapsed] = label
        forbidden[(1, *collapsed)] = label
        forbidden[(n_layers, *collapsed)] = label

  for path, var in _flatten_nnx_vars(nnx.state(model, nnx.Param)):
    if ".experts." not in path:
      continue
    _register(var.get_raw_value().shape[1:], path)

  if tp_size > 1 and i_dim % tp_size == 0:
    i_tp = i_dim // tp_size
    _register((n_exp, h_dim, 2, i_tp), "layers.block_sparse_moe.experts.gate_up_proj[tp_shard]")
    _register((n_exp, i_tp, h_dim), "layers.block_sparse_moe.experts.down_proj[tp_shard]")
  return forbidden


def _assert_zero_expert_all_gather(hlo_text: str, forbidden: Mapping[tuple[int, ...], str], tag: str) -> int:
  """Asserts compiled HLO contains zero expert-weight all-gather ops and returns total all-gather count."""
  all_gather_count = 0
  for line in hlo_text.splitlines():
    stripped = line.strip()
    if stripped.startswith("//"):
      continue
    m = _ALL_GATHER_RE.search(stripped)
    if m is None:
      continue
    all_gather_count += 1
    lhs = stripped[: m.start()]
    for dims_str in _HLO_SHAPE_RE.findall(lhs):
      shape = tuple(int(d) for d in dims_str.split(",") if d.strip())
      if shape in forbidden:
        raise AssertionError(f"[{tag}] Forbidden expert all-gather {forbidden[shape]} {shape}: {stripped}")
  return all_gather_count


@unittest.skipUnless(HAS_TORCH, "PyTorch, transformers, and huggingface_hub required for real-weight parity.")
class GraniteMoeRealWeightTpuParityTest(absltest.TestCase):
  """End-to-end real-weight verification for `ibm-granite/granite-3.0-1b-a400m-instruct` on TPU `v5p-8`."""

  hf_dir: str
  orbax_bf16_dir: str
  orbax_fp32_dir: str
  hf_tokenizer: Any
  hf_model_fp32: Any
  bf16_report: granitemoe_ckpt.CheckpointAuditReport
  fp32_report: granitemoe_ckpt.CheckpointAuditReport

  @classmethod
  def setUpClass(cls):
    """Downloads real Granite-3.0-1B-A400M weights into /dev/shm and converts them to Orbax."""
    super().setUpClass()
    if not _is_tpu_or_opted_in():
      raise unittest.SkipTest(
          "Real-weight 1.3B TPU parity suite requires TPU backend (or GRANITEMOE_RUN_REAL_WEIGHT_ON_CPU=1)."
      )

    os.makedirs(POC_ROOT, exist_ok=True)
    cls.hf_dir = os.path.join(POC_ROOT, "hf_granite_3_0_1b_a400m")
    cls.orbax_bf16_dir = os.path.join(POC_ROOT, "orbax_granite_3_0_1b_a400m_bf16")
    cls.orbax_fp32_dir = os.path.join(POC_ROOT, "orbax_granite_3_0_1b_a400m_fp32")

    huggingface_hub.snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=cls.hf_dir,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab*", "merges*", "special_tokens*"],
    )

    cfg_bf16 = modeling_granitemoe.GraniteMoeConfig.from_hf_config_json(
        cls.hf_dir, dtype="bfloat16", weight_dtype="bfloat16", matmul_precision="highest"
    )
    cfg_fp32 = modeling_granitemoe.GraniteMoeConfig.from_hf_config_json(
        cls.hf_dir, dtype="float32", weight_dtype="float32", matmul_precision="highest"
    )
    cls.bf16_report = granitemoe_ckpt.convert_hf_to_orbax(
        cls.hf_dir, cls.orbax_bf16_dir, config=cfg_bf16, target_dtype="bfloat16", overwrite=True
    )
    cls.fp32_report = granitemoe_ckpt.convert_hf_to_orbax(
        cls.hf_dir, cls.orbax_fp32_dir, config=cfg_fp32, target_dtype="float32", overwrite=True
    )

    cls.hf_tokenizer = transformers.AutoTokenizer.from_pretrained(cls.hf_dir)
    cls.hf_model_fp32 = transformers.AutoModelForCausalLM.from_pretrained(
        cls.hf_dir, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()

  def test_1_full_real_checkpoint_conversion_and_100pct_leaf_audit(self):
    cfg = modeling_granitemoe.GraniteMoeConfig.from_hf_config_json(self.hf_dir)
    self.assertEqual(cfg.vocab_size, 49155)
    self.assertEqual(cfg.hidden_size, 1024)
    self.assertEqual(cfg.intermediate_size, 512)
    self.assertEqual(cfg.num_hidden_layers, 24)
    self.assertEqual(cfg.num_attention_heads, 16)
    self.assertEqual(cfg.num_key_value_heads, 8)
    self.assertEqual(cfg.num_local_experts, 32)
    self.assertEqual(cfg.num_experts_per_tok, 8)
    self.assertTrue(cfg.tie_word_embeddings)
    self.assertAlmostEqual(cfg.embedding_multiplier, 12.0)
    self.assertAlmostEqual(cfg.attention_multiplier, 0.015625)
    self.assertAlmostEqual(cfg.residual_multiplier, 0.22)
    self.assertAlmostEqual(cfg.logits_scaling, 6.0)

    expected_hf_keys = 2 + 24 * 9  # 218 safetensors keys
    for tag, rep in (("bf16", self.bf16_report), ("fp32", self.fp32_report)):
      self.assertEqual(rep.num_hf_keys, expected_hf_keys, f"[{tag}] Unexpected HF key count")
      self.assertEqual(rep.num_m3_leaves, 11, f"[{tag}] Unexpected stacked M3 leaf count")
      self.assertEmpty(rep.unmapped_hf_keys, f"[{tag}] Unmapped HF keys: {rep.unmapped_hf_keys}")
      self.assertEmpty(rep.missing_m3_leaves, f"[{tag}] Missing M3 leaves: {rep.missing_m3_leaves}")
      self.assertEqual(rep.max_abs_diff, 0.0, f"[{tag}] Non-zero roundtrip diff: {rep.max_abs_diff}")

    mesh_4 = m3_sharding.create_mesh(ici={"fsdp": 2, "expert": 2}, devices=list(jax.devices())[:4])
    cfg_bf16 = modeling_granitemoe.GraniteMoeConfig.from_hf_config_json(
        self.hf_dir, dtype="bfloat16", weight_dtype="bfloat16"
    )
    model = modeling_granitemoe.create_granitemoe_model(cfg_bf16, mesh=mesh_4, rngs=nnx.Rngs(0))
    granitemoe_ckpt.restore_orbax_into_model(model, self.orbax_bf16_dir)
    total_params = sum(math.prod(v.get_raw_value().shape) for _, v in _flatten_nnx_vars(nnx.state(model, nnx.Param)))
    hf_total_params = sum(int(p.numel()) for p in self.hf_model_fp32.parameters())
    self.assertEqual(total_params, 1_334_628_352)
    self.assertEqual(total_params, hf_total_params)

    print(
        f"\n[PHASE3-1 CHECKPOINT AUDIT] repo={HF_REPO_ID} poc_dir={POC_ROOT} "
        f"hf_keys={self.bf16_report.num_hf_keys}/218 m3_stacked_leaves={self.bf16_report.num_m3_leaves}/11 "
        f"total_params={total_params:,} bf16_roundtrip_max_abs_diff={self.bf16_report.max_abs_diff:.1e} "
        f"fp32_roundtrip_max_abs_diff={self.fp32_report.max_abs_diff:.1e}",
        flush=True,
    )

  def test_2_real_weight_tpu_v5p_frozen_topk_and_natural_routing_parity(self):
    prompts = [
        "Explain how sparse Mixture-of-Experts transformers route tokens across specialized feed-forward experts.",
        "In distributed TPU training, sharding expert weights across FSDP and expert axes eliminates weight all-gathers.",
    ]
    enc = self.hf_tokenizer(prompts, padding=False, truncation=True, max_length=24)
    min_len = min(len(ids) for ids in enc["input_ids"])
    input_ids_pt = torch.tensor([ids[:min_len] for ids in enc["input_ids"]], dtype=torch.long)
    bsz, seq_len = input_ids_pt.shape

    captured_router_logits: list[np.ndarray] = []
    captured_topk_indices: list[np.ndarray] = []
    hooks: list[Any] = []
    for layer in self.hf_model_fp32.model.layers:

      def _hook(_mod: Any, _inp: Any, out: Any) -> None:
        # HF GraniteMoeTopKGating returns (index_sorted_experts, routing_weights, batch_gates, expert_size, router_logits).
        r_logits = out[-1].detach().cpu().numpy()  # (B*S, E)
        tk_idx = torch.topk(out[-1].detach(), k=8, dim=-1).indices.cpu().numpy()  # (B*S, 8)
        captured_router_logits.append(r_logits.reshape(bsz, seq_len, 32))
        captured_topk_indices.append(tk_idx.reshape(bsz, seq_len, 8))

      hooks.append(layer.block_sparse_moe.router.register_forward_hook(_hook))

    try:
      with torch.no_grad():
        hf_out = self.hf_model_fp32(input_ids=input_ids_pt)
        hf_logits = hf_out.logits.detach().cpu().numpy()
    finally:
      for h in hooks:
        h.remove()

    hf_router_all = np.stack(captured_router_logits, axis=0)  # (24, B, S, 32)
    hf_topk_all = np.stack(captured_topk_indices, axis=0).astype(np.int32)  # (24, B, S, 8)

    cfg_fp32 = modeling_granitemoe.GraniteMoeConfig.from_hf_config_json(
        self.hf_dir, dtype="float32", weight_dtype="float32", scan_layers=True, matmul_precision="highest"
    )
    tokens_jax = jnp.asarray(input_ids_pt.numpy(), dtype=jnp.int32)
    frozen_jax = jnp.asarray(hf_topk_all, dtype=jnp.int32)

    for mesh_tag, ici in (("v5p_fsdp2_tp2", {"fsdp": 2, "tensor": 2}), ("v5p_fsdp2_ep2", {"fsdp": 2, "expert": 2})):
      mesh = m3_sharding.create_mesh(ici=ici, devices=list(jax.devices())[:4])
      model = modeling_granitemoe.create_granitemoe_model(cfg_fp32, mesh=mesh, rngs=nnx.Rngs(0))
      granitemoe_ckpt.restore_orbax_into_model(model, self.orbax_fp32_dir)

      @nnx.jit
      def _fwd_frozen(m: modeling_granitemoe.GraniteMoeModel, toks: jax.Array, frozen: jax.Array):
        return m.forward_with_details(toks, frozen_topk_indices=frozen)

      @nnx.jit
      def _fwd_natural(m: modeling_granitemoe.GraniteMoeModel, toks: jax.Array):
        return m.forward_with_details(toks)

      with jax.sharding.set_mesh(mesh):
        frozen_out = _fwd_frozen(model, tokens_jax, frozen_jax)
        natural_out = _fwd_natural(model, tokens_jax)

      m3_frozen_logits = np.asarray(frozen_out.logits)
      m3_frozen_router = np.asarray(frozen_out.router_logits)
      frozen_mean_kl, frozen_max_kl = _kl_divergence_fp64(hf_logits, m3_frozen_logits)
      frozen_rel_l2 = _rel_l2_error(m3_frozen_logits, hf_logits)
      frozen_max_abs = float(np.max(np.abs(m3_frozen_logits - hf_logits)))
      layer0_router_rel_l2 = _rel_l2_error(m3_frozen_router[0], hf_router_all[0])

      self.assertLessEqual(frozen_mean_kl, 1e-5, f"[{mesh_tag}] FrozenTopK mean KL {frozen_mean_kl} > 1e-5")
      self.assertLessEqual(frozen_rel_l2, 1e-5, f"[{mesh_tag}] FrozenTopK logits rel_l2 {frozen_rel_l2} > 1e-5")
      self.assertLessEqual(
          layer0_router_rel_l2, 1e-5, f"[{mesh_tag}] Layer-0 router rel_l2 {layer0_router_rel_l2} > 1e-5"
      )

      # Unfrozen natural routing agreement across all 24 layers on non-tie tokens (`margin > 8 * ulp`).
      m3_natural_logits = np.asarray(natural_out.logits)
      m3_natural_topk = np.asarray(natural_out.top_k_indices)  # (24, B, S, 8)
      sorted_hf_r = np.sort(hf_router_all, axis=-1)[..., ::-1]
      margin = sorted_hf_r[..., 7] - sorted_hf_r[..., 8]
      ulp_bound = 8.0 * np.spacing(np.abs(sorted_hf_r[..., 7]).astype(np.float32))
      non_tie_mask = margin > ulp_bound

      exact_set_matches: list[bool] = []
      slot_overlaps: list[float] = []
      for l_idx in range(24):
        for b_idx in range(bsz):
          for s_idx in range(seq_len):
            if not non_tie_mask[l_idx, b_idx, s_idx]:
              continue
            s_hf = set(int(x) for x in hf_topk_all[l_idx, b_idx, s_idx])
            s_m3 = set(int(x) for x in m3_natural_topk[l_idx, b_idx, s_idx])
            exact_set_matches.append(s_hf == s_m3)
            slot_overlaps.append(len(s_hf & s_m3) / 8.0)

      exact_set_rate = float(np.mean(exact_set_matches))
      slot_overlap_rate = float(np.mean(slot_overlaps))
      unfrozen_mean_kl, unfrozen_max_kl = _kl_divergence_fp64(hf_logits, m3_natural_logits)

      self.assertGreaterEqual(
          exact_set_rate,
          0.99,
          f"[{mesh_tag}] Non-tie top-8 exact set agreement {exact_set_rate:.4%} < 99%",
      )
      self.assertLessEqual(unfrozen_mean_kl, 1e-4, f"[{mesh_tag}] Unfrozen mean KL {unfrozen_mean_kl} unexpectedly high")

      print(
          f"[PHASE3-2 PARITY {mesh_tag}] shape=({bsz},{seq_len}) "
          f"FrozenTopK: mean_KL={frozen_mean_kl:.3e} max_KL={frozen_max_kl:.3e} "
          f"logits_rel_l2={frozen_rel_l2:.3e} logits_max_abs={frozen_max_abs:.3e} "
          f"layer0_router_rel_l2={layer0_router_rel_l2:.3e} | "
          f"NaturalUnfrozen: non_tie_tokens={int(np.sum(non_tie_mask))}/{non_tie_mask.size} "
          f"top8_exact_set_agree={exact_set_rate:.4%} top8_slot_overlap={slot_overlap_rate:.4%} "
          f"unfrozen_mean_KL={unfrozen_mean_kl:.3e} unfrozen_max_KL={unfrozen_max_kl:.3e}",
          flush=True,
      )

  def test_3_real_weight_16_token_greedy_generation_exact_match(self):
    prompt = "The capital of France is"
    input_ids_pt = self.hf_tokenizer(prompt, return_tensors="pt")["input_ids"].to(torch.long)
    prompt_len = int(input_ids_pt.shape[1])
    max_new_tokens = 16
    total_len = prompt_len + max_new_tokens

    with torch.no_grad():
      hf_gen_ids = (
          self.hf_model_fp32.generate(
              input_ids_pt,
              attention_mask=torch.ones_like(input_ids_pt),
              max_new_tokens=max_new_tokens,
              do_sample=False,
              use_cache=True,
              pad_token_id=self.hf_tokenizer.eos_token_id,
          )[0]
          .cpu()
          .numpy()
      )
    hf_new_ids = [int(t) for t in hf_gen_ids[prompt_len:]]

    # Run M3 autoregressive KV-cache prefill + 15 single-token decode steps on 4-chip sharded v5p-8 (`B=1`).
    mesh = m3_sharding.create_mesh(ici={"fsdp": 2, "tensor": 2}, devices=list(jax.devices())[:4])
    cfg_fp32 = modeling_granitemoe.GraniteMoeConfig.from_hf_config_json(
        self.hf_dir, dtype="float32", weight_dtype="float32", scan_layers=True, matmul_precision="highest"
    )
    model = modeling_granitemoe.create_granitemoe_model(cfg_fp32, mesh=mesh, rngs=nnx.Rngs(0))
    granitemoe_ckpt.restore_orbax_into_model(model, self.orbax_fp32_dir)

    @nnx.jit
    def _prefill(m: modeling_granitemoe.GraniteMoeModel, toks: jax.Array, cache: modeling_granitemoe.GraniteMoeKVCache):
      out = m.forward_with_details(toks, kv_cache=cache, cache_position=0)
      next_tok = jnp.argmax(out.logits[:, -1, :], axis=-1).astype(jnp.int32)
      return next_tok, out.kv_cache

    @nnx.jit
    def _decode_step(
        m: modeling_granitemoe.GraniteMoeModel,
        tok: jax.Array,
        cache: modeling_granitemoe.GraniteMoeKVCache,
        pos: jax.Array,
    ):
      out = m.forward_with_details(tok, kv_cache=cache, cache_position=pos)
      next_tok = jnp.argmax(out.logits[:, -1, :], axis=-1).astype(jnp.int32)
      return next_tok, out.kv_cache

    with jax.sharding.set_mesh(mesh):
      kv_cache = modeling_granitemoe.init_kv_cache(
          cfg_fp32, batch_size=1, max_seq_len=total_len, mesh=mesh, sharding=model.sharding  # type: ignore[arg-type]
      )
      prompt_jax = jnp.asarray(input_ids_pt.numpy(), dtype=jnp.int32)
      next_tok, kv_cache = _prefill(model, prompt_jax, kv_cache)
      m3_new_ids = [int(np.asarray(next_tok)[0])]
      for step_idx in range(1, max_new_tokens):
        cur_pos = jnp.asarray(prompt_len + step_idx - 1, dtype=jnp.int32)
        next_tok, kv_cache = _decode_step(model, next_tok[:, None], kv_cache, cur_pos)
        m3_new_ids.append(int(np.asarray(next_tok)[0]))

    decoded_hf = self.hf_tokenizer.decode(hf_new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    decoded_m3 = self.hf_tokenizer.decode(m3_new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    print(
        f"[PHASE3-3 GREEDY-16] prompt={prompt!r}\n"
        f"  HF_new_ids={hf_new_ids}\n"
        f"  M3_new_ids={m3_new_ids}\n"
        f"  decoded={decoded_m3!r}",
        flush=True,
    )
    self.assertEqual(m3_new_ids, hf_new_ids, f"16-token greedy mismatch: M3={m3_new_ids} vs HF={hf_new_ids}")
    self.assertEqual(decoded_m3, decoded_hf)

  def test_4_sharded_v5p_5_step_training_decreasing_loss_and_zero_expert_all_gather(self):
    corpus = [
        "Mixture-of-Experts models scale parameter capacity while keeping per-token FLOPs small and predictable.",
        "Granite 3.0 MoE applies embedding, attention, residual, and logits scaling multipliers across all 24 layers.",
        "MaxText m3 expresses model sharding over four logical mesh axes: data, fsdp, tensor, and expert.",
        "Switch Transformer load-balancing auxiliary loss encourages balanced token routing across all 32 experts.",
    ]
    tok_out = self.hf_tokenizer(corpus, padding="max_length", truncation=True, max_length=33, return_tensors="np")
    seq_ids = tok_out["input_ids"].astype(np.int32)
    seq_mask = tok_out["attention_mask"].astype(np.int32)
    inputs_np = seq_ids[:, :-1]  # (4, 32)
    targets_np = seq_ids[:, 1:]  # (4, 32)
    in_seg_np = seq_mask[:, :-1]
    tgt_mask_np = seq_mask[:, 1:]

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    yml_path = os.path.join(repo_root, "src", "maxtext", "configs", "models", "granite-3.0-1b-a400m.yml")
    cfg_train = m3_config.initialize(
        [
            "test",
            yml_path,
            "dtype=bfloat16",
            "weight_dtype=float32",
            "scan_layers=true",
            "remat_policy=minimal",
            "router_aux_loss_coef=0.001",
        ],
        unittest=True,
    )

    for mesh_tag, ici in (("v5p_fsdp2_ep2", {"fsdp": 2, "expert": 2}), ("v5p_fsdp2_tp2", {"fsdp": 2, "tensor": 2})):
      mesh = m3_sharding.create_mesh(ici=ici, devices=list(jax.devices())[:4])
      model = m3_models.create_model(
          cfg_train, mesh, rngs=nnx.Rngs(0), quant_mode_str="train", matmul_precision="default"
      )
      granitemoe_ckpt.restore_orbax_into_model(model, self.orbax_fp32_dir)
      forbidden = _forbidden_expert_shapes(model, tp_size=int(mesh.shape["tensor"]))

      batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(("data", "fsdp", "expert"), None))
      inputs_jax = jax.device_put(jnp.asarray(inputs_np), batch_sharding)
      targets_jax = jax.device_put(jnp.asarray(targets_np), batch_sharding)
      in_seg_jax = jax.device_put(jnp.asarray(in_seg_np), batch_sharding)
      tgt_mask_jax = jax.device_put(jnp.asarray(tgt_mask_np), batch_sharding)

      # 1. Audit compiled 24-layer real-weight forward+backward HLO for ZERO expert-weight all-gather!
      graphdef, params_st, rest_st = nnx.split(model, nnx.Param, ...)
      params_pure = nnx.to_pure_dict(params_st)

      @jax.jit
      def _pure_fwd_bwd(p_dict, t_in, t_tgt, t_in_seg, t_tgt_msk, gdef=graphdef, rst=rest_st):
        def _l(p_inner):
          mod = nnx.merge(gdef, p_inner, rst)
          return modeling_granitemoe.compute_granitemoe_loss(
              mod,
              tokens=t_in,
              decoder_target_tokens=t_tgt,
              decoder_target_mask=t_tgt_msk,
              decoder_segment_ids=t_in_seg,
          )

        return jax.value_and_grad(_l, has_aux=True)(p_dict)

      with jax.sharding.set_mesh(mesh):
        compiled = _pure_fwd_bwd.lower(params_pure, inputs_jax, targets_jax, in_seg_jax, tgt_mask_jax).compile()
        hlo_text = compiled.as_text()
        self.assertIsNotNone(hlo_text)
        ag_count = _assert_zero_expert_all_gather(hlo_text, forbidden, mesh_tag)

        # Also verify `granitemoe_engine_loss_fn` executes cleanly on the sharded 24-layer model:
        wm, eng_aux = granitemoe_train.granitemoe_engine_loss_fn(
            model,
            cfg_train,
            {
                "inputs": inputs_jax,
                "targets": targets_jax,
                "inputs_segmentation": in_seg_jax,
                "targets_segmentation": tgt_mask_jax,
            },
            is_train=True,
        )
        self.assertTrue(bool(np.isfinite(np.asarray(wm.unreduced_sum))))
        self.assertGreater(float(eng_aux["moe_lb_loss"]), 0.0)

        # 2. Run 5 sharded AdamW training steps (with standard global-norm clipping + Granite 2e-5 LR) on 4-chip v5p-8!
        tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=2e-5, weight_decay=0.01))
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        @nnx.jit
        def _jit_train_step(m, opt, t_in, t_tgt, t_in_seg, t_tgt_msk):
          return granitemoe_train.train_step(
              m,
              opt,
              tokens=t_in,
              decoder_target_tokens=t_tgt,
              decoder_target_mask=t_tgt_msk,
              decoder_segment_ids=t_in_seg,
              router_aux_loss_coef=1e-3,
          )

        step_losses: list[float] = []
        step_ce: list[float] = []
        step_lb: list[float] = []
        step_gnorm: list[float] = []
        for step in range(5):
          loss_val, metrics = _jit_train_step(model, optimizer, inputs_jax, targets_jax, in_seg_jax, tgt_mask_jax)
          tl = float(loss_val)
          ce = float(metrics["ce_loss"])
          lb = float(metrics["moe_lb_loss"])
          gn = float(metrics["grad_norm"])
          self.assertTrue(math.isfinite(tl) and math.isfinite(ce) and math.isfinite(lb) and math.isfinite(gn))
          self.assertGreater(lb, 0.0)
          self.assertGreater(gn, 0.0)
          step_losses.append(tl)
          step_ce.append(ce)
          step_lb.append(lb)
          step_gnorm.append(gn)
          print(
              f"[PHASE3-4 TRAIN {mesh_tag}] step={step} total_loss={tl:.6f} ce_loss={ce:.6f} "
              f"moe_lb_loss={lb:.6f} grad_norm={gn:.4f} (hlo_non_expert_all_gathers={ag_count})",
              flush=True,
          )

      self.assertLess(step_losses[-1], step_losses[0], f"[{mesh_tag}] Loss did not decrease: {step_losses}")
      for i in range(4):
        self.assertLess(
            step_losses[i + 1],
            step_losses[i],
            f"[{mesh_tag}] Non-monotonic training loss at step {i}->{i+1}: {step_losses}",
        )


if __name__ == "__main__":
  absltest.main()
