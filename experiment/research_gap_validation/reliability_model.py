"""Backward-compatible propagation and late reliability controls for SDT.

No SDT source file is modified. Runtime hooks act on the existing softmax and
feature-reduction modules, preserving checkpoint compatibility.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Mapping, Sequence

import torch
import torch.nn as nn


MODALITIES = ("t", "a", "v")
MODALITY_NAMES = ("text", "audio", "visual")
MODALITY_NAME_TO_INDEX = {name: index for index, name in enumerate(MODALITY_NAMES)}
BRANCHES = ("t_t", "a_t", "v_t", "a_a", "t_a", "v_a", "v_v", "t_v", "a_v")
SOURCE_MODALITY = {name: MODALITIES.index(name.split("_")[0]) for name in BRANCHES}


def identity_logit(value: float) -> float:
    if not 0.5 < value < 1.0:
        raise ValueError("identity gate initialization must be in (0.5, 1)")
    return math.log(value / (1.0 - value))


class ModalityGateNetwork(nn.Module):
    """One gate per target/modality; shared by B1 and B2."""

    def __init__(self, hidden_dim: int, gate_hidden: int, initial_gate: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3 * hidden_dim, gate_hidden), nn.GELU(),
            nn.Linear(gate_hidden, 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, identity_logit(initial_gate))

    def forward(self, target_summary: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(target_summary))


class PairGateNetwork(nn.Module):
    """Efficient source-specific gates without materializing 4H pair features."""

    def __init__(self, hidden_dim: int, gate_hidden: int, initial_gate: float):
        super().__init__()
        self.query = nn.ModuleList([
            nn.Linear(3 * hidden_dim, gate_hidden) for _ in MODALITIES
        ])
        self.key = nn.ModuleList([
            nn.Linear(hidden_dim, gate_hidden) for _ in MODALITIES
        ])
        self.structural = nn.ModuleList([
            nn.Linear(4, 1) for _ in MODALITIES
        ])
        self.content_scale = nn.Parameter(torch.zeros(3))
        self.bias = nn.Parameter(torch.full((3,), identity_logit(initial_gate)))
        for layer in self.structural:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, target_summary: torch.Tensor,
                modality_features: Sequence[torch.Tensor],
                speakers: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, length, _ = target_summary.shape
        positions = torch.arange(length, device=target_summary.device)
        signed = positions[None, :, None] - positions[None, None, :]
        absolute = signed.abs().float()
        scale = max(1, length - 1)
        same_speaker = speakers[:, :, None].eq(speakers[:, None, :]).float()
        structure = torch.stack([
            torch.log1p(absolute).expand(batch, -1, -1),
            (signed.float() / scale).expand(batch, -1, -1),
            same_speaker,
            torch.ones_like(same_speaker),
        ], dim=-1)
        gates = []
        for modality, source in enumerate(modality_features):
            query = self.query[modality](target_summary)
            key = self.key[modality](source)
            content = torch.einsum("bih,bjh->bij", query, key) / math.sqrt(query.size(-1))
            logits = (self.bias[modality]
                      + self.content_scale[modality] * content
                      + self.structural[modality](structure).squeeze(-1))
            gates.append(torch.sigmoid(logits))
        result = torch.stack(gates, dim=-1)
        valid_pairs = valid[:, :, None] & valid[:, None, :]
        result = torch.where(valid_pairs.unsqueeze(-1), result, torch.ones_like(result))
        diagonal = torch.eye(length, dtype=torch.bool, device=result.device)
        return torch.where(diagonal[None, :, :, None], torch.ones_like(result), result)


class CapacityResidual(nn.Module):
    def __init__(self, hidden_dim: int, capacity_hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, capacity_hidden), nn.GELU(),
            nn.Linear(capacity_hidden, hidden_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return fused + self.network(fused)


class ReliabilitySDT(nn.Module):
    """Wrap a checkpoint-compatible SDT with controlled reliability gates.

    Modes: B0, B1 (late modality), B2 (early shared modality), B3/B4/B6
    (early source-specific pair), and B5 (capacity control).
    """

    def __init__(self, base: nn.Module, model_id: str, gate_hidden: int = 64,
                 initial_gate: float = 0.99, capacity_hidden: int = 64,
                 renormalize_attention: bool = False,
                 gate_modalities: Sequence[str] = MODALITY_NAMES):
        super().__init__()
        if model_id not in {f"B{index}" for index in range(7)}:
            raise ValueError(f"Unknown model ID: {model_id}")
        self.base = base
        self.model_id = model_id
        self.renormalize_attention = bool(renormalize_attention)
        unknown_modalities = sorted(set(gate_modalities).difference(MODALITY_NAMES))
        if unknown_modalities:
            raise ValueError(f"Unknown gate modalities: {unknown_modalities}")
        self.gate_modalities = tuple(
            name for name in MODALITY_NAMES if name in set(gate_modalities)
        )
        if model_id in ("B1", "B2", "B3", "B4", "B6") and not self.gate_modalities:
            raise ValueError("At least one gate modality is required for a gated model")
        self.active_gate_indices = tuple(
            MODALITY_NAME_TO_INDEX[name] for name in self.gate_modalities
        )
        hidden = int(base.all_output_layer.in_features)
        self.hidden_dim = hidden
        self.gate_hidden = int(gate_hidden)
        self.capacity_hidden = int(capacity_hidden)
        self.modality_gate = (
            ModalityGateNetwork(hidden, gate_hidden, initial_gate)
            if model_id in ("B1", "B2") else None
        )
        self.pair_gate = (
            PairGateNetwork(hidden, gate_hidden, initial_gate)
            if model_id in ("B3", "B4", "B6") else None
        )
        self.capacity_control = (
            CapacityResidual(hidden, capacity_hidden) if model_id == "B5" else None
        )
        self.last_attention: dict[str, torch.Tensor] = {}
        self.last_enhanced: dict[str, torch.Tensor] = {}
        self.backbone_frozen = True

    def freeze_backbone(self, train_classifier: bool = True) -> None:
        self.backbone_frozen = True
        self.base.requires_grad_(False)
        if train_classifier:
            self.base.all_output_layer.requires_grad_(True)
        for module in (self.modality_gate, self.pair_gate, self.capacity_control):
            if module is not None:
                module.requires_grad_(True)

    def unfreeze_backbone(self) -> None:
        """Secondary protocol: jointly fine-tune the complete checkpoint."""
        self.backbone_frozen = False
        self.base.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # The contextual backbone is frozen in the primary protocol. Keeping it
        # in eval mode also prevents dropout from changing teacher features.
        if self.backbone_frozen:
            self.base.eval()
            if self.base.all_output_layer.weight.requires_grad:
                self.base.all_output_layer.train(mode)
        else:
            self.base.train(mode)
        return self

    def _project_inputs(self, textf, visuf, acouf):
        with torch.no_grad():
            text = self.base.textf_input(textf.permute(1, 2, 0)).transpose(1, 2)
            audio = self.base.acouf_input(acouf.permute(1, 2, 0)).transpose(1, 2)
            visual = self.base.visuf_input(visuf.permute(1, 2, 0)).transpose(1, 2)
        return text.detach(), audio.detach(), visual.detach()

    def _learned_gates(self, textf, visuf, acouf, u_mask, qmask):
        if self.modality_gate is None and self.pair_gate is None:
            return None, None
        features = self._project_inputs(textf, visuf, acouf)
        target_summary = torch.cat(features, dim=-1)
        if self.modality_gate is not None:
            return self._force_inactive_identity(self.modality_gate(target_summary)), None
        if self.pair_gate is not None:
            speakers = qmask.argmax(dim=-1)
            pair = self.pair_gate(target_summary, features, speakers, u_mask.bool())
            return None, self._force_inactive_identity(pair)
        return None, None

    def _force_inactive_identity(self, gates: torch.Tensor) -> torch.Tensor:
        """Keep non-selected modalities at exact identity with zero gradient."""
        active = gates.new_zeros(3)
        active[list(self.active_gate_indices)] = 1.0
        return gates * active + (1.0 - active)

    @staticmethod
    def _force_self_identity(edge_gate: torch.Tensor) -> torch.Tensor:
        length = edge_gate.size(1)
        diagonal = torch.eye(length, dtype=torch.bool, device=edge_gate.device)
        return torch.where(diagonal[None, :, :, None], torch.ones_like(edge_gate), edge_gate)

    @staticmethod
    def _conservative_top_k(edge_gate: torch.Tensor, valid: torch.Tensor,
                            top_k: int) -> torch.Tensor:
        """Apply only the K lowest learned gates per target and modality.

        Ranking uses the dense learned gate. Every non-selected, padded, and
        self edge is restored to identity, so this changes the intervention
        budget without retraining the ranking network.
        """
        if top_k <= 0:
            return edge_gate
        length = edge_gate.size(2)
        diagonal = torch.eye(length, dtype=torch.bool, device=edge_gate.device)
        candidates = (valid[:, :, None] & valid[:, None, :]
                      & ~diagonal[None]).unsqueeze(-1).expand_as(edge_gate)
        scores = torch.where(
            candidates, edge_gate, torch.full_like(edge_gate, float("inf"))
        )
        count = min(int(top_k), length)
        indices = torch.topk(scores, count, dim=2, largest=False).indices
        selected = torch.zeros_like(candidates)
        selected.scatter_(2, indices, True)
        selected &= candidates
        return torch.where(selected, edge_gate, torch.ones_like(edge_gate))

    @contextmanager
    def _runtime_hooks(self, edge_gate, late_gate, active_branches, capture_attention):
        handles = []
        self.last_attention = {}
        self.last_enhanced = {}
        if edge_gate is not None:
            edge_gate = self._force_self_identity(edge_gate)
            for branch in BRANCHES:
                if active_branches is not None and branch not in active_branches:
                    continue
                modality = SOURCE_MODALITY[branch]
                softmax = getattr(self.base, branch).transformer_inter[0].self_attn.softmax

                def attention_hook(_module, _inputs, attention, name=branch, m=modality):
                    gate = edge_gate[..., m].unsqueeze(1).to(attention)
                    gated = attention * gate
                    if self.renormalize_attention:
                        gated = gated / gated.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    if capture_attention:
                        self.last_attention[name] = torch.stack(
                            [attention.detach(), gated.detach()], dim=0
                        )
                    return gated

                handles.append(softmax.register_forward_hook(attention_hook))
        if late_gate is not None:
            reducers = (self.base.features_reduce_t, self.base.features_reduce_a,
                        self.base.features_reduce_v)
            for modality, reducer in enumerate(reducers):
                def late_hook(_module, _inputs, enhanced, m=modality):
                    if capture_attention:
                        self.last_enhanced[MODALITIES[m]] = enhanced.detach()
                    return enhanced * late_gate[..., m].unsqueeze(-1).to(enhanced)
                handles.append(reducer.register_forward_hook(late_hook))
        if self.capacity_control is not None:
            def capacity_pre_hook(_module, inputs):
                return (self.capacity_control(inputs[0]),)
            handles.append(self.base.all_output_layer.register_forward_pre_hook(capacity_pre_hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len,
                fixed_edge_gate: torch.Tensor | None = None,
                fixed_late_gate: torch.Tensor | None = None,
                active_branches: Sequence[str] | None = None,
                capture_attention: bool = False,
                pair_gate_top_k: int = 0):
        modality_gate, pair_gate = self._learned_gates(textf, visuf, acouf, u_mask, qmask)
        edge_gate, late_gate = fixed_edge_gate, fixed_late_gate
        if edge_gate is None:
            if self.model_id == "B2" and modality_gate is not None:
                length = u_mask.size(1)
                edge_gate = modality_gate.unsqueeze(2).expand(-1, -1, length, -1)
            elif self.model_id in ("B3", "B4", "B6"):
                edge_gate = self._conservative_top_k(
                    pair_gate, u_mask.bool(), pair_gate_top_k
                )
        if late_gate is None and self.model_id == "B1":
            late_gate = modality_gate
        with self._runtime_hooks(edge_gate, late_gate, active_branches, capture_attention):
            output = self.base(textf, visuf, acouf, u_mask, qmask, dia_len)
        result = dict(output)
        result["reliability_modality_gate"] = modality_gate
        result["reliability_pair_gate"] = pair_gate
        result["edge_gate_applied"] = edge_gate
        result["late_gate_applied"] = late_gate
        result["attention_audit"] = self.last_attention
        result["late_feature_audit"] = self.last_enhanced
        return result

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters()
                   if parameter.requires_grad)

    def estimated_gate_flops(self, conversation_length: int) -> int:
        """Approximate multiply/add count for added modules; excludes frozen SDT."""
        length, hidden, gate_hidden = conversation_length, self.hidden_dim, self.gate_hidden
        if self.model_id in ("B1", "B2"):
            return int(2 * length * (3 * hidden * gate_hidden + gate_hidden * 3))
        if self.model_id in ("B3", "B4", "B6"):
            projections = 3 * length * (3 * hidden * gate_hidden + hidden * gate_hidden)
            pair_scores = 3 * length * length * (gate_hidden + 4)
            return int(2 * (projections + pair_scores))
        if self.model_id == "B5":
            return int(4 * length * hidden * self.capacity_hidden)
        return 0

    def trainable_state_dict(self) -> Mapping[str, torch.Tensor]:
        trainable = {name for name, parameter in self.named_parameters()
                     if parameter.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items()
                if name in trainable}

    def load_trainable_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        current = self.state_dict()
        unknown = sorted(set(state).difference(current))
        if unknown:
            raise ValueError(f"Unknown trainable keys: {unknown}")
        current.update(state)
        self.load_state_dict(current, strict=True)


def source_branches(modality: int) -> tuple[str, ...]:
    return tuple(name for name in BRANCHES if SOURCE_MODALITY[name] == modality)

