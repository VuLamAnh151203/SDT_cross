"""Checkpoint-compatible implementation of the original SDT model.

The network and objective-facing outputs are the deterministic SDT baseline.
The dictionary return format is intentionally retained because the
cross-utterance experiments consume logits, probabilities, and intermediate
modality representations by name.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sdt_backbone import SDTBackbone


MODALITIES = ("text", "audio", "visual")
FUSION_VARIANTS = ("sdt",)
FUSION_FEATURE_SOURCES = ("enhanced",)


class Transformer_Based_Model(SDTBackbone):
    """Original SDT with a stable experiment-facing output contract.

    Extra keyword arguments are accepted only so that deterministic SDT
    checkpoints produced by the earlier research branch remain loadable.  No
    appraisal, COLD, TiCAL, or emotion-wheel modules are constructed.
    """

    def __init__(
        self,
        dataset,
        temp,
        D_text,
        D_visual,
        D_audio,
        n_head,
        n_classes,
        hidden_dim,
        n_speakers,
        dropout,
        fusion_variant="sdt",
        fusion_feature_source="enhanced",
        **compatibility_options,
    ):
        if fusion_variant != "sdt":
            raise ValueError("SDT_cross contains only the original 'sdt' model")
        if fusion_feature_source != "enhanced":
            raise ValueError("Original SDT requires enhanced modality fusion")
        unsupported = {
            key: value
            for key, value in compatibility_options.items()
            if (key == "use_tical" and bool(value))
            or (key == "use_emotion_wheel" and bool(value))
            or (key == "appraisal_mode" and value not in (None, "none"))
        }
        if unsupported:
            raise ValueError(
                "Checkpoint enables non-SDT extensions: {}".format(
                    ", ".join(sorted(unsupported))
                )
            )
        super().__init__(
            dataset,
            temp,
            D_text,
            D_visual,
            D_audio,
            n_head,
            n_classes,
            hidden_dim,
            n_speakers,
            dropout,
        )
        self.fusion_variant = "sdt"
        self.fusion_feature_source = "enhanced"

    def set_tical_epoch(self, _epoch):
        """Compatibility no-op used by shared experiment runners."""

    def update_tical_anchors(self, *_args, **_kwargs):
        """Compatibility no-op; original SDT has no anchor memory."""
        return 0

    def forward(self, textf, visuf, acouf, u_mask, qmask, dia_len):
        enhanced = self.encode_modalities(
            textf, visuf, acouf, u_mask, qmask, dia_len
        )
        features = torch.stack(enhanced, dim=-2)

        # This is algebraically identical to Multimodal_GatedFusion.forward.
        fusion_weights = torch.softmax(self.last_gate.fc(features), dim=-2)
        fused = (fusion_weights * features).sum(dim=-2)

        student_logits = {
            name: classifier(enhanced[index])
            for index, (name, classifier) in enumerate(
                zip(
                    MODALITIES,
                    (
                        self.t_output_layer,
                        self.a_output_layer,
                        self.v_output_layer,
                    ),
                )
            )
        }
        logits = self.all_output_layer(fused)
        return {
            "logits": logits,
            "log_prob": F.log_softmax(logits, dim=-1),
            "prob": F.softmax(logits, dim=-1),
            "student_logits": student_logits,
            "student_log_prob": {
                name: F.log_softmax(value, dim=-1)
                for name, value in student_logits.items()
            },
            "student_kl_log_prob": {
                name: F.log_softmax(value / self.temp, dim=-1)
                for name, value in student_logits.items()
            },
            "teacher_kl_prob": F.softmax(logits / self.temp, dim=-1),
            "enhanced": features,
            "pure": None,
            "fusion_feature_source": "enhanced",
            "fusion_features": features,
            "distributions": {},
            "latents": features,
            "variance_norm": None,
            "confidence": None,
            "reliability_logits": None,
            "reliability": None,
            "sdt_weights": fusion_weights,
            "fusion_weights": fusion_weights,
            "tical": None,
            "wheel_uncertainty": None,
            "wheel_uncertainty_weights": None,
            "wheel_uncertainty_scales": None,
        }
