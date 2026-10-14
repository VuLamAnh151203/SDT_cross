"""The original masked SDT self-distillation objective."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MODALITIES
from sdt_backbone import MaskedKLDivLoss, MaskedNLLLoss


class SDTLoss(nn.Module):
    def __init__(self, class_weights=None, gamma_1=1.0, gamma_2=1.0,
                 gamma_3=1.0):
        super().__init__()
        self.gamma_1 = float(gamma_1)
        self.gamma_2 = float(gamma_2)
        self.gamma_3 = float(gamma_3)
        self.ce = MaskedNLLLoss(class_weights)
        self.kl = MaskedKLDivLoss()

    def forward(self, outputs, labels, valid_mask, *_unused):
        n_classes = outputs["logits"].size(-1)
        flat_labels = labels.reshape(-1)
        task = self.ce(
            outputs["log_prob"].reshape(-1, n_classes),
            flat_labels,
            valid_mask,
        )
        students = sum(
            self.ce(
                outputs["student_log_prob"][name].reshape(-1, n_classes),
                flat_labels,
                valid_mask,
            )
            for name in MODALITIES
        )
        teacher = outputs["teacher_kl_prob"].reshape(-1, n_classes)
        distillation = sum(
            self.kl(
                outputs["student_kl_log_prob"][name].reshape(-1, n_classes),
                teacher,
                valid_mask,
            )
            for name in MODALITIES
        )
        total = (
            self.gamma_1 * task
            + self.gamma_2 * students
            + self.gamma_3 * distillation
        )
        valid = valid_mask.bool()
        targets = labels[valid]
        errors = {
            name: F.cross_entropy(
                outputs["student_logits"][name][valid], targets, reduction="none"
            )
            for name in MODALITIES
        }
        parts = {
            "total": total,
            "sdt": total,
            "task": task,
            "student_ce": students,
            "distillation": distillation,
        }
        return total, parts, errors
