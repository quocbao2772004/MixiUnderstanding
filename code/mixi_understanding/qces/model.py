"""End-to-end QCES model assembly and checkpoint helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn as nn

from mixi_understanding.qces.composer import PromptComposition, RoleAwarePromptComposer
from mixi_understanding.qces.config import (
    AUDIOSEP_CLAP_FOUNDATION_FEATURES,
    DUAL_ROLE_SEMANTIC_MODE,
    QCESConfig,
)
from mixi_understanding.qces.separators import (
    AudioSepConditionedAdapter,
    ComplementaryMaskSeparator,
    PhaseAwareComplexMaskSeparator,
    SeparationOutput,
)
from mixi_understanding.qces.tokenization import StableHashTokenizer


@dataclass
class QCESOutput:
    composition: PromptComposition
    separation: SeparationOutput

    @property
    def evidence(self) -> torch.Tensor:
        return self.separation.evidence

    @property
    def residual(self) -> torch.Tensor:
        return self.separation.residual


class QCESModel(nn.Module):
    def __init__(self, config: QCESConfig, separator: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.composer = RoleAwarePromptComposer(config)
        self.separator = separator or ComplementaryMaskSeparator(config)
        if (
            config.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE
            and not isinstance(self.separator, AudioSepConditionedAdapter)
        ):
            raise ValueError(
                "dual_role semantic separation requires the frozen "
                "AudioSepConditionedAdapter backend"
            )

    def forward(
        self,
        mixture: torch.Tensor,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        *,
        question_clap: torch.Tensor | None = None,
        scene_clap: torch.Tensor | None = None,
    ) -> QCESOutput:
        composition = self.composer(
            mixture,
            question_ids,
            question_mask,
            question_clap=question_clap,
            scene_clap=scene_clap,
        )
        separation = self.separator(mixture, composition)
        if separation.refined_composition is not None:
            composition = separation.refined_composition
        return QCESOutput(composition=composition, separation=separation)

    def forward_questions(
        self,
        mixture: torch.Tensor,
        questions: Sequence[str],
        tokenizer: StableHashTokenizer | None = None,
        *,
        question_clap: torch.Tensor | None = None,
        scene_clap: torch.Tensor | None = None,
    ) -> QCESOutput:
        foundation_enabled = (
            self.config.foundation_feature_mode
            == AUDIOSEP_CLAP_FOUNDATION_FEATURES
        )
        if foundation_enabled and (question_clap is None or scene_clap is None):
            raise RuntimeError(
                "audiosep_clap inference requires frozen question_clap and "
                "scene_clap features"
            )
        if not foundation_enabled and (
            question_clap is not None or scene_clap is not None
        ):
            raise RuntimeError(
                "foundation features were supplied to a checkpoint whose "
                "foundation_feature_mode is 'none'"
            )
        tokenizer = tokenizer or StableHashTokenizer(
            vocab_size=self.config.vocab_size,
            max_length=self.config.max_question_tokens,
        )
        tokens = tokenizer.batch_encode(questions, device=mixture.device)
        return self(
            mixture,
            tokens.input_ids,
            tokens.attention_mask,
            question_clap=question_clap,
            scene_clap=scene_clap,
        )

    def checkpoint_payload(
        self, backend: str = "mask", extra: Dict[str, object] | None = None
    ) -> Dict[str, object]:
        if backend == "audiosep":
            if not isinstance(self.separator, AudioSepConditionedAdapter):
                raise ValueError(
                    "audiosep checkpoint requires AudioSepConditionedAdapter"
                )
            if any(
                parameter.requires_grad
                for parameter in self.separator.ss_model.parameters()
            ):
                raise ValueError(
                    "AudioSep checkpoint export requires a frozen backbone"
                )
            payload: Dict[str, object] = {
                "format": "qces_v1",
                "config": self.config.to_dict(),
                "backend": backend,
                "composer_state_dict": self.composer.state_dict(),
                "extra": extra or {},
            }
            refiner = self.separator.separator_aware_refiner
            if self.config.separator_aware_refiner:
                if refiner is None:
                    raise ValueError(
                        "config enables separator-aware refiner but adapter has none"
                    )
                if (
                    getattr(refiner, "mode", None)
                    != self.config.separator_aware_refiner_mode
                ):
                    raise ValueError(
                        "adapter refiner mode differs from checkpoint config"
                    )
                payload["separator_aware_refiner_state_dict"] = (
                    refiner.state_dict()
                )
            elif refiner is not None:
                raise ValueError(
                    "adapter has separator-aware refiner but config disables it"
                )
            return payload
        return {
            "format": "qces_v1",
            "config": self.config.to_dict(),
            "backend": backend,
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }


def load_mask_checkpoint(
    payload: Dict[str, object], map_location: torch.device | str = "cpu"
) -> QCESModel:
    if payload.get("format") != "qces_v1":
        raise ValueError("unsupported QCES checkpoint format")
    if payload.get("backend") != "mask":
        raise ValueError("load_mask_checkpoint only supports the mask backend")
    config_payload = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config_payload, dict) or not isinstance(state_dict, dict):
        raise ValueError("checkpoint is missing config or state_dict")
    config = QCESConfig.from_dict(config_payload)
    model = QCESModel(config)
    model.load_state_dict(state_dict)
    return model.to(map_location)


def load_qces_checkpoint(
    payload: Dict[str, object],
    map_location: torch.device | str = "cpu",
    audiosep_repository_root: str | None = None,
    audiosep_config_path: str | None = None,
    audiosep_checkpoint_path: str | None = None,
) -> QCESModel:
    """Load a QCES checkpoint and, when needed, its frozen AudioSep backbone."""

    if payload.get("format") != "qces_v1":
        raise ValueError("unsupported QCES checkpoint format")
    backend = payload.get("backend")
    if backend == "mask":
        return load_mask_checkpoint(payload, map_location=map_location)
    if backend == "audiosep":
        paths = (
            audiosep_repository_root,
            audiosep_config_path,
            audiosep_checkpoint_path,
        )
        if any(path is None for path in paths):
            raise ValueError(
                "loading an audiosep QCES checkpoint requires repository, "
                "config, and separator checkpoint paths"
            )
        config_payload = payload.get("config")
        composer_state = payload.get("composer_state_dict")
        if not isinstance(config_payload, dict) or not isinstance(
            composer_state, dict
        ):
            raise ValueError("AudioSep checkpoint is missing config or composer")
        config = QCESConfig.from_dict(config_payload)
        separator = AudioSepConditionedAdapter.from_repository(
            repository_root=Path(audiosep_repository_root),
            config_path=Path(audiosep_config_path),
            checkpoint_path=Path(audiosep_checkpoint_path),
            device=map_location,
            freeze_separator=True,
            qces_config=config,
        )
        model = QCESModel(config, separator=separator)
        model.composer.load_state_dict(composer_state)
        refiner_state = payload.get("separator_aware_refiner_state_dict")
        if config.separator_aware_refiner:
            if not isinstance(refiner_state, dict):
                raise ValueError(
                    "AudioSep checkpoint enables the separator-aware refiner "
                    "but is missing its state"
                )
            refiner = separator.separator_aware_refiner
            if refiner is None:
                raise RuntimeError("separator-aware refiner was not constructed")
            if (
                getattr(refiner, "mode", None)
                != config.separator_aware_refiner_mode
            ):
                raise RuntimeError(
                    "constructed refiner mode differs from checkpoint config"
                )
            refiner.load_state_dict(refiner_state)
        elif refiner_state is not None:
            raise ValueError(
                "AudioSep checkpoint contains refiner state but config disables it"
            )
        return model.to(map_location)
    if backend != "complex":
        raise ValueError(f"unsupported standalone QCES backend: {backend}")
    config_payload = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config_payload, dict) or not isinstance(state_dict, dict):
        raise ValueError("checkpoint is missing config or state_dict")
    config = QCESConfig.from_dict(config_payload)
    model = QCESModel(config, separator=PhaseAwareComplexMaskSeparator(config))
    model.load_state_dict(state_dict)
    return model.to(map_location)
