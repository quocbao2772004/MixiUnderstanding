#!/usr/bin/env python3
"""Phi-4 MM inference wrapper for ``evaluate_qces_audioqa.py``.

The local Phi-4 Multimodal remote code calls ``set_lora_adapter('speech')`` on
every forward pass.  In the installed PEFT version, ``BaseTunerLayer.set_adapter``
marks the selected adapter trainable via ``requires_grad_(True)``.  That is
unnecessary for frozen evaluation and fails for k-bit quantized layers because
some tensors are non-floating-point.

This wrapper applies an inference-only monkey patch before importing the shared
QCES audio-QA evaluator.  It does not change the evaluator or model weights.
"""

from __future__ import annotations


def patch_peft_set_adapter_for_quantized_inference() -> None:
    from peft.tuners.tuners_utils import BaseTunerLayer

    def set_adapter_inference_only(self, adapter_names):  # type: ignore[no-untyped-def]
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for _key, layer in module_dict.items():
                # In frozen inference the active adapter must be selected, but
                # no adapter needs trainable parameters.  This avoids PEFT
                # calling requires_grad_(True) on k-bit tensors.
                layer.requires_grad_(False)
        self._active_adapter = adapter_names

    BaseTunerLayer.set_adapter = set_adapter_inference_only  # type: ignore[method-assign]


def main() -> None:
    patch_peft_set_adapter_for_quantized_inference()
    from mixi_understanding.scripts.evaluate_qces_audioqa import main as shared_main

    shared_main()


if __name__ == "__main__":
    main()
