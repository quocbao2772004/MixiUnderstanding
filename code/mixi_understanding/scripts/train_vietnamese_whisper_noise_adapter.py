#!/usr/bin/env python3
"""Adapt the final Whisper encoder blocks to dense automotive interference.

The decoder and most of the encoder stay frozen.  A frozen copy of the
pretrained encoder supplies clean-speech targets, while the student sees the
paired noisy mixture.  This avoids teaching the language model the small
training transcript inventory and makes the objective explicitly preserve the
clean acoustic representation expected by the original decoder.
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.modeling_outputs import BaseModelOutput

from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import (
    PROJECT_ROOT,
    _atomic_text,
    _portable,
    _resolve,
)
from mixi_understanding.scripts.build_qces_speech_event_v1 import _max_energy_crop, _normalize, _rms
from mixi_understanding.scripts.build_vieneu_automotive_chaos_v1 import CONTINUOUS, TRANSIENT, _pool


RATE = 16_000


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load(path: Path, start: float, end: float) -> np.ndarray:
    wave, source_rate = sf.read(path, dtype="float32", always_2d=True)
    value = torch.from_numpy(wave.mean(axis=1)).float()
    if source_rate != RATE:
        value = AF.resample(value, source_rate, RATE)
    left = max(0, int(math.floor(start * RATE)))
    right = min(value.numel(), int(math.ceil(end * RATE)))
    value = value[left:right]
    rms = value.square().mean().sqrt().clamp_min(1e-8)
    value = torch.clamp(value * min(10.0, float((10 ** (-24 / 20)) / rms)), -1.0, 1.0)
    return value.numpy()


def _words(value: str) -> list[str]:
    return re.findall(r"\w+", str(value).lower(), flags=re.UNICODE)


def _distance(reference: str, hypothesis: str) -> tuple[int, int]:
    left, right = _words(reference), _words(hypothesis)
    previous = list(range(len(right) + 1))
    for index, source in enumerate(left, 1):
        current = [index]
        for column, target in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (source != target)))
        previous = current
    return previous[-1], max(len(left), 1)


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], seed: int) -> None:
        self.rows = rows
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        margin = rng.uniform(0.05, 0.25)
        start = max(0.0, float(row["speech_onset_seconds"]) - margin)
        end = float(row["speech_offset_seconds"]) + margin
        return {
            "noisy": _load(_resolve(row["mixture_path"]), start, end),
            "clean": _load(_resolve(row["target_path"]), start, end),
            "transcript": str(row["transcript"]),
        }


class DynamicPairDataset(Dataset):
    """Create a fresh dense-noise mixture for every item and every epoch."""

    def __init__(
        self,
        speech_rows: list[dict[str, Any]],
        noise_pool: dict[str, list[dict[str, Any]]],
        scenes_per_epoch: int,
        seed: int,
    ) -> None:
        self.speech_rows = speech_rows
        self.noise_pool = noise_pool
        self.scenes_per_epoch = scenes_per_epoch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.scenes_per_epoch

    @staticmethod
    def _speech(row: dict[str, Any]) -> np.ndarray:
        value, source_rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
        wave = torch.from_numpy(value.mean(axis=1)).float()
        if source_rate != RATE:
            wave = AF.resample(wave, source_rate, RATE)
        return _normalize(wave.numpy(), -22.0)

    @staticmethod
    def _event(row: dict[str, Any], seconds: float) -> np.ndarray:
        wave, source_rate = sf.read(_resolve(row["audio_path"]), dtype="float32", always_2d=True)
        value = torch.from_numpy(wave.mean(axis=1)).float()
        if source_rate != RATE:
            value = AF.resample(value, source_rate, RATE)
        start = max(0, int(float(row.get("active_onset_seconds", 0.0)) * RATE))
        end = min(value.numel(), int(float(row.get("active_offset_seconds", value.numel() / RATE)) * RATE))
        active = value[start:end].numpy() if end > start else value.numpy()
        return _max_energy_crop(active, seconds)

    @staticmethod
    def _fit(wave: np.ndarray, frames: int, rng: random.Random) -> np.ndarray:
        if len(wave) >= frames:
            maximum = len(wave) - frames
            start = rng.randint(0, maximum) if maximum else 0
            return wave[start:start + frames].copy()
        return np.tile(wave, int(math.ceil(frames / max(len(wave), 1))))[:frames].astype(np.float32)

    @staticmethod
    def _place(wave: np.ndarray, frames: int, rng: random.Random) -> np.ndarray:
        result = np.zeros(frames, dtype=np.float32)
        if len(wave) >= frames:
            result[:] = DynamicPairDataset._fit(wave, frames, rng)
        else:
            start = rng.randint(0, frames - len(wave))
            result[start:start + len(wave)] = wave
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.seed + self.epoch * 10_000_019 + index)
        # The coprime stride covers the speech bank before repeating it.
        speech_row = self.speech_rows[(index * 131 + self.epoch * 17) % len(self.speech_rows)]
        clean = self._speech(speech_row); frames = len(clean)
        continuous_labels = rng.sample(list(CONTINUOUS), rng.randint(2, min(4, len(CONTINUOUS))))
        transient_labels = rng.sample(list(TRANSIENT), rng.randint(3, min(7, len(TRANSIENT))))
        components: list[np.ndarray] = []
        for label in continuous_labels:
            source = rng.choice(self.noise_pool[label])
            components.append(self._fit(self._event(source, frames / RATE), frames, rng))
        for label in transient_labels:
            source = rng.choice(self.noise_pool[label])
            event = self._event(source, rng.uniform(0.45, min(2.5, max(0.5, frames / RATE))))
            components.append(self._place(event, frames, rng))
        weights = [10 ** (rng.uniform(-6.0, 3.0) / 20.0) for _ in components]
        interference = sum((weight * value for weight, value in zip(weights, components)), np.zeros(frames, dtype=np.float32))
        target_snr = rng.uniform(-15.0, 5.0)
        interference *= (_rms(clean) / (10 ** (target_snr / 20.0))) / max(_rms(interference), 1e-7)
        noisy = clean + interference
        peak = max(float(np.max(np.abs(noisy))), float(np.max(np.abs(clean))), 1e-8)
        scale = min(1.0, 0.97 / peak)
        return {
            "noisy": (noisy * scale).astype(np.float32),
            "clean": (clean * scale).astype(np.float32),
            "transcript": str(speech_row["transcription"]),
        }


def _dynamic_training_data(
    fleurs_path: Path,
    source_bank_root: Path,
    eval_scenes: list[dict[str, Any]],
    *,
    maximum_duration: float,
    maximum_words: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    speech_rows = pq.read_table(
        fleurs_path,
        columns=["id", "num_samples", "audio", "transcription"],
    ).to_pylist()
    speech_rows = [
        row for row in speech_rows
        if row["audio"]["bytes"] and 1.0 <= int(row["num_samples"]) / RATE <= maximum_duration
        and 2 <= len(str(row["transcription"]).split()) <= maximum_words
    ]
    excluded = {
        str(event["source_id"])
        for scene in eval_scenes for event in scene["events"]
        if event["label"] != "Speech"
    }
    raw_pool = _pool(source_bank_root)
    labels = set(CONTINUOUS) | set(TRANSIENT)
    noise_pool = {
        label: [row for row in raw_pool[label] if str(row.get("source_id")) not in excluded]
        for label in labels
    }
    missing = {label: len(rows) for label, rows in noise_pool.items() if not rows}
    if missing:
        raise RuntimeError(f"empty leakage-free dynamic noise classes: {missing}")
    audit = {
        "speech_recordings": len(speech_rows),
        "speech_transcripts": len({str(row["transcription"]) for row in speech_rows}),
        "excluded_eval_noise_sources": len(excluded),
        "train_noise_sources_by_class": {label: len(rows) for label, rows in sorted(noise_pool.items())},
    }
    return speech_rows, noise_pool, audit


def _collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "noisy": [item["noisy"] for item in items],
        "clean": [item["clean"] for item in items],
        "samples": [len(item["noisy"]) for item in items],
        "transcripts": [item["transcript"] for item in items],
    }


def _scene_audio(scene: dict[str, Any], clean: bool = False) -> tuple[np.ndarray, str]:
    speech = next(event for event in scene["events"] if event["label"] == "Speech")
    path = _resolve(speech["stem_path"] if clean else scene["mixture_path"])
    return _load(path, max(0.0, float(speech["onset_seconds"]) - 0.15), float(speech["offset_seconds"]) + 0.15), str(speech["transcript"])


@torch.inference_mode()
def _evaluate(
    model: torch.nn.Module,
    processor: Any,
    scenes: list[dict[str, Any]],
    *,
    batch_size: int,
    clean: bool = False,
) -> dict[str, Any]:
    model.eval()
    edits = words = correct = 0
    records: list[dict[str, Any]] = []
    for start in range(0, len(scenes), batch_size):
        chosen = scenes[start:start + batch_size]
        examples = [_scene_audio(scene, clean=clean) for scene in chosen]
        features = processor([audio for audio, _ in examples], sampling_rate=RATE, return_tensors="pt").input_features.cuda().half()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            output = model.generate(
                features,
                language="vi",
                task="transcribe",
                max_new_tokens=96,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
            )
        hypotheses = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for scene, (_, reference), hypothesis in zip(chosen, examples, hypotheses):
            distance, count = _distance(reference, hypothesis.strip())
            edits += distance
            words += count
            correct += int(distance / count <= 0.25)
            records.append({
                "scene_id": scene["scene_id"], "split": scene["split"], "reference": reference,
                "hypothesis": hypothesis.strip(), "edit_distance": distance, "reference_words": count,
                "wer_↓": distance / count,
            })
    return {
        "utterances": len(records), "corpus_wer_↓": edits / max(words, 1),
        "accuracy_at_wer_0.25_↑": correct / max(len(records), 1), "total_edits": edits,
        "reference_words": words, "records": records,
    }


def _trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if dict(model.named_parameters()).get(name) is not None and dict(model.named_parameters())[name].requires_grad}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_enhancement_train_v1")
    parser.add_argument("--eval-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_chaos_v1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_whisper_noise_adapter_v1")
    parser.add_argument("--model", default=str(Path.home() / ".cache/huggingface/hub/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--unfrozen-encoder-blocks", type=int, default=2)
    parser.add_argument("--identity-probability", type=float, default=0.2)
    parser.add_argument("--feature-loss-weight", type=float, default=1.0)
    parser.add_argument("--asr-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026081017)
    parser.add_argument("--max-train-scenes", type=int, default=None)
    parser.add_argument("--dynamic-fleurs", type=Path, default=None)
    parser.add_argument("--source-bank-root", type=Path, default=PROJECT_ROOT / "outputs/qces_full200_adaptive_v1/primary_clean")
    parser.add_argument("--dynamic-scenes-per-epoch", type=int, default=1000)
    parser.add_argument("--dynamic-max-duration", type=float, default=20.0)
    parser.add_argument("--dynamic-max-words", type=int, default=70)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.set_prefix_tokens(language="vi", task="transcribe", predict_timestamps=False)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, local_files_only=True, dtype=torch.float16).cuda()
    encoder = model.model.encoder
    teacher = copy.deepcopy(encoder).cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    blocks = encoder.layers[-args.unfrozen_encoder_blocks:]
    for block in blocks:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
        # Keep optimizer/master gradients in FP32 while autocast performs the
        # expensive matmuls in FP16. Frozen blocks remain FP16.
        block.float()
    for parameter in encoder.layer_norm.parameters():
        parameter.requires_grad_(True)
    encoder.layer_norm.float()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    print(json.dumps({"trainable_parameters": sum(p.numel() for _, p in trainable), "tensors": len(trainable)}), flush=True)

    scenes = _jsonl(args.eval_dir.resolve() / "scenes.jsonl")
    validation = [scene for scene in scenes if scene["split"] == "val"]
    test = [scene for scene in scenes if scene["split"] == "test"]
    if args.dynamic_fleurs is not None:
        speech_rows, noise_pool, data_audit = _dynamic_training_data(
            args.dynamic_fleurs.resolve(), args.source_bank_root.resolve(), scenes,
            maximum_duration=args.dynamic_max_duration, maximum_words=args.dynamic_max_words,
        )
        dataset = DynamicPairDataset(speech_rows, noise_pool, args.dynamic_scenes_per_epoch, args.seed)
    else:
        rows = _jsonl(args.train_dir.resolve() / "train.jsonl")
        if args.max_train_scenes is not None:
            rows = rows[:args.max_train_scenes]
        dataset = PairDataset(rows, args.seed)
        data_audit = {"fixed_train_scenes": len(rows)}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=_collate, pin_memory=True)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=args.learning_rate, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda")

    baseline = _evaluate(model, processor, validation, batch_size=4)
    history: list[dict[str, Any]] = [{"epoch": 0, "train_loss": None, "validation": {key: value for key, value in baseline.items() if key != "records"}}]
    best_wer = float(baseline["corpus_wer_↓"]); best_epoch = 0
    torch.save({"epoch": 0, "state": _trainable_state(model), "validation": history[-1]["validation"]}, output / "best.pt")
    print(json.dumps(history[-1], ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch); model.train(); encoder.train(); teacher.eval(); optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []; feature_losses: list[float] = []; asr_losses: list[float] = []
        for step, batch in enumerate(loader, 1):
            noisy = list(batch["noisy"])
            clean = list(batch["clean"])
            # Identity examples prevent the adapter from modifying already clean speech.
            local_rng = random.Random(args.seed + epoch * 10_007 + step)
            student_audio = [clean_item if local_rng.random() < args.identity_probability else noisy_item for noisy_item, clean_item in zip(noisy, clean)]
            student_features = processor(student_audio, sampling_rate=RATE, return_tensors="pt").input_features.cuda().half()
            clean_features = processor(clean, sampling_rate=RATE, return_tensors="pt").input_features.cuda().half()
            with torch.inference_mode():
                clean_hidden = teacher(clean_features).last_hidden_state
            with torch.amp.autocast("cuda", dtype=torch.float16):
                noisy_hidden = encoder(student_features).last_hidden_state
                valid_frames = torch.tensor([(samples + 319) // 320 for samples in batch["samples"]], device="cuda")
                time = torch.arange(noisy_hidden.shape[1], device="cuda")[None]
                mask = (time < valid_frames[:, None]).unsqueeze(-1)
                left = F.layer_norm(noisy_hidden.float(), (noisy_hidden.shape[-1],))
                right = F.layer_norm(clean_hidden.float(), (clean_hidden.shape[-1],))
                feature_loss = F.smooth_l1_loss(left[mask.expand_as(left)], right[mask.expand_as(right)], beta=0.1)
                if args.asr_loss_weight > 0:
                    encoded = processor.tokenizer(batch["transcripts"], padding=True, return_tensors="pt")
                    labels = encoded.input_ids
                    label_mask = encoded.attention_mask
                    if torch.all(labels[:, 0] == model.config.decoder_start_token_id):
                        labels, label_mask = labels[:, 1:], label_mask[:, 1:]
                    labels = labels.masked_fill(label_mask.ne(1), -100).cuda()
                    asr_loss = model(
                        encoder_outputs=BaseModelOutput(last_hidden_state=noisy_hidden),
                        labels=labels,
                        use_cache=False,
                    ).loss.float()
                else:
                    asr_loss = feature_loss.new_zeros(())
                loss = args.feature_loss_weight * feature_loss + args.asr_loss_weight * asr_loss
                loss = loss / args.gradient_accumulation
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * args.gradient_accumulation)
            feature_losses.append(float(feature_loss.detach().cpu()))
            asr_losses.append(float(asr_loss.detach().cpu()))
            if step % args.gradient_accumulation == 0 or step == len(loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if step % 25 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "steps": len(loader), "loss": float(np.mean(losses[-25:])), "feature_loss": float(np.mean(feature_losses[-25:])), "asr_loss": float(np.mean(asr_losses[-25:]))}), flush=True)
        metrics = _evaluate(model, processor, validation, batch_size=4)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_feature_loss": float(np.mean(feature_losses)), "train_asr_loss": float(np.mean(asr_losses)), "validation": {key: value for key, value in metrics.items() if key != "records"}}
        history.append(record); print(json.dumps(record, ensure_ascii=False), flush=True)
        torch.save({"epoch": epoch, "state": _trainable_state(model), "validation": record["validation"]}, output / "last.pt")
        if float(metrics["corpus_wer_↓"]) < best_wer:
            best_wer = float(metrics["corpus_wer_↓"]); best_epoch = epoch
            torch.save({"epoch": epoch, "state": _trainable_state(model), "validation": record["validation"]}, output / "best.pt")
        _atomic_text(output / "history.json", json.dumps(history, ensure_ascii=False, indent=2) + "\n")

    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["state"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected adapter keys: {unexpected}")
    test_metrics = _evaluate(model, processor, test, batch_size=4)
    clean_metrics = _evaluate(model, processor, test, batch_size=4, clean=True)
    records_path = output / "test_items.jsonl"
    _atomic_text(records_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in test_metrics.pop("records")))
    clean_metrics.pop("records")
    receipt = {
        "format": "qces_vietnamese_whisper_noise_adapter_receipt_v1", "complete": True,
        "method": "clean-teacher representation consistency on final Whisper encoder blocks",
        "base_model": args.model, "train_scenes_per_epoch": len(dataset), "training_data_audit": data_audit,
        "dynamic_remix": args.dynamic_fleurs is not None, "unfrozen_encoder_blocks": args.unfrozen_encoder_blocks,
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable), "identity_probability": args.identity_probability,
        "feature_loss_weight": args.feature_loss_weight, "asr_loss_weight": args.asr_loss_weight,
        "selection_protocol": "best epoch selected on chaos validation WER; test evaluated once after locking",
        "baseline_validation": history[0]["validation"], "best_epoch": best_epoch,
        "best_validation": checkpoint["validation"], "locked_test_mixture": test_metrics,
        "locked_test_clean": clean_metrics, "history": _portable(output / "history.json"),
        "adapter_checkpoint": _portable(output / "best.pt"), "test_items": _portable(records_path),
    }
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
