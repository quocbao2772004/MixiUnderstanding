#!/usr/bin/env python3
"""Persistent local Qwen2-Audio service for QCES parsing and speech QA."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import re
import threading
import unicodedata
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from mixi_understanding.audioqa_qwen_contract import (
    extract_json,
    grounded_labels_in_question,
    normalize_program,
    qwen_intent_prompt,
    qwen_parser_prompt,
    qwen_speaker_prompt,
    qwen_slot_prompt,
)


DEFAULT_MODEL = (
    "/home/cuongpv/.cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/"
    "snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a"
)


def explicit_speaker_ordinal(question: str) -> int | None:
    """Extract only a literal speaker index; never infer it from context."""

    decomposed = unicodedata.normalize("NFKD", question).casefold()
    folded = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    folded = " ".join(re.findall(r"[a-z0-9]+", folded))
    # An ordinal is a speaker selector only in an explicit speech-content
    # question.  This includes natural Vietnamese such as "người thứ 2 nói
    # gì", where the word "nói" comes after the ordinal rather than directly
    # after "người".
    content_cue = re.search(
        r"\b(?:noi gi|da noi gi|noi nhung gi|noi cai gi|what did|what does|say|said|transcrib)",
        folded,
    )
    if content_cue is None or not re.search(r"(?:nguoi(?: noi)?|speaker)", folded):
        return None
    patterns = {
        1: r"(?:nguoi(?: noi)?|speaker)\s+(?:thu\s+)?(?:1|mot|nhat|dau tien|one|first)\b",
        2: r"(?:nguoi(?: noi)?|speaker)\s+(?:thu\s+)?(?:2|hai|two|second)\b",
        3: r"(?:nguoi(?: noi)?|speaker)\s+(?:thu\s+)?(?:3|ba|three|third)\b",
    }
    for ordinal, pattern in patterns.items():
        if re.search(pattern, folded):
            return ordinal
    return None


class QwenBackend:
    def __init__(self, model_path: str) -> None:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
            quantization_config=quantization,
            dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).eval()
        self.device = next(self.model.parameters()).device
        self.target_sample_rate = int(self.processor.feature_extractor.sampling_rate)
        self.lock = threading.Lock()
        print(f"[ready] Qwen2-Audio on {self.device}; sample_rate={self.target_sample_rate}", flush=True)

    def generate(self, prompt: str, audio_path: str | None = None, max_new_tokens: int = 160) -> str:
        content: list[dict[str, Any]] = []
        audio: np.ndarray | None = None
        if audio_path:
            waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=1)
            if sample_rate != self.target_sample_rate:
                divisor = math.gcd(int(sample_rate), self.target_sample_rate)
                waveform = resample_poly(
                    waveform,
                    self.target_sample_rate // divisor,
                    int(sample_rate) // divisor,
                )
            audio = np.ascontiguousarray(waveform, dtype=np.float32)
            content.append({"type": "audio", "audio": "in_memory"})
        content.append({"type": "text", "text": prompt})
        chat = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        kwargs: dict[str, Any] = {"text": text, "return_tensors": "pt", "padding": True}
        if audio is not None:
            kwargs.update(audio=audio, sampling_rate=self.target_sample_rate)
        prepared = self.processor(**kwargs)
        allowed = {"input_ids", "attention_mask", "input_features", "feature_attention_mask"}
        inputs = {key: value.to(self.device) for key, value in prepared.items() if key in allowed}
        prefix_width = int(inputs["input_ids"].shape[1])
        with self.lock, self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.float16):
            output = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        continuation = output[0, prefix_width:].detach().cpu().tolist()
        return self.processor.tokenizer.decode(continuation, skip_special_tokens=True).strip()

    def parse(self, question: str, labels: list[str]) -> dict[str, Any]:
        raw_intent = self.generate(qwen_intent_prompt(question), max_new_tokens=72)
        intent_payload = extract_json(raw_intent) or {}
        literal_speaker_ordinal = explicit_speaker_ordinal(question)
        intent_value = intent_payload.get("intent")
        if intent_value is None:
            # Qwen occasionally returns {"speech_content": "..."}; this is
            # schema repair only, not lexical parsing of the user question.
            for candidate in (
                "speech_content",
                "speech_before_quote",
                "speech_after_quote",
                "list",
                "exists",
                "locate",
                "count",
                "first",
                "last",
                "longest",
                "before",
                "after",
                "between",
                "overlap",
                "speaker_count",
                "open_audioqa",
                "unsupported",
            ):
                if candidate in intent_payload:
                    intent_value = candidate
                    break
        intent = str(intent_value or "unsupported").strip().lower()
        if literal_speaker_ordinal is not None:
            # This is a schema repair over an explicit user slot, not an
            # acoustic answer rule. Qwen still handles all free-form wording.
            intent = "speech_content"
        needs_slots = intent in {
            "before",
            "after",
            "between",
            "exists",
            "locate",
            "count",
            "overlap",
            "speech_content",
            "speech_before_quote",
            "speech_after_quote",
        }
        raw_slots = ""
        raw_speaker = ""
        slot_payload: dict[str, Any] = {}
        if needs_slots:
            raw_slots = self.generate(
                qwen_slot_prompt(question, intent, labels), max_new_tokens=96
            )
            slot_payload = extract_json(raw_slots) or {}
        speaker_payload: dict[str, Any] = {}
        if intent in {"speech_content", "speech_before_quote", "speech_after_quote"}:
            raw_speaker = self.generate(qwen_speaker_prompt(question), max_new_tokens=32)
            speaker_payload = extract_json(raw_speaker) or {}
            # Speaker gender is an extractive slot.  Accept a gender prediction
            # only when Qwen also points to an exact, non-empty phrase in the
            # user's question.  This blocks context/example-driven gender
            # hallucinations without consulting audio annotations.
            claimed_speaker = str(speaker_payload.get("speaker") or "any").lower()
            speaker_evidence = str(speaker_payload.get("evidence") or "").strip()
            normalized_question = unicodedata.normalize("NFKC", question).casefold()
            normalized_evidence = unicodedata.normalize("NFKC", speaker_evidence).casefold()
            gender_cues = {
                "female": ("woman", "female", "phụ nữ", "cô", "chị"),
                "male": ("man", "male", "đàn ông", "anh", "chú"),
                "child": ("child", "kid", "trẻ em", "em bé"),
            }
            evidence_has_gender_cue = any(
                cue in normalized_evidence
                for cue in gender_cues.get(claimed_speaker, ())
            )
            if claimed_speaker not in {"male", "female", "child"} or not normalized_evidence:
                speaker_payload["speaker"] = "any"
                speaker_payload["evidence"] = None
            elif normalized_evidence not in normalized_question or not evidence_has_gender_cue:
                speaker_payload["speaker"] = "any"
                speaker_payload["evidence"] = None
        raw_labels = slot_payload.get("labels")
        if raw_labels is None:
            for key in ("sound_labels", "sound_label", "sound", "anchor_label"):
                if key in slot_payload:
                    raw_labels = slot_payload[key]
                    break
        if isinstance(raw_labels, dict):
            raw_labels = [raw_labels]
        if not isinstance(raw_labels, list):
            raw_labels = [] if raw_labels is None else [raw_labels]
        flattened_labels: list[Any] = []
        for value in raw_labels:
            if isinstance(value, dict):
                flattened_labels.append(
                    value.get("label") or value.get("sound_label") or value.get("sound")
                )
            else:
                flattened_labels.append(value)
        # For ontology-grounded intents, prefer labels whose bilingual alias is
        # literally present in the question.  This prevents Qwen from filling
        # the unknown answer slot with a hallucinated event class.
        grounded_labels = list(grounded_labels_in_question(question, labels))
        expected_label_count = 2 if intent in {"between", "overlap"} else 1
        if intent in {"before", "after", "between", "exists", "locate", "count", "overlap"}:
            if len(grounded_labels) >= expected_label_count:
                flattened_labels = grounded_labels[:expected_label_count]
            # before/after ask the executor to discover an answer around one
            # anchor.  Any second label emitted by the language model is an
            # ungrounded answer guess and must never enter the executor.
            flattened_labels = flattened_labels[:expected_label_count]
        raw_ordinals = slot_payload.get("ordinals")
        if raw_ordinals is None:
            for key in ("occurrence_numbers", "occurrence_number", "ordinal"):
                if key in slot_payload:
                    raw_ordinals = slot_payload[key]
                    break
        if not isinstance(raw_ordinals, list):
            raw_ordinals = [] if raw_ordinals is None else [raw_ordinals]
        if intent == "between":
            raw_ordinals = raw_ordinals[:2]
        elif intent in {
            "before", "after", "exists", "locate", "count", "overlap", "speech_content"
        }:
            raw_ordinals = raw_ordinals[:1]
        if intent == "speech_content":
            raw_ordinals = (
                [literal_speaker_ordinal]
                if literal_speaker_ordinal is not None
                else []
            )
        payload = {
            "intent": intent,
            "labels": flattened_labels,
            "ordinals": raw_ordinals,
            "speaker": speaker_payload.get("speaker", intent_payload.get("speaker")),
            "quote": (
                slot_payload.get("quote")
                or slot_payload.get("quoted_text")
                or slot_payload.get("anchor_quote")
                or slot_payload.get("text")
            ),
            "confidence": min(
                float(intent_payload.get("confidence", 0.0) or 0.0),
                float(slot_payload.get("confidence", 1.0) or 0.0),
            ),
        }
        program = normalize_program(payload, labels)
        return {
            "program": program.to_dict(),
            "raw": {"intent": raw_intent, "slots": raw_slots, "speaker": raw_speaker},
            "json": {"intent": intent_payload, "slots": slot_payload, "speaker": speaker_payload},
        }

    def speech_answer(self, audio_path: str, question: str) -> dict[str, Any]:
        prompt = (
            "Listen carefully to the supplied audio evidence. The user asks: "
            f"{question}\n"
            "Return only the exact Vietnamese words spoken by the requested speaker. "
            "Do not describe background sounds. If speech is not intelligible, return NONE."
        )
        return {"answer": self.generate(prompt, audio_path=audio_path, max_new_tokens=192)}

    def audio_answer(self, audio_path: str, question: str) -> dict[str, Any]:
        prompt = (
            "Bạn là trợ lý hỏi đáp âm thanh thân thiện. Hãy trả lời câu hỏi của người dùng "
            "bằng 1-2 câu tiếng Việt tự nhiên, xưng là 'mình' và gọi người dùng là 'bạn'. "
            "Chỉ sử dụng thông tin thực sự nghe được trong audio; không suy đoán hoặc bịa thêm "
            "sự kiện và lời nói. Nếu audio không đủ thông tin, hãy nói tự nhiên rằng mình chưa "
            "xác định được từ đoạn audio của bạn. Không nhắc tới model hay quy trình xử lý.\n"
            f"Câu hỏi của bạn: {question}"
        )
        return {"answer": self.generate(prompt, audio_path=audio_path, max_new_tokens=192)}

    def verbalize(
        self,
        question: str,
        verified_answer: str,
        intent: str,
        evidence_spans: list[list[float]],
    ) -> dict[str, Any]:
        """Turn a verified executor result into natural Vietnamese without changing facts."""

        context = {
            "question": question,
            "intent": intent,
            "verified_answer": verified_answer,
            "evidence_spans_seconds": evidence_spans,
        }
        prompt = (
            "VAI TRÒ: Bạn là một trợ lý hỏi đáp âm thanh thân thiện đang trò chuyện trực tiếp "
            "với người dùng. Bạn xưng là 'mình' và gọi người dùng là 'bạn'.\n\n"
            "NHIỆM VỤ: Dựa vào câu hỏi gốc và đáp án đã được kiểm chứng trong JSON, hãy trả lời "
            "như một trợ lý hội thoại thực sự hiểu câu hỏi. Viết 1-2 câu tiếng Việt tự nhiên; "
            "không được chỉ lặp lại một mình nhãn đáp án.\n\n"
            "RÀNG BUỘC:\n"
            "- Trả lời trực tiếp đúng điều người dùng hỏi.\n"
            "- Phải có cách xưng hô 'mình' và/hoặc 'bạn'.\n"
            "- Chuỗi verified_answer phải xuất hiện nguyên văn trong câu trả lời.\n"
            "- Mở đầu trực tiếp bằng nội dung nghe được; không viết 'Đây là câu trả lời', "
            "'Câu trả lời là' hoặc câu dẫn mang tính meta.\n"
            "- Tuyệt đối không thêm, đổi hoặc suy đoán sự kiện, người nói, nội dung hay thời gian.\n"
            "- Không nhắc đến JSON, đáp án kiểm chứng, detector, model, intent hoặc pipeline.\n"
            "- Không dùng markdown, tiêu đề hoặc giải thích quy trình.\n\n"
            "VÍ DỤ PHONG CÁCH:\n"
            "Câu hỏi: Sau lời nói là âm gì? | Đáp án: Tiếng vỗ tay\n"
            "Trả lời: Mình nghe thấy Tiếng vỗ tay ngay sau phần lời nói trong audio của bạn.\n"
            "Câu hỏi: Trong audio có âm thanh gì? | Đáp án: Tiếng còi, tiếng động cơ\n"
            "Trả lời: Trong audio của bạn, mình nghe thấy Tiếng còi, tiếng động cơ.\n\n"
            f"DỮ LIỆU HIỆN TẠI={json.dumps(context, ensure_ascii=False)}\n"
            "Chỉ trả về câu trả lời tự nhiên:"
        )
        answer = self.generate(prompt, max_new_tokens=96).strip().strip('"')
        return {"answer": answer}


class OllamaQwenBackend(QwenBackend):
    """Use the machine's shared Qwen runtime for text routing and wording.

    The acoustic facts still come from the QCES detector, ASR and separation
    services.  This backend intentionally declines the open-ended multimodal
    route because the installed Ollama model is text-only.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.lock = threading.Lock()
        request = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        installed = {str(row.get("name")) for row in payload.get("models", [])}
        if self.model_name not in installed:
            raise RuntimeError(
                f"Ollama model {self.model_name!r} is not installed; available={sorted(installed)}"
            )
        print(f"[ready] Ollama Qwen text backend: {self.model_name}", flush=True)

    def generate(self, prompt: str, audio_path: str | None = None, max_new_tokens: int = 160) -> str:
        if audio_path is not None:
            raise RuntimeError("The configured Ollama Qwen model is text-only")
        body = json.dumps(
            {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": -1,
                "options": {
                    "temperature": 0,
                    "num_predict": int(max_new_tokens),
                    "num_ctx": 4096,
                },
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.lock, urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
        answer = str(payload.get("response") or "").strip()
        if not answer:
            raise RuntimeError(f"Ollama returned an empty response: {payload}")
        return answer

    def speech_answer(self, audio_path: str, question: str) -> dict[str, Any]:
        return {"answer": "NONE", "degraded": True, "reason": "text_only_qwen_backend"}

    def audio_answer(self, audio_path: str, question: str) -> dict[str, Any]:
        return {
            "answer": (
                "Mình chưa xác định được câu hỏi mở này từ đoạn audio của bạn. "
                "Bạn có thể hỏi về tiếng xe, số người nói hoặc nội dung của từng người."
            ),
            "degraded": True,
            "reason": "text_only_qwen_backend",
        }


BACKEND: QwenBackend | OllamaQwenBackend


class Handler(BaseHTTPRequestHandler):
    server_version = "QCESQwen/1.0"

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            if isinstance(BACKEND, OllamaQwenBackend):
                self._send(
                    200,
                    {
                        "ok": True,
                        "backend": "ollama",
                        "model": BACKEND.model_name,
                        "modalities": ["text"],
                    },
                )
            else:
                self._send(
                    200,
                    {
                        "ok": True,
                        "backend": "transformers",
                        "model": "Qwen2-Audio-7B-Instruct",
                        "quantization": "4bit",
                        "modalities": ["text", "audio"],
                    },
                )
        else:
            self._send(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/parse":
                result = BACKEND.parse(str(payload["question"]), [str(x) for x in payload["labels"]])
            elif self.path == "/speech-answer":
                result = BACKEND.speech_answer(str(payload["audio_path"]), str(payload["question"]))
            elif self.path == "/audio-answer":
                result = BACKEND.audio_answer(str(payload["audio_path"]), str(payload["question"]))
            elif self.path == "/verbalize":
                result = BACKEND.verbalize(
                    question=str(payload["question"]),
                    verified_answer=str(payload["verified_answer"]),
                    intent=str(payload.get("intent") or "unknown"),
                    evidence_spans=[
                        [float(span[0]), float(span[1])]
                        for span in payload.get("evidence_spans", [])
                        if isinstance(span, list) and len(span) == 2
                    ],
                )
            else:
                self._send(404, {"error": "not_found"})
                return
            self._send(200, result)
        except Exception as exc:  # pragma: no cover - service diagnostics
            self._send(500, {"error": type(exc).__name__, "message": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qwen2-audio", "ollama"), default="qwen2-audio")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", default="Qwen-2.5-7b-v2:latest")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8511)
    args = parser.parse_args()
    global BACKEND
    if args.backend == "ollama":
        BACKEND = OllamaQwenBackend(args.ollama_url, args.ollama_model)
    else:
        BACKEND = QwenBackend(args.model)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] http://{args.host}:{args.port}", flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
