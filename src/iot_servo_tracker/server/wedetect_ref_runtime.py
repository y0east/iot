"""Runtime adapter for the official WeDetect-Ref repository.

This module is intentionally imported only by explicit production validation or
production configuration. It expects the official WeChatCV/WeDetect repository
to be available through the ``WEDETECT_REPO`` environment variable or already
importable on ``PYTHONPATH``.
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iot_servo_tracker.common.packets import TrackingResult
from iot_servo_tracker.common.timebase import now_us


_RUNTIME: "_WeDetectRefRuntime | None" = None
_RUNTIME_KEY: tuple[str, str, str, str, str, int] | None = None


def detect(
    frame_bytes: bytes,
    query: str,
    ts_req: int,
    wedetect_ref_model_dir: str,
    wedetect_uni_checkpoint: str,
    device: str = "cuda:0",
) -> dict[str, Any] | TrackingResult:
    """Run official WeDetect-Ref inference and return a TrackingResult payload.

    Environment knobs for RTX laptop validation:

    - ``WEDETECT_REPO``: path to a cloned ``WeChatCV/WeDetect`` repository.
    - ``WEDETECT_ATTN_IMPLEMENTATION``: defaults to ``sdpa`` for Windows/RTX3060.
    - ``WEDETECT_DTYPE``: ``auto``, ``float16``, ``bfloat16``, or ``float32``.
    - ``WEDETECT_NUM_PROPOSALS``: proposal count passed to SimpleYOLOWorldDetector.
    - ``WEDETECT_SCORE_THRE``: negative value means official top-1 behavior.
    """

    repo = _prepare_official_repo()
    attn_implementation = os.environ.get("WEDETECT_ATTN_IMPLEMENTATION", "sdpa")
    dtype_name = os.environ.get("WEDETECT_DTYPE", "auto")
    num_proposals = int(os.environ.get("WEDETECT_NUM_PROPOSALS", "100"))
    score_threshold = float(os.environ.get("WEDETECT_SCORE_THRE", "-1"))

    key = (
        str(Path(wedetect_ref_model_dir).resolve()),
        str(Path(wedetect_uni_checkpoint).resolve()),
        device,
        attn_implementation,
        dtype_name,
        num_proposals,
    )
    global _RUNTIME, _RUNTIME_KEY
    if _RUNTIME is None or _RUNTIME_KEY != key:
        _RUNTIME = _WeDetectRefRuntime(
            repo=repo,
            wedetect_ref_model_dir=wedetect_ref_model_dir,
            wedetect_uni_checkpoint=wedetect_uni_checkpoint,
            device=device,
            attn_implementation=attn_implementation,
            dtype_name=dtype_name,
            num_proposals=num_proposals,
        )
        _RUNTIME_KEY = key

    return _RUNTIME.detect(frame_bytes, query, ts_req, score_threshold)


def _prepare_official_repo() -> Path:
    repo_env = os.environ.get("WEDETECT_REPO", "").strip()
    if repo_env:
        repo = Path(repo_env).expanduser().resolve()
        if not (repo / "infer_wedetect_ref.py").exists():
            raise RuntimeError(
                f"WEDETECT_REPO does not look like WeChatCV/WeDetect: {repo}"
            )
        repo_text = str(repo)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        return repo
    return Path.cwd()


@dataclass
class _WeDetectRefRuntime:
    repo: Path
    wedetect_ref_model_dir: str
    wedetect_uni_checkpoint: str
    device: str
    attn_implementation: str
    dtype_name: str
    num_proposals: int

    def __post_init__(self) -> None:
        try:
            import torch
            from generate_proposal import SimpleYOLOWorldDetector
            from transformers import AutoProcessor
            from wedetect_ref.models.qwen3vl_referring import (
                Qwen3VLGroundingForConditionalGeneration,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Official WeDetect imports failed. Set WEDETECT_REPO to a cloned "
                "WeChatCV/WeDetect repository and install its WeDetect-Ref dependencies."
            ) from exc

        self.torch = torch
        self.AutoProcessor = AutoProcessor
        self.GroundingModel = Qwen3VLGroundingForConditionalGeneration
        self.det_model = self._load_detector(SimpleYOLOWorldDetector)
        self.model, self.processor, self.object_token_index = self._load_ref_model()

    def detect(
        self,
        frame_bytes: bytes,
        query: str,
        ts_req: int,
        score_threshold: float,
    ) -> dict[str, Any]:
        from PIL import Image
        from wedetect_ref.models.vision_process import process_vision_info

        suffix = _image_suffix(frame_bytes)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fp:
            fp.write(frame_bytes)
            image_path = fp.name

        try:
            with self.torch.no_grad():
                outputs = self.det_model([image_path])
                proposal_boxes = outputs[0]["bboxes"].float().cpu().tolist()
            if not proposal_boxes:
                return TrackingResult.empty(ts_req=ts_req, query=query)

            image = Image.open(image_path).convert("RGB")
            proposal_str = "<object>" * len(proposal_boxes)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": copy.deepcopy(image)},
                        {
                            "type": "text",
                            "text": f'Please detect the "{query}" in the image',
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": proposal_str}],
                },
            ]
            image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
            text = [self.processor.apply_chat_template(messages, tokenize=False)]
            model_inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                do_resize=False,
            ).to(self.device)
            proposals = [
                self.torch.tensor(proposal_boxes, device=self.device).to(self.model.dtype)
            ]
            with self.torch.inference_mode():
                pred = self.model(
                    **model_inputs,
                    bboxes=copy.deepcopy(proposals),
                    ori_shapes=[image.size],
                    bboxes_id=self.object_token_index,
                    image_inputs=image_inputs,
                )
            proposal_positions = model_inputs["input_ids"] == self.object_token_index
            pred_scores = pred.logits.sigmoid()[proposal_positions].view(-1)
            pred_bboxes = proposals[0].clone().float()
            if score_threshold < 0:
                topk_values, topk_indexes = self.torch.topk(pred_scores.view(-1), 1, dim=0)
                pred_scores = topk_values
                pred_bboxes = pred_bboxes[topk_indexes]
            else:
                mask = pred_scores > score_threshold
                pred_scores = pred_scores[mask]
                pred_bboxes = pred_bboxes[mask]
            if pred_bboxes.numel() == 0:
                return TrackingResult.empty(ts_req=ts_req, query=query)
            bbox = pred_bboxes[0].detach().cpu().tolist()
            confidence = float(pred_scores[0].detach().cpu().item())
            return {
                "packet": "tracking_result",
                "ts_req": ts_req,
                "ts_resp": now_us(),
                "bbox": bbox,
                "confidence": confidence,
                "track_id": None,
                "query": query,
            }
        finally:
            Path(image_path).unlink(missing_ok=True)

    def _load_detector(self, detector_cls):
        torch = self.torch
        model_size = "base" if "base" in self.wedetect_uni_checkpoint.lower() else "large"
        det_model = detector_cls(
            backbone_size=model_size,
            prompt_dim=768,
            num_prompts=256,
            num_proposals=self.num_proposals,
        )
        checkpoint = torch.load(
            self.wedetect_uni_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint = _rewrite_uni_checkpoint_keys(checkpoint)
        det_model.load_state_dict(checkpoint, strict=False)
        return det_model.to(self.device).eval()

    def _load_ref_model(self):
        torch = self.torch
        dtype = _choose_dtype(torch, self.device, self.dtype_name)
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        model = self.GroundingModel.from_pretrained(
            self.wedetect_ref_model_dir,
            **model_kwargs,
        )
        processor = self.AutoProcessor.from_pretrained(self.wedetect_ref_model_dir)
        object_token_index = processor.tokenizer.convert_tokens_to_ids("<object>")
        model.model.object_token_id = object_token_index
        return model.to(self.device).eval(), processor, object_token_index


def _rewrite_uni_checkpoint_keys(checkpoint: dict[str, Any]) -> dict[str, Any]:
    keys = list(checkpoint.keys())
    for key in keys:
        if "backbone" in key:
            new_key = key.replace("backbone.image_model.model.", "backbone.")
            checkpoint[new_key] = checkpoint.pop(key)
    keys = list(checkpoint.keys())
    for key in keys:
        if "bbox_head" in key:
            new_key = key.replace("bbox_head.head_module.", "bbox_head.")
            new_key = new_key.replace("0.2.", "0.6.")
            new_key = new_key.replace("1.2.", "1.6.")
            new_key = new_key.replace("2.2.", "2.6.")
            new_key = new_key.replace("1.bn", "4")
            new_key = new_key.replace("1.conv", "3")
            new_key = new_key.replace("0.bn", "1")
            new_key = new_key.replace("0.conv", "0")
            checkpoint[new_key] = checkpoint.pop(key)
    return checkpoint


def _choose_dtype(torch, device: str, dtype_name: str):
    normalized = dtype_name.lower()
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float32":
        return torch.float32
    if device.startswith("cuda") and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _image_suffix(frame_bytes: bytes) -> str:
    if frame_bytes.startswith(b"\xff\xd8"):
        return ".jpg"
    if frame_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    return ".jpg"
