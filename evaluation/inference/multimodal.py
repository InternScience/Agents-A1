"""Multimodal message construction for items that may carry an image attachment.

- Items without a `file_name` field pass through unchanged (text-only message).
- Items with a `file_name` produce an OpenAI-style multipart user message with
  a base64 data URL for the image.
- Backend vision support is checked statically (model config) and at runtime
  (API error markers) so misconfigurations surface with actionable messages.
"""

import base64
import json
import os
from io import BytesIO
from typing import Any, Optional


VISION_CONFIG_KEYS = (
    "vision_config",
    "image_token_id",
    "mm_vision_tower",
    "vision_tower",
)

MULTIMODAL_RUNTIME_MARKERS = (
    "at most 0 image",
    "may be provided in one prompt",
    "image_url",
    "image input",
    "multimodal",
    "vision",
    "language-model-only",
    "unsupported content",
    "expected a string",
    "content type",
    "does not support image",
    "does not support vision",
)

DEFAULT_IMAGE_SHORT_SIDE = int(os.getenv("HLE_IMAGE_SHORT_SIDE", "1080"))


def _resize(img, short_side: int):
    from PIL import Image

    width, height = img.size
    if min(width, height) <= short_side:
        return img
    if width <= height:
        new_w, new_h = short_side, int(short_side / width * height)
    else:
        new_h, new_w = short_side, int(short_side / height * width)
    return img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)


def encode_image_as_base64(path: str, max_short_side: int = DEFAULT_IMAGE_SHORT_SIDE) -> str:
    from PIL import Image

    img = Image.open(path)
    if max_short_side > 0:
        img = _resize(img, max_short_side)
    img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def resolve_image_path(item: dict, *, base_dir: Optional[str] = None) -> Optional[str]:
    file_name = str(item.get("file_name") or "").strip()
    if not file_name:
        return None

    candidates: list[str] = [file_name]
    if not os.path.isabs(file_name):
        if base_dir:
            candidates.append(os.path.join(base_dir, file_name))
        candidates.append(os.path.abspath(file_name))

    tried: list[str] = []
    for cand in candidates:
        cand = os.path.abspath(cand)
        if cand in tried:
            continue
        tried.append(cand)
        if os.path.exists(cand):
            return cand

    raise FileNotFoundError(
        f"Image declared by item not found: file_name={file_name!r}, "
        f"base_dir={base_dir!r}, tried={tried}"
    )


def messages_have_image(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") == "image_url" for part in content
        ):
            return True
    return False


def _resolve_config_path(model_path: Optional[str]) -> Optional[str]:
    if not model_path:
        return None
    if os.path.isdir(model_path):
        cand = os.path.join(model_path, "config.json")
        return cand if os.path.exists(cand) else None
    if model_path.endswith("config.json") and os.path.exists(model_path):
        return model_path
    return None


def infer_vision_support(model_path: Optional[str]) -> tuple[Optional[bool], str]:
    cfg_path = _resolve_config_path(model_path)
    if not cfg_path:
        return None, f"config_unavailable model_path={model_path!r}"
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as exc:
        return None, f"config_read_failed path={cfg_path!r} error={exc}"
    return any(k in cfg for k in VISION_CONFIG_KEYS), f"config_path={cfg_path!r}"


def static_multimodal_error(
    messages: list[dict],
    *,
    model_path: Optional[str],
    served_model_name: str,
) -> Optional[str]:
    if not messages_have_image(messages):
        return None
    supports, detail = infer_vision_support(model_path)
    if supports is False:
        return (
            "Image input is enabled for this sample, but the configured model does not "
            f"appear to expose vision support. served_model={served_model_name}, "
            f"model_path={model_path!r}, {detail}"
        )
    return None


def runtime_multimodal_error(
    messages: list[dict],
    *,
    model_path: Optional[str],
    served_model_name: str,
    error_text: str,
    base_url: Optional[str] = None,
) -> Optional[str]:
    if not messages_have_image(messages):
        return None
    supports, detail = infer_vision_support(model_path)
    lowered = error_text.lower()
    if supports is False or any(m in lowered for m in MULTIMODAL_RUNTIME_MARKERS):
        return (
            "Image input is enabled but the backend rejected multimodal messages. "
            f"served_model={served_model_name}, model_path={model_path!r}, "
            f"base_url={base_url!r}, config_support={supports}, {detail}, "
            f"error={error_text}"
        )
    return None


def build_user_message(
    item: dict,
    question: str,
    *,
    model_path: Optional[str],
    served_model_name: str,
    base_dir: Optional[str] = None,
) -> dict[str, Any]:
    image_path = resolve_image_path(item, base_dir=base_dir)
    if image_path is None:
        return {"role": "user", "content": question}

    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": encode_image_as_base64(image_path)}},
        ],
    }
    err = static_multimodal_error(
        [msg], model_path=model_path, served_model_name=served_model_name
    )
    if err:
        raise RuntimeError(err)
    return msg


def flatten_messages_for_tokenization(messages: list[dict]) -> list[dict]:
    """Return a copy of `messages` where multipart content is reduced to its text parts.

    Some tokenizers' `apply_chat_template` cannot handle list-typed `content`; the
    backend is responsible for accurately accounting for image tokens. This helper
    keeps our heuristic token count from blowing up on multipart messages.
    """
    flat: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            flat.append({**msg, "content": " ".join(text_parts)})
        else:
            flat.append(msg)
    return flat
