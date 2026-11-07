"""Gradio demo for the trained Qwen2.5 Assamese cultural-bias classifier.

From project root (next to data_/ and evaluate_lora_v3.py):
    .venv/bin/python app.py

If Gradio is missing, install into the existing environment:
    uv pip install --python .venv/bin/python gradio

Inference matches evaluate_lora_v3.py: same saved SYSTEM_PROMPT, tokenizer
chat template, max input length, deterministic decoding, and JSON validation.
This is a research prototype, not a validated general-purpose bias detector.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT / "data_" / "qwen_lora_adapter_v3_video_disjoint"
CATEGORIES = {"language", "appearance", "food"}


def parse_response(raw: str) -> tuple[int | None, str | None]:
    """Parse a strict JSON prediction; never invent missing labels."""
    candidates = [raw.strip()]
    match = re.search(r"\{[^{}]*\}", raw, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        label, category = obj.get("bias_label"), obj.get("category")
        if type(label) is int and label in (0, 1) and category in CATEGORIES:
            return label, category
    return None, None


def prepare_prompt(text: str, tokenizer, prompt: str, max_comment_tokens: int,
                   max_length: int) -> list[int]:
    """Replicate prompt construction from the V3 evaluation code."""
    comment_ids = tokenizer.encode(text, add_special_tokens=False)
    initial = (
        tokenizer.decode(comment_ids[:max_comment_tokens], skip_special_tokens=True)
        if len(comment_ids) > max_comment_tokens else text
    )
    options = [initial]
    for limit in (120, 80, 48, 24):
        shorter = tokenizer.decode(comment_ids[:limit], skip_special_tokens=True)
        if shorter not in options:
            options.append(shorter)

    sample_answer = '{"bias_label":0,"category":"food"}'
    for current in options:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Comment: {current}\nClassify this comment:"},
        ]
        prefix = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": sample_answer}],
            tokenize=True, add_generation_prompt=False,
        )
        if full[:len(prefix)] != prefix:
            raise ValueError("Chat-template prefix mismatch")
        if len(full) <= max_length:
            return prefix
    raise ValueError("Comment cannot fit within the training token budget")


@lru_cache(maxsize=1)
def load_classifier():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    meta_path = ADAPTER / "pilot_metadata.json"
    if not (ADAPTER / "adapter_config.json").exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Cannot find trained V3 LoRA adapter and metadata under: {ADAPTER}"
        )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if set(metadata.get("categories", [])) != CATEGORIES:
        raise ValueError("This adapter was not trained for the three selected categories")
    if not metadata.get("system_prompt"):
        raise ValueError("Missing saved system prompt in pilot_metadata.json")

    if torch.cuda.is_available():
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "This CUDA GPU does not support BF16. Use a compatible GPU. "
                "FP16 is disabled because it previously generated invalid outputs."
            )
        device, dtype = "cuda", torch.bfloat16
    else:
        # CPU is supported for portability, but substantially slower.
        device, dtype = "cpu", torch.float32

    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"attn_implementation": "eager", "dtype": dtype}
    try:
        base = AutoModelForCausalLM.from_pretrained(metadata["base_model"], **kwargs)
    except (TypeError, ValueError):
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        base = AutoModelForCausalLM.from_pretrained(metadata["base_model"], **kwargs)

    base = base.to(device)
    model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, metadata, device


def predict(comment: str):
    comment = (comment or "").strip()
    if not comment:
        return "Enter a comment to classify.", "—", ""

    try:
        import torch

        model, tokenizer, meta, device = load_classifier()
        prompt_ids = prepare_prompt(
            comment, tokenizer, meta["system_prompt"],
            int(meta["max_comment_tokens"]), int(meta["max_length"]),
        )
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        raw = tokenizer.decode(
            generated[0][len(prompt_ids):], skip_special_tokens=True
        ).strip()
        label, category = parse_response(raw)
        if label is None:
            return "Invalid model response", "—", raw
        return ("Biased (1)" if label == 1 else "Non-biased (0)"), category.title(), raw
    except Exception as error:
        # Display the error during development rather than silently guessing.
        return f"Inference error: {type(error).__name__}: {error}", "—", ""


with gr.Blocks(title="Assamese Cultural Bias Detector") as demo:
    gr.Markdown(
        "# Assamese Cultural Bias Detector\n"
        "**Research prototype · Qwen2.5-0.5B-Instruct + LoRA (Version 3)**\n\n"
        "Classifies text into **biased / non-biased**, with a topic of "
        "**Language**, **Appearance**, or **Food**. "
        "Supports text input in Assamese script, Roman Assamese, or English, "
        "but performance across languages has not been established."
    )
    comment_box = gr.Textbox(
        lines=5,
        label="Comment to analyze",
        placeholder="Paste an Assamese, Roman Assamese, or English comment here...",
    )
    classify = gr.Button("Analyze comment", variant="primary")
    with gr.Row():
        bias_output = gr.Textbox(label="Bias prediction", interactive=False)
        category_output = gr.Textbox(label="Topic category", interactive=False)
    raw_output = gr.Textbox(label="Raw model JSON", lines=2, interactive=False)
    classify.click(
        fn=predict,
        inputs=comment_box,
        outputs=[bias_output, category_output, raw_output],
    )
    gr.Examples(
        examples=[
            ["All Assamese people speak an inferior language."],
            ["I love the Assamese language and enjoy learning it."],
            ["I like Assamese fish curry."],
        ],
        inputs=comment_box,
    )
    gr.Markdown(
        "*Limitations:* This small pilot classifier may mistake criticism or personal "
        "preferences for bias. It always chooses one of the three topic categories "
        "and cannot reliably reject unrelated comments. The demo is not a "
        "moderation or high-stakes decision tool."
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
