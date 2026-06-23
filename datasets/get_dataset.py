#!/usr/bin/env python3
"""
Export a Ximilar VLM dataset to a finetuning-ready conversation ("messages") JSON file.

Supports both dataset modes (auto-detected from the dataset's `mode`):
  * instruction  - single-turn: image(s) + input prompt + labeled variables -> one
                   user->assistant example (answer rendered from the result template).
  * agentic      - multi-turn conversation built from the sample's steps (user / assistant
                   thoughts / tool_call / tool_result / answer) plus the dataset's tools.
                   Thoughts are emitted natively per format (hf: <think>...</think> inline;
                   gpt: a separate `reasoning` field) unless --drop-thoughts is given.

Two output flavours are supported:

  --format hf   HuggingFace TRL / chat-template style (native for Qwen3-VL, Gemma 3).
                `content` is a list of typed blocks; images are {"type": "image"}
                placeholders and the actual images live in a parallel top-level "images" list.

  --format gpt  OpenAI vision finetuning style. content blocks use {"type": "image_url", ...}
                and the assistant content is a plain string (OpenAI forbids images in output).

Output file type follows the --output extension: ".jsonl" -> one object per line
(recommended for gpt / OpenAI uploads), anything else -> a single pretty JSON array.

Images are written to --img_folder and referenced by relative path, unless --img_as_base64
is given, in which case they are embedded inline as `data:<mime>;base64,...` URIs.
(For a real OpenAI upload, use --img_as_base64: OpenAI needs a URL or data-URI, not a path.)

Auth via --api_token (or the XIMILAR_API_KEY env var):
    XIMILAR_API_KEY        (required unless --api_token is given)
    XIMILAR_WORKSPACE_ID   (optional, overridable with --workspace_id)

RUN (from the transformers/ uv env):
    cd transformers && uv run python ../datasets/get_dataset.py \
        --dataset_id <ID> --type train --format hf \
        --output train.json --img_folder train_images [--img_as_base64]
"""

import argparse
import base64
import json
import logging
import os
import re
import sys

import requests
from tqdm import tqdm

from ximilar.client.vlm import VLMClient


logger = logging.getLogger(__name__)

# {{ var }} / {{var}} template placeholders.
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([\w.\-]+)\s*\}\}")

_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

# --type value -> sample `test` filter passed to the API.
_TYPE_TO_TEST = {"train": False, "test": True, "all": None}


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a Ximilar VLM dataset to finetuning messages JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset_id", required=True, help="UUID of the VLM dataset to export")
    parser.add_argument(
        "--type",
        choices=["train", "test", "all"],
        default="train",
        help="Which split to export (maps to the sample test flag). Default: train",
    )
    parser.add_argument(
        "--format",
        choices=["hf", "gpt"],
        default="hf",
        help="Output schema: hf (HuggingFace/TRL) or gpt (OpenAI). Default: hf",
    )
    parser.add_argument("--output", required=True, help="Output file (.json -> array, .jsonl -> lines)")
    parser.add_argument(
        "--img_folder",
        help="Directory to download images into (required unless --img_as_base64)",
    )
    parser.add_argument(
        "--img_as_base64",
        action="store_true",
        help="Embed images inline as data:<mime>;base64,... instead of downloading files",
    )
    parser.add_argument(
        "--no-system-prompt",
        dest="no_system_prompt",
        action="store_true",
        help="Do not emit the dataset's system prompt as a leading system message",
    )
    parser.add_argument(
        "--drop-thoughts",
        dest="drop_thoughts",
        action="store_true",
        help="Agentic mode: omit assistant 'thought' steps from the exported conversation",
    )
    parser.add_argument("--api_token", help="Ximilar API token (defaults to $XIMILAR_API_KEY)")
    parser.add_argument("--workspace_id", help="Workspace id (defaults to $XIMILAR_WORKSPACE_ID)")
    parser.add_argument("--limit", type=int, help="Only export the first N samples (for testing)")
    return parser


def render_template(template: str, values: dict) -> str:
    """Replace {{ name }} placeholders with values[name] (missing -> empty string)."""
    if not template:
        return ""

    def _sub(match):
        key = match.group(1)
        value = values.get(key, "")
        return "" if value is None else str(value)

    return _PLACEHOLDER_RE.sub(_sub, template)


def resolve_prompt(config: dict, key: str):
    """Pull a prompt string from a dataset/sample dict, tolerating a *_ref nested object."""
    if not config:
        return None
    value = config.get(key)
    if isinstance(value, str) and value.strip():
        return value
    ref = config.get(f"{key}_ref")
    if isinstance(ref, dict):
        content = ref.get("content") or ref.get("prompt")
        if isinstance(content, str) and content.strip():
            return content
    return None


def build_question(sample, ds_user_prompt) -> str:
    """User turn text: rendered user_prompt (sample- or dataset-level), else input_meta_data."""
    imd = sample.input_meta_data or {}
    user_prompt = sample.user_prompt or ds_user_prompt
    if user_prompt:
        return render_template(user_prompt, imd)
    if "question" in imd:
        return str(imd["question"])
    if imd:
        return "\n".join(str(v) for v in imd.values())
    return ""


def build_answer(sample, ds_result_template) -> str:
    """Assistant turn text: dataset result_template filled with the sample's variables."""
    values = {v.get("variable_name"): v.get("value") for v in (sample.sample_variables or [])}
    template = sample.result_template or ds_result_template
    if template:
        return render_template(template, values)
    return json.dumps(values, ensure_ascii=False)


def mime_for(url: str) -> str:
    ext = os.path.splitext(url.split("?")[0])[1].lower()
    return _MIME_BY_EXT.get(ext, "image/jpeg")


def image_ref(client, url, img_folder, as_base64) -> str:
    """Return the reference stored in the JSON for one image: a relative path or a data-URI."""
    if as_base64:
        content = requests.get(url, timeout=90).content
        encoded = base64.b64encode(content).decode("ascii")
        return f"data:{mime_for(url)};base64,{encoded}"

    dest = client.download_image(url, destination=img_folder)
    return os.path.join(img_folder, os.path.basename(dest))


def collect_image_refs(client, sample_id, img_folder, as_base64) -> list:
    images, status = client.get_sample_images(sample_id)
    if images is None:
        raise RuntimeError(f"failed to list images: {status}")
    images = sorted(images, key=lambda i: i.get("order") or 0)
    refs = []
    for img in images:
        url = img.get("img_path")
        if not url:
            continue
        refs.append(image_ref(client, url, img_folder, as_base64))
    return refs


def to_hf_messages(image_refs, question, answer, system_prompt) -> dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    user_content = [{"type": "image"} for _ in image_refs]
    user_content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return {"messages": messages, "images": image_refs}


def to_gpt_messages(image_refs, question, answer, system_prompt) -> dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    user_content = [{"type": "text", "text": question}]
    for ref in image_refs:
        user_content.append({"type": "image_url", "image_url": {"url": ref}})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": answer})
    return {"messages": messages}


# --------------------------------------------------------------------------------------
# Agentic mode (multi-turn conversations with thoughts and tool calls)
# --------------------------------------------------------------------------------------


def content_to_str(content) -> str:
    """A step's content is a JSONField: keep strings as-is, serialize objects/arrays."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def build_tools(tools_raw) -> list:
    """Map Ximilar VLMTool dicts to the OpenAI/HF function-tool schema (shared by both formats)."""
    tools = []
    for t in tools_raw or []:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters_schema") or {},
                },
            }
        )
    return tools


def parse_steps(steps, image_fn) -> list:
    """
    Collapse ordered VLMSampleStep records into neutral chat turns. Consecutive assistant
    sub-steps (thought / tool_call / answer) are merged into a single assistant turn, which
    is flushed when a tool result or a new user turn appears.

    Returns a list of dicts shaped per role:
      user      -> {"role": "user", "text": str, "images": [ref, ...]}
      tool      -> {"role": "tool", "tool_call_id": str, "name": str, "content": str, "is_error": bool}
      assistant -> {"role": "assistant", "reasoning": [str], "content": str|None,
                    "tool_calls": [{"id": str, "name": str, "arguments": Any}]}
      system    -> {"role": "system", "text": str}
    """
    messages = []
    cur = None

    def flush():
        nonlocal cur
        if cur and (cur["reasoning"] or cur["content"] is not None or cur["tool_calls"]):
            messages.append(cur)
        cur = None

    def ensure_assistant():
        nonlocal cur
        if cur is None:
            cur = {"role": "assistant", "reasoning": [], "content": None, "tool_calls": []}
        return cur

    for st in steps:
        role, step_type, content = st.get("role"), st.get("type"), st.get("content")
        step_images = sorted(st.get("step_images") or [], key=lambda i: i.get("order") or 0)
        image_refs = [image_fn(i["img_path"]) for i in step_images if i.get("img_path")]

        if role == "user":
            flush()
            messages.append({"role": "user", "text": content_to_str(content), "images": image_refs})
        elif role == "tool":
            flush()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": st.get("tool_call_id"),
                    "name": st.get("tool_name"),
                    "content": content_to_str(content),
                    "is_error": bool(st.get("is_error")),
                }
            )
        elif role == "assistant":
            assistant = ensure_assistant()
            if step_type == "thought":
                assistant["reasoning"].append(content_to_str(content))
            elif step_type == "tool_call":
                name = content.get("name") if isinstance(content, dict) else st.get("tool_name")
                arguments = content.get("arguments") if isinstance(content, dict) else {}
                assistant["tool_calls"].append(
                    {"id": st.get("tool_call_id"), "name": name, "arguments": arguments}
                )
            else:  # answer / text
                assistant["content"] = content_to_str(content)
        else:  # system step
            flush()
            messages.append({"role": "system", "text": content_to_str(content)})

    flush()
    return messages


def agentic_to_hf(parsed, tools, system_prompt, drop_thoughts) -> dict:
    messages = []
    images = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    for m in parsed:
        if m["role"] == "user":
            content = [{"type": "image"} for _ in m["images"]]
            images.extend(m["images"])
            if m["text"]:
                content.append({"type": "text", "text": m["text"]})
            messages.append({"role": "user", "content": content})
        elif m["role"] == "tool":
            messages.append({"role": "tool", "name": m["name"], "content": m["content"]})
        elif m["role"] == "system":
            messages.append({"role": "system", "content": [{"type": "text", "text": m["text"]}]})
        else:  # assistant
            text = ""
            reasoning = "\n\n".join(r for r in m["reasoning"] if r)
            if reasoning and not drop_thoughts:
                text += f"<think>\n{reasoning}\n</think>\n\n"
            if m["content"]:
                text += m["content"]
            msg = {"role": "assistant", "content": text}
            if m["tool_calls"]:
                msg["tool_calls"] = [
                    {"type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in m["tool_calls"]
                ]
            messages.append(msg)

    record = {"messages": messages, "images": images}
    if tools:
        record["tools"] = tools
    return record


def agentic_to_gpt(parsed, tools, system_prompt, drop_thoughts) -> dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for m in parsed:
        if m["role"] == "user":
            content = [{"type": "text", "text": m["text"]}] if m["text"] else []
            for ref in m["images"]:
                content.append({"type": "image_url", "image_url": {"url": ref}})
            messages.append({"role": "user", "content": content or m["text"]})
        elif m["role"] == "tool":
            messages.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        elif m["role"] == "system":
            messages.append({"role": "system", "content": m["text"]})
        else:  # assistant
            msg = {"role": "assistant"}
            reasoning = "\n\n".join(r for r in m["reasoning"] if r)
            if reasoning and not drop_thoughts:
                msg["reasoning"] = reasoning
            if m["tool_calls"]:
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                            if isinstance(tc["arguments"], str)
                            else json.dumps(tc["arguments"], ensure_ascii=False),
                        },
                    }
                    for tc in m["tool_calls"]
                ]
            msg["content"] = m["content"]
            messages.append(msg)

    record = {"messages": messages}
    if tools:
        record["tools"] = tools
    return record


def write_output(path, records):
    if path.lower().endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = get_arg_parser().parse_args()

    if not args.img_as_base64 and not args.img_folder:
        get_arg_parser().error("--img_folder is required unless --img_as_base64 is set")

    api_key = args.api_token or os.environ.get("XIMILAR_API_KEY")
    if not api_key:
        logger.error("Provide an API token via --api_token or the XIMILAR_API_KEY environment variable")
        sys.exit(1)
    workspace = args.workspace_id or os.environ.get("XIMILAR_WORKSPACE_ID", "default")

    if args.img_folder and not args.img_as_base64:
        os.makedirs(args.img_folder, exist_ok=True)

    client = VLMClient(api_key, workspace=workspace)
    client.max_image_size = 0

    dataset = client.get_dataset(args.dataset_id)
    if not dataset or "detail" in dataset:
        logger.error("Could not load dataset %s: %s", args.dataset_id, dataset)
        sys.exit(1)

    ds_user_prompt = resolve_prompt(dataset, "user_prompt")
    ds_result_template = resolve_prompt(dataset, "result_template")
    system_prompt = None if args.no_system_prompt else resolve_prompt(dataset, "system_prompt")
    is_agentic = dataset.get("mode") == "agentic"

    tools = None
    if is_agentic:
        tools_raw, tstatus = client.get_tools(args.dataset_id)
        tools = build_tools(tools_raw or [])
    logger.info(
        "Dataset '%s' loaded (mode=%s, system_prompt=%s, %s)",
        dataset.get("name", args.dataset_id),
        dataset.get("mode", "instruction"),
        bool(system_prompt),
        f"tools={len(tools)}" if is_agentic else f"result_template={bool(ds_result_template)}",
    )

    test_filter = _TYPE_TO_TEST[args.type]
    samples, status = client.get_samples(args.dataset_id, test=test_filter)
    if samples is None:
        logger.error("Could not list samples: %s", status)
        sys.exit(1)
    if args.limit is not None:
        samples = samples[: args.limit]
    logger.info("Exporting %d sample(s) [type=%s mode=%s]", len(samples), args.type, dataset.get("mode"))

    records = []
    succeeded = skipped = failed = 0

    def image_fn(url):
        return image_ref(client, url, args.img_folder, args.img_as_base64)

    for item in tqdm(samples, desc=f"{args.type}:{args.format}"):
        sample_id = item.get("id")
        try:
            if is_agentic:
                steps, sstatus = client.get_sample_steps(sample_id)
                if not steps:
                    tqdm.write(f"[skip] sample {sample_id}: no steps")
                    skipped += 1
                    continue
                steps = sorted(steps, key=lambda s: (s.get("order") or 0, s.get("sub_order") or 0))
                parsed = parse_steps(steps, image_fn)
                builder = agentic_to_hf if args.format == "hf" else agentic_to_gpt
                records.append(builder(parsed, tools, system_prompt, args.drop_thoughts))
            else:
                sample = client.get_sample(sample_id)
                image_refs = collect_image_refs(client, sample_id, args.img_folder, args.img_as_base64)
                if not image_refs:
                    tqdm.write(f"[skip] sample {sample_id}: no images")
                    skipped += 1
                    continue
                question = build_question(sample, ds_user_prompt)
                answer = build_answer(sample, ds_result_template)
                build = to_hf_messages if args.format == "hf" else to_gpt_messages
                records.append(build(image_refs, question, answer, system_prompt))
            succeeded += 1
        except Exception as e:  # noqa: BLE001 - keep going, report per-sample
            tqdm.write(f"[fail] sample {sample_id}: {e}")
            failed += 1

    write_output(args.output, records)
    logger.info(
        "Done. wrote %d record(s) to %s (succeeded=%d skipped=%d failed=%d)",
        len(records),
        args.output,
        succeeded,
        skipped,
        failed,
    )


if __name__ == "__main__":
    main()
