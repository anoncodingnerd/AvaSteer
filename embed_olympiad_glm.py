import argparse
import json
import os
import os.path as op
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from jinja2 import Template
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoTokenizer
from transformers.models.glm4v import Glm4vForConditionalGeneration
from transformers.models.glm4v.processing_glm4v import Glm4vProcessor
from transformers.models.glm4v.video_processing_glm4v import Glm4vVideoProcessor


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        val = v.lower()
        if val in {"yes", "true", "t", "y", "1"}:
            return True
        if val in {"no", "false", "f", "n", "0"}:
            return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed OlympiadBench with GLM-4.1V")
    parser.add_argument(
        "--model_path",
        type=str,
        default="zai-org/GLM-4.1V-9B-Thinking",
        help="Path or HF repo id for the GLM-4.1V model.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="Data/exports/OlympiadBenchZhComp/OlympiadBench.jsonl",
        help="Path to OlympiadBench JSON/JSONL.",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default="",
        help="Optional image root. Defaults to the directory of data_path.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="Data/Representation/OlympiadBench",
        help="Directory to store the resulting embeddings.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of samples to process per batch when all share the same modality.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum sequence length passed to the tokenizer.",
    )
    parser.add_argument(
        "--reasoning",
        type=str2bool,
        default=True,
        help="If true, add reasoning instructions to the prompt.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Threads for pre-loading images per batch to reduce CPU I/O stalls.",
    )
    parser.add_argument(
        "--input_mode",
        type=str,
        default="mixed",
        help=(
            'Input modality: "mixed" (text+image), "text_only" (drop images), '
            'or "gaussian_noise" (replace images with Gaussian noise).'
        ),
    )
    parser.add_argument(
        "--noise_seed",
        type=int,
        default=42,
        help="Random seed for Gaussian noise image generation.",
    )
    parser.add_argument(
        "--noise_mean",
        type=float,
        default=0.5,
        help="Mean for Gaussian noise in [0, 1] pixel space.",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.2,
        help="Std for Gaussian noise in [0, 1] pixel space.",
    )
    parser.add_argument(
        "--noise_size",
        type=int,
        default=224,
        help="Fallback size when an image cannot be loaded.",
    )
    return parser.parse_args()


def load_dataset(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def clean_question(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip()


def resolve_image_paths(
    datapoint: Dict[str, Any],
    images_root: str,
    data_dir: str,
) -> List[str]:
    entries: List[str] = []
    if "images" in datapoint:
        val = datapoint["images"]
        if isinstance(val, list):
            entries.extend(val)
        elif isinstance(val, str):
            entries.append(val)
    elif "image" in datapoint:
        val = datapoint["image"]
        if isinstance(val, list):
            entries.extend(val)
        elif isinstance(val, str):
            entries.append(val)
    else:
        for i in range(1, 10):
            key = f"image_{i}"
            val = datapoint.get(key)
            if not val:
                continue
            if isinstance(val, list):
                entries.extend(val)
            elif isinstance(val, str):
                entries.append(val)

    base_dir = images_root or data_dir
    resolved: List[str] = []
    for entry in entries:
        if not entry:
            continue
        if op.isabs(entry) and op.exists(entry):
            resolved.append(entry)
            continue
        base_name = op.basename(entry)
        cand1 = op.join(base_dir, base_name)
        if op.exists(cand1):
            resolved.append(cand1)
            continue
        cand2 = op.join(base_dir, entry)
        if op.exists(cand2):
            resolved.append(cand2)
            continue
    return resolved


def _noise_rng(seed: Optional[int], sample_id: Optional[int], image_idx: int) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    base = int(seed)
    sid = int(sample_id) if sample_id is not None else 0
    return np.random.default_rng(base + sid * 1000 + image_idx)


def _make_gaussian_noise_image(
    width: int,
    height: int,
    rng: np.random.Generator,
    mean: float,
    std: float,
) -> Image.Image:
    noise = rng.normal(loc=mean, scale=std, size=(height, width, 3))
    noise = np.clip(noise, 0.0, 1.0)
    noise = (noise * 255.0).astype(np.uint8)
    return Image.fromarray(noise)


def load_images(
    image_paths: Sequence[str],
    image_mode: str = "original",
    noise_seed: Optional[int] = None,
    noise_mean: float = 0.5,
    noise_std: float = 0.2,
    noise_size: int = 224,
    sample_id: Optional[int] = None,
) -> List[Image.Image]:
    images: List[Image.Image] = []
    for idx, path in enumerate(image_paths):
        if image_mode == "gaussian_noise":
            width = noise_size
            height = noise_size
            try:
                with Image.open(path) as img:
                    width, height = img.size
            except Exception as exc:  # pragma: no cover - logging only
                print(f"[warn] failed to load image {path}: {exc}; using fallback size.")
            rng = _noise_rng(noise_seed, sample_id, idx)
            images.append(_make_gaussian_noise_image(width, height, rng, noise_mean, noise_std))
            continue
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        except Exception as exc:  # pragma: no cover - logging only
            print(f"[warn] failed to load image {path}: {exc}")
    return images


def build_prompt_text(datapoint: Dict[str, Any], template: Template) -> str:
    question = datapoint.get("question") or datapoint.get("problem") or ""
    question = clean_question(question)
    context = datapoint.get("context")
    if isinstance(context, str) and context.strip():
        question = f"{context.strip()}\n\n{question}"
    return template.render(prompt=question, options_block="")


def gather_prompts_and_images(
    records: Sequence[Dict[str, Any]],
    template: Template,
    images_root: str,
    data_dir: str,
    expect_images: bool,
) -> Tuple[List[str], List[List[str]], List[str]]:
    prompts: List[str] = []
    image_paths: List[List[str]] = []
    missing_images: List[str] = []
    for idx, record in enumerate(records):
        prompt_text = build_prompt_text(record, template)
        resolved_paths = resolve_image_paths(record, images_root, data_dir) if expect_images else []
        if expect_images and not resolved_paths:
            qid = record.get("question_id") or record.get("id") or record.get("uid") or str(idx)
            missing_images.append(qid)
        prompts.append(prompt_text)
        image_paths.append(resolved_paths)
    return prompts, image_paths, missing_images


def make_messages(user_text: str, num_images: int, reasoning: bool) -> List[Dict[str, Any]]:
    if reasoning:
        system_prompt = "请逐步推理，并将最终答案写在\\boxed{}中。"
    else:
        system_prompt = "请将最终答案写在\\boxed{}中。"

    user_content: List[Any] = []
    for _ in range(num_images):
        user_content.append({"type": "image"})
    user_content.append({"type": "text", "text": user_text})

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _count_image_placeholders(prompt_text: str, processor: Glm4vProcessor) -> int:
    token = getattr(processor, "image_token", None)
    if token:
        return prompt_text.count(token)
    if "<|image_pad|>" in prompt_text:
        return prompt_text.count("<|image_pad|>")
    if "<|image|>" in prompt_text:
        return prompt_text.count("<|image|>")
    return 0


def forward_batch(
    prompts: Sequence[str],
    image_paths_list: Sequence[Sequence[str]],
    processor: Glm4vProcessor,
    model: Glm4vForConditionalGeneration,
    reasoning: bool,
    device: torch.device,
    num_workers: int,
    image_mode: str,
    noise_seed: Optional[int],
    noise_mean: float,
    noise_std: float,
    noise_size: int,
) -> Tuple[torch.Tensor, ...]:
    if not prompts:
        return tuple()

    def _load_with_mode(item: Tuple[int, Sequence[str]]) -> List[Image.Image]:
        idx, paths = item
        return load_images(
            paths,
            image_mode=image_mode,
            noise_seed=noise_seed,
            noise_mean=noise_mean,
            noise_std=noise_std,
            noise_size=noise_size,
            sample_id=idx,
        )

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        loaded_images = list(ex.map(_load_with_mode, enumerate(image_paths_list)))

    batched_prompts: List[str] = []
    batched_images: List[List[Image.Image]] = []
    batched_indices: List[int] = []
    per_sample_outputs: List[Optional[Tuple[torch.Tensor, ...]]] = [None] * len(prompts)

    def run_single(prompt_text: str, images: List[Image.Image], slot: int) -> None:
        proc_kwargs = {
            "text": prompt_text,
            "return_tensors": "pt",
            "padding": True,
        }
        if images:
            proc_kwargs["images"] = images
        inputs = processor(**proc_kwargs)
        inputs = move_to_device(inputs, device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        per_sample_outputs[slot] = tuple(
            layer[:, -1, :].detach().cpu().to(torch.float32) for layer in out.hidden_states
        )

    for idx, (prompt, images) in enumerate(zip(prompts, loaded_images)):
        messages = make_messages(prompt, len(images), reasoning)
        prompt_str = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        placeholder_count = _count_image_placeholders(prompt_str, processor)
        if images and placeholder_count != len(images):
            run_single(prompt_str, images, idx)
            continue
        batched_prompts.append(prompt_str)
        batched_images.append(images)
        batched_indices.append(idx)

    if batched_prompts:
        proc_kwargs = {
            "text": batched_prompts,
            "return_tensors": "pt",
            "padding": True,
        }
        if any(len(imgs) > 0 for imgs in batched_images):
            proc_kwargs["images"] = batched_images
        inputs = processor(**proc_kwargs)
        inputs = move_to_device(inputs, device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        for batch_pos, sample_idx in enumerate(batched_indices):
            per_sample_outputs[sample_idx] = tuple(
                layer[batch_pos : batch_pos + 1, -1, :].detach().cpu().to(torch.float32)
                for layer in out.hidden_states
            )

    filtered_outputs: List[Tuple[torch.Tensor, ...]] = [
        output for output in per_sample_outputs if output is not None
    ]
    if not filtered_outputs:
        return tuple()
    num_layers = len(filtered_outputs[0])
    stacked: List[torch.Tensor] = []
    for layer_idx in range(num_layers):
        stacked.append(torch.cat([s[layer_idx] for s in filtered_outputs], dim=0))
    return tuple(stacked)


def accumulate_hidden_states(
    accumulator: List[np.ndarray],
    hidden_states: Tuple[torch.Tensor, ...],
) -> List[np.ndarray]:
    if not hidden_states:
        return accumulator

    seq_embeds: List[np.ndarray] = []
    for layer in hidden_states:
        arr = layer.detach().cpu().to(torch.float32).numpy()
        if arr.ndim == 3:
            arr = arr[:, -1, :]
        elif arr.ndim == 2:
            pass
        else:
            raise RuntimeError(f"Unexpected hidden state rank {arr.ndim}; expected 2 or 3.")
        seq_embeds.append(arr)
    if not accumulator:
        return [embed.copy() for embed in seq_embeds]

    if len(accumulator) != len(seq_embeds):
        raise RuntimeError("Layer count mismatch while aggregating hidden states.")

    for idx, layer_embed in enumerate(seq_embeds):
        accumulator[idx] = np.concatenate((accumulator[idx], layer_embed), axis=0)
    return accumulator


def main() -> None:
    args = parse_args()
    input_mode = (args.input_mode or "mixed").lower()
    if input_mode not in {"mixed", "text_only", "gaussian_noise"}:
        raise ValueError(
            f"Unsupported input_mode={input_mode}; choose from mixed/text_only/gaussian_noise"
        )
    use_images = input_mode in {"mixed", "gaussian_noise"}
    if not use_images:
        print("[info] input_mode=text_only; images will be ignored.")
    image_mode = "gaussian_noise" if input_mode == "gaussian_noise" else "original"

    model_name = args.model_path.split("/")[-1]
    save_dir = op.join(args.save_path, f"{model_name}{'' if args.reasoning else '_no_reasoning'}")
    if not op.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    records = load_dataset(args.data_path)
    data_dir = op.dirname(args.data_path)

    template_str = """\
    题目如下：
    {{prompt}}
    {{options_block}}
    """
    prompt_template = Template(template_str)

    expect_images = use_images and (bool(args.images_root) or any(
        isinstance(record, dict) and any(str(k).startswith("image_") for k in record.keys())
        for record in records
    ))
    prompts, image_path_groups, missing_images = gather_prompts_and_images(
        records, prompt_template, args.images_root, data_dir, expect_images
    )
    if expect_images and missing_images:
        print(f"[info] {len(missing_images)} prompts missing images; falling back to text-only prompts.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    video_processor = Glm4vVideoProcessor()
    processor = Glm4vProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        video_processor=video_processor,
    )
    if not getattr(processor, "chat_template", None):
        processor.chat_template = tokenizer.chat_template

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Glm4vForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target_device)
    print(f"Model device: {next(model.parameters()).device}")

    accumulator: List[np.ndarray] = []
    batch_size = max(1, args.batch_size)
    model_device = next(model.parameters()).device
    for start in tqdm(range(0, len(prompts), batch_size)):
        end = start + batch_size
        batch_prompts = prompts[start:end]
        batch_images = image_path_groups[start:end]
        hidden_states = forward_batch(
            batch_prompts,
            batch_images,
            processor,
            model,
            args.reasoning,
            model_device,
            args.num_workers,
            image_mode,
            args.noise_seed,
            args.noise_mean,
            args.noise_std,
            args.noise_size,
        )
        accumulator = accumulate_hidden_states(accumulator, hidden_states)

    for layer_idx, embeddings in enumerate(accumulator):
        np.save(op.join(save_dir, f"embeds_{layer_idx}.npy"), embeddings)

    print("finished")


if __name__ == "__main__":
    main()
