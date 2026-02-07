import argparse
import ast
import json
import os
import os.path as op
from string import ascii_uppercase
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
    parser = argparse.ArgumentParser(description="Embed MMMU prompts with GLM-4.1V")
    parser.add_argument(
        "--model_path",
        type=str,
        default="zai-org/GLM-4.1V-9B-Thinking",
        help="Path or HF repo id for the GLM-4.1V model.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="Data/Questions/MMMU_Validation.jsonl",
        help="Path to input data (.json or .jsonl).",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default="Data/Images/MMMU_Val",
        help="Directory containing image assets referenced by the dataset.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="Data/Representation/MMMU_Validation",
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


def resolve_image_paths(
    datapoint: Dict[str, Any],
    images_root: str,
    data_path: str,
) -> List[str]:
    entries: List[str] = []
    if "images" in datapoint:
        if isinstance(datapoint["images"], list):
            entries = datapoint["images"]
        elif isinstance(datapoint["images"], str):
            entries = [datapoint["images"]]
    elif "image" in datapoint:
        image_val = datapoint["image"]
        if isinstance(image_val, list):
            entries = image_val
        elif isinstance(image_val, str):
            entries = [image_val]
    else:
        for i in range(1, 8):
            key = f"image_{i}"
            if key not in datapoint:
                continue
            val = datapoint[key]
            if isinstance(val, list):
                entries.extend(val)
            elif isinstance(val, str):
                entries.append(val)

    base_dir = images_root or (op.dirname(data_path) if data_path else os.getcwd())
    resolved: List[str] = []
    for entry in entries:
        if not entry:
            continue
        if op.isabs(entry) and op.exists(entry):
            resolved.append(entry)
            continue
        candidate = op.join(base_dir, op.basename(entry))
        if op.exists(candidate):
            resolved.append(candidate)
            continue
        candidate = op.join(base_dir, entry)
        if op.exists(candidate):
            resolved.append(candidate)
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
            except Exception as exc:
                print(f"[warn] failed to load image {path}: {exc}; using fallback size.")
            rng = _noise_rng(noise_seed, sample_id, idx)
            images.append(_make_gaussian_noise_image(width, height, rng, noise_mean, noise_std))
            continue
        try:
            with Image.open(path) as img:
                if img.mode == "P" and "transparency" in img.info:
                    img = img.convert("RGBA")
                images.append(img.convert("RGB"))
        except Exception as exc:
            print(f"[warn] failed to load image {path}: {exc}")
    return images


def split_options(value: Any) -> Tuple[List[str], str]:
    options_list: Optional[List[str]] = None
    options_text = ""
    if isinstance(value, list):
        options_list = value
    elif isinstance(value, str):
        text = value.strip()
        if text:
            parsed = None
            if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
                try:
                    parsed = ast.literal_eval(text)
                except Exception:
                    parsed = None
            if isinstance(parsed, list):
                options_list = parsed
            else:
                options_text = text
    return options_list or [], options_text


def build_prompt_text(
    datapoint: Dict[str, Any],
    template: Template,
) -> Tuple[str, List[str]]:
    problem = datapoint.get("problem") or datapoint.get("question") or ""
    options_list, options_text = split_options(datapoint.get("options"))

    options_block = ""
    if options_list:
        labels = list(ascii_uppercase[: len(options_list)])
        lines = [f"{label}. {opt}" for label, opt in zip(labels, options_list)]
        if lines:
            options_block = "\nOptions:\n" + "\n".join(lines)
    elif options_text:
        options_block = "\nOptions:\n" + options_text

    prompt_text = template.render(prompt=problem, options_block=options_block)
    return prompt_text, options_list


def gather_prompts_and_images(
    records: Sequence[Dict[str, Any]],
    template: Template,
    images_root: str,
    data_path: str,
    expect_images: bool,
) -> Tuple[List[str], List[List[str]], List[str], List[List[str]]]:
    prompts: List[str] = []
    image_paths: List[List[str]] = []
    missing_images: List[str] = []
    options_list_list: List[List[str]] = []
    for idx, record in enumerate(records):
        prompt_text, options_list = build_prompt_text(record, template)
        resolved_paths = resolve_image_paths(record, images_root, data_path) if expect_images else []
        if expect_images and not resolved_paths:
            qid = record.get("question_id") or record.get("id") or record.get("uid") or str(idx)
            missing_images.append(qid)
        prompts.append(prompt_text)
        image_paths.append(resolved_paths)
        options_list_list.append(options_list)
    return prompts, image_paths, missing_images, options_list_list


def make_messages(
    user_text: str,
    num_images: int,
    reasoning: bool,
    options_list: Sequence[str],
) -> List[Dict[str, Any]]:
    if options_list:
        labels = ", ".join(ascii_uppercase[: max(1, len(options_list))])
        if reasoning:
            system_prompt = (
                "Please reason step by step. Choose the correct option among "
                f"{labels}, and put the chosen letter as your final answer within \\boxed{{}}."
            )
        else:
            system_prompt = (
                "Choose the correct option among "
                f"{labels}, and put the chosen letter as your final answer within \\boxed{{}}."
            )
    else:
        if reasoning:
            system_prompt = (
                "Please reason step by step, and put your final answer within \\boxed{}."
            )
        else:
            system_prompt = "Put your final answer within \\boxed{}."

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


def forward_batch(
    prompts: Sequence[str],
    image_paths_list: Sequence[Sequence[str]],
    processor: Glm4vProcessor,
    model: Glm4vForConditionalGeneration,
    reasoning: bool,
    device: torch.device,
    options_list_list: Sequence[Sequence[str]],
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

    for idx, (prompt, images, options_list) in enumerate(zip(prompts, loaded_images, options_list_list)):
        messages = make_messages(prompt, len(images), reasoning, options_list)
        prompt_str = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_token = getattr(processor, "image_token", "<|image|>")
        placeholder_count = prompt_str.count(image_token)
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

    template_str = "This is the problem:\n{{prompt}}\n{{options_block}}"
    prompt_template = Template(template_str)

    expect_images = use_images and (bool(args.images_root) or any(
        isinstance(record, dict) and any(str(k).startswith("image_") for k in record.keys())
        for record in records
    ))
    prompts, image_path_groups, missing_images, options_list_list = gather_prompts_and_images(
        records, prompt_template, args.images_root, args.data_path, expect_images
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
        batch_options_list = options_list_list[start:end]
        hidden_states = forward_batch(
            batch_prompts,
            batch_images,
            processor,
            model,
            args.reasoning,
            model_device,
            batch_options_list,
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
