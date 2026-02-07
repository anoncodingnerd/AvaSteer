# Think Less, See More: AvaSteer for Adaptive Reasoning Control in Large Vision-Language Models

This repository provides an anonymous release of the code for **AvaSteer**, an inference-time adaptive activation steering method for reducing unnecessary reasoning in large vision-language models while preserving accuracy.

The pipeline in this repo is:

1) Extract embeddings for **image-present** and **image-absent** prompts (via `embed_*.sh` scripts)  
2) Compute a **steering vector** and a **calibration vector** from the embedding difference (`diff_embeds_image_vs_no_image.py`)  
3) Run generation and evaluation under steering (via `generate_and_evaluate_*.sh`)


---

## Repository Structure (Key Files)

- `scripts/`
  - `embed_*.sh`: extract embeddings for (image / no-image) settings
  - `generate_and_evaluate_*.sh`: run inference and evaluation
- `diff_embeds_image_vs_no_image.py`: compute steering + calibration vectors from saved embeddings
- `steer_qwen3_vl_vllm.py`, `steer_glm4_1_vllm.py`: custom vLLM model for Qwen3-VL-Thinking and GLM-4.1V-9B-Thinking
- `utils.py`, `visualization_tools.py`: helper utilities

---

## Environment Requirements

This codebase requires **vLLM 0.11.1**.

Later versions of vLLM are incompatible with the custom model implementations used in this repository.