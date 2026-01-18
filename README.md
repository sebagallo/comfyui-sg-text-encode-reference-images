# ComfyUI Text Encode Reference Images

An advanced text encoding node for ComfyUI that supports a dynamic number of reference images.

Mainly for usage with Qwen Image / Flux 2 models.

![Preview](assets/preview.png)

## Nodes

### Text Encode Reference Images
This node allows you to provide a prompt along with multiple reference images that are dynamically handled and fed to the CLIP model. It supports Vision-Language (VL) templates and reference latents.

- **Category**: `conditioning`
- **Features**:
  - **Stable Dynamic Image Slots**: Automatically adds a new `image` input slot when the last one is connected. Intermediate slots remain stable even if disconnected, preventing naming collisions.
  - **Vision-Language Support**: Toggleable VL processing. Automatically generates image padding tokens (`<|vision_start|><|image_pad|><|vision_end|>`) for each image using native resolution.
  - **Customizable Templates**: Support for custom Llama templates and per-image prompts with `{}` placeholder for index.
  - **Reference Latents**: Optionally encodes input images using a VAE and appends them as `reference_latents` to the conditioning data.
- **Inputs**:
  - `clip`: The CLIP model to use for encoding.
  - `prompt`: The text instruction/prompt.
  - `enable_vl`: (BOOLEAN) Toggle Vision-Language image processing and template formatting.
  - `vae` (Optional): VAE for encoding reference latents.
  - `llama_template` (Optional): Custom system prompt template.
  - `image_prompt` (Optional): Custom prompt format for each image (e.g., `Picture {}: <|vision_start|><|image_pad|><|vision_end|>`).
  - `image1`, `image2`, ...: Dynamic image inputs.
- **Output**:
  - `CONDITIONING`: The encoded conditioning data.

## Installation

1. Place this folder in your ComfyUI `custom_nodes` directory.
2. Restart ComfyUI.
3. The node will appear in the `conditioning` category.
