import math
import torch
import comfy.utils
from comfy_extras import nodes_custom_sampler
from nodes import node_helpers

class TextEncodeReferenceImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "vl_selection": (["none", "qwen image edit", "z-image base omni"], {"default": "none"}),
                "auto_resize_images": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "image_encoder": ("CLIP_VISION",),
                "image1": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"
    
    def encode(self, clip, prompt, vl_selection="none", auto_resize_images=True, vae=None, image_encoder=None, **kwargs):
        ref_latents = []
        # Get all images from kwargs, sorted by name (image1, image2, etc.)
        image_keys = sorted([k for k in kwargs.keys() if k.startswith("image")])
        images = [kwargs[k] for k in image_keys if kwargs[k] is not None]
        
        images_vl = []
        encoded_images = []
        extra_text_embeds = []
        prompt_list = []

        final_image_prompt = ""
        actual_template = None
        
        if vl_selection == "none":
            # No VL processing
            pass
            
        elif vl_selection == "qwen image edit":
            if len(images) > 0:
                actual_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            
            for i, image in enumerate(images):
                images_vl.append(image)
                final_image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
        
        elif vl_selection == "z-image base omni":
            # Omni strategy logic
            if len(images) > 0:
                prompt_list += ["<|im_start|>user\n<|vision_start|>"]
                prompt_list += ["<|vision_end|><|vision_start|>"] * (len(images) - 1)
                prompt_list += ["<|vision_end|><|im_end|>"]
                actual_template = "<|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"
        
        # Image processing: CLIP Vision and VAE encoding
        for i, image in enumerate(images):
            if image_encoder is not None:
                encoded_images.append(image_encoder.encode_image(image))

            if vae is not None:
                samples = image.movedim(-1, 1) # B, C, H, W
                if auto_resize_images:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    samples = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                
                ref_latents.append(vae.encode(samples.movedim(1, -1)[:, :, :, :3]))

        # Final encoding
        tokens = clip.tokenize(
            final_image_prompt + prompt,
            images=images_vl if len(images_vl) > 0 else None,
            llama_template=actual_template if vl_selection != "none" else None
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
            
        for p in prompt_list:
            t = clip.tokenize(p, llama_template="{}")
            text_embeds = clip.encode_from_tokens_scheduled(t)
            extra_text_embeds.append(text_embeds[0][0])

        if len(extra_text_embeds) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents_text_embeds": extra_text_embeds}, append=True)

        if len(encoded_images) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"clip_vision_outputs": encoded_images}, append=True)
            
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        
        return (conditioning,)

