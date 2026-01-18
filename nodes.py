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
                "enable_vl": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "vae": ("VAE",),
                "llama_template": ("STRING", {"multiline": False}),
                "image_prompt": ("STRING", {"multiline": False}),
                "image1": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"
    
    def encode(self, clip, prompt, enable_vl=True, vae=None, llama_template=None, image_prompt=None, **kwargs):
        ref_latents = []
        # Get all images from kwargs, sorted by name (image1, image2, etc.)
        image_keys = sorted([k for k in kwargs.keys() if k.startswith("image")])
        images = [kwargs[k] for k in image_keys if kwargs[k] is not None]
        images_vl = []
        
        if not llama_template or llama_template.strip() == "":
            llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        
        final_image_prompt = ""
        
        # Process all provided images
        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1) # B, C, H, W
                
                if enable_vl:
                    images_vl.append(image) # Use native resolution
                    if image_prompt and image_prompt.strip() != "":
                        # If a custom image prompt is provided, we use it. 
                        # We might need to handle per-image prompts if we wanted to be fancy, 
                        # but for now we'll use the provided one or the default format.
                        final_image_prompt += image_prompt.format(i + 1)
                    else:
                        final_image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
                
                if vae is not None:
                    ref_latents.append(vae.encode(samples.movedim(1, -1)[:, :, :, :3]))
        
        tokens = clip.tokenize(
            final_image_prompt + prompt,
            images=images_vl if enable_vl and len(images_vl) > 0 else None,
            llama_template=llama_template if enable_vl else None
        )
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        
        return (conditioning,)

