import math
import torch
import comfy.utils
from comfy_extras import nodes_custom_sampler
from nodes import node_helpers
from comfy_api.latest import io

class TextEncodeReferenceImages(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TextEncodeReferenceImages",
            display_name="TextEncodeReferenceImages",
            category="conditioning",
            description="Encodes text with reference images for VL models and conditioning.",
            inputs=[
                io.Clip.Input("clip", tooltip="The CLIP model to use for encoding."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True, tooltip="The text prompt to encode."),
                io.Combo.Input("vl_selection", options=["none", "qwen image edit", "z-image base omni"], default="none", tooltip="Select the Vision-Language model strategy."),
                io.Boolean.Input("auto_resize_images", default=True, tooltip="Automatically resize reference images to 1024x1024."),
                io.Vae.Input("vae", optional=True, tooltip="VAE for encoding reference images into latents."),
                io.ClipVision.Input("image_encoder", optional=True, tooltip="CLIP Vision model for encoding reference images."),
                io.Image.Input("images", optional=True, tooltip="Reference images to be used for conditioning (supports image batches)."),
            ],
            outputs=[
                io.Conditioning.Output(),
            ]
        )
    
    @classmethod
    def execute(cls, clip, prompt, vl_selection="none", auto_resize_images=True, vae=None, image_encoder=None, images=None) -> io.NodeOutput:
        ref_latents = []
        
        # In V3, 'images' can be a batch. We convert it to a list for the loops below.
        if images is not None:
            # ComfyUI images are (B, H, W, C)
            if len(images.shape) == 4:
                images_list = [images[i:i+1] for i in range(images.shape[0])]
            else:
                images_list = [images]
        else:
            images_list = []
            
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
            if len(images_list) > 0:
                actual_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            
            for i, image in enumerate(images_list):
                images_vl.append(image)
                final_image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)
        
        elif vl_selection == "z-image base omni":
            # Omni strategy logic
            if len(images_list) > 0:
                prompt_list += ["<|im_start|>user\n<|vision_start|>"]
                prompt_list += ["<|vision_end|><|vision_start|>"] * (len(images_list) - 1)
                prompt_list += ["<|vision_end|><|im_end|>"]
                actual_template = "<|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"
        
        # Image processing: CLIP Vision and VAE encoding
        for i, image in enumerate(images_list):
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
        
        return io.NodeOutput(conditioning)


