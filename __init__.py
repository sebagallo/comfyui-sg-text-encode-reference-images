from .nodes import TextEncodeReferenceImages
from comfy_api.latest import ComfyExtension

class SgTextEncodeRefExtension(ComfyExtension):
    async def get_node_list(self) -> list[type]:
        return [TextEncodeReferenceImages]

async def comfy_entrypoint() -> SgTextEncodeRefExtension:
    return SgTextEncodeRefExtension()

NODE_CLASS_MAPPINGS = {
    "TextEncodeReferenceImages": TextEncodeReferenceImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeReferenceImages": "TextEncodeReferenceImages",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "comfy_entrypoint"]
