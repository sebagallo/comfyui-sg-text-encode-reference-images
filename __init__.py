from .nodes import TextEncodeReferenceImages

NODE_CLASS_MAPPINGS = {
    "TextEncodeReferenceImages": TextEncodeReferenceImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeReferenceImages": "TextEncodeReferenceImages",
}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
