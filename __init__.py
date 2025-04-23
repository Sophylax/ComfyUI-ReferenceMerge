from .combine_and_restitch_nodes import CombineImagesAndMask, RestitchCropNode

# map your class names to the actual Python classes
NODE_CLASS_MAPPINGS = {
    "CombineImagesAndMask": CombineImagesAndMask,
    "RestitchCrop": RestitchCropNode,
}

# human‑friendly display names in the node picker
NODE_DISPLAY_NAME_MAPPINGS = {
    "CombineImagesAndMask": "Combine Images and Mask",
    "RestitchCrop": "Restitch Combined Crop",
}

# tell Python what to export
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
