from .reference_inpaint_nodes import ReferenceInpaintCompositeNode, InpaintRegionRestitcherNode

# map your class names to the actual Python classes
NODE_CLASS_MAPPINGS = {
    "ReferenceInpaintComposite": ReferenceInpaintCompositeNode,
    "InpaintRegionRestitcher": InpaintRegionRestitcherNode,
}

# human‑friendly display names in the node picker
NODE_DISPLAY_NAME_MAPPINGS = {
    "ReferenceInpaintComposite": "Reference Inpainting Composite",
    "InpaintRegionRestitcher": "Inpaint Region Restitcher",
}

# tell Python what to export
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
