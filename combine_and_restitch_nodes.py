import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageOps

class CombineImagesAndMask:
    """Combine base image, base mask, and reference image side-by-side for inpainting workflows.

    Inputs:
        base_image (IMAGE): Original base image.
        base_mask (IMAGE): Binary or grayscale mask for the base image.
        reference_image (IMAGE): Image to be placed alongside the base.
        pixel_buffer (int): Pixel spacing between the base crop and the reference image.
        orientation (str): "left", "top", or "auto".
        matching_method (str): "scale" or "pad" when dims don't align.
        resize (bool): Whether to resize the image to a target size.
        size_target (float): Target size in megapixels (Mpx).
        size_criteria (str): Which region to target: "base", "reference", or "total".
        crop_scale (float): Factor to expand the crop area.
        scale_overflow_strategy (str): "cap" or "extend".
        interpolation_mode (str): "nearest", "bilinear", or "bicubic".
    Outputs:
        combined_image (IMAGE): Side-by-side combined image.
        combined_mask (IMAGE): Combined mask aligned with base crop.
        restitch_data (DICT): Data for reconstructing the base image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "base_mask": ("MASK",),
                "reference_image": ("IMAGE",),
                "pixel_buffer": ("INT", {"default": 10, "min": 0, "max": 1000, "step": 1}),
                "orientation": (["auto", "left", "top"],),
                "matching_method": (["scale", "pad"],),
                "resize": ("BOOLEAN", {"default": True}),
                "size_target": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "size_criteria": (["base", "reference", "total"],),
                "crop_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "scale_overflow_strategy": (["cap", "extend"],),
                "interpolation_mode": (["nearest", "bilinear", "bicubic"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "DICT")
    FUNCTION = "combine"
    CATEGORY = "Custom"

    def combine(self, base_image, base_mask, reference_image, pixel_buffer, orientation,
                matching_method, size_target, size_criteria, crop_scale,
                scale_overflow_strategy, interpolation_mode):
        
        # Torch dimensions for images: B, H, W, C
        # Torch dimensions for masks: B, H, W
        restitch_data = {
            "original_base_image": base_image,
            "original_base_mask": base_mask,
        }

        # Mask should be a 1xHxW tensor, its contents affect image dimensions in the future so we cannot batch that
        #   so we'll just accept only batch size 1 for now
        if base_mask.shape[0] != 1:
            raise ValueError("Mask batch size must be 1")
        # Base image and reference image can be any batch size, but if both are not one, they need to be the same
        if base_image.shape[0] != reference_image.shape[0] and base_image.shape[0] != 1 and reference_image.shape[0] != 1:
            raise ValueError("Base image and reference image batch size must be the same or 1")

        # Find the minimum square covering the entirety of the masked area
        # Get mask as numpy array for processing
        mask_np = base_mask[0].cpu().numpy()
        
        # Find non-zero mask coordinates
        mask_coords = np.nonzero(mask_np)
        if len(mask_coords[0]) == 0:
            raise ValueError("Mask is empty")
            
        # Get bounding box
        min_y, max_y = np.min(mask_coords[0]), np.max(mask_coords[0])
        min_x, max_x = np.min(mask_coords[1]), np.max(mask_coords[1])
        
        # Calculate center point
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2
        
        # Calculate dimensions
        height = max_y - min_y + 3
        width = max_x - min_x + 3
        
        # Make square by taking max dimension
        crop_size = max(height, width)
        
        # Apply crop scale factor
        crop_size = int(crop_size * crop_scale)
        
        # Calculate crop boundaries, we can overflow depending on the scale overflow strategy, so first we'll calculate the virtual boundaries
        half_size = crop_size // 2
        crop_y1 = center_y - half_size
        crop_y2 = center_y + half_size
        crop_x1 = center_x - half_size
        crop_x2 = center_x + half_size

        # Unify batch dimensions by repeating the first dimenions
        target_batch_size = max(base_image.shape[0], reference_image.shape[0])
        if target_batch_size > 1:
            base_image = base_image.repeat(target_batch_size // base_image.shape[0], 1, 1, 1)
            reference_image = reference_image.repeat(target_batch_size // reference_image.shape[0], 1, 1, 1)

        # Overflow strategy Cap: Cap the boundaries to the image dimensions
        if scale_overflow_strategy == "cap":
            crop_y1 = max(0, crop_y1)
            crop_y2 = min(base_image.shape[1], crop_y2)
            crop_x1 = max(0, crop_x1)
            crop_x2 = min(base_image.shape[2], crop_x2)
            cropped_image = base_image[:, crop_y1:crop_y2, crop_x1:crop_x2, :]
            cropped_mask = base_mask[:, crop_y1:crop_y2, crop_x1:crop_x2]
        # Overflow strategy Extend: Pretend the image goes on forever by making a blank image of the expected size and then filling it with whatever we can crop
        elif scale_overflow_strategy == "extend":
            # Create a blank image of the expected size
            cropped_image = torch.ones((1, crop_size, crop_size, 3)) * 0.5
            cropped_mask = torch.zeros((1, crop_size, crop_size))
            
            # Fill the virtual image with the cropped region
            actual_crop_y1 = max(0, crop_y1)
            actual_crop_y2 = min(base_image.shape[1], crop_y2)
            actual_crop_x1 = max(0, crop_x1)
            actual_crop_x2 = min(base_image.shape[2], crop_x2)

            # Where to put the cropped region
            paste_y = -min(crop_y1, 0)
            paste_x = -min(crop_x1, 0)

            cropped_image[:, paste_y:paste_y+actual_crop_y2-actual_crop_y1, paste_x:paste_x+actual_crop_x2-actual_crop_x1, :] = base_image[:, actual_crop_y1:actual_crop_y2, actual_crop_x1:actual_crop_x2, :]
            cropped_mask[:, paste_y:paste_y+actual_crop_y2-actual_crop_y1, paste_x:paste_x+actual_crop_x2-actual_crop_x1] = base_mask[:, actual_crop_y1:actual_crop_y2, actual_crop_x1:actual_crop_x2]

        restitch_data["base_crop_box"] = [crop_x1, crop_y1, crop_x2, crop_y2]

        # Orientation auto: Automatically determine the best orientation based on the relative aspect ratios of the cropped image and the reference image
        #   If the reference image is relatively wider than the cropped image, we'll orient the cropped image to the top
        #   If the reference image is relatively taller than the cropped image, we'll orient the cropped image to the left
        if orientation == "auto":
            if reference_image.shape[2] / reference_image.shape[1] > cropped_image.shape[2] / cropped_image.shape[1]:
                orientation = "top"
            else:
                orientation = "left"

        # Resampler
        resampler = Image.NEAREST if interpolation_mode == "nearest" else Image.BILINEAR if interpolation_mode == "bilinear" else Image.BICUBIC

        # Match the dimensions of the reference image to the cropped image
        # Matching method scale: Scale the reference image to match the cropped image
        if matching_method == "scale":
            if orientation == "top": # Resample the reference image to match the width of the cropped image
                target_height = int(round(cropped_image.shape[2] * reference_image.shape[1] / reference_image.shape[2]))
                target_width = cropped_image.shape[2]
                reference_image = reference_image.permute(0, 3, 1, 2)
                reference_image = torch.nn.functional.interpolate(reference_image, size=(target_height, target_width), mode=interpolation_mode)
                reference_image = reference_image.permute(0, 2, 3, 1)
            elif orientation == "left": # Resample the reference image to match the height of the cropped image
                target_width = int(round(cropped_image.shape[1] * reference_image.shape[2] / reference_image.shape[1]))
                target_height = cropped_image.shape[1]
                reference_image = reference_image.permute(0, 3, 1, 2)
                reference_image = torch.nn.functional.interpolate(reference_image, size=(target_height, target_width), mode=interpolation_mode)
                reference_image = reference_image.permute(0, 2, 3, 1)

        # Create the combined image canvas, grey background
        if orientation == "top":
            # Use actual dimensions from both images
            canvas_height = reference_image.shape[1] + cropped_image.shape[1] + pixel_buffer
            canvas_width = max(reference_image.shape[2], cropped_image.shape[2])
            combined_image = torch.ones((base_image.shape[0], canvas_height, canvas_width, 3)) * 0.5
        elif orientation == "left":
            # Use actual dimensions from both images
            canvas_height = max(reference_image.shape[1], cropped_image.shape[1])
            canvas_width = reference_image.shape[2] + cropped_image.shape[2] + pixel_buffer
            combined_image = torch.ones((base_image.shape[0], canvas_height, canvas_width, 3)) * 0.5

        # Place reference image in top/left corner
        combined_image[:, :reference_image.shape[1], :reference_image.shape[2], :] = reference_image

        # Place cropped image in bottom/right corner with proper offset
        if orientation == "top":
            y_offset = reference_image.shape[1] + pixel_buffer
            combined_image[:, y_offset:y_offset + cropped_image.shape[1], :cropped_image.shape[2], :] = cropped_image
        else:  # left orientation
            x_offset = reference_image.shape[2] + pixel_buffer
            combined_image[:, :cropped_image.shape[1], x_offset:x_offset + cropped_image.shape[2], :] = cropped_image

        # Create the combined mask canvas, black background
        combined_mask = torch.zeros((base_image.shape[0], canvas_height, canvas_width))
        # Place the cropped mask in the bottom/right corner with proper offset
        if orientation == "top":
            y_offset = reference_image.shape[1] + pixel_buffer
            combined_mask[:, y_offset:y_offset + cropped_mask.shape[1], :cropped_mask.shape[2]] = cropped_mask
            # Crop box: x1, y1, x2, y2
            restitch_data["combined_crop_box"] = [0, y_offset, cropped_mask.shape[2], y_offset + cropped_mask.shape[1]]
        else:  # left orientation
            x_offset = reference_image.shape[2] + pixel_buffer
            combined_mask[:, :cropped_mask.shape[1], x_offset:x_offset + cropped_mask.shape[2]] = cropped_mask
            # Crop box: x1, y1, x2, y2
            restitch_data["combined_crop_box"] = [x_offset, 0, x_offset + cropped_mask.shape[2], cropped_mask.shape[1]]

        # Rescale the combined image and mask to the target size
        if resize:
            if size_criteria == "base":
                base_pixel_count = cropped_image.shape[1] * cropped_image.shape[2]
                target_pixel_count = size_target * 1024 * 1024 # In megapixels
                scale_factor = np.sqrt(target_pixel_count / base_pixel_count)
            elif size_criteria == "reference":
                reference_pixel_count = reference_image.shape[1] * reference_image.shape[2]
                target_pixel_count = size_target * 1024 * 1024 # In megapixels
                scale_factor = np.sqrt(target_pixel_count / reference_pixel_count)
            elif size_criteria == "total":
                total_pixel_count = combined_image.shape[1] * combined_image.shape[2]
                target_pixel_count = size_target * 1024 * 1024 # In megapixels
                scale_factor = np.sqrt(target_pixel_count / total_pixel_count)
        else:
            scale_factor = 1.0

        restitch_data["original_combined_size"] = combined_image.shape[1:3]
        restitch_data["original_combined_mask"] = combined_mask
            
        # Output dimensions should divide 8
        target_height = int(round(combined_image.shape[1] * scale_factor / 8) * 8)
        target_width = int(round(combined_image.shape[2] * scale_factor / 8) * 8)

        # Rescale the combined image and mask
        combined_image = combined_image.permute(0, 3, 1, 2)
        # combined_image = torch.nn.functional.interpolate(combined_image, size=(target_height, target_width), mode=interpolation_mode)
        combined_image = F.resize(combined_image, size=(target_height, target_width), interpolation=resampler, antialias=True)
        combined_image = combined_image.permute(0, 2, 3, 1)
        
        # Add channel dimension for mask interpolation
        combined_mask = combined_mask.unsqueeze(1)  # Add channel dimension
        # combined_mask = torch.nn.functional.interpolate(combined_mask, size=(target_height, target_width), mode=interpolation_mode)
        combined_mask = F.resize(combined_mask, size=(target_height, target_width), interpolation=resampler, antialias=True)
        combined_mask = combined_mask.squeeze(1)  # Remove channel dimension

        restitch_data["interpolation_mode"] = interpolation_mode

        return combined_image, combined_mask, restitch_data

class RestitchCropNode:
    """Restore a combined image segment back into the original base image.

    Inputs:
        combined_image (IMAGE): Side-by-side combined image.
        restitch_data (DICT): Metadata from CombineImagesAndMask node.
    Output:
        restored_base (IMAGE): Reconstructed base image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "combined_image": ("IMAGE",),
                "restitch_data": ("DICT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restitch"
    CATEGORY = "Custom"

    def restitch(self, combined_image, restitch_data):
        original_base_image = restitch_data["original_base_image"]

        # Step zero, batch size matching
        #   If one size is one and the other is greater than one, repeat the smaller one to match the larger one
        if combined_image.shape[0] == 1 and original_base_image.shape[0] > 1:
            combined_image = combined_image.repeat(original_base_image.shape[0], 1, 1, 1)
        elif combined_image.shape[0] > 1 and original_base_image.shape[0] == 1:
            original_base_image = original_base_image.repeat(combined_image.shape[0], 1, 1, 1)
            
            

        # Step one, rescale the combined image and mask to the original size
        combined_image = combined_image.permute(0, 3, 1, 2)
        combined_image = F.resize(combined_image, size=(restitch_data["original_combined_size"][0], restitch_data["original_combined_size"][1]), interpolation=resampler, antialias=True)
        #combined_image = torch.nn.functional.interpolate(combined_image, size=(restitch_data["original_combined_size"][0], restitch_data["original_combined_size"][1]), mode=restitch_data["interpolation_mode"])
        combined_image = combined_image.permute(0, 2, 3, 1)

        # combined_mask = restitch_data["original_combined_mask"]

        # Step two, uncrop the base image from the combined image
        # Crop box: x1, y1, x2, y2
        cropped_base_image = combined_image[:, restitch_data["combined_crop_box"][1]:restitch_data["combined_crop_box"][3], restitch_data["combined_crop_box"][0]:restitch_data["combined_crop_box"][2], :]
        # cropped_base_mask = combined_mask[:, restitch_data["combined_crop_box"][1]:restitch_data["combined_crop_box"][3], restitch_data["combined_crop_box"][0]:restitch_data["combined_crop_box"][2]]

        # Step three, handle padding for base cropping beyond original image dimensions
        # Crop box: x1, y1, x2, y2

        crop_x1 = restitch_data["base_crop_box"][0]
        crop_y1 = restitch_data["base_crop_box"][1]
        crop_x2 = restitch_data["base_crop_box"][2]
        crop_y2 = restitch_data["base_crop_box"][3]
        
        # Left padding
        if crop_x1 < 0:
            cropped_base_image = cropped_base_image[:, :, -crop_x1:, :]
            # cropped_base_mask = cropped_base_mask[:, :, -crop_x1:]
            crop_x1 = 0
        # Top padding
        if crop_y1 < 0:
            cropped_base_image = cropped_base_image[:, -crop_y1:, :, :]
            # cropped_base_mask = cropped_base_mask[:, -crop_y1:, :]
            crop_y1 = 0
        # Right padding 
        if crop_x2 > original_base_image.shape[2]:
            pad_size = crop_x2 - original_base_image.shape[2]
            cropped_base_image = cropped_base_image[:, :, :-pad_size, :]
            # cropped_base_mask = cropped_base_mask[:, :, :-pad_size]
            crop_x2 = original_base_image.shape[2]
        # Bottom padding
        if crop_y2 > original_base_image.shape[1]:
            pad_size = crop_y2 - original_base_image.shape[1]
            cropped_base_image = cropped_base_image[:, :-pad_size, :, :]
            # cropped_base_mask = cropped_base_mask[:, :-pad_size, :]
            crop_y2 = original_base_image.shape[1]

        # Step four, uncrop the cropped base image into the original base image
        expanded_cropped_base_image = torch.zeros((cropped_base_image.shape[0], original_base_image.shape[1], original_base_image.shape[2], 3))
        # expanded_cropped_base_mask = torch.zeros((1, restitch_data["original_base_image"].shape[1], restitch_data["original_base_image"].shape[2]))

        expanded_cropped_base_image[:, crop_y1:crop_y2, crop_x1:crop_x2, :] = cropped_base_image
        # expanded_cropped_base_mask[:, restitch_data["base_crop_box"][1]:restitch_data["base_crop_box"][3], restitch_data["base_crop_box"][0]:restitch_data["base_crop_box"][2]] = cropped_base_mask

        # Mask application
        # Image dimensions: B, H, W, C
        # Mask dimensions: 1, H, W
        mask = restitch_data["original_base_mask"].unsqueeze(0).clone().permute(0, 2, 3, 1)

        final_image = expanded_cropped_base_image * mask + original_base_image * (1 - mask)

        return final_image, {}




NODE_CLASS_MAPPINGS = {
    "CombineImagesAndMask": CombineImagesAndMask,
    "RestitchCrop": RestitchCropNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CombineImagesAndMask": "Combine Images and Mask",
    "RestitchCrop": "Restitch Combined Crop"
}
