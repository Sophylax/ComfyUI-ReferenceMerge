import numpy as np
import torch
import torchvision.transforms.functional as F
import math
from PIL import Image, ImageOps

class ReferenceInpaintCompositeNode:
    """
    Given a base image, a mask on the base image, and a reference image, create a composite image/mask pair for inpainting.
    Given that the output dimensions can depend on the mask contents, we cannot batch multiple masks at once.
    We can however batch multiple base images and reference images with the same mask.

    This node expects a batch of base images, reference images, and masks.
        Spatial dimensions of base image batch and the mask batch must match. Reference image spatial dimensions can be different.
        Mask should have a single batch dimension. If the base and reference images have different batch dimensions, one of them should have a batch dimension of 1.
        If necessary, the batch dimensions will be repeated to match the non-zero batch dimension.

    Inputs:
        base_image (IMAGE): Original base image.
        base_mask (MASK): Binary or grayscale mask for the base image.
        reference_image (IMAGE): Image to be placed alongside the base.
        pixel_buffer (int): Pixel spacing between the base crop and the reference image.
        orientation (str): "left", "top", or "auto".
        crop_scale (float): Factor to expand the crop area.
        crop_overflow (bool): Whether to allow the crop to overflow the base image dimensions.
        crop_square (bool): Whether to crop the base image as a square.
        scale_reference (bool): Whether to scale the reference image to match the cropped base image dimensions.
        repeat_base_image (bool): Whether to repeat the base image unmasked for base image, on the opposite side of the reference image.
        resize (bool): Whether to resize the image to a target size.
        resize_target (float): Target size in megapixels (Mpx).
        resize_region (str): Which region to target: "base", "reference", or "total".
        interpolation_mode (str): "nearest", "bilinear", or "bicubic".
    Outputs:
        combined_image (IMAGE): Side-by-side combined image.
        combined_mask (IMAGE): Combined mask aligned with base crop.
        reconstruction_data (DICT): Data for reintegrating the cropped base image from the composite image into the original base image.

    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "base_mask": ("MASK",),
                "reference_image": ("IMAGE",),
                "pixel_buffer": ("INT", {"default": 8, "min": 0, "max": 128, "step": 8}),
                "orientation": (["auto", "left", "top"],),
                "crop_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "crop_overflow": ("BOOLEAN", {"default": False}),
                "crop_square": ("BOOLEAN", {"default": False}),
                "scale_reference": ("BOOLEAN", {"default": True}),
                "repeat_base_image": ("BOOLEAN", {"default": False}),
                "resize": ("BOOLEAN", {"default": True}),
                "resize_target": ("FLOAT", {"default": 1.0, "min": 0, "max": 16, "step": 0.05}),
                "resize_region": (["base", "reference", "total"],),
                "interpolation_mode": (["nearest", "bilinear", "bicubic"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "DICT")
    FUNCTION = "composite"
    CATEGORY = "Reference Inpainting"

    def composite(self, base_image: torch.Tensor, base_mask: torch.Tensor, reference_image: torch.Tensor, pixel_buffer: int, orientation: str,
                        crop_scale: float, crop_overflow: bool, crop_square: bool, scale_reference: bool, repeat_base_image: bool,
                        resize: bool, resize_target: float, resize_region: str, interpolation_mode: str):

        # Torch dimensions for input images: B, H, W, C
        # Torch dimensions for input masks: B, H, W

        # Mask should be a 1xHxW tensor, its contents affect the output dimensions so we cannot accept different masks in a single batch
        if base_mask.shape[0] != 1:
            raise ValueError("Mask batch size must be 1")
        
        # Ensure the spatial dimensions of the base image and mask match
        if base_image.shape[1] != base_mask.shape[1] or base_image.shape[2] != base_mask.shape[2]:
            raise ValueError("Base image and mask spatial dimensions must match")

        # Ensure that the batch sizes of the base image and reference image match OR that at least one of them is 1
        if base_image.shape[0] != reference_image.shape[0] and base_image.shape[0] != 1 and reference_image.shape[0] != 1:
            raise ValueError("Base image and reference image batch size must match or be 1")

        reconstruction_data = {}

        # Deepcopy inputs to avoid modifying the original images
        base_image = torch.clone(base_image)
        base_mask = torch.clone(base_mask)
        reference_image = torch.clone(reference_image)

        # Permute images to B, C, H, W
        base_image = base_image.permute(0, 3, 1, 2)
        reference_image = reference_image.permute(0, 3, 1, 2)

        # Batch repeat if necessary
        if base_image.shape[0] == 1 and reference_image.shape[0] > 1:
            base_image = base_image.repeat(reference_image.shape[0], 1, 1, 1)
        elif reference_image.shape[0] == 1 and base_image.shape[0] > 1:
            reference_image = reference_image.repeat(base_image.shape[0], 1, 1, 1)

        # Also repeat the mask if necessary
        if base_image.shape[0] > 1:
            base_mask = base_mask.repeat(base_image.shape[0], 1, 1)

        reconstruction_data["base_image"] = torch.clone(base_image)
        reconstruction_data["base_mask"] = torch.clone(base_mask)

        # Get the bounding box of the base mask (only the first mask is used, the others are repeats anyway)
        base_mask_np = base_mask[0].cpu().numpy()
        mask_coords = np.nonzero(base_mask_np)
        if len(mask_coords[0]) == 0:
            raise ValueError("Mask is empty")
        bb_y_min, bb_y_max = np.min(mask_coords[0]), np.max(mask_coords[0])
        bb_x_min, bb_x_max = np.min(mask_coords[1]), np.max(mask_coords[1])
        
        if crop_square:
        # Bounding square of the mask
            bb_size = max(bb_y_max - bb_y_min, bb_x_max - bb_x_min)
            bb_y_mid = (bb_y_min + bb_y_max) // 2
            bb_x_mid = (bb_x_min + bb_x_max) // 2
            bb_y_min = bb_y_mid - bb_size // 2
            bb_y_max = bb_y_min + bb_size
            bb_x_min = bb_x_mid - bb_size // 2
            bb_x_max = bb_x_min + bb_size

        # Expand the crop by crop scale
        x_center = (bb_x_min + bb_x_max) // 2
        y_center = (bb_y_min + bb_y_max) // 2
        width = bb_x_max - bb_x_min
        height = bb_y_max - bb_y_min
        scaled_width = math.ceil(width * crop_scale)
        scaled_height = math.ceil(height * crop_scale)
        bb_x_min = x_center - scaled_width // 2
        bb_x_max = bb_x_min + scaled_width
        bb_y_min = y_center - scaled_height // 2
        bb_y_max = bb_y_min + scaled_height

        if crop_overflow:
            # If the crop overflow is allowed, we might need to pad the base image
            pad_value = 0.5 # Gray background
            pad_left = max(0, -bb_x_min)
            pad_right = max(0, bb_x_max - base_image.shape[3])
            pad_top = max(0, -bb_y_min)
            pad_bottom = max(0, bb_y_max - base_image.shape[2])

            padding = [pad_left, pad_top, pad_right, pad_bottom]

            base_image = F.pad(base_image, padding, padding_mode="reflect")
            base_mask = F.pad(base_mask, padding, padding_mode="constant")

            # Shift the bounding box because the padding changed the coordinates
            bb_x_min += pad_left
            bb_x_max += pad_left
            bb_y_min += pad_top
            bb_y_max += pad_top

            reconstruction_data["padding"] = padding
        else:
            # If the crop overflow is not allowed, we need to ensure the crop is within the base image

            if not crop_square: # Easy mode: just cap the crop at the base image boundaries
                bb_x_min = max(0, bb_x_min)
                bb_x_max = min(base_image.shape[3], bb_x_max)
                bb_y_min = max(0, bb_y_min)
                bb_y_max = min(base_image.shape[2], bb_y_max)
            else:
                bb_size = max(bb_y_max - bb_y_min, bb_x_max - bb_x_min)

                # Sanity check: square size should not be greater than the base image dimensions
                if bb_size > base_image.shape[2] or bb_size > base_image.shape[3]:
                    raise ValueError("This mask cannot be cropped to a square without allowing overflow")

                if bb_x_min < 0:
                    bb_x_max -= bb_x_min
                    bb_x_min = 0
                if bb_y_min < 0:
                    bb_y_max -= bb_y_min
                    bb_y_min = 0
                if bb_x_max > base_image.shape[3]:
                    bb_x_min -= bb_x_max - base_image.shape[3]
                    bb_x_max = base_image.shape[3]
                if bb_y_max > base_image.shape[2]:
                    bb_y_min -= bb_y_max - base_image.shape[2]
                    bb_y_max = base_image.shape[2]

            reconstruction_data["padding"] = [0, 0, 0, 0]

        # Get the crop from the base image/mask
        base_image_crop = base_image[:, :, bb_y_min:bb_y_max, bb_x_min:bb_x_max]
        base_mask_crop = base_mask[:, bb_y_min:bb_y_max, bb_x_min:bb_x_max]

        reconstruction_data["base_image_crop_position"] = (bb_x_min, bb_y_min)

        original_base_image_crop_height = base_image_crop.shape[2]
        original_base_image_crop_width = base_image_crop.shape[3]

        # Resolve auto orientation
        if orientation == "auto":
            base_aspect_ratio = base_image_crop.shape[3] / base_image_crop.shape[2]
            reference_aspect_ratio = reference_image.shape[3] / reference_image.shape[2]
            if base_aspect_ratio > reference_aspect_ratio:
                orientation = "left"
            else:
                orientation = "top"

        # Resolve interpolation mode enum
        if interpolation_mode == "nearest":
            interpolation_mode = F.InterpolationMode.NEAREST
        elif interpolation_mode == "bilinear":
            interpolation_mode = F.InterpolationMode.BILINEAR
        elif interpolation_mode == "bicubic":
            interpolation_mode = F.InterpolationMode.BICUBIC

        reconstruction_data["interpolation_mode"] = interpolation_mode

        # Calculate reference dimensions, depending on the scale_reference flag
        if scale_reference:
            if orientation == "left": # Match height
                reference_height = base_image_crop.shape[2]
                reference_width = reference_height * reference_image.shape[3] // reference_image.shape[2]
            else: # Match width
                reference_width = base_image_crop.shape[3]
                reference_height = reference_width * reference_image.shape[2] // reference_image.shape[3]
        else:
            reference_height = reference_image.shape[2]
            reference_width = reference_image.shape[3]

        # Calculate rescale factor
        if resize:
            if resize_region == "base":
                region_size = base_image_crop.shape[2] * base_image_crop.shape[3]
                rescale_factor = resize_target * 1024 * 1024 / region_size
                rescale_factor = math.sqrt(rescale_factor)
            elif resize_region == "reference":
                region_size = reference_height * reference_width
                rescale_factor = resize_target * 1024 * 1024 / region_size
                rescale_factor = math.sqrt(rescale_factor)
            else: # resize_region == "total"
                if orientation == "left":
                    total_region_width = base_image_crop.shape[3] * (2 if repeat_base_image else 1) + reference_width
                    total_region_height = max(base_image_crop.shape[2], reference_height)
                else: # orientation == "top"
                    total_region_width = max(base_image_crop.shape[3], reference_width)
                    total_region_height = base_image_crop.shape[2] * (2 if repeat_base_image else 1) + reference_height
                region_size = total_region_width * total_region_height
                rescale_factor = resize_target * 1024 * 1024 / region_size
                rescale_factor = math.sqrt(rescale_factor)
        else:
            rescale_factor = 1.0

        rescaled_reference_height = math.floor(reference_height * rescale_factor)
        rescaled_reference_width = math.floor(reference_width * rescale_factor)
        rescaled_reference_image = F.resize(reference_image, (rescaled_reference_height, rescaled_reference_width), interpolation=interpolation_mode)
        rescaled_reference_mask = torch.zeros((reference_image.shape[0], rescaled_reference_height, rescaled_reference_width), device=base_mask_crop.device)

        reconstruction_data["original_base_image_crop_shape"] = (original_base_image_crop_height, original_base_image_crop_width)
        rescaled_base_image_crop_height = math.floor(original_base_image_crop_height * rescale_factor)
        rescaled_base_image_crop_width = math.floor(original_base_image_crop_width * rescale_factor)
        rescaled_base_image_crop = F.resize(base_image_crop, (rescaled_base_image_crop_height, rescaled_base_image_crop_width), interpolation=interpolation_mode)
        rescaled_base_mask_crop = F.resize(base_mask_crop, (rescaled_base_image_crop_height, rescaled_base_image_crop_width), interpolation=interpolation_mode)
        reconstruction_data["rescaled_base_image_crop_shape"] = (rescaled_base_image_crop_height, rescaled_base_image_crop_width)

        if orientation == "left":

            # Pad the reference image to match the base image crop height
            base_minus_ref_height = max(0, rescaled_base_image_crop_height - rescaled_reference_height)
            ref_pad_top = base_minus_ref_height // 2
            ref_pad_bottom = base_minus_ref_height - ref_pad_top

            rescaled_reference_image = F.pad(rescaled_reference_image, (0, ref_pad_top, 0, ref_pad_bottom), padding_mode="constant", fill=0.5)
            rescaled_reference_mask = F.pad(rescaled_reference_mask, (0, ref_pad_top, 0, ref_pad_bottom), padding_mode="constant", fill=0)

            # Pad the base image crop to match the reference image height
            ref_minus_base_height = max(0, rescaled_reference_height - rescaled_base_image_crop_height)
            base_pad_top = ref_minus_base_height // 2
            base_pad_bottom = ref_minus_base_height - base_pad_top

            rescaled_base_image_crop = F.pad(rescaled_base_image_crop, (pixel_buffer, base_pad_top, 0, base_pad_bottom), padding_mode="constant", fill=0.5)
            rescaled_base_mask_crop = F.pad(rescaled_base_mask_crop, (pixel_buffer, base_pad_top, 0, base_pad_bottom), padding_mode="constant", fill=0)

            reconstruction_data["rescaled_base_image_crop_position"] = (rescaled_reference_width + pixel_buffer, base_pad_top)

            final_image = torch.cat((rescaled_reference_image, rescaled_base_image_crop), dim=3)
            final_mask = torch.cat((rescaled_reference_mask, rescaled_base_mask_crop), dim=2)

            if repeat_base_image:
                final_image = torch.cat((final_image, rescaled_base_image_crop), dim=3)
                final_mask = torch.cat((final_mask, torch.zeros_like(rescaled_base_mask_crop)), dim=2)
        else:

            # Pad the reference image to match the base image crop width
            base_minus_ref_width = max(0, rescaled_base_image_crop_width - rescaled_reference_width)
            ref_pad_left = base_minus_ref_width // 2
            ref_pad_right = base_minus_ref_width - ref_pad_left

            rescaled_reference_image = F.pad(rescaled_reference_image, (ref_pad_left, 0, ref_pad_right, 0), padding_mode="constant", fill=0.5)
            rescaled_reference_mask = F.pad(rescaled_reference_mask, (ref_pad_left, 0, ref_pad_right, 0), padding_mode="constant", fill=0)

            # Pad the base image crop to match the reference image width
            ref_minus_base_width = max(0, rescaled_reference_width - rescaled_base_image_crop_width)
            base_pad_left = ref_minus_base_width // 2
            base_pad_right = ref_minus_base_width - base_pad_left

            rescaled_base_image_crop = F.pad(rescaled_base_image_crop, (base_pad_left, pixel_buffer, base_pad_right, 0), padding_mode="constant", fill=0.5)
            rescaled_base_mask_crop = F.pad(rescaled_base_mask_crop, (base_pad_left, pixel_buffer, base_pad_right, 0), padding_mode="constant", fill=0)

            reconstruction_data["rescaled_base_image_crop_position"] = (base_pad_left, rescaled_reference_height + pixel_buffer)

            final_image = torch.cat((rescaled_reference_image, rescaled_base_image_crop), dim=2)
            final_mask = torch.cat((rescaled_reference_mask, rescaled_base_mask_crop), dim=1)

            if repeat_base_image:
                final_image = torch.cat((final_image, rescaled_base_image_crop), dim=2)
                final_mask = torch.cat((final_mask, torch.zeros_like(rescaled_base_mask_crop)), dim=1)

        # Final padding on the final image to dimensions divisible by 8
        bottom_padding = 8 - final_image.shape[2] % 8
        right_padding = 8 - final_image.shape[3] % 8

        final_image = F.pad(final_image, (0, 0, right_padding, bottom_padding), padding_mode="constant", fill=0.5)
        final_mask = F.pad(final_mask, (0, 0, right_padding, bottom_padding), padding_mode="constant", fill=0)

        return torch.clone(final_image.permute(0, 2, 3, 1)), torch.clone(final_mask), reconstruction_data
            

class InpaintRegionRestitcherNode:
    """
    Extract a base image from a composite image, then restitch it back into the original base image using the reconstruction data.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "composite_image": ("IMAGE",),
                "reconstruction_data": ("DICT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restitch"
    CATEGORY = "Reference Inpainting"

    def restitch(self, composite_image: torch.Tensor, reconstruction_data: dict):
        
        # Torch dimensions for input images: B, H, W, C
        # Torch dimensions for input masks: B, H, W

        # Deepcopy inputs to avoid modifying the original images
        composite_image = torch.clone(composite_image)
        base_image = torch.clone(reconstruction_data["base_image"])
        base_mask = torch.clone(reconstruction_data["base_mask"])

        # Permute images to B, C, H, W
        composite_image = composite_image.permute(0, 3, 1, 2)

        # Batch size of the composite image should be a multiple of the base image batch size
        if composite_image.shape[0] % base_image.shape[0] != 0:
            raise ValueError("Composite image batch size must be a multiple of the base image batch size")
        
        base_image = base_image.repeat(composite_image.shape[0] // base_image.shape[0], 1, 1, 1)

        # Step One: crop the rescaled base image crop from the composite image
        composite_crop_position = reconstruction_data["rescaled_base_image_crop_position"]
        rescaled_crop_height = reconstruction_data["rescaled_base_image_crop_shape"][0]
        rescaled_crop_width = reconstruction_data["rescaled_base_image_crop_shape"][1]

        rescaled_base_image_crop = composite_image[:, :, composite_crop_position[1]:composite_crop_position[1] + rescaled_crop_height, composite_crop_position[0]:composite_crop_position[0] + rescaled_crop_width]
        
        # Step Two: Unscale the base image crop
        original_base_image_crop_height = reconstruction_data["original_base_image_crop_shape"][0]
        original_base_image_crop_width = reconstruction_data["original_base_image_crop_shape"][1]
        
        original_base_image_crop = F.resize(rescaled_base_image_crop, (original_base_image_crop_height, original_base_image_crop_width), interpolation=reconstruction_data["interpolation_mode"])

        # Step Three: Unpad the base image crop in case overflow was allowed
        if reconstruction_data["padding"] != [0, 0, 0, 0]: # Left, Top, Right, Bottom
            unpad_left = reconstruction_data["padding"][0]
            unpad_top = reconstruction_data["padding"][1]
            unpad_right = original_base_image_crop.shape[3] - reconstruction_data["padding"][2]
            unpad_bottom = original_base_image_crop.shape[2] - reconstruction_data["padding"][3]
            original_base_image_crop = original_base_image_crop[:, :, unpad_top:unpad_bottom, unpad_left:unpad_right]

        # Step Four: Place cropped base image into a blank base image
        base_image_canvas = torch.zeros_like(base_image)
        base_image_crop_position = reconstruction_data["base_image_crop_position"]

        base_image_canvas[:, :, base_image_crop_position[1]:base_image_crop_position[1] + original_base_image_crop.shape[2], base_image_crop_position[0]:base_image_crop_position[0] + original_base_image_crop.shape[3]] = original_base_image_crop
       
        # Step Five: Apply the masked cropped base image to the original base image
        channeled_mask = base_mask.unsqueeze(1).repeat(1, 3, 1, 1)
        base_image_canvas = base_image_canvas * channeled_mask + base_image * (1 - channeled_mask)

        return torch.clone(base_image_canvas.permute(0, 2, 3, 1)), {}
