"""Image transforms for VJEPA AC training."""

from typing import Any

import torch


class DroidRandomResizedCrop(torch.nn.Module):
    """DROID-style random resized crop.

    This is used for VJEPA AC training with the following parameters:
    - Fixed scale: 1.777 (aspect ratio varies)
    - Variable aspect ratio: [0.75, 1.35]
    - Output size: crop_size x crop_size

    This differs from torchvision's RandomResizedCrop which samples both scale and aspect ratio.
    Here scale is fixed and only aspect ratio varies.

    Args:
        scale: Fixed scale value (default: 1.777 from DROID)
        ratio: Aspect ratio range (default: (0.75, 1.35) from DROID)
        target_size: Output size (default: 256)
    """

    def __init__(
        self,
        scale: float = 1.777,
        ratio: tuple[float, float] = (0.75, 1.35),
        target_size: int = 256,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.ratio = ratio
        self.target_size = target_size

    def _get_spatial_crop_params(self, image: torch.Tensor) -> tuple[int, int, int, int]:
        h, w = image.shape[-2:]
        area = h * w
        target_area = area * self.scale

        for _ in range(10):
            aspect_ratio = torch.empty(1).uniform_(self.ratio[0], self.ratio[1]).item()
            crop_h = int(round((target_area * aspect_ratio) ** 0.5))
            crop_w = int(round((target_area / aspect_ratio) ** 0.5))

            if 0 < crop_h <= h and 0 < crop_w <= w:
                top = torch.randint(0, h - crop_h + 1, (1,)).item()
                left = torch.randint(0, w - crop_w + 1, (1,)).item()
                return top, left, crop_h, crop_w

        return 0, 0, h, w

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1
        outputs = []

        for inp in inputs:
            if inp.dim() < 3:
                raise ValueError(f"Expected input with at least 3 dimensions, got {inp.dim()}")
            original_dim = inp.dim()

            if inp.dim() == 4:
                # Video case: [T, C, H, W]
                T, C, H, W = inp.shape
                inp = inp.view(T * C, H, W)  # Flatten time and channels
                inp = inp.unsqueeze(0)  # Add batch dimension: [1, T*C, H, W]
            elif inp.dim() == 3:
                # Single image case: [C, H, W]
                inp = inp.unsqueeze(0)  # Add batch dimension: [1, C, H, W]

            top, left, crop_h, crop_w = self._get_spatial_crop_params(inp)

            cropped = inp[..., top : top + crop_h, left : left + crop_w]
            resized = torch.nn.functional.interpolate(
                cropped,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # Remove batch dimension

            if original_dim == 4:
                # Reshape back to video: [T, C, target_size, target_size]
                resized = resized.view(T, C, self.target_size, self.target_size)

            outputs.append(resized)

        return outputs if needs_unpacking else outputs[0]
