"""
preprocessor.py
Agent 2 — Vision Specialist Agent
OpenCV preprocessing pipeline:
deskew → CLAHE contrast enhancement → DPI normalization
"""

import logging
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

TARGET_DPI = 300
TARGET_SIZE_MULTIPLIER = 300 / 72  # standard PDF render DPI


class Preprocessor:
    """
    Prepares scanned document images for OCR by applying:
    1. DPI normalization to 300 DPI
    2. Deskewing — corrects rotation from scanning
    3. CLAHE — contrast enhancement for stamps and low-contrast areas
    4. Denoising — removes scan artifacts
    """

    def __init__(
        self,
        target_dpi: int = TARGET_DPI,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: tuple = (8, 8)
    ):
        self.target_dpi = target_dpi
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid
        )

    def process(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Full preprocessing pipeline for a single document image.

        Args:
            image: PIL Image or numpy array (BGR or RGB)

        Returns:
            Preprocessed numpy array (BGR, 300 DPI equivalent)
        """
        # normalize to numpy BGR
        img = self._to_numpy_bgr(image)

        # step 1 — DPI normalization
        img = self._normalize_dpi(img)

        # step 2 — deskew
        img = self._deskew(img)

        # step 3 — CLAHE contrast enhancement
        img = self._apply_clahe(img)

        # step 4 — denoising
        img = self._denoise(img)

        return img

    def process_path(self, image_path: str) -> np.ndarray:
        """Load an image from path and preprocess it."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        return self.process(img)

    def _normalize_dpi(self, img: np.ndarray) -> np.ndarray:
        """
        Ensure image is at approximately 300 DPI.
        If image is small (likely low DPI scan), upscale it.
        """
        h, w = img.shape[:2]

        # heuristic: A4 at 300 DPI is ~2480x3508
        # if image is less than half that, it's likely < 150 DPI
        min_dimension = min(h, w)
        if min_dimension < 1000:
            scale = 2.0
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC
            )
            logger.debug(f"Upscaled image by {scale}x for DPI normalization")

        return img

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew angle using Hough line transform.
        Typical scan skew is < 5 degrees.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # binarize for line detection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # find contours and compute minimum area rectangles
        coords = np.column_stack(np.where(binary > 0))

        if len(coords) < 100:
            logger.debug("Not enough content for deskew — skipping")
            return img

        angle = cv2.minAreaRect(coords)[-1]

        # minAreaRect returns angles in [-90, 0)
        if angle < -45:
            angle = 90 + angle

        # only correct if skew is meaningful (> 0.3 degrees)
        if abs(angle) < 0.3:
            return img

        logger.debug(f"Deskewing by {angle:.2f} degrees")

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Enhances local contrast — especially useful for stamps and
        faded certificates.
        Applies per channel in LAB color space to avoid color shift.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # apply CLAHE only to the L (lightness) channel
        l_enhanced = self.clahe.apply(l_channel)

        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_bgr

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Apply non-local means denoising to remove scan artifacts.
        Conservative strength (h=7) to avoid blurring text.
        """
        denoised = cv2.fastNlMeansDenoisingColored(
            img,
            None,
            h=7,           # filter strength — lower = less aggressive
            hColor=7,
            templateWindowSize=7,
            searchWindowSize=21
        )
        return denoised

    def _to_numpy_bgr(
        self, image: Union[np.ndarray, Image.Image]
    ) -> np.ndarray:
        """Convert input to numpy BGR array."""
        if isinstance(image, Image.Image):
            # PIL is RGB — convert to BGR for OpenCV
            arr = np.array(image.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                # grayscale — convert to BGR
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            return image

        raise TypeError(f"Unsupported image type: {type(image)}")

    def to_pil(self, img: np.ndarray) -> Image.Image:
        """Convert preprocessed numpy BGR array back to PIL Image."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)