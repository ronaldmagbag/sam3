#!/usr/bin/env python
"""
SAM3 Annotator - Create mask images with segmented data.
Supports multi-class segmentation with different colors per class.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Try to import cv2 for mask simplification
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] OpenCV (cv2) not available. Mask simplification will be disabled.")


# Default class colors - each class gets a distinct color
DEFAULT_CLASS_COLORS = {
    "building": (255, 0, 0),      # Red
    "road": (128, 128, 128),      # Gray
    "vehicle on the road": (128, 128, 128),
    "parking": (255, 165, 0),     # Orange
    "tree": (0, 128, 0),          # Dark Green
    "grass": (0, 255, 0),         # Bright Green
    "water": (0, 0, 255),         # Blue
    "car": (255, 255, 0),         # Yellow
    "sidewalk": (192, 192, 192),  # Silver
    "fence": (139, 69, 19),       # Brown
    "pool": (0, 191, 255),        # Deep Sky Blue
    "roof": (178, 34, 34),        # Firebrick
    "vegetation": (34, 139, 34),  # Forest Green
}

# Fallback colors for unknown classes
FALLBACK_COLORS = [
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (128, 0, 128),    # Purple
    (255, 192, 203),  # Pink
    (70, 130, 180),   # Steel Blue
    (0, 128, 128),    # Teal
    (128, 128, 0),    # Olive
    (255, 99, 71),    # Tomato
    (147, 112, 219),  # Medium Purple
    (60, 179, 113),   # Medium Sea Green
]


class SAM3Annotator:
    """
    SAM3-based image annotator for creating segmentation masks.
    
    Supports multi-class segmentation where each class gets a different color.
    """
    
    def __init__(self, device: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize SAM3 annotator.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
            confidence_threshold: Minimum confidence score for detections.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        
        print(f"[INFO] Initializing SAM3 Annotator on device: {self.device}")
        print("[INFO] Loading SAM3 model...")
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(
            self.model, 
            device=self.device, 
            confidence_threshold=confidence_threshold
        )
        print("[OK] SAM3 Annotator ready")
    
    def _get_class_color(self, class_name: str, class_index: int) -> Tuple[int, int, int]:
        """Get color for a class name."""
        class_name_lower = class_name.lower().strip()
        if class_name_lower in DEFAULT_CLASS_COLORS:
            return DEFAULT_CLASS_COLORS[class_name_lower]
        return FALLBACK_COLORS[class_index % len(FALLBACK_COLORS)]
    
    def _parse_classes(self, classes_str: str) -> List[str]:
        """Parse comma-separated class string into list."""
        return [c.strip() for c in classes_str.split(",") if c.strip()]
    
    def segment_image(self, image_path: str, text_prompt: str) -> Dict:
        """
        Segment an image using a single text prompt.
        
        Args:
            image_path: Path to the input image.
            text_prompt: Text description of objects to segment.
            
        Returns:
            Dictionary containing masks, boxes, scores, and original image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"[INFO] Loading image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        print(f"[INFO] Image size: {image.size[0]}x{image.size[1]}")
        
        print(f"[INFO] Segmenting with prompt: '{text_prompt}'")
        inference_state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(state=inference_state, prompt=text_prompt)
        
        masks = output.get("masks", torch.tensor([]))
        boxes = output.get("boxes", torch.tensor([]))
        scores = output.get("scores", torch.tensor([]))
        
        print(f"[INFO] Found {len(masks)} objects")
        
        return {
            "image": image,
            "masks": masks,
            "boxes": boxes,
            "scores": scores,
            "image_path": image_path,
            "text_prompt": text_prompt,
        }
    
    def segment_multiclass(
        self, 
        image_path: str, 
        classes: str,
        class_thresholds: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Segment an image with multiple classes, each getting a different color.
        
        Args:
            image_path: Path to the input image.
            classes: Comma-separated class names (e.g., "building, road, tree").
            class_thresholds: Dictionary mapping class names to confidence thresholds.
                            If None, uses default thresholds or self.confidence_threshold.
            
        Returns:
            Dictionary containing per-class masks and combined results.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        class_list = self._parse_classes(classes)
        if not class_list:
            raise ValueError("No valid classes provided")
        
        # Get default thresholds if not provided
        if class_thresholds is None:
            class_thresholds = self._get_default_class_thresholds()
        
        print(f"[INFO] Loading image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        print(f"[INFO] Image size: {image.size[0]}x{image.size[1]}")
        print(f"[INFO] Classes to segment: {class_list}")
        
        # Set image once
        inference_state = self.processor.set_image(image)
        
        # Segment each class
        class_results = {}
        for i, class_name in enumerate(class_list):
            # Get threshold for this class
            threshold = class_thresholds.get(class_name, self.confidence_threshold)
            
            print(f"[INFO] Segmenting class '{class_name}' (threshold: {threshold:.2f})...")
            
            # Set confidence threshold for this class
            self.processor.confidence_threshold = threshold
            
            # Reset prompts for new class
            self.processor.reset_all_prompts(inference_state)
            inference_state = self.processor.set_image(image)
            
            output = self.processor.set_text_prompt(state=inference_state, prompt=class_name)
            
            masks = output.get("masks", torch.tensor([]))
            boxes = output.get("boxes", torch.tensor([]))
            scores = output.get("scores", torch.tensor([]))
            
            color = self._get_class_color(class_name, i)
            
            print(f"[INFO]   Found {len(masks)} '{class_name}' objects (color: {color})")
            
            class_results[class_name] = {
                "masks": masks,
                "boxes": boxes,
                "scores": scores,
                "color": color,
                "threshold": threshold,
            }
        
        # Reset to default threshold
        self.processor.confidence_threshold = self.confidence_threshold
        
        return {
            "image": image,
            "image_path": image_path,
            "classes": class_list,
            "class_results": class_results,
        }
    
    def _get_default_class_thresholds(self) -> Dict[str, float]:
        """
        Get default confidence thresholds for common classes.
        Lower thresholds = more detections (good for trees, grass)
        Higher thresholds = fewer but more confident detections (good for buildings)
        """
        return {
            "building": 0.6,      # Higher - buildings should be clear
            "road": 0.3,         # Medium
            "parking": 0.5,      # Lower - parking lots can be varied
            "tree": 0.3,         # Lower - trees can be partially visible
            "grass": 0.3,        # Lower - grass can be sparse
            "water": 0.5,        # Medium
            "car": 0.5,          # Medium
            "sidewalk": 0.5,     # Medium
            "fence": 0.5,        # Medium
            "pool": 0.5,         # Medium
            "roof": 0.6,         # Higher - roofs should be clear
            "vegetation": 0.5,   # Lower - vegetation can be varied
        }
    
    def _get_class_simplify_method(self, class_name: str) -> str:
        """
        Get the appropriate simplification method for each class type.
        
        Returns:
            "polygon" for geometric/man-made structures
            "smooth" for organic/natural elements
        """
        class_name_lower = class_name.lower()
        
        # Geometric/man-made structures -> polygon simplification
        polygon_classes = [
            "building", "road", "parking", "sidewalk", "fence", 
            "pool", "roof", "car", "vehicle"
        ]
        
        # Organic/natural elements -> smooth simplification
        smooth_classes = [
            "tree", "grass", "vegetation", "water"
        ]
        
        # Check if class name contains any of the keywords
        for keyword in polygon_classes:
            if keyword in class_name_lower:
                return "polygon"
        
        for keyword in smooth_classes:
            if keyword in class_name_lower:
                return "smooth"
        
        # Default: use both methods
        return "both"
    
    def _simplify_mask(
        self,
        mask: np.ndarray,
        method: str = "polygon",
        epsilon: float = 5.0,
        smooth_kernel: int = 5,
        min_area: int = 100
    ) -> np.ndarray:
        """
        Simplify and smooth a binary mask.
        
        Args:
            mask: Binary mask (boolean or uint8, 0/255).
            method: Simplification method - "polygon", "smooth", or "both".
            epsilon: Polygon approximation epsilon (lower = more detailed).
            smooth_kernel: Kernel size for morphological smoothing (must be odd).
            min_area: Minimum area to keep a contour (removes small noise).
            
        Returns:
            Simplified binary mask.
        """
        if not CV2_AVAILABLE:
            return mask
        
        # Convert to uint8 if needed
        if mask.dtype == bool:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.astype(np.uint8)
        
        if method == "polygon" or method == "both":
            # Find contours
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Simplify each contour using polygon approximation
            simplified_contours = []
            for contour in contours:
                # Filter by area
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                
                # Approximate polygon
                epsilon_px = epsilon
                approx = cv2.approxPolyDP(contour, epsilon_px, closed=True)
                simplified_contours.append(approx)
            
            # Recreate mask from simplified contours
            h, w = mask_uint8.shape
            mask_simplified = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask_simplified, simplified_contours, 255)
            mask_uint8 = mask_simplified
        
        if method == "smooth" or method == "both":
            # Morphological operations to smooth boundaries
            kernel = np.ones((smooth_kernel, smooth_kernel), np.uint8)
            
            # Opening: removes small noise
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Closing: fills small holes
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Optional: Gaussian blur for smoother edges
            mask_uint8 = cv2.GaussianBlur(mask_uint8, (smooth_kernel, smooth_kernel), 0)
            _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        
        return mask_uint8.astype(bool)
    
    def create_multiclass_mask(
        self, 
        result: Dict,
        background_color: Tuple[int, int, int] = (0, 0, 0),
        simplify: bool = True,
        simplify_method: Optional[str] = None,
        simplify_epsilon: float = 2.0,
        smooth_kernel: int = 5,
        min_area: int = 100
    ) -> Image.Image:
        """
        Create a colored mask image where each class has a different color.
        
        Args:
            result: Segmentation result from segment_multiclass().
            background_color: RGB color for background.
            simplify: Whether to simplify/smooth masks.
            simplify_method: Method - "polygon", "smooth", or "both". 
                           If None, uses auto per-class method.
            simplify_epsilon: Polygon approximation epsilon (lower = more detailed).
            smooth_kernel: Kernel size for morphological smoothing (must be odd).
            min_area: Minimum area to keep a contour (removes small noise).
            
        Returns:
            PIL Image with colored masks per class.
        """
        image = result["image"]
        class_results = result["class_results"]
        
        width, height = image.size
        
        # Create RGB mask image
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
        colored_mask[:, :] = background_color
        
        # Apply each class's masks with its color
        for class_name, class_data in class_results.items():
            masks = class_data["masks"]
            color = class_data["color"]
            
            if len(masks) == 0:
                continue
            
            # Determine simplification method for this class
            if simplify_method is None:
                class_method = self._get_class_simplify_method(class_name)
            else:
                class_method = simplify_method
            
            # Combine all masks for this class
            for mask in masks:
                mask_np = mask.squeeze().cpu().numpy().astype(bool)
                
                # Simplify mask if requested
                if simplify:
                    mask_np = self._simplify_mask(
                        mask_np,
                        method=class_method,
                        epsilon=simplify_epsilon,
                        smooth_kernel=smooth_kernel,
                        min_area=min_area
                    )
                
                colored_mask[mask_np] = color
        
        return Image.fromarray(colored_mask)
    
    def create_multiclass_overlay(
        self,
        result: Dict,
        alpha: float = 0.5,
        show_boxes: bool = True,
        show_labels: bool = True,
        simplify: bool = True,
        simplify_method: Optional[str] = None,
        simplify_epsilon: float = 2.0,
        smooth_kernel: int = 5,
        min_area: int = 100
    ) -> Image.Image:
        """
        Create an overlay visualization with all class masks on the original image.
        
        Args:
            result: Segmentation result from segment_multiclass().
            alpha: Transparency of mask overlay.
            show_boxes: Whether to draw bounding boxes.
            show_labels: Whether to show class labels.
            simplify: Whether to simplify/smooth masks.
            simplify_method: Method - "polygon", "smooth", or "both".
                           If None, uses auto per-class method.
            simplify_epsilon: Polygon approximation epsilon (lower = more detailed).
            smooth_kernel: Kernel size for morphological smoothing (must be odd).
            min_area: Minimum area to keep a contour (removes small noise).
            
        Returns:
            PIL Image with mask overlay on original image.
        """
        image = result["image"].copy()
        class_results = result["class_results"]
        
        # Convert to numpy for manipulation
        img_array = np.array(image)
        
        # Apply each class's masks with its color
        for class_name, class_data in class_results.items():
            masks = class_data["masks"]
            color = np.array(class_data["color"])
            
            if len(masks) == 0:
                continue
            
            # Determine simplification method for this class
            if simplify_method is None:
                class_method = self._get_class_simplify_method(class_name)
            else:
                class_method = simplify_method
            
            for mask in masks:
                mask_np = mask.squeeze().cpu().numpy().astype(bool)
                
                # Simplify mask if requested
                if simplify:
                    mask_np = self._simplify_mask(
                        mask_np,
                        method=class_method,
                        epsilon=simplify_epsilon,
                        smooth_kernel=smooth_kernel,
                        min_area=min_area
                    )
                
                img_array[mask_np] = (
                    img_array[mask_np] * (1 - alpha) + color * alpha
                ).astype(np.uint8)
        
        # Convert back to PIL
        overlay_image = Image.fromarray(img_array)
        
        # Draw boxes and labels
        if show_boxes:
            draw = ImageDraw.Draw(overlay_image)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
                except:
                    font = ImageFont.load_default()
            
            for class_name, class_data in class_results.items():
                boxes = class_data["boxes"]
                scores = class_data["scores"]
                color = class_data["color"]
                
                if len(boxes) == 0:
                    continue
                
                for i in range(len(boxes)):
                    if isinstance(boxes[i], torch.Tensor):
                        box = boxes[i].cpu().numpy()
                    else:
                        box = np.array(boxes[i])
                    
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    
                    # Draw box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    # Draw label
                    if show_labels:
                        if len(scores) > i:
                            if isinstance(scores[i], torch.Tensor):
                                score = scores[i].item()
                            else:
                                score = float(scores[i])
                            label = f"{class_name}: {score:.2f}"
                        else:
                            label = class_name
                        
                        # Draw text background
                        try:
                            bbox = draw.textbbox((x1, y1 - 16), label, font=font)
                        except AttributeError:
                            text_size = draw.textsize(label, font=font)
                            bbox = (x1, y1 - 16, x1 + text_size[0], y1 - 16 + text_size[1])
                        
                        draw.rectangle(
                            [bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1],
                            fill=(0, 0, 0)
                        )
                        draw.text((x1, y1 - 16), label, fill=color, font=font)
        
        return overlay_image
    
    def _sanitize_prompt(self, prompt: str) -> str:
        """Sanitize text prompt to be safe for use in filenames."""
        import re
        sanitized = re.sub(r'^(a |an |the )', '', prompt.lower())
        sanitized = re.sub(r'[^a-z0-9]+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized or "object"
    
    def save_multiclass_outputs(
        self,
        result: Dict,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        simplify: bool = True,
        simplify_method: Optional[str] = None,
        simplify_epsilon: float = 5.0,
        smooth_kernel: int = 5,
        min_area: int = 100
    ) -> Dict[str, str]:
        """
        Save multi-class mask image to file.
        
        Args:
            result: Segmentation result from segment_multiclass().
            output_dir: Directory to save outputs.
            prefix: Prefix for output filenames.
            simplify: Whether to simplify/smooth masks.
            simplify_method: Method - "polygon", "smooth", or "both".
                           If None, uses auto per-class method (polygon for buildings/roads, smooth for trees/grass).
            simplify_epsilon: Polygon approximation epsilon (lower = more detailed).
            smooth_kernel: Kernel size for morphological smoothing (must be odd).
            min_area: Minimum area to keep a contour (removes small noise).
            
        Returns:
            Dictionary mapping output type to saved file path.
        """
        image_path = Path(result["image_path"])
        
        if output_dir is None:
            output_dir = image_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        if prefix is None:
            prefix = image_path.stem
        
        saved_files = {}
        
        # Save colored multi-class mask
        colored_mask = self.create_multiclass_mask(
            result,
            simplify=simplify,
            simplify_method=simplify_method,
            simplify_epsilon=simplify_epsilon,
            smooth_kernel=smooth_kernel,
            min_area=min_area
        )
        mask_path = output_dir / f"{prefix}_mask.png"
        colored_mask.save(mask_path)
        saved_files["mask"] = str(mask_path)
        print(f"[SAVED] Mask: {mask_path}")
        
        return saved_files
    
    def save_single_outputs(
        self,
        result: Dict,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save single-class mask image to file.
        
        Args:
            result: Segmentation result from segment_image().
            output_dir: Directory to save outputs.
            prefix: Prefix for output filenames.
            
        Returns:
            Dictionary mapping output type to saved file path.
        """
        image_path = Path(result["image_path"])
        text_prompt = result.get("text_prompt", "object")
        prompt_suffix = self._sanitize_prompt(text_prompt)
        
        if output_dir is None:
            output_dir = image_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        if prefix is None:
            prefix = image_path.stem
        
        saved_files = {}
        
        # Save binary mask (combined)
        binary_mask = self.create_binary_mask(result, combine_all=True)
        binary_path = output_dir / f"{prefix}_mask_{prompt_suffix}.png"
        binary_mask.save(binary_path)
        saved_files["mask"] = str(binary_path)
        print(f"[SAVED] Mask: {binary_path}")
        
        return saved_files
    
    def create_binary_mask(
        self, 
        result: Dict, 
        mask_index: Optional[int] = None,
        combine_all: bool = True
    ) -> Image.Image:
        """Create a binary mask image (black and white)."""
        masks = result["masks"]
        image = result["image"]
        
        if len(masks) == 0:
            print("[WARNING] No masks found, returning empty mask")
            return Image.new("L", image.size, 0)
        
        if mask_index is not None:
            if mask_index >= len(masks):
                raise IndexError(f"Mask index {mask_index} out of range")
            mask = masks[mask_index].squeeze().cpu().numpy()
        elif combine_all:
            combined = torch.zeros_like(masks[0].squeeze())
            for m in masks:
                combined = combined | m.squeeze()
            mask = combined.cpu().numpy()
        else:
            mask = masks[0].squeeze().cpu().numpy()
        
        binary_mask = (mask.astype(np.uint8) * 255)
        return Image.fromarray(binary_mask, mode="L")
    
    def create_single_overlay(
        self,
        result: Dict,
        alpha: float = 0.5,
        color: Tuple[int, int, int] = (255, 0, 0),
        show_boxes: bool = True,
        show_scores: bool = True
    ) -> Image.Image:
        """Create overlay for single-class segmentation."""
        image = result["image"].copy()
        masks = result["masks"]
        boxes = result["boxes"]
        scores = result["scores"]
        
        img_array = np.array(image)
        color_arr = np.array(color)
        
        if len(masks) > 0:
            for mask in masks:
                mask_np = mask.squeeze().cpu().numpy().astype(bool)
                img_array[mask_np] = (
                    img_array[mask_np] * (1 - alpha) + color_arr * alpha
                ).astype(np.uint8)
        
        overlay_image = Image.fromarray(img_array)
        
        if show_boxes and len(boxes) > 0:
            draw = ImageDraw.Draw(overlay_image)
            
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
                except:
                    font = ImageFont.load_default()
            
            for i in range(len(boxes)):
                if isinstance(boxes[i], torch.Tensor):
                    box = boxes[i].cpu().numpy()
                else:
                    box = np.array(boxes[i])
                
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                if show_scores and len(scores) > i:
                    if isinstance(scores[i], torch.Tensor):
                        score = scores[i].item()
                    else:
                        score = float(scores[i])
                    
                    label = f"{score:.2f}"
                    
                    try:
                        bbox = draw.textbbox((x1, y1 - 16), label, font=font)
                    except AttributeError:
                        text_size = draw.textsize(label, font=font)
                        bbox = (x1, y1 - 16, x1 + text_size[0], y1 - 16 + text_size[1])
                    
                    draw.rectangle(
                        [bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1],
                        fill=(0, 0, 0)
                    )
                    draw.text((x1, y1 - 16), label, fill=color, font=font)
        
        return overlay_image


def annotate_multiclass(
    image_path: str,
    classes: str,
    output_dir: Optional[str] = None,
    confidence_threshold: float = 0.5,
    class_thresholds: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Convenience function to segment with multiple classes.
    
    Args:
        image_path: Path to input image.
        classes: Comma-separated class names.
        output_dir: Directory to save outputs.
        confidence_threshold: Default confidence threshold (used if class_thresholds not provided).
        class_thresholds: Dictionary mapping class names to confidence thresholds.
        
    Returns:
        Dictionary with segmentation results and saved file paths.
    """
    annotator = SAM3Annotator(confidence_threshold=confidence_threshold)
    result = annotator.segment_multiclass(image_path, classes, class_thresholds)
    saved_files = annotator.save_multiclass_outputs(result, output_dir)
    
    return {
        "result": result,
        "saved_files": saved_files
    }


def parse_class_thresholds(threshold_str: str, class_list: List[str]) -> Dict[str, float]:
    """
    Parse per-class thresholds from string.
    
    Format: "class1:0.6,class2:0.3" or "0.5" (applies to all classes)
    
    Args:
        threshold_str: String with thresholds.
        class_list: List of class names.
        
    Returns:
        Dictionary mapping class names to thresholds.
    """
    thresholds = {}
    
    # If it's a single number, apply to all classes
    try:
        single_threshold = float(threshold_str)
        return {class_name: single_threshold for class_name in class_list}
    except ValueError:
        pass
    
    # Parse per-class thresholds (format: "class1:0.6,class2:0.3")
    parts = threshold_str.split(",")
    for part in parts:
        part = part.strip()
        if ":" in part:
            class_name, threshold_val = part.split(":", 1)
            class_name = class_name.strip()
            try:
                thresholds[class_name] = float(threshold_val.strip())
            except ValueError:
                print(f"[WARNING] Invalid threshold value for '{class_name}': {threshold_val}")
        else:
            print(f"[WARNING] Invalid threshold format: {part}")
    
    return thresholds


def get_image_files(path: str) -> List[str]:
    """
    Get all image files from a path (file or directory).
    
    Args:
        path: Path to a file or directory.
        
    Returns:
        List of image file paths.
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    # If it's a file, return it as a single-item list
    if path_obj.is_file():
        return [str(path_obj)]
    
    # If it's a directory, find all image files
    if path_obj.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(path_obj.glob(f"*{ext}"))
            image_files.extend(path_obj.glob(f"*{ext.upper()}"))
        # Sort for consistent processing order
        return sorted([str(f) for f in image_files])
    
    return []


def print_legend(classes: List[str], class_results: Dict, annotator=None):
    """Print color legend for classes."""
    print("\nClass Color Legend:")
    print("-" * 85)
    for class_name in classes:
        if class_name in class_results:
            color = class_results[class_name]["color"]
            count = len(class_results[class_name]["masks"])
            threshold = class_results[class_name].get("threshold", 0.5)
            
            # Get simplification method if annotator provided
            simplify_info = ""
            if annotator:
                method = annotator._get_class_simplify_method(class_name)
                simplify_info = f" [{method}]"
            
            print(f"  {class_name:20} RGB{str(color):20} threshold={threshold:.2f}{simplify_info:10} ({count} objects)")
    print("-" * 85)


def main():
    """Main entry point for command line usage."""
    print("=" * 60)
    print("SAM3 Annotator - Multi-Class Mask Generator")
    print("=" * 60)
    
    # Default values
    input_path = "tests/images/house.png"
    classes = "building, road, vehicle on the road, parking, tree, grass"
    output_dir = None
    confidence = 0.5
    threshold_str = None
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        classes = sys.argv[2]
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    if len(sys.argv) > 4:
        threshold_str = sys.argv[4]
        # Try to parse as single float first
        try:
            confidence = float(threshold_str)
        except ValueError:
            # Will be parsed as per-class thresholds later
            pass
    
    # Get all image files (handles both file and directory)
    try:
        image_files = get_image_files(input_path)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return None
    
    if not image_files:
        print(f"\n[ERROR] No image files found in: {input_path}")
        return None
    
    print(f"\n[CONFIG] Input: {input_path}")
    print(f"[CONFIG] Found {len(image_files)} image file(s)")
    print(f"[CONFIG] Classes: '{classes}'")
    print(f"[CONFIG] Output dir: {output_dir or 'same as input'}")
    
    # Parse class list
    class_list = [c.strip() for c in classes.split(",") if c.strip()]
    
    # Parse thresholds
    if threshold_str:
        class_thresholds = parse_class_thresholds(threshold_str, class_list)
        print(f"[CONFIG] Thresholds: {class_thresholds}")
    else:
        class_thresholds = None
        print(f"[CONFIG] Default confidence: {confidence} (using class-specific defaults)")
    
    print()
    
    try:
        # Create annotator (only once for all images)
        annotator = SAM3Annotator(confidence_threshold=confidence)
        
        # Process each image
        all_results = []
        all_saved_files = []
        
        for idx, image_path in enumerate(image_files, 1):
            print()
            print("=" * 60)
            print(f"Processing image {idx}/{len(image_files)}: {Path(image_path).name}")
            print("=" * 60)
            
            try:
                # Determine output directory for this image
                image_output_dir = output_dir
                if image_output_dir is None:
                    # Save in same directory as input image
                    image_output_dir = str(Path(image_path).parent)
                
                # Segment image with multiple classes
                result = annotator.segment_multiclass(image_path, classes, class_thresholds)
                
                # Print results summary for this image
                print()
                print("-" * 60)
                print(f"Segmentation Results: {Path(image_path).name}")
                print("-" * 60)
                
                class_results = result["class_results"]
                total_objects = sum(len(cr["masks"]) for cr in class_results.values())
                print(f"Total objects detected: {total_objects}")
                
                print_legend(result["classes"], class_results, annotator)
                
                # Save outputs
                print()
                print("-" * 60)
                print(f"Saving Outputs: {Path(image_path).name}")
                print("-" * 60)
                if CV2_AVAILABLE:
                    if idx == 1:  # Only print once
                        print("[INFO] Mask simplification enabled:")
                        print("[INFO]   - Geometric classes (building, road, parking) -> polygon simplification")
                        print("[INFO]   - Organic classes (tree, grass, vegetation) -> smooth boundaries")
                else:
                    if idx == 1:  # Only print once
                        print("[WARNING] OpenCV not available - masks will not be simplified")
                
                saved_files = annotator.save_multiclass_outputs(
                    result, 
                    image_output_dir, 
                    simplify=CV2_AVAILABLE
                )
                
                all_results.append(result)
                all_saved_files.append(saved_files)
                
            except Exception as e:
                print(f"\n[ERROR] Failed to process {image_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print()
        print("=" * 60)
        print(f"[SUCCESS] Processed {len(all_results)}/{len(image_files)} image(s) successfully!")
        print("=" * 60)
        
        return {
            "results": all_results,
            "saved_files": all_saved_files
        }
        
    except Exception as e:
        print(f"\n[ERROR] Annotation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
