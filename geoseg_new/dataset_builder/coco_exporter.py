"""
COCO format exporter for segmentation datasets.
Converts masks to COCO polygon annotations.
"""

import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
import cv2

# Color mapping for mask visualization (RGB values)
# Must match CLASS_COLORS from colors.py and label_preparation
CLASS_COLORS = {
    0: (0, 0, 0),        # Background - Black
    1: (255, 0, 0),      # Building - Red
    2: (255, 255, 0),    # Road - Yellow
    3: (255, 0, 255),    # Parking - Magenta
    4: (0, 0, 255),      # Water - Blue
    5: (0, 128, 0),      # Tree - Dark Green
    6: (144, 238, 144),  # Grass - Light Green
}


class COCOExporter:
    """
    Exports segmentation masks to COCO format.
    """
    
    def __init__(self):
        """
        Initialize COCO exporter with category definitions.
        """
        # Categories matching the segmentation classes
        self.categories = [
            {"id": 1, "name": "Building", "supercategory": "structure"},
            {"id": 2, "name": "Road", "supercategory": "infrastructure"},
            {"id": 3, "name": "Parking", "supercategory": "infrastructure"},
            {"id": 4, "name": "Water", "supercategory": "terrain"},
            {"id": 5, "name": "Tree", "supercategory": "vegetation"},
            {"id": 6, "name": "Grass", "supercategory": "vegetation"},
        ]
        
        # Map from mask class IDs to COCO category IDs (direct 1:1 mapping)
        self.mask_to_coco_mapping = {
            1: 1,  # Building -> Building
            2: 2,  # Road -> Road
            3: 3,  # Parking -> Parking
            4: 4,  # Water -> Water
            5: 5,  # Tree -> Tree
            6: 6,  # Grass -> Grass
        }
        
        # Note: All classes now have direct mappings
        # - If area is large and spread out -> Grass
        # This can be improved with actual SAM output or additional classification
    
    def export_split(
        self,
        split_dir: str,
        output_json: str,
        split_name: str = "train"
    ) -> None:
        """
        Export a dataset split to COCO format.
        
        Args:
            split_dir: Directory containing images/ and masks/ subdirectories
            output_json: Path to output COCO JSON file
            split_name: Name of the split (train/val/test)
        """
        images_dir = os.path.join(split_dir, "images")
        masks_dir = os.path.join(split_dir, "masks")
        
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
        if not os.path.exists(masks_dir):
            raise ValueError(f"Masks directory not found: {masks_dir}")
        
        # Get all image files
        image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not image_files:
            print(f"  Warning: No image files found in {images_dir}, skipping export")
            return
        
        # Initialize COCO structure
        coco_data = {
            "info": {
                "description": f"GeoSeg Dataset - {split_name} split",
                "version": "1.0",
                "year": 2024,
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": self.categories
        }
        
        # Process each image
        image_id = 1
        annotation_id = 1
        
        for img_file in sorted(image_files):
            image_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file)
            
            # Try different mask extensions if exact match not found
            if not os.path.exists(mask_path):
                base_name = os.path.splitext(img_file)[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    alt_mask_path = os.path.join(masks_dir, f"{base_name}{ext}")
                    if os.path.exists(alt_mask_path):
                        mask_path = alt_mask_path
                        break
                else:
                    print(f"Warning: No mask found for {img_file}, skipping")
                    continue
            
            # Load image to get dimensions
            img = Image.open(image_path)
            width, height = img.size
            
            # Add image entry
            coco_data["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": img_file,
            })
            
            # Process mask and create annotations
            annotations = self._mask_to_annotations(
                mask_path, image_id, annotation_id
            )
            
            coco_data["annotations"].extend(annotations)
            annotation_id += len(annotations)
            image_id += 1
        
        # Save COCO JSON
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"  Exported {len(coco_data['images'])} images, "
              f"{len(coco_data['annotations'])} annotations")
    
    def _load_mask_as_class_ids(self, mask_path: str) -> np.ndarray:
        """
        Load a mask and convert to class IDs (handles both RGB and grayscale).
        
        Args:
            mask_path: Path to mask image
            
        Returns:
            2D numpy array with class IDs
        """
        mask_img = Image.open(mask_path)
        mask = np.array(mask_img)
        
        # If RGB, convert back to class IDs
        if len(mask.shape) == 3 and mask.shape[2] == 3:
            mask_grayscale = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
            # Map RGB colors back to class IDs
            for class_id, color in CLASS_COLORS.items():
                color_array = np.array(color)
                r_match = np.abs(mask[:, :, 0].astype(int) - color_array[0]) < 10
                g_match = np.abs(mask[:, :, 1].astype(int) - color_array[1]) < 10
                b_match = np.abs(mask[:, :, 2].astype(int) - color_array[2]) < 10
                color_match = r_match & g_match & b_match
                mask_grayscale[color_match] = class_id
            return mask_grayscale
        elif len(mask.shape) == 3:
            # Multi-channel but not RGB, take first channel
            return mask[:, :, 0]
        else:
            # Already 2D grayscale
            return mask
    
    def _mask_to_annotations(
        self,
        mask_path: str,
        image_id: int,
        start_annotation_id: int
    ) -> List[Dict]:
        """
        Convert segmentation mask to COCO annotations.
        
        Args:
            mask_path: Path to mask image
            image_id: COCO image ID
            start_annotation_id: Starting annotation ID
            
        Returns:
            List of COCO annotation dictionaries
        """
        # Load mask and convert RGB to class IDs if needed
        mask = self._load_mask_as_class_ids(mask_path)
        height, width = mask.shape
        
        annotations = []
        annotation_id = start_annotation_id
        
        # Process each class in the mask
        unique_classes = np.unique(mask)
        
        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue
            
            # Get COCO category ID (direct mapping for all classes)
            coco_category_id = self.mask_to_coco_mapping.get(class_id)
            
            if coco_category_id is None:
                # Unknown class, skip
                continue
            
            # Extract polygons for this class
            class_mask = (mask == class_id).astype(np.uint8)
            polygons = self._mask_to_polygons(class_mask)
            
            # Create annotation for each polygon
            for polygon in polygons:
                if len(polygon) < 6:  # Need at least 3 points (x,y pairs)
                    continue
                
                # Calculate area and bbox
                area = self._polygon_area(polygon)
                bbox = self._polygon_to_bbox(polygon)
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": coco_category_id,
                    "segmentation": [polygon],  # COCO format: list of polygons
                    "area": area,
                    "bbox": bbox,  # [x, y, width, height]
                    "iscrowd": 0
                }
                
                annotations.append(annotation)
                annotation_id += 1
        
        return annotations
    
    def _classify_natural_features(
        self,
        natural_mask: np.ndarray,
        image_id: int,
        start_annotation_id: int,
        width: int,
        height: int
    ) -> List[Dict]:
        """
        Classify natural features (class 5) into Tree or Grass.
        
        Uses heuristics:
        - Small, compact regions -> Tree
        - Large, spread out regions -> Grass
        
        Args:
            natural_mask: Binary mask of natural features
            image_id: COCO image ID
            start_annotation_id: Starting annotation ID
            width: Image width
            height: Image height
            
        Returns:
            List of COCO annotations for Tree and Grass
        """
        annotations = []
        annotation_id = start_annotation_id
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            natural_mask, connectivity=8
        )
        
        for label_id in range(1, num_labels):  # Skip background (label 0)
            component_mask = (labels == label_id).astype(np.uint8)
            
            # Get component statistics
            area = stats[label_id, cv2.CC_STAT_AREA]
            bbox = stats[label_id, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4]
            bbox_width = bbox[2]
            bbox_height = bbox[3]
            
            # Heuristic: classify based on size and compactness
            compactness = area / (bbox_width * bbox_height + 1e-6)
            
            # Small and compact -> Tree, large or spread out -> Grass
            if area < 5000 and compactness > 0.3:
                category_id = 5  # Tree
            else:
                category_id = 6  # Grass
            
            # Extract polygon
            polygons = self._mask_to_polygons(component_mask)
            
            for polygon in polygons:
                if len(polygon) < 6:
                    continue
                
                poly_area = self._polygon_area(polygon)
                bbox_coco = self._polygon_to_bbox(polygon)
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [polygon],
                    "area": poly_area,
                    "bbox": bbox_coco,
                    "iscrowd": 0
                }
                
                annotations.append(annotation)
                annotation_id += 1
        
        return annotations
    
    def _mask_to_polygons(self, mask: np.ndarray) -> List[List[float]]:
        """
        Convert binary mask to polygon coordinates.
        
        Args:
            mask: Binary mask (0 or 255)
            
        Returns:
            List of polygons, each as [x1, y1, x2, y2, ...]
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        polygons = []
        for contour in contours:
            if len(contour) < 3:  # Need at least 3 points
                continue
            
            # Flatten contour to [x1, y1, x2, y2, ...]
            polygon = contour.reshape(-1, 2).flatten().tolist()
            polygons.append(polygon)
        
        return polygons
    
    def _polygon_area(self, polygon: List[float]) -> float:
        """
        Calculate polygon area using shoelace formula.
        
        Args:
            polygon: Polygon as [x1, y1, x2, y2, ...]
            
        Returns:
            Area of polygon
        """
        if len(polygon) < 6:
            return 0.0
        
        # Reshape to [[x1, y1], [x2, y2], ...]
        points = np.array(polygon).reshape(-1, 2)
        
        # Shoelace formula
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
        return float(area)
    
    def _polygon_to_bbox(self, polygon: List[float]) -> List[float]:
        """
        Calculate bounding box from polygon.
        
        Args:
            polygon: Polygon as [x1, y1, x2, y2, ...]
            
        Returns:
            Bounding box as [x, y, width, height]
        """
        if len(polygon) < 6:
            return [0.0, 0.0, 0.0, 0.0]
        
        points = np.array(polygon).reshape(-1, 2)
        x_min = float(np.min(points[:, 0]))
        y_min = float(np.min(points[:, 1]))
        x_max = float(np.max(points[:, 0]))
        y_max = float(np.max(points[:, 1]))
        
        return [x_min, y_min, x_max - x_min, y_max - y_min]

