# src/core/preprocessing/grid_crop_generator.py

import cv2
import numpy as np
from typing import List, Dict

class GridCropGenerator:
    """
    Genera crops en cuadrícula para imágenes sin daño.
    Asegura que cubrimos todo el vehículo con overlap.
    """
    
    def __init__(self, crop_size: int = 336, overlap: float = 0.25):
        self.crop_size = crop_size
        self.overlap = overlap
        
    def generate_crops(self, image_path: str, vehicle_bbox: List[int] = None) -> List[Dict]:
        image = cv2.imread(str(image_path))
        if image is None: return []
        
        h_img, w_img = image.shape[:2]
        
        # Si no hay detección de vehículo, usamos toda la imagen
        if not vehicle_bbox:
            x1, y1, x2, y2 = 0, 0, w_img, h_img
        else:
            x1, y1, x2, y2 = vehicle_bbox
            
        roi_w = x2 - x1
        roi_h = y2 - y1
        
        stride = int(self.crop_size * (1 - self.overlap))
        crops = []
        
        # Sliding window sobre el ROI (Vehículo)
        for y in range(y1, y2, stride):
            for x in range(x1, x2, stride):
                # Ajustar coordenadas para no salirse
                crop_x1 = x
                crop_y1 = y
                crop_x2 = min(x + self.crop_size, x2)
                crop_y2 = min(y + self.crop_size, y2)
                
                # Si el crop es muy pequeño (borde), lo descartamos o ajustamos
                if (crop_x2 - crop_x1) < self.crop_size * 0.5: continue
                if (crop_y2 - crop_y1) < self.crop_size * 0.5: continue

                crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Resize a 448x448 si es necesario (para mantener consistencia)
                if crop_img.shape[:2] != (self.crop_size, self.crop_size):
                    crop_img = cv2.resize(crop_img, (self.crop_size, self.crop_size))
                
                crops.append({
                    'crop': crop_img,
                    'grid_x': x,
                    'grid_y': y,
                    'is_clean': True
                })
                
        return crops