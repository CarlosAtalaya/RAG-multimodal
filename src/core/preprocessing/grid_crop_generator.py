# src/core/preprocessing/grid_crop_generator.py

import cv2
import numpy as np
from typing import List, Dict, Optional

class GridCropGenerator:
    """
    Genera crops en cuadrícula filtrando por contenido real del vehículo.
    Usa la segmentación del coche para descartar crops que son mayoritariamente fondo.
    """
    
    def __init__(self, crop_size: int = 336, overlap: float = 0.30, min_vehicle_ratio: float = 0.60):
        """
        Args:
            crop_size: Tamaño del crop (336px nativo para CLIP/MetaCLIP)
            overlap: Solapamiento entre crops (30%)
            min_vehicle_ratio: Mínimo % del crop que debe ser coche para guardarlo (60%)
        """
        self.crop_size = crop_size
        self.overlap = overlap
        self.min_vehicle_ratio = min_vehicle_ratio
        
    def generate_crops(self, image_path: str, car_polygon: List[List[int]] = None) -> List[Dict]:
        """
        Genera crops filtrados usando el polígono del coche.
        
        Args:
            image_path: Ruta a la imagen
            car_polygon: Lista de puntos [[x,y], [x,y]...] del contorno del coche completo.
                         Si es None, usa toda la imagen (fallback).
        """
        image = cv2.imread(str(image_path))
        if image is None: return []
        
        h_img, w_img = image.shape[:2]
        
        # 1. Preparar ROI (Region of Interest) y Máscara
        if car_polygon is not None and len(car_polygon) > 0:
            # Convertir a numpy
            pts = np.array(car_polygon, np.int32).reshape((-1, 1, 2))
            
            # Calcular BBox del polígono para limitar la iteración
            x_min, y_min, w_box, h_box = cv2.boundingRect(pts)
            x_max, y_max = x_min + w_box, y_min + h_box
            
            # Crear máscara binaria del coche (1=Coche, 0=Fondo)
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 1)
        else:
            # Si no hay polígono, asumimos toda la imagen
            x_min, y_min, x_max, y_max = 0, 0, w_img, h_img
            mask = np.ones((h_img, w_img), dtype=np.uint8)

        # 2. Configurar Grid
        stride = int(self.crop_size * (1 - self.overlap))
        crops = []
        
        # 3. Iterar Grid sobre el BBox
        for y in range(y_min, y_max, stride):
            for x in range(x_min, x_max, stride):
                
                # Coordenadas propuestas
                crop_x1 = x
                crop_y1 = y
                crop_x2 = min(x + self.crop_size, x_max) # Clamp al bbox
                crop_y2 = min(y + self.crop_size, y_max)
                
                # Ajustar si nos salimos de la imagen real
                crop_x2 = min(crop_x2, w_img)
                crop_y2 = min(crop_y2, h_img)
                
                # Validar tamaño mínimo (evitar tiras finas en bordes)
                current_w = crop_x2 - crop_x1
                current_h = crop_y2 - crop_y1
                
                if current_w < self.crop_size * 0.5 or current_h < self.crop_size * 0.5:
                    continue

                # 4. CRÍTICO: Validar contenido usando la máscara
                # Extraemos el trozo correspondiente de la máscara
                mask_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Calculamos ratio: píxeles de coche / píxeles totales del crop
                vehicle_pixels = cv2.countNonZero(mask_crop)
                total_pixels = current_w * current_h
                ratio = vehicle_pixels / total_pixels
                
                # FILTRO: Si menos del 40% es coche, DESCARTAR.
                if ratio < self.min_vehicle_ratio:
                    continue
                
                # 5. Extraer y procesar crop
                crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Resize final a 336x336 si quedó un poco más pequeño (borde)
                if crop_img.shape[:2] != (self.crop_size, self.crop_size):
                    crop_img = cv2.resize(crop_img, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LANCZOS4)
                
                crops.append({
                    'crop': crop_img,
                    'grid_x': x,
                    'grid_y': y,
                    'vehicle_ratio': float(ratio), # Guardamos esto por si es útil luego
                    'is_clean': True
                })
                
        return crops