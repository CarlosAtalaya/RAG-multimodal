# src/core/preprocessing/vehicle_detector.py

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from ultralytics import YOLO

class VehicleDetector:
    """
    Detector de vehículos utilizando YOLOv8.
    Optimizado para detectar vehículos en planos cercanos (inspección).
    """
    
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.10):
        """
        Args:
            model_name: 'yolov8n.pt' (nano) o 'yolov8m.pt' (medium) si falla mucho.
            confidence: Bajamos default a 0.25 para detectar planos detalle.
        """
        print(f"🚗 Cargando VehicleDetector ({model_name}) | Conf: {confidence}...")
        self.model = YOLO(model_name)
        self.confidence = confidence
        # Clases COCO: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        self.target_classes = [2, 3, 5, 7] 

    def detect(self, image_path: Path) -> Dict:
        """
        Detecta vehículo. Si falla, devuelve un BBox de fallback inteligente (centro).
        
        Returns:
            Dict con 'bbox' y flag 'is_fallback'.
        """
        try:
            results = self.model(str(image_path), verbose=False)
        except Exception as e:
            print(f"⚠️ Error inferencia YOLO: {e}")
            results = []
            
        img = cv2.imread(str(image_path))
        if img is None: return None
        h_img, w_img = img.shape[:2]

        best_box = None
        max_area = 0
        
        if results:
            result = results[0]
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id in self.target_classes and conf >= self.confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Nos quedamos con la detección más grande
                    if area > max_area:
                        max_area = area
                        best_box = {
                            'bbox': [x1, y1, x2, y2],
                            'score': conf,
                            'is_fallback': False
                        }
        
        if best_box:
            # Expandir margen (contexto)
            return self._add_margin(best_box, w_img, h_img)
        else:
            # FALLBACK INTELIGENTE: Center Crop 80%
            # Asumimos que en peritaje el coche está centrado.
            # Evitamos esquinas (cielo/suelo).
            margin_w = int(w_img * 0.10) # 10% a cada lado
            margin_h = int(h_img * 0.10) # 10% arriba/abajo
            
            return {
                'bbox': [margin_w, margin_h, w_img - margin_w, h_img - margin_h],
                'score': 0.0,
                'is_fallback': True
            }

    def _add_margin(self, detection: Dict, img_w: int, img_h: int, margin: float = 0.10) -> Dict:
        """Expande el bbox para asegurar que no cortamos parachoques"""
        x1, y1, x2, y2 = detection['bbox']
        w, h = x2 - x1, y2 - y1
        
        pad_x = int(w * margin)
        pad_y = int(h * margin)
        
        detection['bbox'] = [
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(img_w, x2 + pad_x),
            min(img_h, y2 + pad_y)
        ]
        return detection