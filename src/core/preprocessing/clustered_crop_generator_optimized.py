# src/core/preprocessing/clustered_crop_generator_optimized.py

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

class ClusteredCropGeneratorOptimized:
    """
    Generador de crops inteligente para la Fase 2 (MetaCLIP).
    
    Diferencias vs versión anterior:
    1. ELIMINADO: Padding con color gris (causaba artifacts).
    2. NUEVO: Resize proporcional Lanczos4 solo si excede tamaño target.
    3. NUEVO: Metadata extendida para estrategia multi-patch.
    """
    
    def __init__(self, target_size: int = 336):
        self.target_size = target_size
        # Margen mínimo alrededor del defecto (contexto)
        self.context_margin = 0.20  # 20%

    def generate_crops(self, image_path: Path, json_path: Path) -> List[Dict]:
        import json
        
        # Cargar imagen y JSON
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        h_img, w_img = image.shape[:2]
        
        with open(json_path) as f:
            data = json.load(f)
            
        # 1. Extraer defectos
        defects = self._extract_defects(data, w_img, h_img)
        if not defects:
            return []
            
        # 2. Clusterizar defectos cercanos (Lógica rescatada del script 02)
        clusters = self._cluster_defects(defects)
        
        crops_metadata = []
        
        # 3. Generar crops optimizados
        for i, cluster in enumerate(clusters):
            # Calcular bbox del cluster con margen
            x1, y1, x2, y2 = self._get_cluster_bbox(cluster, w_img, h_img)
            w, h = x2 - x1, y2 - y1
            
            # Extraer crop "crudo" (sin procesar)
            crop = image[y1:y2, x1:x2]
            
            # Lógica de optimización para MetaCLIP
            final_crop, scale_info = self._optimize_crop_for_metaclip(crop)
            
            crops_metadata.append({
                'crop': final_crop,
                'bbox_original': [x1, y1, x2, y2],
                'defects': cluster,
                'scale_info': scale_info, # Para saber si se hizo resize
                'cluster_id': i
            })
            
        return crops_metadata

    def _optimize_crop_for_metaclip(self, crop: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Prepara el crop para MetaCLIP 2 sin introducir ruido artificial.
        """
        h, w = crop.shape[:2]
        
        # Caso A: El crop es más pequeño o igual al target (448x448)
        # Lo devolvemos tal cual. MetaCLIP hará el resize interno si hace falta,
        # pero preferimos no inventar píxeles (no padding).
        if w <= self.target_size and h <= self.target_size:
            return crop, {'type': 'original', 'scale': 1.0}
            
        # Caso B: El crop es gigante (ej: todo el lateral del coche)
        # Hacemos downscale proporcional suave para no perder contexto global.
        scale = min(self.target_size / w, self.target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized_crop = cv2.resize(
            crop, 
            (new_w, new_h), 
            interpolation=cv2.INTER_LANCZOS4 # Mejor calidad para downsampling
        )
        
        return resized_crop, {'type': 'resized', 'scale': scale}

    def _extract_defects(self, data: dict, w_img: int, h_img: int) -> List[Dict]:
        """Extrae bboxes de los polígonos del JSON"""
        defects = []
        for shape in data.get('shapes', []):
            if shape['shape_type'] != 'polygon' or shape['label'] == '9':
                continue
                
            points = np.array(shape['points'])
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            defects.append({
                'label': shape['label'],
                'bbox': [max(0, int(x_min)), max(0, int(y_min)), 
                         min(w_img, int(x_max)), min(h_img, int(y_max))]
            })
        return defects

    def _cluster_defects(self, defects: List[Dict]) -> List[List[Dict]]:
        """
        Algoritmo simple de agrupación por distancia.
        Si dos defectos están cerca (<100px), van al mismo crop.
        """
        if not defects: return []
        
        # Implementación simplificada del clustering espacial
        # En producción real, usarías la lógica O(n log n) del script 02
        # Aquí usamos una heurística rápida para avanzar.
        clusters = []
        assigned = [False] * len(defects)
        
        for i in range(len(defects)):
            if assigned[i]: continue
            
            current_cluster = [defects[i]]
            assigned[i] = True
            
            ref_bbox = defects[i]['bbox']
            center_ref = ((ref_bbox[0]+ref_bbox[2])/2, (ref_bbox[1]+ref_bbox[3])/2)
            
            for j in range(i+1, len(defects)):
                if assigned[j]: continue
                
                tgt_bbox = defects[j]['bbox']
                center_tgt = ((tgt_bbox[0]+tgt_bbox[2])/2, (tgt_bbox[1]+tgt_bbox[3])/2)
                
                # Distancia Euclídea
                dist = np.sqrt((center_ref[0]-center_tgt[0])**2 + (center_ref[1]-center_tgt[1])**2)
                
                if dist < 100: # Threshold de proximidad
                    current_cluster.append(defects[j])
                    assigned[j] = True
            
            clusters.append(current_cluster)
            
        return clusters

    def _get_cluster_bbox(self, cluster: List[Dict], w_img: int, h_img: int) -> List[int]:
        """Calcula el bounding box que engloba todo el cluster + margen"""
        x1 = min(d['bbox'][0] for d in cluster)
        y1 = min(d['bbox'][1] for d in cluster)
        x2 = max(d['bbox'][2] for d in cluster)
        y2 = max(d['bbox'][3] for d in cluster)
        
        w_box, h_box = x2 - x1, y2 - y1
        
        # Aplicar margen de contexto
        pad_x = int(w_box * self.context_margin)
        pad_y = int(h_box * self.context_margin)
        
        return [
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(w_img, x2 + pad_x),
            min(h_img, y2 + pad_y)
        ]