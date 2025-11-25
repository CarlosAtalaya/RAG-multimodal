# src/core/preprocessing/clustered_crop_generator_optimized.py

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

class ClusteredCropGeneratorOptimized:
    """
    Generador de crops inteligente V2.
    
    Características:
    - NO RESIZE: Mantiene la resolución original (1:1).
    - TAMAÑO FIJO: Todos los crops son de target_size x target_size.
    - SMART TILING: Si un defecto es gigante, lo cubre con múltiples tiles.
    """
    
    def __init__(self, target_size: int = 336):
        self.target_size = target_size
        # Margen base alrededor del defecto (solo si cabe en un crop)
        self.base_margin = 0.10
        # Solapamiento mínimo entre tiles para defectos grandes (20%)
        self.tile_overlap = 0.20 

    def generate_crops(self, image_path: Path, json_path: Path) -> List[Dict]:
        import json
        
        # 1. Cargar imagen
        image = cv2.imread(str(image_path))
        if image is None: return []
        h_img, w_img = image.shape[:2]
        
        # 2. Cargar datos
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error JSON {json_path}: {e}")
            return []
            
        # 3. Extraer y Clusterizar
        defects = self._extract_defects(data, w_img, h_img)
        if not defects: return []
            
        clusters = self._cluster_defects(defects)
        crops_metadata = []
        
        # 4. Generar Crops (1 o múltiples por cluster)
        for i, cluster in enumerate(clusters):
            
            # Obtener lista de tiles que cubren este cluster
            tiles = self._generate_tiles_for_cluster(image, cluster, w_img, h_img)
            
            for j, tile_data in enumerate(tiles):
                crops_metadata.append({
                    'crop': tile_data['crop'],             # (336, 336, 3)
                    'bbox_crop': tile_data['bbox'],        # Coordenadas del crop en la imagen original
                    'defects': cluster,                    # Referencia a los defectos que cubre
                    'is_tiled': tile_data['is_tiled'],     # Flag para saber si fue partido
                    'cluster_id': i,
                    'tile_id': j
                })
            
        return crops_metadata

    def _generate_tiles_for_cluster(self, image: np.ndarray, cluster: List[Dict], w_img: int, h_img: int) -> List[Dict]:
        """
        Decide si saca un solo crop centrado o hace tiling si el defecto es muy grande.
        """
        # 1. BBox del cluster completo
        x1 = min(d['bbox'][0] for d in cluster)
        y1 = min(d['bbox'][1] for d in cluster)
        x2 = max(d['bbox'][2] for d in cluster)
        y2 = max(d['bbox'][3] for d in cluster)
        
        w_cluster = x2 - x1
        h_cluster = y2 - y1
        
        # Lista de coordenadas [x, y, w, h] para los crops
        crop_coords = []
        is_tiled = False

        # CASO A: El cluster cabe en un solo crop (con un poco de margen)
        # Verificamos si es más pequeño que el target
        if w_cluster <= self.target_size and h_cluster <= self.target_size:
            # Estrategia: CENTRAR
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Calcular esquinas top-left
            sx = cx - (self.target_size // 2)
            sy = cy - (self.target_size // 2)
            
            # Ajustar si se sale de la imagen (Shift)
            sx = max(0, min(sx, w_img - self.target_size))
            sy = max(0, min(sy, h_img - self.target_size))
            
            crop_coords.append((int(sx), int(sy)))
            
        else:
            # CASO B: El cluster es GIGANTE -> TILING (Teselado)
            # Generamos puntos de corte para X y para Y
            is_tiled = True
            
            xs = self._get_tiling_coords(x1, x2, w_img)
            ys = self._get_tiling_coords(y1, y2, h_img)
            
            # Producto cartesiano de coordenadas
            for sy in ys:
                for sx in xs:
                    crop_coords.append((int(sx), int(sy)))

        # Extraer los recortes reales
        tiles = []
        for sx, sy in crop_coords:
            ex = sx + self.target_size
            ey = sy + self.target_size
            
            # Asegurar límites (por si la imagen original es < 336px, caso raro)
            if ex > w_img or ey > h_img:
                continue 

            crop = image[sy:ey, sx:ex]
            
            # Validación final de integridad
            if crop.shape[:2] == (self.target_size, self.target_size):
                tiles.append({
                    'crop': crop,
                    'bbox': [sx, sy, ex, ey],
                    'is_tiled': is_tiled
                })
                
        return tiles

    def _get_tiling_coords(self, start: int, end: int, max_limit: int) -> List[int]:
        """
        Calcula las coordenadas de inicio para cubrir el rango [start, end]
        con ventanas de tamaño fijo target_size.
        """
        length = end - start
        
        # Si la dimensión cabe, intentamos centrarla en el rango disponible
        # Pero si estamos en lógica de tiling, es porque al menos una dimensión falló.
        # Aquí tratamos la dimensión 1D independientemente.
        
        if length <= self.target_size:
            # Si en este eje cabe, centramos respecto al defecto
            center = (start + end) // 2
            s = center - (self.target_size // 2)
            # Clamp
            return [max(0, min(s, max_limit - self.target_size))]
        
        # Si NO cabe, generamos tiles
        coords = []
        cursor = start
        
        # Stride = tamaño - overlap
        step = int(self.target_size * (1 - self.tile_overlap))
        
        # Avanzar poniendo tiles
        while cursor + self.target_size < end:
            # Clamp para no salirnos por la izquierda (0)
            valid_start = max(0, cursor)
            # Clamp para no salirnos por la derecha
            if valid_start + self.target_size <= max_limit:
                coords.append(valid_start)
            cursor += step
            
        # IMPORTANTE: Añadir el último tile anclado al final del defecto
        # Esto asegura que el borde derecho del defecto siempre sale completo
        last_start = end - self.target_size
        last_start = max(0, min(last_start, max_limit - self.target_size))
        
        if not coords or coords[-1] != last_start:
            coords.append(last_start)
            
        return sorted(list(set(coords))) # Eliminar duplicados y ordenar

    def _extract_defects(self, data: dict, w_img: int, h_img: int) -> List[Dict]:
        defects = []
        for shape in data.get('shapes', []):
            if shape['shape_type'] != 'polygon' or shape['label'] == '9': continue
            points = np.array(shape['points'])
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            defects.append({
                'label': shape['label'],
                'bbox': [max(0, int(x_min)), max(0, int(y_min)), min(w_img, int(x_max)), min(h_img, int(y_max))]
            })
        return defects

    def _cluster_defects(self, defects: List[Dict]) -> List[List[Dict]]:
        if not defects: return []
        clusters = []
        assigned = [False] * len(defects)
        for i in range(len(defects)):
            if assigned[i]: continue
            current_cluster = [defects[i]]
            assigned[i] = True
            ref_bbox = defects[i]['bbox']
            c_ref = ((ref_bbox[0]+ref_bbox[2])/2, (ref_bbox[1]+ref_bbox[3])/2)
            
            for j in range(i+1, len(defects)):
                if assigned[j]: continue
                tgt_bbox = defects[j]['bbox']
                c_tgt = ((tgt_bbox[0]+tgt_bbox[2])/2, (tgt_bbox[1]+tgt_bbox[3])/2)
                dist = np.sqrt((c_ref[0]-c_tgt[0])**2 + (c_ref[1]-c_tgt[1])**2)
                if dist < 100:
                    current_cluster.append(defects[j])
                    assigned[j] = True
            clusters.append(current_cluster)
        return clusters