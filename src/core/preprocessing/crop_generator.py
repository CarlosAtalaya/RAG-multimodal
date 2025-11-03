import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import yaml
import json

class AdaptiveCropGenerator:
    """
    Genera crops de ROIs con padding adaptativo inteligente
    
    Mejoras v2:
    - Umbrales adaptativos basados en distribución del dataset
    - Considera posición relativa en imagen (contexto espacial)
    - Padding ajustado por tipo de daño Y tamaño
    - Preserva información espacial completa
    """
    
    def __init__(self, config_path: str = "config/crop_strategy_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.damage_padding = self.config['padding_factors']
        self.target_size = tuple(self.config['roi_crops']['target_size'])
        self.padding_color = tuple(self.config['roi_crops']['padding_color'])
        
        # Umbrales más flexibles (se ajustarán automáticamente)
        self.min_area = self.config.get('min_bbox_area', 50)  # Más permisivo
        self.max_area = self.config.get('max_bbox_area', 500000)  # Mucho más alto
        
        # Estadísticas para normalización
        self.bbox_areas = []
        self.stats_computed = False
    
    def compute_dataset_statistics(self, manifest_path: Path):
        """
        Calcula estadísticas del dataset para umbrales adaptativos
        """
        print("\n📊 Calculando estadísticas del dataset...")
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        all_areas = []
        
        for item in manifest:
            json_path = manifest_path.parent / item['json']
            
            if not json_path.exists():
                continue
            
            with open(json_path) as f:
                json_data = json.load(f)
            
            for shape in json_data['shapes']:
                if shape['shape_type'] != 'polygon':
                    continue
                
                polygon = np.array(shape['points'], dtype=np.int32)
                x_min, y_min = polygon.min(axis=0)
                x_max, y_max = polygon.max(axis=0)
                
                area = (x_max - x_min) * (y_max - y_min)
                all_areas.append(area)
        
        if not all_areas:
            print("⚠️  No se encontraron áreas válidas")
            return
        
        all_areas = np.array(all_areas)
        
        # Calcular percentiles
        p5 = np.percentile(all_areas, 5)
        p25 = np.percentile(all_areas, 25)
        p50 = np.percentile(all_areas, 50)
        p75 = np.percentile(all_areas, 75)
        p95 = np.percentile(all_areas, 95)
        p99 = np.percentile(all_areas, 99)
        
        # Ajustar umbrales basados en distribución
        # Descartar solo extremos muy anómalos
        self.min_area = max(50, p5 * 0.1)  # 10% del percentil 5
        self.max_area = p99 * 1.5  # 150% del percentil 99
        
        print(f"✅ Estadísticas calculadas:")
        print(f"   - Total defectos: {len(all_areas)}")
        print(f"   - Área mínima: {all_areas.min():.0f} px²")
        print(f"   - Área máxima: {all_areas.max():.0f} px²")
        print(f"   - Mediana (P50): {p50:.0f} px²")
        print(f"   - P25-P75: {p25:.0f} - {p75:.0f} px²")
        print(f"   - P5-P95: {p5:.0f} - {p95:.0f} px²")
        print(f"\n   📏 Umbrales aplicados:")
        print(f"   - Mínimo: {self.min_area:.0f} px²")
        print(f"   - Máximo: {self.max_area:.0f} px²")
        
        self.bbox_areas = all_areas
        self.stats_computed = True
    
    def generate_crops_from_json(
        self,
        image_path: Path,
        json_data: Dict,
        output_dir: Path
    ) -> List[Dict]:
        """
        Genera todos los crops de una imagen con estrategia mejorada
        """
        # Cargar imagen
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"No se pudo cargar imagen: {image_path}")
        
        img_h, img_w = image.shape[:2]
        
        crops_metadata = []
        
        # Procesar cada shape (defecto)
        for idx, shape in enumerate(json_data['shapes']):
            if shape['shape_type'] != 'polygon':
                continue
            
            # Calcular bounding box del polígono
            polygon = np.array(shape['points'], dtype=np.int32)
            x_min, y_min = polygon.min(axis=0)
            x_max, y_max = polygon.max(axis=0)
            
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min
            bbox_area = bbox_w * bbox_h
            
            # Calcular centroide del defecto
            centroid_x = (x_min + x_max) / 2
            centroid_y = (y_min + y_max) / 2
            
            # Posición relativa en imagen (0-1)
            rel_x = centroid_x / img_w
            rel_y = centroid_y / img_h
            
            # Filtrar solo extremos muy anómalos
            if bbox_area < self.min_area:
                print(f"  ⚠️  Skipping ROI {idx}: area {bbox_area:.0f} too small (< {self.min_area:.0f})")
                continue
            
            if bbox_area > self.max_area:
                print(f"  ⚠️  Skipping ROI {idx}: area {bbox_area:.0f} too large (> {self.max_area:.0f})")
                continue
            
            # Obtener tipo de daño
            damage_label = shape['label']
            damage_type = self._label_to_canonical(damage_label)
            
            # Calcular padding adaptativo MEJORADO
            padding_factor = self._calculate_adaptive_padding(
                bbox_area=bbox_area,
                damage_type=damage_type,
                rel_position=(rel_x, rel_y),
                image_size=(img_w, img_h)
            )
            
            pad_x = int(bbox_w * padding_factor)
            pad_y = int(bbox_h * padding_factor)
            
            # Calcular región de crop con límites
            crop_x1 = max(0, x_min - pad_x)
            crop_y1 = max(0, y_min - pad_y)
            crop_x2 = min(img_w, x_max + pad_x)
            crop_y2 = min(img_h, y_max + pad_y)
            
            # Extraer crop
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Resize manteniendo aspect ratio
            crop_resized = self._resize_with_padding(
                crop,
                self.target_size,
                self.padding_color
            )
            
            # Guardar crop
            crop_filename = f"{image_path.stem}_roi_{idx:03d}_{damage_type}.jpg"
            crop_path = output_dir / crop_filename
            cv2.imwrite(str(crop_path), crop_resized)
            
            # Calcular zona de la imagen (división en grilla 3x3)
            zone_x = "left" if rel_x < 0.33 else "center" if rel_x < 0.67 else "right"
            zone_y = "top" if rel_y < 0.33 else "middle" if rel_y < 0.67 else "bottom"
            spatial_zone = f"{zone_y}_{zone_x}"
            
            # Guardar metadata COMPLETA
            metadata = {
                'crop_id': f"{image_path.stem}_roi_{idx:03d}",
                'crop_path': str(crop_path),
                'source_image': str(image_path),
                'source_image_size': [img_w, img_h],
                
                # Información del daño
                'damage_type': damage_type,
                'damage_label': damage_label,
                
                # Geometría del polígono original
                'polygon_coords': polygon.tolist(),
                'bbox': [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)],
                'bbox_center': [int(centroid_x), int(centroid_y)],
                'bbox_area': int(bbox_area),
                'bbox_aspect_ratio': float(bbox_w / bbox_h) if bbox_h > 0 else 1.0,
                
                # Posición espacial (NUEVO)
                'relative_position': {
                    'x': float(rel_x),
                    'y': float(rel_y)
                },
                'spatial_zone': spatial_zone,
                
                # Padding aplicado
                'padding_applied': [int(pad_x), int(pad_y)],
                'padding_factor': float(padding_factor),
                
                # Información del crop final
                'crop_size': list(crop_resized.shape[:2]),
                
                # Contexto adicional (CONVERTIR A TIPOS PYTHON NATIVOS)
                'is_edge_defect': bool(self._is_near_edge(rel_x, rel_y)),  # ← FIX AQUÍ
                'relative_size': self._calculate_relative_size(bbox_area, img_w, img_h)
            }
            
            crops_metadata.append(metadata)
        
        return crops_metadata
    
    def _calculate_adaptive_padding(
        self,
        bbox_area: float,
        damage_type: str,
        rel_position: Tuple[float, float],
        image_size: Tuple[int, int]
    ) -> float:
        """
        Calcula padding adaptativo considerando múltiples factores
        """
        # Padding base por tipo de daño
        damage_label = self._canonical_to_label(damage_type)
        base_padding = self.damage_padding.get(damage_label, 0.30)
        
        # Factor 1: Ajuste por tamaño absoluto
        if bbox_area < 500:
            size_factor = 1.8  # Defectos muy pequeños
        elif bbox_area < 1500:
            size_factor = 1.5
        elif bbox_area < 5000:
            size_factor = 1.2
        elif bbox_area < 15000:
            size_factor = 1.0
        elif bbox_area < 50000:
            size_factor = 0.85
        else:
            size_factor = 0.7  # Defectos muy grandes
        
        # Factor 2: Ajuste por posición (defectos en bordes necesitan menos padding)
        rel_x, rel_y = rel_position
        is_edge = (rel_x < 0.1 or rel_x > 0.9 or rel_y < 0.1 or rel_y > 0.9)
        edge_factor = 0.7 if is_edge else 1.0
        
        # Factor 3: Ajuste por tamaño relativo a la imagen
        img_w, img_h = image_size
        img_area = img_w * img_h
        relative_size = bbox_area / img_area
        
        if relative_size > 0.15:  # Defecto ocupa >15% de la imagen
            relative_factor = 0.6
        elif relative_size > 0.08:
            relative_factor = 0.8
        else:
            relative_factor = 1.0
        
        # Combinar factores
        final_padding = base_padding * size_factor * edge_factor * relative_factor
        
        # Limitar padding entre 0.15 y 0.60
        final_padding = max(0.15, min(0.60, final_padding))
        
        return final_padding
    
    def _is_near_edge(self, rel_x: float, rel_y: float, threshold: float = 0.15) -> bool:
        """Determina si el defecto está cerca del borde"""
        return (rel_x < threshold or rel_x > (1 - threshold) or 
                rel_y < threshold or rel_y > (1 - threshold))
    
    def _calculate_relative_size(self, bbox_area: float, img_w: int, img_h: int) -> str:
        """Calcula tamaño relativo del defecto"""
        img_area = img_w * img_h
        ratio = bbox_area / img_area
        
        if ratio < 0.005:
            return "very_small"
        elif ratio < 0.02:
            return "small"
        elif ratio < 0.08:
            return "medium"
        elif ratio < 0.15:
            return "large"
        else:
            return "very_large"
    
    def _resize_with_padding(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        padding_color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Resize manteniendo aspect ratio con padding
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calcular ratio
        scale = min(target_w / w, target_h / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Crear canvas con padding
        canvas = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
        
        # Centrar imagen
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    def _label_to_canonical(self, label: str) -> str:
        """Convierte label numérico a nombre canónico"""
        mapping = {
            "1": "surface_scratch",
            "2": "dent",
            "3": "paint_peeling",
            "4": "deep_scratch",
            "5": "crack",
            "6": "missing_part",
            "7": "missing_accessory",
            "8": "misaligned_part"
        }
        return mapping.get(label, "unknown")
    
    def _canonical_to_label(self, canonical: str) -> str:
        """Convierte nombre canónico a label numérico"""
        mapping = {
            "surface_scratch": "1",
            "dent": "2",
            "paint_peeling": "3",
            "deep_scratch": "4",
            "crack": "5",
            "missing_part": "6",
            "missing_accessory": "7",
            "misaligned_part": "8"
        }
        return mapping.get(canonical, "1")