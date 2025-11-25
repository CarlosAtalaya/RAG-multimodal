# src/core/embeddings/crop_contextualizer.py

import yaml
from pathlib import Path
from typing import Dict, List
from collections import Counter

class CropContextualizer:
    """
    Genera descripciones textuales enriquecidas específicamente para CROPS.
    Integra reglas de negocio de Fase 2.
    """
    
    def __init__(self, config_path: str = "config/crop_strategy_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Mapeos directos del config
        self.vehicle_zones = self.config.get('vehicle_zones', {})
        self.damage_types_map = {
            k: v for k, v in self.config.get('damage_types', {}).items() 
            # Asegurar compatibilidad si el yaml usa IDs numéricos o nombres string
        }
        # Mapa inverso por si la metadata trae labels de texto directo
        self.damage_label_map = {
            "1": "surface_scratch", "2": "dent", "3": "paint_peeling",
            "4": "deep_scratch", "5": "crack", "6": "missing_part",
            "7": "missing_accessory", "8": "misaligned_part"
        }

    def _get_zone_info(self, zone_id: str) -> dict:
        """Recupera info de la zona (fender, door, etc)"""
        return self.vehicle_zones.get(str(zone_id), {
            "description": f"Vehicle zone {zone_id}",
            "area": "unknown area"
        })

    def _get_damage_name(self, label: str) -> str:
        """Traduce el ID del daño a nombre legible"""
        # Intenta mapear si viene como número string "1"
        if label in self.damage_label_map:
            return self.damage_label_map[label]
        return label.replace('_', ' ')

    def build_context(self, crop_metadata: Dict) -> str:
        """
        Genera el prompt textual basado en si es DAMAGED o CLEAN.
        """
        has_damage = crop_metadata.get('has_damage', False)
        zone_id = str(crop_metadata.get('vehicle_zone', 'unknown'))
        zone_info = self._get_zone_info(zone_id)
        
        zone_desc = zone_info.get('description', 'vehicle part')
        area = zone_info.get('area', 'vehicle')

        # --- CASO 1: CROP CON DAÑO ---
        if has_damage:
            defects = crop_metadata.get('defects', [])
            
            if not defects:
                # Fallback por seguridad si flag es True pero lista vacía
                return f"Damaged surface on {zone_desc} ({area}). Unspecified defects."

            # Contar tipos de daño en este crop específico
            # Asumimos que 'defects' es lista de dicts con 'label' o 'damage_type'
            type_counts = Counter()
            for d in defects:
                label = d.get('label') or d.get('damage_type', 'unknown')
                name = self._get_damage_name(str(label))
                type_counts[name] += 1

            # Construir resumen: "2 dents and 1 scratch"
            summary_parts = []
            for name, count in type_counts.items():
                plural = "s" if count > 1 else ""
                summary_parts.append(f"{count} {name}{plural}")
            
            summary_str = " and ".join(summary_parts)
            
            # Construir lista detallada
            # "dent (confidence 0.95), surface_scratch (large)" - simplificado para este ejemplo
            details_list = ", ".join(list(type_counts.keys()))

            # FORMATO SOLICITADO: 
            # "{N} {tipo_daño} on {zona_vehículo} ({area}). Specific defects visible: {lista_defectos_crop}."
            return f"{summary_str} on {zone_desc} ({area}). Specific defects visible: {details_list}."

        # --- CASO 2: CROP LIMPIO (CLEAN) ---
        else:
            # FORMATO SOLICITADO:
            # "Clean surface of {zona_vehículo} ({area}). No defects visible. Inspection grid section."
            return (f"Clean surface of {zone_desc} ({area}). "
                    f"No defects visible. Inspection grid section.")