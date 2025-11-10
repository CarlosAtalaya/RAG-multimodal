# src/core/rag/retriever.py (ACTUALIZADO para Full Images)

"""
🔍 RAG RETRIEVER - FULL IMAGES + METADATA ENRIQUECIDA

Mejoras vs versión anterior:
- Contexto RAG con descripciones textuales
- Info de zonas del vehículo
- Distribución espacial de defectos
- Filtros por zona y tipo mejorados
"""

from pathlib import Path
import json
import numpy as np
import faiss
import pickle
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .taxonomy_normalizer import TaxonomyNormalizer


@dataclass
class SearchResult:
    """Resultado de búsqueda con metadata enriquecida"""
    index: int
    distance: float
    
    # Info básica
    image_path: str
    image_name: str
    
    # Defectos
    damage_types: List[str] = field(default_factory=list)
    total_defects: int = 0
    defect_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Zona del vehículo
    vehicle_zone: str = ""
    zone_description: str = ""
    zone_area: str = ""
    
    # Descripción textual
    description: str = ""
    
    # Distribución espacial
    spatial_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Bboxes
    bboxes: List[List[int]] = field(default_factory=list)
    
    # Metadata completa
    metadata: Dict = field(default_factory=dict)


class DamageRAGRetriever:
    """Retriever mejorado para Full Images"""
    
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        config_path: Path = None,
        enable_taxonomy_normalization: bool = True
    ):
        print(f"🔧 Inicializando DamageRAGRetriever (Full Images)...")
        
        # Cargar índice y metadata
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.embedding_dim = self.index.d
        
        print(f"   ✅ Índice: {self.index.ntotal} vectores (full images)")
        print(f"   ✅ Metadata: {len(self.metadata)} entries enriquecidas")
        
        # Config opcional
        self.config = {}
        if config_path and config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        
        # Taxonomy normalizer
        self.enable_normalization = enable_taxonomy_normalization
        if self.enable_normalization:
            self.taxonomy_normalizer = TaxonomyNormalizer()
            print(f"   ✅ Taxonomy normalizer activado")
        else:
            self.taxonomy_normalizer = None
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: Optional[Dict] = None,
        return_normalized: bool = True
    ) -> List[SearchResult]:
        """
        Búsqueda con metadata enriquecida
        
        Args:
            query_embedding: Vector de consulta (1024 dims)
            k: Número de resultados
            filters: Filtros opcionales:
                - damage_type: str o List[str]
                - vehicle_zone: str o List[str]
                - min_defects: int
                - max_defects: int
        """
        # Preparar query
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Búsqueda FAISS
        k_search = k * 3 if filters else k  # Más resultados para filtrar
        k_search = min(k_search, self.index.ntotal)
        
        distances, indices = self.index.search(query_embedding, k_search)
        
        # Construir resultados
        results = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            meta = self.metadata[idx]
            
            # Normalizar tipos de daño
            damage_types = meta.get('defect_types', [])
            if return_normalized and self.taxonomy_normalizer:
                damage_types = [
                    self.taxonomy_normalizer.normalize(dt)['benchmark_label']
                    for dt in damage_types
                ]
            
            # Aplicar filtros
            if filters and not self._apply_filters(meta, filters, damage_types):
                continue
            
            result = SearchResult(
                index=int(idx),
                distance=float(dist),
                image_path=meta.get('image_path', ''),
                image_name=meta.get('image_name', ''),
                damage_types=damage_types,
                total_defects=meta.get('total_defects', 0),
                defect_distribution=meta.get('defect_distribution', {}),
                vehicle_zone=meta.get('vehicle_zone', 'unknown'),
                zone_description=meta.get('zone_description', 'unknown'),
                zone_area=meta.get('zone_area', 'unknown'),
                description=meta.get('description', ''),
                spatial_distribution=meta.get('spatial_distribution', {}),
                bboxes=meta.get('bboxes', []),
                metadata=meta
            )
            
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def _apply_filters(
        self, 
        meta: Dict, 
        filters: Dict, 
        normalized_types: List[str] = None
    ) -> bool:
        """Aplica filtros a un resultado"""
        
        # Filtro por tipo de daño
        if 'damage_type' in filters:
            allowed = filters['damage_type']
            if isinstance(allowed, str):
                allowed = [allowed]
            
            if normalized_types:
                if not any(dt in allowed for dt in normalized_types):
                    return False
            else:
                meta_types = meta.get('defect_types', [])
                if not any(dt in allowed for dt in meta_types):
                    return False
        
        # Filtro por zona del vehículo
        if 'vehicle_zone' in filters:
            allowed_zones = filters['vehicle_zone']
            if isinstance(allowed_zones, str):
                allowed_zones = [allowed_zones]
            if meta.get('vehicle_zone', 'unknown') not in allowed_zones:
                return False
        
        # Filtro por área de zona
        if 'zone_area' in filters:
            allowed_areas = filters['zone_area']
            if isinstance(allowed_areas, str):
                allowed_areas = [allowed_areas]
            if meta.get('zone_area', 'unknown') not in allowed_areas:
                return False
        
        # Filtro por número de defectos
        if 'min_defects' in filters:
            if meta.get('total_defects', 0) < filters['min_defects']:
                return False
        
        if 'max_defects' in filters:
            if meta.get('total_defects', 0) > filters['max_defects']:
                return False
        
        return True
    
    def build_rag_context(
        self,
        results: List[SearchResult],
        max_examples: int = 3,
        include_spatial: bool = True
    ) -> str:
        """
        Construye contexto RAG enriquecido con descripciones textuales
        """
        if not results:
            return "No similar examples found in the database."
        
        lines = ["## 🔍 Similar Verified Cases from Database:\n"]
        
        for i, r in enumerate(results[:max_examples], 1):
            lines.append(f"\n### Example {i}:")
            
            # Descripción textual
            lines.append(f"**Description**: {r.description}")
            
            # Similitud
            similarity_pct = (1 - r.distance) * 100
            lines.append(f"**Similarity**: {similarity_pct:.1f}%")
            
            # Detalles de defectos
            lines.append(f"**Total defects**: {r.total_defects}")
            lines.append(f"**Damage types**: {', '.join(r.damage_types)}")
            
            # Info de zona
            lines.append(f"**Vehicle zone**: {r.zone_description} ({r.zone_area})")
            
            # Distribución espacial
            if include_spatial and r.spatial_distribution:
                spatial_info = self._format_spatial_distribution(r.spatial_distribution)
                lines.append(f"**Spatial distribution**: {spatial_info}")
            
            lines.append("")  # Línea en blanco
        
        return "\n".join(lines)
    
    def _format_spatial_distribution(self, spatial_dist: Dict[str, int]) -> str:
        """Formatea distribución espacial en texto"""
        parts = []
        for zone, count in sorted(spatial_dist.items(), key=lambda x: -x[1]):
            zone_formatted = zone.replace('_', ' ').title()
            plural = "s" if count > 1 else ""
            parts.append(f"{count} defect{plural} in {zone_formatted}")
        
        return ", ".join(parts) if parts else "distributed across image"
    
    def get_stats(self) -> Dict:
        """Estadísticas del índice"""
        stats = {
            'n_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'data_type': 'full_images',
            'normalization_enabled': self.enable_normalization
        }
        
        # Estadísticas de dataset
        from collections import Counter
        
        all_types = []
        all_zones = []
        total_defects = 0
        
        for m in self.metadata:
            all_types.extend(m.get('defect_types', []))
            all_zones.append(m.get('vehicle_zone', 'unknown'))
            total_defects += m.get('total_defects', 0)
        
        stats['dataset_stats'] = {
            'total_images': len(self.metadata),
            'total_defects': total_defects,
            'avg_defects_per_image': total_defects / len(self.metadata),
            'damage_type_distribution': dict(Counter(all_types)),
            'zone_distribution': dict(Counter(all_zones))
        }
        
        return stats