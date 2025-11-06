# src/core/rag/retriever.py

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
    """Resultado de búsqueda con labels normalizados"""
    index: int
    distance: float
    crop_path: str
    damage_type: str                      # Normalizado
    damage_type_original: str = ""        # Original
    damage_type_confidence: float = 1.0   # Confianza
    image_path: str = ""
    bbox: List[float] = field(default_factory=list)
    spatial_zone: str = ""
    metadata: Dict = field(default_factory=dict)
    all_damage_types: List[str] = field(default_factory=list)


class DamageRAGRetriever:
    """Retriever con normalización taxonómica automática"""
    
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        config_path: Path = None,
        enable_taxonomy_normalization: bool = True
    ):
        print(f"🔧 Inicializando DamageRAGRetriever...")
        
        # Cargar índice y metadata
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.embedding_dim = self.index.d
        
        print(f"   ✅ Índice: {self.index.ntotal} vectores")
        print(f"   ✅ Metadata: {len(self.metadata)} entries")
        
        # Config opcional
        self.config = {}
        if config_path and config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        
        # Taxonomy normalizer
        self.enable_normalization = enable_taxonomy_normalization
        if self.enable_normalization:
            self.taxonomy_normalizer = TaxonomyNormalizer()
            coverage = self._validate_coverage()
            print(f"   ✅ Taxonomy normalizer: {coverage['coverage_percent']:.1f}% coverage")
            
            if coverage['unmapped_samples'] > 0:
                print(f"   ⚠️  {coverage['unmapped_samples']} samples sin mapeo")
        else:
            self.taxonomy_normalizer = None
            print(f"   ℹ️  Taxonomy normalizer desactivado")
    
    def _validate_coverage(self) -> Dict:
        """Valida cobertura taxonómica del índice"""
        if not self.taxonomy_normalizer:
            return {}
        
        labels = []
        for m in self.metadata:
            label = m.get('dominant_type') or m.get('damage_type') or 'unknown'
            labels.append(label)
            
            # Si es cluster, añadir todos
            if 'damage_types' in m:
                labels.extend(m['damage_types'])
        
        return self.taxonomy_normalizer.get_coverage_stats(labels)
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: Optional[Dict] = None,
        return_normalized: bool = True
    ) -> List[SearchResult]:
        """
        Búsqueda con normalización automática de labels
        
        Args:
            query_embedding: Vector de consulta
            k: Número de resultados
            filters: Filtros opcionales
            return_normalized: Retornar labels normalizados
        """
        # Preparar query
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype('float32')
        
        # Búsqueda FAISS
        k_search = k * 5 if filters else k
        k_search = min(k_search, self.index.ntotal)
        
        distances, indices = self.index.search(query_embedding, k_search)
        
        # Construir resultados
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            meta = self.metadata[idx]
            train_label = meta.get('dominant_type') or meta.get('damage_type') or 'unknown'
            
            # Normalizar label
            if return_normalized and self.taxonomy_normalizer:
                norm = self.taxonomy_normalizer.normalize(train_label)
                benchmark_label = norm['benchmark_label']
                confidence = norm['confidence']
            else:
                benchmark_label = train_label
                confidence = 1.0
            
            # Aplicar filtros
            if filters and not self._apply_filters(meta, filters, benchmark_label):
                continue
            
            # Todos los damage types (si es cluster)
            all_types = []
            if 'damage_types' in meta:
                all_types = meta['damage_types']
                if return_normalized and self.taxonomy_normalizer:
                    all_types = [
                        self.taxonomy_normalizer.normalize(t)['benchmark_label']
                        for t in all_types
                    ]
            
            result = SearchResult(
                index=int(idx),
                distance=float(dist),
                crop_path=meta.get('crop_path', ''),
                damage_type=benchmark_label,
                damage_type_original=train_label,
                damage_type_confidence=confidence,
                image_path=meta.get('source_image', meta.get('image_path', '')),
                bbox=meta.get('cluster_bbox', meta.get('bbox', [])),
                spatial_zone=meta.get('spatial_zone', 'unknown'),
                metadata=meta,
                all_damage_types=all_types
            )
            
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def _apply_filters(self, meta: Dict, filters: Dict, norm_label: str = None) -> bool:
        """Aplica filtros a un resultado"""
        # Filtro por tipo
        if 'damage_type' in filters:
            allowed = filters['damage_type']
            if isinstance(allowed, str):
                allowed = [allowed]
            
            if norm_label and norm_label not in allowed:
                return False
        
        # Filtro por zona
        if 'spatial_zone' in filters:
            allowed = filters['spatial_zone']
            if isinstance(allowed, str):
                allowed = [allowed]
            if meta.get('spatial_zone', 'unknown') not in allowed:
                return False
        
        return True
    
    def build_rag_context(
        self,
        results: List[SearchResult],
        max_examples: int = 3,
        include_confidence: bool = False
    ) -> str:
        """Construye contexto RAG con labels normalizados"""
        if not results:
            return "No similar examples found."
        
        lines = ["## 🔍 Similar Verified Cases:\n"]
        
        for i, r in enumerate(results[:max_examples], 1):
            lines.append(f"\n### Example {i}:")
            lines.append(f"- **Damage Type**: {r.damage_type}")
            lines.append(f"- **Vehicle Area**: {self._format_zone(r.spatial_zone)}")
            lines.append(f"- **Similarity**: {(1 - r.distance) * 100:.1f}%")
            
            # Indicador de confianza (opcional)
            if include_confidence and r.damage_type_confidence < 0.95:
                lines.append(
                    f"- **Note**: Approximate match (original: {r.damage_type_original})"
                )
            
            # Cluster con múltiples tipos
            if r.all_damage_types and len(r.all_damage_types) > 1:
                types = ", ".join(set(r.all_damage_types))
                lines.append(f"- **Additional damages**: {types}")
        
        return "\n".join(lines)
    
    def _format_zone(self, zone: str) -> str:
        """Formatea zonas espaciales"""
        zones = {
            "top_left": "Upper left", "top_center": "Upper center", "top_right": "Upper right",
            "middle_left": "Left side", "middle_center": "Center", "middle_right": "Right side",
            "bottom_left": "Lower left", "bottom_center": "Lower center", "bottom_right": "Lower right"
        }
        return zones.get(zone, zone)
    
    def get_stats(self) -> Dict:
        """Estadísticas del índice"""
        stats = {
            'n_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'normalization_enabled': self.enable_normalization
        }
        
        if self.enable_normalization:
            stats['taxonomy_coverage'] = self._validate_coverage()
        
        return stats