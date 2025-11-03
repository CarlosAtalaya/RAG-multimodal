# src/core/rag/retriever.py

from pathlib import Path
import json
import numpy as np
import faiss
import pickle
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Resultado de una búsqueda"""
    index: int
    distance: float
    crop_path: str
    damage_type: str
    image_path: str
    bbox: List[float]
    spatial_zone: str
    metadata: Dict

class DamageRAGRetriever:
    """
    Retriever para el sistema RAG multimodal
    """
    
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        config_path: Path = None
    ):
        """
        Inicializa el retriever
        
        Args:
            index_path: Ruta al índice FAISS
            metadata_path: Ruta a metadata (pickle)
            config_path: Ruta a configuración del índice
        """
        print(f"🔧 Inicializando DamageRAGRetriever...")
        
        # Cargar índice FAISS
        self.index = faiss.read_index(str(index_path))
        print(f"   ✅ Índice FAISS cargado: {self.index.ntotal} vectores")
        
        # Cargar metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        print(f"   ✅ Metadata cargada: {len(self.metadata)} entries")
        
        # Cargar config si existe
        self.config = {}
        if config_path and config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        
        self.embedding_dim = self.index.d
        print(f"   ✅ Dimensión embeddings: {self.embedding_dim}")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Búsqueda de similitud en el índice
        
        Args:
            query_embedding: Vector de query (1, dim) o (dim,)
            k: Número de resultados a retornar
            filters: Filtros opcionales (damage_type, spatial_zone, etc.)
        
        Returns:
            Lista de SearchResult ordenados por similitud
        """
        # Asegurar shape correcto
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = query_embedding.astype('float32')
        
        # Búsqueda en FAISS
        # Si hay filtros, buscar más resultados y filtrar después
        k_search = k * 5 if filters else k
        k_search = min(k_search, self.index.ntotal)
        
        distances, indices = self.index.search(query_embedding, k_search)
        
        # Construir resultados
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS retorna -1 si no hay suficientes resultados
                continue
            
            meta = self.metadata[idx]
            
            # Aplicar filtros si existen
            if filters:
                if not self._apply_filters(meta, filters):
                    continue
            
            result = SearchResult(
                index=int(idx),
                distance=float(dist),
                crop_path=meta['crop_path'],
                damage_type=meta['damage_type'],
                image_path=meta['image_path'],
                bbox=meta['bbox'],
                spatial_zone=meta['spatial_zone'],
                metadata=meta
            )
            
            results.append(result)
            
            if len(results) >= k:
                break
        
        return results
    
    def _apply_filters(self, meta: Dict, filters: Dict) -> bool:
        """Aplica filtros a un resultado"""
        
        # Filtro por tipo de daño
        if 'damage_type' in filters:
            allowed_types = filters['damage_type']
            if isinstance(allowed_types, str):
                allowed_types = [allowed_types]
            if meta['damage_type'] not in allowed_types:
                return False
        
        # Filtro por zona espacial
        if 'spatial_zone' in filters:
            allowed_zones = filters['spatial_zone']
            if isinstance(allowed_zones, str):
                allowed_zones = [allowed_zones]
            if meta['spatial_zone'] not in allowed_zones:
                return False
        
        # Filtro por tamaño relativo
        if 'size_category' in filters:
            allowed_sizes = filters['size_category']
            if isinstance(allowed_sizes, str):
                allowed_sizes = [allowed_sizes]
            if meta['size_category'] not in allowed_sizes:
                return False
        
        return True
    
    def get_similar_damages(
        self,
        query_embedding: np.ndarray,
        damage_type: Optional[str] = None,
        k: int = 5
    ) -> List[SearchResult]:
        """
        Wrapper conveniente para búsqueda con filtro de tipo de daño
        """
        filters = {'damage_type': damage_type} if damage_type else None
        return self.search(query_embedding, k=k, filters=filters)
    
    def build_rag_context(
        self,
        results: List[SearchResult],
        max_examples: int = 3
    ) -> str:
        """
        Construye contexto para el prompt RAG
        
        Args:
            results: Lista de SearchResult
            max_examples: Número máximo de ejemplos a incluir
        
        Returns:
            String con contexto formateado
        """
        if not results:
            return "No se encontraron ejemplos similares en la base de datos."
        
        context_parts = [
            "## Ejemplos Similares de la Base de Datos:\n"
        ]
        
        for i, result in enumerate(results[:max_examples], 1):
            context_parts.append(f"\n### Ejemplo {i}:")
            context_parts.append(f"- **Tipo de daño**: {result.damage_type.replace('_', ' ')}")
            context_parts.append(f"- **Zona espacial**: {result.spatial_zone}")
            context_parts.append(f"- **Similitud**: {1 - result.distance:.2%}")
            context_parts.append(f"- **Imagen**: {Path(result.image_path).name}")
            
            # Info adicional del metadata
            if 'size_category' in result.metadata:
                context_parts.append(f"- **Tamaño**: {result.metadata['size_category']}")
            
            if 'edge_defect' in result.metadata:
                edge_status = "Sí" if result.metadata['edge_defect'] else "No"
                context_parts.append(f"- **En borde**: {edge_status}")
        
        return "\n".join(context_parts)