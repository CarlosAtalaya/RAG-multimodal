# src/core/evaluation/rag_evaluator.py

from pathlib import Path
import json
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import requests
import base64

@dataclass
class RAGConfig:
    """Configuración de un sistema RAG"""
    name: str
    embedding_type: str  # 'dinov3' o 'qwen3vl'
    density: str  # 'high', 'medium', 'low'
    index_path: Path
    metadata_path: Path
    config_path: Path
    embedding_dim: int


@dataclass
class QueryResult:
    """Resultado de una query RAG"""
    query_id: str
    config_name: str
    
    # Retrieval results
    retrieved_indices: List[int]
    retrieved_distances: List[float]
    retrieved_damage_types: List[str]
    retrieved_crops: List[str]
    
    # Ground truth (si disponible)
    expected_damage_type: Optional[str] = None
    
    # Tiempos
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    
    # Respuesta generada
    generated_answer: str = ""
    rag_context: str = ""


@dataclass
class EvaluationMetrics:
    """Métricas de evaluación de un RAG"""
    config_name: str
    
    # Métricas de retrieval
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    mrr: float  # Mean Reciprocal Rank
    
    # Métricas de latencia
    avg_retrieval_time_ms: float
    avg_generation_time_ms: float
    avg_total_time_ms: float
    
    # Distribución de tipos recuperados
    damage_type_distribution: Dict[str, int]
    
    # Número de queries evaluadas
    num_queries: int


class RAGEvaluator:
    """
    Evaluador comparativo de sistemas RAG multimodales
    
    Funcionalidades:
    1. Carga múltiples configuraciones RAG
    2. Ejecuta queries de test sobre cada config
    3. Calcula métricas cuantitativas
    4. Genera análisis con VLM
    """
    
    def __init__(
        self,
        qwen_api_endpoint: str = "http://localhost:8001",
        output_dir: Path = Path("outputs/evaluation")
    ):
        self.qwen_endpoint = f"{qwen_api_endpoint}/qwen3/chat/completions"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.rag_configs: List[RAGConfig] = []
        self.retrievers: Dict[str, any] = {}
        
        print(f"\n{'='*70}")
        print(f"🧪 RAG EVALUATOR INICIALIZADO")
        print(f"{'='*70}")
        print(f"API Qwen3VL: {qwen_api_endpoint}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def load_rag_configs(
        self,
        base_indices_dir: Path = Path("outputs/vector_indices")
    ):
        """
        Carga automáticamente todas las configuraciones RAG disponibles
        """
        print("📦 Cargando configuraciones RAG...\n")
        
        embedding_types = ['dinov3_indices', 'qwen3_indices']
        densities = ['high', 'medium', 'low']  # Sin sufijo '_density'
        
        for emb_type in embedding_types:
            for density in densities:
                # Construir path: outputs/vector_indices/dinov3_indices/high_density/
                config_dir = base_indices_dir / emb_type / f"{density}_density"
                
                index_path = config_dir / "indexhnswflat_clustered.index"
                metadata_path = config_dir / "metadata_clustered.pkl"
                config_path = config_dir / "index_config_clustered.json"
                
                if not index_path.exists():
                    print(f"⚠️  Skipping {emb_type}/{density}_density: Index not found")
                    print(f"   Buscado en: {index_path}")
                    continue
                
                # Leer config para obtener embedding_dim
                with open(config_path) as f:
                    config_data = json.load(f)
                
                # Nombre limpio: dinov3_high, qwen3_medium, etc.
                emb_name = emb_type.replace('_indices', '')
                config_name = f"{emb_name}_{density}"
                
                config = RAGConfig(
                    name=config_name,
                    embedding_type=emb_name,
                    density=density,
                    index_path=index_path,
                    metadata_path=metadata_path,
                    config_path=config_path,
                    embedding_dim=config_data.get('embedding_dim', 0)
                )
                
                self.rag_configs.append(config)
                
                print(f"✅ {config.name}")
                print(f"   - Embedding: {config.embedding_type} ({config.embedding_dim} dims)")
                print(f"   - Density: {config.density}")
                print(f"   - Index: {index_path.name}\n")
        
        print(f"\n📊 Total configuraciones cargadas: {len(self.rag_configs)}\n")
        
        return self.rag_configs
    
    def initialize_retrievers(self):
        """Inicializa retrievers para cada configuración"""
        from src.core.rag.retriever import DamageRAGRetriever
        
        print("🔧 Inicializando retrievers...\n")
        
        for config in self.rag_configs:
            retriever = DamageRAGRetriever(
                index_path=config.index_path,
                metadata_path=config.metadata_path,
                config_path=config.config_path
            )
            
            self.retrievers[config.name] = retriever
            print(f"✅ {config.name}: {retriever.index.ntotal} vectores\n")
    
    def load_test_queries(
        self,
        test_set_path: Path = None,
        embeddings_type: str = 'dinov3'  # 'dinov3' o 'qwen3vl'
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Carga conjunto de queries de test con sus embeddings
        
        Args:
            test_set_path: Directorio con embeddings de test (opcional)
            embeddings_type: Tipo de embeddings a usar para queries
        
        Returns:
            Lista de tuplas (embedding, metadata)
        """
        # Determinar path según tipo de embedding
        if embeddings_type == 'dinov3':
            base_path = Path("data/processed/embeddings/dinov3_embeddings")
        else:
            base_path = Path("data/processed/embeddings/qwen3vl_embeddings")
        
        # Por defecto usar high_density como test set
        test_dir = base_path / "high_density_20samples"
        
        if not test_dir.exists():
            raise FileNotFoundError(
                f"Test directory not found: {test_dir}\n"
                f"Available directories:\n"
                f"  - {base_path / 'high_density_20samples'}\n"
                f"  - {base_path / 'medium_density_20samples'}\n"
                f"  - {base_path / 'low_density_20samples'}"
            )
        
        # Cargar embeddings
        if embeddings_type == 'dinov3':
            embeddings_file = test_dir / "embeddings_dinov3_vitl.npy"
            metadata_file = test_dir / "enriched_crops_metadata_dinov3_vitl.json"
        else:
            embeddings_file = test_dir / "embeddings_clustered.npy"
            metadata_file = test_dir / "enriched_crops_metadata_clustered.json"
        
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        embeddings = np.load(embeddings_file)
        
        # Cargar metadata
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Crear lista de queries
        queries = []
        for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
            queries.append((emb, meta))
        
        print(f"✅ Cargadas {len(queries)} queries de test ({embeddings_type})")
        print(f"   - Embeddings: {embeddings_file.name}")
        print(f"   - Dimensión: {embeddings.shape[1]}")
        print(f"   - Metadata: {metadata_file.name}\n")
        
        return queries
    
    def run_evaluation(
        self,
        test_queries: List[Tuple[np.ndarray, Dict]],
        k: int = 5,
        num_queries: int = 10,
        use_generation: bool = True
    ) -> Dict[str, List[QueryResult]]:
        """
        Ejecuta evaluación completa sobre todas las configuraciones
        
        Args:
            test_queries: Lista de queries (embedding, metadata)
            k: Número de resultados a recuperar
            num_queries: Número de queries a evaluar (None = todas)
            use_generation: Si True, genera respuestas con VLM
        
        Returns:
            Dict con resultados por configuración
        """
        print(f"\n{'='*70}")
        print(f"🚀 INICIANDO EVALUACIÓN COMPARATIVA")
        print(f"{'='*70}")
        print(f"Queries: {num_queries or len(test_queries)}")
        print(f"Top-k: {k}")
        print(f"Generación VLM: {use_generation}")
        print(f"{'='*70}\n")
        
        # Limitar número de queries
        if num_queries:
            test_queries = test_queries[:num_queries]
        
        # Detectar dimensión de queries
        query_dim = test_queries[0][0].shape[0] if test_queries[0][0].ndim == 1 else test_queries[0][0].shape[1]
        print(f"📏 Dimensión de queries: {query_dim}")
        
        # Filtrar configuraciones compatibles
        compatible_configs = [
            c for c in self.rag_configs 
            if c.embedding_dim == query_dim
        ]
        
        if len(compatible_configs) < len(self.rag_configs):
            incompatible = len(self.rag_configs) - len(compatible_configs)
            print(f"⚠️  {incompatible} configuraciones filtradas por incompatibilidad de dimensiones")
            print(f"✅ Evaluando {len(compatible_configs)} configuraciones compatibles\n")
        
        results_by_config = defaultdict(list)
        
        # Evaluar cada configuración COMPATIBLE
        for config in compatible_configs:
            print(f"\n{'─'*70}")
            print(f"📊 Evaluando: {config.name}")
            print(f"{'─'*70}\n")
            
            retriever = self.retrievers[config.name]
            
            # Evaluar cada query
            for query_idx, (query_emb, query_meta) in enumerate(test_queries):
                # 1. Retrieval
                start_retrieval = time.time()
                search_results = retriever.search(query_emb, k=k)
                retrieval_time = (time.time() - start_retrieval) * 1000
                
                # Extraer información
                retrieved_indices = [r.index for r in search_results]
                retrieved_distances = [r.distance for r in search_results]
                retrieved_damage_types = [r.damage_type for r in search_results]
                retrieved_crops = [r.crop_path for r in search_results]
                
                # Ground truth
                expected_type = query_meta.get('dominant_type') or query_meta.get('damage_type')
                
                # 2. Construcción contexto RAG
                rag_context = retriever.build_rag_context(search_results, max_examples=3)
                
                # 3. Generación (opcional)
                generated_answer = ""
                generation_time = 0.0
                
                if use_generation:
                    try:
                        start_gen = time.time()
                        generated_answer = self._generate_analysis(
                            query_meta['crop_path'],
                            rag_context
                        )
                        generation_time = (time.time() - start_gen) * 1000
                    except Exception as e:
                        print(f"  ⚠️  Error generando respuesta: {e}")
                        generated_answer = "Error en generación"
                
                # Crear resultado
                result = QueryResult(
                    query_id=f"query_{query_idx:03d}",
                    config_name=config.name,
                    retrieved_indices=retrieved_indices,
                    retrieved_distances=retrieved_distances,
                    retrieved_damage_types=retrieved_damage_types,
                    retrieved_crops=retrieved_crops,
                    expected_damage_type=expected_type,
                    retrieval_time_ms=retrieval_time,
                    generation_time_ms=generation_time,
                    generated_answer=generated_answer,
                    rag_context=rag_context
                )
                
                results_by_config[config.name].append(result)
                
                # Log progreso
                if (query_idx + 1) % 5 == 0 or query_idx == len(test_queries) - 1:
                    print(f"  ✓ Procesadas {query_idx + 1}/{len(test_queries)} queries")
        
        print(f"\n{'='*70}")
        print(f"✅ EVALUACIÓN COMPLETADA")
        print(f"{'='*70}\n")
        
        return dict(results_by_config)
    
    def _generate_analysis(
        self,
        image_path: str,
        rag_context: str,
        max_retries: int = 2
    ) -> str:
        """
        Genera análisis usando Qwen3VL con contexto RAG
        """
        # Cargar imagen
        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Prompt con contexto RAG
        prompt = f"""Analiza la imagen de daño vehicular y responde las siguientes preguntas:

1. ¿Qué tipo de daño observas? (surface_scratch, dent, crack, etc.)
2. ¿Dónde está localizado en el vehículo?
3. ¿Cuál es la severidad estimada? (leve, moderada, grave)
4. ¿Hay múltiples daños visibles?

**Contexto de casos similares en la base de datos:**
{rag_context}

**Instrucciones:**
- Usa el contexto de casos similares para fundamentar tu análisis
- Sé preciso y conciso
- Formato: tipo | ubicación | severidad | descripción breve
"""
        
        payload = {
            "model": "Qwen3-VL-4B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300,
            "temperature": 0.1
        }
        
        # Reintentos
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.qwen_endpoint,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                else:
                    raise RuntimeError(f"API status {response.status_code}")
            
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return f"Error: {str(e)}"
        
        return "Error en generación"
    
    def calculate_metrics(
        self,
        results_by_config: Dict[str, List[QueryResult]]
    ) -> Dict[str, EvaluationMetrics]:
        """
        Calcula métricas de evaluación para cada configuración
        """
        print(f"\n{'='*70}")
        print(f"📈 CALCULANDO MÉTRICAS")
        print(f"{'='*70}\n")
        
        metrics_by_config = {}
        
        for config_name, results in results_by_config.items():
            print(f"📊 {config_name}...")
            
            # Métricas de retrieval
            recall_1_scores = []
            recall_3_scores = []
            recall_5_scores = []
            precision_1_scores = []
            precision_3_scores = []
            precision_5_scores = []
            rr_scores = []  # Reciprocal ranks
            
            # Tiempos
            retrieval_times = []
            generation_times = []
            
            # Distribución de tipos
            damage_type_counts = defaultdict(int)
            
            for result in results:
                expected_type = result.expected_damage_type
                retrieved_types = result.retrieved_damage_types
                
                if expected_type:
                    # Recall@k y Precision@k
                    for k in [1, 3, 5]:
                        top_k_types = retrieved_types[:k]
                        
                        # Recall: ¿está el tipo esperado en top-k?
                        recall = 1.0 if expected_type in top_k_types else 0.0
                        
                        # Precision: proporción de tipos correctos en top-k
                        correct = sum(1 for t in top_k_types if t == expected_type)
                        precision = correct / k
                        
                        if k == 1:
                            recall_1_scores.append(recall)
                            precision_1_scores.append(precision)
                        elif k == 3:
                            recall_3_scores.append(recall)
                            precision_3_scores.append(precision)
                        elif k == 5:
                            recall_5_scores.append(recall)
                            precision_5_scores.append(precision)
                    
                    # Mean Reciprocal Rank
                    try:
                        rank = retrieved_types.index(expected_type) + 1
                        rr_scores.append(1.0 / rank)
                    except ValueError:
                        rr_scores.append(0.0)
                
                # Tiempos
                retrieval_times.append(result.retrieval_time_ms)
                generation_times.append(result.generation_time_ms)
                
                # Distribución de tipos
                for dtype in retrieved_types[:5]:
                    damage_type_counts[dtype] += 1
            
            # Calcular promedios
            metrics = EvaluationMetrics(
                config_name=config_name,
                recall_at_1=np.mean(recall_1_scores) if recall_1_scores else 0.0,
                recall_at_3=np.mean(recall_3_scores) if recall_3_scores else 0.0,
                recall_at_5=np.mean(recall_5_scores) if recall_5_scores else 0.0,
                precision_at_1=np.mean(precision_1_scores) if precision_1_scores else 0.0,
                precision_at_3=np.mean(precision_3_scores) if precision_3_scores else 0.0,
                precision_at_5=np.mean(precision_5_scores) if precision_5_scores else 0.0,
                mrr=np.mean(rr_scores) if rr_scores else 0.0,
                avg_retrieval_time_ms=np.mean(retrieval_times),
                avg_generation_time_ms=np.mean(generation_times),
                avg_total_time_ms=np.mean(retrieval_times) + np.mean(generation_times),
                damage_type_distribution=dict(damage_type_counts),
                num_queries=len(results)
            )
            
            metrics_by_config[config_name] = metrics
            
            print(f"  ✓ Recall@5: {metrics.recall_at_5:.2%}")
            print(f"  ✓ MRR: {metrics.mrr:.3f}")
            print(f"  ✓ Avg retrieval: {metrics.avg_retrieval_time_ms:.1f}ms\n")
        
        print(f"{'='*70}\n")
        
        return metrics_by_config
    
    def save_results(
        self,
        results_by_config: Dict[str, List[QueryResult]],
        metrics_by_config: Dict[str, EvaluationMetrics],
        output_name: str = "evaluation_results"
    ):
        """Guarda resultados en JSON"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Convertir a formato serializable
        results_serializable = {}
        for config, results in results_by_config.items():
            results_serializable[config] = [asdict(r) for r in results]
        
        metrics_serializable = {
            config: asdict(m) for config, m in metrics_by_config.items()
        }
        
        # Guardar resultados
        results_path = self.output_dir / f"{output_name}_{timestamp}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        # Guardar métricas
        metrics_path = self.output_dir / f"{output_name}_{timestamp}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"💾 Resultados guardados:")
        print(f"   - {results_path}")
        print(f"   - {metrics_path}\n")
        
        return results_path, metrics_path