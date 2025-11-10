#!/usr/bin/env python3
# scripts/06_evaluate_rag_end_to_end.py (ACTUALIZADO para Full Images)

"""
🧪 EVALUACIÓN RAG END-TO-END - FULL IMAGES

Mejoras:
- Query y database en misma escala (full images)
- Contexto RAG enriquecido con zonas
- Métricas mejoradas
"""

from pathlib import Path
import json
import sys
import time
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rag.retriever_fullimages import DamageRAGRetriever
from src.core.embeddings.dinov3_vitl_embedder import DINOv3ViTLEmbedder
from src.core.api.ollama_client import OllamaVLMClient


@dataclass
class RAGTestCase:
    """Caso de test mejorado"""
    image_id: str
    image_path: str
    ground_truth: Dict  # Tipos de daño reales
    
    # Generado durante evaluación
    query_embedding: np.ndarray = None
    retrieved_images: List[Dict] = None
    rag_context: str = ""
    generated_answer: str = ""
    
    # Métricas
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    recall_at_k: float = 0.0


class EndToEndRAGEvaluator:
    """Evaluador completo con full images"""
    
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedder_type: str = 'dinov3',
        ollama_model: str = "qwen3-vl:4b"
    ):
        print(f"\n{'='*70}")
        print(f"🚀 EVALUADOR RAG END-TO-END (FULL IMAGES)")
        print(f"{'='*70}\n")
        
        # Retriever
        print("📦 Cargando retriever...")
        self.retriever = DamageRAGRetriever(
            index_path=index_path,
            metadata_path=metadata_path
        )
        
        # Stats del índice
        stats = self.retriever.get_stats()
        print(f"   ✅ Índice: {stats['n_vectors']} imágenes completas")
        print(f"   ✅ Total defectos: {stats['dataset_stats']['total_defects']}")
        
        # Embedder
        print(f"\n🔧 Inicializando embedder ({embedder_type})...")
        if embedder_type == 'dinov3':
            self.embedder = DINOv3ViTLEmbedder()
        else:
            raise NotImplementedError(f"Embedder {embedder_type} no implementado")
        
        # VLM client
        print(f"\n🤖 Inicializando Ollama ({ollama_model})...")
        self.vlm_client = OllamaVLMClient(model=ollama_model)
        
        print(f"\n✅ Inicialización completa\n")
    
    def load_test_set(self, test_dir: Path) -> List[RAGTestCase]:
        """Carga casos de test"""
        print(f"📂 Cargando test set desde: {test_dir}\n")
        
        manifest_path = test_dir / "test_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest no encontrado: {manifest_path}")
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        test_cases = []
        
        for item in manifest:
            test_cases.append(RAGTestCase(
                image_id=item['id'],
                image_path=str(test_dir / item['image']),
                ground_truth=item['defect_distribution']
            ))
        
        print(f"✅ {len(test_cases)} casos de test cargados\n")
        return test_cases
    
    def run_rag_pipeline(
        self,
        test_case: RAGTestCase,
        k: int = 5
    ) -> RAGTestCase:
        """Pipeline RAG completo"""
        
        # 1. Generar embedding de query (full image)
        start = time.time()
        query_embedding = self.embedder.generate_embedding(
            image_path=Path(test_case.image_path),
            normalize=True
        )
        test_case.query_embedding = query_embedding
        
        # 2. Búsqueda FAISS
        search_results = self.retriever.search(
            query_embedding=query_embedding,
            k=k
        )
        test_case.retrieval_time_ms = (time.time() - start) * 1000
        
        # 3. Construir contexto RAG
        test_case.rag_context = self.retriever.build_rag_context(
            results=search_results,
            max_examples=3,
            include_spatial=True
        )
        
        test_case.retrieved_images = [
            {
                'image_path': r.image_path,
                'damage_types': r.damage_types,
                'distance': float(r.distance),
                'vehicle_zone': r.vehicle_zone,
                'description': r.description
            }
            for r in search_results
        ]
        
        # 4. Generar respuesta VLM
        start = time.time()
        test_case.generated_answer = self._generate_answer(
            image_path=test_case.image_path,
            rag_context=test_case.rag_context
        )
        test_case.generation_time_ms = (time.time() - start) * 1000
        
        # 5. Calcular métricas
        test_case.recall_at_k = self._calculate_recall(
            retrieved=[r['damage_types'] for r in test_case.retrieved_images],
            ground_truth=list(test_case.ground_truth.keys())
        )
        
        return test_case
    
    def _generate_answer(
        self,
        image_path: str,
        rag_context: str
    ) -> str:
        """Genera respuesta con VLM"""
        
        prompt = f"""You are an expert vehicle damage inspector. Analyze this image and identify ALL visible damages.

**Context from similar verified cases in our database**:
{rag_context}

**Your task**:
1. Identify each damage type (scratch, dent, crack, etc.)
2. Specify the location on the vehicle
3. Estimate severity (mild, moderate, severe)
4. Use the context above to inform your analysis

**Response format** (JSON):
```json
{{
  "damages": [
    {{
      "type": "surface_scratch",
      "location": "hood_center",
      "severity": "moderate",
      "confidence": "high"
    }}
  ],
  "summary": "Brief summary of findings"
}}
```

**Important**: Base your analysis on BOTH the image AND the similar cases provided.
"""
        
        try:
            response = self.vlm_client.generate_with_retry(
                prompt=prompt,
                image_path=Path(image_path),
                max_tokens=1024,
                temperature=0.1,
                max_retries=3,
                retry_delay=5.0
            )
            
            if not response or response.startswith("Error:"):
                return json.dumps({
                    "damages": [],
                    "summary": "Analysis failed",
                    "error": response
                })
            
            return response
        
        except Exception as e:
            print(f"❌ Error en generación: {e}")
            return json.dumps({
                "damages": [],
                "summary": "Error in generation",
                "error": str(e)
            })
    
    def _calculate_recall(
        self,
        retrieved: List[List[str]],
        ground_truth: List[str]
    ) -> float:
        """Calcula Recall@k"""
        if not ground_truth:
            return 0.0
        
        # Flatten retrieved types
        retrieved_flat = set()
        for types_list in retrieved:
            retrieved_flat.update(types_list)
        
        ground_truth_set = set(ground_truth)
        
        hits = len(retrieved_flat & ground_truth_set)
        return hits / len(ground_truth_set)
    
    def evaluate(
        self,
        test_cases: List[RAGTestCase],
        k: int = 5,
        max_cases: int = None
    ) -> Dict:
        """Evalúa el sistema RAG"""
        
        if max_cases:
            test_cases = test_cases[:max_cases]
        
        print(f"\n{'='*70}")
        print(f"🧪 EVALUANDO RAG END-TO-END")
        print(f"{'='*70}")
        print(f"Test cases: {len(test_cases)}")
        print(f"Top-k retrieval: {k}")
        print(f"{'='*70}\n")
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] Evaluando {Path(test_case.image_path).name}...", end=' ')
            
            try:
                result = self.run_rag_pipeline(test_case, k=k)
                results.append(result)
                print(f"✓ Recall@{k}={result.recall_at_k:.2%}")
            
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        # Métricas agregadas
        metrics = self._aggregate_metrics(results)
        
        return {
            'results': results,
            'metrics': metrics
        }
    
    def _aggregate_metrics(self, results: List[RAGTestCase]) -> Dict:
        """Calcula métricas agregadas"""
        
        if not results:
            return {}
        
        metrics = {
            'num_cases': len(results),
            'avg_recall_at_k': np.mean([r.recall_at_k for r in results]),
            'avg_retrieval_time_ms': np.mean([r.retrieval_time_ms for r in results]),
            'avg_generation_time_ms': np.mean([r.generation_time_ms for r in results]),
            'avg_total_time_ms': np.mean([
                r.retrieval_time_ms + r.generation_time_ms 
                for r in results
            ])
        }
        
        return metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test-set',
        type=Path,
        required=True,
        help='Directorio con test set'
    )
    parser.add_argument(
        '--index',
        type=Path,
        default=Path("outputs/vector_indices/fullimages_dinov3/indexhnswflat_fullimages.index"),
        help='Índice FAISS'
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        default=Path("outputs/vector_indices/fullimages_dinov3/metadata_fullimages.pkl"),
        help='Metadata'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5
    )
    parser.add_argument(
        '--max-cases',
        type=int,
        default=None
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("outputs/rag_evaluation_fullimages")
    )
    
    args = parser.parse_args()
    
    # Verificar paths
    if not args.test_set.exists():
        print(f"❌ Test set no encontrado: {args.test_set}")
        return
    
    # Evaluar
    evaluator = EndToEndRAGEvaluator(
        index_path=args.index,
        metadata_path=args.metadata
    )
    
    test_cases = evaluator.load_test_set(args.test_set)
    
    evaluation_results = evaluator.evaluate(
        test_cases=test_cases,
        k=args.k,
        max_cases=args.max_cases
    )
    
    # Mostrar resultados
    metrics = evaluation_results['metrics']
    
    print(f"\n{'='*70}")
    print(f"📊 RESULTADOS FINALES")
    print(f"{'='*70}")
    print(f"Casos evaluados: {metrics['num_cases']}")
    print(f"Recall@{args.k} promedio: {metrics['avg_recall_at_k']:.2%}")
    print(f"Retrieval time: {metrics['avg_retrieval_time_ms']:.1f}ms")
    print(f"Generation time: {metrics['avg_generation_time_ms']:.0f}ms")
    print(f"Total time: {metrics['avg_total_time_ms']:.0f}ms")
    print(f"{'='*70}\n")
    
    # Guardar
    args.output.mkdir(parents=True, exist_ok=True)
    
    results_path = args.output / "evaluation_results_fullimages.json"
    
    results_dict = {
        'metrics': metrics,
        'test_cases': [
            {
                'image_id': r.image_id,
                'image_path': r.image_path,
                'ground_truth': r.ground_truth,
                'retrieved_images': r.retrieved_images,
                'generated_answer': r.generated_answer,
                'recall_at_k': r.recall_at_k,
                'retrieval_time_ms': r.retrieval_time_ms,
                'generation_time_ms': r.generation_time_ms
            }
            for r in evaluation_results['results']
        ]
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"💾 Resultados guardados: {results_path}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluación interrumpida")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()