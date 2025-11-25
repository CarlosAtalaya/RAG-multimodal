#!/usr/bin/env python3
# scripts/06b_evaluate_rag_hybrid.py (VERSIÓN CON ERROR HANDLING)

"""
🧪 EVALUACIÓN RAG CON EMBEDDINGS HÍBRIDOS - VERSIÓN ROBUSTA
"""

from pathlib import Path
import json
import sys
import time
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rag.retriever_fullimages import DamageRAGRetriever
from src.core.embeddings.multimodal_embedder import MultimodalEmbedder
from src.core.api.ollama_client import OllamaVLMClient


@dataclass
class HybridRAGTestCase:
    """Caso de test para evaluación híbrida"""
    image_id: str
    image_path: str
    ground_truth: Dict
    
    # Query metadata
    query_metadata: Dict = None
    
    # Resultados
    query_embedding: np.ndarray = None
    retrieved_images: List[Dict] = None
    rag_context: str = ""
    generated_answer: str = ""
    
    # Métricas
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    recall_at_k: float = 0.0
    
    # Debug
    error: str = ""


class HybridRAGEvaluator:
    """Evaluador para RAG con embeddings híbridos"""
    
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        visual_weight: float = 0.6,
        text_weight: float = 0.4,
        ollama_model: str = "qwen3-vl:4b"
    ):
        print(f"\n{'='*70}")
        print(f"🌟 EVALUADOR RAG HÍBRIDO (Visual + Text)")
        print(f"{'='*70}\n")
        
        # Retriever
        print("📦 Cargando retriever híbrido...")
        self.retriever = DamageRAGRetriever(
            index_path=index_path,
            metadata_path=metadata_path
        )
        
        stats = self.retriever.get_stats()
        print(f"   ✅ Índice: {stats['n_vectors']} vectores")
        print(f"   ✅ Dimensión: {stats['embedding_dim']}")
        print(f"   ✅ Tipo: {stats['data_type']}")
        
        # Multimodal embedder
        print(f"\n🔧 Inicializando MultimodalEmbedder...")
        self.embedder = MultimodalEmbedder(
            visual_weight=visual_weight,
            text_weight=text_weight
        )
        
        # VLM client
        print(f"\n🤖 Inicializando Ollama ({ollama_model})...")
        self.vlm_client = OllamaVLMClient(model=ollama_model)
        
        print(f"\n✅ Inicialización completa\n")
    
    def load_test_set(self, test_dir: Path) -> List[HybridRAGTestCase]:
        """Carga test set"""
        print(f"📂 Cargando test set desde: {test_dir}\n")
        
        manifest_path = test_dir / "test_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest no encontrado: {manifest_path}")
        
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        test_cases = []
        
        for item in manifest:
            # Metadata de query
            query_metadata = {
                'defect_types': list(item['defect_distribution'].keys()),
                'vehicle_zone': item.get('vehicle_zone', 'unknown'),
                'zone_description': item.get('zone_description', 'unknown'),
                'zone_area': item.get('zone_area', 'unknown'),
                'total_defects': item.get('total_defects', 0)
            }
            
            test_cases.append(HybridRAGTestCase(
                image_id=item['id'],
                image_path=str(test_dir / item['image']),
                ground_truth=item['defect_distribution'],
                query_metadata=query_metadata
            ))
        
        print(f"✅ {len(test_cases)} casos de test cargados\n")
        return test_cases
    
    def run_rag_pipeline(
        self,
        test_case: HybridRAGTestCase,
        k: int = 5
    ) -> HybridRAGTestCase:
        """Pipeline RAG completo con error handling"""
        
        try:
            # 1. Generar embedding híbrido de query
            start = time.time()
            
            query_embedding, debug_info = self.embedder.generate_hybrid_embedding(
                image_path=Path(test_case.image_path),
                metadata=test_case.query_metadata,
                normalize=True
            )
            test_case.query_embedding = query_embedding
            
            # 2. Búsqueda FAISS
            search_results = self.retriever.search(
                query_embedding=query_embedding,
                k=k
            )
            test_case.retrieval_time_ms = (time.time() - start) * 1000
            
            if not search_results:
                test_case.error = "No search results returned"
                return test_case
            
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
                    'zone_area': r.zone_area,
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
            
        except Exception as e:
            test_case.error = str(e)
            print(f"\n⚠️  Error detallado: {traceback.format_exc()}")
        
        return test_case
    
    def _generate_answer(self, image_path: str, rag_context: str) -> str:
        """Genera respuesta con VLM"""
        
        prompt = f"""You are an expert vehicle damage inspector. Analyze this image and identify ALL visible damages.

**Context from similar verified cases**:
{rag_context}

**Your task**:
1. Identify each damage type
2. Specify location
3. Estimate severity

**Response format** (JSON):
```json
{{
  "damages": [
    {{
      "type": "surface_scratch",
      "location": "hood_center",
      "severity": "moderate"
    }}
  ],
  "summary": "Brief summary"
}}
```
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
                return json.dumps({"damages": [], "summary": "Analysis failed", "error": response})
            
            return response
        
        except Exception as e:
            return json.dumps({"damages": [], "summary": "Error", "error": str(e)})
    
    def _calculate_recall(
        self,
        retrieved: List[List[str]],
        ground_truth: List[str]
    ) -> float:
        """Calcula Recall@k"""
        if not ground_truth:
            return 0.0
        
        retrieved_flat = set()
        for types_list in retrieved:
            retrieved_flat.update(types_list)
        
        ground_truth_set = set(ground_truth)
        
        # Normalizar
        from src.core.rag.taxonomy_normalizer import TaxonomyNormalizer
        normalizer = TaxonomyNormalizer()
        
        retrieved_normalized = set([
            normalizer.normalize(dt)['benchmark_label']
            for dt in retrieved_flat
        ])
        
        gt_normalized = set([
            normalizer.normalize(dt)['benchmark_label']
            for dt in ground_truth_set
        ])
        
        hits = len(retrieved_normalized & gt_normalized)
        return hits / len(gt_normalized) if gt_normalized else 0.0
    
    def evaluate(
        self,
        test_cases: List[HybridRAGTestCase],
        k: int = 5,
        max_cases: int = None
    ) -> Dict:
        """Evalúa el sistema RAG híbrido"""
        
        if max_cases:
            test_cases = test_cases[:max_cases]
        
        print(f"\n{'='*70}")
        print(f"🧪 EVALUANDO RAG HÍBRIDO END-TO-END")
        print(f"{'='*70}")
        print(f"Test cases: {len(test_cases)}")
        print(f"Top-k retrieval: {k}")
        print(f"{'='*70}\n")
        
        results = []
        errors = []
        
        for i, test_case in enumerate(test_cases, 1):
            img_name = Path(test_case.image_path).name
            print(f"[{i}/{len(test_cases)}] Evaluando {img_name[:60]}...", end=' ')
            
            try:
                result = self.run_rag_pipeline(test_case, k=k)
                
                if result.error:
                    print(f"✗ Error: {result.error}")
                    errors.append((img_name, result.error))
                else:
                    print(f"✓ Recall@{k}={result.recall_at_k:.2%}")
                    results.append(result)
            
            except Exception as e:
                print(f"✗ Exception: {e}")
                errors.append((img_name, str(e)))
        
        # Métricas agregadas
        metrics = self._aggregate_metrics(results, errors, len(test_cases))
        
        return {
            'results': results,
            'errors': errors,
            'metrics': metrics
        }
    
    def _aggregate_metrics(
        self, 
        results: List[HybridRAGTestCase],
        errors: List[tuple],
        total_cases: int
    ) -> Dict:
        """Calcula métricas agregadas con manejo de casos sin datos"""
        
        metrics = {
            'total_cases': total_cases,
            'successful_cases': len(results),
            'failed_cases': len(errors),
            'success_rate': len(results) / total_cases if total_cases > 0 else 0.0
        }
        
        if results:
            recalls = [r.recall_at_k for r in results]
            
            metrics.update({
                'avg_recall_at_k': np.mean(recalls),
                'std_recall_at_k': np.std(recalls),
                'median_recall_at_k': np.median(recalls),
                'min_recall_at_k': min(recalls),
                'max_recall_at_k': max(recalls),
                'avg_retrieval_time_ms': np.mean([r.retrieval_time_ms for r in results]),
                'avg_generation_time_ms': np.mean([r.generation_time_ms for r in results]),
                'avg_total_time_ms': np.mean([
                    r.retrieval_time_ms + r.generation_time_ms 
                    for r in results
                ]),
                'recall_distribution': {
                    'zero': sum(1 for r in results if r.recall_at_k == 0.0),
                    'low': sum(1 for r in results if 0.0 < r.recall_at_k <= 0.3),
                    'medium': sum(1 for r in results if 0.3 < r.recall_at_k <= 0.7),
                    'high': sum(1 for r in results if r.recall_at_k > 0.7)
                }
            })
        else:
            # Si no hay resultados válidos
            metrics.update({
                'avg_recall_at_k': 0.0,
                'std_recall_at_k': 0.0,
                'median_recall_at_k': 0.0,
                'min_recall_at_k': 0.0,
                'max_recall_at_k': 0.0,
                'avg_retrieval_time_ms': 0.0,
                'avg_generation_time_ms': 0.0,
                'avg_total_time_ms': 0.0,
                'recall_distribution': {
                    'zero': 0,
                    'low': 0,
                    'medium': 0,
                    'high': 0
                }
            })
        
        return metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-set', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--visual-weight', type=float, default=0.6)
    parser.add_argument('--text-weight', type=float, default=0.4)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--max-cases', type=int, default=None)
    parser.add_argument('--output', type=Path, default=Path("outputs/rag_evaluation_hybrid"))
    
    args = parser.parse_args()
    
    # Evaluar
    evaluator = HybridRAGEvaluator(
        index_path=args.index,
        metadata_path=args.metadata,
        visual_weight=args.visual_weight,
        text_weight=args.text_weight
    )
    
    test_cases = evaluator.load_test_set(args.test_set)
    
    evaluation_results = evaluator.evaluate(
        test_cases=test_cases,
        k=args.k,
        max_cases=args.max_cases
    )
    
    # Mostrar resultados
    metrics = evaluation_results['metrics']
    errors = evaluation_results['errors']
    
    print(f"\n{'='*70}")
    print(f"📊 RESULTADOS FINALES - HYBRID RAG")
    print(f"{'='*70}")
    print(f"Total casos: {metrics['total_cases']}")
    print(f"Casos exitosos: {metrics['successful_cases']} ({metrics['success_rate']:.1%})")
    print(f"Casos fallidos: {metrics['failed_cases']}")
    
    if metrics['successful_cases'] > 0:
        print(f"\nRecall@{args.k}:")
        print(f"  - Promedio: {metrics['avg_recall_at_k']:.2%} (±{metrics['std_recall_at_k']:.2%})")
        print(f"  - Mediana: {metrics['median_recall_at_k']:.2%}")
        print(f"  - Rango: {metrics['min_recall_at_k']:.2%} - {metrics['max_recall_at_k']:.2%}")
        
        print(f"\nDistribución de Recall:")
        print(f"  - Zero (0%):        {metrics['recall_distribution']['zero']:>3} casos")
        print(f"  - Low (0-30%):      {metrics['recall_distribution']['low']:>3} casos")
        print(f"  - Medium (30-70%):  {metrics['recall_distribution']['medium']:>3} casos")
        print(f"  - High (>70%):      {metrics['recall_distribution']['high']:>3} casos")
        
        print(f"\nTiempos:")
        print(f"  - Retrieval: {metrics['avg_retrieval_time_ms']:.1f}ms")
        print(f"  - Generation: {metrics['avg_generation_time_ms']:.0f}ms")
        print(f"  - Total: {metrics['avg_total_time_ms']:.0f}ms")
    
    if errors:
        print(f"\n⚠️  Errores detectados:")
        for img_name, error in errors[:5]:  # Mostrar primeros 5
            print(f"  - {img_name}: {error}")
        if len(errors) > 5:
            print(f"  ... y {len(errors)-5} más")
    
    print(f"{'='*70}\n")
    
    # Guardar
    args.output.mkdir(parents=True, exist_ok=True)
    
    results_path = args.output / "evaluation_results_hybrid.json"
    
    results_dict = {
        'config': {
            'visual_weight': args.visual_weight,
            'text_weight': args.text_weight,
            'k': args.k
        },
        'metrics': metrics,
        'errors': [{'image': e[0], 'error': e[1]} for e in errors],
        'test_cases': [
            {
                'image_id': r.image_id,
                'image_path': r.image_path,
                'ground_truth': r.ground_truth,
                'query_metadata': r.query_metadata,
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
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()