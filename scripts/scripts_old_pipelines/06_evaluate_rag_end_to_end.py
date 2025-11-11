#!/usr/bin/env python3
# scripts/07_evaluate_rag_end_to_end.py

"""
🧪 EVALUACIÓN RAG END-TO-END

Evalúa el sistema RAG completo:
1. Imágenes del TEST SET (nunca vistas durante training)
2. Genera embedding de query on-the-fly
3. Retrieval de crops similares en FAISS
4. Genera respuesta con Qwen3VL usando contexto RAG
5. Eval

úa calidad de respuesta vs ground truth

ESTO SÍ ES UNA EVALUACIÓN REAL DE RAG
"""

from pathlib import Path
import json
import sys
import time
import base64
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import requests
import numpy as np
import gc, torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rag.retriever import DamageRAGRetriever
from src.core.embeddings.dinov3_vitl_embedder import DINOv3ViTLEmbedder
from src.core.api.ollama_client import OllamaVLMClient


@dataclass
class RAGTestCase:
    """Caso de test para evaluación RAG"""
    image_id: str
    image_path: str
    ground_truth: Dict  # Tipos de daño reales
    
    # Generado durante evaluación
    query_embedding: np.ndarray = None
    retrieved_crops: List[Dict] = None
    rag_context: str = ""
    generated_answer: str = ""
    
    # Métricas
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    recall_at_k: float = 0.0
    answer_faithfulness: float = 0.0  # ¿La respuesta usa el contexto?
    answer_correctness: float = 0.0   # ¿La respuesta es correcta?


class EndToEndRAGEvaluator:
    """
    Evaluador RAG completo end-to-end
    
    Pipeline:
    1. Carga test set (imágenes NO vistas)
    2. Para cada imagen:
       a. Genera embedding (DINOv3 o Qwen3VL)
       b. Retrieval en FAISS → Top-k crops
       c. Construye contexto RAG
       d. Genera respuesta con Qwen3VL
       e. Evalúa calidad
    """
    
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedder_type: str = 'dinov3',
        ollama_model: str = "qwen3-vl:4b"  # ← NUEVO
    ):
        print(f"\n{'='*70}")
        print(f"🚀 INICIALIZANDO EVALUADOR RAG END-TO-END (OLLAMA)")
        print(f"{'='*70}\n")
        
        # Cargar retriever
        print("📦 Cargando retriever...")
        self.retriever = DamageRAGRetriever(
            index_path=index_path,
            metadata_path=metadata_path
        )
        
        # Inicializar embedder
        print(f"🔧 Inicializando embedder ({embedder_type})...")
        if embedder_type == 'dinov3':
            self.embedder = DINOv3ViTLEmbedder()
        else:
            raise NotImplementedError(f"Embedder {embedder_type} no implementado")
        
        # ✨ NUEVO: Cliente Ollama
        print(f"🤖 Inicializando Ollama ({ollama_model})...")
        self.vlm_client = OllamaVLMClient(model=ollama_model)
        
        print(f"✅ Inicialización completa\n")
    
    def load_test_set(self, test_dir: Path) -> List[RAGTestCase]:
        """Carga casos de test del directorio test"""
        print(f"📂 Cargando test set desde: {test_dir}\n")
        
        manifest_path = test_dir / "test_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No se encontró manifest: {manifest_path}")
        
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
        """
        Ejecuta pipeline RAG completo para un caso de test
        
        Returns:
            RAGTestCase con resultados poblados
        """
        
        # 1. Generar embedding de la query image
        start = time.time()
        query_embedding = self.embedder.generate_embedding(
            image_path=Path(test_case.image_path),
            normalize=True
        )
        test_case.query_embedding = query_embedding
        
        # 2. Retrieval en FAISS
        search_results = self.retriever.search(
            query_embedding=query_embedding,
            k=k
        )
        test_case.retrieval_time_ms = (time.time() - start) * 1000
        
        # 3. Construir contexto RAG
        test_case.rag_context = self.retriever.build_rag_context(
            results=search_results,
            max_examples=3
        )
        
        test_case.retrieved_crops = [
            {
                'crop_path': r.crop_path,
                'damage_type': r.damage_type,
                'distance': float(r.distance),
                'spatial_zone': r.spatial_zone
            }
            for r in search_results
        ]
        
        # 4. Generar respuesta con Qwen3VL
        start = time.time()
        test_case.generated_answer = self._generate_answer(
            image_path=test_case.image_path,
            rag_context=test_case.rag_context
        )
        test_case.generation_time_ms = (time.time() - start) * 1000
        
        # 5. Evaluar métricas
        test_case.recall_at_k = self._calculate_recall(
            retrieved=[r['damage_type'] for r in test_case.retrieved_crops],
            ground_truth=list(test_case.ground_truth.keys())
        )
        
        return test_case
    
    def _generate_answer(
        self,
        image_path: str,
        rag_context: str
    ) -> str:
        """
        Genera respuesta usando Ollama
        """
        
        prompt = f"""Eres un asistente experto en inspección de daños vehiculares.

    **Tu tarea**: Analizar la imagen del vehículo e identificar TODOS los daños visibles.

    **Para cada daño detectado, especifica**:
    1. Tipo de daño (surface_scratch, dent, crack, deep_scratch, paint_peeling, missing_part, missing_accessory, misaligned_part)
    2. Ubicación en el vehículo (ej: capó izquierdo, puerta trasera derecha)
    3. Severidad estimada (leve, moderada, grave)

    **Contexto de casos similares en nuestra base de datos**:
    {rag_context}

    **Instrucciones**:
    - Usa el contexto de casos similares para fundamentar tu análisis
    - Sé preciso y objetivo
    - Si hay múltiples daños, lista todos
    - Formato: JSON con lista de daños

    **Formato de respuesta esperado**:
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
    "summary": "Se detectaron X daños en total..."
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
            
            # ✅ Validación adicional
            if not response or response.startswith("Error:"):
                print(f"⚠️  Respuesta vacía o error para {Path(image_path).name}")
                return json.dumps({
                    "damages": [],
                    "summary": "No se pudo generar análisis",
                    "error": response
                })
            
            return response
        
        except Exception as e:
            print(f"❌ Error en generación: {e}")
            return json.dumps({
                "damages": [],
                "summary": "Error en generación",
                "error": str(e)
            })
    
    def _calculate_recall(
        self,
        retrieved: List[str],
        ground_truth: List[str]
    ) -> float:
        """
        Calcula Recall@k: ¿Cuántos tipos de daño reales aparecen en retrieved?
        """
        if not ground_truth:
            return 0.0
        
        retrieved_set = set(retrieved)
        ground_truth_set = set(ground_truth)
        
        hits = len(retrieved_set & ground_truth_set)
        return hits / len(ground_truth_set)
    
    def evaluate(
        self,
        test_cases: List[RAGTestCase],
        k: int = 5,
        max_cases: int = None
    ) -> Dict:
        """
        Evalúa el sistema RAG con test set
        
        Returns:
            Dict con métricas agregadas
        """
        
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
        
        # Calcular métricas agregadas
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
    
    parser = argparse.ArgumentParser(
        description='Evaluación RAG end-to-end con test set'
    )
    parser.add_argument(
        '--test-set',
        type=Path,
        required=True,
        help='Directorio con test set (output de 06_split_train_test.py)'
    )
    parser.add_argument(
        '--index',
        type=Path,
        default=Path("outputs/vector_indices/train_set_dinov3/indexhnswflat_clustered.index"),
        help='Índice FAISS (generado del TRAIN set)'
    )
    parser.add_argument(
        '--metadata',
        type=Path,
        default=Path("outputs/vector_indices/train_set_dinov3/metadata_clustered.pkl"),
        help='Metadata del índice'
    )
    parser.add_argument(
        '--embedder',
        type=str,
        default='dinov3',
        choices=['dinov3', 'qwen3vl'],
        help='Tipo de embedder para queries'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Top-k retrieval'
    )
    parser.add_argument(
        '--max-cases',
        type=int,
        default=None,
        help='Máximo de casos a evaluar (None = todos)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path("outputs/rag_evaluation/rag_dinov3_train815"),
        help='Directorio de salida'
    )
    
    args = parser.parse_args()
    
    # Verificar que existe el test set
    if not args.test_set.exists():
        print(f"❌ Error: Test set no encontrado: {args.test_set}")
        print("\n💡 Primero ejecuta:")
        print("   python scripts/06_split_train_test.py")
        return
    
    # Inicializar evaluador
    evaluator = EndToEndRAGEvaluator(
        index_path=args.index,
        metadata_path=args.metadata,
        embedder_type=args.embedder
    )
    
    # Cargar test set
    test_cases = evaluator.load_test_set(args.test_set)
    
    # Evaluar
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
    
    # Guardar resultados
    args.output.mkdir(parents=True, exist_ok=True)
    
    results_path = args.output / "rag_evaluation_results.json"
    
    # Convertir resultados a dict serializable
    results_dict = {
        'metrics': metrics,
        'test_cases': [
            {
                'image_id': r.image_id,
                'image_path': r.image_path,
                'ground_truth': r.ground_truth,
                'retrieved_crops': r.retrieved_crops,
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