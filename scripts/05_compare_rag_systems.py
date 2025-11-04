#!/usr/bin/env python3
# scripts/05_compare_rag_systems.py

"""
🧪 EVALUACIÓN COMPARATIVA DE SISTEMAS RAG MULTIMODAL

Evalúa y compara 6 configuraciones RAG:
- DINOv3 + {high, medium, low} density
- Qwen3VL + {high, medium, low} density

Genera:
1. Métricas cuantitativas (Recall@k, Precision@k, MRR)
2. Análisis de latencia
3. Reportes comparativos en Markdown
4. Análisis cualitativo con VLM

Uso:
    python scripts/05_compare_rag_systems.py \
        --num-queries 20 \
        --use-generation \
        --output-name experiment_01
"""

from pathlib import Path
import sys
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.evaluation.rag_evaluator import RAGEvaluator
from src.core.evaluation.report_generator import RAGReportGenerator

def main():
    parser = argparse.ArgumentParser(
        description='Evaluación comparativa de sistemas RAG multimodal',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Evaluación rápida (10 queries, sin generación VLM)
  python scripts/05_compare_rag_systems.py --num-queries 10

  # Evaluación completa (20 queries, con generación)
  python scripts/05_compare_rag_systems.py \\
      --num-queries 20 \\
      --use-generation \\
      --output-name full_evaluation

  # Evaluación exhaustiva de todas las queries disponibles
  python scripts/05_compare_rag_systems.py \\
      --num-queries -1 \\
      --use-generation \\
      --k 10

  # Evaluación con embeddings Qwen3VL como queries
  python scripts/05_compare_rag_systems.py \\
      --embeddings-type qwen3vl \\
      --num-queries 15
        """
    )
    
    parser.add_argument(
        '--num-queries',
        type=int,
        default=10,
        help='Número de queries a evaluar (-1 = todas disponibles)'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Número de resultados a recuperar (default: 5)'
    )
    parser.add_argument(
        '--use-generation',
        action='store_true',
        help='Generar respuestas con Qwen3VL (más lento pero más completo)'
    )
    parser.add_argument(
        '--embeddings-type',
        type=str,
        default='dinov3',
        choices=['dinov3', 'qwen3vl', 'both'],
        help='Tipo de embeddings a usar para queries de test (both = evaluar con ambos)'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default='evaluation',
        help='Nombre base para archivos de salida'
    )
    parser.add_argument(
        '--qwen-api',
        type=str,
        default='http://localhost:8001',
        help='Endpoint de la API Qwen3VL'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"🚀 EVALUACIÓN COMPARATIVA DE SISTEMAS RAG MULTIMODAL")
    print(f"{'='*70}")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    print(f"\n📋 Configuración:")
    print(f"   - Queries: {args.num_queries if args.num_queries > 0 else 'TODAS'}")
    print(f"   - Top-k: {args.k}")
    print(f"   - Generación VLM: {'✅ Sí' if args.use_generation else '❌ No'}")
    print(f"   - Tipo embeddings queries: {args.embeddings_type}")
    print(f"   - API Qwen3VL: {args.qwen_api}")
    print(f"   - Output: {args.output_name}")
    print(f"{'='*70}\n")
    
    # Confirmar si usa generación
    if args.use_generation and args.num_queries > 10:
        print("⚠️  AVISO: Generación VLM activada con muchas queries.")
        print("   Esto puede tardar varios minutos.\n")
        response = input("¿Continuar? (y/n): ")
        if response.lower() != 'y':
            print("Evaluación cancelada.")
            return
    
    # 1. Inicializar evaluador
    print("\n" + "="*70)
    print("FASE 1: INICIALIZACIÓN")
    print("="*70 + "\n")
    
    evaluator = RAGEvaluator(
        qwen_api_endpoint=args.qwen_api,
        output_dir=Path("outputs/evaluation")
    )
    
    # 2. Cargar configuraciones RAG
    print("\n" + "="*70)
    print("FASE 2: CARGA DE CONFIGURACIONES RAG")
    print("="*70 + "\n")
    
    configs = evaluator.load_rag_configs()
    
    if not configs:
        print("❌ Error: No se encontraron configuraciones RAG")
        print("\n💡 Asegúrate de haber ejecutado:")
        print("   1. scripts/03_generate_embeddings_*.py")
        print("   2. scripts/04_build_faiss_index.py")
        return
    
    # 3. Inicializar retrievers
    print("\n" + "="*70)
    print("FASE 3: INICIALIZACIÓN DE RETRIEVERS")
    print("="*70 + "\n")
    
    evaluator.initialize_retrievers()
    
    # 4. Cargar queries de test
    print("\n" + "="*70)
    print("FASE 4: CARGA DE QUERIES DE TEST")
    print("="*70 + "\n")
    
    # Si se especifica 'both', evaluar con ambos tipos
    if args.embeddings_type == 'both':
        print("📊 Modo: Evaluación con AMBOS tipos de embeddings\n")
        
        # Cargar queries DINOv3
        print("🔵 Cargando queries DINOv3...")
        test_queries_dinov3 = evaluator.load_test_queries(
            test_set_path=Path("data/processed/embeddings"),
            embeddings_type='dinov3'
        )
        
        # Cargar queries Qwen3VL
        print("🟠 Cargando queries Qwen3VL...")
        test_queries_qwen = evaluator.load_test_queries(
            test_set_path=Path("data/processed/embeddings"),
            embeddings_type='qwen3vl'
        )
        
        # Evaluar con DINOv3
        print("\n" + "="*70)
        print("FASE 5A: EVALUACIÓN CON QUERIES DINOV3")
        print("="*70 + "\n")
        
        queries_to_eval_dinov3 = test_queries_dinov3[:args.num_queries] if args.num_queries > 0 else test_queries_dinov3
        print(f"📊 Evaluando {len(queries_to_eval_dinov3)} queries DINOv3\n")
        
        results_dinov3 = evaluator.run_evaluation(
            test_queries=queries_to_eval_dinov3,
            k=args.k,
            num_queries=len(queries_to_eval_dinov3),
            use_generation=args.use_generation
        )
        
        print(f"\n✅ Resultados DINOv3: {len(results_dinov3)} configuraciones")
        for config_name in results_dinov3.keys():
            print(f"   - {config_name}: {len(results_dinov3[config_name])} queries")
        
        # Evaluar con Qwen3VL
        print("\n" + "="*70)
        print("FASE 5B: EVALUACIÓN CON QUERIES QWEN3VL")
        print("="*70 + "\n")
        
        queries_to_eval_qwen = test_queries_qwen[:args.num_queries] if args.num_queries > 0 else test_queries_qwen
        print(f"📊 Evaluando {len(queries_to_eval_qwen)} queries Qwen3VL\n")
        
        results_qwen = evaluator.run_evaluation(
            test_queries=queries_to_eval_qwen,
            k=args.k,
            num_queries=len(queries_to_eval_qwen),
            use_generation=args.use_generation
        )
        
        print(f"\n✅ Resultados Qwen3VL: {len(results_qwen)} configuraciones")
        for config_name in results_qwen.keys():
            print(f"   - {config_name}: {len(results_qwen[config_name])} queries")
        
        # Combinar resultados
        results_by_config = {**results_dinov3, **results_qwen}
        
        print(f"\n📦 Total configuraciones evaluadas: {len(results_by_config)}")
        print(f"   Configuraciones: {list(results_by_config.keys())}")
        
    else:
        # Modo normal: un solo tipo de embedding
        test_queries = evaluator.load_test_queries(
            test_set_path=Path("data/processed/embeddings"),
            embeddings_type=args.embeddings_type
        )
        
        if args.num_queries > 0:
            test_queries = test_queries[:args.num_queries]
        
        print(f"✅ Queries de test: {len(test_queries)}\n")
        
        # 5. Ejecutar evaluación
        print("\n" + "="*70)
        print("FASE 5: EJECUCIÓN DE EVALUACIÓN")
        print("="*70 + "\n")
        
        results_by_config = evaluator.run_evaluation(
            test_queries=test_queries,
            k=args.k,
            num_queries=len(test_queries),
            use_generation=args.use_generation
        )
    
    # 6. Calcular métricas
    print("\n" + "="*70)
    print("FASE 6: CÁLCULO DE MÉTRICAS")
    print("="*70 + "\n")
    
    # Verificar que tenemos resultados
    if not results_by_config:
        print("❌ Error: No se generaron resultados")
        return
    
    print(f"📊 Configuraciones con resultados: {list(results_by_config.keys())}\n")
    
    metrics_by_config = evaluator.calculate_metrics(results_by_config)
    
    # 7. Guardar resultados
    print("\n" + "="*70)
    print("FASE 7: GUARDADO DE RESULTADOS")
    print("="*70 + "\n")
    
    results_path, metrics_path = evaluator.save_results(
        results_by_config=results_by_config,
        metrics_by_config=metrics_by_config,
        output_name=args.output_name
    )
    
    # 8. Generar reportes
    print("\n" + "="*70)
    print("FASE 8: GENERACIÓN DE REPORTES")
    print("="*70 + "\n")
    
    report_gen = RAGReportGenerator(
        output_dir=Path("outputs/evaluation/reports")
    )
    
    # Reporte comparativo
    comparison_report = report_gen.generate_comparison_report(
        metrics_by_config=metrics_by_config,
        output_name=f"{args.output_name}_comparison"
    )
    
    # Reporte detallado
    detailed_report = report_gen.generate_detailed_results(
        results_by_config=results_by_config,
        output_name=f"{args.output_name}_detailed"
    )
    
    # 9. Resumen final
    print("\n" + "="*70)
    print("✨ EVALUACIÓN COMPLETADA")
    print("="*70 + "\n")
    
    print("📊 Resumen de Métricas (Top-3 por Recall@5):\n")
    
    sorted_metrics = sorted(
        metrics_by_config.items(),
        key=lambda x: x[1].recall_at_5,
        reverse=True
    )
    
    for i, (config, m) in enumerate(sorted_metrics[:3], 1):
        print(f"{i}. {config}")
        print(f"   - Recall@5: {m.recall_at_5:.2%}")
        print(f"   - MRR: {m.mrr:.3f}")
        print(f"   - Latencia: {m.avg_total_time_ms:.1f}ms\n")
    
    print("📁 Archivos generados:")
    print(f"   - Resultados: {results_path}")
    print(f"   - Métricas: {metrics_path}")
    print(f"   - Reporte comparativo: {comparison_report}")
    print(f"   - Reporte detallado: {detailed_report}\n")
    
    print("💡 Próximos pasos:")
    print("   1. Revisar reportes en outputs/evaluation/reports/")
    print("   2. Analizar métricas en outputs/evaluation/*.json")
    print("   3. Iterar sobre configuración si es necesario\n")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluación interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error durante la evaluación:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        print("\n📋 Traceback completo:")
        traceback.print_exc()
        sys.exit(1)