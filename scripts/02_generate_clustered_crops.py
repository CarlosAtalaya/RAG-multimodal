# scripts/02b_generate_clustered_crops.py

from pathlib import Path
import json
from tqdm import tqdm
import sys
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.preprocessing.clustered_crop_generator import ClusteredDefectCropGenerator

def process_dataset_with_clustering(
    dataset_dir: Path,
    output_dir: Path
):
    """
    Procesa el dataset usando estrategia de clustering
    
    Args:
        dataset_dir: Directorio con dataset POC
        output_dir: Directorio de salida
    """
    
    generator = ClusteredDefectCropGenerator()
    
    # Crear directorios
    crops_dir = output_dir / "crops" / "clustered"
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar manifest
    manifest_path = dataset_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"🚀 GENERACIÓN DE CROPS CON CLUSTERING ESPACIAL")
    print(f"{'='*70}\n")
    print(f"Dataset: {len(manifest)} imágenes")
    print(f"Output: {crops_dir}\n")
    
    all_crops_metadata = []
    
    # Estadísticas globales
    total_defects = 0
    total_clusters = 0
    
    for item in tqdm(manifest, desc="Procesando imágenes"):
        json_path = dataset_dir / item['json']
        image_path = dataset_dir / item['image']
        
        if not json_path.exists() or not image_path.exists():
            continue
        
        # Cargar anotaciones
        with open(json_path) as f:
            json_data = json.load(f)
        
        # Contar defectos
        defects_in_image = len([
            s for s in json_data['shapes'] 
            if s['shape_type'] == 'polygon'
        ])
        total_defects += defects_in_image
        
        # Generar crops clustered
        try:
            crops_meta = generator.generate_clustered_crops(
                image_path=image_path,
                json_data=json_data,
                output_dir=crops_dir
            )
            
            total_clusters += len(crops_meta)
            all_crops_metadata.extend(crops_meta)
            
        except Exception as e:
            print(f"❌ Error en {image_path.name}: {e}")
            continue
    
    # Guardar metadata
    metadata_path = output_dir / "metadata" / "clustered_crops_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(all_crops_metadata, f, indent=2)
    
    # Resultados finales
    print(f"\n{'='*70}")
    print(f"✅ RESULTADOS FINALES")
    print(f"{'='*70}\n")
    
    print(f"📊 Estadísticas:")
    print(f"   - Defectos totales: {total_defects}")
    print(f"   - Clusters generados: {total_clusters}")
    print(f"   - Reducción: {(1 - total_clusters/total_defects)*100:.1f}%")
    print(f"   - Ratio: {total_defects/total_clusters:.2f} defectos/cluster\n")
    
    print(f"💾 Archivos generados:")
    print(f"   - Crops: {crops_dir}")
    print(f"   - Metadata: {metadata_path}\n")
    
    print(f"\n{'='*70}\n")
    
    return all_crops_metadata


def analyze_cluster_distribution(metadata: List[Dict]):
    """Analiza distribución de clusters generados"""
    
    print(f"\n{'='*70}")
    print(f"📊 ANÁLISIS DE DISTRIBUCIÓN DE CLUSTERS")
    print(f"{'='*70}\n")
    
    single_defect = sum(1 for m in metadata if m['defect_count'] == 1)
    multi_defect = len(metadata) - single_defect
    
    print(f"Tipo de clusters:")
    print(f"   - Single-defect: {single_defect} ({single_defect/len(metadata)*100:.1f}%)")
    print(f"   - Multi-defect: {multi_defect} ({multi_defect/len(metadata)*100:.1f}%)\n")
    
    if multi_defect > 0:
        avg_defects = sum(
            m['defect_count'] for m in metadata 
            if m['defect_count'] > 1
        ) / multi_defect
        
        max_defects = max(m['defect_count'] for m in metadata)
        
        print(f"Clusters multi-defecto:")
        print(f"   - Promedio defectos: {avg_defects:.1f}")
        print(f"   - Máximo defectos: {max_defects}")
    
    # Distribución por tipo de daño dominante
    print(f"\n📦 Distribución por tipo dominante:")
    from collections import Counter
    dominant_types = Counter([m['dominant_type'] for m in metadata])
    
    for dtype, count in dominant_types.most_common():
        print(f"   - {dtype}: {count} clusters")


if __name__ == "__main__":
    DATASET_DIR = Path("data/raw/stratified_subsets/low_density/")
    OUTPUT_DIR = Path("data/processed/crops/low_density_20samples/")
    
    # Generar crops con clustering
    metadata = process_dataset_with_clustering(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR
    )
    
    # Análisis detallado
    analyze_cluster_distribution(metadata)
    
    print("✨ Proceso completado!\n")