from pathlib import Path
import json
from tqdm import tqdm
import sys

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.preprocessing.crop_generator import AdaptiveCropGenerator

def process_dataset(
    dataset_dir: Path,
    output_dir: Path
):
    """Procesa todo el dataset POC generando crops"""
    
    generator = AdaptiveCropGenerator()
    
    # NUEVO: Calcular estadísticas del dataset primero
    manifest_path = dataset_dir / "poc_manifest.json"
    generator.compute_dataset_statistics(manifest_path)
    
    # Crear directorios
    crops_dir = output_dir / "crops" / "roi"
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    all_crops_metadata = []
    skipped_count = 0
    
    print(f"\n{'='*70}")
    print(f"Procesando {len(manifest)} imágenes del dataset POC")
    print(f"{'='*70}\n")
    
    for item in tqdm(manifest, desc="Generando crops"):
        json_path = dataset_dir / item['json']
        image_path = dataset_dir / item['image']
        
        # Verificar que existen los archivos
        if not json_path.exists():
            print(f"⚠️  JSON no encontrado: {json_path}")
            continue
        if not image_path.exists():
            print(f"⚠️  Imagen no encontrada: {image_path}")
            continue
        
        # Cargar anotaciones
        with open(json_path) as f:
            json_data = json.load(f)
        
        # Contar defectos totales en esta imagen
        total_defects = len([s for s in json_data['shapes'] if s['shape_type'] == 'polygon'])
        
        # Generar crops
        try:
            crops_meta = generator.generate_crops_from_json(
                image_path=image_path,
                json_data=json_data,
                output_dir=crops_dir
            )
            
            crops_generated = len(crops_meta)
            skipped_in_image = total_defects - crops_generated
            skipped_count += skipped_in_image
            
            all_crops_metadata.extend(crops_meta)
            
        except Exception as e:
            print(f"❌ Error procesando {image_path.name}: {e}")
            continue
    
    # Guardar metadata de todos los crops
    metadata_path = output_dir / "metadata" / "crops_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(all_crops_metadata, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✅ Generados {len(all_crops_metadata)} crops")
    print(f"⚠️  Descartados {skipped_count} crops por umbrales")
    print(f"📊 Tasa de aprovechamiento: {len(all_crops_metadata)/(len(all_crops_metadata)+skipped_count)*100:.1f}%")
    print(f"📄 Metadata guardada en: {metadata_path}")
    print(f"{'='*70}\n")
    
    # Estadísticas detalladas
    print("📊 Distribución de crops por tipo de daño:")
    damage_counts = {}
    for meta in all_crops_metadata:
        dtype = meta['damage_type']
        damage_counts[dtype] = damage_counts.get(dtype, 0) + 1
    
    for dtype, count in sorted(damage_counts.items(), key=lambda x: -x[1]):
        print(f"  {dtype}: {count} crops")
    
    # Estadísticas espaciales (NUEVO)
    print("\n📍 Distribución espacial:")
    spatial_counts = {}
    for meta in all_crops_metadata:
        zone = meta['spatial_zone']
        spatial_counts[zone] = spatial_counts.get(zone, 0) + 1
    
    for zone, count in sorted(spatial_counts.items(), key=lambda x: -x[1]):
        print(f"  {zone}: {count} crops")
    
    # Estadísticas de tamaño (NUEVO)
    print("\n📏 Distribución por tamaño relativo:")
    size_counts = {}
    for meta in all_crops_metadata:
        size = meta['relative_size']
        size_counts[size] = size_counts.get(size, 0) + 1
    
    for size, count in sorted(size_counts.items()):
        print(f"  {size}: {count} crops")
    
    return all_crops_metadata

if __name__ == "__main__":
    # Ajusta estas rutas según tu estructura
    DATASET_DIR = Path("data/raw/100_samples")
    OUTPUT_DIR = Path("data/processed")
    
    crops_metadata = process_dataset(DATASET_DIR, OUTPUT_DIR)