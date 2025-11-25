import os
import shutil
from pathlib import Path

# Definir las rutas
carpeta_original = "data/raw/dataset_split_2/"
carpeta_procesada = "../carpetas-datos/resultados_sam3_datasetsplit2/"
carpeta_destino = "data/raw/masked_dataset_split_2/"

# Crear la carpeta de destino si no existe
os.makedirs(carpeta_destino, exist_ok=True)

# Obtener todas las imágenes .jpg de la carpeta original
imagenes_originales = [f for f in os.listdir(carpeta_original) if f.endswith('.jpg')]

print(f"Encontradas {len(imagenes_originales)} imágenes en la carpeta original")

# Contador de éxitos
copiadas = 0
errores = 0

for imagen_original in imagenes_originales:
    # Nombre de la imagen procesada (con prefijo masked_)
    imagen_procesada = f"masked_{imagen_original}"
    
    # Rutas completas
    ruta_procesada = os.path.join(carpeta_procesada, imagen_procesada)
    ruta_destino_img = os.path.join(carpeta_destino, imagen_original)
    
    # Verificar si existe la imagen procesada
    if os.path.exists(ruta_procesada):
        # Copiar la imagen procesada con el nombre original
        shutil.copy2(ruta_procesada, ruta_destino_img)
        print(f"✓ Copiada: {imagen_original}")
        copiadas += 1
        
        # Extraer el identificador base (todo antes de "_imageDANO_original.jpg")
        identificador_base = imagen_original.replace('_imageDANO_original.jpg', '')
        
        # Buscar todos los .json que contengan el identificador base
        for archivo in os.listdir(carpeta_original):
            if archivo.startswith(identificador_base) and archivo.endswith('.json'):
                ruta_json_origen = os.path.join(carpeta_original, archivo)
                ruta_json_destino = os.path.join(carpeta_destino, archivo)
                shutil.copy2(ruta_json_origen, ruta_json_destino)
                print(f"  ✓ JSON copiado: {archivo}")
    else:
        print(f"✗ No encontrada imagen procesada para: {imagen_original}")
        errores += 1

print(f"\n{'='*60}")
print(f"Proceso completado:")
print(f"  - Imágenes copiadas: {copiadas}")
print(f"  - Errores/No encontradas: {errores}")
print(f"  - Carpeta destino: {carpeta_destino}")