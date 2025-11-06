from src.core.api.ollama_client import OllamaVLMClient

client = OllamaVLMClient(model='qwen3-vl:4b')

test_image = "data/raw/train_test_split/test/zona1_ok_2_3_1559568584233_zona_9_imageDANO_original.jpg"

# Ahora debería funcionar
response = client.generate("¿Qué ves?", image_path=test_image, max_tokens=50)
print(f"Respuesta: '{response}'")

# O con ambos campos
# content, thinking = client.generate("¿Qué ves?", image_path=test_image, return_thinking=True)
# print(f"Content: '{content}'")
# print(f"Thinking: '{thinking}'")
