from setuptools import setup, find_packages

setup(
    name="rag-multimodal",
    version="0.1.0",
    description="Sistema RAG Multimodal para Detección de Defectos en Vehículos",
    author="Tu Nombre",
    author_email="tu.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "PyYAML>=6.0",
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
    ],
)
