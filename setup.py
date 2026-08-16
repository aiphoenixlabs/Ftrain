from setuptools import setup, find_packages

setup(
    name="ftrain",
    version="9.0.1",
    packages=find_packages(),
    install_requires=[
        "torch", "transformers", "datasets", "triton", "bitsandbytes",
        "accelerate", "safetensors", "numpy", "gradio", "pandas",
        "trl", "huggingface_hub"
    ],
    author="FTRAIN Engine Team",
    description="High-performance AI model training, data intelligence, and cross-architecture merging framework.",
)
