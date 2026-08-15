from setuptools import setup, find_packages

setup(
    name="ftrain",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "triton",
        "bitsandbytes",
        "accelerate",
        "safetensors",
        "numpy",
        "gradio",
        "pandas",
        "trl",
        "huggingface_hub"
    ],
    author="FTRAIN Engine Team",
    description="High-performance AI model training, data intelligence, and merging framework.",
)
