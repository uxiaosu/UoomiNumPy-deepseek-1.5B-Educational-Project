#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for UoomiNumPy deepseek Educational Project
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Core requirements (without optional dependencies)
core_requirements = [req for req in requirements if not any(opt in req for opt in ["torch", "safetensors", "pytest", "black", "flake8", "markdown"])]

setup(
    name="deepseek-numpy-model",
    version="1.0.0",
    author="uxiaosu (TensorLinx / 上海灵兮矩阵公司)",
    author_email="contact@tensorlinx.com",
    description="A pure NumPy implementation of the DeepSeek language model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uxiaosu/UoomiNumPy-deepseek-1.5B-Educational-Project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=core_requirements,
    extras_require={
        "conversion": ["safetensors>=0.3.0", "torch>=1.9.0"],
        "dev": ["pytest>=6.0.0", "pytest-cov>=2.10.0", "black>=21.0.0", "flake8>=3.8.0"],
        "docs": ["markdown>=3.3.0"],
        "all": ["safetensors>=0.3.0", "torch>=1.9.0", "pytest>=6.0.0", "pytest-cov>=2.10.0", "black>=21.0.0", "flake8>=3.8.0", "markdown>=3.3.0"]
    },
    entry_points={
        "console_scripts": [
            "deepseek-numpy=src.api:main",
            "deepseek-convert=examples.weight_conversion:main",
            "deepseek-chat=examples.chat_example:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["*.py"],
        "examples": ["*.py"],
        "": ["README.md", "requirements.txt", "LICENSE"]
    },
    keywords="deepseek, numpy, language model, nlp, text generation, ai, machine learning",
    project_urls={
        "Bug Reports": "https://github.com/uxiaosu/UoomiNumPy-deepseek-1.5B-Educational-Project/issues",
        "Source": "https://github.com/uxiaosu/UoomiNumPy-deepseek-1.5B-Educational-Project",
        "Documentation": "https://github.com/uxiaosu/UoomiNumPy-deepseek-1.5B-Educational-Project/blob/main/README.md",
    },
)