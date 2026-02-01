"""
Setup configuration for mmWave Radar Detection package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mmwave-radar-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="mmWave radar data parser and ML framework for object detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mmwave-radar-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.4.0", "seaborn>=0.11.0"],
        "dev": ["pytest>=6.2.0", "jupyter>=1.0.0"],
    },
)
