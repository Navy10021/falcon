from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ai-combat-optimization",
    version="2.0.0",
    description="Ontology-Driven GNN + Reinforcement Learning for Force Minimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Research Team",
    url="https://github.com/your-username/ai-combat-optimization",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "networkx>=3.1",
        "gymnasium>=0.29.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
    ],
    extras_require={
        "gnn": ["torch-geometric>=2.3.0"],
        "full": ["stable-baselines3>=2.0.0", "tensorboard>=2.13.0",
                 "seaborn>=0.12.0", "pandas>=2.0.0", "rich>=13.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="reinforcement-learning, graph-neural-network, military-simulation, "
             "combat-optimization, human-in-the-loop, explainable-ai",
)
