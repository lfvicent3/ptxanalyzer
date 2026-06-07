from setuptools import setup, find_packages

setup(
    name="ptx_analyzer",
    version="1.0.0",
    description="Ferramenta de análise e visualização de código PTX para validação de algoritmos de ordenação e grid-stride loops.",
    author="Henrique, Lucas, Luiz",
    packages=find_packages(),
    install_requires=[
        "plotly",
        "ipywidgets",
    ],
    python_requires=">=3.7",
)