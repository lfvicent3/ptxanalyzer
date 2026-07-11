from setuptools import setup, find_packages

setup(
    name="ptx_analyzer",
    version="1.2.0",
    description="Ferramenta enxuta de análise PTX com métricas do ptxas, Mermaid e integração com benchmark.",
    author="Henrique, Lucas, Luiz",
    packages=find_packages(),
    package_data={"ptx_analyzer": ["vendor/*.js", "vendor/*.md"]},
    include_package_data=True,
    install_requires=[],
    python_requires=">=3.7",
)
