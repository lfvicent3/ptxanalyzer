# PTX Analyzer

Versão enxuta do analisador de PTX para o projeto de GPU.

O foco agora é:

- extrair métricas do próprio PTX
- extrair métricas reais do `ptxas`
- gerar fluxo de controle em Mermaid para Colab
- gerar fluxo de controle em Mermaid com descrições humanas por bloco
- ligar kernels PTX/CUDA aos resultados de benchmark

## Uso rápido

```python
from ptx_analyzer import PTXAnalyzer

a = PTXAnalyzer.from_file("kernels/bubble_sort_all.cu")
a.attach_benchmark_csv("resultados.csv")

a.summary()
a.report(section="ptxas")
a.report(section="benchmark")
a.hotspots_report()
a.flowchart(mode="html")
```

## Validação rápida do parser

Para depurar o CFG sem o ruído dos algoritmos de ordenação, use o microkernel
`kernels/cfg_ifelse_smoke.cu`, que contém apenas um `if/else` com soma e
subtração:

```bash
python3 scripts/debug_cfg_smoke.py
```

Esse script compila o `.cu`, roda o analisador e imprime o Mermaid junto com os
blocos anotados em linguagem humana.

## API principal

- `PTXAnalyzer.from_file(path, kernel_index=0, arch="sm_75")`
- `summary()`
- `show_stats()`
- `report(section="summary" | "stats" | "ptxas" | "hotspots" | "benchmark")`
- `hotspots_report(mode="text" | "data")`
- `flowchart(mode="html" | "text" | "raw" | "data")`
- `control_flow(mode="html" | "text" | "raw" | "data")`
- `attach_benchmark_csv(csv_path)`
- `attach_benchmark_output(output_text)`
- `benchmark_rows()`
- `profile_runtime(...)`
- `to_dict()`
- `to_json()`

## O que saiu

Esta versão removeu:

- heurísticas automáticas
- dashboards e gráficos Plotly
- interface `ipywidgets`
- source viewer
- modos antigos de CFG/BRA fora do Mermaid

## Métricas do ptxas

Quando o analisador recebe um `.cu`, ele:

1. compila para `.ptx` com `-lineinfo`
2. compila para `.cubin` com `--ptxas-options=-v`
3. associa ao kernel:

- registradores reais
- `smem`
- `cmem`
- `lmem`
- `stack frame`
- `spill stores`
- `spill loads`

## Benchmark

O analisador entende CSV no formato:

```csv
algorithm,strategy,segment_size,time_ms,validation,baseline_delta_percent
```

Isso permite ligar cada kernel às execuções do benchmark e resumir:

- tempo por segmento
- validação
- diferença para o baseline

## Comparator

```python
from ptx_analyzer import compare_kernels_in_ptx_file

comp = compare_kernels_in_ptx_file("ptx/bubble_sort.ptx")
comp.summary()
print(comp.generate_report_table())
```
