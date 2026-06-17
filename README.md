# PTX Analyzer

Ferramenta de análise estática e visualização de kernels CUDA via código PTX.
Inspeciona instruções, registradores, branches, memória e fluxo de controle
sem precisar executar o kernel na GPU.

```
pip install git+https://github.com/lfvicent3/ptxanalyzer.git
```

> **Requer:** Python ≥ 3.7, CUDA Toolkit (nvcc), plotly, ipywidgets

---

## Índice

1. [Conceito](#conceito)
2. [Início rápido](#início-rápido)
3. [PTXAnalyzer](#ptxanalyzer)
4. [PTXComparator](#ptxcomparator)
5. [PTXSourceView](#ptxsourceview)
6. [Utilitários](#utilitários)
7. [Métricas do kernel](#métricas-do-kernel)
8. [Arquiteturas suportadas](#arquiteturas-suportadas)

---

## Conceito

O compilador NVCC transforma código `.cu` em PTX — uma linguagem assembly virtual
da NVIDIA que documenta cada instrução gerada para o kernel. O **PTX Analyzer**
parseia esse PTX e extrai:

- Contagem e classificação de instruções por categoria
- Registradores declarados (pressão de registradores, risco de spilling)
- Branches condicionais (`@%p bra`) que causam **divergência de warp**
- Acessos de memória: global, shared, local (spill)
- Grafo de Fluxo de Controle (CFG) completo a partir dos labels `$L__BB0_N`
- Mapeamento instrução PTX → linha `.cu` (com `-lineinfo`)

### Auto-lineinfo

Se o `.ptx` foi compilado **sem** `-lineinfo` (sem `.loc`), o analisador procura
automaticamente o `.cu` correspondente nas pastas vizinhas e recompila com
`-lineinfo`. Isso é transparente ao usuário:

```python
# Mesmo que ptx/bubble_sort.ptx não tenha .loc, o analisador
# encontra kernels/bubble_sort.cu e recompila automaticamente
a = PTXAnalyzer.from_file('ptx/bubble_sort.ptx')
```

Ordem de busca: `ptx_dir/` → `../kernels/` → `../` → `CWD/` → `CWD/kernels/`

---

## Início rápido

```python
from ptx_analyzer import PTXAnalyzer, PTXComparator

# Carregar kernel
a = PTXAnalyzer.from_file('ptx/bubble_sort.ptx')   # auto-recompila com -lineinfo
a = PTXAnalyzer.from_file('kernels/bubble_sort.cu') # compila direto do .cu

# Texto
a.show_stats()           # métricas numéricas
a.show_branch_tree()     # árvore de branches com código .cu

# Gráficos
a.plot_mix()             # mix de instruções por categoria
a.plot_branch_cfg()      # CFG interativo
a.plot_decision_tree()   # fluxo simplificado — só blocos de decisão
a.plot_gpu_efficiency()  # dashboard de eficiência de GPU

# Interface completa
a.show()                 # 4 abas: métricas · gráficos · instruções · heurísticas

# Comparar vários kernels
comp = PTXComparator()
for s in ['bubble_sort', 'quick_sort', 'merge_sort']:
    comp.add(s, PTXAnalyzer.from_file(f'ptx/{s}.ptx'))
comp.show_table()
comp.plot_radar()
```

---

## PTXAnalyzer

Analisa um único kernel PTX.

### Carregamento

| Método | Descrição |
|---|---|
| `PTXAnalyzer.from_file(path, kernel_index=0, arch='sm_75')` | Carrega de arquivo `.ptx` ou `.cu`. Auto-recompila com `-lineinfo` se necessário |
| `PTXAnalyzer.from_string(ptx_code, kernel_index=0)` | Carrega de string PTX |
| `PTXAnalyzer.from_upload()` | Upload interativo (Jupyter/Colab) |

**`kernel_index`** — índice do kernel dentro do arquivo (0 = primeiro). Útil quando
um `.ptx` contém múltiplos kernels.

### Texto e terminal

| Método | Descrição |
|---|---|
| `show_stats()` | Métricas completas: instruções, registradores, branches (total/cond/incond), setp, branch ratio, memória, FMA, shfl |
| `show_warnings()` | Apenas os alertas de desempenho gerados pelas heurísticas |
| `summary()` | Resumo completo: métricas + mix de instruções + diagnóstico |
| `show_top_opcodes(n=10)` | Top N opcodes mais frequentes com barra ASCII |
| `show_roofline_text(peak_flops, peak_bw)` | Posição no Roofline em texto (memory-bound vs compute-bound) |
| `show_branch_tree()` | Árvore de desvios: bloco básico + `setp`/`bra` + linha `.cu` correspondente |
| `show_instructions(filter_cat, search)` | Tabela HTML de instruções com filtro por categoria e busca por opcode |

Todos os métodos de texto renderizam como `<pre>` HTML com fonte monoespaçada
no Jupyter/Colab. Fora do Jupyter fazem fallback para `print()`.

### Gráficos interativos (Plotly)

| Método | Descrição |
|---|---|
| `plot_distribution()` | Pizza: proporção de cada categoria de instrução |
| `plot_categories()` | Barras: contagem por categoria |
| `plot_timeline(max_points=400)` | Linha do tempo: categoria de cada instrução em sequência |
| `plot_registers()` | Tipos de registradores declarados (`pred`, `b32`, `f32`, …) |
| `plot_mix()` | Stacked bar do mix de instruções |
| `plot_memory_breakdown()` | Breakdown de acessos: global / shared / local |
| `plot_roofline(peak_flops, peak_bw)` | Posição do kernel no modelo Roofline |

#### CFG — Grafo de Fluxo de Controle

```python
a.plot_branch_cfg(max_blocks=30)
```

Cada retângulo é um bloco básico. Arestas coloridas:
- 🟠 **Laranja** — `@%p bra` (condicional) → divergência de warp possível
- 🔵 **Azul** — `bra.uni` (incondicional) → sem divergência
- ⚫ **Cinza** — fall-through (execução sequencial)

Layout usa **longest-path (algoritmo de Kahn)** para posicionar join-points
abaixo de todos os predecessores. Back-edges (loops) roteados pela parede
direita com linha tracejada — convenção estilo IDA Pro.

Passe o mouse sobre um bloco para ver o código `.cu` do branch terminador.

#### Árvore de Decisão

```python
a.plot_decision_tree(max_decisions=20)
```

Visão simplificada do fluxo: exibe **apenas** os blocos com branch condicional
(`@%p bra`). Blocos sequenciais são omitidos — a contagem de instruções
intermediárias aparece na aresta. Cada nó mostra a condição do código `.cu`.

| Elemento | Significado |
|---|---|
| Aresta **"sim"** | Predicado verdadeiro |
| Aresta **"não"** | Fall-through |
| Número na aresta | Instruções omitidas entre dois pontos de decisão |
| Aresta **↩ loop** tracejada | Back-edge — retorno a bloco anterior |

#### Dashboard de Eficiência de GPU

```python
a.plot_gpu_efficiency(arch='sm_86', threads_per_block=256)
```

6 gauges com análise estática de utilização da GPU:

| Gauge | Fórmula / Fonte |
|---|---|
| **Ocupância** | `⌊(regs/SM ÷ regs/thread) ÷ 32⌋ × 32 ÷ max_threads/SM` |
| **Risco de Divergência** | `min(branch_ratio × 400, 100)%` — quanto maior, pior |
| **Cobertura de Grid** | 100% se `%nctaid` detectado no PTX, 0% caso contrário |
| **Posição Roofline** | `min(AI ÷ ridge_point, 1) × 100%` |
| **Eficiência de Warp** | `100% − risco_divergência` |
| **Score Geral** | Média das 4 métricas principais |

Inclui painel de recomendações gerado automaticamente com ações específicas.

> Análise **puramente estática** — para métricas de runtime use `nvprof` /
> Nsight Compute.

### Interface interativa

```python
a.show()
```

Interface ipywidgets com 4 abas:
- **Visão Geral** — métricas + botões para abrir gráficos
- **Instruções** — tabela pesquisável por categoria e opcode
- **Registradores** — tipos e contagem
- **Diagnóstico** — heurísticas com ícones de severidade

### Exportação

```python
d = a.to_dict()    # dict com todas as métricas
j = a.to_json()    # JSON formatado
```

---

## PTXComparator

Compara múltiplos kernels lado a lado.

```python
from ptx_analyzer import PTXComparator

comp = PTXComparator()
comp.add('bubble_sort',    a_bubble)
comp.add('insertion_sort', a_insertion)
comp.add('quick_sort',     a_quick)
```

| Método | Descrição |
|---|---|
| `show_table()` | Tabela HTML colorida com todas as métricas (melhor valor destacado em verde) |
| `summary()` | Texto: melhor e pior kernel em cada métrica |
| `plot_comparison(*metrics)` | Barras agrupadas para as métricas escolhidas |
| `plot_radar(normalize=True)` | Radar chart normalizado — perfil de cada kernel |
| `plot_mix()` | Mix de instruções empilhado para todos os kernels |
| `plot_memory_breakdown()` | Breakdown de memória comparativo |
| `plot_roofline(peak_flops, peak_bw)` | Roofline com todos os kernels sobrepostos |
| `generate_report_table()` | Tabela Markdown pronta para copiar num relatório |

```python
# Exemplo: comparar métricas específicas
comp.plot_comparison('Instruções', 'Branches', 'ld.global', 'shl/shr')

# Tabela para o relatório
print(comp.generate_report_table())
```

---

## PTXSourceView

Mapeamento lado a lado entre código `.cu` e as instruções PTX geradas.
Requer PTX compilado com `-lineinfo` (ou usa auto-recompile automático).

### Carregamento

```python
from ptx_analyzer import PTXSourceView

# A partir de arquivos separados
view = PTXSourceView.from_files(
    'kernels/bubble_sort.cu',
    'ptx/bubble_sort_li.ptx',
    kernel_index=0
)

# A partir de um .cu — compila automaticamente com -lineinfo
view = PTXSourceView.from_file('kernels/bubble_sort.cu', kernel_index=0, arch='sm_86')

# A partir de um PTXAnalyzer já carregado
view = PTXSourceView.from_analyzer('kernels/bubble_sort.cu', analyzer)
```

### Métodos

| Método | Descrição |
|---|---|
| `show_stats()` | Estatísticas: total de linhas, mapeadas, eliminadas (✂), FMA (⚡), pesadas (📦) |
| `show_text(show_only_mapped=False)` | Mapeamento `.cu` ↔ PTX em texto/ASCII |
| `show_html(show_only_mapped=False)` | Mesmo mapeamento em HTML puro |
| `show()` | Interface ipywidgets com toggle "só linhas mapeadas" e badges coloridos |

### Badges de linha

| Badge | Significado |
|---|---|
| `✂ ELIMINADA` | Linha removida pelo compilador (dead code, constant folding) |
| `⚡ FMA` | `mul + add` fundidos numa instrução `fma.rn.f32` |
| `🔮 Tensor Core` | Instrução `mma.sync` — Tensor Core ativo |
| `📦 PESADA` | ≥ 6 instruções PTX geradas por esta linha de `.cu` |

```python
view.show_stats()
# ══════════════════════════════════════════════
#   _Z11bubble_stepPiii
# ──────────────────────────────────────────────
#   Linhas no .cu              : 92
#   Linhas com PTX mapeado     : 6
#   Linhas eliminadas (✂)      : 1  ← dead code / const fold
#   Linhas com FMA (⚡)         : 0
#   Linhas pesadas ≥6 PTX (📦) : 2
# ══════════════════════════════════════════════
```

---

## Utilitários

```python
from ptx_analyzer import compile_to_ptx, analyze_all_ptx, run_heuristics, build_cfg
```

### `compile_to_ptx`

```python
compile_to_ptx('kernels/bubble_sort.cu', 'ptx/bubble_sort.ptx',
               arch='sm_86', lineinfo=True)
```

Wrapper em torno de `nvcc -ptx`. Imprime a saída do `ptxas` (registradores,
shared memory, spill) quando `lineinfo=True`.

### `analyze_all_ptx`

```python
comp = analyze_all_ptx('ptx/')   # retorna PTXComparator com todos os .ptx da pasta
comp.show_table()
```

### `run_heuristics`

```python
for level, msg in run_heuristics(analyzer.kernel):
    print(f'{level}: {msg}')
```

Retorna lista de `(level, msg)` onde `level` é `"info"`, `"warn"` ou `"error"`.

| Verificação | Condição de alerta |
|---|---|
| Divergência de warp | > 10 branches condicionais ou branch ratio > 10% |
| Pressão de registradores | > 128 registradores → risco de spilling |
| Shared memory | ausente (aviso) ou presente (positivo) |
| Memory-bound | intensidade aritmética < 1.0 |
| Operações atômicas | qualquer `atom` detectado |
| FMA ratio | mul+add separados → sugere `--use_fast_math` |
| Grid-stride | ausência de `%nctaid` → kernel não escala além de 1 grid |
| Vector loads | `ld.v4` detectado → positivo (bom uso de largura de banda) |
| Warp shuffles | `shfl.sync` detectado → positivo (redução eficiente) |
| Local memory (spill) | qualquer `ld.local` / `st.local` |

### `build_cfg`

```python
blocks, bfs_order = build_cfg(analyzer.kernel)
# blocks: dict[label → BasicBlock]
# bfs_order: lista de labels em BFS a partir da entrada
```

`BasicBlock` tem:
- `instructions` — lista de `PTXInstruction`
- `exits` — lista de `(tipo, label_destino)` onde tipo é `"conditional"`, `"jump"` ou `"fallthrough"`
- `is_entry`, `is_terminal` — booleanos

---

## Métricas do kernel

Acessíveis via `analyzer.kernel.<atributo>`:

| Atributo | Tipo | Descrição |
|---|---|---|
| `name` | `str` | Nome do kernel (símbolo mangled do C++) |
| `total_instructions` | `int` | Total de instruções no kernel |
| `total_registers` | `int` | Registradores declarados (proxy para regs/thread) |
| `total_branches` | `int` | Branches totais (`bra` + `bra.uni`) |
| `predicated_branches` | `int` | Branches condicionais (`@%p bra`) |
| `unconditional_branches` | `int` | Branches incondicionais (`bra.uni`) |
| `branch_ratio` | `float` | `predicated_branches / total_instructions` |
| `setp_count` | `int` | Instruções `setp` (comparações) |
| `global_loads` | `int` | Instruções `ld.global` |
| `global_stores` | `int` | Instruções `st.global` |
| `shared_accesses` | `int` | Acessos `ld.shared` + `st.shared` |
| `local_accesses` | `int` | Acessos `ld.local` + `st.local` (register spill) |
| `fma_count` | `int` | Instruções `fma` (mul+add fundidos) |
| `shfl_count` | `int` | Instruções `shfl.sync` (warp shuffle) |
| `atomics` | `int` | Operações atômicas (`atom`) |
| `bit_ops_count` | `int` | Operações `shl` / `shr` |
| `arithmetic_intensity` | `float` | Instruções aritméticas / instruções de memória |
| `uses_nctaid` | `bool` | Usa `%nctaid` (padrão grid-stride) |
| `is_register_only` | `bool` | Zero loads/stores (loop completamente desenrolado) |
| `param_count` | `int` | Número de parâmetros do kernel |
| `file_map` | `dict[int, str]` | Mapeamento índice → caminho `.cu` (via `-lineinfo`) |
| `category_counts` | `dict[str, int]` | Contagem de instruções por categoria |

### Categorias de instrução

| Categoria | Opcodes típicos |
|---|---|
| `arithmetic` | `add`, `sub`, `mul`, `mad`, `fma`, `div`, `neg` |
| `comparison` | `setp`, `set` |
| `logic` | `and`, `or`, `xor`, `not`, `shl`, `shr` |
| `memory` | `ld`, `st`, `prefetch`, `isspacep` |
| `control` | `bra`, `ret`, `exit`, `call`, `brx` |
| `mov_conv` | `mov`, `cvt`, `cvta`, `selp` |
| `special` | `shfl.sync`, `vote`, `bar`, `atom`, `red` |
| `other` | demais |

---

## Arquiteturas suportadas

Parâmetros usados em `plot_gpu_efficiency()`:

| Arquitetura | GPU exemplo | Max threads/SM | Peak TFLOPS | Peak BW |
|---|---|---|---|---|
| `sm_75` | RTX 2080 Ti | 1024 | 14.1 | 448 GB/s |
| `sm_80` | A100 | 2048 | 19.5 | 2000 GB/s |
| `sm_86` | RTX 3050 / 3060 | 1536 | 8.1 | 192 GB/s |
| `sm_89` | RTX 4090 (Ada) | 1536 | 49.7 | 576 GB/s |
| `sm_90` | H100 | 2048 | 133.8 | 3350 GB/s |

---

## Google Colab

```python
# Instalar (--upgrade não quebra o ipython do Colab)
!pip install -q --upgrade git+https://github.com/lfvicent3/ptxanalyzer.git

# Detectar arquitetura da GPU
import subprocess
cap = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                    capture_output=True, text=True).stdout.strip().replace('.', '')
ARCH = f'sm_{cap}' if cap else 'sm_86'

# Compilar kernels
import os
os.makedirs('ptx', exist_ok=True)
!nvcc -ptx kernels/bubble_sort.cu -arch={ARCH} -o ptx/bubble_sort.ptx

# Analisar
from ptx_analyzer import PTXAnalyzer
a = PTXAnalyzer.from_file('ptx/bubble_sort.ptx')  # auto-recompila com -lineinfo
a.show()
```

Todos os gráficos usam `go.FigureWidget` internamente para compatibilidade com Colab.

---

## Autores

Henrique · Lucas · Luiz — Universidade Federal de Viçosa (UFV)
