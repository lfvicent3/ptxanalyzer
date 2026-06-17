"""
Classe principal PTXAnalyzer.
"""

import json
from collections import Counter
from .core import PTXKernel, CATEGORIES
from .parser import parse_ptx
from .heuristics import run_heuristics, LEVEL_ICONS

# ──────────────────────────────────────────────────────────────────────────────
# Helpers de saída em texto
# ──────────────────────────────────────────────────────────────────────────────

def _bar(n: int, max_n: int, width: int = 32) -> str:
    """Barra ASCII proporcional (█ preenchido, ░ vazio)."""
    if max_n == 0:
        return "░" * width
    filled = round(n / max_n * width)
    return "█" * filled + "░" * (width - filled)


def _section(title: str, total_w: int = 60) -> str:
    rest = total_w - len(title) - 4
    return f"── {title} {'─' * max(0, rest)}"
from .core import build_cfg
from .visuals import (
    plot_category_pie, plot_category_bar, plot_instruction_timeline,
    plot_register_types, plot_instruction_mix_stacked, plot_memory_access_breakdown,
    plot_roofline, plot_branch_cfg as _plot_branch_cfg,
)
from .html_render import (
    _render_overview_html, _render_instructions_html,
    _render_registers_html, _render_diagnostics_html
)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Classe principal PTXAnalyzer
# ──────────────────────────────────────────────────────────────────────────────

class PTXAnalyzer:
    """
    Interface principal de análise de um kernel PTX.

    Saída em texto (sem dependências de UI):
        a.summary()              # resumo completo: métricas, mix, memória, diagnóstico
        a.show_stats()           # tabela de métricas com barras ASCII
        a.show_warnings()        # diagnósticos heurísticos
        a.show_top_opcodes(n)    # top-N opcodes mais frequentes
        a.show_roofline_text()   # posição no modelo Roofline (ASCII)
        a.show_branch_tree()     # árvore de desvios (CFG em ASCII)

    Visualizações gráficas (requerem plotly/Jupyter):
        a.show()                 # interface ipywidgets com 4 abas
        a.plot_distribution()    # pie chart
        a.plot_categories()      # bar chart por categoria
        a.plot_timeline()        # sequência de instruções
        a.plot_roofline()        # modelo Roofline interativo
        a.plot_branch_cfg()      # grafo de fluxo de controle (CFG)

    Exportação:
        a.to_dict()              # dict com todas as métricas
        a.to_json()              # JSON string
    """

    def __init__(self, code: str, kernel_index: int = 0):
        """
        Args:
            code:         Conteúdo do arquivo .ptx.
            kernel_index: Índice do kernel a analisar se o arquivo tiver vários.
        """
        self._code = code
        kernels = parse_ptx(code)
        if not kernels:
            raise ValueError("Nenhum kernel encontrado no PTX fornecido.")
        if kernel_index >= len(kernels):
            kernel_index = 0
        self.kernel = kernels[kernel_index]
        self._all_kernels = kernels

    # ── construtores alternativos ────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: str, kernel_index: int = 0, arch: str = "sm_75") -> "PTXAnalyzer":
        import subprocess

        if path.endswith(".cu"):
            out_ptx = path.replace(".cu", ".ptx")
            # --ptxas-options=-v → ptxas imprime registradores, smem e spill no stderr
            cmd = [
                "nvcc", "-ptx", "-lineinfo",
                "--ptxas-options=-v",
                path, f"-arch={arch}", "-o", out_ptx,
            ]
            print(f"Compilando CUDA para PTX: {' '.join(cmd)}")
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Erro ao compilar {path}:\n{res.stderr}")

            # Exibir info do ptxas (vai para stderr mesmo com sucesso)
            ptxas_info = [ln for ln in res.stderr.splitlines()
                          if ln.strip().startswith("ptxas")]
            if ptxas_info:
                print("\n── Informações ptxas (--ptxas-options=-v) ──")
                for ln in ptxas_info:
                    print(" ", ln)
                print()

            path_to_read = out_ptx
        else:
            path_to_read = path

        with open(path_to_read, "r", encoding="utf-8", errors="replace") as f:
            return cls(f.read(), kernel_index)

    @classmethod
    def from_string(cls, ptx_code: str, kernel_index: int = 0) -> "PTXAnalyzer":
        return cls(ptx_code, kernel_index)

    @classmethod
    def from_upload(cls) -> "PTXAnalyzer":
        """Abre diálogo de upload no Colab e carrega o arquivo .ptx."""
        try:
            from google.colab import files
        except ImportError:
            raise RuntimeError("from_upload() só funciona no Google Colab.")
        uploaded = files.upload()
        if not uploaded:
            raise ValueError("Nenhum arquivo enviado.")
        name = next(iter(uploaded))
        code = uploaded[name].decode("utf-8", errors="replace")
        return cls(code)

    # ── interface de texto ───────────────────────────────────────────────────

    def show_stats(self):
        """Imprime métricas resumidas no terminal/célula."""
        k = self.kernel
        print(f"\n{'='*56}")
        print(f"  Kernel: {k.name}")
        print(f"{'='*56}")
        print(f"  Total de instruções : {k.total_instructions}")
        print(f"  Total de registros  : {k.total_registers}")
        print(f"  ld.global           : {k.global_loads}")
        print(f"  st.global           : {k.global_stores}")
        print(f"  Shared memory       : {k.shared_accesses}")
        print(f"  Branches predicados : {k.predicated_branches}")
        print(f"  Operações atômicas  : {k.atomics}")
        print(f"  FMA                 : {k.fma_count}")
        print(f"  shfl.sync           : {k.shfl_count}")
        print(f"  Int. aritmética     : {k.arithmetic_intensity}")
        print(f"\n  Contagem por categoria:")
        for cat, n in sorted(k.category_counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(n, 40)
            print(f"    {cat:<14} {n:>5}  {bar}")
        print()

    def show_warnings(self):
        """Imprime diagnósticos no terminal."""
        for level, msg in run_heuristics(self.kernel):
            icon = LEVEL_ICONS.get(level, "•")
            print(f"  {icon}  {msg}")

    def summary(self):
        """
        Resumo completo em texto: métricas, mix de instruções, memória e diagnóstico.
        Equivalente text-only ao show() interativo — não requer Jupyter nem plotly.
        """
        k = self.kernel
        W = 58

        title = f"PTX Kernel: {k.name}  ({k.param_count} parâm.)"
        print("╔" + "═" * W + "╗")
        print(f"║  {title:<{W - 2}}║")
        print("╚" + "═" * W + "╝")

        # ── Métricas em 2 colunas
        print(f"\n{_section('Métricas', W + 2)}")
        pairs = [
            ("Instruções totais",    k.total_instructions,
             "Registradores",        k.total_registers),
            ("ld.global",            k.global_loads,
             "st.global",            k.global_stores),
            ("ld/st.local (spill)",  k.local_accesses,
             "Shared memory",        k.shared_accesses),
            ("Branches total",       k.total_branches,
             "Branch ratio",         f"{k.branch_ratio:.1%}"),
            ("Cond. (@%p bra)",      k.predicated_branches,
             "Incond. (.uni)",        k.unconditional_branches),
            ("setp (predicados)",    k.setp_count,
             "Blocos básicos",       len(build_cfg(k)[1])),
            ("FMA",                  k.fma_count,
             "shfl.sync",            k.shfl_count),
            ("Int. aritmética",      k.arithmetic_intensity,
             "Só registros",         "Sim" if k.is_register_only else "Não"),
        ]
        for l1, v1, l2, v2 in pairs:
            print(f"  {l1:<22}: {str(v1):>6}    {l2:<18}: {str(v2):>6}")

        # ── Mix de instruções com barras
        print(f"\n{_section('Mix de Instruções', W + 2)}")
        mix = sorted(k.category_counts.items(), key=lambda x: -x[1])
        max_c = max((n for _, n in mix), default=1)
        total = max(k.total_instructions, 1)
        for cat, n in mix:
            pct = n / total * 100
            bar = _bar(n, max_c, 32)
            print(f"  {cat:<14} {pct:5.1f}%  {bar}  {n:>4}")

        # ── Breakdown de memória
        print(f"\n{_section('Memória', W + 2)}")
        mem_items = [
            ("Global loads",  k.global_loads),
            ("Global stores", k.global_stores),
            ("Shared",        k.shared_accesses),
            ("Local (spill)", k.local_accesses),
        ]
        max_mem = max((v for _, v in mem_items), default=1)
        for label, n in mem_items:
            bar = _bar(n, max_mem, 30)
            print(f"  {label:<16}: {n:>5}  {bar}")

        # ── Diagnósticos heurísticos
        print(f"\n{_section('Diagnóstico', W + 2)}")
        for level, msg in run_heuristics(k):
            icon = LEVEL_ICONS.get(level, "•")
            print(f"  {icon}  {msg}")
        print()

    def show_top_opcodes(self, n: int = 10):
        """
        Imprime os N opcodes mais frequentes do kernel com barra ASCII.
        Útil para identificar os hotspots de instrução sem abrir um gráfico.
        """
        k = self.kernel
        counts = Counter(i.op for i in k.instructions)
        top = counts.most_common(n)
        total = max(k.total_instructions, 1)
        max_c = top[0][1] if top else 1
        pad = max((len(op) for op, _ in top), default=20)
        pad = max(pad, 20)

        print(f"\n{_section(f'Top {n} Opcodes — {k.name}', 60)}")
        print(f"  {'#':>3}  {'Opcode':<{pad}}  {'Qtd':>5}  {'%':>5}  Barra")
        print(f"  {'─'*3}  {'─'*pad}  {'─'*5}  {'─'*5}  {'─'*26}")
        for rank, (op, count) in enumerate(top, 1):
            pct = count / total * 100
            bar = _bar(count, max_c, 26)
            print(f"  {rank:>3}  {op:<{pad}}  {count:>5}  {pct:5.1f}%  {bar}")
        print()

    def show_roofline_text(self, peak_flops: float = 15.7, peak_bw: float = 900.0):
        """
        Posição estimada do kernel no modelo Roofline (análise estática).
        A escala usa arith_instrs/mem_instrs como proxy — não FLOPs/Byte reais.
        Para análise precisa, use nvprof / Nsight Compute.
        """
        k = self.kernel
        ai = k.arithmetic_intensity

        if ai < 1.0:
            classif, icon = "MEMORY-BOUND", "⚠️ "
        elif ai < 2.0:
            classif, icon = "LIMÍTROFE", "ℹ️ "
        else:
            classif, icon = "COMPUTE-BOUND", "✅"

        W = 58
        print(f"\n{_section('Roofline — estimativa estática', W + 2)}")
        print(f"  Hardware (padrão):  {peak_flops} TFLOP/s  |  {peak_bw} GB/s BW")
        print(f"\n  Kernel: {k.name}")
        print(f"    Int. aritmética  : {ai}  (arith_instrs / mem_instrs)")
        print(f"    Classificação    : {icon}  {classif}")
        print()

        # Escala 0..4+, barras de 36 chars
        bar_w = 36
        scale_max = 4.0
        pos = min(int(ai / scale_max * bar_w), bar_w - 1)
        p1 = int(1.0 / scale_max * bar_w)   # = 9
        p2 = int(2.0 / scale_max * bar_w)   # = 18

        line = list("─" * bar_w)
        for p in [p1, p2]:
            line[p] = "┼"
        line[pos] = "╪" if line[pos] == "┼" else "▲"
        line_str = "".join(line)

        prefix = "  [MEMÓRIA] "
        print(f"{prefix}{line_str} [CÁLCULO]")

        # Label numérica alinhada com os ticks
        label = (
            "0.0"
            + " " * (p1 - 3)
            + "1.0"
            + " " * (p2 - p1 - 3)
            + "2.0"
            + " " * (p1 - 3)
            + "3.0"
            + " " * (p1 - 3)
            + "4.0+"
        )
        print(" " * len(prefix) + label)

        # Anotação de regiões
        annot = " " * p1 + "└memory " + " " * (p2 - p1 - 8) + "└compute"
        print(" " * len(prefix) + annot)
        print()
        print(f"  Nota: escala em (arith_instrs / mem_instrs), não FLOPs/Byte.")
        print(f"        Para análise precisa use nvprof / Nsight Compute.")
        print()

    def show_instructions(self, filter_cat: str = "all", search: str = ""):
        """Exibe tabela de instruções filtrada."""
        from IPython.display import display
        import ipywidgets as w
        display(w.HTML(
            value='<div style="background:#0f1416;padding:12px;border-radius:8px;">'
            + _render_instructions_html(self.kernel, filter_cat, search)
            + '</div>'
        ))

    # ── gráficos ─────────────────────────────────────────────────────────────

    def plot_distribution(self):
        plot_category_pie(self.kernel)

    def plot_categories(self):
        plot_category_bar(self.kernel)

    def plot_timeline(self, max_points: int = 400):
        plot_instruction_timeline(self.kernel, max_points)

    def plot_registers(self):
        plot_register_types(self.kernel)

    def plot_mix(self):
        """Stacked bar do mix de instruções (% por categoria)."""
        plot_instruction_mix_stacked({self.kernel.name: self.kernel})

    def plot_memory_breakdown(self):
        """Breakdown de acessos: global / shared / local."""
        plot_memory_access_breakdown({self.kernel.name: self.kernel})

    def plot_roofline(self, peak_flops: float = 15.7, peak_bw: float = 900.0):
        """Posição do kernel no modelo Roofline."""
        plot_roofline({self.kernel.name: self.kernel}, peak_flops, peak_bw)

    # ── interface ipywidgets ─────────────────────────────────────────────────

    def show(self):
        """
        Exibe interface interativa com 4 abas:
          0 — Visão Geral (métricas + gráficos)
          1 — Instruções  (tabela pesquisável)
          2 — Registradores
          3 — Diagnóstico
        """
        try:
            import ipywidgets as w
            from IPython.display import HTML, display
        except ImportError:
            print("ipywidgets não disponível. Use show_stats() / show_warnings().")
            self.show_stats()
            self.show_warnings()
            return

        k = self.kernel
        style = ('<style>'
                 'body,div{font-family:ui-monospace,monospace;}'
                 '.ptx-tab-content{background:#0f1416;padding:16px;border-radius:8px;}'
                 '</style>')

        # ── Aba 0: Visão Geral ──────────────────────────────────────────────
        # w.HTML aceita string e é um Widget válido para VBox/HBox
        overview_html = w.HTML(
            value=style
            + '<div class="ptx-tab-content">'
            + f'<h3 style="color:#f1f5f9;margin:0 0 12px;">Kernel: '
            + f'<code style="color:#38bdf8;">{k.name}</code></h3>'
            + _render_overview_html(k)
            + '</div>'
        )
        btn_pie  = w.Button(description="📊 Pizza", button_style="info")
        btn_bar  = w.Button(description="📈 Barras", button_style="info")
        btn_time = w.Button(description="📉 Timeline", button_style="info")
        btn_reg  = w.Button(description="🗂️ Registradores", button_style="info")

        out_charts = w.Output()

        def _make_handler(plot_fn):
            def _handler(_):
                with out_charts:
                    out_charts.clear_output(wait=True)
                    fig = plot_fn()
                    if fig:
                        import plotly.graph_objects as go
                        from IPython.display import display
                        display(go.FigureWidget(fig))
            return _handler

        btn_pie.on_click(_make_handler(self.plot_distribution))
        btn_bar.on_click(_make_handler(self.plot_categories))
        btn_time.on_click(_make_handler(self.plot_timeline))
        btn_reg.on_click(_make_handler(self.plot_registers))

        chart_btns = w.HBox([btn_pie, btn_bar, btn_time, btn_reg])
        tab0 = w.VBox([overview_html, chart_btns, out_charts])

        # ── Aba 1: Instruções ───────────────────────────────────────────────
        search_box = w.Text(placeholder="Buscar opcode ou operando…",
                            layout=w.Layout(width="300px"))
        cat_opts   = ["all"] + sorted(CATEGORIES.keys()) + ["other"]
        cat_dd     = w.Dropdown(options=cat_opts, value="all",
                                layout=w.Layout(width="180px"))

        def _instr_html_val(cat, search):
            return ('<div style="background:#0f1416;padding:12px;border-radius:8px;">'
                    + _render_instructions_html(k, cat, search) + '</div>')

        instr_html_w = w.HTML(value=_instr_html_val("all", ""))

        def _refresh_instrs(*_):
            instr_html_w.value = _instr_html_val(cat_dd.value, search_box.value)

        search_box.observe(_refresh_instrs, names="value")
        cat_dd.observe(_refresh_instrs, names="value")
        tab1 = w.VBox([w.HBox([search_box, cat_dd]), instr_html_w])

        # ── Aba 2: Registradores ────────────────────────────────────────────
        tab2 = w.HTML(value=(
            '<div style="background:#0f1416;padding:16px;border-radius:8px;">'
            + _render_registers_html(k) + '</div>'
        ))

        # ── Aba 3: Diagnóstico ──────────────────────────────────────────────
        tab3 = w.HTML(value=(
            '<div style="background:#0f1416;padding:16px;border-radius:8px;">'
            '<h3 style="color:#f1f5f9;margin:0 0 12px;">Diagnóstico Automático</h3>'
            + _render_diagnostics_html(k) + '</div>'
        ))

        tabs = w.Tab(children=[tab0, tab1, tab2, tab3])
        for idx, title in enumerate(["📊 Visão Geral", "📋 Instruções",
                                      "🗂️ Registradores", "🔍 Diagnóstico"]):
            tabs.set_title(idx, title)

        display(tabs)

    # ── branches / CFG ──────────────────────────────────────────────────────

    def show_branch_tree(self):
        """
        Imprime a árvore de desvios do kernel em texto.

        Para cada bloco básico mostra o par (setp + bra) que causa o desvio
        e, quando o PTX foi compilado com -lineinfo, exibe o trecho do
        código-fonte .cu correspondente a cada branch.

        Legenda de tipos de aresta:
          →[cond]  desvio condicional (@%p bra) — pode causar divergência de warp
          →[jump]  salto incondicional (bra.uni)
          →[fall]  fall-through (execução sequencial)
        """
        import os

        k = self.kernel
        blocks, order = build_cfg(k)
        W = 64

        # ── carrega arquivos-fonte referenciados pelo PTX ─────────────────────
        src_cache: dict = {}   # file_idx → lista de linhas (ou None se ausente)
        for fidx, fpath in k.file_map.items():
            try:
                with open(fpath, encoding="utf-8", errors="replace") as fh:
                    src_cache[fidx] = fh.readlines()
            except OSError:
                src_cache[fidx] = None

        def _src_line(instr):
            """Retorna o conteúdo da linha .cu correspondente à instrução, ou None."""
            if instr.source_line <= 0:
                return None
            lines = src_cache.get(instr.source_file)
            if lines is None:
                return None
            idx = instr.source_line - 1
            if 0 <= idx < len(lines):
                return lines[idx].rstrip()
            return None

        # ── helpers ──────────────────────────────────────────────────────────

        def _find_setp(instrs, bra_idx):
            bra = instrs[bra_idx]
            if not bra.is_predicated or not bra.predicate:
                return None
            pred_reg = bra.predicate.lstrip("@").lstrip("!")
            for i in range(bra_idx - 1, -1, -1):
                ins = instrs[i]
                if ins.op_base == "setp" and ins.operands and ins.operands[0] == pred_reg:
                    return ins
            return None

        def _ptx_loc(instr):
            fname = os.path.basename(k.file_map.get(instr.source_file, ""))
            if instr.source_line > 0 and fname:
                return f"PTX:{instr.line_no}  {fname}:{instr.source_line}"
            return f"PTX:{instr.line_no}"

        def _short(raw, width=46):
            return raw.strip()[:width]

        def _print_branch_detail(detail, instr, label):
            """Imprime linha PTX + código-fonte correspondente."""
            src = _src_line(instr)
            loc = _ptx_loc(instr)
            print(f"{detail}{label}  {_short(instr.raw):<46}  [{loc}]")
            if src:
                src_trimmed = src.strip()[:70]
                print(f"{detail}       ╰─ {src_trimmed}")

        # ── cabeçalho ────────────────────────────────────────────────────────

        has_src = any(i.source_line > 0 for i in k.instructions)
        has_src_files = any(v is not None for v in src_cache.values())

        print(f"\n{'═'*W}")
        print(f"  Árvore de Desvios — {k.name}")
        print(f"{'═'*W}")
        print(f"  Blocos básicos       : {len(blocks)}")
        print(f"  Branches total       : {k.total_branches}")
        print(f"  Condicionais (@%p)   : {k.predicated_branches}"
              "  ← potencial divergência de warp")
        print(f"  Incondicionais (.uni): {k.unconditional_branches}"
              "  ← sem divergência")
        print(f"  Instruções setp      : {k.setp_count}"
              "  ← comparações que geram predicados")
        print(f"  Branch ratio         : {k.branch_ratio:.1%}")
        if not has_src:
            print(f"  (sem mapeamento de fonte — compile com -lineinfo para ver código .cu)")
        elif not has_src_files:
            print(f"  (arquivos .cu não encontrados nos caminhos do file_map)")
        print(f"{'─'*W}")

        EDGE_ICONS = {
            "conditional": "→[cond] ",
            "jump":        "→[jump] ",
            "fallthrough": "→[fall] ",
        }

        # ── DFS ──────────────────────────────────────────────────────────────
        visited: set = set()
        stack = [(order[0], 0)] if order else []

        while stack:
            lbl, depth = stack.pop()
            if lbl not in blocks:
                continue

            block = blocks[lbl]
            display = lbl.replace("__ENTRY__", "ENTRY")
            n = len(block.instructions)

            if block.is_entry:
                tag = "[ENTRADA]"
            elif block.is_terminal:
                tag = "[SAÍDA]  "
            elif any(e == "conditional" for e, _ in block.exits):
                tag = "[BRANCH] "
            else:
                tag = "         "

            prefix = "  " + "│  " * depth
            connector = "└─" if depth else "  "
            already = " (já visitado)" if lbl in visited else ""
            print(f"{prefix}{connector}{tag} {display}  ({n} instrs){already}")

            if lbl in visited:
                continue
            visited.add(lbl)

            # Detalhe do branch terminador do bloco
            detail = "  " + "   " * depth + "     "
            last = block.instructions[-1] if block.instructions else None

            if last and last.op_base == "bra":
                bra_idx = len(block.instructions) - 1
                if last.is_predicated:
                    setp = _find_setp(block.instructions, bra_idx)
                    if setp:
                        # Mostrar setp com sua linha de código-fonte
                        _print_branch_detail(detail, setp, "setp ")
                    # Mostrar bra — só repetir o código-fonte se for linha diferente do setp
                    bra_src = _src_line(last)
                    setp_src = _src_line(setp) if setp else None
                    if bra_src and bra_src != setp_src:
                        _print_branch_detail(detail, last, "bra  ")
                    else:
                        loc = _ptx_loc(last)
                        print(f"{detail}bra    {_short(last.raw):<46}  [{loc}]")
                else:
                    _print_branch_detail(detail, last, "jump ")
            elif last and last.op_base in ("ret", "exit", "brx"):
                loc = _ptx_loc(last)
                print(f"{detail}{last.op_base:<6} {'':<46}  [{loc}]")

            # Arestas de saída
            for etype, target in reversed(block.exits):
                icon = EDGE_ICONS.get(etype, "→ ")
                print(f"{detail}{icon}{target}")
                if target not in visited:
                    stack.append((target, depth + 1))

        # Blocos com back-edges / não alcançados pelo DFS
        unreachable = [l for l in order if l not in visited and l in blocks]
        if unreachable:
            print(f"\n  Blocos com back-edges / não alcançados pelo DFS ({len(unreachable)}):")
            for lbl in unreachable[:12]:
                b = blocks[lbl]
                last = b.instructions[-1] if b.instructions else None
                loc_str = f"  {_loc(last)}" if last else ""
                exits_str = "  ".join(
                    f"{EDGE_ICONS.get(e,'→')}{t}" for e, t in b.exits[:3]
                )
                print(f"    {lbl:<28} ({len(b.instructions):>3} instrs){loc_str}")
                if exits_str:
                    print(f"    {'':28}  {exits_str}")
            if len(unreachable) > 12:
                print(f"    ... e mais {len(unreachable) - 12} blocos")
        print()

    def plot_branch_cfg(self, max_blocks: int = 30):
        """
        Exibe o Grafo de Fluxo de Controle (CFG) interativo com Plotly.

        Cada nó é um bloco básico; arestas mostram para onde cada branch leva.
          Laranja = condicional (@%p bra)   → divergência de warp possível
          Azul    = incondicional (bra.uni) → sem divergência
          Cinza   = fall-through

        Args:
            max_blocks: número máximo de blocos mostrados (BFS a partir da entrada).
        """
        return _plot_branch_cfg(self.kernel, max_blocks)

    # ── exportação ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        k = self.kernel
        return {
            "kernel":      k.name,
            "metrics":     k.metrics_dict(),
            "categories":  k.category_counts,
            "registers":   {t: list(r) for t, r in k.reg_decls.items()},
            "diagnostics": run_heuristics(k),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
