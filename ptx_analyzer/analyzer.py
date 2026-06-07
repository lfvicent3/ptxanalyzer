"""
Classe principal PTXAnalyzer.
"""

import json
from .core import PTXKernel, CATEGORIES
from .parser import parse_ptx
from .heuristics import run_heuristics, LEVEL_ICONS
from .visuals import (
    plot_category_pie, plot_category_bar, plot_instruction_timeline,
    plot_register_types, plot_instruction_mix_stacked, plot_memory_access_breakdown,
    plot_roofline
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

    Uso:
        a = PTXAnalyzer.from_file("kernel.ptx")
        a.show()                    # interface ipywidgets com 4 abas
        a.plot_distribution()       # pie chart
        a.show_stats()              # métricas no terminal
        a.to_dict()                 # exportar como dict
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
    def from_file(cls, path: str, kernel_index: int = 0) -> "PTXAnalyzer":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
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
                    plot_fn()
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
