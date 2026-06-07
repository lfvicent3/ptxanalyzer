"""
Classe PTXComparator para comparar múltiplos kernels.
"""

from typing import Dict, List
from .analyzer import PTXAnalyzer
from .visuals import plot_instruction_mix_stacked, plot_memory_access_breakdown, plot_roofline

# ──────────────────────────────────────────────────────────────────────────────
# 8. PTXComparator — análise de múltiplos kernels
# ──────────────────────────────────────────────────────────────────────────────

class PTXComparator:
    """
    Compara múltiplos kernels PTX lado a lado.

    Uso:
        comp = PTXComparator()
        comp.add("bubble", PTXAnalyzer.from_file("bubble.ptx"))
        comp.add("radix",  PTXAnalyzer.from_file("radix.ptx"))
        comp.show_table()
        comp.plot_comparison("Instruções", "Branches", "ld.global")
        comp.plot_radar()
    """

    def __init__(self):
        self._analyzers: Dict[str, PTXAnalyzer] = {}
        self._order: List[str] = []

    def add(self, name: str, analyzer: PTXAnalyzer) -> "PTXComparator":
        self._analyzers[name] = analyzer
        if name not in self._order:
            self._order.append(name)
        return self

    def _metrics(self) -> Dict[str, dict]:
        return {n: self._analyzers[n].kernel.metrics_dict() for n in self._order}

    # ── tabela HTML com heat map por coluna ─────────────────────────────────

    def show_table(self):
        try:
            from IPython.display import display
            import ipywidgets as w
        except ImportError:
            self._print_table()
            return

        table = self._metrics()
        if not table:
            print("Nenhum kernel adicionado.")
            return

        metric_names = list(next(iter(table.values())).keys())

        # max por métrica para coloração
        max_vals = {m: max((table[n][m] for n in self._order), default=1)
                    for m in metric_names}

        def _cell_bg(val, max_val):
            if max_val == 0:
                return "#1e293b"
            ratio = val / max_val
            r = int(30 + ratio * 180)
            g = int(41 - ratio * 10)
            b = int(59 - ratio * 20)
            return f"rgb({r},{g},{b})"

        header_cells = "".join(
            f'<th style="padding:8px 12px;background:#1e293b;'
            f'color:#94a3b8;text-align:right;font-weight:600;'
            f'font-size:12px;">{m}</th>'
            for m in metric_names
        )
        header = (
            f'<tr>'
            f'<th style="padding:8px 12px;background:#1e293b;'
            f'color:#94a3b8;text-align:left;font-size:12px;">Kernel</th>'
            + header_cells + '</tr>'
        )

        rows_html = ""
        for name in self._order:
            m_dict = table[name]
            cells = "".join(
                f'<td style="padding:6px 12px;text-align:right;'
                f'background:{_cell_bg(float(m_dict[m]) if isinstance(m_dict[m], (int,float)) else 0, float(max_vals[m]) if isinstance(max_vals[m], (int,float)) else 1)};'
                f'color:#e2e8f0;font-size:13px;">'
                f'{m_dict[m]}</td>'
                for m in metric_names
            )
            rows_html += (
                f'<tr>'
                f'<td style="padding:6px 12px;font-weight:600;'
                f'color:#38bdf8;font-size:13px;">{name}</td>'
                + cells + '</tr>'
            )

        html = (
            '<div style="background:#0f1416;padding:16px;border-radius:8px;'
            'overflow-x:auto;">'
            '<h3 style="color:#f1f5f9;margin:0 0 12px;">Comparação de Kernels PTX</h3>'
            '<table style="border-collapse:collapse;width:100%;">'
            '<thead>' + header + '</thead>'
            '<tbody>' + rows_html + '</tbody>'
            '</table></div>'
        )
        display(w.HTML(value=html))

    def _print_table(self):
        table = self._metrics()
        if not table:
            return
        metric_names = list(next(iter(table.values())).keys())
        print(f"{'Kernel':<18}", "  ".join(f"{m[:10]:>10}" for m in metric_names))
        print("-" * (18 + 12 * len(metric_names)))
        for name in self._order:
            vals = table[name]
            row = f"{name:<18}"
            for m in metric_names:
                row += f"  {str(vals[m]):>10}"
            print(row)

    # ── gráficos ─────────────────────────────────────────────────────────────

    def plot_comparison(self, *metrics: str):
        """Bar chart agrupado para as métricas especificadas."""
        import plotly.graph_objects as go
        from .visuals import _DARK_LAYOUT

        if not metrics:
            metrics = ("Instruções", "Branches", "ld.global", "Registradores")

        table = self._metrics()
        names = self._order
        palette = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b",
                   "#8b5cf6", "#ec4899", "#06b6d4", "#f97316"]

        fig = go.Figure()
        for idx, metric in enumerate(metrics):
            vals = [table[n].get(metric, 0) for n in names]
            fig.add_trace(go.Bar(
                name=metric, x=names, y=vals,
                marker_color=palette[idx % len(palette)],
                hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y}}<extra></extra>",
            ))
        fig.update_layout(
            title="Comparação de Métricas PTX",
            barmode="group",
            xaxis_title="Kernel",
            yaxis_title="Valor",
            **_DARK_LAYOUT,
        )
        fig.show()

    def plot_radar(self, normalize: bool = True):
        """Radar chart do perfil de cada kernel."""
        import plotly.graph_objects as go
        from .visuals import _DARK_LAYOUT

        radar_metrics = [
            "Instruções", "Registradores", "ld.global",
            "Branches", "Int.Aritmética", "shfl.sync",
        ]
        table = self._metrics()
        names = self._order

        if normalize:
            max_vals = {
                m: max((float(table[n].get(m, 0)) for n in names), default=1) or 1
                for m in radar_metrics
            }
        else:
            max_vals = {m: 1 for m in radar_metrics}

        palette = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b",
                   "#8b5cf6", "#ec4899", "#06b6d4", "#f97316"]

        fig = go.Figure()
        for idx, name in enumerate(names):
            vals = [float(table[name].get(m, 0)) / max_vals[m]
                    for m in radar_metrics]
            vals.append(vals[0])  # fecha o polígono
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=radar_metrics + [radar_metrics[0]],
                fill="toself",
                name=name,
                line=dict(color=palette[idx % len(palette)]),
                hovertemplate="<b>" + name + "</b><br>%{theta}: %{r:.2f}<extra></extra>",
            ))

        fig.update_layout(
            title="Radar — Perfil de Kernels PTX",
            polar=dict(
                bgcolor="#111a1e",
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="#334155", color="#64748b"),
                angularaxis=dict(gridcolor="#334155", color="#94a3b8"),
            ),
            **_DARK_LAYOUT,
        )
        fig.show()

    def plot_mix(self):
        """Stacked bar: mix de instruções (%) para todos os kernels."""
        plot_instruction_mix_stacked(
            {n: self._analyzers[n].kernel for n in self._order}
        )

    def plot_memory_breakdown(self):
        """Grouped bar: acessos global / shared / local para todos os kernels."""
        plot_memory_access_breakdown(
            {n: self._analyzers[n].kernel for n in self._order}
        )

    def plot_roofline(self, peak_flops: float = 15.7, peak_bw: float = 900.0):
        """
        Posição de todos os kernels no modelo Roofline.
        """
        plot_roofline(
            {n: self._analyzers[n].kernel for n in self._order},
            peak_flops, peak_bw,
        )

    def generate_report_table(self) -> str:
        """Retorna tabela Markdown comparativa para incluir no relatório."""
        table = self._metrics()
        cols = ["Instruções", "Registradores", "ld.global", "ld/st.local",
                "Branches", "BranchRatio", "Int.Aritmética", "SóRegistros"]
        header = "| Algoritmo | " + " | ".join(cols) + " |"
        sep    = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
        rows = []
        for name in self._order:
            m = table[name]
            rows.append("| " + name + " | " +
                        " | ".join(str(m.get(c, "—")) for c in cols) + " |")
        return "\n".join([header, sep] + rows)
