"""
Visualizações com Plotly para análise PTX.
"""

from typing import Dict
from .core import PTXKernel, CATEGORIES, CATEGORY_COLORS

# ──────────────────────────────────────────────────────────────────────────────
# 5. Visualizações (Plotly)
# ──────────────────────────────────────────────────────────────────────────────

_DARK_LAYOUT = dict(
    paper_bgcolor="#0f1416",
    plot_bgcolor="#0f1416",
    font=dict(color="#cbd5e1", size=13),
    margin=dict(l=60, r=40, t=50, b=60),
)


def plot_category_pie(kernel: PTXKernel):
    """Pie chart da distribuição de categorias de instrução."""
    import plotly.graph_objects as go

    counts = kernel.category_counts
    labels = [c for c, n in counts.items() if n > 0]
    values = [counts[c] for c in labels]
    colors = [CATEGORY_COLORS.get(c, "#374151") for c in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=colors, line=dict(color="#0f1416", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value} instruções<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Distribuição — {kernel.name}",
        **_DARK_LAYOUT,
    )
    fig.show()


def plot_category_bar(kernel: PTXKernel):
    """Bar chart de contagem por categoria."""
    import plotly.graph_objects as go

    counts = kernel.category_counts
    cats = sorted(counts, key=lambda c: -counts[c])
    vals = [counts[c] for c in cats]
    colors = [CATEGORY_COLORS.get(c, "#374151") for c in cats]

    fig = go.Figure(go.Bar(
        x=cats, y=vals,
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>%{y} instruções<extra></extra>",
    ))
    fig.update_layout(
        title=f"Contagem por Categoria — {kernel.name}",
        xaxis_title="Categoria",
        yaxis_title="Nº de Instruções",
        **_DARK_LAYOUT,
    )
    fig.show()


def plot_instruction_timeline(kernel: PTXKernel, max_points: int = 400):
    """Scatter plot: sequência de categorias ao longo do kernel."""
    import plotly.graph_objects as go

    instrs = kernel.instructions
    step = max(1, len(instrs) // max_points)
    sampled = instrs[::step]

    cats = sorted(set(i.category for i in instrs))
    cat_idx = {c: j for j, c in enumerate(cats)}

    fig = go.Figure()
    for cat in cats:
        xs = [i.line_no for i in sampled if i.category == cat]
        ys = [cat_idx[cat]] * len(xs)
        if not xs:
            continue
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(color=CATEGORY_COLORS.get(cat, "#374151"), size=5),
            name=cat,
            hovertemplate=f"<b>{cat}</b><br>Linha %{{x}}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Timeline de Instruções — {kernel.name}",
        xaxis_title="Linha no PTX",
        yaxis=dict(tickvals=list(range(len(cats))), ticktext=cats),
        **_DARK_LAYOUT,
    )
    fig.show()


def plot_register_types(kernel: PTXKernel):
    """Bar chart de contagem de registradores por tipo."""
    import plotly.graph_objects as go

    reg_counts = {rtype: len(regs) for rtype, regs in kernel.reg_decls.items()}
    if not reg_counts:
        print("Nenhuma declaração de registrador encontrada.")
        return

    types = sorted(reg_counts)
    vals  = [reg_counts[t] for t in types]
    palette = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444",
               "#06b6d4", "#ec4899", "#f97316"]

    fig = go.Figure(go.Bar(
        x=types, y=vals,
        marker_color=palette[:len(types)],
        hovertemplate="<b>.%{x}</b><br>%{y} registradores<extra></extra>",
    ))
    fig.update_layout(
        title=f"Registradores por Tipo — {kernel.name}",
        xaxis_title="Tipo PTX",
        yaxis_title="Quantidade",
        **_DARK_LAYOUT,
    )
    fig.show()


def plot_instruction_mix_stacked(kernels: Dict[str, "PTXKernel"]):
    """
    Stacked bar chart: mix de instruções (%) por kernel.
    Mostra visualmente a "impressão digital" de cada algoritmo —
    ex: radix sort dominado por logic, bubble sort por control.
    """
    import plotly.graph_objects as go

    cats = list(CATEGORIES.keys()) + ["other"]
    names = list(kernels.keys())

    fig = go.Figure()
    for cat in cats:
        vals = [kernels[n].instruction_mix.get(cat, 0.0) for n in names]
        if sum(vals) == 0:
            continue
        fig.add_trace(go.Bar(
            name=cat, x=names, y=vals,
            marker_color=CATEGORY_COLORS.get(cat, "#374151"),
            hovertemplate=f"<b>%{{x}}</b><br>{cat}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        title="Mix de Instruções por Kernel (%)",
        barmode="stack",
        xaxis_title="Kernel",
        yaxis_title="% de instruções",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", y=-0.25),
        **_DARK_LAYOUT,
    )
    fig.show()


def plot_roofline(kernels: Dict[str, "PTXKernel"],
                  peak_flops: float = 15.7,
                  peak_bw: float = 900.0):
    """
    Modelo Roofline (Williams et al., 2009) para os kernels analisados.

    Cada kernel é representado como um ponto (intensidade aritmética, FLOPs/s estimado).
    A linha do teto mostra onde está o gargalo: memory-bound (esquerda) ou
    compute-bound (direita).

    Args:
        peak_flops: TFLOPs/s do pico de compute (default: T4 ≈ 8.1 TFLOPs fp32,
                    A100 ≈ 19.5 — ajuste para a GPU do Colab).
        peak_bw:    GB/s de largura de banda de memória (default: T4 ≈ 300 GB/s,
                    A100 ≈ 2000 GB/s).

    O eixo X (intensidade aritmética) é calculado estaticamente pelo ptx_analyzer.
    O eixo Y é uma estimativa relativa baseada na contagem de instruções.
    """
    import plotly.graph_objects as go
    import math

    ridge = peak_flops * 1e3 / peak_bw   # FLOPs/byte no ridge point

    # Linha do teto
    xs = [0.01, ridge, ridge * 100]
    ys_roof = [
        peak_bw * 0.01,           # lado esquerdo: memory-bound
        peak_flops * 1e3,         # ridge point
        peak_flops * 1e3,         # lado direito: compute-bound (plano)
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys_roof,
        mode="lines",
        line=dict(color="#94a3b8", width=2, dash="dash"),
        name="Roofline",
        hovertemplate="Roofline<extra></extra>",
    ))

    # Ridge point
    fig.add_vline(x=ridge, line=dict(color="#475569", dash="dot", width=1))
    fig.add_annotation(
        x=math.log10(ridge), y=0.95, xref="x", yref="paper",
        text=f"Ridge ≈ {ridge:.1f} FLOPs/B",
        showarrow=False, font=dict(color="#64748b", size=11),
        xanchor="left",
    )

    # Kernels como pontos
    palette = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b",
               "#8b5cf6", "#ec4899", "#06b6d4", "#f97316"]

    for idx, (name, k) in enumerate(kernels.items()):
        ai = max(k.arithmetic_intensity, 0.01)
        # FLOPs/s estimado: min(compute_bound, memory_bound) — modelo Roofline
        perf = min(peak_flops * 1e3, peak_bw * ai)

        region = "memory-bound" if ai < ridge else "compute-bound"
        fig.add_trace(go.Scatter(
            x=[ai], y=[perf],
            mode="markers+text",
            marker=dict(size=14, color=palette[idx % len(palette)],
                        line=dict(color="white", width=1)),
            text=[name],
            textposition="top center",
            textfont=dict(size=11),
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"Int. Aritmética: {ai:.2f} FLOPs/B<br>"
                f"Perf estimada: {perf:.1f} GFLOPs/s<br>"
                f"Região: {region}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Roofline — Posição dos Kernels no Modelo de Performance",
        xaxis=dict(title="Intensidade Aritmética (FLOPs/byte)",
                   type="log", gridcolor="#1e293b"),
        yaxis=dict(title="Performance Estimada (GFLOPs/s)",
                   type="log", gridcolor="#1e293b"),
        **_DARK_LAYOUT,
    )
    fig.show()


def plot_memory_access_breakdown(kernels: Dict[str, "PTXKernel"]):
    """
    Grouped bar chart: global loads, global stores, shared, local por kernel.
    Destaca o custo relativo de cada espaço de memória —
    mostra visualmente o insight do professor sobre local vs global vs shared.
    """
    import plotly.graph_objects as go

    names = list(kernels.keys())
    memory_types = [
        ("ld.global",  [k.global_loads    for k in kernels.values()], "#f59e0b"),
        ("st.global",  [k.global_stores   for k in kernels.values()], "#ef4444"),
        ("shared",     [k.shared_accesses for k in kernels.values()], "#3b82f6"),
        ("ld/st.local",[k.local_accesses  for k in kernels.values()], "#8b5cf6"),
    ]

    fig = go.Figure()
    for label, vals, color in memory_types:
        if sum(vals) == 0:
            continue
        fig.add_trace(go.Bar(
            name=label, x=names, y=vals,
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        title="Acessos de Memória por Espaço (global / shared / local)",
        barmode="group",
        xaxis_title="Kernel",
        yaxis_title="Nº de instruções de memória",
        **_DARK_LAYOUT,
    )
    fig.show()
