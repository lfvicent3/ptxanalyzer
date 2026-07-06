"""
Visualizações com Plotly para análise PTX.
"""

from typing import Dict
from .core import PTXKernel, CATEGORIES, CATEGORY_COLORS, build_cfg, analyze_control_flow

# ──────────────────────────────────────────────────────────────────────────────
# 5. Visualizações (Plotly)
# ──────────────────────────────────────────────────────────────────────────────

_DARK_LAYOUT = dict(
    paper_bgcolor="#0f1416",
    plot_bgcolor="#0f1416",
    font=dict(color="#cbd5e1", size=13),
    margin=dict(l=60, r=40, t=50, b=60),
)


def _make_block_display_map(blocks, order):
    branch_idx = 1
    seq_idx = 1
    node_idx = 1
    display = {}

    for lbl in order:
        block = blocks[lbl]
        if block.is_entry or lbl == "__ENTRY__":
            display[lbl] = "ENTRY"
        elif block.is_terminal:
            display[lbl] = "END"
        elif lbl.startswith("__seq_"):
            display[lbl] = f"S{seq_idx}"
            seq_idx += 1
        elif any(edge_type == "conditional" for edge_type, _ in block.exits):
            display[lbl] = f"B{branch_idx}"
            branch_idx += 1
        else:
            display[lbl] = f"N{node_idx}"
            node_idx += 1
    return display


def _show_fig(fig):
    """Exibe figura Plotly no Jupyter/Colab via FigureWidget (fallback: fig.show())."""
    try:
        from IPython.display import display
        import plotly.graph_objects as go
        display(go.FigureWidget(fig))
    except Exception:
        fig.show()


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
    _show_fig(fig)
    return fig


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
    _show_fig(fig)
    return fig


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
    _show_fig(fig)
    return fig


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
    _show_fig(fig)
    return fig


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
    _show_fig(fig)
    return fig


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
    _show_fig(fig)
    return fig


def plot_branch_cfg(kernel: PTXKernel, max_blocks: int = 30):
    """
    Grafo de Fluxo de Controle (CFG) interativo.

    Cada nó representa um bloco básico. As arestas mostram o fluxo de controle:
      • Laranja  — desvio condicional (@%p bra) → divergência de warp possível
      • Azul     — salto incondicional (bra.uni) → sem divergência
      • Cinza    — fall-through (execução sequencial)

    Back-edges (loops) aparecem tracejados na lateral direita do grafo.
    Hover sobre o nó mostra a instrução PTX + linha do código-fonte .cu (quando
    compilado com -lineinfo).

    Args:
        max_blocks: número máximo de blocos exibidos (BFS a partir da entrada).
    """
    import plotly.graph_objects as go
    import os as _os
    from collections import deque

    blocks, order = build_cfg(kernel)
    if not blocks:
        print("Nenhum bloco básico encontrado.")
        return

    # ── Carrega arquivos-fonte (.cu) para exibir no hover ─────────────────────
    src_cache: Dict[int, list] = {}
    for fidx, fpath in kernel.file_map.items():
        try:
            with open(fpath, encoding="utf-8", errors="replace") as fh:
                src_cache[fidx] = fh.readlines()
        except OSError:
            src_cache[fidx] = None

    def _get_src(instr):
        if instr.source_line <= 0:
            return None
        lines = src_cache.get(instr.source_file)
        if not lines:
            return None
        idx = instr.source_line - 1
        return lines[idx].rstrip() if 0 <= idx < len(lines) else None

    # ── BFS para descobrir nós (limitado a max_blocks) ───────────────────────
    entry = order[0]
    bfs_order: list = []
    visited_bfs: set = set()
    queue: deque = deque([entry])

    while queue and len(bfs_order) < max_blocks:
        lbl = queue.popleft()
        if lbl in visited_bfs or lbl not in blocks:
            continue
        visited_bfs.add(lbl)
        bfs_order.append(lbl)
        for _, target in blocks[lbl].exits:
            if target in blocks and target not in visited_bfs:
                queue.append(target)

    visible = set(bfs_order)
    display_map = _make_block_display_map(blocks, bfs_order)

    # ── DFS iterativo para identificar back-edges reais (formam ciclos) ───────
    # Back-edge verdadeiro = aresta para ancestral no DFS tree
    dfs_visited: set = set()
    dfs_stack: set = set()
    true_back_edges: set = set()

    dfs_iter: list = [(entry, iter([(e, t) for e, t in blocks[entry].exits
                                    if t in visible]))]
    dfs_visited.add(entry)
    dfs_stack.add(entry)

    while dfs_iter:
        lbl, children = dfs_iter[-1]
        try:
            _, target = next(children)
            if target not in visible:
                continue
            if target in dfs_stack:
                true_back_edges.add((lbl, target))
            elif target not in dfs_visited:
                dfs_visited.add(target)
                dfs_stack.add(target)
                if target in blocks:
                    dfs_iter.append((target, iter([(e, t) for e, t in blocks[target].exits
                                                   if t in visible])))
        except StopIteration:
            dfs_stack.discard(lbl)
            dfs_iter.pop()

    # ── Atribuição de nível: caminho mais longo no DAG (ignora back-edges) ────
    # Usa Kahn's algorithm para ordem topológica + longest-path
    # Isso faz blocos de convergência (join points) ficarem abaixo de todos
    # os predecessores, eliminando arestas que cruzam para cima/mesmo nível.
    in_deg: Dict[str, int] = {l: 0 for l in bfs_order}
    for lbl in bfs_order:
        for _, target in blocks[lbl].exits:
            if target in visible and (lbl, target) not in true_back_edges:
                in_deg[target] = in_deg.get(target, 0) + 1

    dist: Dict[str, int] = {entry: 0}
    topo_q: deque = deque([l for l in bfs_order if in_deg.get(l, 0) == 0])

    while topo_q:
        lbl = topo_q.popleft()
        for _, target in blocks[lbl].exits:
            if target not in visible or (lbl, target) in true_back_edges:
                continue
            new_d = dist.get(lbl, 0) + 1
            if new_d > dist.get(target, 0):
                dist[target] = new_d
            in_deg[target] -= 1
            if in_deg[target] == 0:
                topo_q.append(target)

    level_of: Dict[str, int] = {l: dist.get(l, 0) for l in bfs_order}

    # ── Atribuição de coluna (por nível, em ordem BFS) ────────────────────────
    col_counter: Dict[int, int] = {}
    col_of: Dict[str, int] = {}
    for lbl in sorted(bfs_order, key=lambda l: (level_of.get(l, 0), bfs_order.index(l))):
        lv = level_of[lbl]
        col_counter.setdefault(lv, 0)
        col_of[lbl] = col_counter[lv]
        col_counter[lv] += 1

    # ── Dimensionamento dinâmico ──────────────────────────────────────────────
    max_cols_in_level = max(col_counter.values(), default=1)
    max_depth = max(level_of.values(), default=0) if level_of else 0

    NODE_W = 200
    NODE_H = 60
    # X_STEP: largura de coluna — mais colunas → menor passo (até 240 mín)
    X_STEP = max(NODE_W + 40, min(300, 800 // max(max_cols_in_level, 1)))
    Y_STEP = 160
    # Paredes para arcos: margem fixa baseada no nó, não no X_STEP
    _WALL_MARGIN = NODE_W * 1.6   # 320 px de folga a cada lado

    def _pos(lbl):
        lv = level_of.get(lbl, 0)
        col = col_of.get(lbl, 0)
        n_cols = col_counter.get(lv, 1)
        x = (col - (n_cols - 1) / 2) * X_STEP
        y = -lv * Y_STEP
        return x, y

    fig = go.Figure()

    # ── Arestas ───────────────────────────────────────────────────────────────
    # Convenção visual (inspirada em IDA Pro / Ghidra):
    #   fall-through  → linha reta vertical (caminho normal)
    #   condicional   → arco pelo lado ESQUERDO (branch taken)
    #   jump incond.  → arco pelo lado ESQUERDO
    #   back-edge     → arco pelo lado DIREITO, tracejado (loop/goto)
    EDGE_COLOR = {"conditional": "#f59e0b", "jump": "#3b82f6", "fallthrough": "#6b7280"}

    all_x_vals = [_pos(l)[0] for l in bfs_order]
    right_wall = (max(all_x_vals) if all_x_vals else 0) + NODE_W / 2 + _WALL_MARGIN
    left_wall  = (min(all_x_vals) if all_x_vals else 0) - NODE_W / 2 - _WALL_MARGIN

    def _draw_edge(x0, y0, x1, y1, color, dash, via_left=False, via_right=False, width=1.8):
        if via_right:
            wall = right_wall
            sx, tx = x0 + NODE_W / 2, x1 + NODE_W / 2
        elif via_left:
            wall = left_wall
            sx, tx = x0 - NODE_W / 2, x1 - NODE_W / 2
        else:
            wall = None

        if wall is not None:
            fig.add_trace(go.Scatter(
                x=[sx, wall, wall, tx, None],
                y=[y0, y0, y1, y1, None],
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_annotation(
                x=tx, y=y1,
                ax=wall, ay=y1,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.0,
                arrowwidth=width, arrowcolor=color, text="",
            )
        else:
            fig.add_trace(go.Scatter(
                x=[x0, x1, None], y=[y0 - NODE_H / 2, y1 + NODE_H / 2, None],
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_annotation(
                x=x1, y=y1 + NODE_H / 2,
                ax=x0, ay=y0 - NODE_H / 2,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.2,
                arrowwidth=width, arrowcolor=color, text="",
            )

    for lbl in bfs_order:
        block = blocks[lbl]
        x0, y0 = _pos(lbl)

        for etype, target in block.exits:
            if target not in visible:
                continue
            x1, y1 = _pos(target)
            color = EDGE_COLOR.get(etype, "#6b7280")
            is_back = (lbl, target) in true_back_edges

            if is_back:
                _draw_edge(x0, y0, x1, y1, color, "dot", via_right=True, width=1.5)
            elif etype == "fallthrough":
                _draw_edge(x0, y0, x1, y1, color, "solid")
            else:  # conditional ou jump → lado esquerdo
                _draw_edge(x0, y0, x1, y1, color, "solid", via_left=True)

    # ── Helpers para hover ────────────────────────────────────────────────────
    def _find_setp_hover(instrs):
        if not instrs:
            return None
        bra = instrs[-1]
        if bra.op_base != "bra" or not bra.is_predicated or not bra.predicate:
            return None
        pred_reg = bra.predicate.lstrip("@").lstrip("!")
        for ins in reversed(instrs[:-1]):
            if ins.op_base == "setp" and ins.operands and ins.operands[0] == pred_reg:
                return ins
        return None

    def _loc_str(instr):
        tag = f"PTX:{instr.line_no}"
        if instr.source_line > 0:
            fname = _os.path.basename(kernel.file_map.get(instr.source_file, ""))
            tag += f"  {fname}:{instr.source_line}" if fname else f"  linha:{instr.source_line}"
        return tag

    # ── Nós (retângulos + texto + hover) ─────────────────────────────────────
    NODE_FILL = {
        "entry":    ("#065f46", "#6ee7b7"),
        "terminal": ("#7f1d1d", "#fca5a5"),
        "branch":   ("#78350f", "#fcd34d"),
        "normal":   ("#1e293b", "#475569"),
    }

    for lbl in bfs_order:
        block = blocks[lbl]
        x, y = _pos(lbl)
        n = len(block.instructions)

        if block.is_entry:
            fk = "entry"
        elif block.is_terminal:
            fk = "terminal"
        elif any(e == "conditional" for e, _ in block.exits):
            fk = "branch"
        else:
            fk = "normal"

        fill, border = NODE_FILL[fk]
        display_lbl = display_map.get(lbl, lbl.replace("__ENTRY__", "ENTRY"))
        branch_count = sum(1 for ins in block.instructions if ins.op_base == "bra")
        mem_count = sum(1 for ins in block.instructions if ins.category == "memory")
        arith_count = sum(1 for ins in block.instructions if ins.category == "arithmetic")

        fig.add_shape(
            type="rect",
            x0=x - NODE_W / 2, x1=x + NODE_W / 2,
            y0=y - NODE_H / 2, y1=y + NODE_H / 2,
            fillcolor=fill, line=dict(color=border, width=1.5),
        )
        fig.add_annotation(
            x=x, y=y,
            text=f"<b>{display_lbl}</b><br>"
                 f"<span style='font-size:10px;color:#94a3b8'>{n} instr | mem {mem_count}</span><br>"
                 f"<span style='font-size:9px;color:#94a3b8'>"
                 f"branch {branch_count}</span>",
            showarrow=False,
            font=dict(size=11, color="#f1f5f9"),
            xref="x", yref="y", align="center",
        )

        # Hover detalhado com PTX + fonte .cu
        last_instr = block.instructions[-1] if block.instructions else None
        exits_str = "  ".join(
            f"{e}→{display_map.get(t, t)}" for e, t in block.exits[:3]
        ) or "—"

        setp_instr = _find_setp_hover(block.instructions)
        branch_detail = ""

        if setp_instr:
            setp_src = _get_src(setp_instr)
            branch_detail += (
                f"<br><b>setp:</b> {setp_instr.raw.strip()[:52]}"
                f"<br>  📍 {_loc_str(setp_instr)}"
            )
            if setp_src:
                branch_detail += f"<br>  <i>{setp_src.strip()[:55]}</i>"

        if last_instr and last_instr.op_base in ("bra", "brx"):
            bra_src = _get_src(last_instr)
            branch_detail += (
                f"<br><b>bra:</b>  {last_instr.raw.strip()[:52]}"
                f"<br>  📍 {_loc_str(last_instr)}"
            )
            if bra_src and bra_src != (_get_src(setp_instr) if setp_instr else None):
                branch_detail += f"<br>  <i>{bra_src.strip()[:55]}</i>"
        elif last_instr and last_instr.op_base in ("ret", "exit"):
            branch_detail += (
                f"<br><b>{last_instr.op_base}</b>  {_loc_str(last_instr)}"
            )

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(size=NODE_W * 0.85, opacity=0, symbol="square"),
            hovertemplate=(
                f"<b>{display_lbl}</b><br>"
                f"Bloco PTX: {lbl}<br>"
                f"Instruções: {n}<br>"
                f"Memória: {mem_count}<br>"
                f"Aritméticas: {arith_count}<br>"
                f"Branches: {branch_count}<br>"
                f"Saídas: {exits_str}"
                f"{branch_detail}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    # ── Legenda manual ────────────────────────────────────────────────────────
    for color, dash, name in [
        ("#6b7280", "solid", "Fall-through (reto)"),
        ("#f59e0b", "solid", "Desvio condicional (@%p bra) ← esquerda"),
        ("#3b82f6", "solid", "Salto incondicional (bra.uni) ← esquerda"),
        ("#f59e0b", "dot",   "Back-edge / loop (direita, tracejado)"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=color, width=2, dash=dash),
            name=name, showlegend=True,
        ))
    for color, name in [
        ("#6ee7b7", "Entrada"), ("#fca5a5", "Saída (ret/exit)"),
        ("#fcd34d", "Branch condicional"), ("#475569", "Bloco normal"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, size=10, symbol="square"),
            name=name, showlegend=True,
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    if bfs_order:
        all_y = [_pos(l)[1] for l in bfs_order]
        ymarg = NODE_H * 2
        xr = [left_wall - 20, right_wall + 20]
        yr = [min(all_y) - ymarg, max(all_y) + ymarg]
    else:
        xr = [-400, 400]
        yr = [-200, 50]

    truncated = len(blocks) > max_blocks
    title = f"CFG Simplificado — {kernel.name}"
    if truncated:
        title += f"  (primeiros {max_blocks} de {len(blocks)} blocos)"

    graph_height = max(600, max_depth * Y_STEP + 350)
    fig.update_layout(
        title=title,
        xaxis=dict(visible=False, range=xr),
        yaxis=dict(visible=False, range=yr),
        legend=dict(orientation="h", y=-0.05, font=dict(size=11)),
        height=graph_height,
        **_DARK_LAYOUT,
    )
    _show_fig(fig)
    return fig


def plot_decision_tree(kernel: PTXKernel, max_decisions: int = 20):
    """
    Árvore de Decisão interativa — mostra APENAS os pontos de branch do kernel.

    Diferente do CFG completo, aqui cada nó representa um ponto de decisão
    (bloco com @%p bra) e mostra diretamente a condição do código-fonte .cu.
    Blocos sequenciais sem branch são omitidos (aparecem como contagem de
    instruções nas arestas). Loops aparecem como arestas tracejadas com label.

    Isso permite entender a LÓGICA do algoritmo sem ruído de blocos auxiliares.

    Args:
        max_decisions: número máximo de nós de decisão exibidos.
    """
    import plotly.graph_objects as go
    import os as _os
    from collections import deque

    blocks, order = build_cfg(kernel)
    if not blocks:
        print("Nenhum bloco básico encontrado.")
        return

    # ── Carrega arquivos-fonte ────────────────────────────────────────────────
    src_cache: Dict[int, list] = {}
    for fidx, fpath in kernel.file_map.items():
        try:
            with open(fpath, encoding="utf-8", errors="replace") as fh:
                src_cache[fidx] = fh.readlines()
        except OSError:
            src_cache[fidx] = None

    def _get_src(instr):
        if instr.source_line <= 0:
            return None
        lines = src_cache.get(instr.source_file)
        if not lines:
            return None
        idx = instr.source_line - 1
        return lines[idx].rstrip() if 0 <= idx < len(lines) else None

    def _find_setp(instrs):
        if not instrs:
            return None
        bra = instrs[-1]
        if bra.op_base != "bra" or not bra.is_predicated or not bra.predicate:
            return None
        pred_reg = bra.predicate.lstrip("@").lstrip("!")
        for ins in reversed(instrs[:-1]):
            if ins.op_base == "setp" and ins.operands and ins.operands[0] == pred_reg:
                return ins
        return None

    def _condition_label(block):
        """Retorna a condição .cu do branch, ou descrição PTX se sem lineinfo."""
        last = block.instructions[-1] if block.instructions else None
        if last is None or last.op_base != "bra":
            return ""
        setp = _find_setp(block.instructions)
        if setp:
            src = _get_src(setp)
            if src:
                return src.strip()[:60]
            return setp.raw.strip()[:60]
        return last.raw.strip()[:60]

    # ── BFS para encontrar nós de decisão (blocos com branch condicional) ─────
    entry = order[0]
    visited: set = set()
    queue: deque = deque([entry])
    decision_nodes: list = []   # apenas blocos com exits condicionais
    all_visible: set = set()
    # Mapeia cada bloco → próximo nó de decisão alcançável por cada aresta
    # Format: (src_lbl, edge_type, dst_decision_lbl, n_intermediate_instrs, is_back)

    # Primeiro, descobre todos os blocos em BFS
    bfs_all: list = []
    bfs_visited: set = set()
    bfs_q: deque = deque([entry])
    while bfs_q:
        lbl = bfs_q.popleft()
        if lbl in bfs_visited or lbl not in blocks:
            continue
        bfs_visited.add(lbl)
        bfs_all.append(lbl)
        for _, target in blocks[lbl].exits:
            if target in blocks and target not in bfs_visited:
                bfs_q.append(target)

    # DFS para identificar back-edges
    dfs_visited2: set = set()
    dfs_stack2: set = set()
    true_back: set = set()
    dfs_it = [(entry, iter([(e, t) for e, t in blocks[entry].exits if t in bfs_visited]))]
    dfs_visited2.add(entry)
    dfs_stack2.add(entry)
    while dfs_it:
        lbl, children = dfs_it[-1]
        try:
            _, target = next(children)
            if target not in bfs_visited:
                continue
            if target in dfs_stack2:
                true_back.add((lbl, target))
            elif target not in dfs_visited2:
                dfs_visited2.add(target)
                dfs_stack2.add(target)
                if target in blocks:
                    dfs_it.append((target, iter([(e, t) for e, t in blocks[target].exits
                                                 if t in bfs_visited])))
        except StopIteration:
            dfs_stack2.discard(lbl)
            dfs_it.pop()

    # Identifica blocos de decisão (exits com conditional) e blocos terminais
    is_decision = {lbl for lbl in bfs_all
                   if any(e == "conditional" for e, _ in blocks[lbl].exits)}
    is_terminal = {lbl for lbl in bfs_all if blocks[lbl].is_terminal}
    key_nodes = is_decision | is_terminal | {entry}

    # Nós selecionados: entrada + decisões (limitadas) + terminais (máx 3)
    _decision_list = [l for l in bfs_all if l in is_decision and l != entry]
    _terminal_list = [l for l in bfs_all if l in is_terminal]
    selected = ([entry]
                + _decision_list[:max_decisions - 1]
                + _terminal_list[:3])
    sel_set = set(selected)

    def _next_key(start_lbl, edge_type):
        """Segue o grafo a partir de start_lbl pela aresta edge_type até o
        próximo nó em sel_set (decisão ou terminal)."""
        visited_walk: set = {start_lbl}
        cur = start_lbl
        for etype, target in blocks[cur].exits:
            if etype != edge_type and cur == start_lbl:
                continue
            if (cur, target) in true_back:
                return target, 0, True
            nxt = target
            intermediate = 0
            # Caminha por blocos intermediários (sequenciais sem branch)
            while nxt and nxt not in sel_set and nxt in blocks:
                visited_walk.add(nxt)
                intermediate += len(blocks[nxt].instructions)
                nexts = [t for _, t in blocks[nxt].exits if t not in visited_walk]
                nxt = nexts[0] if nexts else None
            if nxt and nxt in sel_set:
                return nxt, intermediate, False
        return None, 0, False

    # Arestas entre nós selecionados
    edges: list = []   # (src, dst, label, is_back, color)
    for lbl in selected:
        if lbl not in blocks:
            continue
        for etype, target in blocks[lbl].exits:
            if (lbl, target) in true_back:
                color = "#f59e0b" if etype == "conditional" else "#3b82f6"
                tgt_lbl = target if target in sel_set else None
                if tgt_lbl:
                    edges.append((lbl, tgt_lbl, "↩ loop", True, color))
                continue
            # Segue até o próximo nó selecionado
            dst, n_inter, is_bk = _next_key(lbl, etype)
            if dst is None:
                continue
            if etype == "conditional":
                label = "sim"
                color = "#f59e0b"
            elif etype == "fallthrough":
                label = "não" if any(e == "conditional" for e, _ in blocks[lbl].exits) else "→"
                color = "#6b7280"
            else:
                label = "→"
                color = "#3b82f6"
            if n_inter > 0:
                label += f" ({n_inter} instr)"
            edges.append((lbl, dst, label, is_bk, color))

    # ── Layout: longest-path nos nós selecionados (igual ao CFG) ────────────
    # Kahn's algorithm para garantir que join-points fiquem abaixo de todos
    # os predecessores (sem cruzamentos).
    fwd_edges = [(s, d, lb, c) for s, d, lb, ib, c in edges if not ib]

    dt_in_deg: Dict[str, int] = {l: 0 for l in sel_set}
    for s, d, _, _ in fwd_edges:
        if s in sel_set and d in sel_set:
            dt_in_deg[d] = dt_in_deg.get(d, 0) + 1

    dt_dist: Dict[str, int] = {entry: 0}
    dt_topo: deque = deque([l for l in sel_set if dt_in_deg.get(l, 0) == 0])

    while dt_topo:
        lbl = dt_topo.popleft()
        for s, d, _, _ in fwd_edges:
            if s != lbl or d not in sel_set:
                continue
            new_d = dt_dist.get(lbl, 0) + 1
            if new_d > dt_dist.get(d, 0):
                dt_dist[d] = new_d
            dt_in_deg[d] -= 1
            if dt_in_deg[d] == 0:
                dt_topo.append(d)

    level_sel: Dict[str, int] = {l: dt_dist.get(l, 0) for l in sel_set}

    col_sel: Dict[str, int] = {}
    col_cnt: Dict[int, int] = {}
    tree_order = sorted(sel_set, key=lambda l: (level_sel.get(l, 0),
                                                 bfs_all.index(l) if l in bfs_all else 999))

    for lbl in sorted(tree_order, key=lambda l: level_sel.get(l, 0)):
        lv = level_sel[lbl]
        col_cnt.setdefault(lv, 0)
        col_sel[lbl] = col_cnt[lv]
        col_cnt[lv] += 1

    max_cols_d = max(col_cnt.values(), default=1)
    max_depth_d = max(level_sel.values(), default=0) if level_sel else 0

    NODE_W = 260
    NODE_H = 80
    X_STEP = max(NODE_W + 30, min(340, 900 // max(max_cols_d, 1)))
    Y_STEP = 200

    def _pos_d(lbl):
        lv = level_sel.get(lbl, 0)
        col = col_sel.get(lbl, 0)
        n_cols = col_cnt.get(lv, 1)
        x = (col - (n_cols - 1) / 2) * X_STEP
        y = -lv * Y_STEP
        return x, y

    fig = go.Figure()

    # ── Arestas ───────────────────────────────────────────────────────────────
    for src, dst, label, is_bk, color in edges:
        if src not in level_sel or dst not in level_sel:
            continue
        x0, y0 = _pos_d(src)
        x1, y1 = _pos_d(dst)
        dash = "dot" if is_bk else "solid"

        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0 - NODE_H / 2, y1 + NODE_H / 2, None],
            mode="lines",
            line=dict(color=color, width=2, dash=dash),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_annotation(
            x=x1, y=y1 + NODE_H / 2,
            ax=x0, ay=y0 - NODE_H / 2,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowwidth=2, arrowcolor=color, text="",
        )
        # Label no meio da aresta
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2 - 10
        fig.add_annotation(
            x=mx, y=my, text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=10, color=color),
            bgcolor="#0f1416", borderpad=2,
            xref="x", yref="y",
        )

    # ── Nós ───────────────────────────────────────────────────────────────────
    for lbl in tree_order:
        if lbl not in level_sel:
            continue
        x, y = _pos_d(lbl)
        block = blocks[lbl]
        cond = _condition_label(block)
        display = lbl.replace("__ENTRY__", "ENTRY")
        n = len(block.instructions)

        if lbl == entry:
            fill, border, txt_color = "#065f46", "#6ee7b7", "#6ee7b7"
        elif block.is_terminal:
            fill, border, txt_color = "#7f1d1d", "#fca5a5", "#fca5a5"
        elif any(e == "conditional" for e, _ in block.exits):
            fill, border, txt_color = "#1c1917", "#f59e0b", "#fcd34d"
        else:
            fill, border, txt_color = "#1e293b", "#475569", "#94a3b8"

        fig.add_shape(
            type="rect",
            x0=x - NODE_W / 2, x1=x + NODE_W / 2,
            y0=y - NODE_H / 2, y1=y + NODE_H / 2,
            fillcolor=fill, line=dict(color=border, width=2),
        )

        # Linha 1: rótulo do bloco
        fig.add_annotation(
            x=x, y=y + 16,
            text=f"<b style='color:{txt_color}'>{display}</b>"
                 f"<span style='color:#64748b;font-size:10px'>  {n} instr</span>",
            showarrow=False, font=dict(size=12, color="#f1f5f9"),
            xref="x", yref="y", align="center",
        )
        # Linha 2: condição do .cu (se branch)
        if cond:
            fig.add_annotation(
                x=x, y=y - 16,
                text=f"<i style='color:#fbbf24;font-size:10px'>{cond}</i>",
                showarrow=False, font=dict(size=10, color="#fbbf24"),
                xref="x", yref="y", align="center",
            )

        # Hover com detalhes completos
        setp = _find_setp(block.instructions)
        last = block.instructions[-1] if block.instructions else None
        hover_lines = [f"<b>{display}</b>", f"Instruções: {n}"]
        if setp:
            src_cu = _get_src(setp)
            hover_lines.append(f"<br>setp: {setp.raw.strip()[:55]}")
            if src_cu:
                hover_lines.append(f"<i>{src_cu.strip()[:60]}</i>")
        if last and last.op_base in ("bra", "ret", "exit"):
            hover_lines.append(f"<br>{last.op_base}: {last.raw.strip()[:50]}")
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers",
            marker=dict(size=NODE_W * 0.85, opacity=0, symbol="square"),
            hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
            showlegend=False,
        ))

    # ── Layout ────────────────────────────────────────────────────────────────
    if tree_order:
        all_x_d = [_pos_d(l)[0] for l in tree_order]
        all_y_d = [_pos_d(l)[1] for l in tree_order]
        xr = [min(all_x_d) - NODE_W, max(all_x_d) + NODE_W]
        yr = [min(all_y_d) - NODE_H * 2, max(all_y_d) + NODE_H * 2]
    else:
        xr, yr = [-300, 300], [-300, 100]

    truncated = len(is_decision) > max_decisions
    title = f"Árvore de Decisão — {kernel.name}"
    if truncated:
        n_total = len(is_decision)
        title += f"  (primeiros {max_decisions} de {n_total} decisões)"

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False, range=xr),
        yaxis=dict(visible=False, range=yr),
        height=max(500, max_depth_d * Y_STEP + 350),
        **_DARK_LAYOUT,
    )
    _show_fig(fig)
    return fig


def plot_bra_graph(kernel: PTXKernel, max_branches: int = 24):
    """
    Grafo focado apenas nos sites de `BRA`.

    Cada nó à esquerda representa um ponto de branch. À direita aparecem
    os destinos `taken` e `fallthrough`, com destaque visual para risco de
    divergência e pressão de memória nos dois caminhos.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly não disponível. Use control_flow(mode='text', view='bra') "
              "ou instale plotly para o grafo interativo.")
        return None

    cfg = analyze_control_flow(kernel)
    sites = cfg.branch_sites[:max_branches]
    if not sites:
        print("Nenhum site de BRA encontrado.")
        return
    blocks, order = build_cfg(kernel)
    display_map = _make_block_display_map(blocks, order)

    risk_color = {
        "none": "#64748b",
        "low": "#22c55e",
        "medium": "#f59e0b",
        "high": "#ef4444",
    }

    fig = go.Figure()
    y_values = list(range(len(sites)))[::-1]
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_label = []

    for idx, site in enumerate(sites):
        y = y_values[idx]
        branch_id = f"B{idx + 1}"
        total_mem = site.taken_memory_ops + site.fallthrough_memory_ops
        src = f"linha .cu {site.source_line}" if site.source_line > 0 else f"PTX:{site.line_no}"
        block_name = display_map.get(site.block_label, site.block_label)

        node_x.append(0.0)
        node_y.append(y)
        node_label.append(f"{branch_id}\n{site.divergence_risk}")
        node_color.append(risk_color.get(site.divergence_risk, "#64748b"))
        node_text.append(
            f"<b>{branch_id}</b><br>"
            f"Bloco simplificado: {block_name}<br>"
            f"Bloco PTX: {site.block_label}<br>"
            f"Tipo: {site.branch_kind}<br>"
            f"Origem: {src}<br>"
            f"Risco de divergência: {site.divergence_risk}<br>"
            f"Memória nos caminhos: {total_mem}<br>"
            f"<code>{site.raw}</code>"
        )

        for lane, (target_label, mem_ops, inst_count, edge_label, edge_color) in enumerate([
            (site.taken_target, site.taken_memory_ops, site.taken_instruction_count, "taken", "#f59e0b"),
            (site.fallthrough_target, site.fallthrough_memory_ops, site.fallthrough_instruction_count, "fall", "#94a3b8"),
        ]):
            if not target_label:
                continue
            tx = 0.95 + lane * 0.7
            ty = y + (0.16 if lane == 0 else -0.16)
            target_short = display_map.get(target_label, target_label)

            fig.add_trace(go.Scatter(
                x=[0.08, tx - 0.06],
                y=[y, ty],
                mode="lines",
                line=dict(color=edge_color, width=2),
                hoverinfo="skip",
                showlegend=False,
            ))

            fig.add_annotation(
                x=tx - 0.06,
                y=ty,
                ax=0.08,
                ay=y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.0,
                arrowwidth=2,
                arrowcolor=edge_color,
                text="",
            )

            node_x.append(tx)
            node_y.append(ty)
            node_label.append(
                f"{'T' if edge_label == 'taken' else 'F'}\nmem {mem_ops} | i {inst_count}"
            )
            node_color.append("#1e293b")
            node_text.append(
                f"<b>{edge_label}</b><br>"
                f"Destino simplificado: {target_short}<br>"
                f"Destino PTX: {target_label}<br>"
                f"Instruções no bloco: {inst_count}<br>"
                f"Mem ops no bloco: {mem_ops}"
            )

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_label,
        textposition="middle center",
        textfont=dict(size=10),
        marker=dict(
            size=22,
            color=node_color,
            line=dict(color="#cbd5e1", width=1),
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=node_text,
        showlegend=False,
    ))

    fig.add_annotation(x=0.0, y=len(sites) - 0.2, text="branch", showarrow=False, font=dict(size=11, color="#cbd5e1"))
    fig.add_annotation(x=0.95, y=len(sites) - 0.2, text="taken", showarrow=False, font=dict(size=11, color="#f59e0b"))
    fig.add_annotation(x=1.65, y=len(sites) - 0.2, text="fallthrough", showarrow=False, font=dict(size=11, color="#94a3b8"))

    fig.update_layout(
        title=f"Branches Simplificados — {kernel.name}",
        xaxis=dict(visible=False, range=[-0.25, 1.95]),
        yaxis=dict(visible=False, range=[-1, len(sites)]),
        height=max(500, 80 + len(sites) * 44),
        **_DARK_LAYOUT,
    )
    _show_fig(fig)
    return fig


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
    _show_fig(fig)
    return fig


def plot_instruction_roofline(kernels: Dict[str, "PTXKernel"],
                              peak_ipc: float = 128.0,
                              mem_ceiling: float = 32.0):
    """
    Roofline estático em espaço de instrução:
      X = instruction intensity (instruções / acessos de memória)
      Y = throughput estimado de instruções (proxy estático)
    """
    import math
    import plotly.graph_objects as go

    ridge = peak_ipc / max(mem_ceiling, 1e-6)
    xs = [0.1, ridge, ridge * 100]
    ys = [mem_ceiling * xs[0], peak_ipc, peak_ipc]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color="#94a3b8", width=2, dash="dash"),
        name="Instruction Roofline",
        hovertemplate="Roofline estático<extra></extra>",
    ))
    fig.add_vline(x=ridge, line=dict(color="#475569", dash="dot", width=1))
    fig.add_annotation(
        x=math.log10(ridge), y=0.95, xref="x", yref="paper",
        text=f"Ridge ≈ {ridge:.2f} instr/mem",
        showarrow=False, font=dict(color="#64748b", size=11),
        xanchor="left",
    )

    palette = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b",
               "#8b5cf6", "#ec4899", "#06b6d4", "#f97316"]

    for idx, (name, kernel) in enumerate(kernels.items()):
        ii = max(kernel.instruction_intensity, 0.1)
        throughput = min(peak_ipc, mem_ceiling * ii)
        region = "memory-bound" if ii < ridge else "compute-bound"
        fig.add_trace(go.Scatter(
            x=[ii], y=[throughput],
            mode="markers+text",
            marker=dict(size=14, color=palette[idx % len(palette)],
                        line=dict(color="white", width=1)),
            text=[name],
            textposition="top center",
            textfont=dict(size=11),
            name=name,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"Instruction intensity: {ii:.2f}<br>"
                f"Arithmetic ratio: {kernel.arithmetic_ratio:.1%}<br>"
                f"Throughput est.: {throughput:.1f}<br>"
                f"Região: {region}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Instruction Roofline — Espaço Estático de Instruções",
        xaxis=dict(title="Instruction intensity (instr / mem-access)",
                   type="log", gridcolor="#1e293b"),
        yaxis=dict(title="Instruction throughput estimado",
                   type="log", gridcolor="#1e293b"),
        **_DARK_LAYOUT,
    )
    _show_fig(fig)
    return fig


def _kmeans(points, k=3, iterations=20):
    import math

    if not points:
        return []
    k = max(1, min(k, len(points)))
    centers = [list(points[i]) for i in range(k)]
    labels = [0] * len(points)

    def dist2(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    for _ in range(iterations):
        changed = False
        for idx, point in enumerate(points):
            best = min(range(k), key=lambda c: dist2(point, centers[c]))
            if labels[idx] != best:
                labels[idx] = best
                changed = True

        new_centers = []
        for cid in range(k):
            members = [points[i] for i, lbl in enumerate(labels) if lbl == cid]
            if not members:
                new_centers.append(centers[cid])
                continue
            dims = len(members[0])
            new_centers.append([
                sum(p[d] for p in members) / len(members) for d in range(dims)
            ])
        centers = new_centers
        if not changed:
            break
    return labels


def plot_metric_space_pca(kernels: Dict[str, "PTXKernel"]):
    """
    PCA 2D dos kernels no espaço de métricas, com clustering simples.
    """
    import numpy as np
    import plotly.graph_objects as go

    names = list(kernels.keys())
    if len(names) < 2:
        print("PCA requer pelo menos 2 kernels.")
        return

    metric_names = [
        "total_instructions",
        "registers_per_thread",
        "branch_ratio",
        "instruction_intensity",
        "arithmetic_ratio",
        "shared_accesses",
        "local_accesses",
        "basic_block_count",
        "cfg_loop_count",
        "coalescing_est",
        "simd_utilization_est",
    ]
    data = np.array([
        [float(getattr(kernels[name], metric)) for metric in metric_names]
        for name in names
    ], dtype=float)

    means = data.mean(axis=0)
    stds = data.std(axis=0)
    stds[stds == 0] = 1.0
    z = (data - means) / stds

    cov = np.cov(z, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    coords = z @ eigvecs[:, :2]
    explained = eigvals / max(eigvals.sum(), 1e-9)
    labels = _kmeans(coords.tolist(), k=min(3, len(names)))

    palette = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6"]
    fig = go.Figure()
    for idx, name in enumerate(names):
        kernel = kernels[name]
        fig.add_trace(go.Scatter(
            x=[coords[idx, 0]], y=[coords[idx, 1]],
            mode="markers+text",
            marker=dict(size=14,
                        color=palette[labels[idx] % len(palette)],
                        line=dict(color="white", width=1)),
            text=[name],
            textposition="top center",
            name=f"cluster {labels[idx] + 1}",
            legendgroup=f"cluster-{labels[idx]}",
            showlegend=False,
            hovertemplate=(
                f"<b>{name}</b><br>"
                f"Cluster: {labels[idx] + 1}<br>"
                f"Instr: {kernel.total_instructions}<br>"
                f"Regs/thread: {kernel.registers_per_thread}<br>"
                f"Branch eff.: {kernel.branch_efficiency:.1%}<br>"
                f"II: {kernel.instruction_intensity:.2f}<extra></extra>"
            ),
        ))

    for cid in sorted(set(labels)):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=12, color=palette[cid % len(palette)]),
            name=f"Cluster {cid + 1}",
            showlegend=True,
        ))

    fig.update_layout(
        title="PCA / Clustering dos Sorts no Espaço de Métricas",
        xaxis_title=f"PC1 ({explained[0] * 100:.1f}% var.)",
        yaxis_title=f"PC2 ({explained[1] * 100:.1f}% var.)" if len(explained) > 1 else "PC2",
        **_DARK_LAYOUT,
    )
    _show_fig(fig)
    return fig


def plot_branch_efficiency_registers(kernels: Dict[str, "PTXKernel"]):
    """
    Barras comparando branch efficiency estimada e registradores por thread.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = list(kernels.keys())
    branch_eff = [k.branch_efficiency * 100 for k in kernels.values()]
    regs = [k.registers_per_thread for k in kernels.values()]

    fig = make_subplots(rows=1, cols=2, subplot_titles=(
        "Branch Efficiency Estimada",
        "Registradores por Thread",
    ))
    fig.add_trace(go.Bar(
        x=names, y=branch_eff, marker_color="#10b981",
        hovertemplate="<b>%{x}</b><br>Branch efficiency: %{y:.1f}%<extra></extra>",
        name="Branch efficiency",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=names, y=regs, marker_color="#f59e0b",
        hovertemplate="<b>%{x}</b><br>Regs/thread: %{y}<extra></extra>",
        name="Regs/thread",
    ), row=1, col=2)
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="Registradores", row=1, col=2)
    fig.update_layout(
        title="Branches e Pressão de Registradores por Kernel",
        showlegend=False,
        **_DARK_LAYOUT,
    )
    _show_fig(fig)
    return fig


def plot_memory_hierarchy(kernel: PTXKernel):
    """
    Diagrama estático da hierarquia de memória com métricas sobrepostas.
    """
    import plotly.graph_objects as go

    fig = go.Figure()
    levels = [
        ("Registers", kernel.registers_per_thread, "#10b981", 0, 0, 3.8, 0.9),
        ("Shared", kernel.shared_accesses, "#3b82f6", 0.3, -1.3, 3.2, 0.9),
        ("Local", kernel.local_accesses, "#8b5cf6", 0.6, -2.6, 2.6, 0.9),
        ("Global", kernel.global_loads + kernel.global_stores, "#ef4444", 0.9, -3.9, 2.0, 0.9),
    ]

    for label, value, color, x, y, w, h in levels:
        fig.add_shape(
            type="rect", x0=x, y0=y, x1=x + w, y1=y + h,
            fillcolor=color, opacity=0.25,
            line=dict(color=color, width=2),
        )
        fig.add_annotation(
            x=x + w / 2, y=y + h / 2,
            text=f"<b>{label}</b><br>{value}",
            showarrow=False,
            font=dict(size=14, color="#f1f5f9"),
        )

    fig.add_annotation(
        x=2.0, y=1.0,
        text=(
            f"<b>{kernel.name}</b><br>"
            f"bar.sync: {kernel.bar_sync_count}<br>"
            f"coalescência est.: {kernel.coalescing_est:.1%}<br>"
            f"SIMD est.: {kernel.simd_utilization_est:.1%}"
        ),
        showarrow=False,
        align="left",
        font=dict(size=12, color="#cbd5e1"),
        bordercolor="#334155",
        borderwidth=1,
        bgcolor="#111a1e",
    )

    fig.update_layout(
        title="Hierarquia de Memória com Métricas Sobrepostas",
        xaxis=dict(visible=False, range=[-0.2, 4.3]),
        yaxis=dict(visible=False, range=[-4.8, 1.7]),
        height=650,
        **_DARK_LAYOUT,
    )
    _show_fig(fig)
    return fig


def plot_runtime_curves(runtime_profile):
    """
    Curvas de tempo e sorting-rate por tamanho e distribuição de entrada.
    """
    import re
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    dist_re = re.compile(r"(?:dist|distribution)=([A-Za-z0-9_-]+)")
    grouped = {}
    for label, bench in runtime_profile.benchmarks.items():
        for sample in bench.samples:
            match = dist_re.search(sample.extra or "")
            dist = match.group(1) if match else "default"
            key = (label, dist)
            grouped.setdefault(key, []).append(sample)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Tempo por Tamanho", "Sorting Rate por Tamanho"),
    )
    palette = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b",
               "#8b5cf6", "#ec4899", "#06b6d4", "#f97316"]

    for idx, ((label, dist), samples) in enumerate(sorted(grouped.items())):
        by_n = {}
        for sample in samples:
            by_n.setdefault(sample.n, []).append(sample.milliseconds)
        xs = sorted(by_n)
        ys = [sum(by_n[n]) / len(by_n[n]) for n in xs]
        rates = [n / max(ms, 1e-9) for n, ms in zip(xs, ys)]
        color = palette[idx % len(palette)]
        name = f"{label} [{dist}]"

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=color), name=name,
            hovertemplate="<b>%{x}</b><br>tempo: %{y:.3f} ms<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=xs, y=rates, mode="lines+markers",
            line=dict(color=color), name=name, showlegend=False,
            hovertemplate="<b>%{x}</b><br>rate: %{y:.3f} elem/ms<extra></extra>",
        ), row=1, col=2)

    fig.update_xaxes(title_text="Tamanho de entrada (N)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Tamanho de entrada (N)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Tempo (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Sorting rate (elem/ms)", row=1, col=2)
    fig.update_layout(
        title="Curvas de Runtime por Tamanho e Distribuição",
        legend=dict(orientation="h", y=-0.18),
        **_DARK_LAYOUT,
    )
    _show_fig(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard de Eficiência de GPU
# ──────────────────────────────────────────────────────────────────────────────

_ARCH_PARAMS: Dict[str, tuple] = {
    # arch → (max_regs/SM, max_threads/SM, peak_flops_TFLOPS, peak_bw_GBs)
    "sm_75": (65536, 1024,  14.1, 448),   # RTX 2080 Ti
    "sm_80": (65536, 2048,  19.5, 2000),  # A100
    "sm_86": (65536, 1536,   8.1, 192),   # RTX 3050 / 3060
    "sm_89": (65536, 1536,  49.7, 576),   # RTX 4090 (Ada)
    "sm_90": (65536, 2048, 133.8, 3350),  # H100
}


def plot_gpu_efficiency(kernel: PTXKernel,
                        arch: str = "sm_86",
                        threads_per_block: int = 256):
    """
    Dashboard de eficiência de GPU — análise estática do PTX.

    6 gauges:
      • Ocupância estimada   — registradores/thread vs limite/SM
      • Risco de divergência — branches condicionais como % das instruções
      • Cobertura de grid    — usa padrão grid-stride (%nctaid)
      • Posição no Roofline  — intensidade aritmética vs ridge point
      • Eficiência de Warp   — inverso do risco de divergência
      • Score Geral          — média ponderada das 4 métricas

    Args:
        arch: arquitetura alvo (sm_75, sm_80, sm_86, sm_89, sm_90).
        threads_per_block: threads por bloco assumidos para estimativa de ocupância.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    max_regs_sm, max_threads_sm, def_flops, def_bw = _ARCH_PARAMS.get(
        arch, _ARCH_PARAMS["sm_86"]
    )
    ridge_point = def_flops * 1000 / def_bw   # FLOPs/Byte

    # ── 1. Ocupância por registradores ────────────────────────────────────────
    regs = kernel.total_registers
    if regs > 0:
        max_by_regs = (max_regs_sm // regs // 32) * 32   # múltiplo de warp
        active_threads = min(max_by_regs, max_threads_sm)
        occupancy_pct = round(active_threads / max_threads_sm * 100, 1)
        ideal_regs = max_regs_sm // max_threads_sm
    else:
        occupancy_pct, ideal_regs, active_threads = 0.0, 0, 0

    # ── 2. Risco de divergência ───────────────────────────────────────────────
    # branch_ratio de 25%+ → 100% de risco
    divergence_pct = round(min(kernel.branch_ratio * 400, 100), 1)

    # ── 3. Posição no Roofline ────────────────────────────────────────────────
    ai = kernel.arithmetic_intensity
    roofline_pct = round(min(ai / max(ridge_point, 0.001), 1.0) * 100, 1)
    if ai == 0:
        roofline_label = "sem dados"
    elif ai < ridge_point * 0.3:
        roofline_label = "fortemente memory-bound"
    elif ai < ridge_point:
        roofline_label = "memory-bound"
    else:
        roofline_label = "compute-bound"

    # ── 4. Cobertura de grid ──────────────────────────────────────────────────
    grid_pct = 100.0 if kernel.uses_nctaid else 0.0

    # ── Score geral ───────────────────────────────────────────────────────────
    warp_eff = 100 - divergence_pct
    overall = round((occupancy_pct + warp_eff + roofline_pct + grid_pct) / 4, 1)

    # ── Helper: gauge indicator ───────────────────────────────────────────────
    def _gauge(value, title, reverse=False):
        eff = (100 - value) if reverse else value
        bar_color = "#10b981" if eff >= 70 else ("#f59e0b" if eff >= 40 else "#ef4444")
        return go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 30, "color": "#e8eaed"}},
            title={"text": title, "font": {"size": 12, "color": "#94a3b8"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickfont": {"color": "#64748b", "size": 9},
                    "tickcolor": "#334155",
                },
                "bar": {"color": bar_color, "thickness": 0.25},
                "bgcolor": "#1e293b",
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 40],   "color": "#141b22"},
                    {"range": [40, 70],  "color": "#181f25"},
                    {"range": [70, 100], "color": "#152118"},
                ],
                "threshold": {
                    "line": {"color": "#475569", "width": 2},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        )

    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.06,
    )

    fig.add_trace(_gauge(occupancy_pct,
        f"Ocupância<br><sup>{regs} regs/thread · {arch}</sup>"),
        row=1, col=1)
    fig.add_trace(_gauge(divergence_pct,
        f"Risco de Divergência<br><sup>{kernel.predicated_branches} branches cond.</sup>",
        reverse=True),
        row=1, col=2)
    fig.add_trace(_gauge(grid_pct,
        "Cobertura de Grid<br><sup>usa %nctaid (grid-stride)</sup>"),
        row=1, col=3)
    fig.add_trace(_gauge(roofline_pct,
        f"Posição Roofline<br><sup>AI={ai:.2f} — {roofline_label}</sup>"),
        row=2, col=1)
    fig.add_trace(_gauge(warp_eff,
        f"Eficiência de Warp<br><sup>branch ratio={kernel.branch_ratio:.1%}</sup>"),
        row=2, col=2)
    fig.add_trace(_gauge(overall,
        "<b>Score Geral</b><br><sup>média das 4 métricas</sup>"),
        row=2, col=3)

    # ── Recomendações ─────────────────────────────────────────────────────────
    recs = []
    if occupancy_pct < 50 and regs > 0:
        recs.append(
            f"▲ Ocupância baixa: {regs} regs/thread → "
            f"{active_threads} threads/SM ativo. "
            f"Ideal: ≤ {ideal_regs} regs para 100%."
        )
    if divergence_pct > 30:
        recs.append(
            f"▲ Divergência: {kernel.predicated_branches} branches cond. "
            f"({kernel.branch_ratio:.1%} das instruções). "
            "Threads do mesmo warp tomam caminhos diferentes → execução serializada."
        )
    if ai < ridge_point * 0.3:
        recs.append(
            f"▲ Memory-bound: AI={ai:.2f} vs ridge≈{ridge_point:.0f} FLOPs/B. "
            "Aumente reutilização via shared memory ou loop tiling."
        )
    if not kernel.uses_nctaid:
        recs.append(
            "▲ Sem grid-stride: kernel não escala além de gridDim×blockDim. "
            "Adicione loop com stride de grid."
        )
    if kernel.local_accesses > 0:
        recs.append(
            f"▲ Register spilling: {kernel.local_accesses} acessos ld/st.local "
            "(velocidade ≈ global memory)."
        )
    if not recs:
        recs.append("✓ Nenhum problema crítico detectado via análise estática.")

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Eficiência de GPU — {kernel.name}</b><br>"
                f"<sup>Análise estática PTX · {arch} · "
                f"{threads_per_block} threads/bloco · "
                f"Peak {def_flops} TFLOPS / {def_bw} GB/s</sup>"
            ),
            font=dict(color="#e8eaed", size=15),
            x=0.5,
        ),
        paper_bgcolor="#0f1416",
        font=dict(color="#cbd5e1"),
        height=640,
        margin=dict(l=20, r=20, t=110, b=180),
        annotations=[dict(
            x=0.5, y=-0.04,
            xref="paper", yref="paper",
            text=(
                "<b style='color:#64748b'>Recomendações:</b><br>"
                + "<br>".join(
                    f"<span style='color:#94a3b8'>{r}</span>" for r in recs
                )
            ),
            showarrow=False,
            font=dict(size=11),
            align="left",
            bgcolor="#111827",
            bordercolor="#1e293b",
            borderwidth=1,
            borderpad=12,
            xanchor="center",
        )],
    )

    _show_fig(fig)
    return fig
