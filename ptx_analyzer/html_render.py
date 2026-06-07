"""
Funções de renderização HTML para exibição no Jupyter/Colab.
"""

from .core import PTXKernel, CATEGORY_COLORS
from .heuristics import run_heuristics, LEVEL_ICONS, LEVEL_COLORS, LEVEL_TEXT_COLORS

# ──────────────────────────────────────────────────────────────────────────────
# 6. Renderização HTML (para o Colab)
# ──────────────────────────────────────────────────────────────────────────────

def _metric_card(label: str, value, subtitle: str = "") -> str:
    return f"""
    <div style="background:#1e293b;border-radius:8px;padding:14px 18px;
                min-width:120px;flex:1;">
      <div style="font-size:11px;color:#94a3b8;letter-spacing:.05em;
                  text-transform:uppercase;">{label}</div>
      <div style="font-size:26px;font-weight:700;color:#f1f5f9;
                  margin:4px 0;">{value}</div>
      <div style="font-size:11px;color:#64748b;">{subtitle}</div>
    </div>"""


def _render_overview_html(kernel: PTXKernel) -> str:
    cc = kernel.category_counts
    cards = [
        _metric_card("Instruções",    kernel.total_instructions,   "total no kernel"),
        _metric_card("Registradores", kernel.total_registers,       "declarados"),
        _metric_card("ld.global",     kernel.global_loads,          "cargas globais"),
        _metric_card("st.global",     kernel.global_stores,         "escritas globais"),
        _metric_card("FMA",           kernel.fma_count,             "fused mul-add"),
        _metric_card("Branches",      kernel.predicated_branches,   "predicados"),
        _metric_card("Atômicas",      kernel.atomics,               "atom/red"),
        _metric_card("Int.Arit.",     kernel.arithmetic_intensity,  "FLOPs/mem-op"),
        _metric_card("shfl.sync",     kernel.shfl_count,            "warp shuffle"),
    ]
    row = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px;">'
    row += "".join(cards) + "</div>"
    return row


def _render_instructions_html(kernel: PTXKernel,
                               filter_cat: str = "all",
                               search: str = "") -> str:
    rows = []
    for instr in kernel.instructions:
        if filter_cat != "all" and instr.category != filter_cat:
            continue
        if search and search.lower() not in instr.raw.lower():
            continue

        badge_bg = CATEGORY_COLORS.get(instr.category, "#374151")
        pred_badge = (f'<span style="background:#7c3aed;padding:1px 5px;'
                      f'border-radius:3px;font-size:11px;">{instr.predicate}</span> '
                      if instr.is_predicated else "")

        rows.append(
            f'<tr style="border-bottom:1px solid #1e293b;">'
            f'<td style="padding:4px 8px;color:#64748b;font-size:12px;">'
            f'{instr.line_no}</td>'
            f'<td style="padding:4px 8px;">{pred_badge}'
            f'<code style="color:#e2e8f0;">{instr.op}</code></td>'
            f'<td style="padding:4px 8px;color:#94a3b8;font-size:12px;">'
            f'{", ".join(instr.operands[:4])}'
            f'{"..." if len(instr.operands) > 4 else ""}</td>'
            f'<td style="padding:4px 8px;">'
            f'<span style="background:{badge_bg};padding:2px 7px;'
            f'border-radius:10px;font-size:11px;color:#fff;">'
            f'{instr.category}</span></td>'
            f'</tr>'
        )

    if not rows:
        return "<p style='color:#64748b;'>Nenhuma instrução encontrada.</p>"

    return (
        f'<div style="max-height:420px;overflow-y:auto;">'
        f'<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        f'<thead><tr style="background:#1e293b;">'
        f'<th style="padding:6px 8px;text-align:left;color:#94a3b8;">Linha</th>'
        f'<th style="padding:6px 8px;text-align:left;color:#94a3b8;">Instrução</th>'
        f'<th style="padding:6px 8px;text-align:left;color:#94a3b8;">Operandos</th>'
        f'<th style="padding:6px 8px;text-align:left;color:#94a3b8;">Categoria</th>'
        f'</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
    )


def _render_registers_html(kernel: PTXKernel) -> str:
    if not kernel.reg_decls:
        return "<p style='color:#64748b;'>Nenhuma declaração de registrador encontrada.</p>"

    palette = ["#3b82f6", "#8b5cf6", "#10b981", "#f59e0b",
               "#ef4444", "#06b6d4", "#ec4899", "#f97316"]
    html = '<div style="display:flex;flex-direction:column;gap:12px;">'
    for idx, (rtype, regs) in enumerate(sorted(kernel.reg_decls.items())):
        color = palette[idx % len(palette)]
        tags = " ".join(
            f'<span style="background:{color}22;border:1px solid {color}44;'
            f'padding:2px 6px;border-radius:4px;font-size:11px;'
            f'color:{color};margin:2px;">{r}</span>'
            for r in sorted(regs)[:40]
        )
        overflow = f" <em style='color:#64748b;'>+{len(regs)-40} mais…</em>" if len(regs) > 40 else ""
        html += (
            f'<div><div style="font-size:12px;color:{color};margin-bottom:4px;">'
            f'.{rtype} — {len(regs)} registrador(es)</div>'
            f'<div style="display:flex;flex-wrap:wrap;">{tags}{overflow}</div></div>'
        )
    html += "</div>"
    return html


def _render_diagnostics_html(kernel: PTXKernel) -> str:
    results = run_heuristics(kernel)
    items = []
    for level, msg in results:
        icon = LEVEL_ICONS.get(level, "•")
        bg   = LEVEL_COLORS.get(level, "#1e293b")
        tc   = LEVEL_TEXT_COLORS.get(level, "#e2e8f0")
        items.append(
            f'<div style="background:{bg};border-radius:8px;padding:10px 14px;'
            f'margin-bottom:8px;font-size:13px;">'
            f'<span style="margin-right:8px;">{icon}</span>'
            f'<span style="color:{tc};">{msg}</span></div>'
        )
    return "".join(items) if items else "<p style='color:#64748b;'>Sem diagnósticos.</p>"
