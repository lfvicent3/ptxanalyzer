"""
Vista lado a lado: código .cu ↔ PTX correspondente.
"""

import re
from collections import defaultdict
from typing import Dict, List
from .core import PTXKernel, PTXInstruction, CATEGORY_COLORS
from .parser import parse_ptx
from .output import emit_text

# ──────────────────────────────────────────────────────────────────────────────
# 10. PTXSourceView
# ──────────────────────────────────────────────────────────────────────────────

# Palavras-chave C/CUDA para coloração mínima do código-fonte
_CU_KEYWORDS = {
    "__global__", "__device__", "__shared__", "__host__",
    "__forceinline__", "__restrict__",
    "if", "else", "for", "while", "do", "return", "int", "float",
    "double", "void", "const", "unsigned", "char", "short", "long",
    "half", "extern", "static", "inline",
}

def _highlight_cu_line(src: str) -> str:
    """Coloração sintática mínima de uma linha C/CUDA para HTML."""
    import html
    escaped = html.escape(src)
    for kw in _CU_KEYWORDS:
        escaped = re.sub(
            rf'\b({re.escape(kw)})\b',
            r'<span style="color:#79c0ff;">\1</span>',
            escaped,
        )
    escaped = re.sub(
        r'(&quot;.*?&quot;)',
        r'<span style="color:#a5d6ff;">\1</span>',
        escaped,
    )
    escaped = re.sub(
        r'\b(\d+\.?\d*[fFuU]?)\b',
        r'<span style="color:#f2cc60;">\1</span>',
        escaped,
    )
    escaped = re.sub(
        r'(//.*)',
        r'<span style="color:#8b949e;font-style:italic;">\1</span>',
        escaped,
    )
    return escaped


class PTXSourceView:
    """
    Vista lado a lado: código-fonte .cu ↔ instruções PTX correspondentes.

    Requer PTX gerado com -lineinfo para ter as diretivas .loc:
        nvcc -ptx -lineinfo kernel.cu -arch=sm_XX -o kernel.ptx

    Uso:
        view = PTXSourceView.from_files("kernel.cu", "kernel.ptx")
        view.show()          # interface ipywidgets interativa
        view.show_html()     # HTML estático (sem ipywidgets)
        view.show_stats()    # resumo de otimizações no terminal
    """

    def __init__(self, cu_source: str, kernel: PTXKernel):
        self.cu_lines  = cu_source.splitlines()
        self.kernel    = kernel
        self.has_lineinfo = any(i.source_line > 0 for i in kernel.instructions)

        self._by_line: Dict[int, List[PTXInstruction]] = defaultdict(list)
        # _by_col[line][col] → lista de instruções naquela coluna
        self._by_col: Dict[int, Dict[int, List[PTXInstruction]]] = defaultdict(
            lambda: defaultdict(list))
        for instr in kernel.instructions:
            if instr.source_line > 0:
                self._by_line[instr.source_line].append(instr)
                if instr.source_col > 0:
                    self._by_col[instr.source_line][instr.source_col].append(instr)

        self._non_code_lines = self._compute_non_code_lines()
        _mapped = list(self._by_line.keys())
        self._min_mapped = min(_mapped) if _mapped else 0
        self._max_mapped = max(_mapped) if _mapped else 0

    # ── construtores alternativos ────────────────────────────────────────────

    @classmethod
    def from_files(cls, cu_path: str, ptx_path: str,
                   kernel_index: int = 0) -> "PTXSourceView":
        """
        Carrega o .cu e o .ptx e retorna uma PTXSourceView.
        """
        with open(cu_path,  "r", encoding="utf-8", errors="replace") as f:
            cu = f.read()
        with open(ptx_path, "r", encoding="utf-8", errors="replace") as f:
            ptx = f.read()
        kernels = parse_ptx(ptx)
        if not kernels:
            raise ValueError("Nenhum kernel encontrado no PTX.")
        idx = min(kernel_index, len(kernels) - 1)
        return cls(cu, kernels[idx])

    @classmethod
    def from_file(cls, path: str, kernel_index: int = 0, arch: str = "sm_75") -> "PTXSourceView":
        """
        Recebe um arquivo .cu, compila com -lineinfo automaticamente e abre a vista.
        """
        import os
        import subprocess

        if not path.endswith(".cu"):
            raise ValueError("from_file() na PTXSourceView exige um arquivo .cu")
        
        out_ptx = path.replace(".cu", ".ptx")
        cmd = ["nvcc", "-ptx", "-lineinfo", path, f"-arch={arch}", "-o", out_ptx]
        print(f"Compilando CUDA para PTX (com -lineinfo): {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Erro ao compilar {path}:\n{res.stderr}")
        
        return cls.from_files(path, out_ptx, kernel_index)

    @classmethod
    def from_analyzer(cls, cu_path: str,
                      analyzer: "PTXAnalyzer") -> "PTXSourceView": # type: ignore
        with open(cu_path, "r", encoding="utf-8", errors="replace") as f:
            cu = f.read()
        return cls(cu, analyzer.kernel)

    # ── pré-processamento de linhas não-executáveis ──────────────────────────

    def _compute_non_code_lines(self) -> set:
        """
        Detecta linhas que não geram código executável GPU:
        comentários, directivas de pré-processador, tokens estruturais puros
        e linhas de continuação dentro de listas de parâmetros.
        """
        non_code: set = set()
        in_block  = False
        open_pars = 0

        for i, raw in enumerate(self.cu_lines, 1):
            s = raw.strip()

            if not s:
                non_code.add(i)
                continue

            # Dentro de bloco /* ... */
            if in_block:
                non_code.add(i)
                if "*/" in s:
                    in_block = False
                continue

            # Início de bloco de comentário
            if s.startswith("/*"):
                non_code.add(i)
                if "*/" not in s[2:]:
                    in_block = True
                continue

            # Comentário de linha
            if s.startswith("//"):
                non_code.add(i)
                continue

            # Directiva de pré-processador (#include, #define, #pragma …)
            if s.startswith("#"):
                non_code.add(i)
                code = s.split("//")[0]
                open_pars = max(0, open_pars + code.count("(") - code.count(")"))
                continue

            # Tokens estruturais puros (chaves / parêntese de fechamento isolado)
            if s in ("{", "}", "};", ");", ")"):
                non_code.add(i)
                open_pars = max(0, open_pars + s.count("(") - s.count(")"))
                continue

            # Continuação de expressão multi-linha (parâmetros de função, etc.)
            if open_pars > 0:
                non_code.add(i)
                code = s.split("//")[0]
                open_pars = max(0, open_pars + code.count("(") - code.count(")"))
                continue

            # Linha de declaração de função com qualificadores CUDA
            # (a assinatura em si não gera instruções PTX)
            if any(q in s for q in ("__global__", "__device__",
                                     "__host__", "__forceinline__")):
                non_code.add(i)
                code = s.split("//")[0]
                open_pars = max(0, open_pars + code.count("(") - code.count(")"))
                continue

            # Linha de código normal — apenas rastrear parênteses
            code = s.split("//")[0]
            open_pars = max(0, open_pars + code.count("(") - code.count(")"))

        return non_code

    # ── classificação de linhas ──────────────────────────────────────────────

    def _classify_line(self, lineno: int) -> str:
        """
        Classifica uma linha do .cu:
          blank     — comentário, linha vazia, directiva ou token estrutural
          inactive  — fora do intervalo do kernel (código de host, includes …)
          normal    — código com PTX gerado
          optimized — código executável sem PTX (eliminado pelo compilador)
          fma       — fusão mul+add detectada
          heavy     — ≥ 6 instruções PTX geradas
        """
        if lineno in self._non_code_lines:
            return "blank"

        instrs = self._by_line.get(lineno, [])

        if self._min_mapped > 0 and not (self._min_mapped <= lineno <= self._max_mapped):
            return "inactive"

        if not instrs:
            return "optimized"

        cats = {i.op_base for i in instrs}
        if any(i.op_base == "fma" for i in instrs):
            return "fma"
        if len(instrs) >= 6:
            return "heavy"
        return "normal"

    def show_stats(self):
        """Imprime resumo de otimizações detectadas pelo mapeamento .loc."""
        return self.render(mode="html", view="stats")

    def _format_stats_text(self) -> str:
        if not self.has_lineinfo:
            return "⚠  PTX sem informação de linha (.loc ausente).\n   Recompile com:  nvcc -ptx -lineinfo  ...\n"

        n_total = len(self.cu_lines)
        n_mapped = len(self._by_line)
        classes = [self._classify_line(ln) for ln in range(1, n_total + 1)]
        n_optim = classes.count("optimized")
        n_fma = classes.count("fma")
        n_heavy = classes.count("heavy")

        lines = [
            "",
            "═" * 54,
            f"  {self.kernel.name}",
            "─" * 54,
            f"  Linhas no .cu              : {n_total}",
            f"  Linhas com PTX mapeado     : {n_mapped}",
            f"  Linhas eliminadas (✂)      : {n_optim}" + ("  ← dead code / const fold" if n_optim else ""),
            f"  Linhas com FMA (⚡)         : {n_fma}",
            f"  Linhas pesadas ≥6 PTX (📦) : {n_heavy}",
            "═" * 54,
            "",
        ]
        eliminated = [ln for ln, c in enumerate(classes, 1) if c == "optimized"]
        if eliminated:
            lines.append("  Linhas eliminadas pelo compilador:")
            for ln in eliminated[:15]:
                src = self.cu_lines[ln - 1].strip()
                lines.append(f"    L{ln:>4}: {src[:72]}")
            if len(eliminated) > 15:
                lines.append(f"    ... e mais {len(eliminated)-15}")
            lines.append("")
        return "\n".join(lines)

    def _format_mapping_text(self, show_only_mapped: bool = False) -> str:
        if not self.has_lineinfo:
            return "⚠  PTX sem informação de linha (.loc ausente).\n   Recompile com:  nvcc -ptx -lineinfo  ...\n"

        lines = [
            "",
            "═" * 80,
            f"  Mapeamento .cu ↔ PTX  |  Kernel: {self.kernel.name}",
            "═" * 80,
            "",
        ]
        for ln, raw_src in enumerate(self.cu_lines, 1):
            instrs = self._by_line.get(ln, [])
            cls = self._classify_line(ln)

            if show_only_mapped and not instrs:
                continue
            if not show_only_mapped and cls in ("blank", "inactive") and not raw_src.strip():
                continue

            badge = ""
            if cls == "optimized":
                badge = "[✂ ELIMINADA]"
            elif cls == "fma":
                badge = "[⚡ FMA]"
            elif cls == "heavy":
                badge = "[📦 PESADA]"

            src_str = raw_src.strip()
            if len(src_str) > 60:
                src_str = src_str[:57] + "..."

            if not instrs and not badge and cls not in ("blank", "inactive"):
                lines.append(f"L{ln:<4} | {src_str}")
            elif instrs or badge:
                lines.append(f"L{ln:<4} | {src_str:<60} {badge}")
                for i in instrs:
                    pred = f"@{i.predicate} " if i.is_predicated else ""
                    ops = ", ".join(i.operands)
                    lines.append(f"       ↳ {pred}{i.op} {ops}")
                lines.append("─" * 80)
        return "\n".join(lines)

    def show_text(self, show_only_mapped: bool = False):
        """
        Imprime o mapeamento .cu ↔ PTX em formato texto puro (ASCII).
        Ideal para ambientes onde HTML/ipywidgets não estão disponíveis.
        """
        return self.render(mode="text", view="mapping", show_only_mapped=show_only_mapped)

    # ── renderização HTML ────────────────────────────────────────────────────

    # (bg_linha, cor_borda_esq, cor_texto_fonte)
    _CLS_STYLE = {
        "blank":    ("transparent",             "transparent", "#6e7681"),
        "inactive": ("transparent",             "transparent", "#3d444d"),
        "normal":   ("transparent",             "transparent", "#e6edf3"),
        "optimized":("rgba(248,81,73,0.10)",   "#f85149",     "#e6edf3"),
        "fma":      ("rgba(63,185,80,0.10)",   "#3fb950",     "#e6edf3"),
        "heavy":    ("rgba(88,166,255,0.10)",  "#58a6ff",     "#e6edf3"),
    }

    _BADGE_HTML_MAP = {
        "optimized": ('<span style="background:rgba(248,81,73,0.2);color:#f85149;'
                      'padding:0 6px;border-radius:4px;font-size:10px;'
                      'margin-left:8px;vertical-align:middle;">✂ eliminada</span>'),
        "fma":       ('<span style="background:rgba(63,185,80,0.2);color:#3fb950;'
                      'padding:0 6px;border-radius:4px;font-size:10px;'
                      'margin-left:8px;vertical-align:middle;">⚡ FMA</span>'),
        "heavy":     ('<span style="background:rgba(88,166,255,0.2);color:#58a6ff;'
                      'padding:0 6px;border-radius:4px;font-size:10px;'
                      'margin-left:8px;vertical-align:middle;">📦 pesado</span>'),
    }

    def _render_ptx_cell(self, instrs: List[PTXInstruction],
                          uid: str = "") -> str:
        """Renderiza uma célula de instruções PTX com data-iid para hover."""
        import html
        if not instrs:
            return ""
        rows = []
        for i in instrs:
            color     = CATEGORY_COLORS.get(i.category, "#8b949e")
            pred_html = (f'<span style="color:#7c3aed;margin-right:3px;">'
                         f'{i.predicate}</span>' if i.is_predicated else "")
            ops = ", ".join(i.operands[:3])
            if len(i.operands) > 3:
                ops += "…"
            iid = (f"{uid}L{i.source_line}C{i.source_col}"
                   if uid and i.source_col > 0
                   else (f"{uid}L{i.source_line}" if uid and i.source_line > 0 else ""))
            data = f' data-iid="{iid}" class="ptxsv-instr"' if iid else ""
            rows.append(
                f'<div{data} style="padding:1px 0;line-height:1.5;'
                f'border-radius:3px;cursor:crosshair;">'
                f'{pred_html}'
                f'<code style="color:{color};font-size:11px;background:transparent;">'
                f'{i.op}</code>'
                f'<span style="color:#484f58;font-size:10px;margin-left:5px;">'
                f'{html.escape(ops)}</span>'
                f'</div>'
            )
        return "".join(rows)

    def _build_html(self, filter_cls: str = "all") -> str:
        import html as _html

        if not self.has_lineinfo:
            return (
                '<div style="background:#161b22;padding:16px;border-radius:8px;'
                'border:1px solid #f85149;color:#f85149;'
                'font-family:ui-monospace,monospace;">'
                '⚠ PTX sem diretivas <code>.loc</code> — recompile com '
                '<code>nvcc -ptx <b>-lineinfo</b> ...</code></div>'
            )

        n_total  = len(self.cu_lines)
        n_mapped = len(self._by_line)
        classes  = [self._classify_line(ln) for ln in range(1, n_total + 1)]
        n_optim  = classes.count("optimized")
        n_fma    = classes.count("fma")
        n_heavy  = classes.count("heavy")

        def _pill(txt, color, bg):
            return (f'<span style="background:{bg};color:{color};padding:2px 9px;'
                    f'border-radius:10px;font-size:11px;">{txt}</span>')

        pills = []
        if n_optim: pills.append(_pill(f"✂ {n_optim} eliminadas",
                                        "#f85149", "rgba(248,81,73,0.15)"))
        if n_fma:   pills.append(_pill(f"⚡ {n_fma} FMA",
                                        "#3fb950", "rgba(63,185,80,0.15)"))
        if n_heavy: pills.append(_pill(f"📦 {n_heavy} pesadas",
                                        "#58a6ff", "rgba(88,166,255,0.15)"))

        stats_bar = (
            '<div style="background:#161b22;padding:10px 14px;'
            'border-bottom:1px solid #21262d;display:flex;flex-wrap:wrap;'
            'gap:8px;align-items:center;">'
            f'<span style="color:#e6edf3;font-weight:600;font-size:13px;">'
            f'{_html.escape(self.kernel.name)}</span>'
            '<span style="color:#484f58;font-size:12px;">·</span>'
            f'<span style="color:#8b949e;font-size:12px;">'
            f'{self.kernel.total_instructions} instrs PTX'
            f' · {n_mapped}/{n_total} linhas mapeadas</span>'
            '<span style="flex:1"></span>'
            + " ".join(pills) +
            '</div>'
        )

        header = (
            '<tr style="background:#161b22;position:sticky;top:0;z-index:5;">'
            '<th style="width:50px;padding:6px 8px;text-align:right;'
            'color:#484f58;font-size:10px;font-weight:500;'
            'border-right:1px solid #21262d;border-bottom:2px solid #21262d;">Nº</th>'
            '<th style="padding:6px 12px;text-align:left;color:#484f58;'
            'font-size:10px;font-weight:500;'
            'border-right:1px solid #21262d;border-bottom:2px solid #21262d;">'
            'Código-fonte (.cu)</th>'
            '<th style="width:50px;padding:6px 6px;text-align:center;'
            'color:#484f58;font-size:10px;font-weight:500;'
            'border-right:1px solid #21262d;border-bottom:2px solid #21262d;">nPTX</th>'
            '<th style="padding:6px 12px;text-align:left;color:#484f58;'
            'font-size:10px;font-weight:500;border-bottom:2px solid #21262d;">'
            'Instruções PTX</th>'
            '</tr>'
        )

        import uuid as _uuid
        uid = _uuid.uuid4().hex[:8] + "_"   # prefixo único por instância

        rows = []
        for ln, raw_src in enumerate(self.cu_lines, 1):
            cls    = classes[ln - 1]
            instrs = self._by_line.get(ln, [])

            if filter_cls == "optimized" and cls != "optimized": continue
            if filter_cls == "fma"       and cls != "fma":       continue
            if filter_cls == "heavy"     and cls != "heavy":     continue
            if filter_cls == "mapped"    and not instrs:         continue
            if filter_cls == "code"      and cls in ("blank", "inactive"): continue

            row_bg, border_l, text_color = self._CLS_STYLE.get(
                cls, ("transparent", "transparent", "#e6edf3"))
            bl_css = (f"border-left:3px solid {border_l};"
                      if border_l != "transparent"
                      else "border-left:3px solid transparent;")
            badge = self._BADGE_HTML_MAP.get(cls, "")

            n_ptx = len(instrs)
            if n_ptx == 0:
                cnt = '<span style="color:#30363d;font-size:11px;">—</span>'
            elif n_ptx >= 6:
                cnt = (f'<span style="color:#f85149;font-size:11px;'
                       f'font-weight:700;">{n_ptx}</span>')
            elif n_ptx >= 3:
                cnt = (f'<span style="color:#f0883e;font-size:11px;'
                       f'font-weight:600;">{n_ptx}</span>')
            else:
                cnt = f'<span style="color:#8b949e;font-size:11px;">{n_ptx}</span>'

            # ── Segmentação da linha fonte por coluna ──────────────────────
            col_map = self._by_col.get(ln, {})   # col → instrs
            base_color = text_color if cls in ("blank", "inactive") else "#e6edf3"
            span_style = (f"font-size:12px;white-space:pre;"
                          f"color:{base_color};font-family:inherit;")

            if col_map and cls not in ("blank", "inactive"):
                # Temos informação de coluna → criar spans por segmento
                sorted_cols = sorted(col_map.keys())
                # Calcular ranges: [col_i, col_{i+1}) para cada grupo
                ranges = []
                for idx, col in enumerate(sorted_cols):
                    end = sorted_cols[idx + 1] if idx + 1 < len(sorted_cols) else len(raw_src)
                    ranges.append((col, end, col_map[col]))
                # Fragmento antes da primeira coluna
                first_col = sorted_cols[0]
                pre = raw_src[:first_col - 1] if first_col > 1 else ""

                src_parts = []
                if pre:
                    src_parts.append(
                        f'<span style="{span_style}">{_highlight_cu_line(pre)}</span>')
                for col_start, col_end, seg_instrs in ranges:
                    seg_text = raw_src[col_start - 1: col_end - 1]
                    iid      = f"{uid}L{ln}C{col_start}"
                    src_parts.append(
                        f'<span data-iid="{iid}" class="ptxsv-seg" '
                        f'style="{span_style}border-radius:3px;cursor:crosshair;">'
                        f'{_highlight_cu_line(seg_text)}</span>'
                    )
                src_html = "".join(src_parts)
            else:
                # Sem coluna ou linha não-executável → linha inteira como um span
                if cls in ("blank", "inactive"):
                    src_html = (f'<span style="{span_style}">'
                                f'{_html.escape(raw_src)}</span>')
                else:
                    iid      = f"{uid}L{ln}"
                    src_html = (f'<span data-iid="{iid}" class="ptxsv-seg" '
                                f'style="{span_style}border-radius:3px;cursor:crosshair;">'
                                f'{_highlight_cu_line(raw_src)}</span>')

            rows.append(
                f'<tr class="ptxsv-row" style="background:{row_bg};{bl_css}'
                f'border-bottom:1px solid #0d1117;">'
                f'<td style="padding:2px 8px;text-align:right;color:#484f58;'
                f'font-size:11px;vertical-align:top;white-space:nowrap;'
                f'background:#0d1117;border-right:1px solid #21262d;'
                f'user-select:none;">{ln}</td>'
                f'<td class="ptxsv-src" style="padding:2px 12px;vertical-align:top;'
                f'border-right:1px solid #21262d;transition:background .1s;">'
                f'{src_html}{badge}</td>'
                f'<td style="padding:2px 6px;text-align:center;vertical-align:top;'
                f'border-right:1px solid #21262d;">{cnt}</td>'
                f'<td class="ptxsv-ptx" style="padding:2px 12px;vertical-align:top;'
                f'transition:background .1s;">'
                f'{self._render_ptx_cell(instrs, uid=uid)}</td>'
                f'</tr>'
            )

        cid  = uid.rstrip("_")   # ID do container sem o "_"
        css_reset = (
            '<style>'
            '.ptxsv code{background:transparent!important;padding:0!important;'
            'border-radius:0!important;border:none!important;font-size:inherit!important;}'
            # hover coluna-a-coluna: segmento fonte ↔ instrução PTX
            '.ptxsv .ptxsv-seg.ptxsv-hl{'
            'background:rgba(255,220,50,0.25)!important;'
            'outline:1px solid rgba(255,220,50,0.6);}'
            '.ptxsv .ptxsv-instr.ptxsv-hl{'
            'background:rgba(255,220,50,0.18)!important;'
            'outline:1px solid rgba(255,220,50,0.5);border-radius:3px;}'
            # hover geral na linha (quando não há col info)
            '.ptxsv .ptxsv-src:hover{background:rgba(88,166,255,0.06)!important;}'
            '.ptxsv .ptxsv-src:hover~.ptxsv-ptx{background:rgba(88,166,255,0.12)!important;'
            'box-shadow:inset 3px 0 0 #58a6ff;}'
            '.ptxsv tr:has(.ptxsv-ptx:hover) .ptxsv-src{'
            'background:rgba(63,185,80,0.06)!important;'
            'box-shadow:inset -3px 0 0 #3fb950;}'
            '.ptxsv .ptxsv-ptx:hover{background:rgba(63,185,80,0.1)!important;}'
            '.ptxsv tr.ptxsv-row:hover{filter:brightness(1.05);}'
            '</style>'
        )

        # JavaScript: hover em segmento fonte ↔ instrução PTX com mesmo data-iid
        js_block = f"""
<script>
(function(){{
  var root = document.getElementById('ptxsv-{cid}');
  if (!root) return;
  function clear(){{
    root.querySelectorAll('.ptxsv-hl').forEach(function(e){{e.classList.remove('ptxsv-hl');}});
  }}
  root.querySelectorAll('[data-iid]').forEach(function(el){{
    el.addEventListener('mouseenter', function(){{
      clear();
      var id = this.dataset.iid;
      root.querySelectorAll('[data-iid="'+id+'"]').forEach(function(e){{
        e.classList.add('ptxsv-hl');
      }});
    }});
    el.addEventListener('mouseleave', clear);
  }});
}})();
</script>"""

        container_id = f'ptxsv-{cid}'
        return (
            css_reset +
            f'<div class="ptxsv" id="{container_id}" '
            f'style="background:#0d1117;border:1px solid #21262d;'
            f'border-radius:8px;overflow:hidden;font-family:'
            f'ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;">'
            + stats_bar +
            '<div style="max-height:560px;overflow-y:auto;">'
            '<table style="width:100%;border-collapse:collapse;">'
            '<thead>' + header + '</thead>'
            '<tbody>' + "".join(rows) + '</tbody>'
            '</table></div></div>'
            + js_block
        )

    def show_html(self, show_only_mapped: bool = False):
        """Exibe HTML estático (sem ipywidgets)."""
        return self.render(mode="html", view="mapping", show_only_mapped=show_only_mapped)

    def show(self):
        """
        Interface interativa com ipywidgets: barra de filtros, legenda e
        visualização lado a lado código-fonte ↔ PTX com tema GitHub dark.
        """
        return self.render(mode="widget", view="mapping")

    def render(self,
               mode: str = "text",
               view: str = "mapping",
               show_only_mapped: bool = False):
        """
        Interface unificada para o mapeamento fonte ↔ PTX.

        Args:
            mode:
                - "text": ASCII
                - "html": HTML estático
                - "widget": UI interativa ipywidgets
                - "raw": retorna string textual
            view:
                - "mapping": lado a lado .cu ↔ PTX
                - "stats": resumo das linhas mapeadas/otimizadas
        """
        mode = mode.lower()
        view = view.lower()

        if mode in ("text", "raw"):
            if view == "stats":
                return emit_text(self._format_stats_text(), mode=mode)
            return emit_text(self._format_mapping_text(show_only_mapped=show_only_mapped), mode=mode)

        if mode == "html":
            if view == "stats":
                return emit_text(self._format_stats_text(), mode="html")
            try:
                from IPython.display import HTML, display
                display(HTML(self._build_html("mapped" if show_only_mapped else "all")))
                return None
            except ImportError:
                return emit_text(self._format_mapping_text(show_only_mapped=show_only_mapped), mode="text")

        if mode != "widget":
            raise ValueError(f"Modo desconhecido: {mode}")

        if not self.has_lineinfo:
            print("⚠  PTX sem diretivas .loc.")
            print("   Recompile com:  nvcc -ptx -lineinfo kernel.cu -arch=sm_XX ...")
            return

        try:
            import ipywidgets as w
            from IPython.display import display
        except ImportError:
            self.show_html()
            return

        legend_html = (
            '<div style="display:flex;flex-wrap:wrap;gap:6px;padding:6px 0 10px;">'
            '<span style="background:rgba(248,81,73,0.15);color:#f85149;'
            'padding:2px 9px;border-radius:10px;font-size:11px;" '
            'title="Compilador removeu esta linha (dead code / const fold)">'
            '✂ eliminada</span>'
            '<span style="background:rgba(63,185,80,0.15);color:#3fb950;'
            'padding:2px 9px;border-radius:10px;font-size:11px;" '
            'title="mul + add fundidos em fma.rn.f32 — sem mul separado">'
            '⚡ FMA</span>'
            '<span style="background:rgba(88,166,255,0.15);color:#58a6ff;'
            'padding:2px 9px;border-radius:10px;font-size:11px;" '
            'title="Linha gerou ≥ 6 instruções PTX">'
            '📦 pesado</span>'
            '<span style="color:#484f58;font-size:11px;padding:2px 4px;">'
            '— clique no filtro para focar</span>'
            '</div>'
        )

        filter_btn = w.ToggleButtons(
            options=[
                ("Tudo",         "all"),
                ("✂ Eliminadas", "optimized"),
                ("⚡ FMA",       "fma"),
                ("📦 Pesadas",   "heavy"),
                ("💡 Com PTX",   "mapped"),
            ],
            value="all",
            style={"button_width": "auto"},
            layout=w.Layout(margin="0 0 8px 0"),
        )

        table_w = w.HTML(value=self._build_html(filter_cls="all"))

        def _refresh(*_):
            table_w.value = self._build_html(filter_cls=filter_btn.value)

        filter_btn.observe(_refresh, names="value")

        display(w.VBox(
            [w.HTML(value=legend_html), filter_btn, table_w],
            layout=w.Layout(width="100%"),
        ))
