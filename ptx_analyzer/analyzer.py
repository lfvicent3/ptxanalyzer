"""
Classe principal PTXAnalyzer.
"""

import json
from collections import Counter
from typing import Optional
from .core import PTXKernel, CATEGORIES, analyze_control_flow, build_cfg
from .parser import parse_ptx
from .heuristics import run_heuristics, LEVEL_ICONS
from .runtime import RuntimeProfile, profile_cuda_runtime
from .output import emit_text, mermaid_block_html

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
from .visuals import (
    plot_category_pie, plot_category_bar, plot_instruction_timeline,
    plot_register_types, plot_instruction_mix_stacked, plot_memory_access_breakdown,
    plot_roofline,
    plot_instruction_roofline,
    plot_metric_space_pca,
    plot_branch_efficiency_registers,
    plot_memory_hierarchy,
    plot_runtime_curves,
    plot_branch_cfg as _plot_branch_cfg,
    plot_bra_graph as _plot_bra_graph,
    plot_decision_tree as _plot_decision_tree,
    plot_gpu_efficiency as _plot_gpu_efficiency,
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
        self._origin_path = None
        self._source_path = None
        self._ptx_path = None
        self.runtime_profile: Optional[RuntimeProfile] = None

    # ── construtores alternativos ────────────────────────────────────────────

    @classmethod
    def from_file(cls,
                  path: str,
                  kernel_index: int = 0,
                  arch: str = "sm_75",
                  verbose: bool = False) -> "PTXAnalyzer":
        import subprocess, os, re

        if path.endswith(".cu"):
            out_ptx = path.replace(".cu", ".ptx")
            # --ptxas-options=-v → ptxas imprime registradores, smem e spill no stderr
            cmd = [
                "nvcc", "-ptx", "-lineinfo",
                "--ptxas-options=-v",
                path, f"-arch={arch}", "-o", out_ptx,
            ]
            if verbose:
                print(f"Compilando CUDA para PTX: {' '.join(cmd)}")
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Erro ao compilar {path}:\n{res.stderr}")

            # Exibir info do ptxas (vai para stderr mesmo com sucesso)
            ptxas_info = [ln for ln in res.stderr.splitlines()
                          if ln.strip().startswith("ptxas")]
            if verbose and ptxas_info:
                print("\n── Informações ptxas (--ptxas-options=-v) ──")
                for ln in ptxas_info:
                    print(" ", ln)
                print()

            path_to_read = out_ptx
            source_path = os.path.abspath(path)
        else:
            path_to_read = path
            source_path = None

        with open(path_to_read, "r", encoding="utf-8", errors="replace") as f:
            code = f.read()

        has_lineinfo = re.search(r'(?m)^\s*\.loc\s+\d+\s+\d+', code) is not None

        # PTX sem .loc → sem lineinfo → tenta localizar e recompilar o .cu
        if not path.endswith(".cu") and not has_lineinfo:
            stem = os.path.splitext(os.path.basename(path))[0]
            ptx_abs = os.path.abspath(path)
            ptx_dir = os.path.dirname(ptx_abs)
            search_dirs = [
                ptx_dir,
                os.path.normpath(os.path.join(ptx_dir, "..", "kernels")),
                os.path.normpath(os.path.join(ptx_dir, "..")),
                os.getcwd(),
                os.path.join(os.getcwd(), "kernels"),
            ]
            for d in search_dirs:
                candidate = os.path.normpath(os.path.join(d, f"{stem}.cu"))
                if os.path.exists(candidate):
                    if verbose:
                        print(f"[ptx_analyzer] PTX sem lineinfo → tentando recompilar "
                              f"{os.path.basename(candidate)} com -lineinfo")
                    try:
                        return cls.from_file(candidate, kernel_index, arch, verbose=verbose)
                    except Exception as exc:
                        if verbose:
                            print(f"[ptx_analyzer] aviso: recompilação falhou, usando PTX existente sem lineinfo.\n"
                                  f"  motivo: {exc}")
                        break

        analyzer = cls(code, kernel_index)
        analyzer._origin_path = os.path.abspath(path)
        analyzer._ptx_path = os.path.abspath(path_to_read)
        analyzer._source_path = source_path
        return analyzer

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

    def _infer_source_path(self) -> Optional[str]:
        import os

        def _resolve_candidate(candidate: str) -> Optional[str]:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)

            base = os.path.basename(candidate) if candidate else ""
            if not base:
                return None

            search_roots = []
            if self._origin_path:
                search_roots.extend([
                    os.path.dirname(self._origin_path),
                    os.path.join(os.path.dirname(self._origin_path), "..", "kernels"),
                ])
            search_roots.extend([os.getcwd(), os.path.join(os.getcwd(), "kernels")])

            for root in search_roots:
                alt = os.path.abspath(os.path.join(root, base))
                if os.path.exists(alt):
                    return alt
            return None

        if self._source_path and os.path.exists(self._source_path):
            return self._source_path

        file_candidates = []
        for path in self.kernel.file_map.values():
            if path and path.endswith(".cu"):
                file_candidates.append(path)
        for candidate in file_candidates:
            resolved = _resolve_candidate(candidate)
            if resolved:
                self._source_path = resolved
                return self._source_path

        if self._origin_path:
            base = os.path.splitext(os.path.basename(self._origin_path))[0]
            search_roots = [
                os.path.dirname(self._origin_path),
                os.path.join(os.path.dirname(self._origin_path), "..", "kernels"),
                os.getcwd(),
                os.path.join(os.getcwd(), "kernels"),
            ]
            for root in search_roots:
                candidate = os.path.abspath(os.path.join(root, f"{base}.cu"))
                if os.path.exists(candidate):
                    self._source_path = candidate
                    return candidate
        return None

    def _format_stats_text(self) -> str:
        k = self.kernel
        W = 56
        lines = [
            "",
            "═" * W,
            f"  Kernel: {k.name}",
            "═" * W,
            f"  Total de instruções  : {k.total_instructions}",
            f"  Total de registros   : {k.total_registers}",
            "─" * W,
            f"  Branches total       : {k.total_branches}",
            f"  Cond. (@%p bra)      : {k.predicated_branches}  ← divergência possível",
            f"  Incond. (bra.uni)    : {k.unconditional_branches}  ← sem divergência",
            f"  setp (comparações)   : {k.setp_count}",
            f"  Branch ratio         : {k.branch_ratio:.1%}",
            "─" * W,
            f"  ld.global            : {k.global_loads}",
            f"  st.global            : {k.global_stores}",
            f"  Shared memory        : {k.shared_accesses}",
            f"  Local (spill)        : {k.local_accesses}",
            "─" * W,
            f"  Operações atômicas   : {k.atomics}",
            f"  FMA                  : {k.fma_count}",
            f"  shfl.sync            : {k.shfl_count}",
            f"  bar.sync             : {k.bar_sync_count}",
            f"  Basic blocks / loops : {k.basic_block_count} / {k.cfg_loop_count}",
            f"  Branch efficiency    : {k.branch_efficiency:.1%}",
            f"  Arithmetic ratio     : {k.arithmetic_ratio:.1%}",
            f"  Instr. intensity     : {k.instruction_intensity:.3f}",
            f"  Int. aritmética      : {k.arithmetic_intensity:.3f}",
            "",
            "  Contagem por categoria:",
        ]
        for cat, n in sorted(k.category_counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(n, 38)
            lines.append(f"    {cat:<14} {n:>5}  {bar}")
        lines.append("")
        return "\n".join(lines)

    def _format_warnings_text(self) -> str:
        lines = []
        for level, msg in run_heuristics(self.kernel):
            icon = LEVEL_ICONS.get(level, "•")
            lines.append(f"  {icon}  {msg}")
        return "\n".join(lines) + ("\n" if lines else "")

    def _format_memory_text(self) -> str:
        k = self.kernel
        d = k.memory_density
        return "\n".join([
            "",
            _section(f"Memória — {k.name}", 62),
            f"  ld.global             : {k.global_loads}",
            f"  st.global             : {k.global_stores}",
            f"  acessos shared        : {k.shared_accesses}",
            f"  acessos local         : {k.local_accesses}",
            f"  densidade ld.global   : {d['global_load_density']:.1%}",
            f"  densidade st.global   : {d['global_store_density']:.1%}",
            f"  densidade global total: {d['global_memory_density']:.1%}",
            f"  intensidade instr/mem : {k.instruction_intensity:.3f}",
            "",
        ])

    def _format_summary_text(self) -> str:
        k = self.kernel
        W = 58
        title = f"PTX Kernel: {k.name}  ({k.param_count} parâm.)"
        lines = [
            "╔" + "═" * W + "╗",
            f"║  {title:<{W - 2}}║",
            "╚" + "═" * W + "╝",
            "",
            _section("Métricas", W + 2),
        ]
        pairs = [
            ("Instruções totais",    k.total_instructions, "Registradores",        k.total_registers),
            ("ld.global",            k.global_loads,       "st.global",            k.global_stores),
            ("ld/st.local (spill)",  k.local_accesses,     "Shared memory",        k.shared_accesses),
            ("Branches total",       k.total_branches,     "Branch ratio",         f"{k.branch_ratio:.1%}"),
            ("Cond. (@%p bra)",      k.predicated_branches,"Incond. (.uni)",       k.unconditional_branches),
            ("setp (predicados)",    k.setp_count,         "Blocos básicos",       k.basic_block_count),
            ("FMA",                  k.fma_count,          "shfl.sync",            k.shfl_count),
            ("Arithmetic ratio",     f"{k.arithmetic_ratio:.1%}", "Instr. intensity", k.instruction_intensity),
            ("Int. aritmética",      k.arithmetic_intensity, "Branch efficiency",  f"{k.branch_efficiency:.1%}"),
            ("CFG loops",            k.cfg_loop_count,     "Só registros",         "Sim" if k.is_register_only else "Não"),
        ]
        for l1, v1, l2, v2 in pairs:
            lines.append(f"  {l1:<22}: {str(v1):>6}    {l2:<18}: {str(v2):>6}")

        lines.extend(["", _section("Mix de Instruções", W + 2)])
        mix = sorted(k.category_counts.items(), key=lambda x: -x[1])
        max_c = max((n for _, n in mix), default=1)
        total = max(k.total_instructions, 1)
        for cat, n in mix:
            pct = n / total * 100
            bar = _bar(n, max_c, 32)
            lines.append(f"  {cat:<14} {pct:5.1f}%  {bar}  {n:>4}")

        lines.extend(["", _section("Memória", W + 2)])
        mem_items = [
            ("Global loads",  k.global_loads),
            ("Global stores", k.global_stores),
            ("Shared",        k.shared_accesses),
            ("Local (spill)", k.local_accesses),
        ]
        max_mem = max((v for _, v in mem_items), default=1)
        for label, n in mem_items:
            bar = _bar(n, max_mem, 30)
            lines.append(f"  {label:<16}: {n:>5}  {bar}")

        lines.extend(["", _section("Diagnóstico", W + 2)])
        for level, msg in run_heuristics(k):
            icon = LEVEL_ICONS.get(level, "•")
            lines.append(f"  {icon}  {msg}")
        lines.append("")
        return "\n".join(lines)

    def _format_control_flow_text(self, max_blocks: int = 30) -> str:
        cfg = analyze_control_flow(self.kernel)
        visible_blocks = cfg.order[:max_blocks]
        visible_set = set(visible_blocks)
        lines = [
            "",
            "═" * 72,
            f"  Fluxo de Controle / BRA — {self.kernel.name}",
            "═" * 72,
            f"  Basic blocks     : {len(cfg.blocks)}",
            f"  Arestas CFG      : {len(cfg.edges)}",
            f"  Branch sites     : {len(cfg.branch_sites)}",
            f"  Branches cond.   : {self.kernel.predicated_branches}",
            f"  Branch ratio     : {self.kernel.branch_ratio:.1%}",
            "",
            "  Topologia dos blocos:",
        ]
        for label in visible_blocks:
            block = cfg.blocks[label]
            exits = ", ".join(f"{etype}->{target}" for etype, target in block.exits) or "terminal"
            lines.append(f"    {label:<20} instr={len(block.instructions):>3}  exits=[{exits}]")

        lines.extend(["", "  Sites de branch (BRA):"])
        sites = [s for s in cfg.branch_sites if s.block_label in visible_set]
        if not sites:
            lines.append("    nenhum branch BRA encontrado")
        for site in sites:
            src = f" linha={site.source_line}" if site.source_line > 0 else ""
            lines.append(
                f"    {site.block_label}: {site.branch_kind} BRA PTX:{site.line_no}{src}"
            )
            if site.setp_raw:
                lines.append(f"      setp: {site.setp_raw}")
            lines.append(f"      bra : {site.raw}")
            lines.append(
                f"      taken={site.taken_target or '-'} "
                f"(instr={site.taken_instruction_count}, mem={site.taken_memory_ops})"
            )
            lines.append(
                f"      fall ={site.fallthrough_target or '-'} "
                f"(instr={site.fallthrough_instruction_count}, mem={site.fallthrough_memory_ops})"
            )
            lines.append(f"      risco divergência: {site.divergence_risk}")
        lines.append("")
        return "\n".join(lines)

    def _format_bra_text(self, max_branches: int = 24) -> str:
        cfg = analyze_control_flow(self.kernel)
        sites = cfg.branch_sites[:max_branches]
        lines = [
            "",
            "═" * 72,
            f"  Grafo de BRA / Divergência — {self.kernel.name}",
            "═" * 72,
            f"  Sites de branch: {len(cfg.branch_sites)}",
            "",
        ]
        if not sites:
            lines.append("  nenhum site de BRA encontrado")
            lines.append("")
            return "\n".join(lines)

        for idx, site in enumerate(sites, 1):
            loc = f"{site.source_line}" if site.source_line > 0 else f"PTX:{site.line_no}"
            lines.append(
                f"  B{idx:02d}  bloco={site.block_label}  origem={loc}  risco={site.divergence_risk}"
            )
            if site.setp_raw:
                lines.append(f"       setp : {site.setp_raw}")
            lines.append(f"       bra  : {site.raw}")
            lines.append(
                f"       taken: {site.taken_target or '-':<20} "
                f"instr={site.taken_instruction_count:<3} mem={site.taken_memory_ops}"
            )
            lines.append(
                f"       fall : {site.fallthrough_target or '-':<20} "
                f"instr={site.fallthrough_instruction_count:<3} mem={site.fallthrough_memory_ops}"
            )
        lines.append("")
        return "\n".join(lines)

    def _format_hotspots_text(self, max_items: int = 10) -> str:
        cfg = analyze_control_flow(self.kernel)
        k = self.kernel
        sites = sorted(
            cfg.branch_sites,
            key=lambda site: (
                {"high": 3, "medium": 2, "low": 1, "none": 0}.get(site.divergence_risk, 0),
                site.taken_memory_ops + site.fallthrough_memory_ops,
                abs(site.taken_instruction_count - site.fallthrough_instruction_count),
            ),
            reverse=True,
        )[:max_items]
        hotspots = cfg.memory_hotspots[:max_items]

        lines = [
            "",
            "═" * 76,
            f"  Hotspots de Divergência e Memória — {k.name}",
            "═" * 76,
            f"  densidade global total : {k.memory_density['global_memory_density']:.1%}",
            f"  densidade ld.global    : {k.memory_density['global_load_density']:.1%}",
            f"  densidade st.global    : {k.memory_density['global_store_density']:.1%}",
            f"  branches predicados    : {k.predicated_branches}",
            f"  branch efficiency est. : {k.branch_efficiency:.1%}",
            "",
            "  Top branches com maior chance de divergência:",
        ]
        if not sites:
            lines.append("    nenhum BRA encontrado")
        for idx, site in enumerate(sites, 1):
            loc = f"linha .cu {site.source_line}" if site.source_line > 0 else f"PTX:{site.line_no}"
            mem_total = site.taken_memory_ops + site.fallthrough_memory_ops
            inst_delta = abs(site.taken_instruction_count - site.fallthrough_instruction_count)
            lines.append(
                f"    D{idx:02d}  {site.block_label:<18} {loc:<14} risco={site.divergence_risk:<6} "
                f"mem_paths={mem_total:<2} delta_instr={inst_delta}"
            )
            lines.append(
                f"         taken={site.taken_target or '-':<18} "
                f"(instr={site.taken_instruction_count:<3} mem={site.taken_memory_ops})"
            )
            lines.append(
                f"         fall ={site.fallthrough_target or '-':<18} "
                f"(instr={site.fallthrough_instruction_count:<3} mem={site.fallthrough_memory_ops})"
            )

        lines.extend(["", "  Top blocos com pressão de memória:"])
        if not hotspots:
            lines.append("    nenhum bloco com ld/st/atom/red encontrado")
        for idx, hotspot in enumerate(hotspots, 1):
            loc = f"linha .cu {hotspot.source_line}" if hotspot.source_line > 0 else "-"
            lines.append(
                f"    M{idx:02d}  {hotspot.block_label:<18} {loc:<14} "
                f"mem_ops={hotspot.memory_ops:<3} densidade={hotspot.memory_density:.1%} "
                f"instr={hotspot.instruction_count}"
            )
            lines.append(
                f"         gld={hotspot.global_loads:<2} gst={hotspot.global_stores:<2} "
                f"shared={hotspot.shared_accesses:<2} local={hotspot.local_accesses:<2}"
            )
        lines.append("")
        return "\n".join(lines)

    def _format_explain_text(self) -> str:
        k = self.kernel
        cfg = analyze_control_flow(k)
        top_branch = None
        if cfg.branch_sites:
            top_branch = max(
                cfg.branch_sites,
                key=lambda site: (
                    {"high": 3, "medium": 2, "low": 1, "none": 0}.get(site.divergence_risk, 0),
                    site.taken_memory_ops + site.fallthrough_memory_ops,
                ),
            )
        top_memory = cfg.memory_hotspots[0] if cfg.memory_hotspots else None

        if k.global_loads + k.global_stores >= k.arithmetic_instructions:
            dominant = "dependente de memoria global"
        elif k.shared_accesses > 0:
            dominant = "fortemente apoiado em memoria compartilhada"
        elif k.local_accesses > 0:
            dominant = "pressionado por registradores/spill local"
        else:
            dominant = "mais proximo de compute-bound"

        lines = [
            "",
            "═" * 72,
            f"  Leitura Guiada — {k.name}",
            "═" * 72,
            f"  1. Perfil dominante : {dominant}",
            f"  2. Branches cond.   : {k.predicated_branches}  "
            f"(branch ratio {k.branch_ratio:.1%}, eficiencia est. {k.branch_efficiency:.1%})",
            f"  3. Memoria global   : {k.global_loads} loads + {k.global_stores} stores  "
            f"(densidade {k.memory_density['global_memory_density']:.1%})",
            f"  4. Shared / local   : shared={k.shared_accesses}  local={k.local_accesses}",
        ]

        if top_memory is not None:
            loc = f"linha .cu {top_memory.source_line}" if top_memory.source_line > 0 else top_memory.block_label
            lines.append(
                f"  5. Bloco mais pesado: {loc}  "
                f"(mem_ops={top_memory.memory_ops}, densidade={top_memory.memory_density:.1%})"
            )
        else:
            lines.append("  5. Bloco mais pesado: nenhum hotspot de memoria identificado")

        if top_branch is not None:
            loc = f"linha .cu {top_branch.source_line}" if top_branch.source_line > 0 else top_branch.block_label
            lines.append(
                f"  6. Ponto critico de branch: {loc}  "
                f"(risco={top_branch.divergence_risk}, "
                f"taken_mem={top_branch.taken_memory_ops}, fall_mem={top_branch.fallthrough_memory_ops})"
            )
        else:
            lines.append("  6. Ponto critico de branch: nenhum BRA encontrado")

        lines.extend([
            "",
            "  Como ler isso:",
            "    - Se 'memoria global' dominar, o gargalo tende a ser trafego de memoria.",
            "    - Se 'local' aparecer, ha chance de spill e perda de ocupancia.",
            "    - Se houver muitos branches cond., ha mais chance de divergencia de warp.",
            "",
        ])
        return "\n".join(lines)

    def _format_mermaid_text(self,
                             max_decisions: int = 12,
                             label_mode: str = "ptx",
                             source_lines: Optional[list[str]] = None) -> str:
        cfg = analyze_control_flow(self.kernel)
        if not cfg.branch_sites:
            return "graph TD\n    Entry([Inicio]) --> Exit([Sem BRA])\n"

        blocks, order = build_cfg(self.kernel)
        sites = cfg.branch_sites if max_decisions <= 0 else cfg.branch_sites[:max_decisions]
        site_by_label = {site.block_label: site for site in cfg.branch_sites}
        visible_labels = set(order)
        if max_decisions > 0:
            visible_labels = {"__ENTRY__"}
            for site in sites:
                visible_labels.add(site.block_label)
                if site.taken_target in blocks:
                    visible_labels.add(site.taken_target)
                if site.fallthrough_target in blocks:
                    visible_labels.add(site.fallthrough_target)

        alias_by_label = {label: f"N{idx}" for idx, label in enumerate(order, 1)}
        visible_order = [label for label in order if label in visible_labels]
        order_index = {label: idx for idx, label in enumerate(visible_order)}

        raw_sections = []
        for label in visible_order:
            for _, target in blocks[label].exits:
                if target not in order_index:
                    continue
                if order_index[target] <= order_index[label]:
                    start = order_index[target]
                    end = order_index[label]
                    raw_sections.append((start, end))

        raw_sections = sorted(set(raw_sections), key=lambda item: (item[0], -(item[1] - item[0])))
        sections = []
        for start, end in raw_sections:
            sections.append({
                "name": f"Loop_{len(sections) + 1}",
                "start": start,
                "end": end,
                "children": [],
            })

        parents = [None] * len(sections)
        for i, outer in enumerate(sections):
            best_parent = None
            best_span = None
            for j, candidate in enumerate(sections):
                if i == j:
                    continue
                if candidate["start"] <= outer["start"] and outer["end"] <= candidate["end"]:
                    span = candidate["end"] - candidate["start"]
                    own_span = outer["end"] - outer["start"]
                    if span > own_span and (best_span is None or span < best_span):
                        best_parent = j
                        best_span = span
            parents[i] = best_parent

        root_sections = []
        for idx, parent_idx in enumerate(parents):
            if parent_idx is None:
                root_sections.append(idx)
            else:
                sections[parent_idx]["children"].append(idx)

        covered_by_section = set()
        for sec in sections:
            for idx in range(sec["start"], sec["end"] + 1):
                covered_by_section.add(idx)

        outside_sections = [label for idx, label in enumerate(visible_order) if idx not in covered_by_section]

        def _source_snippet(label: str) -> str:
            if not source_lines:
                return ""
            block = blocks[label]
            line_numbers = []
            seen = set()
            for instr in block.instructions:
                src_line = getattr(instr, "source_line", 0)
                if src_line > 0 and src_line not in seen and 1 <= src_line <= len(source_lines):
                    seen.add(src_line)
                    line_numbers.append(src_line)
            if not line_numbers:
                return ""
            snippets = []
            for src_line in line_numbers[:2]:
                text = source_lines[src_line - 1].strip()
                if text:
                    snippets.append(text)
            return "<br>".join(snippets[:2])

        def _node_text(label):
            block = blocks[label]
            if label == "__ENTRY__":
                site = site_by_label.get(label)
                if site:
                    origin = f"linha {site.source_line}" if site.source_line > 0 else f"PTX {site.line_no}"
                    if label_mode == "source":
                        source = _source_snippet(label)
                        if source:
                            return f"ENTRY<br>{origin}<br>{source}"
                    raw = site.raw.strip().replace('"', "'")
                    return f"ENTRY<br>{origin}<br>{raw}"
                return "ENTRY"

            if label in site_by_label:
                site = site_by_label[label]
                origin = f"linha {site.source_line}" if site.source_line > 0 else f"PTX {site.line_no}"
                if label_mode == "source":
                    source = _source_snippet(label)
                    if source:
                        return f"{label}<br>{origin}<br>{source}"
                raw = site.raw.strip().replace('"', "'")
                return f"{label}<br>{origin}<br>{raw}"

            last = block.instructions[-1] if block.instructions else None
            origin = ""
            if last and getattr(last, "source_line", 0) > 0:
                origin = f"linha {last.source_line}<br>"
            elif last:
                origin = f"PTX {last.line_no}<br>"

            if block.is_terminal:
                return f"{label}<br>{origin}{last.op_base if last else 'terminal'}"
            if last:
                if label_mode == "source":
                    source = _source_snippet(label)
                    if source:
                        return f"{label}<br>{origin}{source}"
                last_text = last.raw.strip().replace('"', "'")
                return f"{label}<br>{origin}{last_text}"
            return label

        def _node_decl(label, indent="    "):
            alias = alias_by_label[label]
            text = (
                _node_text(label)
                .replace("\\", "\\\\")
                .replace('"', "&quot;")
                .replace("[", "&#91;")
                .replace("]", "&#93;")
                .replace("{", "&#123;")
                .replace("}", "&#125;")
            )
            if blocks[label].is_terminal:
                return f'{indent}{alias}(["{text}"])'
            return f'{indent}{alias}["{text}"]'

        lines = ["graph TD"]

        def _emit_section(section_idx, indent="    "):
            sec = sections[section_idx]
            lines.append(f"{indent}subgraph {sec['name']}")

            child_ranges = [(sections[c]["start"], sections[c]["end"], c) for c in sec["children"]]
            child_ranges.sort()

            pos = sec["start"]
            for child_start, child_end, child_idx in child_ranges:
                while pos < child_start:
                    label = visible_order[pos]
                    lines.append(_node_decl(label, indent + "    "))
                    pos += 1
                _emit_section(child_idx, indent + "    ")
                pos = child_end + 1

            while pos <= sec["end"]:
                label = visible_order[pos]
                lines.append(_node_decl(label, indent + "    "))
                pos += 1

            lines.append(f"{indent}end")

        for section_idx in root_sections:
            _emit_section(section_idx)

        if outside_sections:
            lines.append("    subgraph CFG")
            for label in outside_sections:
                lines.append(_node_decl(label, "        "))
            lines.append("    end")

        lines.append("")
        lines.append("    %% grafo completo dos blocos e BRAs")
        emitted = set()
        for label in visible_order:
            src_alias = alias_by_label[label]
            for edge_kind, target in blocks[label].exits:
                if target not in visible_labels:
                    continue
                dst_alias = alias_by_label[target]
                edge = (src_alias, dst_alias, edge_kind)
                if edge in emitted:
                    continue
                emitted.add(edge)
                lines.append(f'    {src_alias} -- "{edge_kind}" --> {dst_alias}')

        lines.extend(["", "    %% destaque"])
        lines.append("    style CFG fill:#f8fafc,stroke:#60a5fa,stroke-width:1px,color:#111827;")
        for idx, sec in enumerate(sections, 1):
            fill = "#fff7ed" if idx % 2 == 1 else "#faf5ff"
            stroke = "#ea580c" if idx % 2 == 1 else "#9333ea"
            lines.append(
                f"    style {sec['name']} fill:{fill},stroke:{stroke},stroke-width:1px,color:#111827;"
            )

        top_branch = max(
            cfg.branch_sites,
            key=lambda site: (
                {"high": 3, "medium": 2, "low": 1, "none": 0}.get(site.divergence_risk, 0),
                site.taken_memory_ops + site.fallthrough_memory_ops,
            ),
        )
        for label in visible_order:
            alias = alias_by_label[label]
            if blocks[label].is_terminal:
                lines.append(f"    style {alias} fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#111827;")
            elif label in site_by_label:
                lines.append(f"    style {alias} fill:#ffffff,stroke:#60a5fa,stroke-width:1px,color:#111827;")
            else:
                lines.append(f"    style {alias} fill:#fefce8,stroke:#a16207,stroke-width:1px,color:#111827;")

        if top_branch.block_label in alias_by_label and top_branch.block_label in visible_labels:
            lines.append(
                f"    style {alias_by_label[top_branch.block_label]} fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#111827;"
            )
        return "\n".join(lines) + "\n"

    def _format_mermaid_fence(self, max_decisions: int = 12) -> str:
        graph = self._format_mermaid_text(max_decisions=max_decisions).rstrip()
        return f"```mermaid\n{graph}\n```"

    def report(self, section: str = "summary", mode: str = "text"):
        """
        Interface unificada para saídas textuais/HTML do analisador.

        Args:
            section:
                - "summary"
                - "stats"
                - "warnings"
                - "memory"
                - "hotspots"
                - "explain"
            mode:
                - "text"
                - "html"
                - "raw"
                - "dict"  (somente para summary/stats/memory)
                - "widget" (abre a UI antiga `show()`)
        """
        section = section.lower()
        mode = mode.lower()

        if mode == "widget":
            return self.show()

        if mode == "dict":
            if section in ("summary", "stats"):
                return self.to_dict()
            if section == "memory":
                return {
                    "global_loads": self.kernel.global_loads,
                    "global_stores": self.kernel.global_stores,
                    "shared_accesses": self.kernel.shared_accesses,
                    "local_accesses": self.kernel.local_accesses,
                    **self.kernel.memory_density,
                }
            if section == "hotspots":
                cfg = analyze_control_flow(self.kernel)
                return {
                    "branch_sites": [site.to_dict() for site in cfg.branch_sites],
                    "memory_hotspots": [hotspot.to_dict() for hotspot in cfg.memory_hotspots],
                }
            if section == "explain":
                cfg = analyze_control_flow(self.kernel)
                top_branch = None
                if cfg.branch_sites:
                    top_branch = max(
                        cfg.branch_sites,
                        key=lambda site: (
                            {"high": 3, "medium": 2, "low": 1, "none": 0}.get(site.divergence_risk, 0),
                            site.taken_memory_ops + site.fallthrough_memory_ops,
                        ),
                    )
                return {
                    "kernel": self.kernel.name,
                    "dominant_profile": (
                        "memory-global"
                        if self.kernel.global_loads + self.kernel.global_stores >= self.kernel.arithmetic_instructions
                        else "shared"
                        if self.kernel.shared_accesses > 0
                        else "local-pressure"
                        if self.kernel.local_accesses > 0
                        else "compute-leaning"
                    ),
                    "top_branch": top_branch.to_dict() if top_branch else None,
                    "top_memory_hotspot": cfg.memory_hotspots[0].to_dict() if cfg.memory_hotspots else None,
                }
            raise ValueError(f"Modo dict não suportado para section={section!r}")

        builders = {
            "summary": self._format_summary_text,
            "stats": self._format_stats_text,
            "warnings": self._format_warnings_text,
            "memory": self._format_memory_text,
            "hotspots": self._format_hotspots_text,
            "explain": self._format_explain_text,
        }
        if section not in builders:
            raise ValueError(f"Seção desconhecida: {section}")
        return emit_text(builders[section](), mode=mode)

    def memory_report(self, mode: str = "text"):
        """Relatório padronizado de densidade de memória global/shared/local."""
        if mode == "data":
            return {
                "global_loads": self.kernel.global_loads,
                "global_stores": self.kernel.global_stores,
                "shared_accesses": self.kernel.shared_accesses,
                "local_accesses": self.kernel.local_accesses,
                **self.kernel.memory_density,
            }
        return emit_text(self._format_memory_text(), mode=mode)

    def hotspots_report(self, mode: str = "text", max_items: int = 10):
        """
        Relatório de hotspots com foco em:
          - densidade de acessos globais
          - blocos com maior pressão de memória
          - branches com maior risco estático de divergência
        """
        if mode == "data":
            cfg = analyze_control_flow(self.kernel)
            return {
                "branch_sites": [site.to_dict() for site in cfg.branch_sites[:max_items]],
                "memory_hotspots": [hotspot.to_dict() for hotspot in cfg.memory_hotspots[:max_items]],
                "memory_density": dict(self.kernel.memory_density),
            }
        return emit_text(self._format_hotspots_text(max_items=max_items), mode=mode)

    def flowchart(self, mode: str = "html", max_decisions: int = 0):
        """
        Fluxo lógico simplificado em Mermaid.

        Esta é a visualização recomendada para entendimento humano do kernel.
        """
        return self.control_flow(mode=mode, view="mermaid", max_decisions=max_decisions)

    @property
    def kernel_count(self) -> int:
        return len(self._all_kernels)

    @property
    def kernel_names(self):
        return [kernel.name for kernel in self._all_kernels]

    def explain(self, mode: str = "text"):
        """Leitura curta do kernel em linguagem mais direta."""
        if mode == "data":
            return self.report(section="explain", mode="dict")
        return emit_text(self._format_explain_text(), mode=mode)

    def iter_kernel_analyzers(self):
        """
        Retorna um analisador por kernel encontrado no mesmo PTX.
        Útil para arquivos como baseline/bubble_sort_all com 3 variantes.
        """
        analyzers = []
        for idx in range(len(self._all_kernels)):
            item = PTXAnalyzer(self._code, kernel_index=idx)
            item._origin_path = self._origin_path
            item._source_path = self._source_path
            item._ptx_path = self._ptx_path
            analyzers.append(item)
        return analyzers

    def compare_kernels_in_file(self):
        """
        Compara automaticamente todos os kernels presentes no mesmo arquivo PTX.
        """
        from .comparator import PTXComparator

        comp = PTXComparator()
        for analyzer in self.iter_kernel_analyzers():
            short_name = analyzer.kernel.name
            if "global" in short_name:
                label = "global"
            elif "shared" in short_name:
                label = "shared"
            elif "register" in short_name:
                label = "register"
            else:
                label = short_name
            comp.add(label, analyzer)
        return comp

    def flowcharts_in_file(self,
                           mode: str = "html",
                           max_decisions: int = 0,
                           columns: int = 3):
        """
        Atalho para exibir lado a lado os fluxogramas de todos os kernels
        presentes no mesmo arquivo PTX.
        """
        return self.compare_kernels_in_file().flowcharts(
            mode=mode,
            max_decisions=max_decisions,
            columns=columns,
        )

    def control_flow(self,
                     mode: str = "text",
                     view: str = "cfg",
                     max_blocks: int = 30,
                     max_decisions: int = 20,
                     max_branches: int = 24):
        """
        Interface unificada para análise de fluxo/branches.

        Args:
            mode:
                - "text": relatório textual dos BRAs/CFG
                - "graph": visualização Plotly
                - "data": estrutura serializável com blocos, arestas e branch sites
            view:
                - "cfg": grafo completo de fluxo
                - "decision": árvore simplificada de decisões
                - "mermaid": grafo lógico em Mermaid
        """
        mode = mode.lower()
        view = view.lower()
        if mode == "data":
            cfg = analyze_control_flow(self.kernel)
            if view == "bra":
                return {
                    "branch_sites": [site.to_dict() for site in cfg.branch_sites],
                    "branch_count": len(cfg.branch_sites),
                }
            if view == "mermaid":
                return {"mermaid": self._format_mermaid_text(max_decisions=max_decisions)}
            return cfg.to_dict()
        if mode == "graph":
            if view == "decision":
                return _plot_decision_tree(self.kernel, max_decisions)
            if view == "bra":
                return _plot_bra_graph(self.kernel, max_branches)
            return _plot_branch_cfg(self.kernel, max_blocks)
        if view == "mermaid" and mode == "html":
            graphs = {"ptx": self._format_mermaid_text(max_decisions=max_decisions, label_mode="ptx")}
            src_path = self._infer_source_path()
            if src_path:
                try:
                    with open(src_path, "r", encoding="utf-8", errors="replace") as f:
                        source_lines = f.read().splitlines()
                    if any(instr.source_line > 0 for instr in self.kernel.instructions):
                        graphs["source"] = self._format_mermaid_text(
                            max_decisions=max_decisions,
                            label_mode="source",
                            source_lines=source_lines,
                        )
                except Exception:
                    pass
            try:
                from IPython.display import HTML, display
                display(HTML(mermaid_block_html(graphs, title=self.kernel.name)))
                return None
            except Exception:
                return emit_text(self._format_mermaid_fence(max_decisions=max_decisions), mode="text")
        if mode in ("text", "html", "raw"):
            if view == "mermaid":
                if mode == "text":
                    return emit_text(self._format_mermaid_fence(max_decisions=max_decisions), mode="text")
                if mode == "raw":
                    return self._format_mermaid_text(max_decisions=max_decisions)
                return emit_text(self._format_mermaid_fence(max_decisions=max_decisions), mode="text")
            if view == "bra":
                return emit_text(self._format_bra_text(max_branches=max_branches), mode=mode)
            return emit_text(self._format_control_flow_text(max_blocks=max_blocks), mode=mode)
        raise ValueError(f"Modo desconhecido: {mode}")

    # ── interface de texto ───────────────────────────────────────────────────

    def show_stats(self):
        """Imprime métricas resumidas no terminal/célula."""
        return self.report(section="stats", mode="html")

    def show_warnings(self):
        """Imprime diagnósticos no terminal."""
        return self.report(section="warnings", mode="text")

    def summary(self):
        """
        Resumo completo em texto: métricas, mix de instruções, memória e diagnóstico.
        Equivalente text-only ao show() interativo — não requer Jupyter nem plotly.
        """
        return self.report(section="summary", mode="text")

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

    def plot_instruction_roofline(self, peak_ipc: float = 128.0, mem_ceiling: float = 32.0):
        """Instruction Roofline estático no espaço instruções/memória."""
        plot_instruction_roofline({self.kernel.name: self.kernel}, peak_ipc, mem_ceiling)

    def plot_memory_hierarchy(self):
        """Diagrama da hierarquia de memória com métricas sobrepostas."""
        plot_memory_hierarchy(self.kernel)

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
                    plot_fn()  # _show_fig é chamado internamente
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

    # ── runtime ─────────────────────────────────────────────────────────────

    def profile_runtime(self, sizes=(1024,), repeats: int = 3, arch: str = "sm_75",
                        source_path: Optional[str] = None,
                        executable_path: Optional[str] = None,
                        extra_compile_flags=None,
                        extra_run_args=None):
        """
        Compila o .cu correspondente e executa o benchmark instrumentado.

        Requer que o código-fonte CUDA contenha medição com cudaEvent e imprima
        linhas no formato:
            nome  N=1024  1.234 ms  OK
        """
        src = source_path or self._infer_source_path()
        if not src:
            raise RuntimeError(
                "Não foi possível localizar o .cu correspondente para profiling de runtime. "
                "Passe source_path explicitamente."
            )

        self.runtime_profile = profile_cuda_runtime(
            src_path=src,
            sizes=sizes,
            repeats=repeats,
            arch=arch,
            executable_path=executable_path,
            extra_compile_flags=extra_compile_flags,
            extra_run_args=extra_run_args,
        )
        self._source_path = src
        return self.runtime_profile

    def show_runtime(self):
        """
        Exibe um resumo textual das últimas métricas de runtime coletadas.
        """
        if self.runtime_profile is None:
            print("Nenhum runtime coletado. Use profile_runtime(...) primeiro.")
            return

        rp = self.runtime_profile
        print("\n" + "═" * 62)
        print(f"  Runtime CUDA — {self.kernel.name}")
        print("═" * 62)
        print(f"  Fonte      : {rp.source_path}")
        print(f"  Executável : {rp.executable_path}")
        print(f"  Arquitetura: {rp.arch}")
        print(f"  Tamanhos   : {', '.join(str(n) for n in rp.sizes)}")
        print(f"  Repetições : {rp.repeats}")
        print("─" * 62)

        for label, bench in rp.benchmarks.items():
            print(f"  {label}")
            print(f"    runs     : {bench.runs}")
            print(f"    min/avg  : {bench.min_ms:.3f} / {bench.mean_ms:.3f} ms")
            print(f"    med/max  : {bench.median_ms:.3f} / {bench.max_ms:.3f} ms")
            print(f"    stdev    : {bench.stdev_ms:.3f} ms")
            print(f"    ok rate  : {bench.ok_rate:.1%}")
            for n, stats in bench.by_size().items():
                print(f"    N={n:<8} mean={stats['mean_ms']:.3f} ms  "
                      f"min={stats['min_ms']:.3f}  max={stats['max_ms']:.3f}  "
                      f"ok={stats['ok_rate']:.1%}")
                for sample in stats["samples"]:
                    extra = f"  {sample['extra']}" if sample["extra"] else ""
                    print(f"      run  {sample['milliseconds']:>8.3f} ms  "
                          f"{sample['status']}{extra}")
        print()

    def plot_runtime_curves(self):
        """Curvas de tempo e sorting-rate por tamanho e distribuição."""
        if self.runtime_profile is None:
            print("Nenhum runtime coletado. Use profile_runtime(...) primeiro.")
            return
        return plot_runtime_curves(self.runtime_profile)

    # ── branches / CFG ──────────────────────────────────────────────────────

    def show_branch_tree(self):
        """
        Wrapper compatível com a API antiga.
        Use `control_flow(mode="text" | "graph" | "data")` na API nova.
        """
        return self.control_flow(mode="text", view="cfg")

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
        return self.control_flow(mode="graph", view="cfg", max_blocks=max_blocks)

    def plot_decision_tree(self, max_decisions: int = 0):
        """
        Exibe o fluxo lógico simplificado em Mermaid.

        A versão Plotly da decision tree ficou muito poluída para kernels
        reais; por isso este método agora prioriza Mermaid, que tem melhor
        autorroteamento e leitura mais limpa.
        """
        return self.flowchart(mode="html", max_decisions=max_decisions)

    def plot_bra_graph(self, max_branches: int = 24):
        """
        Exibe um grafo simplificado dos sites de `BRA`, seus destinos e risco de divergência.
        """
        return self.control_flow(mode="graph", view="bra", max_branches=max_branches)

    def plot_gpu_efficiency(self, arch: str = "sm_86", threads_per_block: int = 256):
        """
        Dashboard de eficiência de GPU — 6 gauges com análise estática do PTX.

        Métricas:
          • Ocupância estimada   — registradores/thread vs limite do SM
          • Risco de Divergência — branches condicionais como % das instruções
          • Cobertura de Grid    — kernel usa grid-stride (%nctaid)
          • Posição no Roofline  — intensidade aritmética vs ridge point
          • Eficiência de Warp   — inverso do risco de divergência
          • Score Geral          — média das 4 métricas anteriores

        Inclui painel de recomendações geradas automaticamente.

        Args:
            arch: arquitetura alvo para parâmetros do SM
                  (sm_75, sm_80, sm_86, sm_89, sm_90). Default: sm_86.
            threads_per_block: threads por bloco assumidos (padrão: 256).
        """
        return _plot_gpu_efficiency(self.kernel, arch, threads_per_block)

    # ── exportação ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        k = self.kernel
        return {
            "kernel":      k.name,
            "metrics":     k.metrics_dict(),
            "categories":  k.category_counts,
            "registers":   {t: list(r) for t, r in k.reg_decls.items()},
            "diagnostics": run_heuristics(k),
            "runtime":     self.runtime_profile.to_dict() if self.runtime_profile else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
