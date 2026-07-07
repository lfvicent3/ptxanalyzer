"""
Classe principal PTXAnalyzer.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter, deque
from typing import Optional

from .core import PTXKernel, analyze_control_flow
from .output import emit_text, mermaid_block_html
from .parser import parse_ptx
from .ptxas import parse_ptxas_output
from .runtime import (
    BenchmarkSuite,
    RuntimeProfile,
    load_benchmark_csv,
    parse_benchmark_output,
    profile_cuda_runtime,
)


def _bar(n: int, max_n: int, width: int = 28) -> str:
    if max_n == 0:
        return "░" * width
    filled = round(n / max_n * width)
    return "█" * filled + "░" * (width - filled)


def _section(title: str, total_w: int = 60) -> str:
    rest = total_w - len(title) - 4
    return f"── {title} {'─' * max(0, rest)}"


def _normalize_kernel_name(name: str) -> str:
    return name.strip().rstrip(":")


class PTXAnalyzer:
    """
    Interface principal de análise de um kernel PTX.

    Foco:
      - métricas extraídas do PTX
      - métricas reais do ptxas
      - fluxo de controle em Mermaid
      - ligação com benchmark
    """

    def __init__(self,
                 code: str,
                 kernel_index: int = 0,
                 kernel_name: Optional[str] = None):
        self._code = code
        kernels = parse_ptx(code)
        if not kernels:
            raise ValueError("Nenhum kernel encontrado no PTX fornecido.")
        self._all_kernels = kernels
        selected_index = self._resolve_kernel_index(kernels, kernel_index, kernel_name)
        self.kernel = kernels[selected_index]
        self._origin_path: Optional[str] = None
        self._source_path: Optional[str] = None
        self._ptx_path: Optional[str] = None
        self._ptxas_stderr: str = ""
        self.runtime_profile: Optional[RuntimeProfile] = None
        self.benchmark_suite: Optional[BenchmarkSuite] = None

    @staticmethod
    def _resolve_kernel_index(kernels: list[PTXKernel],
                              kernel_index: int,
                              kernel_name: Optional[str]) -> int:
        if kernel_name:
            normalized_target = _normalize_kernel_name(kernel_name)
            for idx, kernel in enumerate(kernels):
                if _normalize_kernel_name(kernel.name) == normalized_target:
                    return idx
            available = ", ".join(_normalize_kernel_name(kernel.name) for kernel in kernels)
            raise ValueError(
                f"Kernel {kernel_name!r} não encontrado. Disponíveis: {available}"
            )

        if kernel_index < 0 or kernel_index >= len(kernels):
            return 0
        return kernel_index

    @staticmethod
    def _compile_cuda_to_ptx(path: str, arch: str, verbose: bool) -> tuple[str, str, str, str]:
        out_ptx = path.replace(".cu", ".ptx")
        out_cubin = path.replace(".cu", ".cubin")
        src_dir = os.path.dirname(os.path.abspath(path))
        include_dirs = [
            src_dir,
            os.getcwd(),
            os.path.dirname(src_dir),
        ]
        common_flags = [
            path, f"-arch={arch}",
        ]
        for include_dir in include_dirs:
            common_flags.extend(["-I", include_dir])

        ptx_cmd = ["nvcc", "-ptx", "-lineinfo", *common_flags, "-o", out_ptx]
        cubin_cmd = ["nvcc", "-cubin", "-lineinfo", "--ptxas-options=-v", *common_flags, "-o", out_cubin]
        if verbose:
            print(f"Compilando CUDA para PTX: {' '.join(ptx_cmd)}")
            print(f"Compilando CUDA para PTXAS: {' '.join(cubin_cmd)}")

        ptx_res = subprocess.run(ptx_cmd, capture_output=True, text=True)
        if ptx_res.returncode != 0:
            raise RuntimeError(f"Erro ao compilar {path}:\n{ptx_res.stderr}")

        cubin_res = subprocess.run(cubin_cmd, capture_output=True, text=True)
        if cubin_res.returncode != 0 and verbose:
            print(f"[ptx_analyzer] aviso: compilação cubin falhou:\n{cubin_res.stderr}")

        with open(out_ptx, "r", encoding="utf-8", errors="replace") as handle:
            code = handle.read()
        ptxas_stderr = cubin_res.stderr if cubin_res.returncode == 0 else ""
        return code, out_ptx, os.path.abspath(path), ptxas_stderr

    @classmethod
    def from_file(cls,
                  path: str,
                  kernel_index: int = 0,
                  kernel_name: Optional[str] = None,
                  arch: str = "sm_75",
                  verbose: bool = False) -> "PTXAnalyzer":
        source_path = None
        ptx_path = path
        ptxas_stderr = ""

        if path.endswith(".cu"):
            code, ptx_path, source_path, ptxas_stderr = cls._compile_cuda_to_ptx(path, arch, verbose)
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                code = handle.read()

            has_lineinfo = re.search(r"(?m)^\s*\.loc\s+\d+\s+\d+", code) is not None
            source_path = cls._find_related_source(path)
            if source_path is not None:
                try:
                    code, ptx_path, source_path, ptxas_stderr = cls._compile_cuda_to_ptx(source_path, arch, verbose)
                except Exception:
                    if not has_lineinfo:
                        raise

        analyzer = cls(code, kernel_index=kernel_index, kernel_name=kernel_name)
        analyzer._origin_path = os.path.abspath(path)
        analyzer._ptx_path = os.path.abspath(ptx_path)
        analyzer._source_path = os.path.abspath(source_path) if source_path else None
        analyzer._ptxas_stderr = ptxas_stderr
        analyzer._attach_ptxas_info(ptxas_stderr)
        return analyzer

    @classmethod
    def from_string(cls,
                    ptx_code: str,
                    kernel_index: int = 0,
                    kernel_name: Optional[str] = None) -> "PTXAnalyzer":
        return cls(ptx_code, kernel_index=kernel_index, kernel_name=kernel_name)

    @staticmethod
    def _find_related_source(path: str) -> Optional[str]:
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
        for directory in search_dirs:
            candidate = os.path.normpath(os.path.join(directory, f"{stem}.cu"))
            if os.path.exists(candidate):
                return candidate
        return None

    def _attach_ptxas_info(self, stderr_text: str) -> None:
        if not stderr_text:
            return
        infos = parse_ptxas_output(stderr_text)
        if not infos:
            return
        for kernel in self._all_kernels:
            info = infos.get(_normalize_kernel_name(kernel.name))
            if info is not None:
                kernel.ptxas_info = info

    def _infer_algorithm_name(self) -> Optional[str]:
        candidates = []
        if self._origin_path:
            candidates.append(os.path.splitext(os.path.basename(self._origin_path))[0])
        if self._source_path:
            candidates.append(os.path.splitext(os.path.basename(self._source_path))[0])
        candidates.append(self.kernel.name)

        for candidate in candidates:
            text = candidate.lower()
            for name in ("bubble", "insertion", "merge", "oddeven", "odd_even", "quick", "radix", "selection", "baseline"):
                if name in text:
                    return "oddeven" if name == "odd_even" else name
        return None

    def _infer_strategy_name(self) -> Optional[str]:
        text = f"{self.kernel.name} {self._origin_path or ''} {self._source_path or ''}".lower()
        for strategy in ("register", "shared", "global"):
            if strategy in text:
                return strategy
        return None

    def attach_benchmark_csv(self, csv_path: str) -> BenchmarkSuite:
        self.benchmark_suite = load_benchmark_csv(csv_path)
        return self.benchmark_suite

    def attach_benchmark_output(self, output: str) -> BenchmarkSuite:
        self.benchmark_suite = parse_benchmark_output(output)
        return self.benchmark_suite

    def benchmark_rows(self,
                       algorithm: Optional[str] = None,
                       strategy: Optional[str] = None):
        if self.benchmark_suite is None:
            return []
        algorithm = algorithm or self._infer_algorithm_name()
        strategy = strategy or self._infer_strategy_name()
        return self.benchmark_suite.filter(algorithm=algorithm, strategy=strategy)

    def _format_stats_text(self) -> str:
        k = self.kernel
        p = k.ptxas_info
        lines = [
            "",
            "═" * 68,
            f"  Kernel: {k.name}",
            "═" * 68,
            f"  Instruções totais        : {k.total_instructions}",
            f"  Registradores declarados : {k.total_registers}",
            f"  Registradores ptxas      : {k.registers_per_thread}",
            f"  Branches condicionais    : {k.predicated_branches}",
            f"  Branch ratio             : {k.branch_ratio:.1%}",
            f"  ld.global / st.global    : {k.global_loads} / {k.global_stores}",
            f"  shared / local           : {k.shared_accesses} / {k.local_accesses}",
            f"  Basic blocks / loops     : {k.basic_block_count} / {k.cfg_loop_count}",
            f"  Int. aritmética          : {k.arithmetic_intensity:.3f}",
            f"  Arithmetic ratio         : {k.arithmetic_ratio:.1%}",
            f"  Instruction intensity    : {k.instruction_intensity:.3f}",
        ]
        if p is not None:
            lines.extend([
                "─" * 68,
                f"  ptxas smem / cmem / lmem: {p.shared_mem_bytes} / {p.constant_mem_bytes} / {p.local_mem_bytes} bytes",
                f"  ptxas stack frame       : {p.stack_frame_bytes} bytes",
                f"  ptxas spill stores/loads: {p.spill_stores_bytes} / {p.spill_loads_bytes} bytes",
            ])
        lines.append("")
        return "\n".join(lines)

    def _format_ptxas_text(self) -> str:
        p = self.kernel.ptxas_info
        lines = [
            "",
            _section(f"PTXAS — {self.kernel.name}", 72),
        ]
        if p is None:
            lines.append("  Nenhum dado do ptxas disponível para este kernel.")
            lines.append("")
            return "\n".join(lines)

        lines.extend([
            f"  Registradores reais : {p.registers}",
            f"  Shared memory       : {p.shared_mem_bytes} bytes",
            f"  Constant memory     : {p.constant_mem_bytes} bytes",
            f"  Local memory        : {p.local_mem_bytes} bytes",
            f"  Stack frame         : {p.stack_frame_bytes} bytes",
            f"  Spill stores        : {p.spill_stores_bytes} bytes",
            f"  Spill loads         : {p.spill_loads_bytes} bytes",
        ])
        if p.raw_lines:
            lines.append("")
            lines.append("  Saída bruta do ptxas:")
            for line in p.raw_lines:
                lines.append(f"    {line}")
        lines.append("")
        return "\n".join(lines)

    def _format_hotspots_text(self, max_items: int = 10) -> str:
        cfg = analyze_control_flow(self.kernel)
        sites = sorted(
            cfg.branch_sites,
            key=lambda site: (
                {"high": 3, "medium": 2, "low": 1, "none": 0}.get(site.divergence_risk, 0),
                site.taken_memory_ops + site.fallthrough_memory_ops,
            ),
            reverse=True,
        )[:max_items]
        hotspots = cfg.memory_hotspots[:max_items]

        lines = [
            "",
            _section(f"Hotspots — {self.kernel.name}", 76),
            "  Branches:",
        ]
        if not sites:
            lines.append("    nenhum branch `BRA` encontrado")
        for idx, site in enumerate(sites, 1):
            loc = f"linha .cu {site.source_line}" if site.source_line > 0 else f"PTX:{site.line_no}"
            lines.append(
                f"    B{idx:02d} {site.block_label:<18} {loc:<14} risco={site.divergence_risk:<6} "
                f"taken={site.taken_target or '-':<16} fall={site.fallthrough_target or '-'}"
            )

        lines.append("")
        lines.append("  Memória:")
        if not hotspots:
            lines.append("    nenhum bloco com carga/armazenamento relevante")
        for idx, hotspot in enumerate(hotspots, 1):
            loc = f"linha .cu {hotspot.source_line}" if hotspot.source_line > 0 else hotspot.block_label
            lines.append(
                f"    M{idx:02d} {loc:<24} mem_ops={hotspot.memory_ops:<3} "
                f"densidade={hotspot.memory_density:.1%} "
                f"gld={hotspot.global_loads} gst={hotspot.global_stores} "
                f"shared={hotspot.shared_accesses} local={hotspot.local_accesses}"
            )
        lines.append("")
        return "\n".join(lines)

    def _format_benchmark_text(self) -> str:
        rows = self.benchmark_rows()
        lines = [
            "",
            _section(f"Benchmark — {self.kernel.name}", 72),
        ]
        if not rows:
            lines.append("  Nenhum benchmark ligado a este kernel.")
            lines.append("")
            return "\n".join(lines)

        best_time = min(row.time_ms for row in rows)
        for row in rows:
            marker = "★" if row.time_ms == best_time else " "
            lines.append(
                f"  {marker} {row.algorithm:<10} {row.strategy:<8} "
                f"segmento={row.segment_size:<3} tempo={row.time_ms:>8.3f} ms "
                f"validacao={row.validation:<6} vs_baseline={row.baseline_delta_percent:>8.2f}%"
            )
        lines.append("")
        return "\n".join(lines)

    def _format_summary_text(self) -> str:
        k = self.kernel
        mix = sorted(k.category_counts.items(), key=lambda item: -item[1])
        max_c = max((count for _, count in mix), default=1)
        lines = [
            "╔" + "═" * 62 + "╗",
            f"║  PTX Kernel: {k.name:<48} ║",
            "╚" + "═" * 62 + "╝",
            self._format_stats_text().rstrip(),
            _section("Mix de Instruções", 72),
        ]
        for cat, count in mix:
            pct = count / max(k.total_instructions, 1) * 100.0
            lines.append(f"  {cat:<14} {pct:5.1f}%  {_bar(count, max_c)}  {count:>4}")
        lines.append(self._format_ptxas_text().rstrip())
        lines.append(self._format_benchmark_text().rstrip())
        lines.append("")
        return "\n".join(lines)

    def _format_mermaid_text(self, max_decisions: int = 0) -> str:
        cfg = analyze_control_flow(self.kernel)
        if not cfg.branch_sites:
            return (
                "graph TD\n"
                "    Start([START]) --> Entry([ENTRY])\n"
                "    Entry --> Exit([Sem BRA])\n"
                "    style Start fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#111827;\n"
            )

        visible_sites = cfg.branch_sites if max_decisions <= 0 else cfg.branch_sites[:max_decisions]
        anchor_labels = {"__ENTRY__"}
        for site in visible_sites:
            anchor_labels.add(site.block_label)
            if site.taken_target:
                anchor_labels.add(site.taken_target)
            if site.fallthrough_target:
                anchor_labels.add(site.fallthrough_target)

        def _expand_visible_labels() -> set[str]:
            if max_decisions <= 0:
                return set(cfg.order)

            visible = set(anchor_labels)
            if visible:
                last_anchor_idx = max(
                    idx for idx, label in enumerate(cfg.order) if label in visible
                )
                visible.update(cfg.order[:last_anchor_idx + 1])
            terminals = {label for label in cfg.order if _is_visual_terminal(label)}

            for label in list(anchor_labels):
                block = cfg.blocks.get(label)
                if block is None:
                    continue

                for _, start in block.exits:
                    if start not in cfg.blocks or start in visible:
                        continue

                    queue = deque([(start, [start])])
                    seen = {start}
                    connector_path = None

                    while queue:
                        current, path = queue.popleft()
                        if current in anchor_labels or current in terminals:
                            connector_path = path
                            break

                        for _, nxt in cfg.blocks[current].exits:
                            if nxt not in cfg.blocks or nxt in seen:
                                continue
                            seen.add(nxt)
                            queue.append((nxt, path + [nxt]))

                    if connector_path:
                        visible.update(connector_path)

            return visible

        alias = {label: f"N{idx}" for idx, label in enumerate(cfg.order, 1)}
        lines = ["graph TD", "    Start([START])"]

        def _is_visual_terminal(label: str) -> bool:
            block = cfg.blocks[label]
            if not block.instructions:
                return False
            return block.instructions[-1].op_base in ("ret", "exit")

        def _escape_mermaid_text(text: str) -> str:
            return (
                text
                .replace("\\", "\\\\")
                .replace('"', "'")
                .replace("[", "&#91;")
                .replace("]", "&#93;")
                .replace("{", "&#123;")
                .replace("}", "&#125;")
                .replace("(", "&#40;")
                .replace(")", "&#41;")
            )

        visible_labels = _expand_visible_labels()

        for label in cfg.order:
            if label not in visible_labels:
                continue
            block = cfg.blocks[label]
            if label == "__ENTRY__":
                text = "ENTRY"
            elif _is_visual_terminal(label):
                last = block.instructions[-1] if block.instructions else None
                text = f"{label}<br>{last.op_base if last else 'end'}"
            else:
                last = block.instructions[-1] if block.instructions else None
                loc = ""
                if last is not None:
                    if last.source_line > 0:
                        loc = f"linha {last.source_line}<br>"
                    else:
                        loc = f"PTX {last.line_no}<br>"
                last_text = last.raw.strip().replace('"', "'") if last else ""
                text = f"{label}<br>{loc}{last_text}"
            text = _escape_mermaid_text(text)
            shape = f'(["{text}"])' if _is_visual_terminal(label) else f'["{text}"]'
            lines.append(f"    {alias[label]}{shape}")

        if "__ENTRY__" in alias and "__ENTRY__" in visible_labels:
            lines.append(f"    Start --> {alias['__ENTRY__']}")

        emitted = set()

        # Usa o CFG visível inteiro, não só os branch sites. Isso preserva os
        # blocos sequenciais intermediários e evita componentes desconexos.
        for label in cfg.order:
            if label not in visible_labels:
                continue
            block = cfg.blocks[label]
            src = alias[label]
            fallthrough_edges = []
            other_edges = []

            for edge_type, target in block.exits:
                if target not in alias or target not in visible_labels:
                    continue
                edge_key = (label, target, edge_type)
                if edge_key in emitted:
                    continue
                if edge_type == "fallthrough":
                    fallthrough_edges.append((edge_type, target))
                else:
                    other_edges.append((edge_type, target))

            for edge_type, target in fallthrough_edges + other_edges:
                emitted.add((label, target, edge_type))
                edge_label = "taken" if edge_type == "conditional" else edge_type
                lines.append(f'    {src} -- "{edge_label}" --> {alias[target]}')

        for label in cfg.order:
            if label not in visible_labels or label not in alias:
                continue
            block = cfg.blocks[label]
            if _is_visual_terminal(label):
                lines.append(f"    style {alias[label]} fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#111827;")
            elif any(edge_type == "conditional" for edge_type, _ in block.exits):
                lines.append(f"    style {alias[label]} fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#111827;")
            else:
                lines.append(f"    style {alias[label]} fill:#f8fafc,stroke:#60a5fa,stroke-width:1px,color:#111827;")

        lines.append("    style Start fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#111827;")

        return "\n".join(lines) + "\n"

    def _infer_source_path(self) -> Optional[str]:
        if self._source_path and os.path.exists(self._source_path):
            return self._source_path
        for path in self.kernel.file_map.values():
            if path and path.endswith(".cu") and os.path.exists(path):
                self._source_path = os.path.abspath(path)
                return self._source_path
        if self._origin_path:
            self._source_path = self._find_related_source(self._origin_path)
        return self._source_path

    def report(self, section: str = "summary", mode: str = "text", max_items: int = 10):
        section = section.lower()
        if section == "summary":
            return emit_text(self._format_summary_text(), mode=mode)
        if section == "stats":
            return emit_text(self._format_stats_text(), mode=mode)
        if section == "ptxas":
            return emit_text(self._format_ptxas_text(), mode=mode)
        if section == "hotspots":
            return emit_text(self._format_hotspots_text(max_items=max_items), mode=mode)
        if section == "benchmark":
            return emit_text(self._format_benchmark_text(), mode=mode)
        raise ValueError(f"Seção desconhecida: {section}")

    def summary(self):
        return self.report(section="summary", mode="text")

    def show_stats(self):
        return self.report(section="stats", mode="text")

    def show_top_opcodes(self, n: int = 10):
        k = self.kernel
        counts = Counter(instr.op for instr in k.instructions)
        top = counts.most_common(n)
        total = max(k.total_instructions, 1)
        max_c = top[0][1] if top else 1
        pad = max((len(op) for op, _ in top), default=20)

        print(f"\n{_section(f'Top {n} Opcodes — {k.name}', 70)}")
        for rank, (op, count) in enumerate(top, 1):
            pct = count / total * 100.0
            print(f"  {rank:>2}. {op:<{pad}} {count:>5} {pct:5.1f}% {_bar(count, max_c)}")
        print()

    def hotspots_report(self, mode: str = "text", max_items: int = 10):
        if mode == "data":
            cfg = analyze_control_flow(self.kernel)
            return {
                "branch_sites": [site.to_dict() for site in cfg.branch_sites[:max_items]],
                "memory_hotspots": [item.to_dict() for item in cfg.memory_hotspots[:max_items]],
            }
        return self.report(section="hotspots", mode=mode, max_items=max_items)

    def flowchart(self, mode: str = "html", max_decisions: int = 0):
        return self.control_flow(mode=mode, max_decisions=max_decisions)

    def control_flow(self, mode: str = "html", max_decisions: int = 0):
        mode = mode.lower()
        graph = self._format_mermaid_text(max_decisions=max_decisions)
        if mode == "data":
            return {
                "mermaid": graph,
                "control_flow": analyze_control_flow(self.kernel).to_dict(),
            }
        if mode == "raw":
            return graph
        if mode == "text":
            return emit_text(f"```mermaid\n{graph}```", mode="text")
        if mode == "html":
            try:
                from IPython.display import HTML, display
                display(HTML(mermaid_block_html(graph, title=self.kernel.name)))
                return None
            except Exception:
                return emit_text(f"```mermaid\n{graph}```", mode="text")
        raise ValueError(f"Modo desconhecido: {mode}")

    @property
    def kernel_count(self) -> int:
        return len(self._all_kernels)

    @property
    def kernel_names(self):
        return [kernel.name for kernel in self._all_kernels]

    def select_kernel(self,
                      kernel_index: int = 0,
                      kernel_name: Optional[str] = None) -> "PTXAnalyzer":
        item = PTXAnalyzer(
            self._code,
            kernel_index=kernel_index,
            kernel_name=kernel_name,
        )
        item._origin_path = self._origin_path
        item._source_path = self._source_path
        item._ptx_path = self._ptx_path
        item._ptxas_stderr = self._ptxas_stderr
        item.runtime_profile = self.runtime_profile
        item.benchmark_suite = self.benchmark_suite
        for src_kernel, dst_kernel in zip(self._all_kernels, item._all_kernels):
            dst_kernel.ptxas_info = src_kernel.ptxas_info
        return item

    def iter_kernel_analyzers(self):
        analyzers = []
        for idx, kernel in enumerate(self._all_kernels):
            item = PTXAnalyzer(self._code, kernel_index=idx)
            item._origin_path = self._origin_path
            item._source_path = self._source_path
            item._ptx_path = self._ptx_path
            item._ptxas_stderr = self._ptxas_stderr
            item.runtime_profile = self.runtime_profile
            item.benchmark_suite = self.benchmark_suite
            kernel_info = kernel.ptxas_info
            if kernel_info is not None:
                item.kernel.ptxas_info = kernel_info
            analyzers.append(item)
        return analyzers

    def compare_kernels_in_file(self):
        from .comparator import PTXComparator

        comp = PTXComparator()
        for analyzer in self.iter_kernel_analyzers():
            label = analyzer._infer_strategy_name() or analyzer.kernel.name
            comp.add(label, analyzer)
        return comp

    def profile_runtime(self, sizes=(1024,), repeats: int = 3, arch: str = "sm_75",
                        source_path: Optional[str] = None,
                        executable_path: Optional[str] = None,
                        extra_compile_flags=None,
                        extra_run_args=None):
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

    def to_dict(self) -> dict:
        return {
            "kernel": self.kernel.name,
            "metrics": self.kernel.metrics_dict(),
            "categories": self.kernel.category_counts,
            "registers_declared": {rtype: sorted(regs) for rtype, regs in self.kernel.reg_decls.items()},
            "ptxas": self.kernel.ptxas_info.to_dict() if self.kernel.ptxas_info else None,
            "benchmark": [row.to_dict() for row in self.benchmark_rows()],
            "runtime": self.runtime_profile.to_dict() if self.runtime_profile else None,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
