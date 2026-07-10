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

from .core import PTXKernel, analyze_control_flow, explain_instruction, _block_primary_source_line
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
            condition = explain_instruction(
                next(
                    (instr for instr in self.kernel.instructions if instr.line_no == site.setp_line),
                    None,
                )
            ) if site.setp_line > 0 else "sem setp associado"
            lines.append(
                f"    B{idx:02d} {site.block_label:<18} {loc:<14} risco={site.divergence_risk:<6} "
                f"taken={site.taken_target or '-':<16} fall={site.fallthrough_target or '-'}"
            )
            lines.append(f"         {condition}")

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

    def _format_linear_execution_mermaid(self) -> str:
        source_lines = self._load_source_lines()

        def _classify_linear_step(instr):
            if instr.op_base == "ld" and "param" in instr.op:
                return "params"
            if instr.op_base in {"mov", "mad", "mul", "cvt"}:
                return "index"
            if instr.op_base == "cvta":
                return "address"
            if instr.op_base == "add" and any(op.startswith("%rd") for op in instr.operands[:2]):
                return "address"
            if instr.op_base == "shl" and any(op.startswith("%rd") for op in instr.operands[:1]):
                return "address"
            if instr.op_base == "ld" and "global" in instr.op:
                return "load"
            if instr.op_base == "setp":
                return "compare"
            if instr.op_base == "selp":
                return "select"
            if instr.op_base in {"add", "sub"}:
                return "compute"
            if instr.op_base == "call":
                return "call"
            if instr.op_base == "st" and "global" in instr.op:
                return "store"
            if instr.op_base in {"ret", "exit"}:
                return "exit"
            return None

        grouped = []
        current_tag = None
        current_instrs = []
        for instr in self.kernel.instructions:
            tag = _classify_linear_step(instr)
            if tag is None:
                continue
            if tag != current_tag and current_instrs:
                grouped.append((current_tag, current_instrs))
                current_instrs = []
            current_tag = tag
            current_instrs.append(instr)
        if current_instrs:
            grouped.append((current_tag, current_instrs))

        selected = [
            (tag, instrs)
            for tag, instrs in grouped
            if tag in {"compare", "select", "compute", "call", "store", "exit"}
        ]
        if not selected:
            self._last_linear_steps = []
            self._last_linear_metadata = {
                "has_predicated_selection": False,
                "has_compare": False,
                "has_select": False,
                "note": "",
            }
            return (
                "graph LR\n"
                "    Start([START]) --> Exit([Sem etapas semânticas relevantes])\n"
                "    style Start fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#111827;\n"
            )

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

        def _linear_title(tag: str) -> str:
            return {
                "compare": "Decisão",
                "select": "Escolha",
                "compute": "Atualização",
                "call": "Ação",
                "store": "Resultado",
                "exit": "Saída",
            }.get(tag, tag)

        def _linear_description(tag: str, instrs) -> str:
            instr = instrs[-1]
            if tag == "compare":
                return "Compara o valor com o limiar"
            if tag == "select":
                return "Escolhe entre incrementar ou decrementar"
            if tag == "compute":
                return "Aplica o ajuste ao valor"
            if tag == "call":
                return explain_instruction(instr)
            if tag == "store":
                return "Produz o resultado final"
            return explain_instruction(instr)

        def _linear_instruction_name(instrs) -> str:
            if not instrs:
                return ""
            op_names = []
            for instr in instrs:
                if instr.op_base in {"setp", "selp", "add", "sub", "mul", "mad", "fma", "ld", "st", "ret", "exit"}:
                    op_names.append(instr.op)
            if not op_names:
                op_names.append(instrs[-1].op)
            return " + ".join(dict.fromkeys(op_names))

        def _linear_source_info(instrs) -> tuple[int, str]:
            for instr in reversed(instrs):
                if instr.source_line > 0 and instr.op_base not in {"ret", "exit"}:
                    return instr.source_line, source_lines.get(instr.source_line, "")
            for instr in reversed(instrs):
                if instr.source_line > 0:
                    return instr.source_line, source_lines.get(instr.source_line, "")
            return 0, ""

        has_compare = any(tag == "compare" for tag, _ in selected)
        has_select = any(tag == "select" for tag, _ in selected)
        has_predicated_selection = has_compare and has_select
        note = ""
        if has_predicated_selection:
            note = (
                "O PTX não gerou dois caminhos de controle separados. "
                "O compilador linearizou o if/else em comparação + seleção predicada "
                "(por exemplo, setp/selp), então a decisão aparece como escolha de dados, "
                "não como salto bra."
            )

        self._last_linear_steps = []
        for idx, (tag, instrs) in enumerate(selected, 1):
            source_line, source_code = _linear_source_info(instrs)
            raw_instruction_name = _linear_instruction_name(instrs)
            self._last_linear_steps.append({
                "label": f"__linear_{idx}__",
                "title": _linear_title(tag),
                "instruction_name": self._friendly_instruction_caption(raw_instruction_name),
                "raw_instruction_name": raw_instruction_name,
                "description": _linear_description(tag, instrs),
                "tag": tag,
                "source_line": source_line,
                "source_code": source_code,
            })
        self._last_linear_metadata = {
            "has_predicated_selection": has_predicated_selection,
            "has_compare": has_compare,
            "has_select": has_select,
            "note": note,
        }

        lines = ["graph LR", "    Start([START])"]
        prev = "Start"
        for idx, (tag, instrs) in enumerate(selected, 1):
            node_id = f"L{idx}"
            raw_instr_name = _linear_instruction_name(instrs)
            ptx_line = f"<br>PTX: {raw_instr_name}" if raw_instr_name else ""
            text = _escape_mermaid_text(
                f"{_linear_title(tag)}<br>{_linear_description(tag, instrs)}{ptx_line}"
            )
            is_terminal = tag == "exit"
            shape = f'(["{text}"])' if is_terminal else f'["{text}"]'
            lines.append(f"    {node_id}{shape}")
            lines.append(f"    {prev} --> {node_id}")
            prev = node_id

        lines.append("    style Start fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#111827;")
        for idx, (tag, instrs) in enumerate(selected, 1):
            node_id = f"L{idx}"
            if tag == "exit":
                lines.append(f"    style {node_id} fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#111827;")
            elif tag == "store":
                lines.append(f"    style {node_id} fill:#ecfeff,stroke:#0891b2,stroke-width:1px,color:#111827;")
            elif tag in {"compare", "select"}:
                lines.append(f"    style {node_id} fill:#fee2e2,stroke:#dc2626,stroke-width:1px,color:#111827;")
            else:
                lines.append(f"    style {node_id} fill:#f8fafc,stroke:#60a5fa,stroke-width:1px,color:#111827;")
        return "\n".join(lines) + "\n"

    def _format_mermaid_text(self, max_decisions: int = 0) -> str:
        cfg = analyze_control_flow(self.kernel)
        if not cfg.branch_sites:
            return self._format_linear_execution_mermaid()

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
        lines = ["graph LR", "    Start([START])"]

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

        def _is_decision_block(label: str) -> bool:
            block = cfg.blocks[label]
            return any(edge_type == "conditional" for edge_type, _ in block.exits)

        def _decision_source_key(label: str) -> tuple[str, int]:
            block = cfg.blocks[label]
            source_guided = self._source_guided_block_text(block)
            source_line = int(source_guided.get("source_line", 0) or 0)
            inline_line = int(source_guided.get("inline_source_line", 0) or 0)
            source_code = (source_guided.get("source_code") or "").strip()
            inline_source_code = (source_guided.get("inline_source_code") or "").strip()
            reference_code = source_code or inline_source_code
            has_compound_operator = ("&&" in reference_code) or ("||" in reference_code)

            # Só tratamos como "Decisão Composta" quando a origem realmente
            # parece ser uma condição composta no C++ original. Se não houver
            # código-fonte disponível, mantemos o fallback pela linha.
            if reference_code and not has_compound_operator:
                return ("none", 0)
            if source_line > 0:
                return ("source", source_line)
            if inline_line > 0:
                return ("inline", inline_line)
            return ("none", 0)

        def _compound_decision_groups() -> list[list[str]]:
            groups: list[list[str]] = []
            current: list[str] = []
            current_key: tuple[str, int] | None = None
            for label in cfg.order:
                if label not in visible_labels:
                    if len(current) > 1:
                        groups.append(current[:])
                    current = []
                    current_key = None
                    continue
                if _is_decision_block(label):
                    decision_key = _decision_source_key(label)
                    if decision_key == ("none", 0):
                        if len(current) > 1:
                            groups.append(current[:])
                        current = []
                        current_key = None
                        continue
                    if not current:
                        current = [label]
                        current_key = decision_key
                    elif decision_key == current_key:
                        current.append(label)
                    else:
                        if len(current) > 1:
                            groups.append(current[:])
                        current = [label]
                        current_key = decision_key
                else:
                    if len(current) > 1:
                        groups.append(current[:])
                    current = []
                    current_key = None
            if len(current) > 1:
                groups.append(current[:])
            return groups

        def _unroll_groups() -> list[tuple[list[str], int, int]]:
            """Detecta laços desenrolados (loop unrolling) pelo compilador:
            um loop_site sem nenhum outro laço aninhado dentro dele, cujo
            corpo repete um mesmo trecho de código-fonte várias vezes. Cada
            grupo vira uma caixinha "Desenrolado por fator N (linha L)"."""
            order_index = {label: idx for idx, label in enumerate(cfg.order)}

            def _span_of(loop) -> Optional[tuple[int, int]]:
                header_idx = order_index.get(loop.header)
                latch_idx = order_index.get(loop.latch)
                if header_idx is None or latch_idx is None or latch_idx < header_idx:
                    return None
                return header_idx, latch_idx

            groups: list[tuple[list[str], int, int]] = []
            for loop in cfg.loop_sites:
                span_range = _span_of(loop)
                if span_range is None:
                    continue
                header_idx, latch_idx = span_range
                nested = False
                for other in cfg.loop_sites:
                    if other is loop:
                        continue
                    other_range = _span_of(other)
                    if other_range is None:
                        continue
                    other_header_idx, other_latch_idx = other_range
                    if header_idx <= other_header_idx and other_latch_idx <= latch_idx and other_range != span_range:
                        nested = True
                        break
                if nested:
                    continue

                span = [
                    label for label in cfg.order[header_idx:latch_idx + 1]
                    if label in visible_labels
                ]
                if len(span) < 2:
                    continue

                line_counts: dict[int, int] = {}
                for label in span:
                    line = _block_primary_source_line(cfg.blocks[label])
                    if line > 0:
                        line_counts[line] = line_counts.get(line, 0) + 1
                if not line_counts:
                    continue
                body_line, factor = max(line_counts.items(), key=lambda item: item[1])
                if factor < 2:
                    continue
                groups.append((span, factor, body_line))
            return groups

        def _remainder_groups(
            unroll_groups: list[tuple[list[str], int, int]]
        ) -> list[tuple[list[str], int]]:
            """Localiza o "resto" de um laço desenrolado: o trecho reto
            (sem back-edge) logo após o corpo desenrolado, que trata as
            iterações que sobram quando o total não é múltiplo do fator de
            unroll (até fator-1 iterações extras)."""
            order_index = {label: idx for idx, label in enumerate(cfg.order)}
            branch_by_label = {site.block_label: site for site in cfg.branch_sites}

            groups: list[tuple[list[str], int]] = []
            for span, factor, _body_line in unroll_groups:
                header, latch = span[0], span[-1]
                latch_block = cfg.blocks.get(latch)
                if latch_block is None:
                    continue
                exit_target = next(
                    (target for _, target in latch_block.exits if target != header),
                    None,
                )
                if exit_target is None or exit_target not in order_index:
                    continue
                exit_site = branch_by_label.get(exit_target)
                join_target = exit_site.reconvergence_target if exit_site else None
                if not join_target or join_target not in order_index:
                    continue
                start_idx = order_index[exit_target]
                end_idx = order_index[join_target]
                if end_idx <= start_idx:
                    continue
                remainder = [
                    label for label in cfg.order[start_idx:end_idx]
                    if label in visible_labels
                ]
                if len(remainder) < 2:
                    continue
                groups.append((remainder, factor - 1))
            return groups

        def _resolve_visual_target(target: str) -> str:
            seen = set()
            current = target
            while current in cfg.blocks and current not in seen:
                seen.add(current)
                block = cfg.blocks[current]
                if block.display_name.startswith("Salto") and len(block.exits) == 1:
                    current = block.exits[0][1]
                    continue
                break
            return current

        visible_labels = {
            label for label in visible_labels
            if not (label in cfg.blocks and cfg.blocks[label].display_name.startswith("Salto"))
        }

        grouped_labels = set()
        for group_idx, group in enumerate(_compound_decision_groups(), 1):
            lines.append(f'    subgraph DECISION_GROUP_{group_idx}["Decisão Composta {group_idx}"]')
            for label in group:
                grouped_labels.add(label)
                block = cfg.blocks[label]
                source_guided = self._source_guided_block_text(block)
                raw_op_name = self._display_ptx_name(source_guided.get("raw_instruction_name") or "")
                ptx_line = f"<br>PTX: {raw_op_name}" if raw_op_name else ""
                text = (
                    f"{source_guided['title'] or label}<br>"
                    f"{source_guided['description'] or 'Executando etapa da decisão'}"
                    f"{ptx_line}"
                )
                text = _escape_mermaid_text(text)
                shape = f'(["{text}"])' if _is_visual_terminal(label) else f'["{text}"]'
                lines.append(f"        {alias[label]}{shape}")
            lines.append("    end")

        rendered_unroll_groups: list[tuple[list[str], int, int]] = []
        for group_idx, (span, factor, body_line) in enumerate(_unroll_groups(), 1):
            if any(label in grouped_labels for label in span):
                continue
            rendered_unroll_groups.append((span, factor, body_line))
            lines.append(
                f'    subgraph UNROLL_GROUP_{group_idx}["Desenrolado por fator {factor} (linha {body_line})"]'
            )
            for label in span:
                grouped_labels.add(label)
                block = cfg.blocks[label]
                source_guided = self._source_guided_block_text(block)
                raw_op_name = self._display_ptx_name(source_guided.get("raw_instruction_name") or "")
                ptx_line = f"<br>PTX: {raw_op_name}" if raw_op_name else ""
                last = block.instructions[-1] if block.instructions else None
                details = source_guided["description"] or (last.raw.strip().replace('"', "'") if last else "")
                text = f"{source_guided['title'] or label}<br>{details}{ptx_line}"
                text = _escape_mermaid_text(text)
                shape = f'(["{text}"])' if _is_visual_terminal(label) else f'["{text}"]'
                lines.append(f"        {alias[label]}{shape}")
            lines.append("    end")

        for group_idx, (remainder, max_extra) in enumerate(_remainder_groups(rendered_unroll_groups), 1):
            if any(label in grouped_labels for label in remainder):
                continue
            extra_label = "iteração restante" if max_extra == 1 else "iterações restantes"
            lines.append(
                f'    subgraph REMAINDER_GROUP_{group_idx}["Resto do desenrolamento (até {max_extra} {extra_label})"]'
            )
            for label in remainder:
                grouped_labels.add(label)
                block = cfg.blocks[label]
                source_guided = self._source_guided_block_text(block)
                raw_op_name = self._display_ptx_name(source_guided.get("raw_instruction_name") or "")
                ptx_line = f"<br>PTX: {raw_op_name}" if raw_op_name else ""
                last = block.instructions[-1] if block.instructions else None
                details = source_guided["description"] or (last.raw.strip().replace('"', "'") if last else "")
                text = f"{source_guided['title'] or label}<br>{details}{ptx_line}"
                text = _escape_mermaid_text(text)
                shape = f'(["{text}"])' if _is_visual_terminal(label) else f'["{text}"]'
                lines.append(f"        {alias[label]}{shape}")
            lines.append("    end")

        for label in cfg.order:
            if label not in visible_labels:
                continue
            if label in grouped_labels:
                continue
            block = cfg.blocks[label]
            source_guided = self._source_guided_block_text(block)
            raw_op_name = self._display_ptx_name(source_guided.get("raw_instruction_name") or "")
            ptx_line = f"<br>PTX: {raw_op_name}" if raw_op_name else ""
            if label == "__ENTRY__":
                body = source_guided["description"] or "Inicializando contexto do kernel"
                text = f"{source_guided['title'] or 'Entrada'}<br>{body}{ptx_line}"
            elif _is_visual_terminal(label):
                last = block.instructions[-1] if block.instructions else None
                body = source_guided["description"] or (last.op_base if last else "end")
                text = f"{source_guided['title'] or label}<br>{body}{ptx_line}"
            else:
                last = block.instructions[-1] if block.instructions else None
                details = source_guided["description"] or (last.raw.strip().replace('"', "'") if last else "")
                text = f"{source_guided['title'] or label}<br>{details}{ptx_line}"
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
                visual_target = _resolve_visual_target(target)
                if visual_target not in alias or visual_target not in visible_labels:
                    continue
                edge_key = (label, visual_target, edge_type)
                if edge_key in emitted:
                    continue
                if edge_type == "fallthrough":
                    fallthrough_edges.append((edge_type, visual_target))
                else:
                    other_edges.append((edge_type, visual_target))

            for edge_type, target in fallthrough_edges + other_edges:
                visual_target = _resolve_visual_target(target)
                if visual_target not in alias or visual_target not in visible_labels:
                    continue
                emitted.add((label, visual_target, edge_type))
                edge_label = (
                    "Sim" if edge_type == "conditional" else
                    "Não" if edge_type == "fallthrough" and any(t == "conditional" for t, _ in block.exits) else
                    "Segue" if edge_type == "fallthrough" else
                    "Vai para"
                )
                lines.append(f'    {src} -- "{edge_label}" --> {alias[visual_target]}')

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

    def _load_source_lines(self) -> dict[int, str]:
        source_path = self._infer_source_path()
        if not source_path or not os.path.exists(source_path):
            return {}
        try:
            with open(source_path, "r", encoding="utf-8", errors="replace") as handle:
                return {
                    idx: line.rstrip()
                    for idx, line in enumerate(handle, 1)
                }
        except Exception:
            return {}

    def _block_source_info(self, block) -> dict:
        source_lines = self._load_source_lines()

        def _pick_representative_line(attr_name: str) -> int:
            for instr in reversed(block.instructions):
                line_no = getattr(instr, attr_name, 0)
                if line_no <= 0:
                    continue
                if instr.op_base in {"ret", "exit"}:
                    continue
                return line_no
            for instr in reversed(block.instructions):
                line_no = getattr(instr, attr_name, 0)
                if line_no > 0:
                    return line_no
            return 0

        source_line = _pick_representative_line("source_line")
        inline_source_line = _pick_representative_line("inline_source_line")

        info = {
            "source_line": source_line,
            "source_code": source_lines.get(source_line, "") if source_line > 0 else "",
            "inline_source_line": inline_source_line,
            "inline_source_code": source_lines.get(inline_source_line, "") if inline_source_line > 0 else "",
        }
        return info

    def _friendly_instruction_caption(self, instruction_name: str, title: str = "") -> str:
        text = (instruction_name or "").strip()
        lowered = text.lower()
        title_lower = (title or "").lower()

        if "setp" in lowered and "bra" in lowered:
            return "comparação condicional"
        if lowered.startswith("setp"):
            return "comparação"
        if lowered.startswith("selp"):
            return "seleção do ajuste"
        if lowered.startswith("ld.global"):
            return "leitura em memória global"
        if lowered.startswith("ld.shared"):
            return "leitura em memória compartilhada"
        if lowered.startswith("st.global"):
            return "escrita em memória global"
        if lowered.startswith("st.shared"):
            return "escrita em memória compartilhada"
        if lowered.startswith("add"):
            return "atualização de valor"
        if lowered.startswith("sub"):
            return "decremento"
        if lowered.startswith("mov"):
            if "sequência" in title_lower:
                return "preparo de registradores"
            return "movimentação de valor"
        if lowered.startswith("ret") or lowered.startswith("exit"):
            return "retorno"
        if lowered.startswith("call"):
            return "chamada auxiliar"
        return text or "etapa"

    def _display_ptx_name(self, raw_instruction_name: str) -> str:
        raw = (raw_instruction_name or "").strip()
        if " + " in raw:
            return raw.split(" + ", 1)[0].strip()
        return raw

    def _extract_source_condition(self, source_line: str) -> str:
        compact = " ".join((source_line or "").split())
        if not compact:
            return ""

        match = re.search(r"\b(if|while)\s*\((.*)\)\s*\{?$", compact)
        if match:
            return match.group(2).strip()

        match = re.search(r"\bfor\s*\(([^;]*);([^;]*);([^\)]*)\)\s*\{?$", compact)
        if match:
            return match.group(2).strip()

        return ""

    def _source_guided_block_text(self, block) -> dict:
        info = self._block_source_info(block)
        source_code = (info.get("source_code") or "").strip()
        inline_source_code = (info.get("inline_source_code") or "").strip()
        reference_code = source_code or inline_source_code

        title = block.display_name
        instruction_name = block.instruction_name
        raw_instruction_name = getattr(block, "raw_instruction_name", block.instruction_name)
        raw_display_name = self._display_ptx_name(raw_instruction_name)
        description = block.description

        if reference_code:
            compact = " ".join(reference_code.split())
            extracted_condition = self._extract_source_condition(compact)

            if compact.startswith("if (base + segment_size > total_elements)"):
                title = "Limite"
                instruction_name = "checagem de fronteira"
                description = "segmento válido"
            elif compact.startswith("for (int end = count - 1; end > 0; --end)"):
                if raw_display_name.startswith("setp"):
                    title = "Laço"
                    instruction_name = "controle de iteração"
                    description = extracted_condition or "end > 0"
                else:
                    title = "Atualização"
                    instruction_name = "avanço do laço"
                    description = "preparo do laço"
            elif compact.startswith("for (int end = segment_size - 1; end > 0; --end)"):
                if raw_display_name.startswith("setp"):
                    title = "Laço"
                    instruction_name = "controle de iteração"
                    description = extracted_condition or "end > 0"
                else:
                    title = "Atualização"
                    instruction_name = "avanço do laço"
                    description = "preparo do laço"
            elif compact.startswith("for (int i = 0; i < end; ++i)"):
                if raw_display_name.startswith("setp"):
                    title = "Laço"
                    instruction_name = "controle de iteração"
                    description = extracted_condition or "i < end"
                else:
                    title = "Atualização"
                    instruction_name = "avanço do índice"
                    description = "preparo do próximo i"
            elif compact.startswith("if (data[base + i] > data[base + i + 1])"):
                if raw_display_name.startswith("setp"):
                    title = "Comparação"
                    instruction_name = "teste de troca"
                    description = "data[i] e data[i+1]"
                elif raw_display_name.startswith(("add", "sub", "mul", "mad", "mov", "ld")):
                    title = "Preparação"
                    instruction_name = "endereçamento"
                    description = "prepara a comparação"
                elif raw_display_name.startswith("st"):
                    title = "Escrita"
                    instruction_name = "troca"
                    description = "troca parcial"
            elif compact.startswith("for (int i = 1; i < count; ++i)"):
                if raw_display_name.startswith("setp"):
                    title = "Laço"
                    instruction_name = "controle de iteração"
                    description = extracted_condition or "i < count"
                else:
                    title = "Atualização"
                    instruction_name = "avanço do índice"
                    description = "preparo do próximo i"
            elif compact.startswith("while (j >= 0 && values[j] > key)"):
                if raw_display_name.startswith("setp"):
                    title = "Laço"
                    instruction_name = "teste de deslocamento"
                    description = extracted_condition or "j >= 0 && values[j] > key"
                else:
                    title = "Preparação"
                    instruction_name = "endereçamento"
                    description = "prepara a verificação"
            elif compact.startswith("data_t key = values[i]"):
                title = "Leitura"
                instruction_name = "captura da chave"
                description = "chave"
            elif compact.startswith("values[j + 1] = values[j]"):
                title = "Escrita"
                instruction_name = "movimento à direita"
                description = "deslocamento"
            elif compact.startswith("values[j + 1] = key"):
                title = "Escrita"
                instruction_name = "posicionamento da chave"
                description = "inserção"
            elif compact.startswith("output[idx] = result"):
                title = "Resultado"
                instruction_name = "escrita final"
                description = "Escreve o resultado final na memória global"
            elif compact.startswith("if (value > threshold)"):
                title = "Decisão"
                instruction_name = "comparação com limiar"
                description = extracted_condition or "value > threshold"
            elif compact.startswith("result = value + 1"):
                title = "Atualização"
                instruction_name = "ajuste positivo"
                description = "incremento"
            elif compact.startswith("result = value - 1"):
                title = "Atualização"
                instruction_name = "ajuste negativo"
                description = "decremento"

        instruction_name = self._friendly_instruction_caption(raw_instruction_name, title)

        if block.is_terminal:
            title = "Saída"
            instruction_name = self._friendly_instruction_caption(raw_instruction_name or "ret", title)
            description = "Encerra a execução do kernel"

        info.update({
            "title": title,
            "instruction_name": instruction_name,
            "raw_instruction_name": raw_instruction_name,
            "repeated_source_instance": getattr(block, "repeated_source_instance", 0),
            "description": description,
        })
        return info

    def _control_flow_data_with_source(self) -> dict:
        cfg = analyze_control_flow(self.kernel)
        data = cfg.to_dict()
        for label in data["order"]:
            block = cfg.blocks[label]
            source_guided = self._source_guided_block_text(block)
            data["blocks"][label].update(source_guided)
        return data

    def _visual_flow_data(self) -> dict:
        cfg = analyze_control_flow(self.kernel)
        if not cfg.branch_sites:
            self._format_linear_execution_mermaid()
            return {
                "kind": "linear",
                "nodes": list(getattr(self, "_last_linear_steps", [])),
                "metadata": dict(getattr(self, "_last_linear_metadata", {})),
            }

        nodes = []
        for label in cfg.order:
            block = cfg.blocks[label]
            if block.display_name.startswith("Salto"):
                continue
            source_guided = self._source_guided_block_text(block)
            node = {
                "label": label,
                "title": source_guided["title"],
                "instruction_name": source_guided["instruction_name"],
                "description": source_guided["description"],
            }
            node.update(source_guided)
            nodes.append(node)
        return {
            "kind": "cfg",
            "nodes": nodes,
        }

    def _dynamic_block_lookup(self, control_flow: dict, visual_flow: dict) -> dict:
        line_to_labels: dict[int, list[str]] = {}
        for label in control_flow.get("order", []):
            block = control_flow["blocks"][label]
            line_no = int(block.get("source_line", 0) or 0)
            if line_no <= 0:
                continue
            line_to_labels.setdefault(line_no, []).append(label)

        visible_labels = set()
        if visual_flow.get("kind") == "cfg":
            visible_labels = {node["label"] for node in visual_flow.get("nodes", [])}
        else:
            for node in visual_flow.get("nodes", []):
                line_no = int(node.get("source_line", 0) or 0)
                label = node.get("label")
                if line_no > 0 and label:
                    line_to_labels.setdefault(line_no, []).append(label)
                    visible_labels.add(label)
        return {
            "line_to_labels": line_to_labels,
            "visible_labels": visible_labels,
        }

    def _resolve_block_from_line(self, line_no: int, lookup: dict, fallback: Optional[str] = None) -> Optional[str]:
        labels = list(lookup.get("line_to_labels", {}).get(line_no, []))
        visible = lookup.get("visible_labels", set())
        if visible:
            visible_matches = [label for label in labels if label in visible]
            if visible_matches:
                return visible_matches[0]
        if labels:
            return labels[0]
        return fallback

    def _format_dynamic_step_mermaid(self,
                                     control_flow: dict,
                                     visual_flow: dict,
                                     active_labels: list[str],
                                     completed_labels: Optional[list[str]],
                                     active_edges: Optional[list[tuple[str, str]]],
                                     completed_edges: Optional[list[tuple[str, str]]],
                                     caption: str) -> str:
        active = set(label for label in active_labels if label)
        completed = set(label for label in (completed_labels or []) if label)
        active_edge_set = set(edge for edge in (active_edges or []) if edge[0] and edge[1])
        completed_edge_set = set(edge for edge in (completed_edges or []) if edge[0] and edge[1])
        if visual_flow.get("kind") == "linear":
            lines = ["graph LR", '    Start([START])']
            prev = "Start"
            edge_index = 0
            for idx, node in enumerate(visual_flow.get("nodes", []), 1):
                node_id = f"L{idx}"
                text = f"{node['title']}<br>{node['description']}"
                shape = f'(["{text}"])' if node["title"] == "Saída" else f'["{text}"]'
                lines.append(f"    {node_id}{shape}")
                lines.append(f"    {prev} --> {node_id}")
                logical_src = "Start" if idx == 1 else visual_flow.get("nodes", [])[idx - 2].get("label")
                logical_dst = node.get("label")
                if (logical_src, logical_dst) in active_edge_set:
                    lines.append(f"    linkStyle {edge_index} stroke:#d97706,stroke-width:4px;")
                elif (logical_src, logical_dst) in completed_edge_set:
                    lines.append(f"    linkStyle {edge_index} stroke:#16a34a,stroke-width:3px;")
                edge_index += 1
                prev = node_id
            lines.append("    Caption[" + '"' + caption.replace('"', "'") + '"' + "]")
            lines.append(f"    {prev} --> Caption")
            lines.append("    style Start fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#111827;")
            for idx, node in enumerate(visual_flow.get("nodes", []), 1):
                node_id = f"L{idx}"
                label = node.get("label")
                if label in active:
                    lines.append(f"    style {node_id} fill:#fef3c7,stroke:#d97706,stroke-width:3px,color:#111827;")
                elif label in completed:
                    lines.append(f"    style {node_id} fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#111827;")
                elif node["title"] == "Saída":
                    lines.append(f"    style {node_id} fill:#e5e7eb,stroke:#94a3b8,stroke-width:1px,color:#111827;")
                else:
                    lines.append(f"    style {node_id} fill:#f8fafc,stroke:#94a3b8,stroke-width:1px,color:#111827;")
            return "\n".join(lines) + "\n"

        node_order = visual_flow.get("nodes", [])
        alias = {node["label"]: f"N{idx}" for idx, node in enumerate(node_order, 1)}
        lines = ["graph LR", '    Start([START])']
        for node in node_order:
            text = f"{node['title']}<br>{node['description']}"
            shape = f'(["{text}"])' if "Saída" in node["title"] else f'["{text}"]'
            lines.append(f"    {alias[node['label']]}{shape}")
        if node_order:
            lines.append(f"    Start --> {alias[node_order[0]['label']]}")
        edge_index = 0
        if node_order:
            first_edge = ("Start", node_order[0]["label"])
            if first_edge in active_edge_set:
                lines.append(f"    linkStyle {edge_index} stroke:#d97706,stroke-width:4px;")
            elif first_edge in completed_edge_set:
                lines.append(f"    linkStyle {edge_index} stroke:#16a34a,stroke-width:3px;")
            edge_index += 1
        emitted = set()
        for edge in control_flow.get("edges", []):
            src = edge["source"]
            dst = edge["target"]
            if src not in alias or dst not in alias:
                continue
            key = (src, dst, edge["edge_type"])
            if key in emitted:
                continue
            emitted.add(key)
            lines.append(f'    {alias[src]} -- "{edge["edge_type"]}" --> {alias[dst]}')
            if (src, dst) in active_edge_set:
                lines.append(f"    linkStyle {edge_index} stroke:#d97706,stroke-width:4px;")
            elif (src, dst) in completed_edge_set:
                lines.append(f"    linkStyle {edge_index} stroke:#16a34a,stroke-width:3px;")
            edge_index += 1
        lines.append("    Caption[" + '"' + caption.replace('"', "'") + '"' + "]")
        if node_order:
            lines.append(f"    {alias[node_order[-1]['label']]} --> Caption")
        lines.append("    style Start fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#111827;")
        for node in node_order:
            node_id = alias[node["label"]]
            if node["label"] in active:
                lines.append(f"    style {node_id} fill:#fef3c7,stroke:#d97706,stroke-width:3px,color:#111827;")
            elif node["label"] in completed:
                lines.append(f"    style {node_id} fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#111827;")
            elif "Saída" in node["title"]:
                lines.append(f"    style {node_id} fill:#e5e7eb,stroke:#94a3b8,stroke-width:1px,color:#111827;")
            else:
                lines.append(f"    style {node_id} fill:#f8fafc,stroke:#94a3b8,stroke-width:1px,color:#111827;")
        return "\n".join(lines) + "\n"

    def _simulate_smoke_dynamic(self,
                                sample_input: list[int],
                                threshold: int,
                                control_flow: dict,
                                visual_flow: dict,
                                threads_per_block: int,
                                warp_size: int) -> dict:
        lookup = self._dynamic_block_lookup(control_flow, visual_flow)
        cfg = analyze_control_flow(self.kernel)
        branch_site = cfg.branch_sites[0] if cfg.branch_sites else None
        linear_nodes = visual_flow.get("nodes", []) if visual_flow.get("kind") == "linear" else []
        entry_label = branch_site.block_label if branch_site else (
            next((node["label"] for node in linear_nodes if node.get("tag") == "compare"), "__ENTRY__")
        )
        taken_label = branch_site.taken_target if branch_site else None
        fallthrough_label = branch_site.fallthrough_target if branch_site else None
        linear_select = next((node["label"] for node in linear_nodes if node.get("tag") == "select"), None)
        linear_update = next((node["label"] for node in linear_nodes if node.get("tag") == "compute"), None)
        linear_store = next((node["label"] for node in linear_nodes if node.get("tag") == "store"), None)
        exit_label = branch_site.reconvergence_target if branch_site else (
            next((node["label"] for node in linear_nodes if node.get("tag") == "exit"), None)
        )
        if not exit_label:
            exit_label = self._resolve_block_from_line(34, lookup)

        threads = []
        block_hits = Counter()
        edge_hits = Counter()
        taken_count = 0
        fallthrough_count = 0
        output = []
        step_frames = []

        for thread_id, value in enumerate(sample_input):
            path = [entry_label]
            if value > threshold:
                taken_count += 1
                next_label = taken_label or self._resolve_block_from_line(29, lookup)
                result = value + 1
                branch_kind = "taken"
            else:
                fallthrough_count += 1
                next_label = fallthrough_label or self._resolve_block_from_line(31, lookup)
                result = value - 1
                branch_kind = "fallthrough"

            if next_label:
                path.append(next_label)
            elif linear_select:
                path.append(linear_select)
                if linear_update:
                    path.append(linear_update)
                if linear_store:
                    path.append(linear_store)
            if exit_label:
                path.append(exit_label)

            for label in path:
                if label:
                    block_hits[label] += 1
            for src, dst in zip(path, path[1:]):
                if src and dst and src != dst:
                    edge_hits[(src, dst)] += 1

            threads.append({
                "thread_id": thread_id,
                "lane": thread_id % warp_size,
                "warp_id": thread_id // warp_size,
                "input": value,
                "output": result,
                "path": [label for label in path if label],
                "branch": branch_kind,
            })
            output.append(result)

        total_threads = len(sample_input)
        active_threads = max(taken_count, fallthrough_count)
        activity_factor = round(active_threads / max(min(total_threads, warp_size), 1), 4)
        if not branch_site and linear_select:
            activity_factor = 1.0
        branch_activity = []
        if branch_site is not None or linear_select:
            branch_activity.append({
                "block_label": branch_site.block_label if branch_site else entry_label,
                "taken_target": taken_label,
                "fallthrough_target": fallthrough_label,
                "taken_count": taken_count,
                "fallthrough_count": fallthrough_count,
                "active_threads": min(total_threads, warp_size) if not branch_site and linear_select else active_threads,
                "warp_size": min(total_threads, warp_size),
                "activity_factor": activity_factor,
                "reconvergence_target": exit_label,
            })

        step_values = {
            "entrada": list(sample_input),
            "ajustes": [1 if value > threshold else -1 for value in sample_input],
            "saida": list(output),
        }
        select_label = linear_select or taken_label or fallthrough_label
        update_label = linear_update or taken_label or fallthrough_label
        store_label = linear_store or exit_label
        frame_specs = [
            {
                "title": "Entrada dos dados",
                "active_labels": [entry_label],
                "completed_labels": [],
                "active_edges": [("Start", entry_label)],
                "completed_edges": [],
                "state": {"dados": step_values["entrada"], "threshold": threshold},
            },
            {
                "title": "Decisão por comparação",
                "active_labels": [entry_label],
                "completed_labels": [],
                "active_edges": [("Start", entry_label)],
                "completed_edges": [],
                "state": {
                    "dados": step_values["entrada"],
                    "taken_threads": [t["thread_id"] for t in threads if t["branch"] == "taken"],
                    "fallthrough_threads": [t["thread_id"] for t in threads if t["branch"] == "fallthrough"],
                },
            },
            {
                "title": "Seleção do ajuste",
                "active_labels": [select_label],
                "completed_labels": [entry_label],
                "active_edges": [(entry_label, select_label)],
                "completed_edges": [("Start", entry_label)],
                "state": {"ajustes": step_values["ajustes"]},
            },
            {
                "title": "Aplicação do ajuste",
                "active_labels": [update_label],
                "completed_labels": [entry_label, select_label],
                "active_edges": [(select_label, update_label)],
                "completed_edges": [("Start", entry_label), (entry_label, select_label)],
                "state": {"parcial": step_values["saida"]},
            },
            {
                "title": "Escrita do resultado",
                "active_labels": [store_label],
                "completed_labels": [entry_label, select_label, update_label],
                "active_edges": [(update_label, store_label)],
                "completed_edges": [("Start", entry_label), (entry_label, select_label), (select_label, update_label)],
                "state": {"saida": step_values["saida"]},
            },
            {
                "title": "Fim do kernel",
                "active_labels": [exit_label],
                "completed_labels": [entry_label, select_label, update_label, store_label],
                "active_edges": [(store_label, exit_label)],
                "completed_edges": [("Start", entry_label), (entry_label, select_label), (select_label, update_label), (update_label, store_label)],
                "state": {"saida": step_values["saida"]},
            },
        ]
        for idx, frame in enumerate(frame_specs, 1):
            caption = f"Etapa {idx}: {frame['title']}"
            step_frames.append({
                "step": idx,
                "title": frame["title"],
                "active_labels": [label for label in frame["active_labels"] if label],
                "completed_labels": [label for label in frame.get("completed_labels", []) if label],
                "active_edges": [edge for edge in frame.get("active_edges", []) if edge[0] and edge[1]],
                "completed_edges": [edge for edge in frame.get("completed_edges", []) if edge[0] and edge[1]],
                "state": frame["state"],
                "mermaid": self._format_dynamic_step_mermaid(
                    control_flow=control_flow,
                    visual_flow=visual_flow,
                    active_labels=[label for label in frame["active_labels"] if label],
                    completed_labels=[label for label in frame.get("completed_labels", []) if label],
                    active_edges=[edge for edge in frame.get("active_edges", []) if edge[0] and edge[1]],
                    completed_edges=[edge for edge in frame.get("completed_edges", []) if edge[0] and edge[1]],
                    caption=caption,
                ),
            })

        return {
            "model": "source_guided_dynamic_trace",
            "sample_input": list(sample_input),
            "sample_output": output,
            "threads": threads,
            "block_hits": {label: count for label, count in block_hits.items()},
            "edge_hits": {f"{src}->{dst}": count for (src, dst), count in edge_hits.items()},
            "branch_activity": branch_activity,
            "timeline": [],
            "step_frames": step_frames,
            "notes": [
                "Traço dinâmico guiado pelo CFG do PTX e pelo comportamento semântico do kernel.",
                "Para o microkernel smoke, cada thread executa um único teste e segue um dos dois ramos.",
                "Quando o PTX é linearizado com predicação, a decisão continua existindo nos dados, mas sem dividir o fluxo de controle do warp.",
            ],
        }

    def _simulate_insertion_dynamic(self,
                                    sample_input: list[int],
                                    segment_size: int,
                                    threads_per_block: int,
                                    warp_size: int,
                                    control_flow: dict,
                                    visual_flow: dict) -> dict:
        lookup = self._dynamic_block_lookup(control_flow, visual_flow)
        cfg = analyze_control_flow(self.kernel)
        guard_site = cfg.branch_sites[0] if cfg.branch_sites else None
        outer_check = self._resolve_block_from_line(43, lookup)
        key_block = self._resolve_block_from_line(44, lookup)
        while_check = self._resolve_block_from_line(47, lookup)
        shift_block = self._resolve_block_from_line(48, lookup)
        place_block = self._resolve_block_from_line(52, lookup)
        exit_label = self._resolve_block_from_line(70, lookup) or (visual_flow.get("nodes", [{}])[-1].get("label") if visual_flow.get("nodes") else None)

        threads = []
        block_hits = Counter()
        edge_hits = Counter()
        global_branch_counts: dict[str, dict[str, int]] = {}
        timeline = []
        step_frames = []
        total_elements = len(sample_input)
        segment_count = (total_elements + segment_size - 1) // segment_size

        def _count_branch(block_label: Optional[str], edge_kind: str) -> None:
            if not block_label:
                return
            item = global_branch_counts.setdefault(block_label, {"taken": 0, "fallthrough": 0})
            item[edge_kind] += 1

        for thread_id in range(segment_count):
            base = thread_id * segment_size
            segment = sample_input[base:base + segment_size]
            path = []
            if base + segment_size > total_elements:
                if guard_site:
                    path = [guard_site.block_label, guard_site.taken_target]
                    _count_branch(guard_site.block_label, "taken")
                threads.append({
                    "thread_id": thread_id,
                    "lane": thread_id % warp_size,
                    "warp_id": thread_id // warp_size,
                    "segment_in": segment,
                    "segment_out": segment,
                    "path": [label for label in path if label],
                    "inactive": True,
                })
                continue

            values = list(segment)
            if guard_site:
                path.append(guard_site.block_label)
                if guard_site.fallthrough_target:
                    path.append(guard_site.fallthrough_target)
                _count_branch(guard_site.block_label, "fallthrough")

            for i in range(1, len(values)):
                if outer_check:
                    path.append(outer_check)
                if key_block:
                    path.append(key_block)
                key = values[i]
                j = i - 1
                while True:
                    if while_check:
                        path.append(while_check)
                    cond = j >= 0 and values[j] > key
                    _count_branch(while_check, "taken" if cond else "fallthrough")
                    if not cond:
                        break
                    if shift_block:
                        path.append(shift_block)
                    values[j + 1] = values[j]
                    j -= 1
                if place_block:
                    path.append(place_block)
                values[j + 1] = key
            if exit_label:
                path.append(exit_label)

            for label in path:
                if label:
                    block_hits[label] += 1
            for src, dst in zip(path, path[1:]):
                if src and dst and src != dst:
                    edge_hits[(src, dst)] += 1

            threads.append({
                "thread_id": thread_id,
                "lane": thread_id % warp_size,
                "warp_id": thread_id // warp_size,
                "segment_in": segment,
                "segment_out": values,
                "path": [label for label in path if label],
                "inactive": False,
            })
            timeline.append({
                "thread_id": thread_id,
                "comparisons": sum(1 for label in path if label == while_check),
                "shifts": sum(1 for label in path if label == shift_block),
                "placements": sum(1 for label in path if label == place_block),
            })
            if thread_id == 0 and not threads[-1]["inactive"]:
                values = list(segment)
                frame_id = 0
                step_frames.append({
                    "step": frame_id + 1,
                    "title": "Entrada do segmento",
                    "active_labels": [guard_site.block_label] if guard_site else [],
                    "completed_labels": [],
                    "active_edges": [("Start", guard_site.block_label)] if guard_site else [],
                    "completed_edges": [],
                    "state": {"segmento": list(values)},
                    "mermaid": self._format_dynamic_step_mermaid(control_flow, visual_flow, [guard_site.block_label] if guard_site else [], [], [("Start", guard_site.block_label)] if guard_site else [], [], "Etapa 1: entrada do segmento"),
                })
                for i in range(1, len(values)):
                    key = values[i]
                    j = i - 1
                    frame_id += 1
                    step_frames.append({
                        "step": frame_id + 1,
                        "title": f"Seleciona chave i={i}",
                        "active_labels": [outer_check, key_block],
                        "completed_labels": [guard_site.block_label] if guard_site else [],
                        "active_edges": [(guard_site.block_label, outer_check), (outer_check, key_block)] if guard_site and outer_check and key_block else [],
                        "completed_edges": [("Start", guard_site.block_label)] if guard_site else [],
                        "state": {"segmento": list(values), "key": key, "j": j},
                        "mermaid": self._format_dynamic_step_mermaid(control_flow, visual_flow, [outer_check, key_block], [guard_site.block_label] if guard_site else [], [(guard_site.block_label, outer_check), (outer_check, key_block)] if guard_site and outer_check and key_block else [], [("Start", guard_site.block_label)] if guard_site else [], f"Etapa {frame_id + 1}: chave i={i}"),
                    })
                    while j >= 0 and values[j] > key:
                        frame_id += 1
                        step_frames.append({
                            "step": frame_id + 1,
                            "title": f"Desloca valor em j={j}",
                            "active_labels": [while_check, shift_block],
                            "completed_labels": [label for label in [guard_site.block_label, outer_check, key_block] if label],
                            "active_edges": [(key_block, while_check), (while_check, shift_block)] if key_block and while_check and shift_block else [],
                            "completed_edges": [edge for edge in [("Start", guard_site.block_label if guard_site else None), (guard_site.block_label if guard_site else None, outer_check), (outer_check, key_block)] if edge[0] and edge[1]],
                            "state": {"antes": list(values), "key": key, "j": j},
                            "mermaid": self._format_dynamic_step_mermaid(control_flow, visual_flow, [while_check, shift_block], [label for label in [guard_site.block_label, outer_check, key_block] if label], [(key_block, while_check), (while_check, shift_block)] if key_block and while_check and shift_block else [], [edge for edge in [("Start", guard_site.block_label if guard_site else None), (guard_site.block_label if guard_site else None, outer_check), (outer_check, key_block)] if edge[0] and edge[1]], f"Etapa {frame_id + 1}: shift em j={j}"),
                        })
                        values[j + 1] = values[j]
                        j -= 1
                    values[j + 1] = key
                    frame_id += 1
                    step_frames.append({
                        "step": frame_id + 1,
                        "title": f"Insere chave na posição {j + 1}",
                        "active_labels": [place_block],
                        "completed_labels": [label for label in [guard_site.block_label, outer_check, key_block, while_check, shift_block] if label],
                        "active_edges": [(shift_block, place_block)] if shift_block and place_block else [],
                        "completed_edges": [edge for edge in [("Start", guard_site.block_label if guard_site else None), (guard_site.block_label if guard_site else None, outer_check), (outer_check, key_block), (key_block, while_check), (while_check, shift_block)] if edge[0] and edge[1]],
                        "state": {"depois": list(values), "key": key, "posicao": j + 1},
                        "mermaid": self._format_dynamic_step_mermaid(control_flow, visual_flow, [place_block], [label for label in [guard_site.block_label, outer_check, key_block, while_check, shift_block] if label], [(shift_block, place_block)] if shift_block and place_block else [], [edge for edge in [("Start", guard_site.block_label if guard_site else None), (guard_site.block_label if guard_site else None, outer_check), (outer_check, key_block), (key_block, while_check), (while_check, shift_block)] if edge[0] and edge[1]], f"Etapa {frame_id + 1}: inserção da chave"),
                    })
                frame_id += 1
                step_frames.append({
                    "step": frame_id + 1,
                    "title": "Segmento final ordenado",
                    "active_labels": [exit_label],
                    "completed_labels": [label for label in [guard_site.block_label, outer_check, key_block, while_check, shift_block, place_block] if label],
                    "active_edges": [(place_block, exit_label)] if place_block and exit_label else [],
                    "completed_edges": [edge for edge in [("Start", guard_site.block_label if guard_site else None), (guard_site.block_label if guard_site else None, outer_check), (outer_check, key_block), (key_block, while_check), (while_check, shift_block), (shift_block, place_block)] if edge[0] and edge[1]],
                    "state": {"saida": list(values)},
                    "mermaid": self._format_dynamic_step_mermaid(control_flow, visual_flow, [exit_label], [label for label in [guard_site.block_label, outer_check, key_block, while_check, shift_block, place_block] if label], [(place_block, exit_label)] if place_block and exit_label else [], [edge for edge in [("Start", guard_site.block_label if guard_site else None), (guard_site.block_label if guard_site else None, outer_check), (outer_check, key_block), (key_block, while_check), (while_check, shift_block), (shift_block, place_block)] if edge[0] and edge[1]], f"Etapa {frame_id + 1}: fim do segmento"),
                })

        branch_activity = []
        for site in cfg.branch_sites:
            counts = global_branch_counts.get(site.block_label, {"taken": 0, "fallthrough": 0})
            active_threads = max(counts["taken"], counts["fallthrough"])
            normalized_active = min(active_threads, min(len(threads), warp_size))
            branch_activity.append({
                "block_label": site.block_label,
                "taken_target": site.taken_target,
                "fallthrough_target": site.fallthrough_target,
                "taken_count": counts["taken"],
                "fallthrough_count": counts["fallthrough"],
                "active_threads": normalized_active,
                "warp_size": min(len(threads), warp_size),
                "activity_factor": round(normalized_active / max(min(len(threads), warp_size), 1), 4),
                "reconvergence_target": site.reconvergence_target,
            })

        sample_output = []
        for item in threads:
            sample_output.extend(item["segment_out"])

        return {
            "model": "source_guided_dynamic_trace",
            "sample_input": list(sample_input),
            "sample_output": sample_output[:len(sample_input)],
            "threads": threads,
            "block_hits": {label: count for label, count in block_hits.items()},
            "edge_hits": {f"{src}->{dst}": count for (src, dst), count in edge_hits.items()},
            "branch_activity": branch_activity,
            "timeline": timeline,
            "step_frames": step_frames,
            "notes": [
                "Traço dinâmico aproximado por segmento/thread, projetado sobre os blocos básicos do PTX.",
                "O laço interno do insertion sort reaparece no caminho dinâmico conforme o número real de deslocamentos.",
            ],
        }

    def _simulate_bubble_dynamic(self,
                                 sample_input: list[int],
                                 segment_size: int,
                                 threads_per_block: int,
                                 warp_size: int,
                                 control_flow: dict,
                                 visual_flow: dict) -> dict:
        lookup = self._dynamic_block_lookup(control_flow, visual_flow)
        cfg = analyze_control_flow(self.kernel)
        guard_site = cfg.branch_sites[0] if cfg.branch_sites else None
        outer_block = self._resolve_block_from_line(50, lookup)
        inner_block = self._resolve_block_from_line(51, lookup)
        compare_block = self._resolve_block_from_line(52, lookup)
        swap_block = self._resolve_block_from_line(53, lookup)
        exit_label = self._resolve_block_from_line(73, lookup) or (visual_flow.get("nodes", [{}])[-1].get("label") if visual_flow.get("nodes") else None)

        threads = []
        block_hits = Counter()
        edge_hits = Counter()
        global_branch_counts: dict[str, dict[str, int]] = {}
        timeline = []
        step_frames = []
        total_elements = len(sample_input)
        segment_count = (total_elements + segment_size - 1) // segment_size

        def _count_branch(block_label: Optional[str], edge_kind: str) -> None:
            if not block_label:
                return
            item = global_branch_counts.setdefault(block_label, {"taken": 0, "fallthrough": 0})
            item[edge_kind] += 1

        for thread_id in range(segment_count):
            base = thread_id * segment_size
            segment = sample_input[base:base + segment_size]
            path = []
            if base + segment_size > total_elements:
                if guard_site:
                    path = [guard_site.block_label, guard_site.taken_target]
                    _count_branch(guard_site.block_label, "taken")
                threads.append({
                    "thread_id": thread_id,
                    "lane": thread_id % warp_size,
                    "warp_id": thread_id // warp_size,
                    "segment_in": segment,
                    "segment_out": segment,
                    "path": [label for label in path if label],
                    "inactive": True,
                })
                continue

            values = list(segment)
            if guard_site:
                path.append(guard_site.block_label)
                if guard_site.fallthrough_target:
                    path.append(guard_site.fallthrough_target)
                _count_branch(guard_site.block_label, "fallthrough")

            swaps = 0
            comparisons = 0
            for end in range(len(values) - 1, 0, -1):
                if outer_block:
                    path.append(outer_block)
                for i in range(0, end):
                    if inner_block:
                        path.append(inner_block)
                    if compare_block:
                        path.append(compare_block)
                    comparisons += 1
                    cond = values[i] > values[i + 1]
                    _count_branch(compare_block, "taken" if cond else "fallthrough")
                    if cond:
                        if swap_block:
                            path.append(swap_block)
                        values[i], values[i + 1] = values[i + 1], values[i]
                        swaps += 1
            if exit_label:
                path.append(exit_label)

            for label in path:
                if label:
                    block_hits[label] += 1
            for src, dst in zip(path, path[1:]):
                if src and dst and src != dst:
                    edge_hits[(src, dst)] += 1

            threads.append({
                "thread_id": thread_id,
                "lane": thread_id % warp_size,
                "warp_id": thread_id // warp_size,
                "segment_in": segment,
                "segment_out": values,
                "path": [label for label in path if label],
                "inactive": False,
            })
            timeline.append({
                "thread_id": thread_id,
                "comparisons": comparisons,
                "swaps": swaps,
            })
            if thread_id == 0 and not threads[-1]["inactive"]:
                values = list(segment)
                frame_id = 0
                step_frames.append({
                    "step": frame_id + 1,
                    "title": "Entrada do segmento",
                    "active_labels": [guard_site.block_label] if guard_site else [],
                    "completed_labels": [],
                    "active_edges": [("Start", guard_site.block_label)] if guard_site else [],
                    "completed_edges": [],
                    "state": {"segmento": list(values)},
                    "mermaid": self._format_dynamic_step_mermaid(control_flow, visual_flow, [guard_site.block_label] if guard_site else [], [], [("Start", guard_site.block_label)] if guard_site else [], [], "Etapa 1: entrada do segmento"),
                })
                for end in range(len(values) - 1, 0, -1):
                    frame_id += 1
                    step_frames.append({
                        "step": frame_id + 1,
                        "title": f"Novo passe end={end}",
                        "active_labels": [outer_block],
                        "completed_labels": [guard_site.block_label] if guard_site else [],
                        "active_edges": [(guard_site.block_label, outer_block)] if guard_site and outer_block else [],
                        "completed_edges": [("Start", guard_site.block_label)] if guard_site else [],
                        "state": {"segmento": list(values), "end": end},
                        "mermaid": self._format_dynamic_step_mermaid(control_flow, visual_flow, [outer_block], [guard_site.block_label] if guard_site else [], [(guard_site.block_label, outer_block)] if guard_site and outer_block else [], [("Start", guard_site.block_label)] if guard_site else [], f"Etapa {frame_id + 1}: passe end={end}"),
                    })
                    for i in range(0, end):
                        frame_id += 1
                        step_frames.append({
                            "step": frame_id + 1,
                            "title": f"Compara posições {i} e {i + 1}",
                            "active_labels": [inner_block, compare_block],
                            "completed_labels": [label for label in [guard_site.block_label, outer_block] if label],
                            "active_edges": [(outer_block, inner_block), (inner_block, compare_block)] if outer_block and inner_block and compare_block else [],
                            "completed_edges": [edge for edge in [("Start", guard_site.block_label if guard_site else None), (guard_site.block_label if guard_site else None, outer_block)] if edge[0] and edge[1]],
                            "state": {"segmento": list(values), "i": i, "par": [values[i], values[i + 1]]},
                            "mermaid": self._format_dynamic_step_mermaid(control_flow, visual_flow, [inner_block, compare_block], [label for label in [guard_site.block_label, outer_block] if label], [(outer_block, inner_block), (inner_block, compare_block)] if outer_block and inner_block and compare_block else [], [edge for edge in [("Start", guard_site.block_label if guard_site else None), (guard_site.block_label if guard_site else None, outer_block)] if edge[0] and edge[1]], f"Etapa {frame_id + 1}: comparação {i}/{i + 1}"),
                        })
                        if values[i] > values[i + 1]:
                            values[i], values[i + 1] = values[i + 1], values[i]
                            frame_id += 1
                            step_frames.append({
                                "step": frame_id + 1,
                                "title": f"Swap entre {i} e {i + 1}",
                                "active_labels": [swap_block],
                                "completed_labels": [label for label in [guard_site.block_label, outer_block, inner_block, compare_block] if label],
                                "active_edges": [(compare_block, swap_block)] if compare_block and swap_block else [],
                                "completed_edges": [edge for edge in [("Start", guard_site.block_label if guard_site else None), (guard_site.block_label if guard_site else None, outer_block), (outer_block, inner_block), (inner_block, compare_block)] if edge[0] and edge[1]],
                                "state": {"segmento": list(values), "i": i},
                                "mermaid": self._format_dynamic_step_mermaid(control_flow, visual_flow, [swap_block], [label for label in [guard_site.block_label, outer_block, inner_block, compare_block] if label], [(compare_block, swap_block)] if compare_block and swap_block else [], [edge for edge in [("Start", guard_site.block_label if guard_site else None), (guard_site.block_label if guard_site else None, outer_block), (outer_block, inner_block), (inner_block, compare_block)] if edge[0] and edge[1]], f"Etapa {frame_id + 1}: swap {i}/{i + 1}"),
                            })
                frame_id += 1
                step_frames.append({
                    "step": frame_id + 1,
                    "title": "Segmento final ordenado",
                    "active_labels": [exit_label],
                    "completed_labels": [label for label in [guard_site.block_label, outer_block, inner_block, compare_block, swap_block] if label],
                    "active_edges": [(swap_block, exit_label)] if swap_block and exit_label else [],
                    "completed_edges": [edge for edge in [("Start", guard_site.block_label if guard_site else None), (guard_site.block_label if guard_site else None, outer_block), (outer_block, inner_block), (inner_block, compare_block), (compare_block, swap_block)] if edge[0] and edge[1]],
                    "state": {"saida": list(values)},
                    "mermaid": self._format_dynamic_step_mermaid(control_flow, visual_flow, [exit_label], [label for label in [guard_site.block_label, outer_block, inner_block, compare_block, swap_block] if label], [(swap_block, exit_label)] if swap_block and exit_label else [], [edge for edge in [("Start", guard_site.block_label if guard_site else None), (guard_site.block_label if guard_site else None, outer_block), (outer_block, inner_block), (inner_block, compare_block), (compare_block, swap_block)] if edge[0] and edge[1]], f"Etapa {frame_id + 1}: fim do segmento"),
                })

        branch_activity = []
        for site in cfg.branch_sites:
            counts = global_branch_counts.get(site.block_label, {"taken": 0, "fallthrough": 0})
            active_threads = max(counts["taken"], counts["fallthrough"])
            normalized_active = min(active_threads, min(len(threads), warp_size))
            branch_activity.append({
                "block_label": site.block_label,
                "taken_target": site.taken_target,
                "fallthrough_target": site.fallthrough_target,
                "taken_count": counts["taken"],
                "fallthrough_count": counts["fallthrough"],
                "active_threads": normalized_active,
                "warp_size": min(len(threads), warp_size),
                "activity_factor": round(normalized_active / max(min(len(threads), warp_size), 1), 4),
                "reconvergence_target": site.reconvergence_target,
            })

        sample_output = []
        for item in threads:
            sample_output.extend(item["segment_out"])

        return {
            "model": "source_guided_dynamic_trace",
            "sample_input": list(sample_input),
            "sample_output": sample_output[:len(sample_input)],
            "threads": threads,
            "block_hits": {label: count for label, count in block_hits.items()},
            "edge_hits": {f"{src}->{dst}": count for (src, dst), count in edge_hits.items()},
            "branch_activity": branch_activity,
            "timeline": timeline,
            "step_frames": step_frames,
            "notes": [
                "Traço dinâmico aproximado por segmento/thread, projetado sobre os blocos básicos do PTX.",
                "Se o vetor já estiver ordenado, o ramo de swap perde atividade e o fluxo cai preferencialmente no fallthrough.",
            ],
        }

    def _format_dynamic_mermaid(self, control_flow: dict, visual_flow: dict, dynamic: dict) -> str:
        if visual_flow.get("kind") == "linear":
            lines = ["graph LR", "    Start([START])"]
            prev = "Start"
            for idx, node in enumerate(visual_flow.get("nodes", []), 1):
                node_id = f"L{idx}"
                hits = dynamic.get("block_hits", {}).get(node.get("label", ""), 0)
                label = f"{node['title']}<br>{node['description']}<br>hits={hits}"
                shape = f'(["{label}"])' if node["title"] == "Saída" else f'["{label}"]'
                lines.append(f"    {node_id}{shape}")
                lines.append(f"    {prev} --> {node_id}")
                prev = node_id
            return "\n".join(lines) + "\n"

        node_order = visual_flow.get("nodes", [])
        alias = {node["label"]: f"N{idx}" for idx, node in enumerate(node_order, 1)}
        edge_hits = dynamic.get("edge_hits", {})
        lines = ["graph LR", "    Start([START])"]
        for node in node_order:
            hits = dynamic.get("block_hits", {}).get(node["label"], 0)
            text = f"{node['title']}<br>{node['description']}<br>hits={hits}"
            shape = f'(["{text}"])' if "Saída" in node["title"] else f'["{text}"]'
            lines.append(f"    {alias[node['label']]}{shape}")
        if node_order:
            lines.append(f"    Start --> {alias[node_order[0]['label']]}")
        emitted = set()
        for edge in control_flow.get("edges", []):
            src = edge["source"]
            dst = edge["target"]
            if src not in alias or dst not in alias:
                continue
            key = (src, dst, edge["edge_type"])
            if key in emitted:
                continue
            emitted.add(key)
            hits = edge_hits.get(f"{src}->{dst}", 0)
            edge_label = edge["edge_type"]
            if hits:
                edge_label = f"{edge_label} | {hits} thread(s)"
            lines.append(f'    {alias[src]} -- "{edge_label}" --> {alias[dst]}')
        return "\n".join(lines) + "\n"

    def dynamic_flow(self,
                     sample_input: list[int],
                     segment_size: Optional[int] = None,
                     threads_per_block: int = 32,
                     threshold: int = 0,
                     warp_size: int = 32,
                     mode: str = "data"):
        mode = mode.lower()
        control_flow = self._control_flow_data_with_source()
        visual_flow = self._visual_flow_data()
        kernel_text = f"{self.kernel.name} {self._source_path or ''} {self._origin_path or ''}".lower()

        if "cfg_ifelse_smoke" in kernel_text:
            dynamic = self._simulate_smoke_dynamic(
                sample_input=sample_input,
                threshold=threshold,
                control_flow=control_flow,
                visual_flow=visual_flow,
                threads_per_block=threads_per_block,
                warp_size=warp_size,
            )
        elif "insertion" in kernel_text:
            dynamic = self._simulate_insertion_dynamic(
                sample_input=sample_input,
                segment_size=segment_size or len(sample_input),
                threads_per_block=threads_per_block,
                warp_size=warp_size,
                control_flow=control_flow,
                visual_flow=visual_flow,
            )
        elif "bubble" in kernel_text:
            dynamic = self._simulate_bubble_dynamic(
                sample_input=sample_input,
                segment_size=segment_size or len(sample_input),
                threads_per_block=threads_per_block,
                warp_size=warp_size,
                control_flow=control_flow,
                visual_flow=visual_flow,
            )
        else:
            raise ValueError(
                "Ainda não há simulador dinâmico para este kernel. "
                "Os casos suportados nesta versão são cfg_ifelse_smoke, insertion e bubble."
            )

        mermaid = self._format_dynamic_mermaid(control_flow, visual_flow, dynamic)
        if mode == "data":
            return {
                "mermaid": mermaid,
                "control_flow": control_flow,
                "visual_flow": visual_flow,
                "dynamic_flow": dynamic,
            }
        if mode == "raw":
            return mermaid
        if mode == "text":
            return emit_text(f"```mermaid\n{mermaid}```", mode="text")
        raise ValueError(f"Modo desconhecido: {mode}")

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
                "control_flow": self._control_flow_data_with_source(),
                "visual_flow": self._visual_flow_data(),
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
