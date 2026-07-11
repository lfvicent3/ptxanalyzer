"""
Classe principal PTXAnalyzer.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter, deque
from typing import Optional, Union

from .core import PTXKernel, analyze_control_flow, explain_instruction, _block_primary_source_line
from .interpreter import PTXInterpreter
from .output import emit_text
from .parser import parse_ptx
from .ptxas import parse_ptxas_output
from .runtime import (
    BenchmarkSuite,
    RuntimeProfile,
    load_benchmark_csv,
    parse_benchmark_output,
    profile_cuda_runtime,
)
from .state import KernelArg, KernelLaunchConfig, Number


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


_RE_ITANIUM_NAME_PREFIX = re.compile(r'^_Z(\d+)')


def _demangled_kernel_name(name: str) -> str:
    """Extrai o nome "amigável" (sem mangling C++) de um símbolo PTX.

    Kernels `__global__` têm linkage C++ por padrão (portanto mangled no
    PTX), a menos que sejam declarados `extern "C"`. Isto usa o prefixo de
    comprimento do Itanium ABI (`_Z<N><identificador>...`) para isolar o
    nome original — não decodifica namespaces/templates, só o identificador
    da função, que é suficiente para casar com o nome usado no `.cu`.
    """
    match = _RE_ITANIUM_NAME_PREFIX.match(name)
    if not match:
        return name
    length = int(match.group(1))
    rest = name[match.end():]
    return rest[:length] if len(rest) >= length else name


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
        units = parse_ptx(code)
        # `parse_ptx` também captura funções device auxiliares (`.func`,
        # chamadas via `call.uni`) — elas não são pontos de entrada e não
        # entram na seleção de kernel, mas o executor dinâmico genérico
        # precisa delas para resolver chamadas (ver `dynamic_flow`).
        kernels = [unit for unit in units if unit.is_entry_point]
        if not kernels:
            raise ValueError("Nenhum kernel encontrado no PTX fornecido.")
        self._all_units = units
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
            # Fallback: casa pelo nome "amigável" (sem mangling C++), já
            # que o nome usado no .cu raramente é o símbolo mangled do PTX.
            for idx, kernel in enumerate(kernels):
                if _demangled_kernel_name(kernel.name) == normalized_target:
                    return idx
            available = ", ".join(
                kernel.name if _demangled_kernel_name(kernel.name) == kernel.name
                else f"{kernel.name} ({_demangled_kernel_name(kernel.name)})"
                for kernel in kernels
            )
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
        # Saltos incondicionais (bra.uni de reconvergência/else) não são decisões
        # reais e nunca têm risco de divergência: mostrá-los aqui só dilui o
        # ranking de hotspots com "branches" que não escolhem caminho nenhum.
        decision_sites = [site for site in cfg.branch_sites if site.branch_kind == "predicated"]
        sites = sorted(
            decision_sites,
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
            elif compact.startswith(
                "if (base + segment_size > total_elements || local_thread >= segment_size)"
            ):
                # Mesma guarda de fronteira do bloco anterior, só que a variante
                # shared também descarta threads sem segmento (uma por elemento).
                title = "Limite"
                instruction_name = "checagem de fronteira"
                description = "segmento válido e thread dentro do segmento"
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

    def _function_map(self) -> dict:
        """Funções device (`.func`) disponíveis no mesmo PTX, indexadas
        pelo nome mangled — usadas pelo executor genérico para resolver
        `call.uni` a partir do kernel de entrada."""
        return {unit.name: unit for unit in self._all_units if not unit.is_entry_point}

    def dynamic_flow(self,
                     kernel_args: Optional[Union[dict, list]] = None,
                     kernel_name: Optional[str] = None,
                     grid_dim: Optional[tuple[int, int, int]] = None,
                     block_dim: Optional[tuple[int, int, int]] = None,
                     warp_size: int = 32,
                     max_threads: int = 4096,
                     max_steps: int = 200_000,
                     expected_output: Optional[dict[str, list]] = None,
                     mode: str = "data"):
        """Executa o modo dinâmico de forma genérica: interpreta o PTX de
        verdade (via `ptx_analyzer.interpreter.PTXInterpreter`) sobre os
        parâmetros/buffers concretos informados em `kernel_args`, para a
        configuração de grid/block dada. Não há nenhuma ramificação por
        nome de kernel/algoritmo aqui — o mesmo código atende qualquer
        kernel cujas instruções estejam dentro do subconjunto suportado
        (ver `ptx_analyzer.interpreter.SUPPORTED_OPCODES`).

        Uso comum (`kernel_args` como dict simples — a ordem das chaves
        define a posição do parâmetro na assinatura PTX, e listas/tuplas
        viram buffers automaticamente)::

            analyzer.dynamic_flow(
                kernel_name="bubble_sort_global_kernel",
                kernel_args={"data": [4, 2, 1, 3], "total_elements": 4, "segment_size": 4},
                grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
            )

        Uso avançado (controle fino de largura/sinal/tipo de cada buffer):
        passe uma `list[KernelArg]` em `kernel_args`, como antes.

        `kernel_name`, se informado e diferente do kernel atual, seleciona
        outro `.visible .entry` do mesmo arquivo (equivalente a chamar
        `select_kernel(kernel_name=...)` antes).

        `expected_output`, se fornecido, mapeia rótulo de buffer (a chave
        usada em `kernel_args`, ou `param_<indice>` se um `list[KernelArg]`
        sem `label` foi passado) para os valores esperados, e o resultado
        inclui uma comparação honesta em `dynamic_flow["validation"]`.
        """
        if kernel_name and _normalize_kernel_name(kernel_name) != _normalize_kernel_name(self.kernel.name):
            return self.select_kernel(kernel_name=kernel_name).dynamic_flow(
                kernel_args=kernel_args,
                grid_dim=grid_dim,
                block_dim=block_dim,
                warp_size=warp_size,
                max_threads=max_threads,
                max_steps=max_steps,
                expected_output=expected_output,
                mode=mode,
            )

        mode = mode.lower()
        resolved_args = self._resolve_kernel_args(kernel_args)
        dims_omitted = grid_dim is None and block_dim is None
        resolved_grid_dim = grid_dim or (1, 1, 1)
        resolved_block_dim = block_dim or (1, 1, 1)

        control_flow = self._control_flow_data_with_source()
        visual_flow = self._visual_flow_data()
        dynamic = self._simulate_generic_dynamic(
            kernel_args=resolved_args,
            grid_dim=resolved_grid_dim,
            block_dim=resolved_block_dim,
            warp_size=warp_size,
            max_threads=max_threads,
            max_steps=max_steps,
            expected_output=expected_output,
            control_flow=control_flow,
            visual_flow=visual_flow,
        )
        if dims_omitted:
            dynamic["notes"].insert(0,
                "grid_dim/block_dim não informados — assumindo (1, 1, 1) (uma única "
                "thread). Se o kernel espera uma thread por elemento/segmento, informe "
                "grid_dim/block_dim explicitamente."
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

    def dynamic_trace_html(self,
                           kernel_args: Optional[Union[dict, list]] = None,
                           grid_dim: Optional[tuple[int, int, int]] = None,
                           block_dim: Optional[tuple[int, int, int]] = None,
                           warp_size: int = 32,
                           max_threads: int = 4096,
                           max_steps: int = 200_000,
                           expected_output: Optional[dict[str, list]] = None,
                           sections: Optional[list[str]] = None,
                           height: str = "480px",
                           title: Optional[str] = None) -> str:
        """Executa `dynamic_flow(...)` e devolve o HTML interativo e
        navegável por passo (grafo, threads/warps, memória, E/S, linha do
        tempo das threads e código do kernel — ver
        `ptx_analyzer.dynamic_view` para o design por trás disso).

        `sections`, se informado, restringe quais blocos aparecem — um
        subconjunto de `dynamic_view.ALL_SECTIONS` =
        ("graph", "threads", "memory", "io", "timeline", "code"); por
        padrão (`None`) mostra a visão completa. Útil pra embutir só um
        pedaço (ex.: `sections=["graph"]` pra só o grafo, ou
        `sections=["timeline", "code"]` pra dispensar o grafo).

        Uso em Colab/Jupyter:

            from IPython.display import HTML
            HTML(kernel.dynamic_trace_html(kernel_args=..., grid_dim=..., block_dim=...))

        ou direto com `kernel.show_dynamic_trace(...)`.
        """
        from .dynamic_view import render_dynamic_trace_html

        result = self.dynamic_flow(
            kernel_args=kernel_args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            warp_size=warp_size,
            max_threads=max_threads,
            max_steps=max_steps,
            expected_output=expected_output,
            mode="data",
        )
        return render_dynamic_trace_html(
            result,
            title=title or f"{self.kernel.name} — traço dinâmico",
            warp_size=warp_size,
            grid_dim=grid_dim,
            block_dim=block_dim,
            source_path=self.source_path,
            kernel_name=self.friendly_kernel_name,
            sections=sections,
            height=height,
        )

    def show_dynamic_trace(self, **kwargs) -> None:
        """Chama `dynamic_trace_html(**kwargs)` e exibe via
        `IPython.display.HTML` — uso direto em uma célula do Colab/Jupyter:

            kernel.show_dynamic_trace(kernel_args=..., grid_dim=..., block_dim=...)
        """
        from .dynamic_view import render_dynamic_trace_iframe_html
        from IPython.display import HTML, display
        result = self.dynamic_flow(
            kernel_args=kwargs.get("kernel_args"),
            grid_dim=kwargs.get("grid_dim"),
            block_dim=kwargs.get("block_dim"),
            warp_size=kwargs.get("warp_size", 32),
            max_threads=kwargs.get("max_threads", 4096),
            max_steps=kwargs.get("max_steps", 200_000),
            expected_output=kwargs.get("expected_output"),
            mode="data",
        )
        display(HTML(render_dynamic_trace_iframe_html(
            result,
            title=kwargs.get("title") or f"{self.kernel.name} — traço dinâmico",
            warp_size=kwargs.get("warp_size", 32),
            grid_dim=kwargs.get("grid_dim"),
            block_dim=kwargs.get("block_dim"),
            height=kwargs.get("height", "480px"),
            source_path=self.source_path,
            kernel_name=self.friendly_kernel_name,
            sections=kwargs.get("sections"),
        )))

    def control_flow_html(self, sections: Optional[list[str]] = None, height: str = "480px") -> str:
        """Visão estática do CFG (sem executar nada), no mesmo grafo
        (Cytoscape.js + dagre) usado pelo traço dinâmico — mesmos nós e
        arestas, só sem dados de execução (threads=0, sem heat/timeline
        úteis). Substitui o antigo grafo Mermaid usado por
        `control_flow(mode="html")`, que tinha problemas reais de
        escala/fit no Colab (ver docstring de `ptx_analyzer.dynamic_view`
        pra detalhes do porquê).

        `sections` por padrão mostra só `["graph"]` (não há threads pra
        preencher os outros painéis numa visão puramente estática); passe
        explicitamente pra incluir outros (ex.: `sections=["graph","code"]`).
        """
        from .dynamic_view import render_dynamic_trace_html

        result = {"control_flow": self._control_flow_data_with_source()}
        return render_dynamic_trace_html(
            result,
            title=f"{self.kernel.name} — CFG estático",
            source_path=self.source_path,
            kernel_name=self.friendly_kernel_name,
            sections=sections if sections is not None else ["graph"],
            height=height,
        )

    def _kernel_display_name(self) -> str:
        """Nome do kernel para mensagens de erro: prefere o nome "amigável"
        (sem mangling C++) quando ele existe e difere do símbolo PTX cru."""
        demangled = _demangled_kernel_name(self.kernel.name)
        if demangled != self.kernel.name:
            return f"{demangled} ({self.kernel.name})"
        return self.kernel.name

    def _resolve_kernel_args(self, kernel_args: Optional[Union[dict, list]]) -> list[KernelArg]:
        """Aceita tanto o formato ergonômico (`dict {rótulo: valor}`, com
        listas/tuplas viram buffers e escalares viram `int`/`float`) quanto
        o formato de baixo nível (`list[KernelArg]`, para controle fino de
        largura/sinal/tipo). Sempre valida contra `self.kernel.params`
        (a assinatura real, extraída do PTX) antes de seguir."""
        if kernel_args is None:
            kernel_args = {}

        if isinstance(kernel_args, dict):
            resolved = self._kernel_args_from_dict(kernel_args)
        elif isinstance(kernel_args, list) and all(isinstance(item, KernelArg) for item in kernel_args):
            resolved = list(kernel_args)
        else:
            raise TypeError(
                "kernel_args deve ser um dict {rótulo: valor} (uso comum: listas/tuplas "
                "viram buffers, int/float viram escalares) ou uma list[KernelArg] (uso "
                f"avançado). Recebido: {type(kernel_args).__name__}."
            )

        self._validate_kernel_args(resolved)
        return resolved

    def _kernel_args_from_dict(self, kernel_args: dict) -> list[KernelArg]:
        declared = self.kernel.params
        resolved: list[KernelArg] = []
        for position, (label, value) in enumerate(kernel_args.items()):
            decl = declared[position] if position < len(declared) else None
            looks_like_pointer = decl is not None and decl.ptx_type == "u64"

            if isinstance(value, (list, tuple)):
                if decl is not None and not looks_like_pointer:
                    raise ValueError(
                        f"parâmetro {position} ({label!r}) do kernel {self._kernel_display_name()} é "
                        f"escalar (.{decl.ptx_type}), mas foi passada uma lista/tupla — "
                        "buffers só fazem sentido em parâmetros de ponteiro (tipicamente .u64)."
                    )
                resolved.append(KernelArg(index=position, kind="buffer", values=list(value), label=label))
            elif isinstance(value, bool):
                raise TypeError(f"parâmetro {label!r}: bool não é um argumento de kernel válido.")
            elif isinstance(value, (int, float)):
                if looks_like_pointer:
                    raise ValueError(
                        f"parâmetro {position} ({label!r}) do kernel {self._kernel_display_name()} espera "
                        f"um ponteiro (.{decl.ptx_type}), mas foi passado um escalar ({value!r}) — "
                        "passe uma lista/tupla de valores para esse parâmetro."
                    )
                resolved.append(KernelArg(index=position, kind="scalar", value=value, label=label))
            else:
                raise TypeError(
                    f"parâmetro {label!r}: tipo {type(value).__name__} não suportado; "
                    "use int/float para escalares ou list/tuple para buffers."
                )
        return resolved

    def _validate_kernel_args(self, resolved: list[KernelArg]) -> None:
        declared = self.kernel.params
        if not declared:
            return  # assinatura não capturada pelo parser (raro) — nada a validar
        provided = {arg.index: arg for arg in resolved}
        missing = [decl for decl in declared if decl.index not in provided]
        if missing:
            details = ", ".join(f"posição {decl.index} (.{decl.ptx_type})" for decl in missing)
            raise ValueError(
                f"faltando argumento(s) para o kernel {self._kernel_display_name()}: {details}. "
                f"A assinatura tem {len(declared)} parâmetro(s) — confira a ordem/quantidade "
                "de chaves em kernel_args (analyzer.kernel.params mostra a assinatura completa)."
            )
        extra = sorted(idx for idx in provided if idx >= len(declared))
        if extra:
            raise ValueError(
                f"kernel {self._kernel_display_name()} tem apenas {len(declared)} parâmetro(s), mas "
                f"foram informados argumentos nas posições extras {extra} — remova-os de kernel_args."
            )

    def _simulate_generic_dynamic(self,
                                  kernel_args: list[KernelArg],
                                  grid_dim: tuple[int, int, int],
                                  block_dim: tuple[int, int, int],
                                  warp_size: int,
                                  max_threads: int,
                                  max_steps: int,
                                  expected_output: Optional[dict[str, list]],
                                  control_flow: dict,
                                  visual_flow: dict) -> dict:
        interpreter = PTXInterpreter(
            self.kernel,
            functions=self._function_map(),
            raw_ptx=self._code,
            max_steps=max_steps,
        )
        interpreter.load_args(kernel_args)
        launch = KernelLaunchConfig(grid_dim=grid_dim, block_dim=block_dim, warp_size=warp_size)
        ktrace = interpreter.run(launch, max_threads=max_threads)

        threads = [
            {
                "thread_id": t.thread_id,
                "tid": list(t.tid),
                "ctaid": list(t.ctaid),
                "lane": t.thread_id % warp_size,
                "warp_id": t.thread_id // warp_size,
                "path": list(t.blocks_visited),
                "edges": [list(edge) for edge in t.edges_taken],
                "branch_decisions": [bd.to_dict() for bd in t.branch_decisions],
                "halt_reason": t.halt_reason,
                "unsupported_ops": list(t.unsupported_ops),
            }
            for t in ktrace.threads
        ]

        # Atividade por branch: agregada a partir das decisões REAIS de
        # cada thread (não estimada por heurística de algoritmo).
        branch_totals: dict = {}
        for t in ktrace.threads:
            for bd in t.branch_decisions:
                item = branch_totals.setdefault(bd.block_label, {
                    "taken": 0, "fallthrough": 0,
                    "taken_target": bd.taken_target,
                    "fallthrough_target": bd.fallthrough_target,
                })
                if bd.taken_target is not None and bd.chosen_target == bd.taken_target:
                    item["taken"] += 1
                else:
                    item["fallthrough"] += 1

        cfg = analyze_control_flow(self.kernel)
        reconvergence = {site.block_label: site.reconvergence_target for site in cfg.branch_sites}
        total_threads = max(len(ktrace.threads), 1)
        branch_activity = []
        for label, counts in branch_totals.items():
            active_threads = max(counts["taken"], counts["fallthrough"])
            warp_population = min(total_threads, warp_size)
            normalized = min(active_threads, warp_population)
            branch_activity.append({
                "block_label": label,
                "taken_target": counts["taken_target"],
                "fallthrough_target": counts["fallthrough_target"],
                "taken_count": counts["taken"],
                "fallthrough_count": counts["fallthrough"],
                "active_threads": normalized,
                "warp_size": warp_population,
                "activity_factor": round(normalized / max(warp_population, 1), 4),
                "reconvergence_target": reconvergence.get(label),
            })

        buffers_before = {
            ktrace.buffer_labels.get(idx, f"param_{idx}"): vals
            for idx, vals in ktrace.buffers_before.items()
        }
        buffers_after = {
            ktrace.buffer_labels.get(idx, f"param_{idx}"): vals
            for idx, vals in ktrace.buffers_after.items()
        }

        validation = []
        if expected_output:
            for label, expected in expected_output.items():
                actual = buffers_after.get(label)
                validation.append({
                    "buffer": label,
                    "expected": list(expected),
                    "actual": actual,
                    "match": actual == list(expected),
                })

        notes = [
            "Traço dinâmico obtido executando o PTX real (executor genérico, não uma "
            "heurística por algoritmo): cada thread simulada percorre o CFG a partir dos "
            "valores concretos fornecidos, avaliando predicados/branches de verdade.",
            "Threads são executadas sequencialmente (sem concorrência real); "
            "bar.sync/membar são tratados como marcadores, não como barreiras de fato — "
            "kernels que dependem de __syncthreads() para dividir trabalho entre threads "
            "de um mesmo bloco podem produzir saída divergente da GPU real (ver docstring "
            "de ptx_analyzer.interpreter).",
        ]
        if ktrace.unsupported_ops:
            notes.append(
                "Instruções fora do subconjunto suportado hoje foram encontradas e "
                f"interromperam a(s) thread(s) nesse ponto: {', '.join(ktrace.unsupported_ops)}."
            )
        if expected_output:
            if all(item["match"] for item in validation):
                notes.append("Saída simulada bate com a saída esperada informada.")
            else:
                notes.append("Divergência entre saída simulada e saída esperada — ver 'validation'.")
        else:
            notes.append("Nenhuma saída esperada foi fornecida para validação automática.")

        step_frames = self._build_generic_step_frames(ktrace, control_flow, visual_flow)

        return {
            "model": "generic_ptx_cfg_trace",
            "sample_input": buffers_before,
            "sample_output": buffers_after,
            "threads": threads,
            "block_hits": dict(ktrace.block_hits),
            "edge_hits": dict(ktrace.edge_hits),
            "branch_activity": branch_activity,
            "timeline": [],
            "step_frames": step_frames,
            "validation": validation,
            "unsupported_ops": list(ktrace.unsupported_ops),
            "notes": notes,
        }

    def _build_generic_step_frames(self,
                                   ktrace,
                                   control_flow: dict,
                                   visual_flow: dict,
                                   max_frames: int = 120) -> list:
        """Um quadro por bloco visitado por uma thread representativa
        (normalmente a de índice 0), com o conteúdo real dos buffers
        naquele ponto da execução — não uma reconstrução manual do
        algoritmo, e sim o que a memória simulada de fato continha."""
        snapshot_trace = next((t for t in ktrace.threads if t.block_snapshots), None)
        if snapshot_trace is None:
            return []

        snapshots = snapshot_trace.block_snapshots[:max_frames]
        frames = []
        prev_label = None
        completed_labels: list = []
        completed_edges: list = []
        for idx, (label, buffers) in enumerate(snapshots, 1):
            active_edges = [(prev_label, label)] if prev_label else []
            state = {
                ktrace.buffer_labels.get(buf_idx, f"param_{buf_idx}"): vals
                for buf_idx, vals in buffers.items()
            }
            frames.append({
                "step": idx,
                "title": f"Bloco {label}",
                "active_labels": [label],
                "completed_labels": list(completed_labels),
                "active_edges": list(active_edges),
                "completed_edges": list(completed_edges),
                "state": state,
                "mermaid": self._format_dynamic_step_mermaid(
                    control_flow=control_flow,
                    visual_flow=visual_flow,
                    active_labels=[label],
                    completed_labels=list(completed_labels),
                    active_edges=list(active_edges),
                    completed_edges=list(completed_edges),
                    caption=f"Etapa {idx}: bloco {label}",
                ),
            })
            if prev_label:
                completed_edges.append((prev_label, label))
            completed_labels.append(label)
            prev_label = label

        if len(snapshot_trace.block_snapshots) > max_frames:
            frames.append({
                "step": len(frames) + 1,
                "title": "Traço truncado",
                "active_labels": [],
                "completed_labels": list(completed_labels),
                "active_edges": [],
                "completed_edges": list(completed_edges),
                "state": {"nota": f"exibindo apenas os primeiros {max_frames} blocos visitados"},
                "mermaid": self._format_dynamic_step_mermaid(
                    control_flow=control_flow,
                    visual_flow=visual_flow,
                    active_labels=[],
                    completed_labels=list(completed_labels),
                    active_edges=[],
                    completed_edges=list(completed_edges),
                    caption="Traço truncado",
                ),
            })
        return frames

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
                from .dynamic_view import render_dynamic_trace_iframe_html
                from IPython.display import HTML, display
                display(HTML(render_dynamic_trace_iframe_html(
                    {"control_flow": self._control_flow_data_with_source()},
                    title=f"{self.kernel.name} — CFG estático",
                    source_path=self.source_path,
                    kernel_name=self.friendly_kernel_name,
                    sections=["graph"],
                )))
                return None
            except Exception:
                return emit_text(f"```mermaid\n{graph}```", mode="text")
        raise ValueError(f"Modo desconhecido: {mode}")

    @property
    def source_path(self) -> Optional[str]:
        """Caminho absoluto do .cu associado a este kernel, se localizado
        ou compilado (mesma resolução usada para anotar `source_line`
        nos blocos do CFG). `None` se não houver fonte CUDA disponível."""
        return self._infer_source_path()

    @property
    def friendly_kernel_name(self) -> str:
        """Nome do kernel como aparece escrito no `.cu` (sem mangling C++
        do símbolo PTX) — o que precisa pra localizar a função certa
        dentro do arquivo-fonte quando ele tem mais de um kernel."""
        return _demangled_kernel_name(self.kernel.name)

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
