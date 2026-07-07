"""
Extração de métricas reais emitidas pelo ptxas.
"""

from __future__ import annotations

import re
from typing import Dict, List

from .core import PTXASInfo


_FUNCTION_RE = re.compile(r"Function properties for (\S+)")
_COMPILE_RE = re.compile(r"Compiling entry function '([^']+)'")
_STACK_RE = re.compile(
    r"(\d+)\s+bytes stack frame,\s+(\d+)\s+bytes spill stores,\s+(\d+)\s+bytes spill loads"
)
_USAGE_RE = re.compile(r"Used\s+(\d+)\s+registers(?:,\s*(.*))?$")
_MEM_FIELD_RE = re.compile(r"(\d+)\s+bytes\s+([a-z]+(?:\[\d+\])?)")


def _normalize_kernel_name(name: str) -> str:
    return name.strip().rstrip(":")


def parse_ptxas_output(stderr_text: str) -> Dict[str, PTXASInfo]:
    """
    Converte o stderr do nvcc/ptxas em um mapa kernel -> métricas reais.
    """
    infos: Dict[str, PTXASInfo] = {}
    current: PTXASInfo | None = None

    for raw_line in stderr_text.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("ptxas info"):
            continue

        function_match = _FUNCTION_RE.search(line)
        compile_match = _COMPILE_RE.search(line)
        kernel_name = None
        if function_match:
            kernel_name = _normalize_kernel_name(function_match.group(1))
        elif compile_match:
            kernel_name = _normalize_kernel_name(compile_match.group(1))

        if kernel_name is not None:
            current = infos.setdefault(kernel_name, PTXASInfo(kernel_name=kernel_name))
            current.raw_lines.append(line)
            continue

        if current is None:
            continue

        current.raw_lines.append(line)

        stack_match = _STACK_RE.search(line)
        if stack_match:
            current.stack_frame_bytes = int(stack_match.group(1))
            current.spill_stores_bytes = int(stack_match.group(2))
            current.spill_loads_bytes = int(stack_match.group(3))
            continue

        usage_match = _USAGE_RE.search(line)
        if usage_match:
            current.registers = int(usage_match.group(1))
            remainder = usage_match.group(2) or ""
            for bytes_value, field_name in _MEM_FIELD_RE.findall(remainder):
                value = int(bytes_value)
                if field_name == "smem":
                    current.shared_mem_bytes = value
                elif field_name.startswith("cmem"):
                    current.constant_mem_bytes += value
                elif field_name == "lmem":
                    current.local_mem_bytes = value

    return infos
