"""
Parser PTX.
"""

import re
from typing import Dict, List, Optional

from .core import PTXInstruction, PTXKernel, _OP_TO_CAT

# ──────────────────────────────────────────────────────────────────────────────
# 3. Parser PTX
# ──────────────────────────────────────────────────────────────────────────────

_RE_KERNEL  = re.compile(r'\.visible\s+\.entry\s+(\w+)\s*\(')
_RE_REG     = re.compile(r'\.reg\s+\.(pred|[bsuf]\d+)\s+(%[\w<>]+)\s*;')
_RE_PARAM   = re.compile(r'\.param\s+')
_RE_LABEL   = re.compile(r'^(\$?[\w.]+):')
_RE_INSTR   = re.compile(
    r'^(?:(@[!]?%\w+)\s+)?'   # predicado opcional
    r'(\w[\w.]*)\s*'           # opcode (com qualificadores de tipo)
    r'(.*?)\s*;',              # operandos
    re.DOTALL,
)

def _op_base(op: str) -> str:
    return op.split(".")[0].lower()


def _split_operands(raw: str) -> List[str]:
    raw = raw.strip()
    # remove chaves externas de vetores  {%r0, %r1, ...}
    raw = re.sub(r'^\{(.*)\}$', r'\1', raw.strip())
    return [o.strip() for o in re.split(r',\s*', raw) if o.strip()]


_RE_FILE = re.compile(r'\.file\s+(\d+)\s+"([^"]+)"')
_RE_LOC  = re.compile(
    r'\.loc\s+(\d+)\s+(\d+)(?:\s+(\d+))?'
    r'(?:,.*?\binlined_at\s+(\d+)\s+(\d+)\s+(\d+))?'
)


def parse_ptx(code: str) -> List[PTXKernel]:
    """
    Parseia código PTX e retorna lista de PTXKernel.

    Captura diretivas .file/.loc (geradas com -lineinfo) para mapear cada
    instrução de volta à linha correspondente no código-fonte .cu.
    """
    kernels: List[PTXKernel] = []
    current: Optional[PTXKernel] = None
    pending_label = ""
    inside_kernel = False
    depth = 0
    global_file_map: Dict[int, str] = {}
    cur_src_file = 0
    cur_src_line = 0
    cur_src_col  = 0
    cur_inline_src_file = 0
    cur_inline_src_line = 0
    cur_inline_src_col = 0
    pending_stmt_parts: List[str] = []
    pending_stmt_raw_parts: List[str] = []
    pending_stmt_lineno = 0
    pending_stmt_src_file = 0
    pending_stmt_src_line = 0
    pending_stmt_src_col = 0
    pending_stmt_inline_src_file = 0
    pending_stmt_inline_src_line = 0
    pending_stmt_inline_src_col = 0

    def _start_multiline_statement(clean_line: str, raw_line: str, lineno: int) -> None:
        nonlocal pending_stmt_parts, pending_stmt_raw_parts
        nonlocal pending_stmt_lineno, pending_stmt_src_file, pending_stmt_src_line, pending_stmt_src_col
        nonlocal pending_stmt_inline_src_file, pending_stmt_inline_src_line, pending_stmt_inline_src_col
        pending_stmt_parts = [clean_line]
        pending_stmt_raw_parts = [raw_line.strip()]
        pending_stmt_lineno = lineno
        pending_stmt_src_file = cur_src_file
        pending_stmt_src_line = cur_src_line
        pending_stmt_src_col = cur_src_col
        pending_stmt_inline_src_file = cur_inline_src_file
        pending_stmt_inline_src_line = cur_inline_src_line
        pending_stmt_inline_src_col = cur_inline_src_col

    def _append_multiline_statement(clean_line: str, raw_line: str) -> None:
        pending_stmt_parts.append(clean_line)
        pending_stmt_raw_parts.append(raw_line.strip())

    for lineno, raw in enumerate(code.splitlines(), 1):
        line = raw.strip()

        # comentários
        if not line or line.startswith("//"):
            continue
        line = re.sub(r'//.*$', '', line).strip()
        if not line:
            continue

        if pending_stmt_parts:
            _append_multiline_statement(line, raw)
            if ";" not in line:
                continue
            line = " ".join(pending_stmt_parts)
            raw = " ".join(part for part in pending_stmt_raw_parts if part)
            lineno = pending_stmt_lineno
            stmt_src_file = pending_stmt_src_file
            stmt_src_line = pending_stmt_src_line
            stmt_src_col = pending_stmt_src_col
            stmt_inline_src_file = pending_stmt_inline_src_file
            stmt_inline_src_line = pending_stmt_inline_src_line
            stmt_inline_src_col = pending_stmt_inline_src_col
            pending_stmt_parts = []
            pending_stmt_raw_parts = []
        else:
            stmt_src_file = cur_src_file
            stmt_src_line = cur_src_line
            stmt_src_col = cur_src_col
            stmt_inline_src_file = cur_inline_src_file
            stmt_inline_src_line = cur_inline_src_line
            stmt_inline_src_col = cur_inline_src_col

        # .file N "path" — mapeamento de índice para nome de arquivo
        # Pode aparecer em qualquer posição do PTX (dentro ou fora de kernels),
        # então propaga para todos os kernels já parseados.
        m = _RE_FILE.match(line)
        if m:
            idx, path = int(m.group(1)), m.group(2)
            global_file_map[idx] = path
            for k in kernels:
                k.file_map[idx] = path
            continue

        # .loc N line [col] — localização no código-fonte
        m = _RE_LOC.match(line)
        if m:
            cur_src_file = int(m.group(1))
            cur_src_line = int(m.group(2))
            cur_src_col  = int(m.group(3)) if m.group(3) else 0
            cur_inline_src_file = int(m.group(4)) if m.group(4) else 0
            cur_inline_src_line = int(m.group(5)) if m.group(5) else 0
            cur_inline_src_col = int(m.group(6)) if m.group(6) else 0
            continue

        depth += line.count("{") - line.count("}")

        # cabeçalho do kernel
        m = _RE_KERNEL.match(line)
        if m:
            current = PTXKernel(name=m.group(1), file_map=dict(global_file_map))
            kernels.append(current)
            inside_kernel = True
            continue

        if not inside_kernel or current is None:
            continue

        # saiu do kernel
        if depth < 0:
            inside_kernel = False
            depth = 0
            current = None
            continue

        # parâmetros
        if _RE_PARAM.search(line):
            current.param_count += line.count(".param")
            continue

        # declarações de registrador
        m = _RE_REG.match(line)
        if m:
            rtype, rname = m.group(1), m.group(2)
            if rtype not in current.reg_decls:
                current.reg_decls[rtype] = set()
            range_m = re.match(r'%(\w+)<(\d+)>', rname)
            if range_m:
                base, n = range_m.group(1), int(range_m.group(2))
                for k in range(n):
                    current.reg_decls[rtype].add(f"%{base}{k}")
            else:
                current.reg_decls[rtype].add(rname)
            continue

        # rótulo
        m = _RE_LABEL.match(line)
        if m and ";" not in line:
            pending_label = m.group(1)
            continue

        if ";" not in line and not line.endswith("{") and not line.endswith("}"):
            _start_multiline_statement(line, raw, lineno)
            continue

        # instrução
        m = _RE_INSTR.match(line)
        if m:
            pred    = m.group(1) or ""
            op      = m.group(2)
            ob      = _op_base(op)
            ops_raw = m.group(3)
            cat     = _OP_TO_CAT.get(ob, "other")

            instr = PTXInstruction(
                line_no=lineno,
                raw=raw.strip(),
                is_predicated=bool(pred),
                predicate=pred,
                op=op,
                op_base=ob,
                operands=_split_operands(ops_raw),
                category=cat,
                label=pending_label,
                source_file=stmt_src_file,
                source_line=stmt_src_line,
                source_col =stmt_src_col,
                inline_source_file=stmt_inline_src_file,
                inline_source_line=stmt_inline_src_line,
                inline_source_col=stmt_inline_src_col,
            )
            current.instructions.append(instr)
            pending_label = ""

    return kernels
