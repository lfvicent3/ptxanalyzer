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
_RE_LOC  = re.compile(r'\.loc\s+(\d+)\s+(\d+)(?:\s+(\d+))?')


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

    for lineno, raw in enumerate(code.splitlines(), 1):
        line = raw.strip()

        # comentários
        if not line or line.startswith("//"):
            continue
        line = re.sub(r'//.*$', '', line).strip()
        if not line:
            continue

        # .file N "path" — mapeamento de índice para nome de arquivo
        m = _RE_FILE.match(line)
        if m:
            idx, path = int(m.group(1)), m.group(2)
            global_file_map[idx] = path
            if current is not None:
                current.file_map[idx] = path
            continue

        # .loc N line [col] — localização no código-fonte
        m = _RE_LOC.match(line)
        if m:
            cur_src_file = int(m.group(1))
            cur_src_line = int(m.group(2))
            cur_src_col  = int(m.group(3)) if m.group(3) else 0
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
                source_file=cur_src_file,
                source_line=cur_src_line,
                source_col =cur_src_col,
            )
            current.instructions.append(instr)
            pending_label = ""

    return kernels
