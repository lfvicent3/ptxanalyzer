"""
Taxonomia e modelo de dados básicos para a análise PTX.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# 1. Taxonomia de instruções PTX
# ──────────────────────────────────────────────────────────────────────────────

CATEGORIES: Dict[str, Set[str]] = {
    "arithmetic": {
        "add", "sub", "mul", "fma", "mad", "div", "neg", "abs",
        "min", "max", "rcp", "sqrt", "rsqrt", "sin", "cos", "lg2",
        "ex2", "popc", "clz", "brev", "sad", "dp2a", "dp4a",
    },
    "memory": {
        "ld", "st", "atom", "red", "prefetch", "isspacep",
        "ldu", "suld", "sust", "sured", "tex", "tld4",
    },
    "control": {
        "bra", "brx", "call", "ret", "exit", "bar", "membar",
        "vote", "nanosleep",
    },
    "mov_conv": {
        "mov", "cvt", "cvta", "addrspacecast", "selp",
    },
    "comparison": {
        "setp", "set",
    },
    "logic": {
        "and", "or", "xor", "not", "cnot", "shl", "shr",
        "bfi", "bfe", "prmt",
    },
    "warp": {
        "shfl", "redux", "activemask", "elect", "match",
    },
    "special": {
        "trap", "brkpt", "nop", "pmevent", "setmaxnreg",
        "griddepcontrol", "fence",
    },
}

_OP_TO_CAT: Dict[str, str] = {
    op: cat for cat, ops in CATEGORIES.items() for op in ops
}

CATEGORY_COLORS = {
    "arithmetic": "#3b82f6",   # azul
    "memory":     "#f59e0b",   # laranja
    "control":    "#ef4444",   # vermelho
    "mov_conv":   "#10b981",   # verde
    "comparison": "#06b6d4",   # ciano
    "logic":      "#f97316",   # laranja escuro
    "warp":       "#ec4899",   # rosa
    "special":    "#6b7280",   # cinza
    "other":      "#374151",   # cinza escuro
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Modelo de dados
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PTXInstruction:
    line_no: int
    raw: str
    is_predicated: bool
    predicate: str           # "@%p1" ou "@!%p1"
    op: str                  # opcode completo: "ld.global.f32"
    op_base: str             # prefixo: "ld"
    operands: List[str]
    category: str
    label: str = ""          # rótulo que precede esta instrução
    source_file: int = 0     # índice do .file no PTX (0 = desconhecido)
    source_line: int = 0     # linha do .cu original (via .loc, 0 = desconhecido)
    source_col:  int = 0     # coluna do .cu original (via .loc, 0 = desconhecido)


@dataclass
class BasicBlock:
    """Bloco básico do grafo de fluxo de controle (CFG)."""
    label: str
    instructions: List[PTXInstruction]
    # (tipo_aresta, rótulo_destino): "conditional"|"jump"|"fallthrough"
    exits: List[Tuple[str, str]] = field(default_factory=list)
    is_entry: bool = False
    is_terminal: bool = False   # ret/exit ou bra.uni sem fall-through


@dataclass
class PTXKernel:
    name: str
    instructions: List[PTXInstruction] = field(default_factory=list)
    reg_decls: Dict[str, Set[str]] = field(default_factory=dict)
    param_count: int = 0
    file_map: Dict[int, str] = field(default_factory=dict)  # idx → path do .cu

    # ── métricas derivadas ──────────────────────────────────────────────────

    @property
    def total_instructions(self) -> int:
        return len(self.instructions)

    @property
    def total_registers(self) -> int:
        return sum(len(v) for v in self.reg_decls.values())

    @property
    def category_counts(self) -> Dict[str, int]:
        return dict(Counter(i.category for i in self.instructions))

    @property
    def global_loads(self) -> int:
        return sum(1 for i in self.instructions
                   if i.op_base == "ld" and "global" in i.op)

    @property
    def global_stores(self) -> int:
        return sum(1 for i in self.instructions
                   if i.op_base == "st" and "global" in i.op)

    @property
    def shared_accesses(self) -> int:
        return sum(1 for i in self.instructions
                   if i.op_base in ("ld", "st") and "shared" in i.op)

    @property
    def atomics(self) -> int:
        return sum(1 for i in self.instructions
                   if i.op_base in ("atom", "red"))

    @property
    def predicated_branches(self) -> int:
        """Branches condicionais (@%p bra) — causam divergência de warp."""
        return sum(1 for i in self.instructions
                   if i.op_base == "bra" and i.is_predicated)

    @property
    def total_branches(self) -> int:
        """Todas as instruções bra/brx (condicionais + incondicionais)."""
        return sum(1 for i in self.instructions if i.op_base in ("bra", "brx"))

    @property
    def unconditional_branches(self) -> int:
        """Saltos incondicionais (bra.uni ou bra sem predicado) — sem divergência."""
        return sum(1 for i in self.instructions
                   if i.op_base == "bra" and not i.is_predicated)

    @property
    def setp_count(self) -> int:
        """Instruções setp — cada uma define um predicado que pode levar a um branch."""
        return sum(1 for i in self.instructions if i.op_base == "setp")

    @property
    def fma_count(self) -> int:
        return sum(1 for i in self.instructions if i.op_base == "fma")

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs / acessos de memória — estimativa estática."""
        arith = self.category_counts.get("arithmetic", 0)
        mem   = max(self.category_counts.get("memory", 1), 1)
        return round(arith / mem, 3)

    @property
    def uses_nctaid(self) -> bool:
        return any("%nctaid" in i.raw for i in self.instructions)

    @property
    def shfl_count(self) -> int:
        return sum(1 for i in self.instructions if "shfl" in i.op_base)

    @property
    def bit_ops_count(self) -> int:
        return sum(1 for i in self.instructions
                   if i.op_base in ("shl", "shr", "and", "xor", "bfe", "bfi"))

    @property
    def local_accesses(self) -> int:
        """Acessos a .local (stack/spill) — caro como global, sem coalescência."""
        return sum(1 for i in self.instructions
                   if i.op_base in ("ld", "st") and "local" in i.op)

    @property
    def min_max_count(self) -> int:
        """Instruções min/max — selection sort pode usar ao invés de branch+swap."""
        return sum(1 for i in self.instructions if i.op_base in ("min", "max"))

    @property
    def total_mem_accesses(self) -> int:
        return (self.global_loads + self.global_stores
                + self.shared_accesses + self.local_accesses)

    @property
    def is_register_only(self) -> bool:
        """Sem loads/stores → compilador desenrolou tudo em registradores (unroll total)."""
        return self.total_mem_accesses == 0 and self.total_instructions > 0

    @property
    def branch_ratio(self) -> float:
        """Fração de instruções que são branches predicados."""
        if self.total_instructions == 0:
            return 0.0
        return round(self.predicated_branches / self.total_instructions, 4)

    @property
    def instruction_mix(self) -> Dict[str, float]:
        """Distribuição percentual de categorias de instrução."""
        total = max(self.total_instructions, 1)
        return {cat: round(n / total * 100, 1)
                for cat, n in self.category_counts.items()}

    def metrics_dict(self) -> dict:
        return {
            "Instruções":           self.total_instructions,
            "Registradores":        self.total_registers,
            "ld.global":            self.global_loads,
            "st.global":            self.global_stores,
            "ld/st.local":          self.local_accesses,
            "Shared":               self.shared_accesses,
            "BranchTotal":          self.total_branches,
            "BranchCondicional":    self.predicated_branches,
            "BranchIncondicional":  self.unconditional_branches,
            "Setp":                 self.setp_count,
            "BranchRatio":          self.branch_ratio,
            "Atômicas":             self.atomics,
            "FMA":                  self.fma_count,
            "min/max":              self.min_max_count,
            "shfl.sync":            self.shfl_count,
            "shl/shr":              self.bit_ops_count,
            "SóRegistros":          int(self.is_register_only),
            "Int.Aritmética":       self.arithmetic_intensity,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Grafo de Fluxo de Controle (CFG)
# ──────────────────────────────────────────────────────────────────────────────

_TERMINATOR_OPS = {"ret", "exit", "brx"}


def build_cfg(kernel: PTXKernel) -> Tuple[Dict[str, BasicBlock], List[str]]:
    """
    Constrói o CFG do kernel a partir de suas instruções.

    Retorna (blocks, order):
        blocks: dict rótulo → BasicBlock
        order:  lista de rótulos na ordem de aparição no PTX
    """
    blocks: Dict[str, BasicBlock] = {}
    order: List[str] = []
    current_label = "__ENTRY__"
    current_instrs: List[PTXInstruction] = []

    def _flush():
        nonlocal current_label, current_instrs
        if not current_instrs:
            return
        if current_label not in blocks:
            blocks[current_label] = BasicBlock(current_label, current_instrs[:], [])
            order.append(current_label)
        current_instrs = []

    for instr in kernel.instructions:
        # Nova label → novo bloco básico
        if instr.label:
            _flush()
            current_label = instr.label

        current_instrs.append(instr)

        # Terminador de bloco (bra, ret, exit, brx)
        if instr.op_base in ("bra", "brx", "ret", "exit"):
            _flush()
            # Bloco sequencial anônimo que começa após o branch
            current_label = f"__seq_{len(blocks)}__"

    _flush()

    # Adicionar arestas
    for i, lbl in enumerate(order):
        block = blocks[lbl]
        if not block.instructions:
            continue
        last = block.instructions[-1]

        if last.op_base == "bra":
            target = last.operands[-1] if last.operands else None
            if last.is_predicated:
                # Condicional: vai para target OU cai no próximo bloco
                if target and target in blocks:
                    block.exits.append(("conditional", target))
                if i + 1 < len(order):
                    block.exits.append(("fallthrough", order[i + 1]))
            else:
                # Incondicional (bra.uni): só vai para target
                if target and target in blocks:
                    block.exits.append(("jump", target))
                block.is_terminal = True  # sem fall-through sequencial
        elif last.op_base in _TERMINATOR_OPS:
            block.is_terminal = True
        else:
            # Fall-through implícito
            if i + 1 < len(order):
                block.exits.append(("fallthrough", order[i + 1]))

    # Marcar terminais sem saídas
    for block in blocks.values():
        if not block.exits:
            block.is_terminal = True

    if order:
        blocks[order[0]].is_entry = True

    return blocks, order
