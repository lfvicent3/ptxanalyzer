"""
Taxonomia e modelo de dados básicos para a análise PTX.
"""

from __future__ import annotations

from collections import Counter, deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

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
    inline_source_file: int = 0
    inline_source_line: int = 0
    inline_source_col: int = 0


@dataclass
class BasicBlock:
    """Bloco básico do grafo de fluxo de controle (CFG)."""
    label: str
    instructions: List[PTXInstruction]
    # (tipo_aresta, rótulo_destino): "conditional"|"jump"|"fallthrough"
    exits: List[Tuple[str, str]] = field(default_factory=list)
    is_entry: bool = False
    is_terminal: bool = False   # ret/exit ou bra.uni sem fall-through
    display_name: str = ""
    description: str = ""


@dataclass
class CFGEdge:
    source: str
    target: str
    edge_type: str
    instruction_line: int = 0
    source_file: int = 0
    source_line: int = 0
    predicate: str = ""
    is_back_edge: bool = False


@dataclass
class BranchSite:
    block_label: str
    branch_kind: str
    line_no: int
    raw: str
    predicate: str = ""
    source_file: int = 0
    source_line: int = 0
    taken_target: Optional[str] = None
    fallthrough_target: Optional[str] = None
    setp_line: int = 0
    setp_raw: str = ""
    taken_instruction_count: int = 0
    fallthrough_instruction_count: int = 0
    taken_memory_ops: int = 0
    fallthrough_memory_ops: int = 0
    divergence_risk: str = "none"
    reconvergence_target: Optional[str] = None
    idle_threads_possible: bool = False
    simt_effect: str = ""

    def to_dict(self) -> dict:
        return {
            "block_label": self.block_label,
            "branch_kind": self.branch_kind,
            "line_no": self.line_no,
            "raw": self.raw,
            "predicate": self.predicate,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "taken_target": self.taken_target,
            "fallthrough_target": self.fallthrough_target,
            "setp_line": self.setp_line,
            "setp_raw": self.setp_raw,
            "taken_instruction_count": self.taken_instruction_count,
            "fallthrough_instruction_count": self.fallthrough_instruction_count,
            "taken_memory_ops": self.taken_memory_ops,
            "fallthrough_memory_ops": self.fallthrough_memory_ops,
            "divergence_risk": self.divergence_risk,
            "reconvergence_target": self.reconvergence_target,
            "idle_threads_possible": self.idle_threads_possible,
            "simt_effect": self.simt_effect,
        }


@dataclass
class MemoryHotspot:
    block_label: str
    instruction_count: int
    memory_ops: int
    global_loads: int = 0
    global_stores: int = 0
    shared_accesses: int = 0
    local_accesses: int = 0
    source_file: int = 0
    source_line: int = 0

    @property
    def memory_density(self) -> float:
        return round(self.memory_ops / max(self.instruction_count, 1), 4)

    def to_dict(self) -> dict:
        return {
            "block_label": self.block_label,
            "instruction_count": self.instruction_count,
            "memory_ops": self.memory_ops,
            "global_loads": self.global_loads,
            "global_stores": self.global_stores,
            "shared_accesses": self.shared_accesses,
            "local_accesses": self.local_accesses,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "memory_density": self.memory_density,
        }


@dataclass
class LoopSite:
    header: str
    latch: str
    edge_type: str
    source_line: int = 0

    def to_dict(self) -> dict:
        return {
            "header": self.header,
            "latch": self.latch,
            "edge_type": self.edge_type,
            "source_line": self.source_line,
        }


@dataclass
class ControlFlowAnalysis:
    blocks: Dict[str, BasicBlock]
    order: List[str]
    edges: List[CFGEdge]
    branch_sites: List[BranchSite]
    memory_hotspots: List[MemoryHotspot]
    loop_sites: List[LoopSite] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "blocks": {
                label: {
                    "instruction_count": len(block.instructions),
                    "is_entry": block.is_entry,
                    "is_terminal": block.is_terminal,
                    "display_name": block.display_name,
                    "description": block.description,
                    "exits": [{"type": et, "target": target} for et, target in block.exits],
                }
                for label, block in self.blocks.items()
            },
            "order": list(self.order),
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "edge_type": edge.edge_type,
                    "instruction_line": edge.instruction_line,
                    "source_file": edge.source_file,
                    "source_line": edge.source_line,
                    "predicate": edge.predicate,
                    "is_back_edge": edge.is_back_edge,
                }
                for edge in self.edges
            ],
            "branch_sites": [site.to_dict() for site in self.branch_sites],
            "memory_hotspots": [hotspot.to_dict() for hotspot in self.memory_hotspots],
            "loop_sites": [loop.to_dict() for loop in self.loop_sites],
        }


@dataclass
class PTXASInfo:
    kernel_name: str
    registers: int = 0
    shared_mem_bytes: int = 0
    constant_mem_bytes: int = 0
    local_mem_bytes: int = 0
    stack_frame_bytes: int = 0
    spill_stores_bytes: int = 0
    spill_loads_bytes: int = 0
    raw_lines: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kernel_name": self.kernel_name,
            "registers": self.registers,
            "shared_mem_bytes": self.shared_mem_bytes,
            "constant_mem_bytes": self.constant_mem_bytes,
            "local_mem_bytes": self.local_mem_bytes,
            "stack_frame_bytes": self.stack_frame_bytes,
            "spill_stores_bytes": self.spill_stores_bytes,
            "spill_loads_bytes": self.spill_loads_bytes,
            "raw_lines": list(self.raw_lines),
        }


@dataclass
class PTXKernel:
    name: str
    instructions: List[PTXInstruction] = field(default_factory=list)
    reg_decls: Dict[str, Set[str]] = field(default_factory=dict)
    param_count: int = 0
    file_map: Dict[int, str] = field(default_factory=dict)  # idx → path do .cu
    ptxas_info: Optional[PTXASInfo] = None

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
    def arithmetic_instructions(self) -> int:
        return self.category_counts.get("arithmetic", 0)

    @property
    def memory_instructions(self) -> int:
        return self.category_counts.get("memory", 0)

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
    def bar_sync_count(self) -> int:
        return sum(1 for i in self.instructions if i.op_base == "bar" and "sync" in i.op)

    @property
    def fma_count(self) -> int:
        return sum(1 for i in self.instructions if i.op_base == "fma")

    @property
    def arithmetic_intensity(self) -> float:
        """Razão aritmética/memória — mantém compatibilidade com versões anteriores."""
        arith = self.arithmetic_instructions
        mem   = max(self.memory_instructions, 1)
        return round(arith / mem, 3)

    @property
    def arithmetic_ratio(self) -> float:
        """Fração de instruções aritméticas no kernel."""
        total = max(self.total_instructions, 1)
        return round(self.arithmetic_instructions / total, 4)

    @property
    def instruction_intensity(self) -> float:
        """Instruções por transação/acesso de memória aproximado."""
        mem_tx = max(self.total_mem_accesses, 1)
        return round(self.total_instructions / mem_tx, 3)

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
    def memory_density(self) -> Dict[str, float]:
        total = max(self.total_instructions, 1)
        return {
            "global_load_density": round(self.global_loads / total, 4),
            "global_store_density": round(self.global_stores / total, 4),
            "shared_density": round(self.shared_accesses / total, 4),
            "local_density": round(self.local_accesses / total, 4),
            "global_memory_density": round((self.global_loads + self.global_stores) / total, 4),
        }

    @property
    def registers_per_thread(self) -> int:
        if self.ptxas_info is not None and self.ptxas_info.registers > 0:
            return self.ptxas_info.registers
        return self.total_registers

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
    def divergent_branch_count(self) -> int:
        return self.predicated_branches

    @property
    def instruction_mix(self) -> Dict[str, float]:
        """Distribuição percentual de categorias de instrução."""
        total = max(self.total_instructions, 1)
        return {cat: round(n / total * 100, 1)
                for cat, n in self.category_counts.items()}

    @property
    def cfg_stats(self) -> Dict[str, int]:
        cfg = analyze_control_flow(self)
        edges = len(cfg.edges)
        loops = sum(1 for edge in cfg.edges if edge.is_back_edge)
        return {
            "basic_blocks": len(cfg.blocks),
            "edges": edges,
            "loops": loops,
        }

    @property
    def control_flow(self) -> ControlFlowAnalysis:
        return analyze_control_flow(self)

    @property
    def branch_sites(self) -> List[BranchSite]:
        return self.control_flow.branch_sites

    @property
    def memory_hotspots(self) -> List[MemoryHotspot]:
        return self.control_flow.memory_hotspots

    @property
    def basic_block_count(self) -> int:
        return self.cfg_stats["basic_blocks"]

    @property
    def cfg_edge_count(self) -> int:
        return self.cfg_stats["edges"]

    @property
    def cfg_loop_count(self) -> int:
        return self.cfg_stats["loops"]

    def metrics_dict(self) -> dict:
        ptxas = self.ptxas_info
        return {
            "Instruções":           self.total_instructions,
            "RegistradoresDeclarados": self.total_registers,
            "Regs/Thread":          self.registers_per_thread,
            "ld.global":            self.global_loads,
            "st.global":            self.global_stores,
            "ld/st.local":          self.local_accesses,
            "Shared":               self.shared_accesses,
            "bar.sync":             self.bar_sync_count,
            "BranchTotal":          self.total_branches,
            "BranchCondicional":    self.predicated_branches,
            "BranchDivergente":     self.divergent_branch_count,
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
            "ArithmeticRatio":      self.arithmetic_ratio,
            "InstructionIntensity": self.instruction_intensity,
            "BasicBlocks":          self.basic_block_count,
            "CFGEdges":             self.cfg_edge_count,
            "CFGLoops":             self.cfg_loop_count,
            "GlobalLoadDensity":    self.memory_density["global_load_density"],
            "GlobalStoreDensity":   self.memory_density["global_store_density"],
            "GlobalMemDensity":     self.memory_density["global_memory_density"],
            "PTXASRegs":            ptxas.registers if ptxas else 0,
            "PTXASSmemBytes":       ptxas.shared_mem_bytes if ptxas else 0,
            "PTXASCmemBytes":       ptxas.constant_mem_bytes if ptxas else 0,
            "PTXASLocalBytes":      ptxas.local_mem_bytes if ptxas else 0,
            "PTXASStackFrame":      ptxas.stack_frame_bytes if ptxas else 0,
            "PTXASSpillStores":     ptxas.spill_stores_bytes if ptxas else 0,
            "PTXASSpillLoads":      ptxas.spill_loads_bytes if ptxas else 0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Grafo de Fluxo de Controle (CFG)
# ──────────────────────────────────────────────────────────────────────────────

_TERMINATOR_OPS = {"ret", "exit", "brx"}


def _humanize_operand(operand: str) -> str:
    operand = operand.strip()
    return operand if operand else "valor"


def explain_instruction(instr: Optional[PTXInstruction]) -> str:
    if instr is None:
        return "Sem instrução associada"

    op = instr.op_base
    ops = instr.operands

    if op == "setp":
        if any(token in instr.op for token in (".le.", ".lt.", ".gt.", ".ge.")):
            if len(ops) >= 3:
                return (
                    "Comparando valores para decidir o próximo passo "
                    f"entre {_humanize_operand(ops[1])} e {_humanize_operand(ops[2])}"
                )
        if len(ops) >= 3:
            return f"Testando condição entre {_humanize_operand(ops[1])} e {_humanize_operand(ops[2])}"
        return "Testando condição"
    if op == "bra":
        target = ops[-1] if ops else "destino"
        return f"Desvio condicional para {target}" if instr.is_predicated else f"Saltando para {target}"
    if op == "ld":
        scope = (
            "global" if "global" in instr.op else
            "shared" if "shared" in instr.op else
            "local" if "local" in instr.op else
            "de parâmetros" if "param" in instr.op else
            "memória"
        )
        if any(offset in instr.raw for offset in ("+4", "-4", "+8", "-8", "+12", "-12", "+16", "-16")):
            return f"Carregando elementos adjacentes da memória {scope}"
        return f"Carregando dados da memória {scope}"
    if op == "st":
        scope = (
            "global" if "global" in instr.op else
            "shared" if "shared" in instr.op else
            "local" if "local" in instr.op else
            "de parâmetros" if "param" in instr.op else
            "memória"
        )
        if "shared" in instr.op:
            return "Escrevendo resultado do swap na memória shared"
        return f"Escrevendo dados na memória {scope}"
    if op == "add":
        return "Efetuando soma"
    if op == "sub":
        return "Efetuando subtração"
    if op in {"mul", "mad", "fma"}:
        return "Efetuando multiplicação"
    if op in {"and", "or", "xor", "shl", "shr"}:
        return "Aplicando operação lógica"
    if op == "mov":
        return "Movendo valor entre registradores"
    if op == "selp":
        return "Selecionando valor por predicado"
    if op == "call":
        callee = ops[1] if len(ops) >= 2 else ops[0] if ops else "função"
        if "add" in callee:
            return "Incrementa o valor"
        if "sub" in callee:
            return "Decrementa o valor"
        return f"Chamando {callee}"
    if op in {"ret", "exit"}:
        return "Encerrando kernel"
    return instr.raw.strip()


def _block_primary_source_line(block: BasicBlock) -> int:
    for instr in block.instructions:
        if instr.source_line > 0 and instr.op_base not in {"ret", "exit"}:
            return instr.source_line
    for instr in block.instructions:
        if instr.source_line > 0:
            return instr.source_line
    return 0


def describe_block(block: BasicBlock) -> tuple[str, str]:
    last = block.instructions[-1] if block.instructions else None
    setp_instr = next((instr for instr in reversed(block.instructions) if instr.op_base == "setp"), None)
    shared_loads = [instr for instr in block.instructions if instr.op_base == "ld" and "shared" in instr.op]
    shared_stores = [instr for instr in block.instructions if instr.op_base == "st" and "shared" in instr.op]

    if last and last.op_base == "bra" and last.is_predicated:
        if shared_loads and setp_instr is not None:
            return "Decisão", "Compara elementos adjacentes e decide se precisa trocar"
        condition_text = explain_instruction(setp_instr) if setp_instr else "Avaliando predicado para decidir o próximo bloco"
        if block.is_entry:
            return "Entrada", condition_text
        return "Decisão", condition_text

    if block.is_entry:
        return "Entrada", "Inicializando contexto do kernel"
    if block.is_terminal:
        return "Saída", explain_instruction(last) if last else "Encerrando fluxo"

    call_instr = next((instr for instr in block.instructions if instr.op_base == "call"), None)
    if call_instr is not None:
        return "Chamada", explain_instruction(call_instr)

    if shared_stores:
        return "Escrita", "Realiza swap / atualização dos valores na memória shared"

    for instr in block.instructions:
        if instr.op_base == "st":
            return "Escrita", explain_instruction(instr)
        if instr.op_base == "ld":
            return "Leitura", explain_instruction(instr)
        if instr.op_base in {"add", "sub", "mul", "mad", "fma"}:
            return "Cálculo", explain_instruction(instr)

    if last and last.op_base == "bra":
        return "Salto", "Segue para o próximo ramo do fluxo"
    if block.label.startswith("__seq_"):
        return "Sequência", "Executando instruções sequenciais"
    if block.label == "__ENTRY__":
        return "Entrada", "Inicializando contexto do kernel"
    return "Bloco", explain_instruction(last) if last else "Executando instruções"


def annotate_basic_blocks(blocks: Dict[str, BasicBlock], order: List[str]) -> None:
    title_counts: Dict[str, int] = defaultdict(int)
    repeated_lines: Dict[int, int] = defaultdict(int)
    repeated_line_totals = Counter(
        _block_primary_source_line(block)
        for block in blocks.values()
        if _block_primary_source_line(block) > 0
    )
    for label in order:
        block = blocks[label]
        title, description = describe_block(block)
        source_line = _block_primary_source_line(block)
        if label == "__ENTRY__":
            display_name = "Entrada"
        else:
            title_counts[title] += 1
            display_name = f"{title} {title_counts[title]}"
        if source_line > 0 and repeated_line_totals[source_line] > 1:
            repeated_lines[source_line] += 1
            description = f"{description} (instância desenrolada {repeated_lines[source_line]})"
        block.display_name = display_name
        block.description = description


def _merge_redundant_same_line_fallthrough_blocks(
    blocks: Dict[str, BasicBlock],
    order: List[str],
) -> Tuple[Dict[str, BasicBlock], List[str]]:
    changed = True
    while changed:
        changed = False
        predecessors: Dict[str, List[str]] = defaultdict(list)
        for src in order:
            for _, target in blocks[src].exits:
                if target in blocks:
                    predecessors[target].append(src)

        for idx in range(len(order) - 1):
            current_label = order[idx]
            next_label = order[idx + 1]
            current = blocks[current_label]
            nxt = blocks[next_label]
            current_line = _block_primary_source_line(current)
            next_line = _block_primary_source_line(nxt)

            if current_line == 0 or current_line != next_line:
                continue
            if current.is_terminal or nxt.is_entry:
                continue
            if len(current.exits) != 1 or current.exits[0] != ("fallthrough", next_label):
                continue
            if predecessors.get(next_label, []) != [current_label]:
                continue

            current.instructions.extend(nxt.instructions)
            current.exits = list(nxt.exits)
            current.is_terminal = nxt.is_terminal
            del blocks[next_label]
            order.pop(idx + 1)
            changed = True
            break

    return blocks, order


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

    blocks, order = _merge_redundant_same_line_fallthrough_blocks(blocks, order)
    annotate_basic_blocks(blocks, order)

    return blocks, order


def _count_cfg_loops(blocks: Dict[str, BasicBlock], order: List[str]) -> int:
    """Conta loops por back-edges detectados via DFS."""
    if not order:
        return 0

    entry = order[0]
    visited: Set[str] = set()
    stack: Set[str] = set()
    back_edges: Set[Tuple[str, str]] = set()
    dfs_iter = [(entry, iter(blocks[entry].exits))]
    visited.add(entry)
    stack.add(entry)

    while dfs_iter:
        lbl, children = dfs_iter[-1]
        try:
            _, target = next(children)
            if target not in blocks:
                continue
            if target in stack:
                back_edges.add((lbl, target))
            elif target not in visited:
                visited.add(target)
                stack.add(target)
                dfs_iter.append((target, iter(blocks[target].exits)))
        except StopIteration:
            stack.discard(lbl)
            dfs_iter.pop()

    return len(back_edges)


def _find_back_edges(blocks: Dict[str, BasicBlock], order: List[str]) -> Set[Tuple[str, str]]:
    if not order:
        return set()

    entry = order[0]
    visited: Set[str] = set()
    stack: Set[str] = set()
    back_edges: Set[Tuple[str, str]] = set()
    dfs_iter = [(entry, iter(blocks[entry].exits))]
    visited.add(entry)
    stack.add(entry)

    while dfs_iter:
        lbl, children = dfs_iter[-1]
        try:
            _, target = next(children)
            if target not in blocks:
                continue
            if target in stack:
                back_edges.add((lbl, target))
            elif target not in visited:
                visited.add(target)
                stack.add(target)
                dfs_iter.append((target, iter(blocks[target].exits)))
        except StopIteration:
            stack.discard(lbl)
            dfs_iter.pop()

    return back_edges


def _successor_map(blocks: Dict[str, BasicBlock], order: List[str]) -> Dict[str, Set[str]]:
    return {
        label: {target for _, target in blocks[label].exits if target in blocks}
        for label in order
    }


def _compute_postdominators(blocks: Dict[str, BasicBlock], order: List[str]) -> Dict[str, Set[str]]:
    if not order:
        return {}

    all_nodes = set(order)
    successors = _successor_map(blocks, order)
    terminals = {label for label in order if not successors[label]}
    if not terminals:
        terminals = {order[-1]}

    postdom: Dict[str, Set[str]] = {}
    for label in order:
        postdom[label] = {label} if label in terminals else set(all_nodes)

    changed = True
    while changed:
        changed = False
        for label in reversed(order):
            if label in terminals:
                continue
            succs = successors[label]
            if not succs:
                new_set = {label}
            else:
                intersection = set(all_nodes)
                for succ in succs:
                    intersection &= postdom[succ]
                new_set = {label} | intersection
            if new_set != postdom[label]:
                postdom[label] = new_set
                changed = True
    return postdom


def _compute_immediate_postdominators(blocks: Dict[str, BasicBlock], order: List[str]) -> Dict[str, Optional[str]]:
    postdom = _compute_postdominators(blocks, order)
    ipdom: Dict[str, Optional[str]] = {label: None for label in order}

    for label in order:
        candidates = postdom.get(label, set()) - {label}
        if not candidates:
            continue
        chosen = None
        for candidate in candidates:
            if all(candidate not in postdom.get(other, set()) for other in candidates if other != candidate):
                chosen = candidate
                break
        ipdom[label] = chosen
    return ipdom


def _count_block_memory_ops(block: Optional[BasicBlock]) -> int:
    if block is None:
        return 0
    return sum(1 for instr in block.instructions if instr.op_base in ("ld", "st", "atom", "red"))


def _build_memory_hotspot(block: BasicBlock) -> MemoryHotspot:
    global_loads = 0
    global_stores = 0
    shared_accesses = 0
    local_accesses = 0
    memory_ops = 0
    source_file = 0
    source_line = 0

    for instr in block.instructions:
        if source_line == 0 and instr.source_line > 0:
            source_file = instr.source_file
            source_line = instr.source_line
        if instr.op_base not in ("ld", "st", "atom", "red"):
            continue
        memory_ops += 1
        if instr.op_base == "ld" and "global" in instr.op:
            global_loads += 1
        if instr.op_base == "st" and "global" in instr.op:
            global_stores += 1
        if instr.op_base in ("ld", "st") and "shared" in instr.op:
            shared_accesses += 1
        if instr.op_base in ("ld", "st") and "local" in instr.op:
            local_accesses += 1

    return MemoryHotspot(
        block_label=block.label,
        instruction_count=len(block.instructions),
        memory_ops=memory_ops,
        global_loads=global_loads,
        global_stores=global_stores,
        shared_accesses=shared_accesses,
        local_accesses=local_accesses,
        source_file=source_file,
        source_line=source_line,
    )


def _find_setp_for_branch(instrs: List[PTXInstruction], bra_idx: int) -> Optional[PTXInstruction]:
    bra = instrs[bra_idx]
    if not bra.is_predicated or not bra.predicate:
        return None
    pred_reg = bra.predicate.lstrip("@").lstrip("!")
    for i in range(bra_idx - 1, -1, -1):
        ins = instrs[i]
        if ins.op_base == "setp" and ins.operands and ins.operands[0] == pred_reg:
            return ins
    return None


def _estimate_divergence_risk(instr: PTXInstruction,
                              taken_block: Optional[BasicBlock],
                              fallthrough_block: Optional[BasicBlock]) -> str:
    if not instr.is_predicated:
        return "none"

    score = 0
    if taken_block is not None:
        score += 1
        if taken_block.instructions:
            score += 1
    if fallthrough_block is not None:
        score += 1
        if fallthrough_block.instructions:
            score += 1

    mem_pressure = _count_block_memory_ops(taken_block) + _count_block_memory_ops(fallthrough_block)
    if mem_pressure > 0:
        score += 1
    if mem_pressure >= 4:
        score += 1

    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def analyze_control_flow(kernel: PTXKernel) -> ControlFlowAnalysis:
    """
    Monta uma representação estruturada do fluxo de controle do kernel.

    Inclui:
      - blocos básicos e ordem de aparição
      - arestas com tipo (condicional/jump/fallthrough)
      - pontos de branch (`bra`) com alvo, fall-through e risco de divergência
    """
    blocks, order = build_cfg(kernel)
    back_edges = _find_back_edges(blocks, order)
    ipdom = _compute_immediate_postdominators(blocks, order)

    edges: List[CFGEdge] = []
    branch_sites: List[BranchSite] = []
    memory_hotspots: List[MemoryHotspot] = []
    loop_sites: List[LoopSite] = []

    for src, target in sorted(back_edges):
        src_block = blocks.get(src)
        last = src_block.instructions[-1] if src_block and src_block.instructions else None
        edge_type = next((kind for kind, dst in src_block.exits if dst == target), "loop") if src_block else "loop"
        loop_sites.append(LoopSite(
            header=target,
            latch=src,
            edge_type=edge_type,
            source_line=last.source_line if last else 0,
        ))

    for lbl in order:
        block = blocks[lbl]
        last = block.instructions[-1] if block.instructions else None
        hotspot = _build_memory_hotspot(block)
        if hotspot.memory_ops > 0:
            memory_hotspots.append(hotspot)

        for edge_type, target in block.exits:
            edges.append(CFGEdge(
                source=lbl,
                target=target,
                edge_type=edge_type,
                instruction_line=last.line_no if last else 0,
                source_file=last.source_file if last else 0,
                source_line=last.source_line if last else 0,
                predicate=last.predicate if last and last.is_predicated else "",
                is_back_edge=(lbl, target) in back_edges,
            ))

        if not last or last.op_base != "bra":
            continue

        taken_target = last.operands[-1] if last.operands else None
        fallthrough_target = None
        for edge_type, target in block.exits:
            if edge_type == "fallthrough":
                fallthrough_target = target
                break

        taken_block = blocks.get(taken_target) if taken_target else None
        fallthrough_block = blocks.get(fallthrough_target) if fallthrough_target else None
        setp_instr = _find_setp_for_branch(block.instructions, len(block.instructions) - 1)
        idle_threads_possible = bool(last.is_predicated and taken_target and fallthrough_target)
        simt_effect = (
            "Threads do mesmo warp podem se dividir entre os dois caminhos até a reconvergência."
            if idle_threads_possible else
            "Sem perda SIMT relevante neste desvio."
        )

        branch_sites.append(BranchSite(
            block_label=lbl,
            branch_kind="predicated" if last.is_predicated else "unconditional",
            line_no=last.line_no,
            raw=last.raw,
            predicate=last.predicate,
            source_file=last.source_file,
            source_line=last.source_line,
            taken_target=taken_target,
            fallthrough_target=fallthrough_target,
            setp_line=setp_instr.line_no if setp_instr else 0,
            setp_raw=setp_instr.raw if setp_instr else "",
            taken_instruction_count=len(taken_block.instructions) if taken_block else 0,
            fallthrough_instruction_count=len(fallthrough_block.instructions) if fallthrough_block else 0,
            taken_memory_ops=_count_block_memory_ops(taken_block),
            fallthrough_memory_ops=_count_block_memory_ops(fallthrough_block),
            divergence_risk=_estimate_divergence_risk(last, taken_block, fallthrough_block),
            reconvergence_target=ipdom.get(lbl),
            idle_threads_possible=idle_threads_possible,
            simt_effect=simt_effect,
        ))

    return ControlFlowAnalysis(
        blocks=blocks,
        order=order,
        edges=edges,
        branch_sites=branch_sites,
        memory_hotspots=sorted(
            memory_hotspots,
            key=lambda item: (item.memory_ops, item.memory_density, item.instruction_count),
            reverse=True,
        ),
        loop_sites=loop_sites,
    )
