"""
Executor genérico de PTX/CFG — o "modo dinâmico" real do analisador.

Este módulo substitui os simuladores anteriores, que eram escritos um a
um por algoritmo (`_simulate_bubble_dynamic`, `_simulate_insertion_dynamic`,
`_simulate_smoke_dynamic` em `analyzer.py`). Em vez de reconhecer o nome
do kernel e reproduzir manualmente a lógica do algoritmo em Python, o
`PTXInterpreter` interpreta as instruções PTX de verdade sobre o CFG
estático já construído por `core.build_cfg`, usando valores concretos de
entrada fornecidos pelo usuário.

Suportado hoje (ver `SUPPORTED_OPCODES`):
  - controle: `bra` (condicional/incondicional), `ret`/`exit`, `call.uni`
    para funções device não-inline (convenção nvcc: `stN.param` para
    argumentos, `st.param [func_retvalN+0], ...` para o retorno)
  - dados: `mov`, `cvt`, `cvta`
  - aritmética: `add`, `sub`, `mul` (`.wide` incluso, `.hi` não), `mad`
    (idem), `min`, `max`, `neg`, `abs`
  - bit a bit: `and`, `or`, `xor`, `not`, `shl`, `shr`
  - comparação/seleção: `setp` (eq/ne/lt/le/gt/ge e variantes
    unsigned lo/ls/hi/hs), `selp`
  - memória: `ld`/`st` em `.global`/`.shared`/`.local`/`.param`,
    endereçamento genérico byte a byte (qualquer largura/sinal/float
    conforme o sufixo do opcode)
  - sincronização: `bar.sync` sincroniza threads do mesmo CTA (scheduler
    cooperativo por bloco quando o kernel usa barreiras); `membar`/
    `fence`/`nop` seguem como marcadores sem efeito de dados

Não suportado ainda (fica como `unsupported_ops` no traço, sem travar o
processo inteiro nem inventar um resultado): `div`/`rem`, `mul.hi`/
`mad.hi`, transcendentais (`sin`/`cos`/`sqrt`/...), operações atômicas,
`shfl`/warp vote, `ld`/`st` vetoriais (`.v2`/`.v4`), chamadas por
ponteiro de função. Para estender: adicione um novo ramo em
`PTXInterpreter._execute_instruction` (ou em `SUPPORTED_OPCODES` + handler
dedicado) — o resto do motor (CFG, memória, traço) já é genérico e não
precisa mudar.

Limitação estrutural conhecida (não é bug, é o modelo escolhido): fora
dos pontos de `bar.sync`, as threads continuam sendo executadas de forma
determinística pelo scheduler do simulador, não por um modelo completo de
concorrência/warp scheduling da GPU real. Isso já é suficiente para
kernels cooperativos clássicos com `__shared__ + __syncthreads()`, mas
não modela interleavings arbitrários, atomics ou efeitos de memória mais
avançados.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .core import BasicBlock, PTXInstruction, PTXKernel, build_cfg
from .state import (
    BranchDecision,
    ByteMemory,
    Frame,
    KernelArg,
    KernelDynamicTrace,
    KernelLaunchConfig,
    Number,
    ThreadContext,
    ThreadTrace,
    UnsupportedInstruction,
)

# ──────────────────────────────────────────────────────────────────────────────
# Regex/tipos auxiliares
# ──────────────────────────────────────────────────────────────────────────────

_TYPE_TOKEN_RE = re.compile(r'^[bsuf](8|16|32|64)$')
_PARAM_SYMBOL_RE = re.compile(r'_param_(\d+)$')
# O offset pode vir com sinal duplo (`[%rd15+-4]` — nvcc gera "+" seguido
# de um literal negativo), então o sinal e a magnitude são capturados
# separadamente em vez de assumir um único caractere de sinal.
_MEM_OPERAND_RE = re.compile(r'^\[\s*([^\[\]+-]+?)\s*(?:([+-])\s*([+-]?\d+))?\s*\]$')
_HEX_FLOAT32_RE = re.compile(r'^0[fF]([0-9a-fA-F]{8})$')
_HEX_FLOAT64_RE = re.compile(r'^0[dD]([0-9a-fA-F]{16})$')
_INT_LITERAL_RE = re.compile(r'^-?0[xX][0-9a-fA-F]+$|^-?\d+$')
_SPACE_TOKENS = ("global", "shared", "local", "const", "param")

# Declaração de um array nomeado em memória `.shared`/`.local` (ex.:
# `.extern .shared .align 4 .b8 shared_segment[];` para memória shared
# dinâmica, ou `.local .align 4 .b8 __local_depot0[32];` para um array
# que o compilador decidiu manter na pilha/local em vez de registrador).
# O endereço desses arrays é obtido via `mov.TYPE %reg, NOME;` — capturado
# genericamente a partir do texto bruto do PTX, não de um kernel específico.
_RE_NAMED_MEM_DECL = re.compile(r'\.(shared|local)\b[^;\n{}]*?\b([A-Za-z_$][\w$]*)\s*\[')


def _discover_named_symbols(raw_ptx: str) -> Dict[str, str]:
    symbols: Dict[str, str] = {}
    for match in _RE_NAMED_MEM_DECL.finditer(raw_ptx or ""):
        space, name = match.group(1), match.group(2)
        symbols.setdefault(name, space)
    return symbols

# `call.uni` costuma vir espalhado por várias linhas no PTX original; o
# parser já reconstitui `instr.raw` como uma única string legível, então
# extraímos a lista de retorno/nome/argumentos daqui em vez de depender do
# split ingênuo por vírgula de `instr.operands` (que quebraria com mais de
# um argumento).
_RE_CALL_UNI = re.compile(
    r'call(?:\.uni)?\s*'
    r'(?:\(\s*([^()]*?)\s*\)\s*,\s*)?'
    r'([\w$.]+)\s*'
    r'(?:,\s*\(\s*([^()]*?)\s*\))?'
    r'\s*;?\s*$'
)

SUPPORTED_OPCODES = {
    "mov", "cvt", "cvta",
    "add", "sub", "mul", "mad", "min", "max", "neg", "abs",
    "and", "or", "xor", "not", "shl", "shr",
    "setp", "selp",
    "ld", "st",
    "bra", "ret", "exit",
    "call",
    "bar", "membar", "fence", "nop",
}


def _type_width_bytes(ptx_type: str) -> int:
    if ptx_type == "pred":
        return 1
    match = _TYPE_TOKEN_RE.match(ptx_type)
    if match:
        return int(match.group(1)) // 8
    return 4


def _type_is_float(ptx_type: str) -> bool:
    return ptx_type.startswith("f")


def _type_is_signed(ptx_type: str) -> bool:
    return ptx_type.startswith("s")


def _last_type_token(op: str) -> Optional[str]:
    for part in reversed(op.split(".")):
        if _TYPE_TOKEN_RE.match(part):
            return part
    return None


def _vector_width(op: str) -> int:
    for part in op.split("."):
        if part.startswith("v") and part[1:].isdigit():
            return int(part[1:])
    return 1


def _classify_space(op: str) -> Optional[str]:
    parts = set(op.split("."))
    for token in _SPACE_TOKENS:
        if token in parts:
            return token
    return None


def _normalize_vector_operands(ops: List[str]) -> List[str]:
    return [op.strip().strip("{}").strip() for op in ops]


def build_reg_type_map(kernel: PTXKernel) -> Dict[str, str]:
    """Tipo declarado (`.reg .TYPE`) de cada registrador do kernel/função.
    É a partir daqui — não do sufixo de cada instrução — que decidimos a
    largura/sinal usados para mascarar o valor escrito, o que é mais fiel
    ao PTX real do que assumir um tamanho fixo."""
    mapping: Dict[str, str] = {}
    for ptx_type, names in kernel.reg_decls.items():
        for name in names:
            mapping[name] = ptx_type
    return mapping


@dataclass
class _KernelScope:
    kernel: PTXKernel
    blocks: Dict[str, BasicBlock]
    order: List[str]
    reg_types: Dict[str, str]


@dataclass
class _ThreadExecState:
    ctx: ThreadContext
    trace: ThreadTrace
    frame: Frame
    current_label: Optional[str]
    instr_index: int = 0
    steps: int = 0
    halted: bool = False
    halt_reason: str = ""
    waiting_barrier: Optional[Tuple[str, int, str]] = None
    snapshot_buffers: bool = False
    needs_block_entry: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Avaliação de operandos / registradores
# ──────────────────────────────────────────────────────────────────────────────

def _write_register(reg_types: Dict[str, str], frame: Frame, name: str, value: Number) -> None:
    if not name.startswith("%"):
        return
    ptx_type = reg_types.get(name)
    if ptx_type == "pred":
        frame.predicates[name] = bool(value)
        return
    if ptx_type is None:
        frame.registers[name] = value
        return
    if _type_is_float(ptx_type):
        frame.registers[name] = float(value)
        return
    width = _type_width_bytes(ptx_type)
    mask = (1 << (width * 8)) - 1
    ival = int(value) & mask
    if _type_is_signed(ptx_type) and ival >= (1 << (width * 8 - 1)):
        ival -= (1 << (width * 8))
    frame.registers[name] = ival


def _eval_operand(reg_types: Dict[str, str], ctx: ThreadContext, frame: Frame, op: str) -> Number:
    op = op.strip()
    if op.startswith("%"):
        special = ctx.special_register(op)
        if special is not None:
            return special
        if op in frame.predicates:
            return frame.predicates[op]
        return frame.registers.get(op, 0)
    match = _HEX_FLOAT32_RE.match(op)
    if match:
        return struct.unpack(">f", bytes.fromhex(match.group(1)))[0]
    match = _HEX_FLOAT64_RE.match(op)
    if match:
        return struct.unpack(">d", bytes.fromhex(match.group(1)))[0]
    if _INT_LITERAL_RE.match(op):
        return int(op, 0)
    try:
        return float(op)
    except ValueError as exc:
        raise UnsupportedInstruction(f"operando não reconhecido: {op!r}") from exc


def _eval_predicate_operand(frame: Frame, op: str) -> bool:
    op = op.strip()
    negate = op.startswith("!")
    name = op.lstrip("!")
    value = bool(frame.predicates.get(name, False))
    return (not value) if negate else value


def _parse_memory_operand(op: str) -> Tuple[str, int]:
    match = _MEM_OPERAND_RE.match(op.strip())
    if not match:
        raise UnsupportedInstruction(f"operando de memória não reconhecido: {op!r}")
    base = match.group(1).strip()
    sign, magnitude = match.group(2), match.group(3)
    if magnitude is None:
        offset = 0
    else:
        offset = int(magnitude)
        if sign == "-":
            offset = -offset
    return base, offset


# ──────────────────────────────────────────────────────────────────────────────
# Executor
# ──────────────────────────────────────────────────────────────────────────────

class PTXInterpreter:
    """Interpreta um `PTXKernel` (mais as funções device que ele chama,
    se houver) sobre valores concretos de entrada, produzindo um traço
    dinâmico real: blocos visitados, arestas tomadas, predicados
    avaliados e o conteúdo dos buffers antes/depois da execução.

    Não recebe nem consulta o nome do kernel/algoritmo em nenhum ponto de
    decisão — a única coisa que importa é o PTX e os `KernelArg`
    fornecidos pelo usuário.
    """

    def __init__(self,
                 kernel: PTXKernel,
                 functions: Optional[Dict[str, PTXKernel]] = None,
                 raw_ptx: str = "",
                 max_steps: int = 200_000,
                 max_call_depth: int = 32) -> None:
        self.kernel = kernel
        blocks, order = build_cfg(kernel)
        self._entry_scope = _KernelScope(kernel, blocks, order, build_reg_type_map(kernel))
        self._function_kernels = functions or {}
        self._function_scopes: Dict[str, _KernelScope] = {}
        self.max_steps = max_steps
        self.max_call_depth = max_call_depth

        self.global_mem = ByteMemory()
        self.param_values: Dict[int, Number] = {}
        self.pointer_param_indices: Set[int] = set()
        self.buffer_specs: Dict[int, dict] = {}
        self._next_buffer_base = 0x0001_0000
        self.unsupported_ops: List[str] = []

        # Arrays `.shared`/`.local` nomeados (ver `_discover_named_symbols`):
        # descobertos uma vez a partir do texto bruto do PTX; o endereço
        # de cada um é alocado sob demanda, na primeira vez que aparece
        # como operando de `mov`.
        self._named_symbol_space = _discover_named_symbols(raw_ptx)
        self._named_symbol_base: Dict[str, int] = {}
        self._named_symbol_next: Dict[str, int] = {}
        self._entry_has_barrier = any(instr.op_base == "bar" for instr in kernel.instructions)

    def _alloc_named_symbol(self, space: str) -> int:
        base = self._named_symbol_next.get(space, 0)
        self._named_symbol_next[space] = base + 0x1000
        return base

    # ── carregamento de argumentos ──────────────────────────────────────

    def load_args(self, args: List[KernelArg]) -> None:
        for arg in args:
            if arg.kind == "buffer":
                values = list(arg.values or [])
                base = self._next_buffer_base
                span = max(len(values) * arg.element_width, arg.element_width)
                self._next_buffer_base += span + 0x1000
                for i, value in enumerate(values):
                    self.global_mem.write_value(base + i * arg.element_width, value,
                                                 arg.element_width, arg.element_is_float)
                self.param_values[arg.index] = base
                self.pointer_param_indices.add(arg.index)
                self.buffer_specs[arg.index] = {
                    "base": base,
                    "length": len(values),
                    "width": arg.element_width,
                    "is_float": arg.element_is_float,
                    "signed": arg.element_signed,
                    "label": arg.label or f"param_{arg.index}",
                }
            else:
                self.param_values[arg.index] = arg.value

    def read_buffer(self, index: int) -> List[Number]:
        spec = self.buffer_specs[index]
        return self.global_mem.snapshot(spec["base"], spec["length"], spec["width"],
                                         spec["is_float"], spec["signed"])

    def buffers_snapshot(self) -> Dict[int, List[Number]]:
        return {idx: self.read_buffer(idx) for idx in self.buffer_specs}

    def buffer_labels(self) -> Dict[int, str]:
        return {idx: spec["label"] for idx, spec in self.buffer_specs.items()}

    # ── resolução de parâmetros/funções ─────────────────────────────────

    def _resolve_param_symbol(self, symbol: str) -> Optional[int]:
        match = _PARAM_SYMBOL_RE.search(symbol)
        return int(match.group(1)) if match else None

    def _get_function_scope(self, name: str) -> Optional[_KernelScope]:
        name = name.strip()
        if name in self._function_scopes:
            return self._function_scopes[name]
        func_kernel = self._function_kernels.get(name)
        if func_kernel is None:
            return None
        blocks, order = build_cfg(func_kernel)
        scope = _KernelScope(func_kernel, blocks, order, build_reg_type_map(func_kernel))
        self._function_scopes[name] = scope
        return scope

    def _memory_for(self, space: Optional[str], ctx: ThreadContext) -> ByteMemory:
        if space == "shared":
            return ctx.shared_mem
        if space == "local":
            return ctx.local_mem
        # "global"/"const"/genérico (sem tag conhecida): tratados como a
        # mesma memória global plana — é o caso comum destes kernels.
        return ctx.global_mem

    # ── laço de execução (thread) ───────────────────────────────────────

    def run_thread(self, ctx: ThreadContext, snapshot_buffers: bool = False) -> ThreadTrace:
        trace = ThreadTrace(thread_id=ctx.thread_id, tid=ctx.tid, ctaid=ctx.ctaid)
        entry_frame = Frame(param_values=dict(self.param_values),
                             pointer_param_indices=set(self.pointer_param_indices))
        halt_reason = self._run_frame_serial(self._entry_scope, ctx, entry_frame, call_depth=0,
                                             trace=trace, snapshot_buffers=snapshot_buffers)
        trace.halted = True
        trace.halt_reason = halt_reason
        return trace

    def _run_frame_serial(self,
                          scope: _KernelScope,
                          ctx: ThreadContext,
                          frame: Frame,
                          call_depth: int,
                          trace: Optional[ThreadTrace] = None,
                          snapshot_buffers: bool = False) -> str:
        current_label = scope.order[0] if scope.order else None
        steps = 0
        while current_label:
            steps += 1
            if steps > self.max_steps:
                return "limite de passos excedido (possível laço sem convergência)"
            block = scope.blocks.get(current_label)
            if block is None:
                return f"bloco desconhecido: {current_label}"
            if trace is not None:
                trace.blocks_visited.append(current_label)
                trace.steps_executed = steps

            next_label = None
            halt_reason = None
            for instr in block.instructions:
                if instr.op_base != "bra" and instr.is_predicated:
                    pred_name = instr.predicate.lstrip("@").lstrip("!")
                    negate = instr.predicate.startswith("@!")
                    pred_val = bool(frame.predicates.get(pred_name, False))
                    if negate:
                        pred_val = not pred_val
                    if not pred_val:
                        continue  # instrução com predicado falso: não executa

                if instr.op_base == "bra":
                    next_label = self._resolve_branch(frame, block, instr, trace)
                    break
                if instr.op_base in ("ret", "exit"):
                    halt_reason = "ret"
                    break
                try:
                    self._execute_instruction(scope, ctx, frame, instr, call_depth)
                except UnsupportedInstruction as exc:
                    self.unsupported_ops.append(instr.op)
                    if trace is not None:
                        trace.unsupported_ops.append(instr.op)
                    halt_reason = f"instrução não suportada ({instr.op}): {exc}"
                    break

            if snapshot_buffers and trace is not None:
                trace.block_snapshots.append((current_label, self.buffers_snapshot()))
            if halt_reason:
                return halt_reason
            if next_label is None:
                exits = block.exits
                next_label = exits[0][1] if exits else None
                if next_label and trace is not None:
                    trace.edges_taken.append((current_label, next_label))
            current_label = next_label
        return "ret"

    def _leave_block(self, state: _ThreadExecState, block_label: str) -> None:
        if state.snapshot_buffers:
            state.trace.block_snapshots.append((block_label, self.buffers_snapshot()))

    def _advance_to_next_block(self,
                               state: _ThreadExecState,
                               block: BasicBlock,
                               chosen: Optional[str] = None) -> None:
        next_label = chosen
        if next_label is None:
            exits = block.exits
            next_label = exits[0][1] if exits else None
            if next_label and state.trace is not None:
                state.trace.edges_taken.append((block.label, next_label))
        self._leave_block(state, block.label)
        state.current_label = next_label
        state.instr_index = 0
        state.needs_block_entry = True
        if next_label is None:
            state.halted = True
            state.halt_reason = "ret"
            state.trace.halted = True
            state.trace.halt_reason = "ret"

    def _barrier_token(self, block_label: str, instr_index: int, instr: PTXInstruction) -> Tuple[str, int, str]:
        barrier_id = instr.operands[0].strip() if instr.operands else "0"
        return (block_label, instr_index, barrier_id)

    def _step_thread_state(self, state: _ThreadExecState) -> bool:
        if state.halted or state.waiting_barrier is not None:
            return False
        if state.current_label is None:
            state.halted = True
            state.halt_reason = "ret"
            state.trace.halted = True
            state.trace.halt_reason = "ret"
            return True

        block = self._entry_scope.blocks.get(state.current_label)
        if block is None:
            state.halted = True
            state.halt_reason = f"bloco desconhecido: {state.current_label}"
            state.trace.halted = True
            state.trace.halt_reason = state.halt_reason
            return True

        if state.needs_block_entry:
            state.steps += 1
            if state.steps > self.max_steps:
                state.halted = True
                state.halt_reason = "limite de passos excedido (possível laço sem convergência)"
                state.trace.halted = True
                state.trace.halt_reason = state.halt_reason
                return True
            state.trace.blocks_visited.append(state.current_label)
            state.trace.steps_executed = state.steps
            state.needs_block_entry = False

        if state.instr_index >= len(block.instructions):
            self._advance_to_next_block(state, block)
            return True

        instr = block.instructions[state.instr_index]
        frame = state.frame
        ctx = state.ctx

        if instr.op_base != "bra" and instr.is_predicated:
            pred_name = instr.predicate.lstrip("@").lstrip("!")
            negate = instr.predicate.startswith("@!")
            pred_val = bool(frame.predicates.get(pred_name, False))
            if negate:
                pred_val = not pred_val
            if not pred_val:
                state.instr_index += 1
                return True

        if instr.op_base == "bra":
            chosen = self._resolve_branch(frame, block, instr, state.trace)
            self._advance_to_next_block(state, block, chosen=chosen)
            return True

        if instr.op_base in ("ret", "exit"):
            self._leave_block(state, block.label)
            state.halted = True
            state.halt_reason = "ret"
            state.trace.halted = True
            state.trace.halt_reason = "ret"
            return True

        if instr.op_base == "bar":
            state.waiting_barrier = self._barrier_token(block.label, state.instr_index, instr)
            return True

        try:
            self._execute_instruction(self._entry_scope, ctx, frame, instr, call_depth=0)
        except UnsupportedInstruction as exc:
            self.unsupported_ops.append(instr.op)
            state.trace.unsupported_ops.append(instr.op)
            state.halted = True
            state.halt_reason = f"instrução não suportada ({instr.op}): {exc}"
            state.trace.halted = True
            state.trace.halt_reason = state.halt_reason
            return True

        state.instr_index += 1
        return True

    def _run_cta_threads(self, cta_states: List[_ThreadExecState]) -> List[ThreadTrace]:
        while True:
            if all(state.halted for state in cta_states):
                break

            progress = False
            for state in cta_states:
                if self._step_thread_state(state):
                    progress = True

            active_states = [state for state in cta_states if not state.halted]
            waiting_states = [state for state in active_states if state.waiting_barrier is not None]
            if waiting_states:
                tokens = {state.waiting_barrier for state in waiting_states}
                if len(waiting_states) == len(active_states) and len(tokens) == 1:
                    for state in waiting_states:
                        state.waiting_barrier = None
                        state.instr_index += 1
                    progress = True
                elif len(waiting_states) == len(active_states):
                    deadlock = (
                        "deadlock em bar.sync: threads ativas do CTA chegaram a barreiras "
                        "diferentes, o que indica sincronização divergente não suportada"
                    )
                    for state in active_states:
                        state.halted = True
                        state.halt_reason = deadlock
                        state.trace.halted = True
                        state.trace.halt_reason = deadlock
                    break

            if not progress:
                stalled = [state for state in cta_states if not state.halted]
                reason = "execução estagnou sem progresso (possível sincronização não suportada)"
                for state in stalled:
                    state.halted = True
                    state.halt_reason = reason
                    state.trace.halted = True
                    state.trace.halt_reason = reason
                break

        return [state.trace for state in cta_states]

    def _resolve_branch(self,
                         frame: Frame,
                         block: BasicBlock,
                         instr: PTXInstruction,
                         trace: Optional[ThreadTrace]) -> Optional[str]:
        if instr.is_predicated:
            pred_name = instr.predicate.lstrip("@").lstrip("!")
            negate = instr.predicate.startswith("@!")
            pred_val = bool(frame.predicates.get(pred_name, False))
            taken = (not pred_val) if negate else pred_val
            conditional_target = next((t for et, t in block.exits if et == "conditional"), None)
            fallthrough_target = next((t for et, t in block.exits if et == "fallthrough"), None)
            chosen = conditional_target if taken else fallthrough_target
            if trace is not None:
                trace.branch_decisions.append(BranchDecision(
                    block_label=block.label,
                    predicate=pred_name,
                    predicate_value=pred_val,
                    taken_target=conditional_target,
                    fallthrough_target=fallthrough_target,
                    chosen_target=chosen,
                ))
        else:
            chosen = next((t for et, t in block.exits if et == "jump"), None)
        if chosen and trace is not None:
            trace.edges_taken.append((block.label, chosen))
        return chosen

    # ── dispatch de instruções ──────────────────────────────────────────

    def _execute_instruction(self,
                              scope: _KernelScope,
                              ctx: ThreadContext,
                              frame: Frame,
                              instr: PTXInstruction,
                              call_depth: int) -> None:
        op_base = instr.op_base
        reg_types = scope.reg_types
        ops = instr.operands

        if op_base == "mov":
            src = ops[1]
            try:
                value = _eval_operand(reg_types, ctx, frame, src)
            except UnsupportedInstruction:
                space = self._named_symbol_space.get(src.strip())
                if space is None:
                    raise
                address = self._named_symbol_base.get(src.strip())
                if address is None:
                    address = self._alloc_named_symbol(space)
                    self._named_symbol_base[src.strip()] = address
                _write_register(reg_types, frame, ops[0], address)
                frame.reg_space[ops[0]] = space
                return
            _write_register(reg_types, frame, ops[0], value)
            return

        if op_base == "cvt":
            parts = instr.op.split(".")
            dtype = parts[-2] if len(parts) >= 2 else parts[-1]
            stype = parts[-1]
            value = _eval_operand(reg_types, ctx, frame, ops[1])
            if dtype.startswith("f") and not stype.startswith("f"):
                value = float(value)
            elif not dtype.startswith("f") and stype.startswith("f"):
                value = int(value)
            _write_register(reg_types, frame, ops[0], value)
            return

        if op_base == "cvta":
            space = _classify_space(instr.op) or "global"
            value = _eval_operand(reg_types, ctx, frame, ops[1])
            _write_register(reg_types, frame, ops[0], value)
            frame.reg_space[ops[0]] = space
            return

        if op_base in ("add", "sub"):
            a = _eval_operand(reg_types, ctx, frame, ops[1])
            b = _eval_operand(reg_types, ctx, frame, ops[2])
            _write_register(reg_types, frame, ops[0], a + b if op_base == "add" else a - b)
            return

        if op_base == "mul":
            modifiers = set(instr.op.split(".")[1:-1])
            if "hi" in modifiers:
                raise UnsupportedInstruction(f"mul.hi ainda não é suportado: {instr.raw}")
            a = _eval_operand(reg_types, ctx, frame, ops[1])
            b = _eval_operand(reg_types, ctx, frame, ops[2])
            _write_register(reg_types, frame, ops[0], a * b)
            return

        if op_base == "mad":
            modifiers = set(instr.op.split(".")[1:-1])
            if "hi" in modifiers:
                raise UnsupportedInstruction(f"mad.hi ainda não é suportado: {instr.raw}")
            a = _eval_operand(reg_types, ctx, frame, ops[1])
            b = _eval_operand(reg_types, ctx, frame, ops[2])
            c = _eval_operand(reg_types, ctx, frame, ops[3])
            _write_register(reg_types, frame, ops[0], a * b + c)
            return

        if op_base in ("min", "max"):
            a = _eval_operand(reg_types, ctx, frame, ops[1])
            b = _eval_operand(reg_types, ctx, frame, ops[2])
            _write_register(reg_types, frame, ops[0], min(a, b) if op_base == "min" else max(a, b))
            return

        if op_base == "neg":
            a = _eval_operand(reg_types, ctx, frame, ops[1])
            _write_register(reg_types, frame, ops[0], -a)
            return

        if op_base == "abs":
            a = _eval_operand(reg_types, ctx, frame, ops[1])
            _write_register(reg_types, frame, ops[0], abs(a))
            return

        if op_base in ("and", "or", "xor"):
            a = int(_eval_operand(reg_types, ctx, frame, ops[1]))
            b = int(_eval_operand(reg_types, ctx, frame, ops[2]))
            value = {"and": a & b, "or": a | b, "xor": a ^ b}[op_base]
            _write_register(reg_types, frame, ops[0], value)
            return

        if op_base == "not":
            a = int(_eval_operand(reg_types, ctx, frame, ops[1]))
            _write_register(reg_types, frame, ops[0], ~a)
            return

        if op_base in ("shl", "shr"):
            a = int(_eval_operand(reg_types, ctx, frame, ops[1]))
            b = int(_eval_operand(reg_types, ctx, frame, ops[2]))
            _write_register(reg_types, frame, ops[0], (a << b) if op_base == "shl" else (a >> b))
            return

        if op_base == "setp":
            self._execute_setp(reg_types, ctx, frame, instr)
            return

        if op_base == "selp":
            a = _eval_operand(reg_types, ctx, frame, ops[1])
            b = _eval_operand(reg_types, ctx, frame, ops[2])
            chosen = a if _eval_predicate_operand(frame, ops[3]) else b
            _write_register(reg_types, frame, ops[0], chosen)
            return

        if op_base == "ld":
            self._execute_ld(reg_types, ctx, frame, instr)
            return

        if op_base == "st":
            self._execute_st(reg_types, ctx, frame, instr)
            return

        if op_base == "call":
            self._execute_call(ctx, frame, instr, call_depth)
            return

        if op_base in ("bar", "membar", "fence", "nop"):
            return  # marcador sem efeito de dados no modelo serial atual

        raise UnsupportedInstruction(f"opcode fora do subconjunto suportado: {instr.op}")

    def _execute_setp(self, reg_types, ctx, frame, instr: PTXInstruction) -> None:
        parts = instr.op.split(".")
        cmpop = parts[1] if len(parts) > 1 else ""
        type_token = parts[-1]
        ops = instr.operands
        if len(ops) != 3:
            raise UnsupportedInstruction(f"forma de setp não suportada (esperado 3 operandos): {instr.raw}")
        pd = ops[0]
        a = _eval_operand(reg_types, ctx, frame, ops[1])
        b = _eval_operand(reg_types, ctx, frame, ops[2])
        is_float = type_token.startswith("f")
        unsigned_cmp = cmpop in ("lo", "ls", "hi", "hs") or (not is_float and type_token[:1] in ("u", "b"))
        if not is_float and unsigned_cmp:
            width = _type_width_bytes(type_token) if _TYPE_TOKEN_RE.match(type_token) else 4
            mask = (1 << (width * 8)) - 1
            a = int(a) & mask
            b = int(b) & mask
        base_cmpop = {"lo": "lt", "ls": "le", "hi": "gt", "hs": "ge"}.get(cmpop, cmpop)
        comparisons = {
            "eq": a == b, "ne": a != b,
            "lt": a < b, "le": a <= b,
            "gt": a > b, "ge": a >= b,
        }
        if base_cmpop not in comparisons:
            raise UnsupportedInstruction(f"comparação setp não suportada ({cmpop}): {instr.raw}")
        frame.predicates[pd] = bool(comparisons[base_cmpop])

    def _eval_address(self, reg_types, ctx, frame, base: str) -> int:
        base = base.strip()
        if base.startswith("%"):
            return int(_eval_operand(reg_types, ctx, frame, base))
        raise UnsupportedInstruction(f"endereço base não suportado (símbolo global direto): {base!r}")

    def _execute_ld(self, reg_types, ctx, frame, instr: PTXInstruction) -> None:
        space = _classify_space(instr.op)
        type_token = _last_type_token(instr.op) or "u32"
        width = _type_width_bytes(type_token)
        is_float = type_token.startswith("f")
        signed = type_token.startswith("s")
        vec_width = _vector_width(instr.op)
        if len(instr.operands) < vec_width + 1:
            raise UnsupportedInstruction(f"forma vetorial de ld não suportada: {instr.raw}")
        dst_operands = _normalize_vector_operands(instr.operands[:vec_width])
        mem_operand = instr.operands[vec_width]

        if space == "param":
            if vec_width != 1:
                raise UnsupportedInstruction(f"ld.param vetorial ainda não é suportado: {instr.raw}")
            symbol = mem_operand.strip("[]").split("+")[0].strip()
            idx = self._resolve_param_symbol(symbol)
            if idx is not None:
                if idx not in frame.param_values:
                    raise UnsupportedInstruction(f"valor não fornecido para o parâmetro {idx} ({symbol})")
                _write_register(reg_types, frame, dst_operands[0], frame.param_values[idx])
                if idx in frame.pointer_param_indices:
                    frame.reg_space[dst_operands[0]] = "global"
            else:
                # `.param` local de um callseq (ex.: retval0) — não é um
                # parâmetro da assinatura do kernel/função.
                _write_register(reg_types, frame, dst_operands[0], frame.call_scratch.get(symbol, 0))
                if frame.call_scratch_space.get(symbol):
                    frame.reg_space[dst_operands[0]] = frame.call_scratch_space[symbol]
            return

        base, offset = _parse_memory_operand(mem_operand)
        addr = self._eval_address(reg_types, ctx, frame, base) + offset
        mem_space = space or frame.reg_space.get(base, "global")
        memory = self._memory_for(mem_space, ctx)
        for i, dst in enumerate(dst_operands):
            value = memory.read_value(addr + i * width, width, is_float, signed)
            _write_register(reg_types, frame, dst, value)

    def _execute_st(self, reg_types, ctx, frame, instr: PTXInstruction) -> None:
        space = _classify_space(instr.op)
        type_token = _last_type_token(instr.op) or "u32"
        width = _type_width_bytes(type_token)
        is_float = type_token.startswith("f")
        vec_width = _vector_width(instr.op)
        if len(instr.operands) < vec_width + 1:
            raise UnsupportedInstruction(f"forma vetorial de st não suportada: {instr.raw}")
        mem_operand = instr.operands[0]
        src_operands = _normalize_vector_operands(instr.operands[1:1 + vec_width])

        if space == "param":
            if vec_width != 1:
                raise UnsupportedInstruction(f"st.param vetorial ainda não é suportado: {instr.raw}")
            symbol = mem_operand.strip("[]").split("+")[0].strip()
            value = _eval_operand(reg_types, ctx, frame, src_operands[0])
            frame.call_scratch[symbol] = value
            if src_operands[0].strip() in frame.reg_space:
                frame.call_scratch_space[symbol] = frame.reg_space[src_operands[0].strip()]
            return

        base, offset = _parse_memory_operand(mem_operand)
        addr = self._eval_address(reg_types, ctx, frame, base) + offset
        mem_space = space or frame.reg_space.get(base, "global")
        memory = self._memory_for(mem_space, ctx)
        for i, src in enumerate(src_operands):
            value = _eval_operand(reg_types, ctx, frame, src)
            memory.write_value(addr + i * width, value, width, is_float)

    def _execute_call(self, ctx: ThreadContext, frame: Frame, instr: PTXInstruction, call_depth: int) -> None:
        match = _RE_CALL_UNI.match(instr.raw.strip())
        if not match:
            raise UnsupportedInstruction(f"forma de chamada não suportada: {instr.raw}")
        ret_names = [name.strip() for name in (match.group(1) or "").split(",") if name.strip()]
        callee_name = match.group(2).strip()
        arg_names = [name.strip() for name in (match.group(3) or "").split(",") if name.strip()]

        callee_scope = self._get_function_scope(callee_name)
        if callee_scope is None:
            raise UnsupportedInstruction(
                f"função {callee_name!r} chamada mas não encontrada no PTX "
                "(chamada indireta ou função externa não suportada)"
            )
        if call_depth >= self.max_call_depth:
            raise UnsupportedInstruction(f"profundidade de chamada excedida ao chamar {callee_name!r}")

        callee_param_values: Dict[int, Number] = {}
        callee_pointer_indices: Set[int] = set()
        for idx, arg_name in enumerate(arg_names):
            callee_param_values[idx] = frame.call_scratch.get(arg_name, 0)
            if frame.call_scratch_space.get(arg_name):
                callee_pointer_indices.add(idx)

        callee_frame = Frame(param_values=callee_param_values, pointer_param_indices=callee_pointer_indices)
        halt_reason = self._run_frame_serial(callee_scope, ctx, callee_frame, call_depth + 1, trace=None)
        if halt_reason != "ret":
            raise UnsupportedInstruction(f"chamada a {callee_name!r} não terminou normalmente: {halt_reason}")

        if ret_names:
            return_value = callee_frame.call_scratch.get("func_retval0")
            if return_value is None:
                # convenção alternativa mais rara: o retorno é escrito com
                # o mesmo nome usado no lado do chamador.
                return_value = callee_frame.call_scratch.get(ret_names[0], 0)
            frame.call_scratch[ret_names[0]] = return_value

    # ── execução completa do grid ───────────────────────────────────────

    def run(self, launch: KernelLaunchConfig, max_threads: int = 4096,
            snapshot_thread: int = 0) -> KernelDynamicTrace:
        buffers_before = self.buffers_snapshot()
        total_threads = launch.total_threads
        if total_threads > max_threads:
            raise ValueError(
                f"grid*block = {total_threads} thread(s) excede max_threads={max_threads}; "
                "reduza grid_dim/block_dim para depuração (isto é um limite de segurança "
                "do simulador, não do hardware real)."
            )

        gx, gy, gz = launch.grid_dim
        bx, by, bz = launch.block_dim
        threads: List[ThreadTrace] = []
        block_hits: Dict[str, int] = {}
        edge_hits: Dict[str, int] = {}
        shared_mem_by_cta: Dict[int, ByteMemory] = {}

        thread_id = 0
        for cta_z in range(gz):
            for cta_y in range(gy):
                for cta_x in range(gx):
                    cta_index = cta_x + gx * (cta_y + gy * cta_z)
                    shared_mem = shared_mem_by_cta.setdefault(cta_index, ByteMemory())
                    cta_states: List[_ThreadExecState] = []
                    for tz in range(bz):
                        for ty in range(by):
                            for tx in range(bx):
                                linear = thread_id
                                ctx = ThreadContext(
                                    thread_id=thread_id,
                                    tid=(tx, ty, tz),
                                    ctaid=(cta_x, cta_y, cta_z),
                                    ntid=(bx, by, bz),
                                    nctaid=(gx, gy, gz),
                                    global_mem=self.global_mem,
                                    shared_mem=shared_mem,
                                    local_mem=ByteMemory(),
                                    laneid=linear % launch.warp_size,
                                    warpid=linear // launch.warp_size,
                                )
                                if self._entry_has_barrier:
                                    trace = ThreadTrace(thread_id=thread_id, tid=ctx.tid, ctaid=ctx.ctaid)
                                    entry_frame = Frame(
                                        param_values=dict(self.param_values),
                                        pointer_param_indices=set(self.pointer_param_indices),
                                    )
                                    cta_states.append(_ThreadExecState(
                                        ctx=ctx,
                                        trace=trace,
                                        frame=entry_frame,
                                        current_label=self._entry_scope.order[0] if self._entry_scope.order else None,
                                        snapshot_buffers=(thread_id == snapshot_thread),
                                    ))
                                else:
                                    trace = self.run_thread(ctx, snapshot_buffers=(thread_id == snapshot_thread))
                                    threads.append(trace)
                                    for label in trace.blocks_visited:
                                        block_hits[label] = block_hits.get(label, 0) + 1
                                    for src, dst in trace.edges_taken:
                                        key = f"{src}->{dst}"
                                        edge_hits[key] = edge_hits.get(key, 0) + 1
                                thread_id += 1
                    if self._entry_has_barrier:
                        cta_traces = self._run_cta_threads(cta_states)
                        threads.extend(cta_traces)
                        for trace in cta_traces:
                            for label in trace.blocks_visited:
                                block_hits[label] = block_hits.get(label, 0) + 1
                            for src, dst in trace.edges_taken:
                                key = f"{src}->{dst}"
                                edge_hits[key] = edge_hits.get(key, 0) + 1

        return KernelDynamicTrace(
            threads=threads,
            block_hits=block_hits,
            edge_hits=edge_hits,
            buffers_before=buffers_before,
            buffers_after=self.buffers_snapshot(),
            buffer_labels=self.buffer_labels(),
            unsupported_ops=sorted(set(self.unsupported_ops)),
        )
