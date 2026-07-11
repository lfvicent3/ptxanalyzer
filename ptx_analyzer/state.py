"""
Estado de execução genérico para a simulação dinâmica de kernels PTX.

Este módulo define a estrutura de dados usada pelo executor genérico
(`interpreter.py`) — não conhece nada sobre nenhum algoritmo específico
(bubble/insertion/smoke/etc.). O objetivo é dar uma base robusta e
extensível para "rodar" um kernel PTX de verdade sobre um CFG, com:

  - registradores e predicados por frame de execução (kernel de entrada
    ou função device chamada por ele)
  - memória global (compartilhada entre threads, como na GPU real)
  - memória shared (por bloco CUDA)
  - memória local simplificada (por thread)
  - parâmetros do kernel/função, na ordem declarada na assinatura PTX

Duas coisas persistem durante toda a vida de uma thread simulada
(`ThreadContext`): identidade da thread (tid/ctaid/...) e as três
memórias. Tudo o resto — registradores, predicados, parâmetros — vive
num `Frame`, criado do zero a cada chamada (kernel de entrada ou função
device), exatamente como acontece de verdade no PTX: os registradores de
uma função chamada não têm nenhuma relação com os do chamador.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

Number = Union[int, float]


# ──────────────────────────────────────────────────────────────────────────────
# Memória byte-endereçável genérica
# ──────────────────────────────────────────────────────────────────────────────

_FLOAT_FORMATS = {4: "<f", 8: "<d"}


class ByteMemory:
    """Espaço de memória esparso, endereçável por byte.

    Não assume nenhum tamanho de elemento fixo: cada leitura/escrita
    recebe a largura (em bytes) e o tipo (inteiro com/sem sinal ou ponto
    flutuante) extraídos do próprio opcode PTX (`ld.global.u32`,
    `st.shared.f32`, ...), então a mesma memória serve para buffers de
    `int`, `float`, `double` etc. sem tratamento especial por caso.
    """

    def __init__(self) -> None:
        self._bytes: Dict[int, int] = {}

    def write_bytes(self, address: int, data: bytes) -> None:
        for offset, byte_value in enumerate(data):
            self._bytes[address + offset] = byte_value

    def read_bytes(self, address: int, size: int) -> bytes:
        return bytes(self._bytes.get(address + offset, 0) for offset in range(size))

    def write_value(self, address: int, value: Number, width: int, is_float: bool) -> None:
        if is_float:
            fmt = _FLOAT_FORMATS.get(width)
            if fmt is None:
                raise ValueError(f"largura de ponto flutuante não suportada: {width} bytes")
            data = struct.pack(fmt, float(value))
        else:
            mask = (1 << (width * 8)) - 1
            data = (int(value) & mask).to_bytes(width, "little", signed=False)
        self.write_bytes(address, data)

    def read_value(self, address: int, width: int, is_float: bool, signed: bool) -> Number:
        data = self.read_bytes(address, width)
        if is_float:
            fmt = _FLOAT_FORMATS.get(width)
            if fmt is None:
                raise ValueError(f"largura de ponto flutuante não suportada: {width} bytes")
            return struct.unpack(fmt, data)[0]
        return int.from_bytes(data, "little", signed=signed)

    def snapshot(self, base: int, length: int, width: int, is_float: bool, signed: bool) -> List[Number]:
        return [self.read_value(base + i * width, width, is_float, signed) for i in range(length)]


# ──────────────────────────────────────────────────────────────────────────────
# Entrada do usuário: parâmetros do kernel
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class KernelArg:
    """Um argumento concreto fornecido pelo usuário para um parâmetro do
    kernel, identificado por posição (a mesma ordem em que aparece na
    assinatura `.visible .entry NOME(...)` do PTX).

    kind="scalar": valor único (int ou float), ex.: `total_elements=8`.
    kind="buffer": vetor de valores que vira um buffer de memória global
                   (ex.: o array a ser ordenado). `element_width`/
                   `element_is_float`/`element_signed` descrevem o tipo
                   de cada elemento (default: inteiro de 4 bytes com
                   sinal, o caso mais comum nos kernels deste projeto).
    """
    index: int
    kind: str  # "scalar" | "buffer"
    value: Optional[Number] = None
    values: Optional[List[Number]] = None
    element_width: int = 4
    element_is_float: bool = False
    element_signed: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("scalar", "buffer"):
            raise ValueError(f"kind inválido para KernelArg: {self.kind!r}")
        if self.kind == "scalar" and self.value is None:
            raise ValueError(f"KernelArg escalar (index={self.index}) precisa de 'value'")
        if self.kind == "buffer" and self.values is None:
            raise ValueError(f"KernelArg buffer (index={self.index}) precisa de 'values'")


@dataclass
class KernelLaunchConfig:
    """Configuração de execução (grid/block), sempre genérica em 1D/2D/3D
    — nunca amarrada a um algoritmo específico."""
    grid_dim: Tuple[int, int, int] = (1, 1, 1)
    block_dim: Tuple[int, int, int] = (1, 1, 1)
    warp_size: int = 32

    @property
    def total_threads(self) -> int:
        gx, gy, gz = self.grid_dim
        bx, by, bz = self.block_dim
        return gx * gy * gz * bx * by * bz


# ──────────────────────────────────────────────────────────────────────────────
# Contexto (persistente) e frame (por chamada) de uma thread simulada
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ThreadContext:
    """O que persiste durante toda a vida de uma thread simulada,
    incluindo através de chamadas a funções device: identidade da thread
    e as três memórias."""
    thread_id: int
    tid: Tuple[int, int, int]
    ctaid: Tuple[int, int, int]
    ntid: Tuple[int, int, int]
    nctaid: Tuple[int, int, int]
    global_mem: ByteMemory
    shared_mem: ByteMemory
    local_mem: ByteMemory
    laneid: int = 0
    warpid: int = 0

    def special_register(self, name: str) -> Optional[Number]:
        table = {
            "%tid.x": self.tid[0], "%tid.y": self.tid[1], "%tid.z": self.tid[2],
            "%ctaid.x": self.ctaid[0], "%ctaid.y": self.ctaid[1], "%ctaid.z": self.ctaid[2],
            "%ntid.x": self.ntid[0], "%ntid.y": self.ntid[1], "%ntid.z": self.ntid[2],
            "%nctaid.x": self.nctaid[0], "%nctaid.y": self.nctaid[1], "%nctaid.z": self.nctaid[2],
            "%laneid": self.laneid, "%warpid": self.warpid,
        }
        return table.get(name)


@dataclass
class Frame:
    """Escopo de uma única chamada (o kernel de entrada, ou uma função
    device invocada por ele via `call.uni`).

    Registradores, predicados e parâmetros são sempre locais ao frame —
    nunca compartilhados entre chamador e chamado, exatamente como no
    PTX real. `call_scratch` guarda os `.param` locais de um callseq
    (`param0`, `retval0`, `func_retvalN`, ...) usados para marshalling de
    argumentos/retorno; `call_scratch_space` propaga a tag de espaço de
    endereço (para o caso raro de um ponteiro ser passado como argumento).
    """
    param_values: Dict[int, Number]
    pointer_param_indices: Set[int]
    registers: Dict[str, Number] = field(default_factory=dict)
    predicates: Dict[str, bool] = field(default_factory=dict)
    reg_space: Dict[str, str] = field(default_factory=dict)
    call_scratch: Dict[str, Number] = field(default_factory=dict)
    call_scratch_space: Dict[str, str] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# Traço dinâmico
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BranchDecision:
    block_label: str
    predicate: str
    predicate_value: bool
    taken_target: Optional[str]
    fallthrough_target: Optional[str]
    chosen_target: Optional[str]

    def to_dict(self) -> dict:
        return {
            "block_label": self.block_label,
            "predicate": self.predicate,
            "predicate_value": self.predicate_value,
            "taken_target": self.taken_target,
            "fallthrough_target": self.fallthrough_target,
            "chosen_target": self.chosen_target,
        }


@dataclass
class ThreadTrace:
    thread_id: int
    tid: Tuple[int, int, int]
    ctaid: Tuple[int, int, int]
    blocks_visited: List[str] = field(default_factory=list)
    edges_taken: List[Tuple[str, str]] = field(default_factory=list)
    branch_decisions: List[BranchDecision] = field(default_factory=list)
    halted: bool = False
    halt_reason: str = ""
    unsupported_ops: List[str] = field(default_factory=list)
    steps_executed: int = 0
    # Estado intermediário relevante: snapshot dos buffers logo após cada
    # bloco visitado. Só é preenchido para uma thread representativa (ver
    # `PTXInterpreter.run(..., snapshot_thread=...)`), pois tirar um
    # snapshot completo dos buffers a cada bloco para todas as threads
    # seria caro e desnecessário para depuração.
    block_snapshots: List[Tuple[str, Dict[int, List[Number]]]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "tid": list(self.tid),
            "ctaid": list(self.ctaid),
            "path": list(self.blocks_visited),
            "edges": [list(edge) for edge in self.edges_taken],
            "branch_decisions": [bd.to_dict() for bd in self.branch_decisions],
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "unsupported_ops": list(self.unsupported_ops),
            "steps_executed": self.steps_executed,
        }


@dataclass
class KernelDynamicTrace:
    """Resultado agregado de rodar o kernel para todas as threads
    simuladas: percurso real (não uma reprodução do algoritmo), entradas
    e saídas reais lidas da memória simulada, e o que ficou sem suporte."""
    threads: List[ThreadTrace]
    block_hits: Dict[str, int]
    edge_hits: Dict[str, int]
    buffers_before: Dict[int, List[Number]]
    buffers_after: Dict[int, List[Number]]
    buffer_labels: Dict[int, str]
    unsupported_ops: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "threads": [t.to_dict() for t in self.threads],
            "block_hits": dict(self.block_hits),
            "edge_hits": dict(self.edge_hits),
            "buffers_before": {self.buffer_labels.get(idx, f"param_{idx}"): vals
                                for idx, vals in self.buffers_before.items()},
            "buffers_after": {self.buffer_labels.get(idx, f"param_{idx}"): vals
                               for idx, vals in self.buffers_after.items()},
            "unsupported_ops": list(self.unsupported_ops),
        }


class UnsupportedInstruction(Exception):
    """Levantada quando o executor encontra uma instrução PTX (ou uma
    variante dela) fora do subconjunto suportado hoje. É tratada como uma
    limitação a ser relatada honestamente no traço, nunca silenciada nem
    mascarada com um resultado inventado — ver `interpreter.py`."""
