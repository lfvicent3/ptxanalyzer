"""
Heurísticas de diagnóstico estático para kernels PTX.
"""

from typing import List, Tuple
from .core import PTXKernel

# ──────────────────────────────────────────────────────────────────────────────
# 4. Heurísticas de diagnóstico
# ──────────────────────────────────────────────────────────────────────────────

# Cada heurística retorna (nível, mensagem) onde nível ∈ {"ok","info","warn","error"}
# ou None se não aplicável.

def _check_warp_divergence(k: PTXKernel):
    n = k.predicated_branches
    if n == 0:
        return ("ok", "Sem branches predicados — divergência de warp improvável.")
    if n > 20:
        return ("warn", f"{n} branches predicados — alta probabilidade de "
                        "divergência de warp, impacto severo em desempenho.")
    return ("info", f"{n} branches predicados — divergência moderada.")


def _check_register_pressure(k: PTXKernel):
    n = k.total_registers
    if n > 128:
        return ("warn", f"{n} registradores declarados — risco de register spilling "
                        "(limite típico: 255; threads por SM cai acima de ~64).")
    if n > 64:
        return ("info", f"{n} registradores — ocupância pode ser limitada "
                        "dependendo do número de threads por bloco.")
    return ("ok", f"{n} registradores — pressão de registradores normal.")


def _check_shared_memory(k: PTXKernel):
    sa = k.shared_accesses
    gl = k.global_loads
    if sa == 0 and gl > 10:
        return ("warn", "Nenhum acesso a shared memory com muitos ld.global — "
                        "considere usar shared memory para reduzir latência.")
    if sa > 0:
        ratio = round(sa / max(gl, 1), 2)
        return ("ok", f"Shared memory usada: {sa} acessos "
                      f"({ratio:.0%} do total de loads).")
    return None


def _check_memory_bound(k: PTXKernel):
    ai = k.arithmetic_intensity
    if ai < 1.0:
        return ("warn", f"Intensidade aritmética = {ai} — kernel provavelmente "
                        "memory-bound (eixo X do Roofline abaixo do ridge point).")
    if ai < 2.0:
        return ("info", f"Intensidade aritmética = {ai} — limítrofe entre "
                        "memory-bound e compute-bound.")
    return ("ok", f"Intensidade aritmética = {ai} — kernel provavelmente compute-bound.")


def _check_atomics(k: PTXKernel):
    n = k.atomics
    if n == 0:
        return None
    if n > 10:
        return ("warn", f"{n} operações atômicas — pode causar serialização "
                        "em acessos concorrentes à mesma posição.")
    return ("info", f"{n} operações atômicas detectadas.")


def _check_fma_ratio(k: PTXKernel):
    fma = k.fma_count
    mul = sum(1 for i in k.instructions if i.op_base == "mul")
    add = sum(1 for i in k.instructions if i.op_base == "add")
    if mul + add > 10 and fma == 0:
        return ("warn", f"{mul} mul + {add} add separados sem FMA — "
                        "compilador pode não estar fundindo operações "
                        "(verifique --use_fast_math).")
    if fma > 0:
        return ("ok", f"{fma} instruções FMA — fusão de mul+add ativa.")
    return None


def _check_grid_stride(k: PTXKernel):
    if k.uses_nctaid and k.predicated_branches >= 1:
        return ("ok", "Padrão grid-stride loop detectado "
                      "(%nctaid + branch de loop predicado).")
    if k.uses_nctaid:
        return ("info", "%nctaid presente — grid-stride parcial ou acesso a gridDim.")
    return ("info", "Padrão grid-stride não detectado — "
                    "kernel pode não suportar N > gridDim × blockDim.")


def _check_vector_loads(k: PTXKernel):
    v4 = sum(1 for i in k.instructions if "v4" in i.op)
    v2 = sum(1 for i in k.instructions if "v2" in i.op)
    if v4 + v2 == 0 and k.global_loads > 10:
        return ("info", "Sem loads vetorizados (v2/v4) — considere "
                        "ld.global.v4.f32 para melhor throughput de memória.")
    if v4 > 0:
        return ("ok", f"{v4} loads vetorizados v4 — bom uso da largura de banda.")
    return None


def _check_warp_ops(k: PTXKernel):
    n = k.shfl_count
    if n > 0:
        return ("ok", f"{n} instrução(ões) shfl.sync — reduções de warp "
                      "sem shared memory (eficiente).")
    return None


def _check_local_memory(k: PTXKernel):
    """
    .local = memória de pilha do thread — tão lento quanto global, sem coalescência.
    Ocorre quando o compilador não consegue manter um array indexado em registradores
    (ex: insert sort com vetor local indexado por variável).
    """
    n = k.local_accesses
    if n == 0:
        return None
    return ("warn",
            f"{n} acesso(s) a .local — array indexado spilled para memória local "
            "(tão lento quanto global; prefira acesso por registradores fixos).")


def _check_unroll(k: PTXKernel):
    """
    Quando o compilador consegue desenrolar um loop de ordenação com vetor pequeno,
    todas as trocas ficam em registradores: zero loads/stores, zero branches.
    Isso é o comportamento ideal detectado pelo professor para bubble sort local.
    """
    if k.is_register_only:
        return ("ok",
                "Kernel 100% em registradores — loop completamente desenrolado "
                "(unroll total). Zero loads/stores: comportamento ótimo para "
                "vetores pequenos (ex: bubble sort local com N fixo).")
    if k.total_mem_accesses > 0 and k.local_accesses == 0 and k.global_loads == 0:
        return ("info",
                "Sem loads globais/locais — acesso exclusivamente via shared memory "
                "ou operações em registradores.")
    return None


def _check_branch_ratio(k: PTXKernel):
    """Alta fração de branches → divergência severa no modelo SIMT."""
    r = k.branch_ratio
    if r == 0:
        return None
    if r > 0.15:
        return ("warn",
                f"{r:.1%} das instruções são branches predicados — "
                "divergência severa esperada (Quick Sort / Bubble Sort global).")
    if r > 0.05:
        return ("info",
                f"{r:.1%} das instruções são branches predicados — "
                "divergência moderada.")
    return ("ok", f"{r:.1%} de branches predicados — divergência baixa.")


def _check_min_max(k: PTXKernel):
    n = k.min_max_count
    if n > 0:
        return ("info",
                f"{n} instrução(ões) min/max — compilador pode ter substituído "
                "branch+swap por instrução escalar (bom sinal em selection sort).")
    return None


HEURISTICS = [
    _check_unroll,           # primeiro: detectar o caso ótimo logo de cara
    _check_warp_divergence,
    _check_branch_ratio,
    _check_register_pressure,
    _check_local_memory,
    _check_shared_memory,
    _check_memory_bound,
    _check_atomics,
    _check_fma_ratio,
    _check_min_max,
    _check_grid_stride,
    _check_vector_loads,
    _check_warp_ops,
]

LEVEL_ICONS = {
    "ok":    "✅",
    "info":  "ℹ️",
    "warn":  "⚠️",
    "error": "🚨",
}
LEVEL_COLORS = {
    "ok":    "#065f46",
    "info":  "#1e3a5f",
    "warn":  "#78350f",
    "error": "#7f1d1d",
}
LEVEL_TEXT_COLORS = {
    "ok":    "#6ee7b7",
    "info":  "#93c5fd",
    "warn":  "#fcd34d",
    "error": "#fca5a5",
}


def run_heuristics(k: PTXKernel) -> List[Tuple[str, str]]:
    """Executa todas as heurísticas e retorna lista de (nível, mensagem)."""
    results = []
    for fn in HEURISTICS:
        r = fn(k)
        if r is not None:
            results.append(r)
    return results
