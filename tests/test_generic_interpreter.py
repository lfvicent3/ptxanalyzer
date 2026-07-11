"""
Testes do executor genérico de PTX/CFG (`ptx_analyzer.interpreter`).

Cada teste usa um PTX sintético auto-contido (sem depender de nvcc) que
exercita uma parte específica do subconjunto suportado: laços com
back-edge real, chamada a função device (`call.uni`), e o
comportamento honesto diante de um opcode fora do subconjunto suportado.
"""

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ptx_analyzer.parser import parse_ptx
from ptx_analyzer.interpreter import PTXInterpreter
from ptx_analyzer.state import KernelArg, KernelLaunchConfig


# Soma os elementos de `values` (um por thread, num laço `while (i < n)`)
# num único slot de saída, só para gerar um back-edge de verdade no CFG.
LOOP_SUM_PTX = r"""
.version 7.0
.target sm_75
.address_size 64

.visible .entry loop_sum_kernel(
    .param .u64 loop_sum_kernel_param_0,
    .param .u32 loop_sum_kernel_param_1
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<6>;

    ld.param.u64 %rd1, [loop_sum_kernel_param_0];
    ld.param.u32 %r1, [loop_sum_kernel_param_1];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
$L_loop:
    setp.ge.s32 %p1, %r2, %r1;
    @%p1 bra $L_end;
    mul.wide.s32 %rd3, %r2, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.u32 %r4, [%rd4];
    add.s32 %r3, %r3, %r4;
    add.s32 %r2, %r2, 1;
    bra $L_loop;
$L_end:
    cvta.to.global.u64 %rd5, %rd1;
    st.global.u32 [%rd5], %r3;
    ret;
}
"""

# Kernel de entrada que chama uma função device (`smoke_add_one`), na
# mesma convenção que o nvcc gera para funções __noinline__.
CALL_PTX = r"""
.version 7.0
.target sm_75
.address_size 64

.func  (.param .b32 func_retval0) smoke_add_one(
    .param .b32 smoke_add_one_param_0
)
{
    .reg .b32 %r<3>;

    ld.param.u32 %r1, [smoke_add_one_param_0];
    add.s32 %r2, %r1, 1;
    st.param.b32 [func_retval0+0], %r2;
    ret;
}

.visible .entry call_kernel(
    .param .u64 call_kernel_param_0
)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;

    ld.param.u64 %rd1, [call_kernel_param_0];
    cvta.to.global.u64 %rd2, %rd1;
    ld.global.u32 %r1, [%rd2];
    {
        .param .b32 param0;
        st.param.b32 [param0+0], %r1;
        .param .b32 retval0;
        call.uni (retval0), smoke_add_one, (param0);
        ld.param.b32 %r2, [retval0+0];
    }
    st.global.u32 [%rd2], %r2;
    ret;
}
"""

# `div.s32` está fora do subconjunto suportado hoje — a thread deve
# parar honestamente nesse ponto, sem travar o processo nem inventar um
# resultado.
UNSUPPORTED_OP_PTX = r"""
.version 7.0
.target sm_75
.address_size 64

.visible .entry unsupported_div_kernel(
    .param .u64 unsupported_div_kernel_param_0
)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [unsupported_div_kernel_param_0];
    cvta.to.global.u64 %rd2, %rd1;
    ld.global.u32 %r1, [%rd2];
    div.s32 %r2, %r1, 2;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""


def _units_by_role(code: str):
    units = parse_ptx(code)
    kernel = next(u for u in units if u.is_entry_point)
    functions = {u.name: u for u in units if not u.is_entry_point}
    return kernel, functions


class GenericInterpreterTest(unittest.TestCase):
    def test_loop_back_edge_is_actually_executed_the_right_number_of_times(self):
        kernel, functions = _units_by_role(LOOP_SUM_PTX)
        interp = PTXInterpreter(kernel, functions=functions, raw_ptx=LOOP_SUM_PTX)
        values = [1, 2, 3, 4, 5]
        interp.load_args([
            KernelArg(index=0, kind="buffer", values=list(values), label="data"),
            KernelArg(index=1, kind="scalar", value=len(values), label="n"),
        ])
        trace = interp.run(KernelLaunchConfig(grid_dim=(1, 1, 1), block_dim=(1, 1, 1)))

        self.assertEqual(trace.buffers_after[0][0], sum(values))
        thread = trace.threads[0]
        self.assertEqual(thread.halt_reason, "ret")
        # Uma passagem pelo cabeçalho do laço por elemento, mais a
        # passagem final que sai do laço.
        self.assertEqual(thread.blocks_visited.count("$L_loop"), len(values) + 1)
        self.assertEqual(trace.unsupported_ops, [])

    def test_call_to_device_function_executes_and_returns_value(self):
        kernel, functions = _units_by_role(CALL_PTX)
        self.assertIn("smoke_add_one", functions)
        interp = PTXInterpreter(kernel, functions=functions, raw_ptx=CALL_PTX)
        interp.load_args([KernelArg(index=0, kind="buffer", values=[41], label="data")])
        trace = interp.run(KernelLaunchConfig(grid_dim=(1, 1, 1), block_dim=(1, 1, 1)))

        self.assertEqual(trace.buffers_after[0], [42])
        self.assertEqual(trace.threads[0].halt_reason, "ret")

    def test_unsupported_opcode_halts_that_thread_without_crashing_or_faking_a_result(self):
        kernel, functions = _units_by_role(UNSUPPORTED_OP_PTX)
        interp = PTXInterpreter(kernel, functions=functions, raw_ptx=UNSUPPORTED_OP_PTX)
        interp.load_args([KernelArg(index=0, kind="buffer", values=[10], label="data")])
        trace = interp.run(KernelLaunchConfig(grid_dim=(1, 1, 1), block_dim=(1, 1, 1)))

        thread = trace.threads[0]
        self.assertIn("div.s32", thread.unsupported_ops)
        self.assertIn("não suportada", thread.halt_reason)
        # A memória não foi escrita (a instrução falhou antes do st.global),
        # então o buffer permanece com o valor original — nada foi inventado.
        self.assertEqual(trace.buffers_after[0], [10])
        self.assertIn("div.s32", trace.unsupported_ops)

    def test_multiple_threads_share_global_memory_like_real_grid(self):
        kernel, functions = _units_by_role(CALL_PTX)
        interp = PTXInterpreter(kernel, functions=functions, raw_ptx=CALL_PTX)
        interp.load_args([KernelArg(index=0, kind="buffer", values=[1], label="data")])
        # Um grid maior não muda o resultado de uma única palavra, mas
        # confirma que múltiplas threads (mesmo endereço) rodam sem erro.
        trace = interp.run(KernelLaunchConfig(grid_dim=(1, 1, 1), block_dim=(2, 1, 1)))
        self.assertEqual(len(trace.threads), 2)
        for thread in trace.threads:
            self.assertEqual(thread.halt_reason, "ret")


if __name__ == "__main__":
    unittest.main()
