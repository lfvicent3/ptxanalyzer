"""
Testes da API de alto nível de `PTXAnalyzer.dynamic_flow`: `kernel_args`
como dict simples (o caminho comum de uso), `kernel_name` casando pelo
nome "amigável" (sem mangling C++), e mensagens de erro claras quando o
usuário informa algo incompatível com a assinatura real do kernel.
"""

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ptx_analyzer import PTXAnalyzer

# Mesma assinatura de `cfg_ifelse_smoke_kernel` (extern "C", sem mangling):
# dois ponteiros (u64) e um escalar (u32).
PARAM_IF_ELSE_PTX = r"""
.version 7.0
.target sm_75
.address_size 64

.visible .entry cfg_ifelse_smoke_kernel(
    .param .u64 cfg_ifelse_smoke_kernel_param_0,
    .param .u64 cfg_ifelse_smoke_kernel_param_1,
    .param .u32 cfg_ifelse_smoke_kernel_param_2
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [cfg_ifelse_smoke_kernel_param_0];
    ld.param.u64 %rd2, [cfg_ifelse_smoke_kernel_param_1];
    ld.param.u32 %r1, [cfg_ifelse_smoke_kernel_param_2];
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    mov.u32 %r2, %tid.x;
    mul.wide.s32 %rd5, %r2, 4;
    add.s64 %rd6, %rd3, %rd5;
    ld.global.u32 %r3, [%rd6];
    setp.gt.s32 %p1, %r3, %r1;
    @%p1 bra $L_then;
    sub.s32 %r4, %r3, 1;
    bra $L_join;
$L_then:
    add.s32 %r4, %r3, 1;
$L_join:
    add.s64 %rd7, %rd4, %rd5;
    st.global.u32 [%rd7], %r4;
    ret;
}
"""

# Duas variantes com nomes mangled diferentes, para testar seleção por
# nome "amigável" (o nome do .cu, não o símbolo PTX).
TWO_KERNEL_PTX = r"""
.version 7.0
.target sm_75
.address_size 64

.visible .entry _Z11double_it_vPi(
    .param .u64 _Z11double_it_vPi_param_0
)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [_Z11double_it_vPi_param_0];
    cvta.to.global.u64 %rd2, %rd1;
    ld.global.u32 %r1, [%rd2];
    add.s32 %r2, %r1, %r1;
    st.global.u32 [%rd2], %r2;
    ret;
}

.visible .entry _Z11triple_it_vPi(
    .param .u64 _Z11triple_it_vPi_param_0
)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [_Z11triple_it_vPi_param_0];
    cvta.to.global.u64 %rd2, %rd1;
    ld.global.u32 %r1, [%rd2];
    add.s32 %r2, %r1, %r1;
    add.s32 %r3, %r2, %r1;
    st.global.u32 [%rd2], %r3;
    ret;
}
"""


class ErgonomicDynamicApiTest(unittest.TestCase):
    def test_dict_kernel_args_infer_buffer_vs_scalar_by_position(self):
        analyzer = PTXAnalyzer.from_string(PARAM_IF_ELSE_PTX)
        result = analyzer.dynamic_flow(
            kernel_args={"input": [4, 1, 9, 2], "output": [0, 0, 0, 0], "threshold": 3},
            grid_dim=(1, 1, 1),
            block_dim=(4, 1, 1),
        )
        self.assertEqual(result["dynamic_flow"]["sample_output"]["output"], [5, 0, 10, 1])

    def test_grid_block_omitted_defaults_to_single_thread_with_note(self):
        analyzer = PTXAnalyzer.from_string(PARAM_IF_ELSE_PTX)
        result = analyzer.dynamic_flow(
            kernel_args={"input": [4], "output": [0], "threshold": 3},
        )
        self.assertIn("assumindo (1, 1, 1)", result["dynamic_flow"]["notes"][0])

    def test_missing_argument_raises_clear_error(self):
        analyzer = PTXAnalyzer.from_string(PARAM_IF_ELSE_PTX)
        with self.assertRaises(ValueError) as ctx:
            analyzer.dynamic_flow(kernel_args={"input": [4], "output": [0]})
        self.assertIn("faltando argumento", str(ctx.exception))

    def test_scalar_where_buffer_expected_raises_clear_error(self):
        analyzer = PTXAnalyzer.from_string(PARAM_IF_ELSE_PTX)
        with self.assertRaises(ValueError) as ctx:
            analyzer.dynamic_flow(kernel_args={"input": 4, "output": [0], "threshold": 3})
        self.assertIn("ponteiro", str(ctx.exception))

    def test_buffer_where_scalar_expected_raises_clear_error(self):
        analyzer = PTXAnalyzer.from_string(PARAM_IF_ELSE_PTX)
        with self.assertRaises(ValueError) as ctx:
            analyzer.dynamic_flow(kernel_args={"input": [4], "output": [0], "threshold": [3]})
        self.assertIn("escalar", str(ctx.exception))

    def test_kernel_name_matches_friendly_demangled_name(self):
        analyzer = PTXAnalyzer.from_string(TWO_KERNEL_PTX)
        result = analyzer.dynamic_flow(
            kernel_name="triple_it_v",
            kernel_args={"data": [10]},
            grid_dim=(1, 1, 1), block_dim=(1, 1, 1),
        )
        self.assertEqual(result["dynamic_flow"]["sample_output"]["data"], [30])

    def test_unknown_kernel_name_lists_friendly_names_in_error(self):
        analyzer = PTXAnalyzer.from_string(TWO_KERNEL_PTX)
        with self.assertRaises(ValueError) as ctx:
            analyzer.dynamic_flow(kernel_name="does_not_exist", kernel_args={"data": [1]})
        message = str(ctx.exception)
        self.assertIn("triple_it_v", message)
        self.assertIn("double_it_v", message)


if __name__ == "__main__":
    unittest.main()
