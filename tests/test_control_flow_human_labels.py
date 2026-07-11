import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ptx_analyzer import KernelArg, PTXAnalyzer, build_cfg
from ptx_analyzer.dynamic_view import _build_payload


SIMPLE_IF_ELSE_PTX = r"""
.version 7.0
.target sm_75
.address_size 64

.visible .entry cfg_ifelse_smoke_kernel()
{
    .reg .pred %p<2>;
    .reg .b32 %r<5>;

    mov.u32 %r1, %tid.x;
    setp.gt.s32 %p1, %r1, %r2;
    @%p1 bra $L_then;
    sub.s32 %r3, %r1, 1;
    bra $L_end;
$L_then:
    add.s32 %r4, %r1, 1;
$L_end:
    ret;
}
"""

# Variante do smoke test com assinatura PTX real (dois ponteiros + um
# escalar) e ld.param/ld.global/st.global de verdade, para exercitar o
# executor dinâmico genérico (ptx_analyzer.interpreter) fim a fim, sem
# depender de nvcc/CUDA no ambiente de teste.
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

INLINE_AND_REDUNDANT_LOC_PTX = r"""
.version 7.0
.target sm_75
.address_size 64

.visible .entry redundant_loc_kernel()
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;

    .loc 1 52 13, function_name $L__info_string0, inlined_at 1 94 9
    ld.shared.u32 %r1, [%r2];
    $L_dup:
    .loc 1 52 13, function_name $L__info_string0, inlined_at 1 94 9
    add.s32 %r3, %r1, 1;
    .loc 1 53 17, function_name $L__info_string1, inlined_at 1 94 9
    setp.gt.s32 %p1, %r3, %r4;
    @%p1 bra $L_exit;
    st.shared.u32 [%r2], %r3;
$L_exit:
    ret;
}
"""

LINEARIZED_IF_ELSE_PTX = r"""
.version 7.0
.target sm_75
.address_size 64

.visible .entry linearized_if_else_kernel()
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;

    ld.global.u32 %r1, [%rd1];
    setp.gt.s32 %p1, %r1, %r2;
    selp.b32 %r3, 1, -1, %p1;
    add.s32 %r4, %r3, %r1;
    st.global.u32 [%rd2], %r4;
    ret;
}
"""

LOOP_PTX = r"""
.version 7.0
.target sm_75
.address_size 64

.visible .entry loop_kernel()
{
    .reg .pred %p<2>;
    .reg .b32 %r<5>;

    mov.u32 %r1, 0;
$L_loop:
    add.s32 %r1, %r1, 1;
    setp.lt.s32 %p1, %r1, %r2;
    @%p1 bra $L_loop;
    ret;
}
"""

NESTED_LOOP_FALSE_UNROLL_PTX = r"""
.version 7.0
.target sm_75
.address_size 64

.visible .entry nested_loop_kernel()
{
    .reg .pred %p<3>;
    .reg .b32 %r<8>;

    mov.u32 %r1, 0;
$L_outer:
    .loc 1 89 5
    setp.lt.s32 %p1, %r1, %r2;
    @!%p1 bra $L_end;
    mov.u32 %r3, 0;
$L_inner:
    .loc 1 89 9
    setp.lt.s32 %p2, %r3, %r4;
    @!%p2 bra $L_after_inner;
    add.s32 %r3, %r3, 1;
    bra $L_inner;
$L_after_inner:
    add.s32 %r1, %r1, 1;
    bra $L_outer;
$L_end:
    ret;
}
"""

SEQUENTIAL_CONDITIONS_PTX = r"""
.version 7.0
.target sm_75
.address_size 64

.visible .entry sequential_conditions_kernel()
{
    .reg .pred %p<3>;
    .reg .b32 %r<6>;

    setp.gt.s32 %p1, %r1, %r2;
    @%p1 bra $L_skip;
    setp.lt.s32 %p2, %r3, %r4;
    @%p2 bra $L_then;
$L_skip:
    sub.s32 %r5, %r1, 1;
    bra $L_end;
$L_then:
    add.s32 %r5, %r1, 1;
$L_end:
    ret;
}
"""

SEQUENTIAL_CONDITIONS_SAME_LINE_PTX = r"""
.version 7.0
.target sm_75
.address_size 64

.visible .entry sequential_conditions_same_line_kernel()
{
    .reg .pred %p<3>;
    .reg .b32 %r<6>;

    .loc 1 47 5
    setp.gt.s32 %p1, %r1, %r2;
    @%p1 bra $L_skip;
    .loc 1 47 5
    setp.lt.s32 %p2, %r3, %r4;
    @%p2 bra $L_then;
$L_skip:
    sub.s32 %r5, %r1, 1;
    bra $L_end;
$L_then:
    add.s32 %r5, %r1, 1;
$L_end:
    ret;
}
"""


class ControlFlowHumanLabelsTest(unittest.TestCase):
    def test_control_flow_data_contains_human_labels(self):
        analyzer = PTXAnalyzer.from_string(SIMPLE_IF_ELSE_PTX)
        data = analyzer.control_flow(mode="data")
        blocks = data["control_flow"]["blocks"]
        order = data["control_flow"]["order"]
        mermaid = data["mermaid"]

        self.assertIn("__ENTRY__", blocks)
        self.assertEqual(blocks["__ENTRY__"]["display_name"], "Entrada")
        self.assertIn("Comparando valores", blocks["__ENTRY__"]["description"])

        synthetic_blocks = [blocks[label] for label in order if label.startswith("__seq_")]
        block_values = list(blocks.values())
        self.assertTrue(any(block["display_name"].endswith("1") for block in synthetic_blocks))
        self.assertTrue(any("subtração" in block["description"].lower() for block in synthetic_blocks))
        self.assertTrue(any("soma" in block["description"].lower() for block in block_values))

        self.assertIn("Comparando valores", mermaid)
        self.assertIn("PTX: setp.gt.s32", mermaid)
        self.assertIn("Efetuando soma", mermaid)
        self.assertIn("Efetuando subtração", mermaid)
        self.assertEqual(data["control_flow"]["branch_sites"][0]["reconvergence_target"], "$L_end")
        self.assertTrue(data["control_flow"]["branch_sites"][0]["idle_threads_possible"])

    def test_parser_tracks_inlined_at_and_merges_redundant_same_line_fallthrough(self):
        analyzer = PTXAnalyzer.from_string(INLINE_AND_REDUNDANT_LOC_PTX)
        first_instr = analyzer.kernel.instructions[0]
        self.assertEqual(first_instr.source_line, 52)
        self.assertEqual(first_instr.inline_source_line, 94)

        blocks, order = build_cfg(analyzer.kernel)
        self.assertEqual(order[0], "__ENTRY__")
        self.assertEqual(len(order), 3)
        self.assertEqual(len(blocks["__ENTRY__"].instructions), 4)
        self.assertTrue(any(blocks[label].repeated_source_instance > 0 for label in order[1:]))

    def test_linear_execution_graph_when_kernel_has_no_bra(self):
        analyzer = PTXAnalyzer.from_string(LINEARIZED_IF_ELSE_PTX)
        mermaid = analyzer.control_flow(mode="raw")
        data = analyzer.control_flow(mode="data")

        self.assertIn("Decisão", mermaid)
        self.assertIn("Escolha", mermaid)
        self.assertIn("Atualização", mermaid)
        self.assertIn("Resultado", mermaid)
        self.assertIn("Escolhe entre incrementar ou decrementar", mermaid)
        self.assertEqual(data["visual_flow"]["kind"], "linear")
        self.assertTrue(data["visual_flow"]["metadata"]["has_predicated_selection"])
        self.assertIn("não gerou dois caminhos de controle", data["visual_flow"]["metadata"]["note"])

    def test_loop_back_edge_is_exposed_as_loop_site(self):
        analyzer = PTXAnalyzer.from_string(LOOP_PTX)
        data = analyzer.control_flow(mode="data")
        loop_sites = data["control_flow"]["loop_sites"]

        self.assertEqual(len(loop_sites), 1)
        self.assertEqual(loop_sites[0]["header"], "$L_loop")
        self.assertEqual(loop_sites[0]["latch"], "$L_loop")

    def test_compound_decision_group_is_rendered_for_same_source_line_conditions(self):
        analyzer = PTXAnalyzer.from_string(SEQUENTIAL_CONDITIONS_SAME_LINE_PTX)
        mermaid = analyzer.control_flow(mode="raw")
        data = analyzer.control_flow(mode="data")

        self.assertIn("Decisão Composta 1", mermaid)
        self.assertIn("PTX: setp.gt.s32", mermaid)
        first_block = data["control_flow"]["blocks"]["__ENTRY__"]
        second_block = data["control_flow"]["blocks"]["__seq_1__"]
        self.assertEqual(first_block["instruction_name"], "comparação condicional")
        self.assertEqual(second_block["instruction_name"], "comparação condicional")
        self.assertEqual(first_block["raw_instruction_name"], "setp.gt.s32 + bra")
        self.assertEqual(second_block["raw_instruction_name"], "setp.lt.s32 + bra")

        payload = _build_payload(data, warp_size=32, meta={})
        group_labels = [group["label"] for group in payload["graph_groups"]]
        self.assertIn("Decisão Composta 1", group_labels)
        node_entry = next(node for node in payload["nodes"] if node["id"] == "__ENTRY__")
        self.assertEqual(node_entry["label"], "Entrada")
        self.assertEqual(node_entry["ptx_name"], "setp.gt.s32")

    def test_compound_decision_group_is_not_rendered_for_distinct_conditions(self):
        analyzer = PTXAnalyzer.from_string(SEQUENTIAL_CONDITIONS_PTX)
        mermaid = analyzer.control_flow(mode="raw")

        self.assertNotIn("Decisão Composta", mermaid)

    def test_nested_loops_are_not_mistaken_for_unroll_groups_in_payload(self):
        analyzer = PTXAnalyzer.from_string(NESTED_LOOP_FALSE_UNROLL_PTX)
        data = analyzer.control_flow(mode="data")
        payload = _build_payload(data, warp_size=32, meta={})

        unroll_labels = [group["label"] for group in payload["graph_groups"] if group["kind"] == "unroll"]
        self.assertEqual(unroll_labels, [])

    def test_dynamic_smoke_trace_counts_taken_and_fallthrough_threads(self):
        # Executa o PTX de verdade (executor genérico) em vez de reconhecer
        # o kernel pelo nome e reproduzir a lógica em Python.
        analyzer = PTXAnalyzer.from_string(PARAM_IF_ELSE_PTX)
        kernel_args = [
            KernelArg(index=0, kind="buffer", values=[4, 1, 9, 2], label="input"),
            KernelArg(index=1, kind="buffer", values=[0, 0, 0, 0], label="output"),
            KernelArg(index=2, kind="scalar", value=3, label="threshold"),
        ]
        data = analyzer.dynamic_flow(
            kernel_args=kernel_args,
            grid_dim=(1, 1, 1),
            block_dim=(4, 1, 1),
            expected_output={"output": [5, 0, 10, 1]},
            mode="data",
        )
        dynamic = data["dynamic_flow"]

        self.assertEqual(dynamic["sample_output"]["output"], [5, 0, 10, 1])
        self.assertEqual(dynamic["branch_activity"][0]["taken_count"], 2)
        self.assertEqual(dynamic["branch_activity"][0]["fallthrough_count"], 2)
        self.assertIn("__ENTRY__", dynamic["block_hits"])
        self.assertEqual(dynamic["unsupported_ops"], [])
        self.assertTrue(all(item["match"] for item in dynamic["validation"]))

    def test_dynamic_trace_reports_validation_mismatch_honestly(self):
        analyzer = PTXAnalyzer.from_string(PARAM_IF_ELSE_PTX)
        kernel_args = [
            KernelArg(index=0, kind="buffer", values=[4, 1, 9, 2], label="input"),
            KernelArg(index=1, kind="buffer", values=[0, 0, 0, 0], label="output"),
            KernelArg(index=2, kind="scalar", value=3, label="threshold"),
        ]
        data = analyzer.dynamic_flow(
            kernel_args=kernel_args,
            grid_dim=(1, 1, 1),
            block_dim=(4, 1, 1),
            expected_output={"output": [0, 0, 0, 0]},
            mode="data",
        )
        dynamic = data["dynamic_flow"]
        self.assertFalse(dynamic["validation"][0]["match"])
        self.assertTrue(any("Divergência" in note for note in dynamic["notes"]))


if __name__ == "__main__":
    unittest.main()
