import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ptx_analyzer import PTXAnalyzer, build_cfg


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

    def test_compound_decision_group_is_not_rendered_for_distinct_conditions(self):
        analyzer = PTXAnalyzer.from_string(SEQUENTIAL_CONDITIONS_PTX)
        mermaid = analyzer.control_flow(mode="raw")

        self.assertNotIn("Decisão Composta", mermaid)

    def test_dynamic_smoke_trace_counts_taken_and_fallthrough_threads(self):
        analyzer = PTXAnalyzer.from_string(SIMPLE_IF_ELSE_PTX)
        data = analyzer.dynamic_flow(sample_input=[4, 1, 9, 2], threshold=3, mode="data")
        dynamic = data["dynamic_flow"]

        self.assertEqual(dynamic["sample_output"], [5, 0, 10, 1])
        self.assertEqual(dynamic["branch_activity"][0]["taken_count"], 2)
        self.assertEqual(dynamic["branch_activity"][0]["fallthrough_count"], 2)
        self.assertIn("__ENTRY__", dynamic["block_hits"])


if __name__ == "__main__":
    unittest.main()
