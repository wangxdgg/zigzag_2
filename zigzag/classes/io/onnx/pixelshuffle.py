import onnx
from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import get_node_input_output_dimension_shapes
from zigzag.classes.workload.layer_node import LayerNode
import logging

logger = logging.getLogger(__name__)


class PixelShuffleParser(Parser):
    """Simplified parser for ONNX PixelShuffle node"""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def run(self):
        """Run the parser and return the created LayerNode object."""
        layer_node = self.generate_layer_node_for_pixelshuffle()
        return layer_node

    def generate_layer_node_for_pixelshuffle(self):
        def get_layer_node_input_format(B, C, OY, OX, upscale_factor, node_mapping, nodes_outputs):
            """
            Generate the necessary dictionary items required for the LayerNode creation.
            """
            # Equation
            d = {}
            d["equation"] = "O[b][c][oy][ox] = I[b][c][y][x]"

            # Calculate the new dimension sizes after pixel shuffle
            IY = OY * upscale_factor
            IX = OX * upscale_factor

            # Get dimension sizes from input parameters
            B = B
            C = C
            IY = IY
            IX = IX
            OY = OY
            OX = OX
            d["loop_dim_size"] = {"B": B, "C": C, "IY": IY, "IX": IX, "OY": OY, "OX": OX}
            d["dimension_relations"] = ["y = oy / upscale_factor", "x = ox / upscale_factor"]
            d["operand_precision"] = {"O": 16, "I": 8}
            d["operand_source"] = {"I": []}
            d["constant_operands"] = []

            d["core_allocation"] = node_mapping["core_allocation"]
            d["spatial_mapping"] = node_mapping["spatial_mapping"]
            d["temporal_ordering"] = node_mapping.get("temporal_ordering", None)
            d["memory_operand_links"] = {"O": "O", "I": "I1"}

            # Find the previous layer(s) that should be this node's parent(s)
            node_inputs = self.node.input
            preds = []
            for node_input in node_inputs:
                for n in nodes_outputs:
                    if node_input in nodes_outputs[n]:
                        preds.append(n)
            d["operand_source"] = {"I": preds}

            return d

        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(
            self.node, self.onnx_model
        )

        assert len(ia_dimension_shape) == len(oa_dimension_shape) == 4, "Input and output dimensions must match in 4D"

        B = ia_dimension_shape[0]
        C = ia_dimension_shape[1]
        OY = oa_dimension_shape[2]  # 使用OY表示高度
        OX = oa_dimension_shape[3]  # 使用OX表示宽度

        # 您提供的放大因子，例如 upscale_factor = 2
        upscale_factor = 2

        # Get the hw mapping of this node (可以根据需要自定义)
        node_mapping = {
            "core_allocation": {"core_id": 0},
            "spatial_mapping": {"x": 0, "y": 0},
        }

        node_attrs = get_layer_node_input_format(
            B, C, OY, OX, upscale_factor, node_mapping, self.nodes_outputs
        )

        node_obj = LayerNode(
            self.node_id,
            node_attrs,
            node_name=self.node.name,
            type=self.node.op_type.lower(),
        )

        logger.info(f"Parsed PixelShuffle node {self.node.name}")

        return node_obj
