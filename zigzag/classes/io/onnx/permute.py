import onnx
from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import get_node_input_output_dimension_shapes
from zigzag.classes.workload.layer_node import LayerNode
import logging

"""
假设输入和输出张量的形状在维度上匹配，且没有考虑其他可能的参数（如输入数据类型）
"""

logger = logging.getLogger(__name__)

class PermuteParser(Parser):
    """Simplified parser for ONNX Permute node"""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def run(self):
        """Run the parser and return the created LayerNode object."""
        layer_node = self.generate_layer_node_for_permute()
        return layer_node

    def generate_layer_node_for_permute(self):
        def get_layer_node_input_format(B, C, H, W, permute_order, node_mapping, nodes_outputs):
            """
            Generate the necessary dictionary items required for the LayerNode creation.
            """
            # Equation
            d = {}
            d["equation"] = "O[b][c][h][w] = I[b][permute_order[0]][permute_order[1]][permute_order[2]]"

            # Get dimension sizes from input parameters
            B = B
            C = C
            H = H
            W = W
            permute_order = permute_order  # New dimension permutation order
            d["loop_dim_size"] = {"B": B, "C": C, "H": H, "W": W}
            d["dimension_relations"] = []
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

        assert len(ia_dimension_shape) == len(oa_dimension_shape), "Input and output dimensions must match"

        B = ia_dimension_shape[0]
        C = ia_dimension_shape[1]
        H = ia_dimension_shape[2]
        W = ia_dimension_shape[3]

        # 您提供的维度排列顺序，例如 [0, 2, 3, 1]
        permute_order = [0, 2, 3, 1]

        # Get the hw mapping of this node (可以根据需要自定义)
        node_mapping = {
            "core_allocation": {"core_id": 0},
            "spatial_mapping": {"x": 0, "y": 0},
        }

        node_attrs = get_layer_node_input_format(
            B, C, H, W, permute_order, node_mapping, self.nodes_outputs
        )

        node_obj = LayerNode(
            self.node_id,
            node_attrs,
            node_name=self.node.name,
            type=self.node.op_type.lower(),
        )

        logger.info(f"Parsed Permute node {self.node.name}")

        return node_obj
