import onnx
from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import get_node_input_output_dimension_shapes
from zigzag.classes.workload.layer_node import LayerNode
import logging

logger = logging.getLogger(__name__)

"""
请注意，这是一个简化的示例，假设输入和输出特征图的维度为 [B, C, D]，其中 B 代表批处理大小，C 代表通道数，D 代表特征图维度。
您需要根据您的实际需求和硬件配置进行进一步定制，特别是关于硬件映射和常数操作数值。
"""

class LayerNormParser(Parser):
    """Simplified parser for ONNX LayerNormalization (LayerNorm) node"""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def run(self):
        """Run the parser and return the created LayerNode object."""
        layer_node = self.generate_layer_node_for_layernorm()
        return layer_node

    def generate_layer_node_for_layernorm(self):
        def get_layer_node_input_format(B, C, node_mapping, nodes_outputs):
            """
            Generate the necessary dictionary items required for the LayerNode creation.
            """
            # Equation
            d = {}
            d["equation"] = "O[b][c] = (I[b][c] - mean[c]) / sqrt(variance[c] + epsilon) * scale[c] + bias[c]"

            # Get dimension sizes from input parameters
            B = B
            C = C
            d["loop_dim_size"] = {"B": B, "C": C}
            d["dimension_relations"] = []
            d["operand_precision"] = {"O": 16, "I": 8, "mean": 8, "variance": 8, "scale": 8, "bias": 8}
            d["operand_source"] = {"I": [], "mean": [], "variance": [], "scale": [], "bias": []}
            d["constant_operands"] = ["mean", "variance", "scale", "bias"]

            d["core_allocation"] = node_mapping["core_allocation"]
            d["spatial_mapping"] = node_mapping["spatial_mapping"]
            d["temporal_ordering"] = node_mapping.get("temporal_ordering", None)
            d["memory_operand_links"] = {"O": "O", "I": "I1", "mean": "I2", "variance": "I3", "scale": "I4", "bias": "I5"}

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

        assert (
            len(ia_dimension_shape) == len(oa_dimension_shape) == 3
        )  # Assumes [B, C, D] format for input and output

        B = ia_dimension_shape[0]
        C = ia_dimension_shape[1]

        # Get the hw mapping of this node (you can customize this)
        node_mapping = {
            "core_allocation": {"core_id": 0},
            "spatial_mapping": {"x": 0, "y": 0},
        }

        # Define the constant operands (mean, variance, scale, bias)
        # You should provide the correct values for these constants
        mean_value = [0.0] * C  # Example: All zeros
        variance_value = [1.0] * C  # Example: All ones
        scale_value = [1.0] * C  # Example: All ones
        bias_value = [0.0] * C  # Example: All zeros

        node_attrs = get_layer_node_input_format(
            B, C, node_mapping, self.nodes_outputs
        )
        node_attrs["constant_operands_values"] = {"mean": mean_value, "variance": variance_value, "scale": scale_value, "bias": bias_value}
        node_obj = LayerNode(
            self.node_id,
            node_attrs,
            node_name=self.node.name,
            type=self.node.op_type.lower(),
        )

        logger.info(f"Parsed LayerNormalization node {self.node.name}")

        return node_obj
