import onnx
from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import get_node_input_output_dimension_shapes
from zigzag.classes.workload.layer_node import LayerNode
import logging

logger = logging.getLogger(__name__)

"""
请注意，代码中的 mean_value、variance_value、scale_value 和 bias_value 是常数张量，它们的值应根据您的 LayerNormalization 操作的要求提供正确的值。
这个示例中，它们只是简单的示例值，您需要根据实际需求进行更改。此外，还需要根据您的硬件配置和需求进行适当的内存分配、空间映射和其他配置。
"""

class LayerNormParser(Parser):
    """Parses an ONNX LayerNormalization operator into a LayerNode"""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def run(self):
        """Run the parser"""
        layer_node = self.generate_layer_node_for_layernorm()
        return layer_node

    def generate_layer_node_for_layernorm(self):
        def get_layer_node_input_format(B, C, node_mapping, nodes_outputs):
            """
            Generate the necessary dictionary items required for the Node creation.
            """
            # convert the data types to precisions based on the ONNX definition

            # Equation
            d = {}
            d["equation"] = "O[b][c] = (A[b][c] - mean[c]) / sqrt(variance[c] + epsilon) * scale[c] + bias[c]"

            # Get dimension sizes from input parameters
            B = B
            C = C
            d["loop_dim_size"] = {"B": B, "C": C}
            d["dimension_relations"] = []
            d["operand_precision"] = {"O": 16, "A": 8, "mean": 8, "variance": 8, "scale": 8, "bias": 8}
            d["operand_source"] = {"A": [], "mean": [], "variance": [], "scale": [], "bias": []}
            d["constant_operands"] = ["mean", "variance", "scale", "bias"]

            d["core_allocation"] = node_mapping["core_allocation"]
            d["spatial_mapping"] = node_mapping["spatial_mapping"]
            d["temporal_ordering"] = node_mapping.get("temporal_ordering", None)
            d["memory_operand_links"] = {"O": "O", "A": "I1", "mean": "I2", "variance": "I3", "scale": "I4", "bias": "I5"}

            # Find the previous layer(s) that should be this node's parent(s)
            node_inputs = self.node.input
            preds = []
            for node_input in node_inputs:
                for n in nodes_outputs:
                    if node_input in nodes_outputs[n]:
                        preds.append(n)
            d["operand_source"] = {"A": preds}

            return d

        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(
            self.node, self.onnx_model
        )

        assert (
            len(ia_dimension_shape) == len(oa_dimension_shape) == 2
        )  # First element is batch size, second is input/output channel
        B = ia_dimension_shape[0]
        C = ia_dimension_shape[1]

        # Get the hw mapping of this node.
        if self.node.name in self.mapping:
            node_mapping = self.mapping[self.node.name]
        else:
            try:
                node_mapping = self.mapping["default"]
            except:
                raise ValueError(
                    f"There is no mapping provided for node {self.node.name}, nor a default one."
                )

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
