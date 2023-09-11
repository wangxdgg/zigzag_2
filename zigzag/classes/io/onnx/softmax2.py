from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import get_node_input_output_dimension_shapes
from zigzag.classes.workload.layer_node import LayerNode

import logging

logger = logging.getLogger(__name__)

class SoftmaxParser(Parser):
    """Parser for ONNX Softmax nodes into LayerNode."""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model):
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def run(self):
        """Run the parser and return the created LayerNode object."""
        layer_node = self.generate_layer_node_for_softmax()
        return layer_node

    def generate_layer_node_for_softmax(self):
        def get_layer_node_input_format(B, C, K, node_mapping, nodes_outputs):
            """
            Generate the necessary dictionary items required for the Node creation.
            """
            # Convert the data types to precisions based on the ONNX definition

            # Equation
            d = {}
            # Update the equation for Softmax
            d["equation"] = "O[b][c] = exp(I[b][c]) / (reduce_sum(exp(I[b]), axis=1))"
            d["dimension_relations"] = []
            d["operand_precision"] = {"O": 16, "I": 8}  # Modify precision as needed
            d["operand_source"] = {"I": []}

            # Core allocation and spatial mapping
            d["core_allocation"] = node_mapping["core_allocation"]
            d["spatial_mapping"] = node_mapping["spatial_mapping"]

            # Find the previous layer(s) that should be this node's parent(s)
            node_inputs = self.node.input
            preds = []
            for node_input in node_inputs:
                for n in nodes_outputs:
                    if node_input in nodes_outputs[n]:
                        preds.append(n)
            d["operand_source"]["I"] = preds

            return d

        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(
            self.node, self.onnx_model
        )

        # Get the batch size, input channels, and output channels
        B = ia_dimension_shape[0] if ia_dimension_shape else 1
        C = ia_dimension_shape[1] if ia_dimension_shape else 0
        K = oa_dimension_shape[1] if oa_dimension_shape else 0

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

        node_attrs = get_layer_node_input_format(
            B, C, K, node_mapping, self.nodes_outputs
        )
        node_obj = LayerNode(
            self.node_id,
            node_attrs,
            node_name=self.node.name,
            type=self.node.op_type.lower(),
        )

        logger.info(f"Parsed Softmax node {self.node.name}")

        return node_obj
