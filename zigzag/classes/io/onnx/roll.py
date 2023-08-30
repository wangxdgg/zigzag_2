from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import get_node_input_output_dimension_shapes
from zigzag.classes.workload.layer_node import LayerNode

import logging

logger = logging.getLogger(__name__)


class RollParser(Parser):
    """Parses an ONNX Roll operator into a LayerNode"""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def run(self):
        """Run the parser"""
        layer_node = self.generate_layer_node_for_roll()
        return layer_node

    def generate_layer_node_for_roll(self):
        def get_layer_node_input_format(B, D, H, W, node_mapping, nodes_outputs):
            """
            Generate the necessary dictionary items required for the Node creation.
            """
            # convert the data types to precisions based on the onnx definition

            # Equation
            d = {}
            d["equation"] = "O[b][d][h][w]=I[b][d][h][w]"

            # Get dimension sizes from input parameters
            B = B
            D = D
            H = H
            W = W
            d["loop_dim_size"] = {"B": B, "D": D, "H": H, "W": W}
            d["dimension_relations"] = []
            d["operand_precision"] = {"O": 16, "O_final": 8, "I": 8}
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

        assert (
            len(ia_dimension_shape) == len(oa_dimension_shape) == 4
        )  # First element is batch size, rest are dimensions
        assert (
            ia_dimension_shape[0] == oa_dimension_shape[0]
        )  # Batch size should be the same for input and output
        B = ia_dimension_shape[0]
        D = ia_dimension_shape[1]
        H = ia_dimension_shape[2]
        W = ia_dimension_shape[3]

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
            B, D, H, W, node_mapping, self.nodes_outputs
        )
        node_obj = LayerNode(
            self.node_id,
            node_attrs,
            node_name=self.node.name,
            type=self.node.op_type.lower(),
        )

        logger.info(f"Parsed Roll node {self.node.name}")

        return node_obj
