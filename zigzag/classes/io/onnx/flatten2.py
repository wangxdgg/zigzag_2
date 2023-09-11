from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import get_node_input_output_dimension_shapes
from zigzag.classes.workload.layer_node import LayerNode

import logging

logger = logging.getLogger(__name__)


class FlattenParser(Parser):
    """Parser for ONNX Flatten nodes into LayerNode."""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def run(self):
        """Run the parser and return the created LayerNode object."""
        layer_node = self.generate_layer_node_for_flatten()
        return layer_node

    def generate_layer_node_for_flatten(self):
        def get_layer_node_input_format(ia_shape, oa_shape, node_mapping):
            """
            Generate the necessary dictionary items required for the LayerNode creation.
            """
            # Equation
            d = {}
            d["equation"] = "O[b][c][oy][ox] = I[b][c][ix][iy]"

            # Get dimension sizes from input parameters
            B = ia_shape[0]
            C = ia_shape[1]
            IX = ia_shape[2]
            IY = ia_shape[3]
            OX = oa_shape[2]
            OY = oa_shape[3]

            d["loop_dim_size"] = {"B": B, "C": C, "IX": IX, "IY": IY, "OX": OX, "OY": OY}
            d["pr_loop_dim_size"] = {}

            d["operand_precision"] = {"O": 16, "O_final": 8, "I": 8}
            d["constant_operands"] = []

            d["core_allocation"] = node_mapping["core_allocation"]
            d["spatial_mapping"] = node_mapping["spatial_mapping"]
            d["temporal_ordering"] = node_mapping.get("temporal_ordering", None)
            d["memory_operand_links"] = node_mapping["memory_operand_links"]

            # Find the previous layer(s) that should be this node's parent(s)
            node_inputs = self.node.input
            preds = []
            for node_input in node_inputs:
                for n in self.nodes_outputs:
                    if node_input in self.nodes_outputs[n]:
                        preds.append(n)
            d["operand_source"] = {"I": preds}

            return d

        # Get the input and output activation shapes
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(
            self.node, self.onnx_model
        )

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

        # Take a deepcopy of the mapping, otherwise it will be changed for other layers if using default
        node_mapping = pickle_deepcopy(node_mapping)

        node_attrs = get_layer_node_input_format(
            ia_dimension_shape, oa_dimension_shape, node_mapping
        )

        node_obj = LayerNode(
            self.node_id,
            node_attrs,
            node_name=self.node.name,
            type=self.node.op_type.lower(),
        )

        logger.info(f"Parsed Flatten node {self.node.name}")

        return node_obj
