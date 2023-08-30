import pickle

from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import (
    get_attribute_ints_with_name,
    get_node_input_output_dimension_shapes, get_onnx_tensor_type,
)
from zigzag.classes.workload.layer_node import LayerNode
from zigzag.utils import pickle_deepcopy

import logging

logger = logging.getLogger(__name__)


class AddParser(Parser):
    """Parser for ONNX Add nodes into LayerNode."""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def run(self):
        """Run the parser and return the created LayerNode object."""
        layer_node = self.generate_layer_node_for_add()
        return layer_node

    def generate_layer_node_for_add(self):
        def get_input_output_data_type(node, model):
            """
            Return the data type of the input and output tensors of this node.
            """
            input_name = node.input[0]
            output_name = node.output[0]

            input_elem_type = get_onnx_tensor_type(input_name, model).elem_type
            output_elem_type = get_onnx_tensor_type(output_name, model).elem_type

            return input_elem_type, output_elem_type

        def get_layer_node_input_format(input_shape, output_shape, node_mapping):
            """
            Generate the necessary dictionary items required for the LayerNode creation.
            """
            # Generate the necessary dictionary items required for the LayerNode creation
            d = {}

            # Define the equation for the add operation
            d["equation"] = "O[b][c][y][x]=I1[b][c][y][x]+I2[b][c][y][x]"

            # Get dimension sizes from input parameters
            B = input_shape[0]
            C = input_shape[1]
            Y = input_shape[2]
            X = input_shape[3]

            # Set the loop dimension sizes
            d["loop_dim_size"] = {
                "B": B,
                "C": C,
                "Y": Y,
                "X": X
            }

            d["operand_precision"] = {"O": 8, "I1": 8, "I2": 8}

            d["core_allocation"] = node_mapping["core_allocation"]
            d["spatial_mapping"] = node_mapping["spatial_mapping"]
            d["temporal_ordering"] = node_mapping.get("temporal_ordering", None)
            d["memory_operand_links"] = node_mapping["memory_operand_links"]

            # Find the previous layers that should be this node's parents
            node_inputs = self.node.input
            preds = []
            for node_input in node_inputs:
                for n in self.nodes_outputs:
                    if node_input in self.nodes_outputs[n]:
                        preds.append(n)
            d["operand_source"] = {"I1": preds, "I2": preds}

            return d

        # Get the input and output activation shapes
        input_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)[0]
        output_shape = get_node_input_output_dimension_shapes(self.node, self.onnx_model)[1]

        # Get the input and output activation data types
        input_data_type, output_data_type = get_input_output_data_type(self.node, self.onnx_model)

        # Get the hw mapping of this node
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
        node_mapping = pickle

        node_attrs = get_layer_node_input_format(input_shape, output_shape, node_mapping)

        node_obj = LayerNode(
            self.node_id,
            node_attrs,
            node_name=self.node.name,
            type=self.node.op_type.lower(),
        )

        logger.info(f"Parsed Add node {self.node.name}")

        return node_obj

