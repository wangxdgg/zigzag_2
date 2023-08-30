from math import ceil

from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import (
    get_attribute_ints_with_name,
    get_node_input_output_dimension_shapes,
    get_onnx_tensor_type,
)
from zigzag.classes.workload.layer_node import LayerNode
from zigzag.utils import pickle_deepcopy

import logging

logger = logging.getLogger(__name__)


class LinearParser(Parser):
    """Parser for ONNX Linear nodes into LayerNode."""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def run(self):
        """Run the parser and return the created LayerNode object."""
        layer_node = self.generate_layer_node_for_linear()
        return layer_node

    def generate_layer_node_for_linear(self):
        def get_weight_name(node):
            """Return the name of the weight input of this node depending on its operator type.
            Args:
                node (NodeProto): The node
            """
            op_type = node.op_type  # 'Conv', 'QLinearConv', ...
            if op_type == "Linear":
                return node.input[1]
            else:
                raise NotImplementedError(
                    f"Retrieving weight name for onnx node of type {op_type} is not supported."
                )

        def get_input_output_weight_data_type(node, model):
            """
            Return the data type of the input, output and weight tensors of this node.
            """

            input_name = node.input[0]
            output_name = node.output[0]
            weight_name = get_weight_name(node)

            input_elem_type = get_onnx_tensor_type(input_name, model).elem_type
            output_elem_type = get_onnx_tensor_type(output_name, model).elem_type
            weight_elem_type = get_onnx_tensor_type(weight_name, model).elem_type

            return input_elem_type, output_elem_type, weight_elem_type

        def get_layer_node_input_format(
            ia_shape, oa_shape, node_mapping
        ):
            """
            Generate the necessary dictionary items required for the LayerNode creation.
            """
            # convert the data types to precisions based on the onnx definition

            # Equation
            d = {}
            # IMPORTANT: If any of the input loops require padding, they should be defined as the rightmost dimensions in the equation
            # This is because we construct the dimensionality order and then add the padding to those last dimensions in the order
            d["equation"] = "O[b][g][c]=W[g][c][o]*I[b][g][c][i]"

            # Get dimension sizes from input parameters
            B = oa_shape[0]
            if B == 0:
                B = 1
            G = 1
            O = oa_shape[1]
            C = ia_shape[1]
            IY = ia_shape[2]
            IX = ia_shape[3]
            FX = 1
            FY = 1
            d["loop_dim_size"] = {
                "B": B,
                "G": G,
                "O": O,
                "C": C,
                "IY": IY,
                "IX": IX,
                "FX": FX,
                "FY": FY,
            }
            d["pr_loop_dim_size"] = {"IY": IY, "IX": IX}
            d["dimension_relations"] = [
                f"ix={strides[0]}*ox+{dilations[0]}*fx",
                f"iy={strides[1]}*oy+{dilations[1]}*fy",
            ]
            d["operand_precision"] = {"O": 16, "O_final": 8, "W": 8, "I": 8}
            # d["operand_source"] =  {'W': [], 'I': []}
            d["constant_operands"] = ["W"]

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

            # Add padding information

            d["padding"] = {"IY": (0, 0), "IX": (0, 0)}

            return d

        attrs = self.node.attribute
        # Find kernel shape in attrs
        # Find strides in attrs
        # Find dilation rate in attrs
        # Find number of groups in attrs
        # Find padding in attrs

        # Get the input and output activation shapes from the ONNX model.
        ia_dimension_shape, oa_dimension_shape = get_node_input_output_dimension_shapes(
            self.node, self.onnx_model
        )

        # Get the input and output activation data types (precision) from the ONNX model.
        ia_data_type, oa_data_type, w_data_type = get_input_output_weight_data_type(
            self.node, self.onnx_model
        )

        # Get the hw mapping of the current node.
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

        logger.info(f"Parsed Linear node {self.node.name}")

        return node_obj
