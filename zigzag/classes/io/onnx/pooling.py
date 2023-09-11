import onnx
from onnx import numpy_helper
from onnx.helper import make_node
from onnx import TensorProto
import numpy as np

from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import (
    get_node_input_output_dimension_shapes,
    get_attribute_ints_with_name,
)
from zigzag.classes.workload.layer_node import LayerNode

import logging

from zigzag.utils import pickle_deepcopy

logger = logging.getLogger(__name__)


class PoolingParser(Parser):
    """Parser for ONNX pooling nodes into LayerNode."""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, mapping, onnx_model)

    def run(self):
        """Run the parser and return the created LayerNode object."""
        layer_node = self.generate_layer_node_for_pooling()
        return layer_node

    def generate_layer_node_for_pooling(self):
        def get_layer_node_input_format(
            pooling_type, kernel_shape, strides, padding, ia_shape, oa_shape, node_mapping
        ):
            """
            Generate the necessary dictionary items required for the LayerNode creation.
            """
            # convert the data types to precisions based on the onnx definition

            # Equation
            d = {}
            if pooling_type == "max":
                d["equation"] = "O[b][c][oy][ox] = max(O[b][c][oy][ox], I[b][c][iy][ix])"
            elif pooling_type == "average":
                d["equation"] = "O[b][c][oy][ox] += I[b][c][iy][ix]"
            elif pooling_type == "global_average":
                d["equation"] = "O[b][c][0][0] += I[b][c][iy][ix]"

            # Get dimension sizes from input parameters
            B = oa_shape[0]
            C = oa_shape[1]
            OY = oa_shape[2]
            OX = oa_shape[3]
            IY = ia_shape[2]
            IX = ia_shape[3]
            d["loop_dim_size"] = {"B": B, "C": C, "OY": OY, "OX": OX}
            d["pr_loop_dim_size"] = {"IY": IY, "IX": IX}
            d["dimension_relations"] = [
                f"ix = {strides[1]} * ox + {kernel_shape[1]} * fx",
                f"iy = {strides[0]} * oy + {kernel_shape[0]} * fy",
            ]
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

            # Add padding information
            d["padding"] = {
                "IY": (padding[0], padding[2]),
                "IX": (padding[1], padding[3]),
            }

            return d

        attrs = self.node.attribute
        # Find pooling type in attrs
        pooling_type = self.node.op_type.lower()

        # Find kernel shape in attrs
        kernel_shape = get_attribute_ints_with_name("kernel_shape", attrs, default=None)
        if kernel_shape is None:
            raise ValueError("Kernel shape must be specified for pooling operation.")

        # Find strides in attrs
        strides = get_attribute_ints_with_name("strides", attrs, default=[1, 1])

        # Find padding in attrs
        padding = get_attribute_ints_with_name("pads", attrs, default=[0, 0, 0, 0])

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
            pooling_type, kernel_shape, strides, padding, ia_dimension_shape, oa_dimension_shape, node_mapping
        )

        node_obj = LayerNode(
            self.node_id,
            node_attrs,
            node_name=self.node.name,
            type=pooling_type,
        )

        logger.info(f"Parsed Pooling node {self.node.name}")

        return node_obj
