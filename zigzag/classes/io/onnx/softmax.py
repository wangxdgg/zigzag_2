import onnx
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
        def get_layer_node_input_format(node_mapping, nodes_outputs):
            """
            Generate the necessary dictionary items required for the LayerNode creation.
            """
            # Equation
            d = {}
            d["equation"] = "O[b][c][y][x] = exp(I[b][c][y][x]) / sum(exp(I[b][c'][y][x]))"

            # Get dimension sizes from input parameters
            B = node_mapping["batch_size"]
            C = node_mapping["num_classes"]
            H = node_mapping["height"]
            W = node_mapping["width"]

            d["loop_dim_size"] = {"B": B, "C": C, "H": H, "W": W}
            d["pr_loop_dim_size"] = {}
            d["dimension_relations"] = []
            d["operand_precision"] = {"I": 16, "O": 16}
            d["constant_operands"] = []

            d["core_allocation"] = node_mapping["core_allocation"]
            d["spatial_mapping"] = node_mapping["spatial_mapping"]
            d["temporal_ordering"] = node_mapping.get("temporal_ordering", None)
            d["memory_operand_links"] = {"I": "I", "O": "O"}

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

        # Check that Softmax is applied along the correct axis (usually axis=1)
        axis_attr = self.node.attribute[0]
        if axis_attr.name == "axis" and axis_attr.i != 1:
            raise ValueError("Softmax should typically be applied along axis=1.")

        # Get the number of classes from the output shape
        num_classes = oa_dimension_shape[1]

        # Get the batch size from the input shape
        batch_size = ia_dimension_shape[0]

        # Get the height and width from the input shape
        height = ia_dimension_shape[2]
        width = ia_dimension_shape[3]

        # Create the node mapping
        node_mapping = {
            "batch_size": batch_size,
            "num_classes": num_classes,
            "height": height,
            "width": width,
            "core_allocation": {"core_id": 0},
            "spatial_mapping": {"x": 0, "y": 0},
        }

        node_attrs = get_layer_node_input_format(node_mapping, self.nodes_outputs)

        node_obj = LayerNode(
            self.node_id,
            node_attrs,
            node_name=self.node.name,
            type=self.node.op_type.lower(),
        )

        logger.info(f"Parsed Softmax node {self.node.name}")

        return node_obj
