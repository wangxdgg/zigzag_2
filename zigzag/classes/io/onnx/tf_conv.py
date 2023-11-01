from math import ceil

from zigzag.classes.io.onnx.parser import Parser
from zigzag.classes.io.onnx.utils import (
    get_attribute_ints_with_name,
    get_node_input_output_dimension_shapes, get_onnx_tensor_type,
)
from zigzag.classes.workload.layer_node import LayerNode
from zigzag.utils import pickle_deepcopy

import logging

logger = logging.getLogger(__name__)


from math import ceil
import tensorflow as tf


class ConvParser:
    def __init__(self, node_id, node, nodes_outputs, mapping, pb_model_path):
        self.node_id = node_id
        self.node = node
        self.nodes_outputs = nodes_outputs
        self.mapping = mapping
        self.pb_model_path = pb_model_path
        self.graph = self.load_pb_model()

    def load_pb_model(self):
        graph = tf.Graph()
        with tf.compat.v1.Session(graph=graph) as sess:
            with tf.io.gfile.GFile(self.pb_model_path, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

        return graph

    def run(self):
        layer_node = self.generate_layer_node_for_conv()
        return layer_node

    def generate_layer_node_for_conv(self):
        def get_weight_name(node):
            op_type = node.op_type
            if op_type == "Conv":
                return node.input[1]
            elif op_type == "QLinearConv":
                return node.input[3]
            else:
                raise NotImplementedError(f"Retrieving weight name for node of type {op_type} is not supported.")

        def get_input_output_weight_data_type(node):
            input_name = node.input[0]
            output_name = node.output[0]
            weight_name = get_weight_name(node)

            input_elem_type = self.get_tensor_data_type(input_name)
            output_elem_type = self.get_tensor_data_type(output_name)
            weight_elem_type = self.get_tensor_data_type(weight_name)

            return input_elem_type, output_elem_type, weight_elem_type

        def get_layer_node_input_format(
                kernel_shape,
                strides,
                dilations,
                groups,
                padding,
                ia_shape,
                oa_shape,
                node_mapping,
        ):
            d = {}
            d["equation"] = "O[b][g][k][oy][ox]+=W[g][k][c][fy][fx]*I[b][g][c][iy][ix]"

            B = oa_shape[0]
            if B == 0:
                B = 1
            G = groups
            K = ceil(oa_shape[1] / G)
            OX = oa_shape[2]
            OY = oa_shape[3]
            C = ceil(ia_shape[1] / G)
            IX = ia_shape[2]
            IY = ia_shape[3]
            FX = kernel_shape[0]
            FY = kernel_shape[1]

            d["loop_dim_size"] = {
                "B": B,
                "K": K,
                "G": G,
                "OX": OX,
                "OY": OY,
                "C": C,
                "FX": FX,
                "FY": FY,
            }

            d["pr_loop_dim_size"] = {"IX": IX, "IY": IY}
            d["dimension_relations"] = [
                f"ix={strides[0]}*ox+{dilations[0]}*fx",
                f"iy={strides[1]}*oy+{dilations[1]}*fy",
            ]

            d["operand_precision"] = {"O": 16, "O_final": 8, "W": 8, "I": 8}
            d["constant_operands"] = ["W"]

            d["core_allocation"] = node_mapping["core_allocation"]
            d["spatial_mapping"] = node_mapping["spatial_mapping"]
            d["temporal_ordering"] = node_mapping.get("temporal_ordering", None)
            d["memory_operand_links"] = node_mapping["memory_operand_links"]

            node_inputs = self.node.input
            preds = []

            for node_input in node_inputs:
                for n in self.nodes_outputs:
                    if node_input in self.nodes_outputs[n]:
                        preds.append(n)

            d["operand_source"] = {"I": preds}

            d["padding"] = {
                "IY": (padding[0], padding[2]),
                "IX": (padding[1], padding[3]),
            }

            return d

        attrs = self.node.attribute
        kernel_shape = get_attribute_ints_with_name("kernel_shape", attrs, default=None)
        strides = get_attribute_ints_with_name("strides", attrs, default=[1, 1])
        dilations = get_attribute_ints_with_name("dilations", attrs, default=[1, 1])
        groups = get_attribute_ints_with_name("group", attrs, default=1)
        padding = get_attribute_ints_with_name("pads", attrs, default=[0, 0, 0, 0])

        ia_dimension_shape, oa_dimension_shape = self.get_node_input_output_dimension_shapes()

        ia_data_type, oa_data_type, w_data_type = get_input_output_weight_data_type(self.node)

        if self.node.name in self.mapping:
            node_mapping = self.mapping[self.node.name]
        else:
            try:
                node_mapping = self.mapping["default"]
            except:
                raise ValueError(f"There is no mapping provided for node {self.node.name}, nor a default one.")

        node_mapping = pickle_deepcopy(node_mapping)

        node_attrs = get_layer_node_input_format(
            kernel_shape,
            strides,
            dilations,
            groups,
            padding,
            ia_dimension_shape,
            oa_dimension_shape,
            node_mapping,
        )

        node_obj = LayerNode(
            self.node_id,
            node_attrs,
            node_name=self.node.name,
            type=self.node.op_type.lower(),
        )

        logger.info(f"Parsed Conv node {self.node.name}")

        return node_obj
