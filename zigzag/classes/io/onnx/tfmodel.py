from typing import Union

from zigzag.classes.io.onnx.default import DefaultNodeParser
from zigzag.classes.io.onnx.gemm import GemmParser
from zigzag.classes.io.onnx.matmul import MatMulParser
from zigzag.classes.io.onnx.conv import ConvParser
from zigzag.classes.io.onnx.utils import (
    parse_mapping_from_path,
    parse_onnx_model_from_path,
)
from zigzag.classes.workload.pb_workload import PBWorkload

import tensorflow as tf


import logging

logger = logging.getLogger(__name__)

class PBModelParser:
    """Parse the protobuf model into a workload."""

    def __init__(self, pb_model: Union[str, bytes], mapping_path: str) -> None:
        # Sanity checks on given pb_model
        if isinstance(pb_model, str):
            self.pb_model_path = pb_model
            self.pb_model = None
        elif isinstance(pb_model, bytes):
            self.pb_model_path = None
            self.pb_model = pb_model
        else:
            raise TypeError(f"Given pb_model is of type {type(pb_model)}.")

        # Sanity checks on given mapping
        if isinstance(mapping_path, str):
            self.mapping_path = mapping_path
            self.mapping = None
        else:
            raise TypeError(f"Given mapping_path is of type {type(mapping_path)}.")

        self.workload = None

    def run(self):
        """Run the parser:
        - parse the pb_model_path into a protobuf model
        - parse the mapping_path into a mapping dict
        - iterate through the protobuf model and generate the workload consisting of LayerNodes and DummyNodes
        """
        if not self.pb_model:
            with open(self.pb_model_path, "rb") as f:
                pb_model = f.read()
            self.pb_model = pb_model

        mapping = parse_mapping_from_path(self.mapping_path)
        self.mapping = mapping

        workload = self.parse_workload_from_pb_model_and_mapping()
        self.workload = workload

    def parse_workload_from_pb_model_and_mapping(self):
        nodes_inputs = {}
        nodes_outputs = {}

        # Workload Graph
        workload = PBWorkload()

        with tf.Graph().as_default():
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(self.pb_model)
            tf.import_graph_def(graph_def, name="")

            for node in graph_def.node:
                nodes_inputs[node.name] = [inp.split(":")[0] for inp in node.input]
                nodes_outputs[node.name] = [out.split(":")[0] for out in node.output]

                if node.op in ["Conv2D", "DepthwiseConv2dNative"]:
                    parser = ConvParser(
                        node.name, node, nodes_outputs, self.mapping, graph_def
                    )
                elif node.op in ["MatMul"]:
                    parser = MatMulParser(
                        node.name, node, nodes_outputs, self.mapping, graph_def
                    )
                # elif node.op in ["Add", "BiasAdd"]:
                #     parser = AddParser(
                #         node.name, node, nodes_outputs, self.mapping, graph_def
                #     )
                else:  # it is not a convolutional node, so create a DummyNode
                    parser = DefaultNodeParser(node.name, node, nodes_outputs)
                node_obj = parser.run()
                # Add the node_obj to the PBWorkload
                workload.add(node.name, node_obj)

        logger.info(
            f"Created PBWorkload graph with {workload.number_of_nodes()} nodes and {workload.number_of_edges()} edges."
        )
        return workload

    def get_pb_model(self):
        return self.pb_model

    def get_mapping(self):
        return self.mapping

    def get_workload(self):
        return self.workload
