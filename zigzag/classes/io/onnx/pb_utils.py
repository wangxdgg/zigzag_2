import enum
import importlib
import logging
from dataclasses import dataclass
from enum import auto
from os import path
from typing import List


import tensorflow as tf

logger = logging.getLogger(__name__)


def parse_mapping_from_path(mapping_path):
    """
    Parse the input accelerator residing in accelerator_path.
    """
    # Sanity check on mapping_path
    if mapping_path is None:
        # Update the mapping_path to the default mapping file
        if path.exists("inputs/examples/mapping/default.py"):
            mapping_path = "zigzag.inputs.examples.mapping.default"
        else:
            raise ValueError(
                "No mapping path/dict provided, and default was not found."
            )
    global module
    module = importlib.import_module(mapping_path)
    mapping = module.mapping
    if "default" in mapping:
        default_present = "\u2705"
    else:
        default_present = "\u274C"
    logger.debug(
        f"Parsed mapping with {len(mapping)} different entries. Default: {default_present}."
    )
    return mapping


def parse_pb_model_from_path(pb_model_path):
    with tf.io.gfile.GFile(pb_model_path, 'rb') as f:
        pb_model = f.read()
    return pb_model


def get_attribute_ints_with_name(name, attrs, default=None):
    """
    Retrieves the attrs[name_idx].ints from attrs.
    If attrs[name_idx] is of type INTS, attrs[name_idx].ints is returned.
    If attrs[name_idx] is of type INT, attrs[name_idx].i is returned.
    If name does not exist in attrs, the default provided by the caller is used.
    If the caller doesn't supply a default, an error is thrown.
    """
    attrs_names = [attr.name for attr in attrs]
    try:
        name_idx = attrs_names.index(name)
        attr_type = attrs[name_idx].type
        if attr_type == tf.AttrValue.AttrType.INT:
            return attrs[name_idx].i
        elif attr_type == tf.AttrValue.AttrType.LIST_INT:
            return attrs[name_idx].list.i
        else:
            raise NotImplementedError(
                f"Attribute extraction of type {attr_type} not supported."
            )
    except ValueError:
        if default is not None:
            return default
        else:
            raise ValueError(
                f"attrs has no attribute called {name} and no default was given. Names = {attrs_names}."
            )


class PBTensorCategory(enum.Enum):
    Input = auto()
    Output = auto()
    Hidden = auto()
    Constant = auto()

    @property
    def is_output(self):
        return self == PBTensorCategory.Output

    @property
    def is_input(self):
        return self == PBTensorCategory.Input

    @property
    def is_hidden(self):
        return self == PBTensorCategory.Hidden

    @property
    def is_constant(self):
        return self == PBTensorCategory.Constant


@dataclass
class PBTensorType:
    shape: List[int]
    dtype: str
    category: PBTensorCategory

    @staticmethod
    def from_tensor_info(tensor_info, category: PBTensorCategory):
        shape = [dim.size for dim in tensor_info.shape.dim]
        dtype = tensor_info.dtype
        return PBTensorType(shape, dtype, category)


def get_pb_tensor_type(name, graph):
    for op in graph.node:
        if name in op.output:
            tensor_info = graph.get_tensor_by_name(name)
            return PBTensorType.from_tensor_info(tensor_info, PBTensorCategory.Output)
    for init in graph.initializer:
        if init.name == name:
            return PBTensorType(list(init.dims), init.data_type.decode("utf-8"), PBTensorCategory.Constant)
    raise KeyError(
        f"Could not find type for value {name} in graph."
        f"Make sure you are loading in an inferred model, "
        f"see https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#running-shape-inference-on-an-onnx-model"
    )


def get_node_input_output_dimension_shapes(node, graph):
    # assumed it is the first input, don't see a way to otherwise know
    input_name = node.input[0]
    input_shape = get_pb_tensor_type(input_name, graph).shape

    output_name = node.output[0]
    output_shape = get_pb_tensor_type(output_name, graph).shape

    return input_shape, output_shape
