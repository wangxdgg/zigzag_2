from typing import Generator

from zigzag.classes.io.onnx.tfmodel import PBModelParser
from zigzag.classes.stages.Stage import Stage

import logging

logger = logging.getLogger(__name__)


class TensorFlowModelParserStage(Stage):
    """
    Stage that parses a TensorFlow model and creates a PBModelParser.
    """
    def __init__(self, list_of_callables, *, pb_model, mapping, **kwargs):
        super().__init__(list_of_callables, **kwargs)
        self.pb_model_parser = PBModelParser(pb_model, mapping)

    def run(self) -> Generator:
        self.pb_model_parser.run()
        pb_model = self.pb_model_parser.get_pb_model()
        mapping = self.pb_model_parser.get_mapping()
        workload = self.pb_model_parser.get_workload()

        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            pb_model=pb_model,
            mapping=mapping,
            workload=workload,
            **self.kwargs
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    # For testing purposes
    def is_leaf(self) -> bool:
        return True