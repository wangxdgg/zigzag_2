# mapping = {
#     "default": {
#         "core_allocation": 1,
#         "spatial_mapping": {"D1": ("K", 32), "D2": ("C", 32)},
#         "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
#     },
#     "Add": {
#         "core_allocation": 1,
#         "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 1)},
#         "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
#     },
#     "Pooling": {
#         "core_allocation": 1,
#         "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 1)},
#         "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
#     },
# }


mapping = {
    "default": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("K", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
    "Add": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 1)},
        "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
    },
    "Pooling": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 1)},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
    "Gelu": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 16), "D2": ("C", 4)},
        "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
    },
    "Linear": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 64), "D2": ("C", 8)},
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
    "Mul": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 16), "D2": ("C", 8)},
        "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
    },
    "Gemm": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "A": "I2", "B": "I1"},
    },
    "LayerNorm": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "X": "I1", "Scale": "I2", "Bias": "I3"},
    },
    "Transpose": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "A": "I1"},
    },
    "Reshape": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "I": "I1", "Shape": "I2"},
    },
    "Flatten": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "I": "I1"},
    },
    "Roll": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "I": "I1", "Shifts": "I2"},
    },
}

