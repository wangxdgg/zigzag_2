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
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "X": "I2"},
    },
    "Linear": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("K", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "W": "I2", "X": "I1"},
    },
    "Mul": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("K", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
    },
    "Gemm": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("K", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "A": "I1", "B": "I2", "C": "O"},
    },
    "LayerNorm": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "X": "I1", "W": "I2"},
    },
    "Transpose": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "X": "I1"},
    },
    "Reshape": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "X": "I1"},
    },
    "Flatten": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "X": "I1"},
    },
    "Roll": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "X": "I1"},
    },
    "Softmax": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "X": "I1"},
    },
    "Permute": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "X": "I1"},
    },
    "PixelShuffle": {
        "core_allocation": 1,
        "spatial_mapping": {"D1": ("G", 32), "D2": ("C", 32)},
        "memory_operand_links": {"O": "O", "X": "I1"},
    },
}


