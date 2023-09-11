mapping = {
    "default": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("K", 16),
            "D2": ("C", 16),
            "D3": ("OX", 2),
            "D4": ("OY", 2),
        },
        "memory_operand_links": {"O": "O", "W": "I2", "I": "I1"},
    },
    "Add": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("G", 16),
            "D2": ("C", 1),
            "D3": ("OX", 1),
            "D4": ("OY", 1),
        },
        "memory_operand_links": {"O": "O", "X": "I2", "Y": "I1"},
    },
    "Gelu": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("G", 16),
            "D2": ("C", 16),
            "D3": ("OX", 2),
            "D4": ("OY", 2),
        },
        "memory_operand_links": {"O": "O", "X": "I1"},
    },
    "Linear": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("N", 16),
            "D2": ("K", 16),
        },
        "memory_operand_links": {"O": "O", "X": "I2", "W": "I1"},
    },
    "Mul": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("N", 16),
            "D2": ("K", 16),
        },
        "memory_operand_links": {"O": "O", "X": "I1", "Y": "I2"},
    },
    "Gemm": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("M", 16),
            "D2": ("N", 16),
            "D3": ("K", 16),
        },
        "memory_operand_links": {"O": "O", "A": "I1", "B": "I2"},
    },
    "LayerNorm": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("N", 16),
            "D2": ("C", 16),
        },
        "memory_operand_links": {"O": "O", "X": "I1", "W": "I2", "B": "I3"},
    },
    "Transpose": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("N", 16),
            "D2": ("C", 16),
            "D3": ("H", 16),
            "D4": ("W", 16),
        },
        "memory_operand_links": {"O": "O", "X": "I1"},
    },
    "Reshape": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("N", 16),
            "D2": ("C", 16),
            "D3": ("H", 16),
            "D4": ("W", 16),
        },
        "memory_operand_links": {"O": "O", "X": "I1", "S": "I2"},
    },
    "Flatten": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("N", 16),
            "D2": ("C", 16),
            "D3": ("H", 16),
            "D4": ("W", 16),
        },
        "memory_operand_links": {"O": "O", "X": "I1"},
    },
    "Roll": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("N", 16),
            "D2": ("C", 16),
            "D3": ("H", 16),
            "D4": ("W", 16),
        },
        "memory_operand_links": {"O": "O", "X": "I1", "S": "I2"},
    },
    "Softmax": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("N", 16),
            "D2": ("C", 16),
        },
        "memory_operand_links": {"O": "O", "X": "I1"},
    },
    "Permute": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("N", 16),
            "D2": ("C", 16),
            "D3": ("H", 16),
            "D4": ("W", 16),
        },
        "memory_operand_links": {"O": "O", "X": "I1", "P": "I2"},
    },
    "PixelShuffle": {
        "core_allocation": 1,
        "spatial_mapping": {
            "D1": ("N", 16),
            "D2": ("C", 16),
            "D3": ("H", 16),
            "D4": ("W", 16),
        },
        "memory_operand_links": {"O": "O", "X": "I1"},
    },
}

