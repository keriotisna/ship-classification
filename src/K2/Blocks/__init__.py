from .DepthwiseSeparableConv2d import *
from .ResidualBlock import *
from .ResidualBlockVersatile import *
from .SeparableConv2d import *
from .BlurPool import *
from .DepthwiseSeparableConv2dAntiAliased import *

# Explicitly define what's exported
__all__ = [
    'DepthwiseSeparableConv2d',
    'ResidualBlock',
    'ResidualBlockVersatile',
    'SeparableConv2d'
]


# K2/Blocks/__init__.py
# Optional way to import all files if they remain consistent
# import os
# import importlib
# import inspect
# import sys

# # Get the current package path
# package_dir = os.path.dirname(os.path.abspath(__file__))

# # Initialize empty __all__ list
# __all__ = []

# # Process each Python file in the directory
# for filename in os.listdir(package_dir):
#     # Skip __init__.py and non-Python files
#     if filename == '__init__.py' or not filename.endswith('.py'):
#         continue
    
#     # Get module name (filename without .py)
#     module_name = filename[:-3]
    
#     # Import the module
#     module = importlib.import_module(f'.{module_name}', package=__name__)
    
#     # Find all classes defined in this module
#     for name, obj in inspect.getmembers(module, inspect.isclass):
#         # Only include classes defined in this module (not imported classes)
#         if obj.__module__ == f'{__name__}.{module_name}':
#             # Add to globals so it's accessible from the package
#             globals()[name] = obj
#             # Add to __all__ so it's exported with "from K2.Blocks import *"
#             __all__.append(name)

# # Optional: print what was discovered for debugging
# # print(f"K2.Blocks imported: {__all__}")