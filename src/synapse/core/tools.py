import ast
import logging
import importlib
import math

import awkward as ak
import numpy as np
import torch

_logger = logging.getLogger("SynapseLogger")

def _array(arr):
    return ak.Array(arr) if isinstance(arr, (list, tuple)) else arr

safe_builtins = {
    "abs": np.abs,
    "round": np.round,
    "sum": ak.sum,
    "min": ak.min,
    "max": ak.max,
    "sqrt": np.sqrt,
    "log": np.log,
    "log1p": np.log1p,
    "exp": np.exp,
    "expm1": np.expm1,
    "sin": np.sin,
    "cos": np.cos,
    "mean": ak.mean,
    "std": ak.std,
    "any": ak.any,
    "all": ak.all,
    "len": len,
    "array": _array,
}  # TODO: add more functions if needed

supported_modules = {
    "ak": ak,
    "np": np,
    "math": math,
    # Add more modules as needed
}

def customized_eval(data: ak.Array, expr: str):
    locals_env = {
        '_data': data,
        **safe_builtins,
        **supported_modules,
        **{k: data[k] for k in data.fields},
    }
    return eval(expr, {"__builtins__": None}, locals_env)

def build_new_variables(data: ak.Array, new_var_entries: dict | None, supress_warning: bool = False) -> ak.Array:
    """
    Build new variables from the data

    Args:
        data (ak.Array): input data
        new_var_entries (dict | None): new variables to build
        supress_warning (bool): if True, suppress warnings for existing variables
    Returns:
        ak.Array: data with new variables
    """
    if new_var_entries:
        for var_name, var_expr in new_var_entries.items():
            if var_name in data.fields:
                if not supress_warning:
                    _logger.warning(f"{var_name} already exists in the input data, skipping")
                continue
            try:
                data[var_name] = customized_eval(data, var_expr)
            except Exception as e:
                _logger.error(f"Error evaluating '{var_name}' with expression '{var_expr}': {e}")
    return data

def apply_selection(data: ak.Array, selection_expr: str|None, new_var_entries: dict = None) -> ak.Array:
    """
    Apply a selection to the data

    Args:
        data (ak.Array): input data
        selection_expr (str | None): selection expression to apply
        new_var_entries (dict): new variables to build
    Returns:
        ak.Array: selected data
    Raises:
        ValueError: If the selection expression produces a non-boolean mask
        KeyError: If the selection expression uses field(s) not in data
    """
    if selection_expr is None:
        return data
    try:
        # Check if all required fields are available
        fields_in_expr = extract_fields_from_expr(selection_expr)
        missing_fields = [field for field in fields_in_expr if field not in data.fields]
        if new_var_entries:
            if any(field in missing_fields for field in new_var_entries.keys()):
                build_new_variables(data, new_var_entries)
                missing_fields = [field for field in missing_fields if field not in new_var_entries.keys()]
        if missing_fields:
            raise KeyError(f"Fields not found in data: {', '.join(missing_fields)}")

        # Evaluate the selection expression
        mask = customized_eval(data, selection_expr)

        # Verify mask is boolean
        if not isinstance(mask.type, ak.types.ArrayType) or mask.type.content.primitive != 'bool':
            raise ValueError(f"Selection expression '{selection_expr}' produced a non-boolean mask of type {mask.type}")

        # Apply the mask
        return data[mask]
    except Exception as e:
        _logger.error(f"Error applying selection '{selection_expr}': {e}")
        return ak.Array([])  # Return an empty array on error # FIXME: should we raise an error instead?

def extract_fields_from_expr(expr: str) -> set:
    """
    Extract field names from an expression.

    Args:
        expr (str): Expression to analyze

    Returns:
        set: Set of potential field names in the expression
    """
    # Filter out Python keywords and known safe functions and modules
    # FIXME: sooo ugly, how to improve this?
    python_keywords = {
        'and', 'or', 'not', 'if', 'else', 'for',
        'in', 'True', 'False', 'None'
    }
    safe_functions = safe_builtins.keys()
    module_names = supported_modules.keys()

    try:
        # Parse the expression using ast to extract variable names
        tree = ast.parse(expr)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}

        # Return potential field names by excluding known non-field tokens
        return set(name for name in names
                 if name not in python_keywords
                 and name not in safe_functions
                 and name not in module_names)
    except SyntaxError:
        # Fall back to regex in case of invalid Python syntax
        import re
        tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expr)

        # Filter out known tokens as before
        return set(token for token in tokens
                  if token not in python_keywords
                  and token not in safe_functions
                  and token not in module_names)

def dynamic_import(dotted_path: str):
    """
    Dynamically import a module or object from a dotted path.
    """
    try:
        module_path, obj_name = dotted_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, obj_name)
        return obj
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"'{dotted_path}': {str(e)}") from e

def flatten_nested_list(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten_nested_list(item))
        else:
            result.append(item)
    return result

def is_scalar(value):
    """
    Check if the value is a scalar.
    """
    if isinstance(value, (int, float, bool)):
        return True

    if isinstance(value, np.generic):
        return True

    if isinstance(value, np.ndarray):
        return value.ndim == 0

    if torch.is_tensor(value):
        return value.dim() == 0

    return False
