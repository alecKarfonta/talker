def print_dict_structure(d, indent=0, parent_is_list=False):
    """
    Recursively prints the structure of a dictionary, including the type of each object.
    
    Args:
    d (dict): The dictionary to print.
    indent (int): The current indentation level (used for recursion).
    parent_is_list (bool): Indicates if the parent element is a list.
    """
    prefix = '|' if indent > 0 else ''
    for i, (key, value) in enumerate(d.items()):
        is_last = (i == len(d) - 1)
        if is_last and not parent_is_list:
            branch = '└─'
        else:
            branch = '├─'
        
        print(f"{prefix}{'    ' * (indent - 1)}{branch}{key} ({type(value).__name__})")
        
        if isinstance(value, dict):
            print_dict_structure(value, indent + 1)
        elif isinstance(value, list):
            print(f"{prefix}{'    ' * (indent)}├─[")
            for j, item in enumerate(value):
                item_is_last = (j == len(value) - 1)
                if item_is_last:
                    sub_branch = '└─'
                else:
                    sub_branch = '├─'
                if isinstance(item, dict):
                    print(f"{prefix}{'    ' * (indent + 1)}{sub_branch}item ({type(item).__name__})")
                    print_dict_structure(item, indent + 2, parent_is_list=True)
                else:
                    print(f"{prefix}{'    ' * (indent + 1)}{sub_branch}{item} ({type(item).__name__})")
            print(f"{prefix}{'    ' * (indent)}└─]")

import logging



def get_model_size(model, in_gb: bool = True) -> str:
    """
    Calculate the size of a PyTorch model's parameters and buffers.

    Args:
        model (torch.nn.Module): The PyTorch model whose size is to be calculated.
        in_gb (bool): If True, return the size in gigabytes; otherwise, return the size in megabytes.

    Returns:
        str: The size of the model in GB or MB as a string.
    """
    logging.debug(f"{__name__}(): Starting calculation of model size")

    # Initialize parameter size accumulator
    param_size: int = 0
    # Iterate over all model parameters and accumulate their sizes
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        logging.debug(f"{__name__}(): Adding parameter size: {param.nelement() * param.element_size()} bytes")

    # Initialize buffer size accumulator
    buffer_size: int = 0
    # Iterate over all model buffers and accumulate their sizes
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        logging.debug(f"{__name__}(): Adding buffer size: {buffer.nelement() * buffer.element_size()} bytes")

    # Calculate the total size in bytes
    total_size = param_size + buffer_size
    logging.debug(f"{__name__}(): Total size in bytes: {total_size}")

    # Convert total size to GB or MB based on the 'in_gb' flag
    if in_gb:
        size_str = f"{round(total_size / 1024**3, 1)} GB"
        logging.debug(f"{__name__}(): Converted size to GB: {size_str}")
    else:
        size_str = f"{int(total_size / 1024**2)} MB"
        logging.debug(f"{__name__}(): Converted size to MB: {size_str}")

    # Return the formatted size string
    return size_str