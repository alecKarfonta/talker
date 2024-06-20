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