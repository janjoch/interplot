"""Add up all values in dicts, lists or tuples.

Nested structures are added up recursively.
"""


def sum_nested(inp):
    """Add up all values in dicts, lists or tuples.

    Nested structures are added up recursively.
    """
    if(type(inp) is dict):
        inp = [elem for key, elem in inp.items()]

    if(type(inp) is list or type(inp) is tuple):
        val = 0
        for elem in inp:
            if(
                type(elem) is dict
                or type(elem) is list
                or type(elem) is tuple
            ):
                val += sum_nested(elem)

            else:
                val += elem

        return(val)

    return(inp)
