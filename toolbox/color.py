"""Change the brightness of a hex color."""


def change_hex_brightness(color, factor, to_white=False, hash_out=True):
    """
    Change the brightness of a hex color.

    Parameters
    ----------
    color: str
        6-digit hex colour, with or withoud leading hash.
    factor: float
        Factor by which to brighten the color.
        >1 will brighten up.
        <1 will darken.
    to_white: bool, optional
        Instead of multiplying the brightness,
        divide the remainder to full white.
        Default False.
    hash_out: bool, optional
        Return the color with a hash.
        Default True.

    Returns
    -------
    str:
        New color, with leading hash (default)
    """
    # input validation
    if len(color) != 6:
        if len(color) == 7 and color[0] == "#":
            color = color[1:]
        else:
            raise ValueError("Expected 6 digit hex color.")
        if factor < 0:
            raise ValueError("Factor must be a positive value.")

    out = "#" if hash_out else ""

    for i in range(3):
        if to_white:
            c = int(color[2 * i: 2 * i + 2], 16)
            rest = 256 - c
            c = 256 - (rest / factor)
            if c < 0:
                c = 0
            else:
                c = int(c)

        else:
            c = int(color[2 * i: 2 * i + 2], 16) * factor
            if c > 255:
                c = 255
            else:
                c = int(c)

        out = f"{out}{c:02X}"

    return out
