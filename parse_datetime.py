def parse_datetime(self, time_str, drop_ms=True):
    """
    Parse timestamps.

    Input Format: 2022-07-12T13:42:15.622628
    Output Format datetime.datetime
    """
    match = re.match(
        r'^\s*([0-9]+)'
        r'-([0-9]+)'
        r'-([0-9]+)'
        r'T([0-9]+)'
        r':([0-9]+)'
        r':([0-9]+)'
        r'\.([0-9]+)\s*$',
        time_str,
    )
    ints = np.zeros(7,dtype=np.int64)
    if(match):
        for i in range(7):
            ints[i] = int(match[i+1])
    else:
        raise Exception("nothing found in "+time_str)
    if(drop_ms):
        ints[6] = 0
    return dt.datetime(*ints)
