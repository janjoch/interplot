import datetime
import re


def parse_datetime(timestr):
    match = re.search(
        r'^([0-9]{4})'
        r'-([0-9]{2})'
        r'-([0-9]{2})'
        r'T([0-9]{2})'
        r':([0-9]{2})'
        r':([0-9]{2})'
        r'\.([0-9]{3})$',
        timestr,
    )
    if(match):
        timelist = [
            int(match_strs)
            for match_strs
            in match.group(*range(1, 8))
        ]
        timelist[6] = timelist[6] * 1000
        parsed = datetime.datetime(*timelist)
        return(parsed)
    else:
        raise Exception("cannot parse " + str(timestr))
