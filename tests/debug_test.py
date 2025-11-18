import interplot as ip

# import pytest


def test_debug_log(capsys):

    @ip.debug.wiretap
    def inner(foo):
        return foo + foo

    inner("bar1")
    ip.debug.start_logging()
    inner("bar2.1")
    inner(foo="bar2.2")
    ip.debug.stop_logging()
    inner("bar3")

    assert len(ip.debug.log) == 2

    assert ip.debug.log[0]["args"] == ("bar2.1",)
    assert ip.debug.log[0]["kwargs"] == {}

    assert ip.debug.get_log(1)["args"] == ()
    assert ip.debug.log[1]["kwargs"] == dict(foo="bar2.2")

    ip.debug.clear_log()

    assert len(ip.debug.log) == 0

    ip.debug.start_logging(save_to_log=False, verbose=False)
    inner("bar4")
    ip.debug.stop_logging()

    assert len(ip.debug.log) == 0

    out, _ = capsys.readouterr()
    print(out)
    out = out.split("\n")
    assert out[0] == "Wiretap log: {"
    assert out[2][:]
    assert len(out) == 19
