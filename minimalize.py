"""
This is the compressed version of interplot.

Unpacking:

>>> from interplot_minimal import unpack
... unpack()

Then either:
```bash
cd interplot_module
pip install -e .
```

or
1. move to the directory interplot_module
2. create a new notebook
3. >>> pip install -e .
4. if no imports possible: >>> pip install --no-dependencies -e .

or
1. move the inner `interplot` folder to the directory where you want to use it.


to use it:
>>> import interplot as ip
"""


import re
import io
from pathlib import Path


files_to_minimalize = [
    "setup.py",
    "requirements.txt",
    "README.md",
    "interplot",
]


def minimalize(files: None):
    if files is None:
        files = files_to_minimalize

    with io.open(Path(__file__), "r", encoding="utf-8") as f:
        minimalizer = f.read()

    with io.open(Path("interplot_minimal.py"), "w+", encoding="utf-8") as f:

        f.write(minimalizer)

        for file in files:
            minimalize_path_object(file, f)

        f.write(
            "\n#############\n"
            "#### END ####\n"
            "#############\n"
        )


def minimalize_path_object(path, f):
    if isinstance(path, str):
        path = Path(path)
    if path.is_dir():
        for path_ in path.iterdir():
            minimalize_path_object(path_, f)
    else:
        if re.search(r"\.(py|txt|md)$", path.name):
            minimalize_file(path, f)


def minimalize_file(file, f):
    print("WRITING", file)

    path_str = str(file)

    f.write(
        "\n"
        + ("#" * (len(path_str) + 10)) + "\n"
        + "#### " + path_str + " ####\n"
        + ("#" * (len(path_str) + 10))
    )

    with io.open(file, "r", encoding="utf-8") as f_:
        content = f_.read()
        content = re.sub(
            r"\n",
            r"\n#   ",
            content,
        )
        f.write("\n#   ")
        f.write(content)


def unpack(dir="interplot_module"):
    """Unpack the module to the current directory."""
    this_file = Path(__file__)
    with io.open(this_file, "r", encoding="utf-8") as file:
        content = file.read()
    dir = this_file.parent / dir
    dir.mkdir(parents=True, exist_ok=True)
    (dir / "interplot").mkdir(parents=True, exist_ok=True)

    for match in re.finditer(
        (
            r"^#### ([a-zA-Z0-9._/-]*?) ####\n####+?\n"  # file flag
            r"(.*?)"  # file content
            r"####+?\n"  # next file flag
        ),
        content,
        re.DOTALL | re.MULTILINE
    ):
        path = dir / match.group(1).strip()
        filecontent = re.sub(
            r"\n#[ ]{0,3}",
            r"\n",
            match.group(2),
        )[4:]
        print("unpacking:", path)
        # print(filecontent[:100])
        with io.open(path, "w+", encoding="utf-8") as file:
            file.write(filecontent)

    print("Successfully unpacked interplot to current directory.\n")
    print("Now call:")
    print("cd interplot_module")
    print("pip install -e .")
    print("or")
    print("pip install --no-dependencies -e .")
    print("")
    print("If pip installs are not possible,")
    print("just move the inner `interplot` folder")
    print("to the directory where you want to use it.")


if __name__ == "__main__":
    minimalize(None)

    print("Successfully minimized interplot to interplot_minimal.py.")

    # unpack()
