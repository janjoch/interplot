from setuptools import setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='interplot',
    version='0.1.1',
    description=(
        "Create matplotlib/plotly hybrid plots with a few lines of code."
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/janjoch/interplot',
    author='Janosch Jörg',
    author_email='janjo@duck.com',
    license='GPL v3',
    packages=['interplot'],
    install_requires=requirements,

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
