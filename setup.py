from setuptools import setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name='interplot',
    version='0.0.1',
    description=(
        "Create matplotlib/plotly hybrid plots with a few lines of code."
    ),
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
    ],
)
