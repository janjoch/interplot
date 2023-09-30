from setuptools import setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name='toolbox',
    version='0.2.3',
    description=(
        "Janosch's small Python code snippets "
        "making life a bit easier."
    ),
    url='https://github.com/janjoch/toolbox',
    author='Janosch Jörg',
    author_email='janjo@duck.com',
    license='GPL v3',
    packages=['toolbox'],
    install_requires=requirements,

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
