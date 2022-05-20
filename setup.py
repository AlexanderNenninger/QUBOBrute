import pathlib
from gettext import install

from setuptools import setup

from qubobrute import __version__

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="qubobrute",
    version=__version__,
    description="A Python package for solving QUBOs using brute force on an NVidia GPU.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/AlexanderNenninger/QUBOBrute",
    author="Alexander Nenninger",
    author_email="alexander.nenninger.devoutlook.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    packages=["qubobrute"],
    install_requires=["numpy", "numba", "cudatoolkit", "pyqubo", "scipy"],
)
