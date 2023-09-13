# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys, shutil
from pathlib import Path

sys.path.insert(0, os.path.abspath("../src"))

# The following enables read-in and parsing of the README.md source package file into sphinx docs.
# It was adapted from:
# https://www.lieret.net/2021/05/20/include-readme-sphinx/
# to also rewrite the path of Figure1.png
import pathlib

readme_path = pathlib.Path(__file__).parent.resolve().parent / "README.md"
readme_target = pathlib.Path(__file__).parent / "readme.md"
with readme_target.open("w") as outf:
    outf.write(
        "\n".join(
            [
                "Readme",
                "======",
            ]
        )
    )
    lines = []
    for line in readme_path.read_text().split("\n")[1:]:
        if line.startswith("#"):
            line = line[1:]
        if "Figure1.png" in line:
            line = line.replace("Figure1.png", "_static/Figure1.png")
        lines.append(line)
    outf.write("\n".join(lines))
# End of README.md importer/parser
print(Path(".").cwd())

# Copy Figure1.png to docsource/_static
Path("_static").mkdir(exist_ok=True)
shutil.copyfile("../Figure1.png", Path("_static") / "Figure1.png")

project = 'OpenFEPOPS'
copyright = '2023, Yan-Kai Chen'
author = 'Yan-Kai Chen'
release = '1.6.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
source_suffix = ['.rst', '.md']
