"""Sphinx configuration for the Copenhagen documentation."""

project = "Copenhagen"
author = "Alexander Kolpakov"
copyright = "2026, Alexander Kolpakov"
release = "0.1.0"

extensions = [
    "sphinx.ext.mathjax",      # LaTeX math in HTML
    "sphinx.ext.autodoc",      # pull docstrings if/when the Python API is documented
    "sphinx.ext.napoleon",     # Google/NumPy docstring styles
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "alabaster"
html_static_path = ["_static"]

# Number equations so the text can reference them as :eq:`label`.
math_number_all = False
math_eqref_format = "Eq. {number}"

# MathJax: enable common macros / AMS environments.
mathjax3_config = {
    "tex": {
        "macros": {
            "argmax": r"\operatorname{arg\,max}",
            "argmin": r"\operatorname{arg\,min}",
        },
        "tags": "ams",
    }
}
