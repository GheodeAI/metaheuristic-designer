""" Configuration file for the Sphinx documentation builder."""

import os
import sys
import matplotlib
from dataclasses import is_dataclass

matplotlib.use("Agg")
plot_rcparams = {"backend": "Agg"}

sys.path.insert(0, os.path.abspath("../src/metaheuristic_designer"))
sys.path.insert(0, os.path.abspath("./src/metaheuristic_designer"))


# -- Project information -----------------------------------------------------

project = "metaheuristic-designer"
copyright = "2023, Eugenio Lorente-Ramos"
author = "Eugenio Lorente-Ramos"

release = "1.1.0"

extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc"

]

autodoc_typehints = "description" 

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

add_module_names = False

autosummary_generate = True

autodoc_member_order = "bysource"

autodoc_default_options = {
    'exclude-members': 'a_long_list, of, fields, to, exclude'
}

# -- Options for HTML output -------------------------------------------------

# html_theme = "sphinx_material"
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    # "nav_title": "metaheuristic-designer docs",
    "navbar_align": "left"
}

html_sidebars = {
    "api_reference*": [], 
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["custom.css"]

suppress_warnings = ["autosectionlabel.*"]

nitpicky = False

# exclude_patterns = [
#     "api_reference.plotting.rst"
# ]

def skip_properties(app, what, name, obj, skip, options):
    if isinstance(obj, property):
        return True
    return None

def skip_dataclass_fields(app, what, name, obj, skip, options):
    """Skip documenting fields of dataclasses."""
    if is_dataclass(obj):
        # If the class is a dataclass, check if the member is a field
        if hasattr(obj, '__dataclass_fields__') and name in obj.__dataclass_fields__:
            return True
    return None

def setup(app):
    app.connect('autodoc-skip-member', skip_properties)
    app.connect('autodoc-skip-member', skip_dataclass_fields)