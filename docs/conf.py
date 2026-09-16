"""Sphinx configuration.

The site is generated from the docstrings, not written separately.  Adding a
metric means writing its docstring; the API page picks it up on the next push.
"""

from importlib.metadata import version as _version

project = "veloeval"
copyright = "2026, VeloBench contributors"
author = "VeloBench contributors"
release = _version("veloeval")

extensions = [
    "sphinx.ext.autodoc",        # pull docstrings out of the code
    "sphinx.ext.autosummary",    # build the metric tables automatically
    "sphinx.ext.napoleon",       # render numpydoc Parameters / Returns sections
    "sphinx.ext.intersphinx",    # link numpy / pandas / anndata types
    "sphinx.ext.viewcode",       # "[source]" link next to every function
    "sphinx.ext.mathjax",        # the formula in phase_dir
    "myst_parser",               # pages in Markdown, not reStructuredText
    "sphinx_rtd_theme",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# Import-time stand-ins: readthedocs installs the package but not the heavy
# optional extras, and autodoc only needs the module to import.
autodoc_mock_imports = ["scvelo"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable", None),
}

myst_enable_extensions = ["deflist", "colon_fence", "dollarmath"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "veloeval"
html_static_path = []
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "style_external_links": True,
}

nitpick_ignore_regex = [("py:class", r".*")]
