# -*- coding: utf-8 -*-

from datetime import date
import os.path as op
import sys

import mnefun

sys.path.append(op.dirname(__file__))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.graphviz',
    'sphinxext.flow',
    'sphinx_bootstrap_theme',
    'sphinx_fontawesome',
    'numpydoc',
]
autosummary_generate = True
autodoc_default_options = {'inherited-members': None}
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = u'mnefun'
copyright = u'2013-{0}, mnefun Developers'.format(date.today().year)
version = mnefun.__version__
release = version
exclude_trees = ['_build']
default_role = 'py:obj'
pygments_style = 'manni'
modindex_common_prefix = ['mnefun.']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/devdocs', None),
    'mne': ('https://mne.tools/dev', None),
}

numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {
    'instance', 'of', 'shape', 'ndarray', 'or',
    'n_dip',
}
numpydoc_xref_aliases = {
    # Python
    'ConductorModel': 'mne.bem.ConductorModel',
    'Info': 'mne.Info',
    'Transform': 'mne.transforms.Transform',
    'Projection': 'mne.Projection',
}
html_favicon = "_static/favicon.ico"
html_logo = "_static/ilabs.png"
html_static_path = ['_static']
html_css_files = ['default.css']
html_last_updated_fmt = '%b %d, %Y'
html_use_modindex = False
html_use_index = False
html_show_sourcelink = False
htmlhelp_basename = 'mnefun-doc'
html_theme = 'bootstrap'
html_theme_options = {
    'navbar_title': 'mnefun',  # we replace this with an image
    'navbar_site_name': '',
    'source_link_position': "nav",  # default
    'bootswatch_theme': "lumen",  # yeti paper lumen
    'navbar_sidebarrel': False,  # Render the next/prev links in navbar?
    'navbar_pagenav': False,
    'navbar_class': ("navbar",),
    'bootstrap_version': "3",  # default
    'navbar_links': [
        ("Overview", "overview"),
        ("API reference", "python_reference"),
        ("What's new", "whats_new"),
    ],
}

trim_doctests_flags = True
graphviz_output_format = 'svg'