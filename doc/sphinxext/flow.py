import os.path as op
from sphinx.util import logging
from mnefun._flow import _create_flowchart
logger = logging.getLogger(__name__)


def setup(app):
    """Setup extension."""
    app.connect('builder-inited', run)
    return {'parallel_read_safe': True,
            'parallel_write_safe': True,
            'version': '0.1'}


def run(app):
    """Run the flowchart generation."""
    fname = op.join(op.dirname(__file__), '..', '_static', 'flow.dot')
    _create_flowchart(fname)
