import pygraphviz as pgv

font_face = 'Arial'
node_size = 9
node_small_size = 11
edge_size = 11
acq_color = ('#AA4499', '#FFFFFF')  # (background color, text color)
sss_color = ('#999933', '#000000')
user_color = ('#332288', '#FFFFFF')
pipe_color = ('#88CCEE', '#000000')

legend = """
<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4" CELLPADDING="4">
<TR><TD BGCOLOR="%s">    </TD><TD>Acquisiton machine</TD></TR>
<TR><TD BGCOLOR="%s">    </TD><TD>SSS machine</TD></TR>
<TR><TD BGCOLOR="%s">    </TD><TD>User-created files</TD></TR>
<TR><TD BGCOLOR="%s">    </TD><TD>Pipeline-created files</TD></TR>
</TABLE>>""" % (acq_color[0], sss_color[0], user_color[0], pipe_color[0])
legend = ''.join(legend.split('\n'))

nodes = dict(
    # Input files (presumed to be available / manually created)
    sco='./\nscore.py',
    bem='structural/bem/\n*-bem-sol.fif',
    pbd='./subj/raw_fif/\n*_prebad.txt',
    bad='./subj/bads/\n*_post-sss.txt',
    tra='./subj/trans/\n*-trans.fif',
    acq='user@minea\n*_raw.fif',

    # Files created by various steps
    mfr='user@kasga:/data00/user/subj/\n*_raw.fif, *_prebad.txt',
    mfs='user@kasga:/data00/user/subj/\n*_raw_sss.fif, *_log.txt',
    cov='./subj/cov/\n*-cov.fif',

    lst='./subj/lists/\nALL_*-eve.lst',
    raw='./subj/raw_fif/\n*_raw.fif',
    sss='./subj/sss_fif/\n*_raw_sss.fif',
    ssl='./subj/sss_log/\n*_log.txt',
    pca='./subj/sss_pca_fif/\n*_raw_sss.fif',
    pro='./subj/sss_pca_fif/\n*-proj.fif',
    evo='./subj/inverse/, ./subj/epochs/\n*-evo.fif, *-epo.fif',
    # epo='./subj/epochs/\nsubj-All-epo.fif',
    src='structural/bem/\n*-oct-6-src.fif',
    fwd='./subj/forward/\n*-fwd.fif',
    inv='./subj/inverse/\n*-inv.fif',
    # htm='./subj/\nsubj.hmtl',
    legend=legend,
)

edges = (
    ('acq', 'raw', '1. fetch_raw'),
    ('sco', 'lst', '2. do_score'),
    ('raw', 'lst'),
    ('raw', 'mfr', '3. push_raw'),
    ('pbd', 'mfr',),
    ('mfr', 'mfs', '4. do_sss'),
    ('mfs', 'sss', '5. fetch_sss'),
    ('mfs', 'ssl'),
    ('sss', 'sss', '6. do_ch_fix'),
    ('bad', 'pro'),
    ('sss', 'pro', '7. gen_ssp'),
    ('bad', 'pca'),
    ('pro', 'pca', '8. apply_ssp'),
    ('sss', 'pca'),
    ('pca', 'evo', '9. write_epochs'),
    ('lst', 'evo'),
    ('pca', 'cov', '10. gen_covs'),
    ('src', 'src'),
    ('bem', 'fwd'),
    ('pca', 'fwd'),
    ('tra', 'fwd', '11. gen_fwd'),
    ('src', 'fwd'),
    ('fwd', 'inv', '12. gen_inv'),
    ('cov', 'inv'),
    # ('evo', 'htm', 'gen_report'),
    # ('cov', 'htm', 'gen_report'),
    # ('bem', 'htm', 'gen_report'),
)

grouped_nodes = [
    [('acq', 'pbd', 'sco', 'tra', 'bad'), user_color],
    [('acq',), acq_color],
    [('mfr', 'mfs'), sss_color],
]
grouped_nodes.append([[node for node in nodes.keys()
                       if not any(node in x[0] for x in grouped_nodes) and
                       node != 'legend'],
                      pipe_color])

g = pgv.AGraph(directed=True)

for key, label in nodes.items():
    label = label.split('\n')
    if len(label) > 1:
        label[0] = '<<FONT POINT-SIZE="%s">' % node_size + label[0] + '</FONT>'
        for li in range(1, len(label)):
            label[li] = ('<FONT POINT-SIZE="%s">' % node_small_size
                         + label[li] + '</FONT>')
        label[-1] = label[-1] + '>'
        label = '<BR/>'.join(label)
    else:
        label = label[0]
    g.add_node(key, label=label)

# Create and customize nodes and edges
for edge in edges:
    g.add_edge(*edge[:2])
    e = g.get_edge(*edge[:2])
    if len(edge) > 2:
        e.attr['label'] = '<<B>%s</B>>' % edge[2]
    e.attr['fontsize'] = edge_size

# Change colors
for these_nodes, (bgcolor, fgcolor) in grouped_nodes:
    for node in these_nodes:
        g.get_node(node).attr['fillcolor'] = bgcolor
        g.get_node(node).attr['style'] = 'filled'
        g.get_node(node).attr['fontcolor'] = fgcolor

# Format (sub)graphs
for gr in g.subgraphs() + [g]:
    for x in [gr.node_attr, gr.edge_attr]:
        x['fontname'] = font_face
        x['fontsize'] = node_size
g.node_attr['shape'] = 'box'
g.get_node('legend').attr.update(shape='plaintext', margin=0)

g.layout('dot')
g.draw('flow.svg', format='svg')
