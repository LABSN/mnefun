"""Create a flow chart for mnefun."""


def _create_flowchart(fname):
    """Run the flow chart generation."""
    import pygraphviz as pgv
    font_face = 'sans-serif'
    node_size = 8
    node_small_size = 10
    edge_size = 10
    margin = '0.1,0.05'
    # (background color, text color)
    acq_color = ('#CCBB44', '#000000')
    user_color = ('#EE6677', '#000000')
    user_opt_color = ('#FFD8DF', '#000000')
    pipe_color = ('#66CCEE', '#000000')

    legend = """
<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="4" CELLPADDING="4">
<TR><TD BGCOLOR="%s">    </TD><TD ALIGN="left">Remote: acquisiton machine</TD></TR>
<TR><TD BGCOLOR="%s">    </TD><TD ALIGN="left">Local: obligatory user-created files</TD></TR>
<TR><TD BGCOLOR="%s">    </TD><TD ALIGN="left">Local: optional user-created files</TD></TR>
<TR><TD BGCOLOR="%s">    </TD><TD ALIGN="left">Local: pipeline-created files</TD></TR>
</TABLE>>""" % (acq_color[0], user_color[0], user_opt_color[0], pipe_color[0])  # noqa
    legend = ''.join(legend.split('\n'))

    # XXX the SVG mouseovers for these are the dict keys. We should make
    # human-readable at some point, probably by swapping in names during
    # node creation.
    nodes = dict(
        # Input files (presumed to be available / manually created)
        sco='./\nscore.py',
        mri='structural/mri/\nT1.mgz',
        bem='structural/bem/\n*-bem-sol.fif',
        pbd='params.mf_prebad[SUBJ]',
        bad='./SUBJ/bads/\n*_post-sss.txt',
        tra='./SUBJ/trans/\n*-trans.fif',
        acq='user@minea\n*_raw.fif',
        can='./SUBJ/raw_fif/\n*-custom-annot.fif',
        pex='./SUBJ/sss_pca_fif/\n*-extra-proj.fif',

        # Files created by various steps
        mfb='./SUBJ/raw_fif/\n*_raw_maxbad.txt',
        mfp='./SUBJ/raw_fif/\n*_raw.pos',
        aan='./SUBJ/raw_fif/\n*-annot.fif',
        cov='./SUBJ/cov/\n*-cov.fif',
        lst='./SUBJ/lists/\nALL_*-eve.lst',
        raw='./SUBJ/raw_fif/\n*_raw.fif',
        otp='./SUBJ/otp_fif/\n*_raw_otp.fif',
        sss='./SUBJ/sss_fif/\n*_raw_sss.fif',
        pca='./SUBJ/sss_pca_fif/\n*_fil55_raw_sss.fif',
        pro='./SUBJ/sss_pca_fif/\n*-proj.fif',
        evo='./SUBJ/inverse/, ./SUBJ/epochs/\n*-evo.fif, *-epo.fif',

        # epo='./SUBJ/epochs/\nsubj-All-epo.fif',
        src='structural/bem/\n*-src.fif',
        fwd='./SUBJ/forward/\n*-fwd.fif',
        inv='./SUBJ/inverse/\n*-inv.fif',
        htm='./SUBJ/\nsubj_fil*_report.html',
        legend=legend,
    )

    edges = (
        ('mri', 'bem', 'Freesurfer'),
        ('mri', 'src'),
        ('mri', 'tra', 'mne coreg'),
        ('acq', 'raw', '1. fetch_raw'),
        ('raw', 'otp', '2. do_otp'),
        ('otp', 'sco'),
        ('sco', 'lst', '3. do_score'),
        ('raw', 'mfb'),
        ('otp', 'pbd'),
        ('pbd', 'mfb', '4. do_sss'),
        ('mfb', 'mfp'),
        ('pbd', 'mfp'),
        ('mfp', 'sss'),
        ('mfp', 'aan'),
        ('otp', 'sss'),
        ('pbd', 'sss'),
        ('mfb', 'sss'),
        ('aan', 'sss'),
        ('otp', 'can'),
        ('can', 'sss'),
        ('sss', 'sss', '5. do_ch_fix'),
        ('bad', 'pro'),
        ('pex', 'pro'),
        ('sss', 'pro', '6. gen_ssp'),
        ('sss', 'bad'),
        ('sss', 'pex'),
        ('bad', 'pca'),
        ('pex', 'pca'),
        ('pro', 'pca', '7. apply_ssp'),
        ('sss', 'pca'),
        ('pca', 'evo', '8. write_epochs'),
        ('lst', 'evo'),
        ('pca', 'cov', '9. gen_covs'),
        ('bem', 'fwd'),
        ('pca', 'fwd'),
        ('tra', 'fwd', '10. gen_fwd'),
        ('src', 'fwd'),
        ('fwd', 'inv', '11. gen_inv'),
        ('cov', 'inv'),
        ('evo', 'htm', '12. gen_report'),
        ('cov', 'htm'),
        ('inv', 'htm'),
    )

    grouped_nodes = [
        [('acq', 'pbd', 'sco', 'tra', 'bad'), user_color],
        [('can', 'pex'), user_opt_color],
        [('acq',), acq_color],
    ]
    grouped_nodes.append([[node for node in nodes.keys()
                           if not any(node in x[0] for x in grouped_nodes) and
                           node != 'legend'], pipe_color])

    g = pgv.AGraph(
        name='mnefun flow diagram', directed=True,
        ranksep=0.1, nodesep=0.2, bgcolor='#00000000',
    )
    for key, label in nodes.items():
        label = label.split('\n')
        if len(label) > 1:
            label[0] = \
                '<<FONT POINT-SIZE="%s">' % node_size + label[0] + '</FONT>'
            for li in range(1, len(label)):
                label[li] = ('<FONT POINT-SIZE="%s">' % node_small_size +
                             label[li] + '</FONT>')
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
            e.attr['label'] = edge[2]
            e.attr['labeltooltip'] = edge[2]
            if edge[2][0].isnumeric():
                e.attr['label'] = '<<B>%s</B>>' % (e.attr['label'],)
                anchor = edge[2].split(' ')[-1].replace('_', '-')
                e.attr['URL'] = '../overview.html#' + anchor
                e.attr['target'] = '_top'
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
    g.node_attr['margin'] = margin
    g.get_node('legend').attr.update(shape='plaintext', margin=0)
    g.add_subgraph(['legend', 'htm'], rank='same')
    # g.add_subgraph(['acq', 'raw'], rank='same')
    # g.add_subgraph(['eve', 'aan'], rank='same')
    g.layout('dot')
    g.write(fname)
