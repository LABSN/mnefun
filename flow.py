import pygraphviz as pgv

font_face = 'OpenSans'
node_size = 12
edge_size = 8

# Input files (presumed to be available / manually created)
sco = 'score.py'
bem = 'SUBJECTS_DIR/struc/bem/struc-5120-5120-5120-bem-sol.fif'
pbd = 'subj/raw_fif/subj_prebad.txt'
bads = 'subj/bads/subj_post-sss.txt'
tra = 'subj/trans/subj-trans.fif'
acq = 'user@minea:/data/somewhere/*_raw.fif'

# Files created by various steps
mfr = 'user@kasga:/data00/user/subj/*_raw.fif'
mfs = 'user@kasga:/data00/user/subj/*_raw_sss.fif'
mfb = 'user@kasga:/data00/user/subj/*_prebad.txt'
mfl = 'user@kasga:/data00/user/subj/*_log.txt'
cov = 'subj/cov/subj[-erm]-cov.fif'

lst = 'subj/lists/ALL_subj-eve.lst'
raw = 'subj/raw_fif/*_raw.fif'
sss = 'subj/sss_fif/*_raw_sss.fif'
ssl = 'subj/sss_log/*_log.txt'
pca = 'subj/sss_pca_fif/*_allclean_fil55_raw_sss.fif'
pro = 'subj/sss_pca_fif/*-proj.fif'
epo = 'subj/epochs/subj-All-epo.fif'
evo = 'subj/inverse/*-ave.fif'
src = 'SUBJECTS_DIR/struc/bem/struc-oct-6-src.fif'
fwd = 'subj/forward/subj-fwd.fif'
inv = 'subj/inverse/subj[-meg][-eeg][-erm]-inv.fif'
htm = 'subj/subj.hmtl'

# steps / processes
fetch_raw = 'fetch_raw'
push_raw = 'push_raw'
do_sss = 'do_sss'
fetch_sss = 'fetch_sss'
do_score = 'do_score'
do_ch_fix = 'do_ch_fix'
gen_ssp = 'gen_ssp'
apply_ssp = 'apply_ssp'
gen_covs = 'gen_covs'
gen_fwd = 'gen_fwd'
gen_inv = 'gen_inv'
write_epochs = 'write_epochs'
gen_report = 'gen_report'

edges = (
    (acq, raw, fetch_raw),
    (raw, mfr, push_raw),
    (pbd, mfb, push_raw),
    (mfr, mfs, do_sss),
    (pbd, mfs, do_sss),
    (mfs, sss, fetch_sss),
    (mfl, ssl, fetch_sss),
    (raw, lst, do_score),
    (sss, sss, do_ch_fix),
    (sss, pro, gen_ssp),
    (pro, pca, apply_ssp),
    (sss, pca, apply_ssp),
    (pca, cov, gen_covs),
    (src, src, gen_fwd),
    (bem, fwd, gen_fwd),
    (pca, fwd, gen_fwd),
    (tra, fwd, gen_fwd),
    (src, fwd, gen_fwd),
    (fwd, inv, gen_inv),
    (cov, inv, gen_inv),
    (pca, evo, write_epochs),
    (lst, evo, write_epochs),
    (pca, epo, write_epochs),
    (lst, epo, write_epochs),
    (epo, htm, gen_report),
    (evo, htm, gen_report),
    (cov, htm, gen_report),
    (bem, htm, gen_report),
)

g = pgv.AGraph(directed=True)
for x in (g.node_attr, g.edge_attr):
    x['fontname'] = font_face
    x['fontsize'] = node_size
g.node_attr['shape'] = 'box'

for edge in edges:
    g.add_edge(*edge[:2])
    e = g.get_edge(*edge[:2])
    if len(edge) > 2:
        e.attr['label'] = edge[2]
    e.attr['fontsize'] = edge_size

g.layout('dot')
g.draw('flow.svg', format='svg')
