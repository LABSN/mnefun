from mne.utils import _check_option

# Copied from KeyToElectrodesAtlasesOther.docx

# Hammers, “IXI” atlas
IXI = """\
1	Hippocampus_right
2	Hippocampus_left
3	Amygdala_right
4	Amygdala_left
5	Anterior_temporal_lobe,_medial_part_right
6	Anterior_temporal_lobe,_medial_part_left
7	Anterior_temporal_lobe,_lateral_part_right
8	Anterior_temporal_lobe,_lateral_part_left
9	Gyri_parahippocampalis_et_ambiens_right
10	Gyri_parahippocampalis_et_ambiens_left
11	Superior_temporal_gyrus,_central_part_right
12	Superior_temporal_gyrus,_central_part_left
13	Medial_and_inferior_temporal_gyri_right
14	Medial_and_inferior_temporal_gyri_left
15	Lateral_occipitotemporal_gyrus_(gyrus_fusiformis)_right
16	Lateral_occipitotemporal_gyrus_(gyrus_fusiformis)_left
17	Cerebellum_right
18	Cerebellum_left
19	Brainstem_(spans_the_midline)
20	Insula_left
21	Insula_right
22	Lateral_remainder_of_occipital_lobe_left
23	Lateral_remainder_of_occipital_lobe_right
24	Cingulate_gyrus,_anterior_(supragenual)_part_left
25	Cingulate_gyrus,_anterior_(supragenual)_part_right
26	Cingulate_gyrus,_posterior_part_left
27	Cingulate_gyrus,_posterior_part_right
28	Middle_frontal_gyrus_left
29	Middle_frontal_gyrus_right
30	Posterior_temporal_lobe_left
31	Posterior_temporal_lobe_right
32	Remainder_of_parietal_lobe_left_(including_supramarginal_and_angular_gyrus)
33	Remainder_of_parietal_lobe_right_(including_supramarginal_and_angular_gyrus)
34	Caudate_nucleus_left
35	Caudate_nucleus_right
36	Nucleus_accumbens_left
37	Nucleus_accumbens_right
38	Putamen_left
39	Putamen_right
40	Thalamus_left
41	Thalamus_right
42	Pallidum_(globus_pallidus)_left
43	Pallidum_(globus_pallidus)_right
44	Corpus_callosum
45	Lateral_ventricle,_frontal_horn,_central_part,_and_occipital_horn_right
46	Lateral_ventricle,_frontal_horn,_central_part,_and_occipital_horn_left
47	Lateral_ventricle,_temporal_horn_right
48	Lateral_ventricle,_temporal_horn_left
49	Third_ventricle
50	Precentral_gyrus_left
51	Precentral_gyrus_right
52	Straight_gyrus_(gyrus_rectus)_left
53	Straight_gyrus_(gyrus_rectus)_right
54	Anterior_orbital_gyrus_left
55	Anterior_orbital_gyrus_right
56	Inferior_frontal_gyrus_left
57	Inferior_frontal_gyrus_right
58	Superior_frontal_gyrus_left
59	Superior_frontal_gyrus_right
60	Postcentral_gyrus_left
61	Postcentral_gyrus_right
62	Superior_parietal_gyrus_left
63	Superior_parietal_gyrus_right
64	Lingual_gyrus_left
65	Lingual_gyrus_right
66	Cuneus_left
67	Cuneus_right
68	Medial_orbital_gyrus_left
69	Medial_orbital_gyrus_right
70	Lateral_orbital_gyrus_left
71	Lateral_orbital_gyrus_right
72	Posterior_orbital_gyrus_left
73	Posterior_orbital_gyrus_right
74	Substantia_nigra_left
75	Substantia_nigra_right
76	Subgenual_anterior_cingulate_gyrus_left
77	Subgenual_anterior_cingulate_gyrus_right
78	Subcallosal_area_left
79	Subcallosal_area_right
80	Pre-subgenual_anterior_cingulate_gyrus_left
81	Pre-subgenual_anterior_cingulate_gyrus_right
82	Superior_temporal_gyrus,_anterior_part_left
83	Superior_temporal_gyrus,_anterior_part_right"""

# Loni Probabilstic Brain Atlas
LBPA40 = """\
21	L_superior_frontal_gyrus
22	R_superior_frontal_gyrus
23	L_middle_frontal_gyrus
24	R_middle_frontal_gyrus
25	L_inferior_frontal_gyrus
26	R_inferior_frontal_gyrus
27	L_precentral_gyrus
28	R_precentral_gyrus
29	L_middle_orbitofrontal_gyrus
30	R_middle_orbitofrontal_gyrus
31	L_lateral_orbitofrontal_gyrus
32	R_lateral_orbitofrontal_gyrus
33	L_gyrus_rectus
34	R_gyrus_rectus
41	L_postcentral_gyrus
42	R_postcentral_gyrus
43	L_superior_parietal_gyrus
44	R_superior_parietal_gyrus
45	L_supramarginal_gyrus
46	R_supramarginal_gyrus
47	L_angular_gyrus
48	R_angular_gyrus
49	L_precuneus
50	R_precuneus
61	L_superior_occipital_gyrus
62	R_superior_occipital_gyrus
63	L_middle_occipital_gyrus
64	R_middle_occipital_gyrus
65	L_inferior_occipital_gyrus
66	R_inferior_occipital_gyrus
67	L_cuneus
68	R_cuneus
81	L_superior_temporal_gyrus
82	R_superior_temporal_gyrus
83	L_middle_temporal_gyrus
84	R_middle_temporal_gyrus
85	L_inferior_temporal_gyrus
86	R_inferior_temporal_gyrus
87	L_parahippocampal_gyrus
88	R_parahippocampal_gyrus
89	L_lingual_gyrus
90	R_lingual_gyrus
91	L_fusiform_gyrus
92	R_fusiform_gyrus
101	L_insular_cortex
102	R_insular_cortex
121	L_cingulate_gyrus
122	R_cingulate_gyrus
161	L_caudate
162	R_caudate
163	L_putamen
164	R_putamen
165	L_hippocampus
166	R_hippocampus
181	cerebellum
182	brainstem"""


def get_atlas_mapping(atlas):
    """Get an atlas mapping.

    Parameters
    ----------
    atlas : str
        The atlas name. Can be "LBPA40" or "IXI".

    Returns
    -------
    mapping : dict
        The mapping from string name to atlas integer.
    """
    _check_option('kind', atlas, ('LBPA40', 'IXI'))
    out = globals()[atlas].split('\n')
    out = {line.split('\t')[1]: int(line.split('\t')[0]) for line in out}
    return out
