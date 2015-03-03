# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 08:45:29 2015

@author: rkmaddox
"""

    import os
    from glob import glob
    from os.path import join

    n_runs = 10
    sub_str = 'ross_pw1v2a_'
    subs = sorted([p for p in os.listdir('.') if
                   os.path.isdir(p) and p[:len(sub_str)] == sub_str])
    steps_all = {}

    for sub in subs:
        fetch_raw = prebads = coreg = fetch_sss = do_score = do_ch_fix = False
        do_ssp = apply_ssp = gen_covs = gen_fwd = gen_inv = write_epochs = False

        # check if raws fetched (+1 is for erm)
        if len(glob(join(sub, 'raw_fif', sub + '*_raw.fif'))) >= n_runs + 1:
            fetch_raw = True

        # check if prebads created
        if os.path.exists(join(sub, 'raw_fif', sub + '_prebad.txt')):
            prebads = True

        # check if coreg has been done
        if os.path.exists(join(sub, 'trans', sub + '-trans.fif')):
            coreg = True

        # check if sss has been fetched (+1 is for erm)
        if len(glob(join(sub, 'sss_fif', sub + '*_raw_sss.fif'))) >= n_runs + 1:
            fetch_sss = True

        # check if scoring has been done
        if len(glob(join(sub, 'lists', 'ALL_' + sub + '*-eve.lst'))) == n_runs:
            do_score = True

        # check if channel orders have been fixed

        # check if SSPs have been generated:
        if len(glob(join(sub, 'sss_pca_fif', 'preproc_*-proj.fif'))) == 4:
            do_ssp = True

        # check if SSPs have been applied:
        if len(glob(join(sub, 'sss_pca_fif', sub +
                         '*allclean_fil*_raw_sss.fif'))) >= n_runs + 1:
            apply_ssp = True

        # check if covariance has been calculated
        if len(glob(join(sub, 'covariance', sub + '*-sss-cov.fif'))) == 2:
            gen_covs = True

        # check if forward solution has been calculated
        if os.path.exists(join(sub, 'forward', sub + '-sss-fwd.fif')):
            gen_fwd = True

        # check if inverses have been calculated
        if len(glob(join(sub, 'inverse', sub + '*-inv.fif'))) == 8:
            gen_inv = True

        # check if epechs have been made
        if len(glob(join(sub, 'epochs', 'All_*' + sub + '*-epo.fif'))) >= 1:
            write_epochs = True

        steps = [
            ['raw fetch', fetch_raw],
            ['prebads', prebads],
            ['coreg', coreg],
            ['sss fetched', fetch_sss],
            ['scored', do_score],
            #['chan fix', do_ch_fix],
            ['gen ssp', do_ssp],
            ['apply ssp', apply_ssp],
            ['gen covs', gen_covs],
            ['gen fwd', gen_fwd],
            ['gen inv', gen_inv],
            ['make ep&ev', write_epochs],
        ]
        steps_all[sub] = steps

    # =========================================================================
    # Print it out in a tabular manner
    # =========================================================================
    n_name_spaces = max([len(s[0]) for s in steps])
    step_names = [s[0] for s in steps]
    n_col_spaces = 3
    ft = '.|'

    # print the headings
    row = ' ' * n_name_spaces
    for sub in subs:
        row += ('%%%ii' % n_col_spaces) % int(sub[-3:])
    print(row)

    # print the statuses
    for si, step_name in enumerate(step_names):
        row = ('%%%is' % n_name_spaces) % step_name
        for sub in subs:
            row += ('%%%is' % n_col_spaces) % ft[steps_all[sub][si][1]]
        row += ' ' + step_name
        print(row)
