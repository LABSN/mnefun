import os
import os.path as op
import subprocess
import time
from mne import pick_types, read_raw_fif
from ._paths import get_raw_fnames
from mne.preprocessing import oversampled_temporal_projection
from ._sss import _maxbad, _read_raw_prebad, _load_meg_bads, _python_autobads

def run_otp(p, subjects, run_indices):
    """ Run Oversampled Temporal Projection (OTP) on raw data.
    """
    
    assert isinstance(p.otp_dur, float) and p.tsss_dur > 0
    duration = p.otp_dur
    if p.otp_dur is not None:
        assert p.otp_dur == p.tsss_dur

    for si, subj in enumerate(subjects):
        if p.disp_files:
            print('    Denoising subject %g/%g (%s).'
                  % (si + 1, len(subjects), subj))
        # locate raw files with splits
        otp_dir = op.join(p.work_dir, subj, p.otp_dir)
        if not op.isdir(otp_dir):
            os.mkdir(otp_dir)
        raw_files = get_raw_fnames(p, subj, 'raw', erm=False,
                                   run_indices=run_indices[si])
        raw_files_out = get_raw_fnames(p, subj, 'otp', erm=False,
                                       run_indices=run_indices[si])
        erm_files = get_raw_fnames(p, subj, 'raw', 'only')
        erm_files_out = get_raw_fnames(p, subj, 'otp', 'only')

        #  process raw files
        assert len(raw_files) > 0
        for ii, (r, o) in enumerate(zip(raw_files + erm_files,
                                        raw_files_out + erm_files_out)):
            if not op.isfile(r):
                raise NameError('File not found (' + r + ')')
            raw = _read_raw_prebad(p, subj, r, disp=True).load_data()

            # apply maxwell filter
            t0 = time.time()
            print('        Running OTP ...', end='')
            raw_otp = oversampled_temporal_projection(
                raw, duration=duration)
            print('%i sec' % (time.time() - t0,))
            raw_otp.save(o, overwrite=True, buffer_size_sec=None)
