import os

# Fix OPENLBAS threads to 1
NP_THREAD_VARS = ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OPM_NUM_THREADS']
for ev in NP_THREAD_VARS:
    os.environ[ev] = '1'
