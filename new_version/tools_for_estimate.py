import re
import os
import logging
import oct2py
from pesq import pesq
import numpy as np
import ctypes
from scipy.io import wavfile
from pystoi import stoi
import config as cfg


############################################################################
#                       for plotting the samples                           #
############################################################################
# Reference
# https://github.com/usimarit/semetrics # https://ecs.utdallas.edu/loizou/speech/software.htm
logging.basicConfig(level=logging.ERROR)
oc = oct2py.Oct2Py(logger=logging.getLogger())

COMPOSITE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "composite.m")


def pesq_mos(clean: str, enhanced: str):
    sr1, clean_wav = wavfile.read(clean)
    sr2, enhanced_wav = wavfile.read(enhanced)
    assert sr1 == sr2
    mode = "nb" if sr1 < 16000 else "wb"
    return pesq(sr1, clean_wav, enhanced_wav, mode)


def composite(clean: str, enhanced: str):
    pesq_score = pesq_mos(clean, enhanced)
    csig, cbak, covl, ssnr = oc.feval(COMPOSITE, clean, enhanced, nout=4)
    csig += 0.603 * pesq_score
    cbak += 0.478 * pesq_score
    covl += 0.805 * pesq_score
    return csig, cbak, covl, ssnr


###############################################################################
#                           PESQ (another ref)                                #
###############################################################################
pesq_dll = ctypes.CDLL('./PESQ.so')
pesq_dll.pesq.restype = ctypes.c_double


# interface to PESQ evaluation, taking in two filenames as input
def run_pesq_filenames(clean, to_eval):
    pesq_regex = re.compile("\(MOS-LQO\):  = ([0-9]+\.[0-9]+)")

    pesq_out = os.popen("./PESQ" + cfg.fs + "wb " + clean + " " + to_eval).read()
    regex_result = pesq_regex.search(pesq_out)

    if (regex_result is None):
        return 0.0
    else:
        return float(regex_result.group(1))


def run_pesq_waveforms(dirty_wav, clean_wav):
    clean_wav = clean_wav.astype(np.double)
    dirty_wav = dirty_wav.astype(np.double)
    # return pesq(clean_wav, dirty_wav, fs=8000)
    return pesq_dll.pesq(ctypes.c_void_p(clean_wav.ctypes.data),
                         ctypes.c_void_p(dirty_wav.ctypes.data),
                         len(clean_wav),
                         len(dirty_wav))


# interface to PESQ evaluation, taking in two waveforms as input
def cal_pesq(dirty_wavs, clean_wavs):
    scores = []
    for i in range(len(dirty_wavs)):
        pesq = run_pesq_waveforms(dirty_wavs[i], clean_wavs[i])
        scores.append(pesq)
    return scores


###############################################################################
#                                     STOI                                    #
###############################################################################
def cal_stoi(estimated_speechs, clean_speechs):
    stoi_scores = []
    for i in range(len(estimated_speechs)):
        stoi_score = stoi(clean_speechs[i], estimated_speechs[i], cfg.fs, extended=False)
        stoi_scores.append(stoi_score)
    return stoi_scores
