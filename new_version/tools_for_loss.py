import torch
import math
import numpy as np
import config as cfg
from asteroid.losses import SingleSrcPMSQE, PITLossWrapper
from asteroid_filterbanks import STFTFB, Encoder


############################################################################
#               for model structure & loss function                        #
############################################################################
L1Loss = torch.nn.L1Loss()


def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True)
    data = data - mean
    return data


def l2_norm(s1, s2):
    # norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    # norm = torch.norm(s1*s2, 1, keepdim=True)

    norm = torch.sum(s1 * s2, -1, keepdim=True)
    return norm


def sdr_linear(s1, s2, eps=1e-8):
    sn = l2_norm(s1, s1)
    sn_m_shn = l2_norm(s1 - s2, s1 - s2)
    sdr_loss = sn**2 / (sn_m_shn**2 + eps)
    return torch.mean(sdr_loss)


def sdr(s1, s2, eps=1e-8):
    sn = l2_norm(s1, s1)
    sn_m_shn = l2_norm(s1 - s2, s1 - s2)
    sdr_loss = 10 * torch.log10(sn**2 / (sn_m_shn**2 + eps))
    return torch.mean(sdr_loss)


def si_snr(s1, s2, eps=1e-8):
    # s1 = remove_dc(s1)
    # s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10((target_norm) / (noise_norm + eps) + eps)
    return torch.mean(snr)


def si_sdr(reference, estimation, eps=1e-8):
    """
        Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
        Args:
            reference: numpy.ndarray, [..., T]
            estimation: numpy.ndarray, [..., T]
        Returns:
            SI-SDR
        [1] SDR– Half- Baked or Well Done?
        http://www.merl.com/publications/docs/TR2019-013.pdf
        >>> np.random.seed(0)
        >>> reference = np.random.randn(100)
        >>> si_sdr(reference, reference)
        inf
        >>> si_sdr(reference, reference * 2)
        inf
        >>> si_sdr(reference, np.flip(reference))
        -25.127672346460717
        >>> si_sdr(reference, reference + np.flip(reference))
        0.481070445785553
        >>> si_sdr(reference, reference + 0.5)
        6.3704606032577304
        >>> si_sdr(reference, reference * 2 + 1)
        6.3704606032577304
        >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
        nan
        >>> si_sdr([reference, reference], [reference * 2 + 1, reference * 1 + 0.5])
        array([6.3704606, 6.3704606])
        :param reference:
        :param estimation:
        :param eps:
        """

    reference_energy = torch.sum(reference ** 2, axis=-1, keepdims=True)

    # This is $\alpha$ after Equation (3) in [1].
    optimal_scaling = torch.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy + eps

    # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * reference

    # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimation - projection

    ratio = torch.sum(projection ** 2, axis=-1) / torch.sum(noise ** 2, axis=-1) + eps

    ratio = torch.mean(ratio)
    return 10 * torch.log10(ratio + eps)


class rmse(torch.nn.Module):
    def __init__(self):
        super(rmse, self).__init__()

    def forward(self, y_true, y_pred):
        mse = torch.mean((y_pred - y_true) ** 2, axis=-1)
        rmse = torch.sqrt(mse + 1e-7)

        return torch.mean(rmse)


# ====================================================================
#  MFCC (Mel Frequency Cepstral Coefficients)
# ====================================================================

# based on a combination of this article:
#     http://practicalcryptography.com/miscellaneous/machine-learning/...
#         guide-mel-frequency-cepstral-coefficients-mfccs/
# and some of this code:
#     http://stackoverflow.com/questions/5835568/...
#         how-to-get-mfcc-from-an-fft-on-a-signal

# conversions between Mel scale and regular frequency scale
def freqToMel(freq):
    return 1127.01048 * math.log(1 + freq / 700.0)


def melToFreq(mel):
    return 700 * (math.exp(mel / 1127.01048) - 1)


# generate Mel filter bank
def melFilterBank(numCoeffs, fftSize=None):
    minHz = 0
    maxHz = cfg.fs / 2  # max Hz by Nyquist theorem
    if (fftSize is None):
        numFFTBins = cfg.win_len
    else:
        numFFTBins = int(fftSize / 2) + 1

    maxMel = freqToMel(maxHz)
    minMel = freqToMel(minHz)

    # we need (numCoeffs + 2) points to create (numCoeffs) filterbanks
    melRange = np.array(range(numCoeffs + 2))
    melRange = melRange.astype(np.float32)

    # create (numCoeffs + 2) points evenly spaced between minMel and maxMel
    melCenterFilters = melRange * (maxMel - minMel) / (numCoeffs + 1) + minMel

    for i in range(numCoeffs + 2):
        # mel domain => frequency domain
        melCenterFilters[i] = melToFreq(melCenterFilters[i])

        # frequency domain => FFT bins
        melCenterFilters[i] = math.floor(numFFTBins * melCenterFilters[i] / maxHz)

    # create matrix of filters (one row is one filter)
    filterMat = np.zeros((numCoeffs, numFFTBins))

    # generate triangular filters (in frequency domain)
    for i in range(1, numCoeffs + 1):
        filter = np.zeros(numFFTBins)

        startRange = int(melCenterFilters[i - 1])
        midRange = int(melCenterFilters[i])
        endRange = int(melCenterFilters[i + 1])

        for j in range(startRange, midRange):
            filter[j] = (float(j) - startRange) / (midRange - startRange)
        for j in range(midRange, endRange):
            filter[j] = 1 - ((float(j) - midRange) / (endRange - midRange))

        filterMat[i - 1] = filter

    # return filterbank as matrix
    return filterMat


# ====================================================================
#  Finally: a perceptual loss function (based on Mel scale)
# ====================================================================
# Set device
DEVICE = torch.device('cuda')

FFT_SIZE = cfg.fft_len

# multi-scale MFCC distance
MEL_SCALES = [16, 32, 64]  # for LMS
# PAM : MEL_SCALES = [32, 64]


# given a (symbolic Theano) array of size M x WINDOW_SIZE
#     this returns an array M x N where each window has been replaced
#     by some perceptual transform (in this case, MFCC coeffs)
def perceptual_transform(x):
    # precompute Mel filterbank: [FFT_SIZE x NUM_MFCC_COEFFS]
    MEL_FILTERBANKS = []
    for scale in MEL_SCALES:
        filterbank_npy = melFilterBank(scale, FFT_SIZE).transpose()
        torch_filterbank_npy = torch.from_numpy(filterbank_npy).type(torch.FloatTensor)
        MEL_FILTERBANKS.append(torch_filterbank_npy.to(DEVICE))

    transforms = []
    # powerSpectrum = torch_dft_mag(x, DFT_REAL, DFT_IMAG)**2

    powerSpectrum = x.view(-1, FFT_SIZE // 2 + 1)
    powerSpectrum = 1.0 / FFT_SIZE * powerSpectrum

    for filterbank in MEL_FILTERBANKS:
        filteredSpectrum = torch.mm(powerSpectrum, filterbank)
        filteredSpectrum = torch.log(filteredSpectrum + 1e-7)
        transforms.append(filteredSpectrum)

    return transforms


# perceptual loss function
class perceptual_distance(torch.nn.Module):

    def __init__(self):
        super(perceptual_distance, self).__init__()

    def forward(self, y_true, y_pred):
        rmse_loss = rmse()
        # y_true = torch.reshape(y_true, (-1, WINDOW_SIZE))
        # y_pred = torch.reshape(y_pred, (-1, WINDOW_SIZE))

        pvec_true = perceptual_transform(y_true)
        pvec_pred = perceptual_transform(y_pred)

        distances = []
        for i in range(0, len(pvec_true)):
            error = rmse_loss(pvec_pred[i], pvec_true[i])
            error = error.unsqueeze(dim=-1)
            distances.append(error)
        distances = torch.cat(distances, axis=-1)

        loss = torch.mean(distances, axis=-1)
        return torch.mean(loss)


get_mel_loss = perceptual_distance()


def get_array_lms_loss(clean_array, est_array):
    array_mel_loss = 0
    for i in range(len(clean_array)):
        mel_loss = get_mel_loss(clean_array[i], est_array[i])
        array_mel_loss += mel_loss

    avg_mel_loss = array_mel_loss / len(clean_array)
    return avg_mel_loss


############################################################################
#                            for pmsqe loss                                #
############################################################################
pmsqe_stft = Encoder(STFTFB(kernel_size=512, n_filters=512, stride=256))
pmsqe = PITLossWrapper(SingleSrcPMSQE(), pit_from='pw_pt')
