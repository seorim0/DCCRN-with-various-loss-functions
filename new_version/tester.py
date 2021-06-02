import numpy as np
import tools_for_model as tools
from tools_for_estimate import cal_pesq, cal_stoi, composite
import config as cfg
from scipy.io.wavfile import write as wav_write


#######################################################################
#                           For evaluation                            #
#######################################################################
def model_test(model, validation_loader, noise_type, snr, direct, dir_to_save, epoch, DEVICE):
    # initialize
    batch_num = 0

    avg_pesq = 0
    avg_stoi = 0
    avg_csig = 0
    avg_cbak = 0
    avg_cvol = 0

    all_batch_input = []
    all_batch_target = []
    all_batch_output = []
    all_batch_pesq = []
    all_batch_stoi = []

    # for record the score each samples
    f_score = open(dir_to_save + '/Epoch_' + '%d_SCORES' % epoch, 'a')
    for inputs, targets in tools.Bar(validation_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        _, _, outputs = model(inputs, direct_mapping=direct)

        # estimate the output speech with pesq and stoi
        estimated_wavs = outputs.cpu().detach().numpy()
        clean_wavs = targets.cpu().detach().numpy()

        pesq = cal_pesq(estimated_wavs, clean_wavs)
        stoi = cal_stoi(estimated_wavs, clean_wavs)

        # reshape for sum
        pesq = np.reshape(pesq, (1, -1))
        stoi = np.reshape(stoi, (1, -1))

        # all batch data array
        all_batch_input.extend(inputs)
        all_batch_target.extend(targets)
        all_batch_output.extend(outputs)
        all_batch_pesq.extend(pesq[0])
        all_batch_stoi.extend(stoi[0])

        avg_pesq += sum(pesq[0]) / len(inputs)
        avg_stoi += sum(stoi[0]) / len(inputs)

    estfile_path = './output/{}/{}dB/'.format(noise_type, snr)
    for m in range(len(all_batch_output)):
        est_file_name = '{}dB_{}_est_{}_{:.5}.wav'.format(snr, m + 1, all_batch_pesq[m])

        est_wav = all_batch_output[m].cpu().detach().numpy()
        wav_write(estfile_path+est_file_name, cfg.fs, est_wav)

        noisy_file_name = '{}dB/{}dB_{}_noisy_{:.5}.wav'.format(snr, m + 1, all_batch_input[m])
        noisy_wav = all_batch_input[m].cpu().detach().numpy()
        wav_write(estfile_path+noisy_file_name, cfg.fs, noisy_wav)

        clean_file_name = '{}dB_{}_clean.wav'.format(snr, m + 1)
        clean_wav = all_batch_target[m].cpu().detach().numpy()
        wav_write(estfile_path+clean_file_name, cfg.fs, clean_wav)

        CSIG, CBAK, CVOL, _ = composite(estfile_path+clean_file_name, estfile_path+est_file_name)
        avg_csig += CSIG
        avg_cbak += CBAK
        avg_cvol += CVOL

        # pesq: 0.1 better / stoi: 0.01 better
        f_score.write('PESQ {:.6f} | STOI {:.6f} | CSIG {:.6f} | CBAK {:.6f} | CVOL {:.6f}\n'
                          .format(all_batch_pesq[m], all_batch_stoi[m], CSIG, CBAK, CVOL))

    avg_pesq /= batch_num
    avg_stoi /= batch_num
    avg_csig /= batch_num
    avg_cbak /= batch_num
    avg_cvol /= batch_num
    f_score.close()
    return avg_pesq, avg_stoi, avg_csig, avg_cbak, avg_cvol
