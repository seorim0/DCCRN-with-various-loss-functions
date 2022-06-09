"""
Run the trainer and tester
"""
import torch
from estimate import run_pesq_waveforms_array, cal_stoi
import numpy as np
from scipy.io.wavfile import write as wav_write
from tools_for_model import near_avg_index, max_index, min_index, Bar
from config import fs, info, mode


def model_train(model, optimizer, train_loader, epoch, DEVICE):
    # initialization
    train_loss = 0
    batch_num = 0

    # train
    model.train()
    for inputs, labels in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        labels = labels.float().to(DEVICE)

        _, _, real_spec, img_spec, outputs = model(inputs)
        loss = model.loss(outputs, labels, real_spec, img_spec)
        # loss = model.pmsqe_loss(labels, outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss /= batch_num

    return train_loss


def model_validate(model, validation_loader, dir_to_save, writer, epoch, DEVICE):
    # initialization
    batch_num = 0
    validation_loss = 0
    avg_pesq = 0
    avg_stoi = 0

    all_batch_input = []
    all_batch_label = []
    all_batch_output = []
    all_batch_real_spec = []
    all_batch_img_spec = []
    all_batch_pesq = []

    f_pesq = open(dir_to_save + '/pesq_epoch_' + '%d' % epoch, 'a')
    f_stoi = open(dir_to_save + '/stoi_epoch_' + '%d' % epoch, 'a')

    model.eval()

    with torch.no_grad():
        for inputs, labels in Bar(validation_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            labels = labels.float().to(DEVICE)

            mask_real, mask_imag, real_spec, img_spec, outputs = model(inputs)
            loss = model.loss(outputs, labels, real_spec, img_spec)

            # loss = model.pmsqe_loss(labels, outputs)

            # estimate the output speech with pesq and stoi
            # save pesq & stoi score at each epoch
            estimated_wavs = outputs.cpu().detach().numpy()
            clean_wavs = labels.cpu().detach().numpy()

            pesq = run_pesq_waveforms_array(estimated_wavs, clean_wavs)  ## 98
            stoi = cal_stoi(estimated_wavs, clean_wavs)

            # pesq: 0.1 better / stoi: 0.01 better
            for i in range(len(pesq)):
                f_pesq.write('{:.6f}\n'.format(pesq[i]))
                f_stoi.write('{:.4f}\n'.format(stoi[i]))

            # reshape for sum
            pesq = np.reshape(pesq, (1, -1))
            stoi = np.reshape(stoi, (1, -1))

            avg_pesq += sum(pesq[0]) / len(inputs)
            avg_stoi += sum(stoi[0]) / len(inputs)

            if epoch % 10 == 0:
                # all batch data array
                all_batch_input.extend(inputs)
                all_batch_label.extend(labels)
                all_batch_output.extend(outputs)
                all_batch_real_spec.extend(mask_real)
                all_batch_img_spec.extend(mask_imag)
                all_batch_pesq.extend(pesq[0])

            validation_loss += loss

        # save the samples to tensorboard
        if epoch % 10 == 0:
            all_batch_pesq = np.reshape(all_batch_pesq, (-1, 1))

            # find the best & worst pesq model
            max_pesq_index = max_index(all_batch_pesq)
            min_pesq_index = min_index(all_batch_pesq)

            # find the avg pesq model
            avg_pesq_index = near_avg_index(all_batch_pesq)

            # save the samples to tensorboard
            # the best pesq
            writer.save_samples_we_want('max_pesq', all_batch_input[max_pesq_index], all_batch_label[max_pesq_index],
                                             all_batch_output[max_pesq_index], epoch)
            # the worst pesq
            writer.save_samples_we_want('min_pesq', all_batch_input[min_pesq_index], all_batch_label[min_pesq_index],
                                             all_batch_output[min_pesq_index], epoch)
            # the avg pesq
            writer.save_samples_we_want('avg_pesq', all_batch_input[avg_pesq_index], all_batch_label[avg_pesq_index],
                                             all_batch_output[avg_pesq_index], epoch)

            # save the same sample
            clip_num = 10
            writer.save_samples_we_want('n{}_sample'.format(clip_num), all_batch_input[clip_num], all_batch_label[clip_num],
                                        all_batch_output[clip_num], epoch)

        validation_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

        # save average score
        f_pesq.write('Avg: {:.6f}\n'.format(avg_pesq))
        f_stoi.write('Avg: {:.4f}\n'.format(avg_stoi))

    f_pesq.close()
    f_stoi.close()
    return validation_loss, avg_pesq, avg_stoi


def model_test(noise_type, snr, model, test_loader, dir_to_save, DEVICE):
    model.eval()
    with torch.no_grad():
        # initialization
        batch_num = 0
        test_loss = 0
        avg_pesq = 0
        avg_stoi = 0

        all_batch_input = []
        all_batch_label = []
        all_batch_output = []
        all_batch_real_spec = []
        all_batch_img_spec = []
        all_batch_pesq = []

        # f_pesq = open(dir_to_save + '/test_pesq_epoch{}_{}_{}dB'
        #                 .format(min_index + 1, noise_type, snr), 'a')
        # f_stoi = open(dir_to_save + '/test_stoi_epoch{}_{}_{}dB'
        #                 .format(min_index + 1, noise_type, snr), 'a')
        for inputs, labels in Bar(test_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            labels = labels.float().to(DEVICE)

            mask_real, mask_imag, real_spec, img_spec, outputs = model(inputs)
            loss = model.loss(outputs, labels, real_spec, img_spec)
            # loss = model.pmsqe_loss(labels, outputs)
            # estimate the output speech with pesq and stoi
            # save pesq & stoi score at each epoch
            # [18480, 1]
            estimated_wavs = outputs.cpu().detach().numpy()
            clean_wavs = labels.cpu().detach().numpy()

            pesq = run_pesq_waveforms_array(estimated_wavs, clean_wavs)
            stoi = cal_stoi(estimated_wavs, clean_wavs)

            # # pesq: 0.1 better / stoi: 0.01 better
            # for i in range(len(pesq)):
            #     f_pesq.write('{:.6f}\n'.format(pesq[i]))
            #     f_stoi.write('{:.4f}\n'.format(stoi[i]))

            test_loss += loss

            # reshape for sum
            pesq = np.reshape(pesq, (1, -1))
            stoi = np.reshape(stoi, (1, -1))

            avg_pesq += sum(pesq[0]) / len(inputs)
            avg_stoi += sum(stoi[0]) / len(inputs)

            # all batch data array
            all_batch_input.extend(inputs)
            all_batch_label.extend(labels)
            all_batch_output.extend(outputs)
            all_batch_real_spec.extend(mask_real)
            all_batch_img_spec.extend(mask_imag)
            all_batch_pesq.extend(pesq[0])

        # find the best & worst pesq model
        max_pesq_index = all_batch_pesq.index(max(all_batch_pesq))
        min_pesq_index = all_batch_pesq.index(min(all_batch_pesq))

        test_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

        max_pesq = all_batch_pesq[max_pesq_index]
        min_pesq = all_batch_pesq[min_pesq_index]

        # save average score
        # f_pesq.write('Max: {:.6f} | Min: {:.6f} | Avg: {:.6f}\n'.format(max_pesq, min_pesq, avg_pesq))
        # f_stoi.write('Avg: {:.4f}\n'.format(avg_stoi))
        # f_pesq.close()
        # f_stoi.close()
    return test_loss, avg_pesq, avg_stoi
