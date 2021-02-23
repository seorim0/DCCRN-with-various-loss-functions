"""
DCCRN: Deep complex convolution recurrent network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
from tools_for_model import ConvSTFT, ConviSTFT, \
    ComplexConv2d, ComplexConvTranspose2d, NavieComplexLSTM, complex_cat, ComplexBatchNorm
from tools_for_loss import si_snr, si_sdr, get_array_mel_loss, pmsqe_stft, pmsqe_loss, sdr
from asteroid.filterbanks import transforms


class DCCRN(nn.Module):

    def __init__(
            self,
            rnn_layers=cfg.rnn_layers,
            rnn_units=cfg.rnn_units,
            win_len=cfg.win_len,
            win_inc=cfg.win_inc,
            fft_len=cfg.fft_len,
            win_type=cfg.window_type,
            masking_mode='E',
            use_clstm=False,
            use_cbn=False,
            kernel_size=5,
            kernel_num=[16, 32, 64, 128, 256, 256]
    ):
        '''

            rnn_layers: the number of lstm layers in the crn,
            rnn_units: for clstm, rnn_units = real+imag
        '''

        super(DCCRN, self).__init__()

        # for fft
        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        input_dim = win_len
        output_dim = win_len

        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = rnn_layers
        self.kernel_size = kernel_size
        # self.kernel_num = [2, 8, 16, 32, 128, 128, 128]
        # self.kernel_num = [2, 16, 32, 64, 128, 256, 256]
        self.kernel_num = [2] + kernel_num
        self.masking_mode = masking_mode
        self.use_clstm = use_clstm

        # bidirectional=True
        bidirectional = False
        fac = 2 if bidirectional else 1

        fix = True
        self.fix = fix
        self.stft = ConvSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, fft_len, self.win_type, 'complex', fix=fix)

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                nn.Sequential(
                    # nn.ConstantPad2d([0, 0, 0, 0], 0),
                    ComplexConv2d(
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1),
                        padding=(2, 1)
                    ),
                    nn.BatchNorm2d(self.kernel_num[idx + 1]) if not use_cbn else ComplexBatchNorm(
                        self.kernel_num[idx + 1]),
                    nn.PReLU()
                )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        if self.use_clstm:
            rnns = []
            for idx in range(rnn_layers):
                rnns.append(
                    NavieComplexLSTM(
                        input_size=hidden_dim * self.kernel_num[-1] if idx == 0 else self.rnn_units,
                        hidden_size=self.rnn_units,
                        bidirectional=bidirectional,
                        batch_first=False,
                        projection_dim=hidden_dim * self.kernel_num[-1] if idx == rnn_layers - 1 else None,
                    )
                )
                self.enhance = nn.Sequential(*rnns)
        else:
            self.enhance = nn.LSTM(
                input_size=hidden_dim * self.kernel_num[-1],
                hidden_size=self.rnn_units,
                num_layers=2,
                dropout=0.0,
                bidirectional=bidirectional,
                batch_first=False
            )
            self.tranform = nn.Linear(self.rnn_units * fac, hidden_dim * self.kernel_num[-1])

        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                        nn.BatchNorm2d(self.kernel_num[idx - 1]) if not use_cbn else ComplexBatchNorm(
                            self.kernel_num[idx - 1]),
                        # nn.ELU()
                        nn.PReLU()
                    )
                )
            else:
                self.decoder.append(
                    nn.Sequential(
                        ComplexConvTranspose2d(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        ),
                    )
                )

        self.flatten_parameters()

    def flatten_parameters(self):
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()

    def forward(self, inputs, lens=None):
        specs = self.stft(inputs)
        real = specs[:, :self.fft_len // 2 + 1]
        imag = specs[:, self.fft_len // 2 + 1:]
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        spec_mags = spec_mags

        ##

        ##
        spec_phase = torch.atan2(imag, real)
        spec_phase = spec_phase
        cspecs = torch.stack([real, imag], 1)
        cspecs = cspecs[:, :, 1:]
        '''
        means = torch.mean(cspecs, [1,2,3], keepdim=True)
        std = torch.std(cspecs, [1,2,3], keepdim=True )
        normed_cspecs = (cspecs-means)/(std+1e-8)
        out = normed_cspecs
        '''

        out = cspecs
        encoder_out = []

        for idx, layer in enumerate(self.encoder):
            out = layer(out)
            #    print('encoder', out.size())
            encoder_out.append(out)

        batch_size, channels, dims, lengths = out.size()
        out = out.permute(3, 0, 1, 2)
        if self.use_clstm:
            r_rnn_in = out[:, :, :channels // 2]
            i_rnn_in = out[:, :, channels // 2:]
            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2 * dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2 * dims])

            r_rnn_in, i_rnn_in = self.enhance([r_rnn_in, i_rnn_in])

            r_rnn_in = torch.reshape(r_rnn_in, [lengths, batch_size, channels // 2, dims])
            i_rnn_in = torch.reshape(i_rnn_in, [lengths, batch_size, channels // 2, dims])
            out = torch.cat([r_rnn_in, i_rnn_in], 2)

        else:
            # to [L, B, C, D]
            out = torch.reshape(out, [lengths, batch_size, channels * dims])
            out, _ = self.enhance(out)
            out = self.tranform(out)
            out = torch.reshape(out, [lengths, batch_size, channels, dims])

        out = out.permute(1, 2, 3, 0)

        for idx in range(len(self.decoder)):
            out = complex_cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)
            out = out[..., 1:]
        #    print('decoder', out.size())
        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        mask_real = F.pad(mask_real, [0, 0, 1, 0])
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0])

        if self.masking_mode == 'E':
            mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
            real_phase = mask_real / (mask_mags + 1e-8)
            imag_phase = mask_imag / (mask_mags + 1e-8)
            mask_phase = torch.atan2(
                imag_phase,
                real_phase
            )

            # mask_mags = torch.clamp_(mask_mags,0,100)
            mask_mags = torch.tanh(mask_mags)
            est_mags = mask_mags * spec_mags
            est_phase = spec_phase + mask_phase
            real = est_mags * torch.cos(est_phase)
            imag = est_mags * torch.sin(est_phase)
        elif self.masking_mode == 'C':
            real, imag = real * mask_real - imag * mask_imag, real * mask_imag + imag * mask_real
        elif self.masking_mode == 'R':
            real, imag = real * mask_real, imag * mask_imag

        out_spec = torch.cat([real, imag], 1)
        out_wav = self.istft(out_spec)

        out_wav = torch.squeeze(out_wav, 1)
        # out_wav = torch.tanh(out_wav)
        out_wav = torch.clamp_(out_wav, -1, 1)
        return mask_real, mask_imag, real, imag, out_wav  # out_spec, out_wav

    def get_params(self, weight_decay=0.0):
        # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
            'params': weights,
            'weight_decay': weight_decay,
        }, {
            'params': biases,
            'weight_decay': 0.0,
        }]
        return params

    def loss(self, inputs, labels, real_spec, img_spec, loss_mode=cfg.loss_mode):
        if loss_mode == 'MSE':
            #
            return F.mse_loss(inputs, labels, reduction='mean')

        elif loss_mode == 'SDR':
            return -sdr(labels, inputs)

        elif loss_mode == 'SI-SNR':
            # return -torch.mean(si_snr(inputs, labels))
            return -(si_snr(inputs, labels))

        elif loss_mode == 'SI-SDR':
            # return -torch.mean(si_sdr(inputs, labels))
            return -(si_sdr(labels, inputs))

        elif loss_mode == 'MSE+LMS':

            mse_loss = F.mse_loss(inputs, labels, reduction='mean')

            # for mel loss calculation
            clean_specs = self.stft(labels)
            clean_real = clean_specs[:, :self.fft_len // 2 + 1]
            clean_imag = clean_specs[:, self.fft_len // 2 + 1:]
            clean_mags = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

            est_clean_mags = torch.sqrt(real_spec ** 2 + img_spec ** 2 + 1e-7)
            mel_loss = get_array_mel_loss(clean_mags, est_clean_mags)

            r1 = 1e+3
            r2 = 1
            r = r1 + r2

            loss = (r1 * mse_loss + r2 * mel_loss) / r

            return loss

        elif loss_mode == 'MSE+SI-SNR':
            snr_loss = -(si_snr(inputs, labels))
            mse_loss = F.mse_loss(inputs, labels, reduction='mean')

            r1 = 1
            r2 = 100
            r = r1 + r2

            loss = (r1 * snr_loss + r2 * mse_loss) / r

            return loss

        elif loss_mode == 'MSE+PMSQE':
            ref_wav = labels.reshape(-1, 3, 16000)
            est_wav = inputs.reshape(-1, 3, 16000)
            ref_wav = ref_wav.cpu()
            est_wav = est_wav.cpu()

            ref_spec = transforms.take_mag(pmsqe_stft(ref_wav))
            est_spec = transforms.take_mag(pmsqe_stft(est_wav))

            loss = pmsqe_loss(ref_spec, est_spec)

            loss = loss.cuda()

            return loss

        elif loss_mode == 'SI-SNR+SI-SDR':
            snr_loss = -(si_snr(inputs, labels))
            sdr_loss = -(si_sdr(inputs, labels))

            r1 = 1
            r2 = 1
            r = r1 + r2

            loss = (r1 * snr_loss + r2 * sdr_loss) / r

            return loss

        elif loss_mode == 'SDR+LMS':
            sdr_loss = -sdr(labels, inputs)

            # for mel loss calculation
            clean_specs = self.stft(labels)
            clean_real = clean_specs[:, :self.fft_len // 2 + 1]
            clean_imag = clean_specs[:, self.fft_len // 2 + 1:]
            clean_mags = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

            est_clean_mags = torch.sqrt(real_spec ** 2 + img_spec ** 2 + 1e-7)
            mel_loss = get_array_mel_loss(clean_mags, est_clean_mags)

            r1 = 1
            r2 = 2
            r = r1 + r2

            loss = (r1 * sdr_loss + r2 * mel_loss) / r
            return loss

        elif loss_mode == 'SDR+PMSQE':
            sdr_loss = -sdr(labels, inputs)

            ref_wav = labels.reshape(-1, 3, 16000)
            est_wav = inputs.reshape(-1, 3, 16000)
            ref_wav = ref_wav.cpu()
            est_wav = est_wav.cpu()

            ref_spec = transforms.take_mag(pmsqe_stft(ref_wav))
            est_spec = transforms.take_mag(pmsqe_stft(est_wav))

            # p_loss = pmsqe_loss(ref_spec, est_spec) wrong
            p_loss = pmsqe_loss(est_spec, ref_spec)

            r1 = 1
            r2 = 15
            r = r1 + r2

            loss = (r1 * sdr_loss + r2 * p_loss) / r
            return loss

        elif loss_mode == 'SI-SNR+LMS':
            snr_loss = -(si_snr(inputs, labels))

            # for mel loss calculation
            clean_specs = self.stft(labels)
            clean_real = clean_specs[:, :self.fft_len // 2 + 1]
            clean_imag = clean_specs[:, self.fft_len // 2 + 1:]
            clean_mags = torch.sqrt(clean_real ** 2 + clean_imag ** 2 + 1e-7)

            est_clean_mags = torch.sqrt(real_spec ** 2 + img_spec ** 2 + 1e-7)
            mel_loss = get_array_mel_loss(clean_mags, est_clean_mags)

            r1 = 1
            r2 = 2
            r = r1 + r2

            loss = (r1 * snr_loss + r2 * mel_loss) / r

            return loss

        elif loss_mode == 'SI-SNR+PMSQE':
            ref_wav = labels.reshape(-1, 3, 16000)
            est_wav = inputs.reshape(-1, 3, 16000)
            ref_wav = ref_wav.cpu()
            est_wav = est_wav.cpu()

            ref_spec = transforms.take_mag(pmsqe_stft(ref_wav))
            est_spec = transforms.take_mag(pmsqe_stft(est_wav))

            # p_loss = pmsqe_loss(ref_spec, est_spec) wrong
            p_loss = pmsqe_loss(est_spec, ref_spec)

            snr_loss = -(si_snr(est_wav, ref_wav))

            r1 = 8
            r2 = 1
            r = r1 + r2

            loss = (r1 * p_loss + r2 * snr_loss) / r

            return loss

