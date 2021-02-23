"""
Configuration for program
"""

# model
mode = 'DCCRN'  # DCUNET / DCCRN
info = 'SDR+PMSQE'

test = True

# path
job_dir = './job/'
logs_dir = './logs/'
chkpt_model = '2.17_DCCRN_SDR+PMSQE/'
chkpt_path = job_dir + chkpt_model + 'chkpt_88.pt'
#chkpt_path = None

# model information
fs = 16000
win_len = 400
win_inc = 100
ola_ratio = win_inc / win_len
fft_len = 512
sam_sec = fft_len / fs
frm_samp = fs * (fft_len / fs)
window_type = 'hanning'

rnn_layers = 2
rnn_units = 256
masking_mode = 'E'
use_clstm = True
kernel_num = [32, 64, 128, 256, 256, 256]  # DCCRN
#kernel_num = [72, 72, 144, 144, 144, 160, 160, 180]  # DCUNET
loss_mode = 'SDR+PMSQE'

# hyperparameters for model train
max_epochs = 100
learning_rate = 0.0005
batch = 15
