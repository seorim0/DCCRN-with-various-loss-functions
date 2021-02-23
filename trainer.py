"""
Interface for train
"""

import os
import time
import torch
import shutil
import numpy as np
import config as cfg
from run import model_train, model_validate, model_test
from dataloader import create_dataloader, create_dataloader_for_test
from model import DCCRN, DCUNET, DCCRN_direct, DCCRN_no_skip
from write_on_tensorboard import Writer


###############################################################################
#                        Helper function definition                           #
###############################################################################
# Write training related parameters into the log file.
def write_status_to_log_file(fp, total_parameters):
    fp.write('adsfasdfsdfds')
    fp.write('%d-%d-%d %d:%d:%d\n' %
             (time.localtime().tm_year, time.localtime().tm_mon,
              time.localtime().tm_mday, time.localtime().tm_hour,
              time.localtime().tm_min, time.localtime().tm_sec))
    fp.write('mode                : %s_%s\n' % (cfg.mode, cfg.info))
    fp.write('learning rate       : %g\n' % cfg.learning_rate)
    fp.write('total params   : %d (%.2f M, %.2f MBytes)\n' %
             (total_parameters,
              total_parameters / 1000000.0,
              total_parameters * 4.0 / 1000000.0))


# Calculate the size of total network.
def calculate_total_params(our_model):
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters


###############################################################################
#                          Parameter Initialization                           #
###############################################################################
print('***********************************************************')
print('*    Python library for DNN-based speech enhancement      *')
print('*                        using Pytorch API                *')
print('***********************************************************')

# Set device
DEVICE = torch.device("cuda")

# Set model
if cfg.mode == 'DCCRN':
    model = DCCRN(rnn_units=cfg.rnn_units, masking_mode=cfg.masking_mode, use_clstm=cfg.use_clstm,
                  kernel_num=cfg.kernel_num).to(DEVICE)
elif cfg.mode == 'DCUNET':
    model = DCUNET(masking_mode=cfg.masking_mode, kernel_num=cfg.kernel_num).to(DEVICE)
elif cfg.mode == 'DCCRN_direct':
    model = DCCRN_direct(rnn_units=cfg.rnn_units, use_clstm=cfg.use_clstm, kernel_num=cfg.kernel_num).to(DEVICE)

###############################################################################
#                    Set optimizer and learning rate                          #
###############################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
total_params = calculate_total_params(model)

###############################################################################
#                          Confirm model information                          #
###############################################################################
print('%d-%d-%d %d:%d:%d\n' %
      (time.localtime().tm_year, time.localtime().tm_mon,
       time.localtime().tm_mday, time.localtime().tm_hour,
       time.localtime().tm_min, time.localtime().tm_sec))
print('mode                : %s_%s\n' % (cfg.mode, cfg.info))
print('learning rate       : %g\n' % cfg.learning_rate)
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))

###############################################################################
#                              Create Dataloader                              #
###############################################################################
# Set device
DEVICE = torch.device("cuda")

train_loader = create_dataloader(mode='train')
validation_loader = create_dataloader(mode='valid')

###############################################################################
#                        Set a log file to store progress.                    #
#               Set a hps file to store hyper-parameters information.         #
###############################################################################
# Load the checkpoint
if cfg.chkpt_path is not None:
    print('Resuming from checkpoint: %s' % cfg.chkpt_path)

    # Set a log file to store progress.
    dir_to_save = cfg.job_dir + cfg.chkpt_model
    dir_to_logs = cfg.logs_dir + cfg.chkpt_model

    checkpoint = torch.load(cfg.chkpt_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start_idx = checkpoint['epoch'] + 1
    mse_vali_total = np.load(str(dir_to_save + '/mse_vali_total.npy'))
    if len(mse_vali_total) < cfg.max_epochs:
        plus = cfg.max_epochs - len(mse_vali_total)
        mse_vali_total = np.concatenate((mse_vali_total, np.zeros(plus)), 0)
else:
    print('Starting new training run')
    epoch_start_idx = 1
    mse_vali_total = np.zeros(cfg.max_epochs)

    # Set a log file to store progress.
    dir_to_save = str(cfg.job_dir) + '%d.%d' % (time.localtime().tm_mon, time.localtime().tm_mday) \
                  + '_%s' % cfg.mode + '_%s' % cfg.info
    dir_to_logs = str(cfg.logs_dir) + '%d.%d' % (time.localtime().tm_mon, time.localtime().tm_mday) \
                  + '_%s' % cfg.mode + '_%s' % cfg.info

if not os.path.exists(dir_to_save):
    os.mkdir(dir_to_save)
    os.mkdir(dir_to_logs)

log_fname = str(dir_to_save + '/log.txt')
if not os.path.exists(log_fname):
    fp = open(log_fname, 'w')
    write_status_to_log_file(fp, total_params)
else:
    fp = open(log_fname, 'a')

# Set a hps file to store hyper-parameters information.
hps_fname = str(dir_to_save + '/hp_str.txt')
fp_h = open(hps_fname, 'w')

with open('config.py', 'r') as f:
    hp_str = ''.join(f.readlines())
fp_h.write(hp_str)
fp_h.close()

###############################################################################
###############################################################################
#                             Main program start !!                           #
###############################################################################
###############################################################################

# Writer initialize
writer = Writer(dir_to_logs)

###############################################################################
#                                    Train                                    #
###############################################################################
for epoch in range(epoch_start_idx, cfg.max_epochs + 1):
    start_time = time.time()
    train_loss = model_train(model, optimizer, train_loader, epoch, DEVICE)
    vali_loss, vali_pesq, vali_stoi = model_validate(model, validation_loader,
                                                     dir_to_save, writer, epoch, DEVICE)

    mse_vali_total[epoch - 1] = vali_loss
    np.save(str(dir_to_save + '/mse_vali_total.npy'), mse_vali_total)

    # write the loss on tensorboard
    writer.log_loss(train_loss, vali_loss, epoch)

    # save checkpoint file to resume training
    save_path = str(dir_to_save + '/' + ('chkpt_%d.pt' % epoch))
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

    print('Epoch [{}] | {:.6f} | {:.6} | {:.6} | {:.6} takes {:.2f} seconds'
          .format(epoch, train_loss, vali_loss, vali_pesq, vali_stoi, time.time() - start_time))
    fp.write('Epoch [{}] | {:.6f} | {:.6f} | {:.6f} | {:.6f} takes {:.2f} seconds\n'
             .format(epoch, train_loss, vali_loss, vali_pesq, vali_stoi, time.time() - start_time))

print('Training has been finished.')

# Copy optimum model that has minimum MSE.
print('Save optimum models...')
min_index = np.argmin(mse_vali_total)
print('Minimum validation loss is at '+str(min_index+1)+'.')
