import os
import time
import torch
import shutil
import numpy as np
import config as cfg
import itertools
from model import complex_model
from write_on_tensorboard import Writer
from dataloader import create_dataloader
from trainer import model_train, model_validate, model_eval


###############################################################################
#                        Helper function definition                           #
###############################################################################
# Write training related parameters into the log file.
def write_status_to_log_file(fp, total_parameters):
    fp.write('%d-%d-%d %d:%d:%d\n' %
             (time.localtime().tm_year, time.localtime().tm_mon,
              time.localtime().tm_mday, time.localtime().tm_hour,
              time.localtime().tm_min, time.localtime().tm_sec))
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
#         Parameter Initialization and Setting for model training             #
###############################################################################
# Set device
DEVICE = torch.device('cuda')

# Set model
model = complex_model().to(DEVICE)
# Set optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
total_params = calculate_total_params(model)

if cfg.masking_mode == 'Direct(None make)':
    direct = True
else:
    direct = False
###############################################################################
#                        Confirm model information                            #
###############################################################################
print('%d-%d-%d %d:%d:%d\n' %
      (time.localtime().tm_year, time.localtime().tm_mon,
       time.localtime().tm_mday, time.localtime().tm_hour,
       time.localtime().tm_min, time.localtime().tm_sec))
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))

###############################################################################
#                              Create Dataloader                              #
###############################################################################
train_loader = create_dataloader(mode='train')
validation_loader = create_dataloader(mode='valid')

###############################################################################
#                        Set a log file to store progress.                    #
#               Set a hps file to store hyper-parameters information.         #
###############################################################################
if cfg.chkpt_model is not None:  # Load the checkpoint
    print('Resuming from checkpoint: %s' % cfg.chkpt_path)

    # Set a log file to store progress.
    dir_to_save = cfg.job_dir + cfg.chkpt_model
    dir_to_logs = cfg.logs_dir + cfg.chkpt_model

    checkpoint = torch.load(cfg.chkpt_path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start_idx = checkpoint['epoch'] + 1
    mse_vali_total = np.load(str(dir_to_save + '/mse_vali_total.npy'))
    # if the loaded length is shorter than I expected, extend the length
    if len(mse_vali_total) < cfg.max_epochs:
        plus = cfg.max_epochs - len(mse_vali_total)
        mse_vali_total = np.concatenate((mse_vali_total, np.zeros(plus)), 0)
else:  # First learning
    print('Starting new training run...')
    epoch_start_idx = 1
    mse_vali_total = np.zeros(cfg.max_epochs)

    # Set a log file to store progress.
    dir_to_save = cfg.job_dir + cfg.expr_num + '_%d.%d' % (time.localtime().tm_mon,
                                                           time.localtime().tm_mday) + '_%s' % cfg.model + '_%s' % cfg.loss
    dir_to_logs = cfg.logs_dir + cfg.expr_num + '_%d.%d' % (time.localtime().tm_mon,
                                                            time.localtime().tm_mday) + '_%s' % cfg.model + '_%s' % cfg.loss

# make the file directory
if not os.path.exists(dir_to_save):
    os.mkdir(dir_to_save)
    os.mkdir(dir_to_logs)

# logging
log_fname = str(dir_to_save + '/log.txt')
if not os.path.exists(log_fname):
    fp = open(log_fname, 'w')
    write_status_to_log_file(fp, total_params)
else:
    fp = open(log_fname, 'a')

# Set a config file to store hyper-parameters information.
hps_fname = str(dir_to_save + '/config_setting.txt')
if not os.path.exists(hps_fname):
    fp_h = open(hps_fname, 'w')
else:
    fp_h = open(hps_fname, 'a')
    fp_h.write('\n\n')
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
for epoch in range(epoch_start_idx, cfg.max_epochs):
        start_time = time.time()
        # Training
        if cfg.perceptual != 'False':  # The case of using perceptual loss function
            train_loss, train_main_loss, train_perceptual_loss = \
                model_train(model, optimizer, train_loader, direct, DEVICE)
        else:
            train_loss = model_train(model, optimizer, train_loader, direct, DEVICE)

        # save checkpoint file to resume training
        save_path = str(dir_to_save + '/' + ('chkpt_%d.pt' % epoch))
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

        # Validation
        if cfg.perceptual != 'False':  # The case of using perceptual loss function
            vali_loss, vali_main_loss, vali_perceptual_loss = \
                model_validate(model, validation_loader, direct, writer, epoch, DEVICE)
            # write the loss on tensorboard
            writer.log_loss(train_loss, vali_loss, epoch)
            writer.log_perceptual_loss(train_main_loss, train_perceptual_loss,
                                       vali_main_loss, vali_perceptual_loss, epoch)

            print('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                  .format(epoch, train_loss, vali_loss, time.time() - start_time))
            print('             main loss: T {:.6f} V {:.6f} | perceptual loss: T {:.6f} V {:.6f}'
                  .format(train_main_loss, vali_main_loss, train_perceptual_loss, vali_perceptual_loss))
            # log file save
            fp.write('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                     .format(epoch, train_loss, vali_loss, time.time() - start_time))
            fp.write('             main loss: T {:.6f} V {:.6f} | perceptual loss: T {:.6f} V {:.6f}'
                     .format(train_main_loss, vali_main_loss, train_perceptual_loss, vali_perceptual_loss))
        else:
            vali_loss = model_validate(model, validation_loader, direct, writer, epoch, DEVICE)
            # write the loss on tensorboard
            writer.log_loss(train_loss, vali_loss, epoch)

            print('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                  .format(epoch, train_loss, vali_loss, time.time() - start_time))
            # log file save
            fp.write('Epoch [{}] | T {:.6f} | V {:.6} takes {:.2f} seconds\n'
                     .format(epoch, train_loss, vali_loss, time.time() - start_time))

        mse_vali_total[epoch - 1] = vali_loss
        np.save(str(dir_to_save + '/mse_vali_total.npy'), mse_vali_total)

        # It takes quite some time to evaluate the scores for each epoch, so you can control the frequency
        if epoch % 1 == 0:
            vali_pesq, vali_stoi = model_eval(model, validation_loader, direct, dir_to_save, epoch, DEVICE)
            # write the loss on tensorboard per 5 epochs
            writer.log_score(vali_pesq, vali_stoi, epoch)
            print('          | V PESQ: {:.6f} | STOI: {:.6f} '.format(vali_pesq, vali_stoi))
            fp.write('          | V PESQ: {:.6f} | STOI: {:.6f} \n'.format(vali_pesq, vali_stoi))

fp.close()
print('Training has been finished.')

# Copy optimum model that has minimum MSE.
print('Save optimum models...')
min_index = np.argmin(mse_vali_total)
print('Minimum validation loss is at ' + str(min_index + 1) + '.')
src_file = str(dir_to_save + '/' + ('chkpt_%d.pt' % (min_index + 1)))
tgt_file = str(dir_to_save + '/chkpt_opt.pt')
shutil.copy(src_file, tgt_file)
