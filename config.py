##################################################
# Config
##################################################

##################################################
# Training Config
##################################################

# model parameter
batch_size = 32  # batch size
workers = 4  # number of Dataloader workers

# traning parameter
crop_size = 48
upscale_factor = 3

epochs = 30000

adam_lr = 5e-4
sgd_lr = 1e-4

beta1 = 0.5  # for adam

T_0 = 5000
T_mult = 2

sgd_eta_min = 1e-5
adam_eta_min = 5e-5
##################################################
# Testing Config
##################################################
model_name = "generator.pth"
