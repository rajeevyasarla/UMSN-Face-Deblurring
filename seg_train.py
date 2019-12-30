from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

from misc import *
import models.face_fed as net



from myutils.vgg16 import Vgg16
from myutils import utils
import pdb
import torch.nn.functional as F
#from PIL import Image
from torchvision import transforms

import h5py
from os import listdir
from os.path import isfile, join

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False,
  default='pix2pix_class',  help='')
parser.add_argument('--dataroot', required=False,
  default='', help='path to trn dataset')
parser.add_argument('--valDataroot', required=False,
  default='', help='path to val dataset')
parser.add_argument('--mode', type=str, default='B2A', help='B2A: facade, A2B: edges2shoes')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--valBatchSize', type=int, default=120, help='input batch size')
parser.add_argument('--originalSize', type=int,
  default=160, help='the height / width of the original input image')
parser.add_argument('--imageSize', type=int,
  default=128, help='the height / width of the cropped input image to network')
parser.add_argument('--inputChannelSize', type=int,
  default=3, help='size of the input channels')
parser.add_argument('--outputChannelSize', type=int,
  default=3, help='size of the output channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=5000, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--annealStart', type=int, default=0, help='annealing learning rate start to')
parser.add_argument('--annealEvery', type=int, default=1000, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaGAN', type=float, default=0.01, help='lambdaGAN')
parser.add_argument('--lambdaIMG', type=float, default=3.2, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--modeclean', type=int,default= 1, help='segmentation network training mode, by it is default trained using clean images')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args()
print(opt)

create_exp_dir(opt.exp)
opt.manualSeed = random.randint(1, 10000)
# opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

# get dataloader
opt.dataset='pix2pix_val'
print (opt.dataroot)
dataloader = getLoader(opt.dataset,
                       opt.dataroot,
                       opt.originalSize,
                       opt.imageSize,
                       opt.batchSize,
                       opt.workers,
                       mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                       split='train',
                       shuffle=True,
                       seed=opt.manualSeed)

opt.dataset='pix2pix_val'
valDataloader = getLoader(opt.dataset,
                          opt.valDataroot,
                          opt.originalSize,
                          opt.imageSize,
                          opt.valBatchSize,
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='val',
                          shuffle=False,
                          seed=opt.manualSeed)


# get logger
trainLogger = open('%s/train.log' % opt.exp, 'w')

def gradient(y):
    gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_y=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

    return gradient_h, gradient_y


ngf = opt.ngf
ndf = opt.ndf
inputChannelSize = opt.inputChannelSize
outputChannelSize= opt.outputChannelSize

# get models
# netG = net.G(inputChannelSize, outputChannelSize, ngf)
netG=net.Segmentation()

if opt.netG != '':
  netG.load_state_dict(torch.load(opt.netG))
print(netG)
#netG.load_state_dict(torch.load('./segs_faceblr/SMaps_60.pth'))
from scipy import signal
import h5py
from scipy import signal
import random
k_filename ='./kernel.mat'
kfp = h5py.File(k_filename)
kernels = np.array(kfp['kernels'])
kernels = kernels.transpose([0,2,1])


netG.train()
criterionCAE = nn.BCELoss()

target= torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
target_128= torch.FloatTensor(opt.batchSize, outputChannelSize, (opt.imageSize//4), (opt.imageSize//4))
input_128 = torch.FloatTensor(opt.batchSize, inputChannelSize, (opt.imageSize//4), (opt.imageSize//4))
target_256= torch.FloatTensor(opt.batchSize, outputChannelSize, (opt.imageSize//2), (opt.imageSize//2))
input_256 = torch.FloatTensor(opt.batchSize, inputChannelSize, (opt.imageSize//2), (opt.imageSize//2))




val_target= torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_target_128= torch.FloatTensor(opt.batchSize, outputChannelSize, (opt.imageSize//4), (opt.imageSize//4))
val_input_128 = torch.FloatTensor(opt.batchSize, inputChannelSize, (opt.imageSize//4), (opt.imageSize//4))
val_target_256= torch.FloatTensor(opt.batchSize, outputChannelSize, (opt.imageSize//2), (opt.imageSize//2))
val_input_256 = torch.FloatTensor(opt.batchSize, inputChannelSize, (opt.imageSize//2), (opt.imageSize//2))
label_d = torch.FloatTensor(opt.batchSize)


target = torch.FloatTensor(opt.batchSize, outputChannelSize, opt.imageSize, opt.imageSize)
input = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
depth = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)
ato = torch.FloatTensor(opt.batchSize, inputChannelSize, opt.imageSize, opt.imageSize)


val_target = torch.FloatTensor(opt.valBatchSize, outputChannelSize, opt.imageSize, opt.imageSize)
val_input = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_depth = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)
val_ato = torch.FloatTensor(opt.valBatchSize, inputChannelSize, opt.imageSize, opt.imageSize)




# NOTE: size of 2D output maps in the discriminator
sizePatchGAN = 30
real_label = 1
fake_label = 0

# image pool storing previously generated samples from G
imagePool = ImagePool(opt.poolSize)

# NOTE weight for L_cGAN and L_L1 (i.e. Eq.(4) in the paper)
lambdaGAN = opt.lambdaGAN
lambdaIMG = opt.lambdaIMG

netG.cuda()
criterionCAE.cuda()




target, input, depth, ato = target.cuda(), input.cuda(), depth.cuda(), ato.cuda()
val_target, val_input, val_depth, val_ato = val_target.cuda(), val_input.cuda(), val_depth.cuda(), val_ato.cuda()

target = Variable(target)
input = Variable(input)

target_128, input_128 = target_128.cuda(), input_128.cuda()
val_target_128, val_input_128 = val_target_128.cuda(), val_input_128.cuda()
target_256, input_256 = target_256.cuda(), input_256.cuda()
val_target_256, val_input_256 = val_target_256.cuda(), val_input_256.cuda()

target_128 = Variable(target_128)
input_128 = Variable(input_128)
target_256 = Variable(target_256)
input_256 = Variable(input_256)
# input = Variable(input,requires_grad=False)
# depth = Variable(depth)
ato = Variable(ato)

# Initialize VGG-16
vgg = Vgg16()
utils.init_vgg16('./models/')
vgg.load_state_dict(torch.load(os.path.join('./models/', "vgg16.weight")))
vgg.cuda()


label_d = Variable(label_d.cuda())

# get randomly sampled validation images and save it
print(len(dataloader))
val_iter = iter(valDataloader)
data_val = val_iter.next()


val_target_cpu, val_input_cpu = data_val


val_target_cpu, val_input_cpu = val_target_cpu.float().cuda(), val_input_cpu.float().cuda()



val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)

vutils.save_image(val_target, '%s/real_target.png' % opt.exp, normalize=True)
vutils.save_image(val_input, '%s/real_input.png' % opt.exp, normalize=True)

# pdb.set_trace()
# get optimizer
optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=0.00005)
# NOTE training loop
ganIterations = 0
count = 1
Best_Fs = 0
Best_epoch = 0
if opt.modeclean == 1:
  Num_rn = 0
else:
  Num_rn = 34
for epoch in range(1000):
    if epoch%60 == 0 and epoch>0:
        opt.lrG = opt.lrG/1.25
        for param_group in optimizerG.param_groups:
            param_group['lr'] = opt.lrG
    if epoch == 200:
        opt.lrG = 0.000001
        for param_group in optimizerG.param_groups:
            param_group['lr'] = opt.lrG
    if epoch >= opt.annealStart:
        adjust_learning_rate(optimizerG, opt.lrG, epoch, None, opt.annealEvery)
    for set_i in range(1,5):
        cl_filename ='./seg_mat_files/Images_m%d.mat'%set_i
        sm_filename ='./seg_mat_files/Labels_m%d.mat'%set_i
        fcl = h5py.File(cl_filename)
        fsm = h5py.File(sm_filename)
        clean_images = np.array(fcl['Images'])
        sm_maps = np.array(fsm['Labels'])
        #print(clean_images.shape)

        clean_images = clean_images.transpose([0,1,3,2])
        sm_maps = sm_maps.transpose([0,1,3,2])
        
        
        
        clean_images = clean_images/255.0
        clean_images = (clean_images-0.5)/0.5
        sm_maps = sm_maps/(1*255.0)
        sm_maps[sm_maps>0.49] = 1
        sm_maps[sm_maps<=0.49] = 0
        for i in range(50):
            sm_map = sm_maps[i*10:i*10+10,:,:,:]
            cl_image = clean_images[i*10:i*10+10,:,:,:]
            #print(np.amax(sm_map))
            #print(sm_map.shape)
            cl_image = np.reshape(cl_image,[10,3,160,160])
            sm_map = np.reshape(sm_map,[10,4,160,160])
            input_cpu = Variable(torch.from_numpy(cl_image))
            target_cpu = Variable(torch.from_numpy(sm_map))
            x1 = int((160-opt.imageSize)/2)
            y1 = int((160-opt.imageSize)/2)
            input_cpu = input_cpu.numpy()
            target_cpu = target_cpu.numpy()
            if (random.randint(0,100)<Num_rn) :
                
                for j in range(10):
                    index = random.randint(0,24000)
                    for k in range(4):
                        target_cpu[j,k,:,:]= signal.convolve(target_cpu[j,k,:,:],kernels[index,:,:],mode='same')
                    input_cpu[j,0,:,:]= signal.convolve(input_cpu[j,0,:,:],kernels[index,:,:],mode='same')
                    input_cpu[j,1,:,:]= signal.convolve(input_cpu[j,1,:,:],kernels[index,:,:],mode='same')
                    input_cpu[j,2,:,:]= signal.convolve(input_cpu[j,2,:,:],kernels[index,:,:],mode='same')
            #input_cpu = input_cpu + (1.0/255.0)* np.random.normal(0,4,input_cpu.shape)
            #input_cpu = input_cpu[:,:,x1:x1+opt.imageSize,y1:y1+opt.imageSize]
            #target_cpu = target_cpu[:,:,x1:x1+opt.imageSize,y1:y1+opt.imageSize]
                
            input_cpu = input_cpu[:,:,x1:x1+opt.imageSize,y1:y1+opt.imageSize]
            target_cpu = target_cpu[:,:,x1:x1+opt.imageSize,y1:y1+opt.imageSize]
            target_cpu[target_cpu>0.49]=1
            target_cpu[target_cpu<=0.49]=0
            input_cpu = torch.from_numpy(input_cpu)
            target_cpu = torch.from_numpy(target_cpu)
            target_cpu, input_cpu = target_cpu.float().cuda(), input_cpu.float().cuda()
            target.data.resize_as_(target_cpu).copy_(target_cpu)
            input.data.resize_as_(input_cpu).copy_(input_cpu)
            input_256 = torch.nn.functional.interpolate(input,scale_factor=0.5)
            target_256 = torch.nn.functional.interpolate(target,scale_factor=0.5)
            
            x_hat, x_hat64 = netG(input,input_256)
            #print(x_hat.size())
            netG.zero_grad()
            
            L_img_ = criterionCAE(x_hat, target) + 0.5*criterionCAE(x_hat64, target_256)
            
            L_img = lambdaIMG * L_img_

            if lambdaIMG != 0:
                L_img.backward(retain_graph=True)
                #print("came")
                
            optimizerG.step()
            ganIterations += 1
            if ganIterations % (10*opt.display) == 0:
                print('[%d/%d][%d/%d] L_D: %f L_img: %f L_G: %f D(x): %f D(G(z)): %f / %f'
                    % (epoch, opt.niter, 10*i, 2000,
                    L_img.data[0], L_img.data[0], L_img.data[0], L_img.data[0], L_img.data[0], L_img.data[0]))
            if ganIterations % (40*opt.display) == 0:
                val_batch_output = torch.zeros([16,3,128,128], dtype=torch.float32)#torch.FloatTensor([10,3,128,128]).fill_(0)
                Fs = 0
                for idx in range(33):
                    tcl_filename ='./seg_mat_files/Images_test.mat'
                    tsm_filename ='./seg_mat_files/Labels_test.mat'
                    tfcl = h5py.File(tcl_filename)
                    tfsm = h5py.File(tsm_filename)
                    tclean_images = np.array(tfcl['Images'])
                    tsm_maps = np.array(tfsm['Labels'])
                    #print(clean_images.shape)

                    tclean_images = tclean_images.transpose([0,1,3,2])
                    tsm_maps = tsm_maps.transpose([0,1,3,2])
                    
                    
                    tclean_images = tclean_images/255.0
                    tclean_images = (tclean_images-0.5)/0.5
                    tsm_maps = tsm_maps/(1*255.0)
                    tsm_maps[tsm_maps>0.49] = 1
                    tsm_maps[tsm_maps<=0.49] = 0
                    x1 = int((160-opt.imageSize)/2)
                    y1 = int((160-opt.imageSize)/2)
                    # print(tclean_images.shape,x1,y1,opt.imageSize)
                    single_img = tclean_images[idx*10:idx*10+10,:,:,:]
                    single_img = np.reshape(single_img,[10,3,160,160])#val_input[idx,:,:,:].unsqueeze(0)
                    val_inputv = Variable(torch.from_numpy(single_img), volatile=True)
                    yval = tsm_maps[idx*10:idx*10+10,:,:,:]
                    yval = np.reshape(yval,[10,4,160,160])#val_input[idx,:,:,:].unsqueeze(0)
                    
                    # if epoch>=0:
                    #     index = idx+24500
                    #     val_inputv = val_inputv.cpu().numpy()
                    #     val_inputv[0,0,:,:]= signal.convolve(val_inputv[0,0,:,:],kernels[index,:,:],mode='same')
                    #     val_inputv[0,1,:,:]= signal.convolve(val_inputv[0,1,:,:],kernels[index,:,:],mode='same')
                    #     val_inputv[0,2,:,:]= signal.convolve(val_inputv[0,2,:,:],kernels[index,:,:],mode='same')
                    #     val_inputv = val_inputv[:,:,x1:x1+opt.imageSize,y1:y1+opt.imageSize]
                    #     val_inputv = val_inputv + (1.0/255.0)* np.random.normal(0,4,val_inputv.shape)
                    #     val_inputv = torch.from_numpy(val_inputv)
                    #     val_inputv = val_inputv.float().cuda()
                    # else :
                    val_inputv = val_inputv[:,:,x1:x1+opt.imageSize,y1:y1+opt.imageSize]
                    yval = yval[:,:,x1:x1+opt.imageSize,y1:y1+opt.imageSize]
                    val_inputv = val_inputv.float().cuda()
                    val_inputv_256 = torch.nn.functional.interpolate(val_inputv,scale_factor=0.5)
                    with torch.no_grad():
                        x_hat_val,x_hat_val64 = netG(val_inputv,val_inputv_256)
                        
                        val_batch_output[idx%16,:,:,:].copy_(x_hat_val.data[0,idx%4,:,:])
                        # print(val_inputv.size(),yval.min(),yval.max(),x_hat_val.min(),x_hat_val.max())                    
                        x_hat_val = x_hat_val.cpu().numpy()
                        x_hat_val[x_hat_val>0.5] = 1
                        x_hat_val[x_hat_val<=0.5] = 0
                        # print(val_inputv.size(),yval.min(),yval.max(),x_hat_val.min(),x_hat_val.max())
                        for jdx in range(10):
                            ttp = np.sum(yval[jdx,:,:,:]==1)
                            tp = np.sum((x_hat_val[jdx,:,:,:]==1) & (yval[jdx,:,:,:]==1))
                            fp = np.sum((x_hat_val[jdx,:,:,:] == 1) & (yval[jdx,:,:,:] == 0))
                            fn = np.sum((x_hat_val[jdx,:,:,:] == 0) & (yval[jdx,:,:,:] == 1))
                            tot = tp+fp
                            tot2 = tp+fn
                            if tot==0 or tot2 ==0 or tp==0:
                             
                              f1_s = 1
                            else:
                              precision = tp / (tp + fp);
                              recall = tp / (tp + fn);
                              f1_s = (2 * precision * recall) / (precision + recall);
                            #print(f1_s)

                            Fs = Fs + f1_s
                        # print(f1_s)
                        
                        
                        
                Fs = Fs/330
                if Best_Fs<Fs:
                    Best_Fs = Fs
                    Best_epoch = epoch
                print("Best epoch: %d"%(Best_epoch))
                print("Best Fs: %f\t Fs:%f"%(Best_Fs,Fs))
            if ganIterations % (40*opt.display) == 0:
                vutils.save_image(val_batch_output, '%s/generated_epoch_iter%08d.png' % \
                    (opt.exp, ganIterations), normalize=True, scale_each=False)
            
        sys.stdout.flush()
        trainLogger.write('%d\t%f\t%f\t%f\t%f\t%f\t%f\n' % \
                        (i, L_img.data[0], L_img.data[0], L_img.data[0], L_img.data[0], L_img.data[0], L_img.data[0]))
        trainLogger.flush()
    torch.save(netG.state_dict(), '%s/SMaps_%d.pth' % (opt.exp, epoch))
        
trainLogger.close()


