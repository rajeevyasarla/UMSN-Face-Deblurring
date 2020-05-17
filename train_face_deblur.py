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
  default=175, help='the height / width of the original input image')
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
parser.add_argument('--annealEvery', type=int, default=400, help='epoch to reaching at learning rate of 0')
parser.add_argument('--lambdaGAN', type=float, default=0.01, help='lambdaGAN')
parser.add_argument('--lambdaIMG', type=float, default=2.0, help='lambdaIMG')
parser.add_argument('--poolSize', type=int, default=50, help='Buffer size for storing previously generated samples from G')
parser.add_argument('--wd', type=float, default=0.0000, help='weight decay in D')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--exp', default='sample', help='folder to output images and model checkpoints')
parser.add_argument('--display', type=int, default=5, help='interval for displaying train-logs')
parser.add_argument('--evalIter', type=int, default=500, help='interval for evauating(generating) images from valDataroot')
opt = parser.parse_args()
print(opt)

from scipy import signal
import h5py
from scipy import signal
import random

#loading kernels mat file
k_filename ='./kernel.mat'
kfp = h5py.File(k_filename)
kernels = np.array(kfp['kernels'])
kernels = kernels.transpose([0,2,1])


vgg = Vgg16()
#utils.init_vgg16('./models/')
state_dict_g = torch.load('VGG_FACE.pth')
new_state_dict_g = {}
new_state_dict_g["conv1_1.weight"]= state_dict_g["0.weight"]
new_state_dict_g["conv1_1.bias"]= state_dict_g["0.bias"]
new_state_dict_g["conv1_2.weight"]= state_dict_g["2.weight"]
new_state_dict_g["conv1_2.bias"]= state_dict_g["2.bias"]
new_state_dict_g["conv2_1.weight"]= state_dict_g["5.weight"]
new_state_dict_g["conv2_1.bias"]= state_dict_g["5.bias"]
new_state_dict_g["conv2_2.weight"]= state_dict_g["7.weight"]
new_state_dict_g["conv2_2.bias"]= state_dict_g["7.bias"]
new_state_dict_g["conv3_1.weight"]= state_dict_g["10.weight"]
new_state_dict_g["conv3_1.bias"]= state_dict_g["10.bias"]
new_state_dict_g["conv3_2.weight"]= state_dict_g["12.weight"]
new_state_dict_g["conv3_2.bias"]= state_dict_g["12.bias"]
new_state_dict_g["conv3_3.weight"]= state_dict_g["14.weight"]
new_state_dict_g["conv3_3.bias"]= state_dict_g["14.bias"]
new_state_dict_g["conv4_1.weight"]= state_dict_g["17.weight"]
new_state_dict_g["conv4_1.bias"]= state_dict_g["17.bias"]
new_state_dict_g["conv4_2.weight"]= state_dict_g["19.weight"]
new_state_dict_g["conv4_2.bias"]= state_dict_g["19.bias"]
new_state_dict_g["conv4_3.weight"]= state_dict_g["21.weight"]
new_state_dict_g["conv4_3.bias"]= state_dict_g["21.bias"]
new_state_dict_g["conv5_1.weight"]= state_dict_g["24.weight"]
new_state_dict_g["conv5_1.bias"]= state_dict_g["24.bias"]
new_state_dict_g["conv5_2.weight"]= state_dict_g["26.weight"]
new_state_dict_g["conv5_2.bias"]= state_dict_g["26.bias"]
new_state_dict_g["conv5_3.weight"]= state_dict_g["28.weight"]
new_state_dict_g["conv5_3.bias"]= state_dict_g["28.bias"]
vgg.load_state_dict(new_state_dict_g)

vgg = torch.nn.DataParallel(vgg)
vgg.cuda()

create_exp_dir(opt.exp)
opt.manualSeed = random.randint(1, 10000)
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
netS=net.Segmentation()
netG=net.Deblur_segdl()


netS.load_state_dict(torch.load('./pretrained_models/SMaps_Best.pth'))

netG.apply(weights_init)
if opt.netG != '':
    state_dict_g = torch.load(opt.netG)
    new_state_dict_g = {}
    for k, v in state_dict_g.items():
        name = k[7:] 
        new_state_dict_g[name] = v
    # load params
    netG.load_state_dict(new_state_dict_g)
print(netG)


netG = torch.nn.DataParallel(netG)
netS = torch.nn.DataParallel(netS)
netG.train()
criterionCAE = nn.L1Loss()
criterionCAE1 = nn.SmoothL1Loss()

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


# image pool storing previously generated samples from G
lambdaGAN = opt.lambdaGAN
lambdaIMG = opt.lambdaIMG

netG.cuda()
netS.cuda()
criterionCAE.cuda()
criterionCAE1.cuda()




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


val_input_cpu, val_target_cpu = data_val

val_target_cpu, val_input_cpu = val_target_cpu.float().cuda(), val_input_cpu.float().cuda()



val_target.resize_as_(val_target_cpu).copy_(val_target_cpu)
val_input.resize_as_(val_input_cpu).copy_(val_input_cpu)

vutils.save_image(val_target, '%s/real_target.png' % opt.exp, normalize=True)
vutils.save_image(val_input, '%s/real_input.png' % opt.exp, normalize=True)




optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (opt.beta1, 0.999), weight_decay=0.00005)
# NOTE training loop
ganIterations = 0
count = 0


for epoch in range(opt.niter):
  if epoch % 19 == 0  and epoch>0:
      opt.lrG = opt.lrG/2.0
      for param_group in optimizerG.param_groups:
          param_group['lr'] = opt.lrG
  if epoch >= opt.annealStart:
    adjust_learning_rate(optimizerG, opt.lrG, epoch, None, opt.annealEvery)


  for i, data in enumerate(dataloader, 0):

    input_cpu, target_cpu = data
    batch_size = target_cpu.size(0)
    b,ch,x,y = target_cpu.size()
    x1 = int((x-opt.imageSize)/2)
    y1 = int((y-opt.imageSize)/2)

    #generating blurry image
    input_cpu = input_cpu.numpy()
    target_cpu = target_cpu.numpy()
    for j in range(batch_size):
        index = random.randint(0,24500)
        input_cpu[j,0,:,:]= signal.convolve(input_cpu[j,0,:,:],kernels[index,:,:],mode='same')
        input_cpu[j,1,:,:]= signal.convolve(input_cpu[j,1,:,:],kernels[index,:,:],mode='same')
        input_cpu[j,2,:,:]= signal.convolve(input_cpu[j,2,:,:],kernels[index,:,:],mode='same')
    input_cpu = input_cpu + (1.0/255.0)* np.random.normal(0,4,input_cpu.shape)
    input_cpu = input_cpu[:,:,x1:x1+opt.imageSize,y1:y1+opt.imageSize]
    target_cpu = target_cpu[:,:,x1:x1+opt.imageSize,y1:y1+opt.imageSize]
    input_cpu = torch.from_numpy(input_cpu)
    target_cpu = torch.from_numpy(target_cpu)
    
        
    

    target_cpu, input_cpu = target_cpu.float().cuda(), input_cpu.float().cuda()

    # getting input and target image at 0.5 scale
    target.data.resize_as_(target_cpu).copy_(target_cpu)
    input.data.resize_as_(input_cpu).copy_(input_cpu)
    input_256 = torch.nn.functional.interpolate(input,scale_factor=0.5)
    target_256 = torch.nn.functional.interpolate(target,scale_factor=0.5)



    # computing segmentation masks for input and target 
    with torch.no_grad():
        smaps_i,smaps_i64 = netS(input,input_256)
        smaps,smaps64 = netS(target,target_256)
        class1 = torch.zeros([batch_size,1,128,128], dtype=torch.float32)
        class1[:,0,:,:] = smaps_i[:,0,:,:]
        class2 = torch.zeros([batch_size,1,128,128], dtype=torch.float32)
        class2[:,0,:,:] = smaps_i[:,1,:,:]
        class3 = torch.zeros([batch_size,1,128,128], dtype=torch.float32)
        class3[:,0,:,:] = smaps_i[:,2,:,:]
        class4 = torch.zeros([batch_size,1,128,128], dtype=torch.float32)
        class4[:,0,:,:] = smaps_i[:,3,:,:]
        class_msk1 = torch.zeros([batch_size,3,128,128], dtype=torch.float32)
        class_msk1[:,0,:,:] = smaps[:,0,:,:] 
        class_msk1[:,1,:,:] = smaps[:,0,:,:] 
        class_msk1[:,2,:,:] = smaps[:,0,:,:] 
        class_msk2 = torch.zeros([batch_size,3,128,128], dtype=torch.float32)
        class_msk2[:,0,:,:] = smaps[:,1,:,:]
        class_msk2[:,1,:,:] = smaps[:,1,:,:]
        class_msk2[:,2,:,:] = smaps[:,1,:,:]
        class_msk3 = torch.zeros([batch_size,3,128,128], dtype=torch.float32)
        class_msk3[:,0,:,:] = smaps[:,2,:,:] 
        class_msk3[:,1,:,:] = smaps[:,2,:,:] 
        class_msk3[:,2,:,:] = smaps[:,2,:,:] 
        class_msk4 = torch.zeros([batch_size,3,128,128], dtype=torch.float32)
        class_msk4[:,0,:,:] = smaps[:,3,:,:]  
        class_msk4[:,1,:,:] = smaps[:,3,:,:]
        class_msk4[:,2,:,:] = smaps[:,3,:,:]

    class1 = class1.float().cuda()
    class2 = class2.float().cuda()
    class3 = class3.float().cuda()
    class4 = class4.float().cuda()
    class_msk4 = class_msk4.float().cuda()
    class_msk3 = class_msk3.float().cuda()
    class_msk2 = class_msk2.float().cuda()
    class_msk1 = class_msk1.float().cuda()

    # Forward step
    x_hat1,x_hat64,xmask1,xmask2,xmask3,xmask4,xcl_class1,xcl_class2,xcl_class3,xcl_class4 = netG(input,input_256,smaps_i,class1,class2,class3,class4,target,class_msk1,class_msk2,class_msk3,class_msk4)

    x_hat = x_hat1
    
    

    if ganIterations % 2 == 0:
        netG.zero_grad() # start to update G


    if epoch>-1:
      # with torch.no_grad():
      #   smaps,smaps64 = netS(target,target_256)
      L_img_ = 0.33*criterionCAE(x_hat64, target_256) #+ 0.5*criterionCAE(smaps_hat, smaps)
      L_img_ = L_img_ + 1.2 *criterionCAE(xmask1*class_msk1*x_hat+(1-xmask1)*class_msk1*target, class_msk1*target) 
      L_img_ = L_img_ + 1.2 *criterionCAE(xmask2*class_msk2*x_hat+(1-xmask2)*class_msk2*target, class_msk2*target)
      L_img_ = L_img_ + 3.6 *criterionCAE(xmask3*class_msk3*x_hat+(1-xmask3)*class_msk3*target, class_msk3*target)
      L_img_ = L_img_ + 1.2 *criterionCAE(xmask4*class_msk4*x_hat+(1-xmask4)*class_msk4*target, class_msk4*target)
      if ganIterations % (25*opt.display) == 0:
          print(L_img_.data[0])
          sys.stdout.flush()
      if  ganIterations< -1:
          lam_cmp = 1.0
      else :
          lam_cmp = 0.06
      sng = 0.00000001
      L_img_ = L_img_ - (lam_cmp/(4.0))*torch.mean(torch.log(xmask1+sng))
      L_img_ = L_img_ - (lam_cmp/(4.0))*torch.mean(torch.log(xmask2+sng))
      L_img_ = L_img_ - (lam_cmp/(4.0))*torch.mean(torch.log(xmask3+sng))
      L_img_ = L_img_ - (lam_cmp/(4.0))*torch.mean(torch.log(xmask4+sng))
      if ganIterations % (50*opt.display) == 0:
          print(L_img_.data[0])
          sys.stdout.flush()
      
      gradh_xhat,gradv_xhat=gradient(x_hat)
      gradh_tar,gradv_tar=gradient(target)
      gradh_xhat64,gradv_xhat64=gradient(x_hat64)
      gradh_tar64,gradv_tar64=gradient(target_256)
      L_img_ = L_img_ + 0.15*criterionCAE(gradh_xhat,gradh_tar)+ 0.15*criterionCAE(gradv_xhat,gradv_tar)+ 0.08*criterionCAE(gradh_xhat64,gradh_tar64)+0.08*criterionCAE(gradv_xhat64,gradv_tar64)
      if ganIterations % (25*opt.display) == 0:
          print(L_img_.data[0])
          print((torch.mean(torch.log(xmask1)).data),(torch.mean(torch.log(xmask2)).data),(torch.mean(xmask3).data),(torch.mean(xmask4).data))
          sys.stdout.flush()

      L_img = lambdaIMG * L_img_
      #Backward step or computing gradients
      if lambdaIMG != 0:
        L_img.backward(retain_graph=True)

      # Perceptual Loss 1
      features_content = vgg(target)
      f_xc_c = Variable(features_content[1].data, requires_grad=False)
      f_xc_c5 = Variable(features_content[4].data, requires_grad=False)
      features_y = vgg(x_hat)
      
      features_content = vgg(target_256)
      f_xc_c64 = Variable(features_content[1].data, requires_grad=False)
      features_y64 = vgg(x_hat64)
      lambda_p=0.00018
      content_loss =  lambda_p*lambdaIMG* criterionCAE(features_y[1], f_xc_c) + lambda_p*0.33*lambdaIMG* criterionCAE(features_y64[1], f_xc_c64) + lambda_p*lambdaIMG* criterionCAE(features_y[4], f_xc_c5) 
      content_loss.backward(retain_graph=True)

      # Perceptual Loss 2
      features_content = vgg(target)
      f_xc_c = Variable(features_content[0].data, requires_grad=False)
      features_y = vgg(x_hat)
      
      features_content = vgg(target_256)
      f_xc_c64 = Variable(features_content[0].data, requires_grad=False)
      features_y64 = vgg(x_hat64)
      
      content_loss1 =  lambda_p*lambdaIMG* criterionCAE(features_y[0], f_xc_c) + lambda_p*0.33*lambdaIMG* criterionCAE(features_y64[0], f_xc_c64)
      content_loss1.backward(retain_graph=True)


    else:
      L_img_ = 1.2 *criterionCAE(xcl_class1, target) 
      L_img_ = L_img_ + 1.2 *criterionCAE(xcl_class2, target)
      L_img_ = L_img_ + 3.6 *criterionCAE(xcl_class3, target)
      L_img_ = L_img_ + 1.2 *criterionCAE(xcl_class4, target)
      L_img = lambdaIMG * L_img_
      if lambdaIMG != 0:
        L_img.backward(retain_graph=True)
      if ganIterations % (25*opt.display) == 0:
          print(L_img_.data[0])
          print("updating fisrt stage parameters")
          sys.stdout.flush()

    

    if ganIterations % 2 == 0:
        optimizerG.step()
    ganIterations += 1

    if ganIterations % opt.display == 0:
      print('[%d/%d][%d/%d] Loss: %f '
          % (epoch, opt.niter, i, len(dataloader),
             L_img.data[0]))
      sys.stdout.flush()
      trainLogger.write('%d\t%f\n' % \
                        (i, L_img.data[0]))
      trainLogger.flush()

    #validation
    if ganIterations % (int(len(dataloader)/2)) == 0:
      val_batch_output = torch.zeros([16,3,128,128], dtype=torch.float32)#torch.FloatTensor([10,3,128,128]).fill_(0)
      for idx in range(val_input.size(0)):
        single_img = val_input[idx,:,:,:].unsqueeze(0)
        val_inputv = Variable(single_img, volatile=True)
        with torch.no_grad():
            index = idx+24500
            val_inputv = val_inputv.cpu().numpy()
            val_inputv[0,0,:,:]= signal.convolve(val_inputv[0,0,:,:],kernels[index,:,:],mode='same')
            val_inputv[0,1,:,:]= signal.convolve(val_inputv[0,1,:,:],kernels[index,:,:],mode='same')
            val_inputv[0,2,:,:]= signal.convolve(val_inputv[0,2,:,:],kernels[index,:,:],mode='same')
            val_inputv = val_inputv[:,:,x1:x1+opt.imageSize,y1:y1+opt.imageSize]
            val_inputv = val_inputv + (1.0/255.0)* np.random.normal(0,4,val_inputv.shape)
            val_inputv = torch.from_numpy(val_inputv)
            val_inputv = val_inputv.float().cuda()
            val_inputv_256 = torch.nn.functional.interpolate(val_inputv,scale_factor=0.5)
            #rint(val_inputv.size())
            smaps,smaps64 = netS(val_inputv,val_inputv_256)
            class1 = torch.zeros([1,1,128,128], dtype=torch.float32)
            class1[:,0,:,:] = smaps[:,0,:,:]
            class2 = torch.zeros([1,1,128,128], dtype=torch.float32)
            class2[:,0,:,:] = smaps[:,1,:,:]
            class3 = torch.zeros([1,1,128,128], dtype=torch.float32)
            class3[:,0,:,:] = smaps[:,2,:,:]
            class4 = torch.zeros([1,1,128,128], dtype=torch.float32)
            class4[:,0,:,:] = smaps[:,3,:,:]
            class_msk1 = torch.zeros([1,3,128,128], dtype=torch.float32)
            class_msk1[:,0,:,:] = smaps[:,0,:,:] 
            class_msk1[:,1,:,:] = smaps[:,0,:,:] 
            class_msk1[:,2,:,:] = smaps[:,0,:,:] 
            class_msk2 = torch.zeros([1,3,128,128], dtype=torch.float32)
            class_msk2[:,0,:,:] = smaps[:,1,:,:]
            class_msk2[:,1,:,:] = smaps[:,1,:,:]
            class_msk2[:,2,:,:] = smaps[:,1,:,:]
            class_msk3 = torch.zeros([1,3,128,128], dtype=torch.float32)
            class_msk3[:,0,:,:] = smaps[:,2,:,:] 
            class_msk3[:,1,:,:] = smaps[:,2,:,:] 
            class_msk3[:,2,:,:] = smaps[:,2,:,:] 
            class_msk4 = torch.zeros([1,3,128,128], dtype=torch.float32)
            class_msk4[:,0,:,:] = smaps[:,3,:,:]  
            class_msk4[:,1,:,:] = smaps[:,3,:,:]
            class_msk4[:,2,:,:] = smaps[:,3,:,:]
            x_hat_val, x_hat_val64,xmask1,xmask2,xmask3,xmask4,xcl_class1,xcl_class2,xcl_class3,xcl_class4 = netG(val_inputv,val_inputv_256,smaps,class1,class2,class3,class4,val_inputv,class_msk1,class_msk2,class_msk3,class_msk4)
            #x_hat_val.data[0,:,:,:] = masks*x_hat_val.data[0,:,:,:]
            val_batch_output[idx,:,:,:].copy_(x_hat_val.data[0,:,:,:])
        ###  We use a random label here just for intermediate result visuliztion (No need to worry about the label here) ##

                
    if ganIterations % (int(len(dataloader)/2)) == 0:
        vutils.save_image(val_batch_output, '%s/generated_epoch_iter%08d.png' % \
            (opt.exp, ganIterations), normalize=True, scale_each=False)
        del val_batch_output
    if ganIterations % (int(len(dataloader)/2)) == 0:
        torch.save(netG.state_dict(), '%s/Deblur_epoch_%d.pth' % (opt.exp, count))
        #torch.save(netC.state_dict(), '%s/Deblur_first_epoch_%d.pth' % (opt.exp, count))
        count = count +1
trainLogger.close()

