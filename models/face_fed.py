import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable



def conv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                       nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
  return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.UpsamplingNearest2d(scale_factor=2))


def blockUNet1(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False))
  else:
    block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 3, 1, 1, bias=False))
  if bn:
    block.add_module('%s.bn' % name, nn.InstanceNorm2d(out_c))
  if dropout:
    block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block

def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
  block = nn.Sequential()
  if relu:
    block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
  else:
    block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
  if not transposed:
    block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
  else:
    block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
  if bn:
    block.add_module('%s.bn' % name, nn.InstanceNorm2d(out_c))
  if dropout:
    block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
  return block


class D1(nn.Module):
  def __init__(self, nc, ndf, hidden_size):
    super(D1, self).__init__()

    # 256
    self.conv1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1),
                               nn.ELU(True))
    # 256
    self.conv2 = conv_block(ndf,ndf)
    # 128
    self.conv3 = conv_block(ndf, ndf*2)
    # 64
    self.conv4 = conv_block(ndf*2, ndf*3)
    # 32
    self.encode = nn.Conv2d(ndf*3, hidden_size, kernel_size=1,stride=1,padding=0)
    self.decode = nn.Conv2d(hidden_size, ndf, kernel_size=1,stride=1,padding=0)
    # 32
    self.deconv4 = deconv_block(ndf, ndf)
    # 64
    self.deconv3 = deconv_block(ndf, ndf)
    # 128
    self.deconv2 = deconv_block(ndf, ndf)
    # 256
    self.deconv1 = nn.Sequential(nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                                 nn.ELU(True),
                                 nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                                 nn.ELU(True),
                                 nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1),
                                 nn.Tanh())
    """
    self.deconv1 = nn.Sequential(nn.Conv2d(ndf,nc,kernel_size=3,stride=1,padding=1),
                                 nn.Tanh())
    """
  def forward(self,x):
    out1 = self.conv1(x)
    out2 = self.conv2(out1)
    out3 = self.conv3(out2)
    out4 = self.conv4(out3)
    out5 = self.encode(out4)
    dout5= self.decode(out5)
    dout4= self.deconv4(dout5)
    dout3= self.deconv3(dout4)
    dout2= self.deconv2(dout3)
    dout1= self.deconv1(dout2)
    return dout1

class D(nn.Module):
  def __init__(self, nc, nf):
    super(D, self).__init__()

    main = nn.Sequential()
    # 256
    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s.conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    # 128
    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 64
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    # 32
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    main.add_module('%s.bn' % name, nn.InstanceNorm2d(nf*2))

    # 31
    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s.conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    main.add_module('%s.sigmoid' % name , nn.Sigmoid())
    # 30 (sizePatchGAN=30)

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output

class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight,
                        None, 1, self.padding, 1, inc)


class BottleneckBlockdls(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlockdls, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_o = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.sharewconv1 = ShareSepConv(3)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.sharewconv2 = ShareSepConv(3)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=2, dilation=2, bias=False)
        self.bn4 = nn.BatchNorm2d(inter_planes)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=2, dilation=2, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        outx = self.conv_o(x)
        out = outx + self.conv4(self.sharewconv2(self.relu(self.bn4(out))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlockdl(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlockdl, self).__init__()
        inter_planes = out_planes * 3
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_o = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, dilation=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=2, dilation=2, bias=False)
        self.bn4 = nn.InstanceNorm2d(inter_planes)
        self.sharewconv = ShareSepConv(3)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=2, dilation=2, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        outx = self.conv_o(x)
        out = outx + self.conv4(self.sharewconv(self.relu(self.bn4(out))))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlockrs1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlockrs1, self).__init__()
        inter_planes = out_planes * 3
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_o = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=2, dilation=2, bias=False)
        self.bn4 = nn.InstanceNorm2d(inter_planes)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        outx = self.conv_o(x)
        out = outx + self.conv4(self.relu(self.bn4(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlockrs(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlockrs, self).__init__()
        inter_planes = out_planes * 3
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_o = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(inter_planes)
        self.conv3 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn4 = nn.InstanceNorm2d(inter_planes)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        outx = self.conv_o(x)
        out = outx + self.conv4(self.relu(self.bn4(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)





class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)



class TransitionBlock1(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock1, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)



class TransitionBlock3(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock3, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out






class vgg19ca(nn.Module):
    def __init__(self):
        super(vgg19ca, self).__init__()




        ############# 256-256  ##############
        haze_class = models.vgg19_bn(pretrained=True)
        self.feature = nn.Sequential(haze_class.features[0])

        for i in range(1,3):
            self.feature.add_module(str(i),haze_class.features[i])

        self.conv16=nn.Conv2d(64, 24, kernel_size=3,stride=1,padding=1)  # 1mm
        self.dense_classifier=nn.Linear(127896, 512)
        self.dense_classifier1=nn.Linear(512, 4)


    def forward(self, x):

        feature=self.feature(x)
        # feature = Variable(feature.data, requires_grad=True)

        feature=self.conv16(feature)
        # print feature.size()

        # feature=Variable(feature.data,requires_grad=True)



        out = F.relu(feature, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(out.size(0), -1)
        # print out.size()

        # out=Variable(out.data,requires_grad=True)
        out = F.relu(self.dense_classifier(out))
        out = (self.dense_classifier1(out))


        return out
class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 3
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_o = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        self.bn4 = nn.InstanceNorm2d(inter_planes)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        outx = self.conv_o(x)
        out = outx + self.conv4(self.relu(self.bn4(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out

class TransitionBlockbil(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlockbil, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = F.upsample_bilinear(out, scale_factor=2)
        return self.conv2(out)


class Deblur_first(nn.Module):
    def __init__(self,in_channels):
        super(Deblur_first, self).__init__()

        self.dense_block1=BottleneckBlockrs(in_channels,32-in_channels)
        self.trans_block1=TransitionBlock1(32,16)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlockdl(16,16)
        self.trans_block2=TransitionBlock3(32,16)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlockdl(16,16)
        self.trans_block3=TransitionBlock3(32,16)
        

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlockdl(32,16)
        self.trans_block5=TransitionBlock3(48,16)

        self.dense_block6=BottleneckBlockrs(16,16)
        self.trans_block6=TransitionBlockbil(32,16)


        self.conv_refin=nn.Conv2d(16,16,3,1,1)
        self.conv_refin_in=nn.Conv2d(in_channels,16,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(16, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1= nn.Conv2d(3, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)


    def forward(self, x,smaps):
        ## 256x256
        x1=self.dense_block1(torch.cat([x,smaps],1))
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(x1))
        x2=self.trans_block2(x2)

        #print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        x5_in=torch.cat([x3, x1], 1)
        x5_i=(self.dense_block5(x5_in))
        
        x5=self.trans_block5(x5_i)
        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))
        
        x7=self.relu(self.conv_refin_in(torch.cat([x,smaps],1))) - self.relu(self.conv_refin(x6))
        residual=self.tanh(self.refine3(x7))
        clean = x - residual
        clean = self.relu(self.refineclean1(clean))
        clean = self.tanh(self.refineclean2(clean))
        

        return clean,x5

class Deblur_class(nn.Module):
    def __init__(self):
        super(Deblur_class, self).__init__()
        
        ##### stage class networks ###########
        self.deblur_class1 = Deblur_first(4)
        self.deblur_class2 = Deblur_first(4)
        self.deblur_class3 = Deblur_first(11)
        self.deblur_class4 = Deblur_first(4)
        ######################################
        
    def forward(self, x_input1,x_input2,x_input3,x_input4,class1,class2,class3,class4):
        xh_class1,x_lst1 = self.deblur_class1(x_input1,class1)
        xh_class2,x_lst2 = self.deblur_class2(x_input2,class2)
        xh_class3,x_lst3 = self.deblur_class3(x_input3,class3)
        xh_class4,x_lst4 = self.deblur_class4(x_input4,class4)
        
        return xh_class1,xh_class2,xh_class3,xh_class4,x_lst1,x_lst2,x_lst3,x_lst4

class BottleneckBlockcf(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlockcf, self).__init__()
        inter_planes = out_planes * 3
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_o = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, inter_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        self.bn4 = nn.InstanceNorm2d(inter_planes)
        self.conv4 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv4(self.relu(self.bn4(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out

class scale_kernel_conf(nn.Module):
    def __init__(self):
        super(scale_kernel_conf, self).__init__()

        self.conv1 = nn.Conv2d(6,16,3,1,1)#BottleneckBlock(35, 16)
        self.trans_block1 = TransitionBlock1(16, 16)
        self.conv2 = BottleneckBlockcf(16, 32)
        self.trans_block2 = TransitionBlock1(32, 16)
        self.conv3 = BottleneckBlockcf(16, 32)
        self.trans_block3 = TransitionBlock1(32, 16)
        self.conv4 = BottleneckBlockcf(16, 32)
        self.trans_block4 = TransitionBlock3(32, 16)
        self.conv_refin = nn.Conv2d(16, 3, 1, 1, 0)
        self.sig = torch.nn.Sigmoid()

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x,target):
        x1=self.conv1(torch.cat([x,target],1))
        x1 = self.trans_block1(x1)
        x2=self.conv2(x1)
        x2 = self.trans_block2(x2)
        x3=self.conv3(x2)
        x3 = self.trans_block3(x3)
        x4=self.conv3(x3)
        x4 = self.trans_block4(x4)
        #print(x4.size())
        residual = self.sig(self.conv_refin(self.sig(F.avg_pool2d(x4,16))))
        #print(residual)
        residual = F.upsample_nearest(residual, scale_factor=128)
        #print(residual.size())
        return residual



class Deblur_segdl(nn.Module):
    def __init__(self):
        super(Deblur_segdl, self).__init__()
        self.deblur_class1 = Deblur_first(4)
        self.deblur_class2 = Deblur_first(4)
        self.deblur_class3 = Deblur_first(4)
        self.deblur_class4 = Deblur_first(4)
        self.dense_block1=BottleneckBlockrs(7,57)
        self.dense_block_cl=BottleneckBlock(64,32)
        #self.trans_block_cl=TransitionBlock1(64,32)
        self.trans_block1=TransitionBlock1(64,32)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlockrs1(67,64)
        self.trans_block2=TransitionBlock3(131,64)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlockdl(64,64)
        self.trans_block3=TransitionBlock3(128,64)
        
        self.dense_block3_1=BottleneckBlockdl(64,64)
        self.trans_block3_1=TransitionBlock3(128,64)

        self.dense_block3_2=BottleneckBlockdl(64,64)
        self.trans_block3_2=TransitionBlock3(128,64)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlockdl(64,64)
        self.trans_block4=TransitionBlock3(128,64)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlockrs1(128,64)
        self.trans_block5=TransitionBlockbil(195,64)

        self.dense_block6=BottleneckBlockrs(71,64)
        self.trans_block6=TransitionBlock3(135,16)


        self.conv_refin=nn.Conv2d(23,16,3,1,1)
        self.conv_refin_in=nn.Conv2d(7,16,3,1,1)
        self.conv_refin_in64=nn.Conv2d(3,16,3,1,1)
        self.conv_refin64=nn.Conv2d(192,16,3,1,1)
        self.tanh=nn.Tanh()


        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(16, 3, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1= nn.Conv2d(3, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 3, kernel_size=3,stride=1,padding=1)
        
        self.conv11 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv21 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv31 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv3_11 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv3_21 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv41 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv51 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        #self.conv61 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm




        self.batchnorm20=nn.InstanceNorm2d(20)
        self.batchnorm1=nn.InstanceNorm2d(1)
        self.conf_ker = scale_kernel_conf()



    def forward(self, x,x_64,smaps,class1,class2,class3,class4,target,class_msk1,class_msk2,class_msk3,class_msk4):
        ## 256x256
        xcl_class1,xh_class1 = self.deblur_class1(x,class1)
        xcl_class2,xh_class2 = self.deblur_class2(x,class2)
        xcl_class3,xh_class3 = self.deblur_class3(x,class3)
        xcl_class4,xh_class4 = self.deblur_class4(x,class4)
        x_cl = self.dense_block_cl(torch.cat([xh_class1,xh_class2,xh_class3,xh_class4],1))
        x1=self.dense_block1(torch.cat([x,smaps],1))
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(torch.cat([x1,x_64,x_cl],1)))
        x2=self.trans_block2(x2)

        #print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        x3_1 = (self.dense_block3_1(x3))
        x3_1 = self.trans_block3_1(x3_1)
        #print x3_1.size()
        x3_2 = (self.dense_block3_2(x3_1))
        x3_2 = self.trans_block3_2(x3_2)

        ## Classifier  ##
        #x4_in = torch.cat([x3_2, x2], 1)
        x4=(self.dense_block4(x3_2))
        x4=self.trans_block4(x4)
        x5_in=torch.cat([x4, x1,x_cl], 1)
        x5_i=(self.dense_block5(x5_in))
        
        xhat64 = self.relu(self.conv_refin_in64(x_64)) - self.relu(self.conv_refin64(x5_i))
        xhat64 = self.tanh(self.refine3(xhat64))
        x5=self.trans_block5(torch.cat([x5_i,xhat64],1))
        x6=(self.dense_block6(torch.cat([x5,x,smaps],1)))
        x6=(self.trans_block6(x6))
        shape_out = x6.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        x11 = self.upsample(self.relu((self.conv11(torch.cat([x1,x_cl], 1)))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv21(x2))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv31(x3))), size=shape_out)
        x3_11 = self.upsample(self.relu((self.conv3_11(x3_1))), size=shape_out)
        x3_21 = self.upsample(self.relu((self.conv3_21(x3_2))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv41(x4))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv51(x5))), size=shape_out)
        x6=torch.cat([x6,x51,x41,x3_21,x3_11,x31,x21,x11],1)
        x7=self.relu(self.conv_refin_in(torch.cat([x,smaps],1))) - self.relu(self.conv_refin(x6))
        residual=self.tanh(self.refine3(x7))
        clean = x - residual
        clean = self.relu(self.refineclean1(clean))
        clean = self.tanh(self.refineclean2(clean))
        
        clean64 = x_64 - xhat64
        clean64 = self.relu(self.refineclean1(clean64))
        clean64 = self.tanh(self.refineclean2(clean64))
        
        xmask1 = self.conf_ker(clean*class_msk1,target*class_msk1)
        xmask2 = self.conf_ker(clean*class_msk2,target*class_msk2)
        xmask3 = self.conf_ker(clean*class_msk3,target*class_msk3)
        xmask4 = self.conf_ker(clean*class_msk4,target*class_msk4)

        return clean,clean64,xmask1,xmask2,xmask3,xmask4,xcl_class1,xcl_class2,xcl_class3,xcl_class4

class Segmentation(nn.Module):
    def __init__(self):
        super(Segmentation, self).__init__()

        self.dense_block1=BottleneckBlockrs1(3,61)
        self.trans_block1=TransitionBlock1(64,64)

        ############# Block2-down 32-32  ##############
        self.dense_block2=BottleneckBlockdls(67,64)
        self.trans_block2=TransitionBlock1(131,64)

        ############# Block3-down  16-16 ##############
        self.dense_block3=BottleneckBlockdls(64,64)
        self.trans_block3=TransitionBlock3(128,64)
        
        self.dense_block3_1=BottleneckBlockdls(64,64)
        self.trans_block3_1=TransitionBlock3(128,64)

        self.dense_block3_2=BottleneckBlockdls(64,64)
        self.trans_block3_2=TransitionBlock3(128,64)

        ############# Block4-up  8-8  ##############
        self.dense_block4=BottleneckBlockdls(128,64)
        self.trans_block4=TransitionBlock(192,64)

        ############# Block5-up  16-16 ##############
        self.dense_block5=BottleneckBlockdls(128,64)
        self.trans_block5=TransitionBlock(196,64)

        self.dense_block6=BottleneckBlockrs1(64,64)
        self.trans_block6=TransitionBlock3(128,16)


        self.conv_refin=nn.Conv2d(23,16,3,1,1)
        self.conv_refin64=nn.Conv2d(192,16,3,1,1)
        self.tanh=nn.Sigmoid()

        self.refine3= nn.Conv2d(16, 4, kernel_size=3,stride=1,padding=1)
        self.refine3_i= nn.Conv2d(16, 4, kernel_size=3,stride=1,padding=1)
        # self.refine3= nn.Conv2d(20+4, 3, kernel_size=7,stride=1,padding=3)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)
        self.refineclean1= nn.Conv2d(4, 8, kernel_size=7,stride=1,padding=3)
        self.refineclean2= nn.Conv2d(8, 4, kernel_size=3,stride=1,padding=1)
        
        self.conv11 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv21 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv31 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv3_11 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv3_21 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv41 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        self.conv51 = nn.Conv2d(64, 1, kernel_size=3,stride=1,padding=1)  # 1mm
        #self.conv61 = nn.Conv2d(8, 1, kernel_size=3,stride=1,padding=1)  # 1mm




        self.batchnorm20=nn.BatchNorm2d(20)
        self.batchnorm1=nn.BatchNorm2d(1)



    def forward(self, x,x_64):
        ## 256x256
        x1=self.dense_block1(x)
        x1=self.trans_block1(x1)

        ###  32x32
        x2=(self.dense_block2(torch.cat([x1,x_64],1)))
        x2=self.trans_block2(x2)

        #print x2.size()
        ### 16 X 16
        x3=(self.dense_block3(x2))
        x3=self.trans_block3(x3)

        x3_1 = (self.dense_block3_1(x3))
        x3_1 = self.trans_block3_1(x3_1)
        #print x3_1.size()
        x3_2 = (self.dense_block3_2(x3_1))
        x3_2 = self.trans_block3_2(x3_2)

        ## Classifier  ##
        x4_in = torch.cat([x3_2, x2], 1)
        x4=(self.dense_block4(x4_in))
        x4=self.trans_block4(x4)
        x5_in=torch.cat([x4, x1], 1)
        x5_i=(self.dense_block5(x5_in))
        xhat64 = self.relu(self.conv_refin64(x5_i))
        xhat64 = self.tanh(self.refine3_i(xhat64))
        x5=self.trans_block5(torch.cat([x5_i, xhat64], 1))
        
        x6=(self.dense_block6(x5))
        x6=(self.trans_block6(x6))
        shape_out = x6.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]
        x11 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
        x21 = self.upsample(self.relu((self.conv21(x2))), size=shape_out)
        x31 = self.upsample(self.relu((self.conv31(x3))), size=shape_out)
        x3_11 = self.upsample(self.relu((self.conv3_11(x3_1))), size=shape_out)
        x3_21 = self.upsample(self.relu((self.conv3_21(x3_2))), size=shape_out)
        x41 = self.upsample(self.relu((self.conv41(x4))), size=shape_out)
        x51 = self.upsample(self.relu((self.conv51(x5))), size=shape_out)
        x6 = torch.cat([x6,x51,x41,x3_21,x3_11,x31,x21,x11],1)
        x7 = self.relu(self.conv_refin(x6))
        residual = self.tanh(self.refine3(x7))

        return residual,xhat64
