import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

###################################################################ResNet#####################################################################
def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
       
        # x = F.normalize(x,dim=1)
        #x = F.normalize(self.fc(x),dim=1)
      
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = 
    #     load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', BasicBlock, [3, 2, 2, 2], pretrained, progress,
                   **kwargs)



class ResNet_sup(nn.Module):
    
    def __init__(self, in_channels, *args, **kwargs):
    #def __init__(self, in_channels,n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = resnet50(pretrained=False)
        self.decoder = ResnetDecoder(head='mlp',feat_dim=10)
        #self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features=1024, head='mlp', feat_dim=10):
        super().__init__()
        #self.avg = nn.AdaptiveAvgPool2d((1, 1))
        dim_in = in_features
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        # x = self.avg(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        #x = self.decoder(x)
        x = F.normalize(self.head(x), dim=1)
       
        return x

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, feat_dim, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return F.normalize(self.fc(x),dim=1)
    

def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()

def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

################################################################# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print("How many GPUs are availabe: ", torch.cuda.device_count())
#print("GPU Name: " , torch.cuda.get_device_name(0))



############################################################ Image preprocessing modules

#cifar10
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
# cifar100
# mean = (0.5071, 0.4867, 0.4408)
# std = (0.2675, 0.2565, 0.2761)
 
normalize = transforms.Normalize(mean=mean, std=std)

train_transform = transforms.Compose([
    transforms.Pad(5),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),normalize])
val_transform = transforms.Compose([
        transforms.ToTensor(),normalize])


############################################################ Hyper-parameters
num_epochs = 200
batchsize = 256

############################################################ CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./',
                                             train=True, 
                                             transform=train_transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./',
                                            train=False, 
                                            transform=val_transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batchsize, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batchsize, 
                                           shuffle=False)


############################################################ Representation Learning
model = ResNet_sup(3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(num_epochs):
    total_train, correct_train= 0,0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ("epoch=",epoch, "loss=", loss.item(),"acc=", correct_train/total_train)

######################################### Save and Load Trained base encoder + Projection head
PATH = './cifar10_ResNet_Epoch=200_BatchSize=256_lr=0.001_PHead_mlp.pt'
model.load_state_dict(torch.load(PATH))
# torch.save(model.state_dict(), PATH)

######################################### Linear Classification Training
classifier = LinearClassifier(1024,10).to(device)

criterion = nn.CrossEntropyLoss()
opt1 = optim.Adam(classifier.parameters(), lr=0.001)

num_epochs = 50

for t in range(num_epochs):
    total_loss_cls, total_err_train_cls = 0., 0.
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
              h= model.encoder(images)
              # h= model(images)
        yp = classifier(h)
        total_err_train_cls += (yp.max(dim=1)[1] != labels).sum().item()
        loss_train_cls = criterion(yp,labels)
        opt1.zero_grad()
        loss_train_cls.backward()
        opt1.step()
        total_loss_cls += loss_train_cls.item() * labels.shape[0]
    print('Epoch =' , t , 'Loss_Train_cls =', total_loss_cls/len(train_loader.dataset),'Error_Train =',total_err_train_cls/len(train_loader.dataset), sep="\t")


########################################## Load and Save Linear Classifier
PATH = './cifar10_CLS_Res_Epoch=50_BatchSize=256_lr=0.001_Phead_MLP.pt'
# torch.save(classifier.state_dict(), PATH)
classifier.load_state_dict(torch.load(PATH))

########################################### Test Phase
total_loss_test,total_err_test = 0., 0.
for idx, (X, y) in enumerate(test_loader):
    X,y = X.to(device), y.to(device)
    h = model.encoder(X)
    # h = model(X)
    yp = classifier(h)
    loss_test = nn.CrossEntropyLoss()(yp,y)
    total_loss_test += loss_test.item() * y.shape[0]
    total_err_test += (yp.max(dim=1)[1] != y).sum().item()
print('Loss_Test =',total_loss_test/len(test_loader.dataset), 'Error_Test =', total_err_test / len(test_loader.dataset),sep="\t")

########################################### FGSM scheme 1 and scheme 2
# total_loss_adv,total_err_adv = 0., 0.
# for idx, (X, y) in enumerate(test_loader):
#     X,y = X.to(device), y.to(device)
#     delta = fgsm(model, X, y, epsilon=4/255)
#     h = model.encoder(X+delta)
#     yp = classifier(h)
#     loss_test_adv = nn.CrossEntropyLoss()(yp,y)
#     total_loss_adv += loss_test_adv.item() * y.shape[0]
#     total_err_adv += (yp.max(dim=1)[1] != y).sum().item()
# print('Loss_Test_adv =',total_loss_adv/len(test_loader.dataset), 'Error_Test_adv =', total_err_adv/ len(test_loader.dataset),sep="\t")

############################################ PGD scheme 1 and scheme 2
# total_loss_adv,total_err_adv = 0., 0.
# for idx, (X, y) in enumerate(test_loader):
#     X,y = X.to(device), y.to(device)
#     delta = pgd_linf(model, X, y, epsilon=16/255, alpha=0.01, num_iter=500)
#     h = model.encoder(X+delta)
#     yp = classifier(h)
#     loss_test_adv = nn.CrossEntropyLoss()(yp,y)
#     total_loss_adv += loss_test_adv.item() * y.shape[0]
#     total_err_adv += (yp.max(dim=1)[1] != y).sum().item()
# print('Loss_Test_adv =',total_loss_adv/len(test_loader.dataset), 'Error_Test_adv =', total_err_adv/ len(test_loader.dataset),sep="\t")

############################################# FGSM scheme 3 
total_loss_test_adv1,total_err_test_adv1 = 0., 0.
for idx, (X, y) in enumerate(test_loader):
    X,y = X.to(device), y.to(device)
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(classifier(model.encoder(X + delta)), y)
    loss.backward()
    epsilon = 4/255
    delta_adv1 = epsilon * delta.grad.detach().sign()
    h = model.encoder(X+delta_adv1)
    yp = classifier(h)
    loss_test_adv1 = nn.CrossEntropyLoss()(yp,y)
    total_loss_test_adv1 += loss_test_adv1.item() * y.shape[0]
    total_err_test_adv1 += (yp.max(dim=1)[1] != y).sum().item()
print('Loss_Test =',total_loss_test_adv1/len(test_loader.dataset), 'Error_Test =', total_err_test_adv1 / len(test_loader.dataset),sep="\t")


############################################# PGD scheme 3
# total_loss_test_adv2,total_err_test_adv2 = 0., 0.
# for idx, (X, y) in enumerate(test_loader):
#     X,y = X.to(device), y.to(device)
#     delta = torch.zeros_like(X, requires_grad=True)
#     num_iter = 100
#     alpha = 0.01
#     epsilon = 4/255
#     for t in range(num_iter):
#         loss = nn.CrossEntropyLoss()(classifier(model.encoder(X + delta)), y)
#         loss.backward()
#         delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
#         delta.grad.zero_()
#     delta_adv2 = delta.detach()
#     h = model.encoder(X+delta_adv2)
#     yp = classifier(h)
#     loss_test_adv2 = nn.CrossEntropyLoss()(yp,y)
#     total_loss_test_adv2 += loss_test_adv2.item() * y.shape[0]
#     total_err_test_adv2 += (yp.max(dim=1)[1] != y).sum().item()
# print('Loss_Test =',total_loss_test_adv2/len(test_loader.dataset), 'Error_Test =', total_err_test_adv2 / len(test_loader.dataset),sep="\t")

