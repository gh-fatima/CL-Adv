

###############################################################################################ResNet
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt 
import matplotlib.image as img

# np.random.seed(0)
# random.seed(0) 
# torch.manual_seed(0) 
# torch.cuda.manual_seed(0) 
# torch.cuda.manual_seed_all(9) 
# torch.backends.cudnn.enabled = False 
# torch.backends.cudnn.benchmark = False 
# torch.backends.cudnn.deterministic = True

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
        self.decoder = ResnetDecoder(head='mlp',feat_dim=128)
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
    def __init__(self, in_features=1024, head='mlp', feat_dim=128):
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






class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = 2
       

    def forward(self, features):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
       
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss =  -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, feat_dim, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return F.normalize(self.fc(x),dim=1)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.

num_epochs = 50
learning_rate = 0.001
bs = 512
#cifar10
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)
# cifar100
# mean = (0.5071, 0.4867, 0.4408)
# std = (0.2675, 0.2565, 0.2761)
 
normalize = transforms.Normalize(mean=mean, std=std)

train_transform_sup = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

  
train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, 
                                          transform=TwoCropTransform(train_transform_sup),
                                          download=True)
    

train_sampler = None
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = bs, shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler)


###############################################################################Contrastive Learning
model = ResNet_sup(3).to(device)
# model= nn.DataParallel(model,device_ids = [0, 1])
# model.to(device)
opt = optim.Adam(model.parameters(), lr=learning_rate)
# for t in range(num_epochs):
#     total_loss_Sup = 0.
#     # Self-Supervised Learning
#     for idx, (images, labels) in enumerate(train_loader):
#         images = torch.cat([images[0]+delta, images[1]], dim=0)
#         images = images.to(device)
#         labels = labels.to(device)
#         bsz = labels.shape[0]
#         features = model(images)
#         f1, f2 = torch.split(features, [bsz, bsz], dim=0)
#         features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
#         criterion = SupConLoss()
#         loss_Sup = criterion(features)
#         opt.zero_grad()
#         loss_Sup.backward()
#         opt.step()
#         total_loss_Sup += loss_Sup.item() * labels.shape[0]
#     print('Epoch =', t , 'Loss_Sup =', total_loss_Sup/len(train_loader.dataset), sep="\t")

############################################################################### Save and Load base encoder + projection head
PATH = './cifar10_NewSimCLR_Res_Epoch=1000_BatchSize=512_lr=0.001_T=0.5.pt'
# torch.save(model.state_dict(), PATH)
model.load_state_dict(torch.load(PATH))
# model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))




############################################################################### Train Linear Classifier
train_transform_cls = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])


train_dataset = torchvision.datasets.CIFAR10(root='./',
                                       train=True,
                                       transform=train_transform_cls)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=bs, 
                                           shuffle=True)


classifier = LinearClassifier(1024,10).to(device)
# classifier = nn.DataParallel(classifier,device_ids = [0, 1])
# classifier.to(device)
criterion = nn.CrossEntropyLoss()
opt1 = optim.Adam(classifier.parameters(), lr=learning_rate)


# for t in range(num_epochs):
#     # Linear Classifier Training
#     total_loss_cls, total_err_train_cls = 0., 0.
#     for idx, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#         with torch.no_grad():
#               h= model.encoder(images)
#               # h= model(images)
#         yp = classifier(h)
#         total_err_train_cls += (yp.max(dim=1)[1] != labels).sum().item()
#         loss_train_cls = criterion(yp,labels)
#         opt1.zero_grad()
#         loss_train_cls.backward()
#         opt1.step()
#         #total_loss_cls += 1
#         total_loss_cls += loss_train_cls.item() * labels.shape[0]
#     print('Epoch =' , t , 'Loss_Train_cls =', total_loss_cls/len(train_loader.dataset),'Error_Train =',total_err_train_cls/len(train_loader.dataset), sep="\t")

############################################################################### Load and Save Linear Classifier
PATH = './cifar10_NewCLS_Res_Epoch=50_BatchSize=512_lr=0.001_T=0.5.pt'
# model.load_state_dict(torch.load(PATH))

# PATH = './cifar10_NewSimCLR_Res_Epoch=120_BatchSize=512_lr=0.001.pt'


# PATH = './cifar10_NewCLS_Res_Epoch=1000_BatchSize=512_lr=0.001_T=0.2.pt'
# torch.save(classifier.state_dict(), PATH)
classifier.load_state_dict(torch.load(PATH))

############################################################################### Test phase
val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

test_dataset = torchvision.datasets.CIFAR10(root='./',
                                            train=False, 
                                            transform=val_transform)
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size= bs, shuffle=(train_sampler is None))


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

############################################################################### FGSM Scheme 1 and Scheme 2
total_loss_test_adv1,total_err_test_adv1 = 0., 0.
for idx, (X, y) in enumerate(test_loader):
    X,y = X.to(device), y.to(device)
    delta = torch.zeros_like(X, requires_grad=True)
    z = model(X)
    z_adv = model(X+delta)
    features1 = torch.cat([z.unsqueeze(1), z_adv.unsqueeze(1)], dim=1)
    criterion = SupConLoss()
    loss_Sup = criterion(features1)
    opt.zero_grad()
    loss_Sup.backward()
    epsilon = 4/255
    delta_adv1 = epsilon * delta.grad.detach().sign()
    h = model.encoder(X+delta_adv1)
    yp = classifier(h)
    loss_test_adv1 = nn.CrossEntropyLoss()(yp,y)
    total_loss_test_adv1 += loss_test_adv1.item() * y.shape[0]
    total_err_test_adv1 += (yp.max(dim=1)[1] != y).sum().item()
print('Loss_Test_adv =',total_loss_test_adv1/len(test_loader.dataset), 'Error_Test_adv =', total_err_test_adv1/ len(test_loader.dataset),sep="\t")


############################################################################### PGD Scheme 1 and Scheme 2
# total_loss_test_adv1,total_err_test_adv1 = 0., 0.
# num_iter = 100
# epsilon = 4/255
# alpha =0.01
# for idx, (X, y) in enumerate(test_loader):
#     X,y = X.to(device), y.to(device)
#     delta = torch.zeros_like(X, requires_grad=True)
#     for t in range(num_iter):
#         z = model(X)
#         z_adv = model(X+delta)
#         features1 = torch.cat([z.unsqueeze(1), z_adv.unsqueeze(1)], dim=1)
#         criterion = SupConLoss()
#         loss_Sup = criterion(features1)
#         opt.zero_grad()
#         loss_Sup.backward(retain_graph=True)
        
#         delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
#         delta.grad.zero_()
#     delta_adv1 = delta.detach()
#     h = model.encoder(X+delta_adv1)
#     yp = classifier(h)
#     loss_test_adv1 = nn.CrossEntropyLoss()(yp,y)
#     total_loss_test_adv1 += loss_test_adv1.item() * y.shape[0]
#     total_err_test_adv1 += (yp.max(dim=1)[1] != y).sum().item()
# print('Loss_Test_adv =',total_loss_test_adv1/len(test_loader.dataset), 'Error_Test_adv =', total_err_test_adv1/ len(test_loader.dataset),sep="\t")
 
       
############################################################################### FGSM Scheme 3
total_loss_test_adv1,total_err_test_adv1 = 0., 0.
for idx, (X, y) in enumerate(test_loader):
    X,y = X.to(device), y.to(device)
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(classifier(model.encoder(X + delta)), y)
    loss.backward()
    epsilon = 16/255
    delta_adv1 = epsilon * delta.grad.detach().sign()
    h = model(X+delta_adv1)
    yp = classifier(h)
    loss_test_adv1 = nn.CrossEntropyLoss()(yp,y)
    total_loss_test_adv1 += loss_test_adv1.item() * y.shape[0]
    total_err_test_adv1 += (yp.max(dim=1)[1] != y).sum().item()
print('Loss_Test =',total_loss_test_adv1/len(test_loader.dataset), 'Error_Test =', total_err_test_adv1 / len(test_loader.dataset),sep="\t")


############################################################################### PGD scheme 3
# total_loss_test_adv2,total_err_test_adv2 = 0., 0.
# for idx, (X, y) in enumerate(test_loader):
#     X,y = X.to(device), y.to(device)
#     delta = torch.zeros_like(X, requires_grad=True)
#     num_iter = 100
#     alpha = 0.01
#     epsilon = 8/255
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













   


