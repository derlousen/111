# class ONet(nn.Module):
#     def __init__(self):
#         super(ONet, self).__init__()
#         self.model_path, _ = os.path.split(os.path.realpath(__file__))
#         self.features = nn.Sequential(OrderedDict([
#             ('conv1', nn.Conv2d(3, 32, 3, 1)),
#             ('prelu1', nn.PReLU(32)),
#             ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),
#             ('conv2', nn.Conv2d(32, 64, 3, 1)),
#             ('prelu2', nn.PReLU(64)),
#             ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),
#             ('conv3', nn.Conv2d(64, 64, 3, 1)),
#             ('prelu3', nn.PReLU(64)),
#             ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),
#             ('conv4', nn.Conv2d(64, 128, 2, 1)),
#             ('prelu4', nn.PReLU(128)),
#             ('flatten', Flatten()),
#             ('conv5', nn.Linear(1152, 256)),
#             ('drop5', nn.Dropout(0.25)),
#             ('prelu5', nn.PReLU(256)),
#         ]))
#         self.conv6_1 = nn.Linear(256, 2)
#         self.conv6_2 = nn.Linear(256, 4)
#         self.conv6_3 = nn.Linear(256, 136)
#         # weights = np.load(os.path.join(self.model_path, 'weights', 'onet.npy'))[()]
#         # for n, p in self.named_parameters():
#         #     p.data = torch.FloatTensor(weights[n])
#
#     def forward(self, x):
#         x = self.features(x)
#         a = self.conv6_1(x)
#         b = self.conv6_2(x)
#         c = self.conv6_3(x)
#         a = F.softmax(a, dim=1)
#         return a, b, c


# class RNet(nn.Module):
#     def __init__(self):
#         super(RNet, self).__init__()
#         self.model_path, _ = os.path.split(os.path.realpath(__file__))
#         self.features = nn.Sequential(OrderedDict([
#             ('conv1', nn.Conv2d(3, 28, 3, 1)),
#             ('prelu1', nn.PReLU(28)),
#             ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),
#             ('conv2', nn.Conv2d(28, 48, 3, 1)),
#             ('prelu2', nn.PReLU(48)),
#             ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),
#             ('conv3', nn.Conv2d(48, 64, 2, 1)),
#             ('prelu3', nn.PReLU(64)),
#             ('flatten', Flatten()),
#             ('conv4', nn.Linear(576, 128)),
#             ('prelu4', nn.PReLU(128))
#         ]))
#         self.conv5_1 = nn.Linear(128, 2)
#         self.conv5_2 = nn.Linear(128, 4)
#         # weights = np.load(os.path.join(self.model_path, 'weights', 'rnet.npy'))[()]
#         # for n, p in self.named_parameters():
#         #     p.data = torch.FloatTensor(weights[n])
#
#     def forward(self, x):
#         x = self.features(x)
#         a = self.conv5_1(x)
#         b = self.conv5_2(x)
#         # b = F.sigmoid(b)
#         a = F.softmax(a, dim=1)
#         return a, b

# class PNet(nn.Module):
#     def __init__(self):
#         super(PNet, self).__init__()
#         self.model_path, _ = os.path.split(os.path.realpath(__file__))
#         self.features = nn.Sequential(OrderedDict([
#             ('conv1', nn.Conv2d(3, 10, 3, 1)),
#             ('prelu1', nn.PReLU(10)),
#             ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),
#             ('conv2', nn.Conv2d(10, 16, 3, 1)),
#             ('prelu2', nn.PReLU(16)),
#             ('conv3', nn.Conv2d(16, 32, 3, 1)),
#             ('prelu3', nn.PReLU(32))
#         ]))
#
#         self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
#
#         # ############################
#         # self.conv4_2 = nn.Conv2d(32, 4, 1, 1)
#         # ############################
#
#         ############################
#         self.conv4_2 = nn.Conv2d(32, 32, 1, 1)
#         self.conv4_3 = nn.Conv2d(32, 4, 1, 1)
#         ############################
#
#         # weights = np.load(os.path.join(self.model_path, 'weights', 'pnet.npy'))[()]
#         # for n, p in self.named_parameters():
#         #     p.data = torch.FloatTensor(weights[n])
#         self.std_1 = 0
#
#     def forward(self, x):
#         # if self.training:
#         #     self.std_1 = torch.abs(torch.sum(self.features.conv1.weight))
#         # if self.std_1 == 0:
#         #     self.std_1 = torch.abs(torch.sum(self.features.conv1.weight))
#
#         x = self.features(x)
#         # self.std_1.data.to(x.device)
#
#         # y = self.std_1.data.to(x.device) * (torch.randn(x.size()).data.to(x.device))
#         # y.data.to(self.device)
#
#         # print(self.std_1)
#         # x = x+y
#         # print(torch.sum(self.features.conv3.weight))
#
#         a = self.conv4_1(x)
#         b = self.conv4_2(x)
#         b = self.conv4_3(b)
#         a = F.softmax(a, dim=1)
#         return a, b