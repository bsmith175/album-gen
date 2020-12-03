import torch
import numpy as np
from preprocess import get_data

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = torch.nn.Conv2d(3, 128, 4, stride=2, padding=1)
        # self.batch_norm1 = torch.nn.BatchNorm2d(128)
        # self.conv2 = torch.nn.Conv2d(128, 256, 4, stride=2, padding=1)
        # self.batch_norm2 = torch.nn.BatchNorm2d(256)
        # self.conv3 = torch.nn.Conv2d(256, 512, 4, stride=2, padding=1)
        # self.batch_norm3 = torch.nn.BatchNorm2d(512)
        # self.conv4 = torch.nn.Conv2d(512, 1024, 4, stride=2, padding=1)
        # self.batch_norm4 = torch.nn.BatchNorm2d(1024)
        pad = 2
        num_categories = 5
        self.conv1 = torch.nn.Conv2d(3, 64, 5, stride=2, padding=pad)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, 5, stride=2, padding=pad)
        self.batch_norm2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, 5, stride=2, padding=pad)
        self.batch_norm3 = torch.nn.BatchNorm2d(256)
        self.conv4 = torch.nn.Conv2d(256, 512, 5, stride=2, padding=pad)
        self.batch_norm4 = torch.nn.BatchNorm2d(512)
        self.dense = torch.nn.Linear(8192, num_categories)
        self.learning_rate = 0.00002
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))

    def forward(self, X):
        X = torch.nn.functional.leaky_relu(self.batch_norm1(self.conv1(X)), negative_slope=0.1)
        X = torch.nn.functional.leaky_relu(self.batch_norm2(self.conv2(X)), negative_slope=0.1)
        X = torch.nn.functional.leaky_relu(self.batch_norm3(self.conv3(X)), negative_slope=0.1)
        X = torch.nn.functional.leaky_relu(self.batch_norm4(self.conv4(X)), negative_slope=0.1)
        # X = torch.reshape(X, (-1, 16384))
        X = torch.reshape(X, (-1, 8192))
        return self.dense(X)

    def loss(self, logits, labels):
        return torch.nn.functional.cross_entropy(logits, labels, reduction='mean')

    def accuracy(self, logits, labels):
        predicted = torch.argmax(logits, 1)
        # true_label = torch.argmax(labels, 1)
        matches = torch.eq(predicted, labels)
        matches = matches.type(torch.FloatTensor)
        return torch.mean(matches).item()


def train(net, inputs, labels):
    logits = net(inputs)
    loss = net.loss(logits, labels)
    loss.backward()
    net.optimizer.step()
    return net.accuracy(logits, labels)

def test(net, inputs, labels):
    logits = net(inputs)
    return net.accuracy(logits, labels)


if torch.cuda.is_available():
    dev = 'cuda:0'
    print("Training on GPU")
else:
    dev = 'cpu'
    print("Training on CPU")

dev = torch.device(dev)

net = Net().to(dev)
net = net.to(dev)
net = net.double()
# trials = 10
# X = np.random.normal(size=(trials,3,64,64))
# Y = np.zeros(trials)
# for i in range(trials):
#     Y[i] = i%5
# X_tensor = torch.from_numpy(X)
# X_tensor = X_tensor.double()
# Y_tensor = torch.from_numpy(Y)
# Y_tensor = Y_tensor.long()



num_epochs = 500
batch_size = 128
for epoch in range(num_epochs):
    print("Epoch: ", epoch)
    train_acc = 0
    test_acc = 0
    count = 0
    for batch in get_data('data/inputs.npy', 'data/labels.npy', batch_size):
        count += 1
        X = batch[0]
        Y = batch[1]
        X = torch.from_numpy(X).to(dev)
        X = X.double()
        Y = torch.from_numpy(Y).to(dev)
        Y = Y.long()
        if count < 58:
            train_acc += train(net, X, Y)
        else:
            test_acc += test(net, X, Y)
    print("Train Accuracy:", train_acc / count)
    print("Test Accuracy:", test_acc / count)


