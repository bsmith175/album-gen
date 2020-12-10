import torch
import numpy as np

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.num_categories = 5
        # self.conv1 = torch.nn.Conv2d(3, 128, 4, stride=2, padding=1)
        # self.batch_norm1 = torch.nn.BatchNorm2d(128)
        # self.conv2 = torch.nn.Conv2d(128, 256, 4, stride=2, padding=1)
        # self.batch_norm2 = torch.nn.BatchNorm2d(256)
        # self.conv3 = torch.nn.Conv2d(256, 512, 4, stride=2, padding=1)
        # self.batch_norm3 = torch.nn.BatchNorm2d(512)
        # self.conv4 = torch.nn.Conv2d(512, 1024, 4, stride=2, padding=1)
        # self.batch_norm4 = torch.nn.BatchNorm2d(1024)
        # self.dense = torch.nn.Linear(16384, self.num_categories)

        self.pad = 2
        self.stride = 2
        self.conv1 = torch.nn.Conv2d(3, 64, 5, stride=self.stride, padding=self.pad)
        self.batch_norm1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, 5, stride=self.stride, padding=self.pad)
        self.batch_norm2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, 5, stride=self.stride, padding=self.pad)
        self.batch_norm3 = torch.nn.BatchNorm2d(256)
        self.conv4 = torch.nn.Conv2d(256, 512, 5, stride=self.stride, padding=self.pad)
        self.batch_norm4 = torch.nn.BatchNorm2d(512)
        self.dense1 = torch.nn.Linear(8192, 1)
        self.dense2 = torch.nn.Linear(8192, self.num_categories)
        self.dense3 = torch.nn.Linear(8192, 2)
        self.dropout = torch.nn.Dropout()
        self.dropout2d = torch.nn.Dropout2d()
        self.learning_rate = .0001
        self.beta1 = 0.5
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999))
        self.latent_loss = torch.nn.MSELoss()
        self.softmax = torch.nn.Softmax()

    def forward(self, X):
        X = torch.nn.functional.leaky_relu(self.batch_norm1(self.conv1(X)), negative_slope=0.1)
        X = torch.nn.functional.leaky_relu(self.batch_norm2(self.conv2(X)), negative_slope=0.1)
        X = torch.nn.functional.leaky_relu(self.batch_norm3(self.conv3(X)), negative_slope=0.1)
        X = torch.nn.functional.leaky_relu(self.batch_norm4(self.conv4(X)), negative_slope=0.1)
        X = X.view(-1, 8192)
        X = self.dropout(X)
        return self.dense1(X), self.dense2(X), self.dense3(X)

    def real_loss(self, logits, labels):
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def class_loss(self, logits, labels):
        return torch.nn.functional.cross_entropy(logits, labels, reduction='mean')

    def real_accuracy(self, predicted, labels):
        logits = self.softmax(predicted)
        print(logits)
        logits = logits > 0.5
        return (logits == labels).sum().item() / labels.size(0)

    def class_accuracy(self, logits, labels):
        predicted = torch.argmax(logits, 1)
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

if __name__ == "__main__":
    if torch.cuda.is_available():
        dev = 'cuda:0'
        print("Training on GPU")
    else:
        dev = 'cpu'
        print("Training on CPU")
    dev = torch.device(dev)
    to_load = False
    PATH = "net.pth"
    net = Net().to(dev)
    net = net.double()
    if to_load:
        net.load_state_dict(torch.load(PATH))

    num_epochs = 500
    batch_size = 128
    for epoch in range(num_epochs):
        print("Epoch: ", epoch)
        train_acc = 0
        test_acc = 0
        count = 0
        net.train()
        for batch in get_data('./data/inputs.npy', './data/labels.npy', batch_size):
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
                net.eval()
                test_acc += test(net, X, Y)
        print("Train Accuracy:", train_acc / count)
        print("Test Accuracy:", test_acc / count)

    torch.save(net.state_dict(), PATH)



