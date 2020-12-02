import torch
import numpy as np
from preprocess import get_data

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.batch_size = 128
        self.stride = 2
        self.kernel_size = 5

        self.cat_dim = 2
        self.con_dim = 2
        self.rand_dim = 100 #idk what these are needed for
        self.noise_dim = self.cat_dim + self.con_dim + self.rand_dim
        self.padding = 0

        self.dense = torch.nn.Linear(self.noise_dim, 8192, True) #linear: in_feats, out_feats, bias
        self.norm1 = torch.nn.BatchNorm2d(512) #paper's batch norm uses tf maximum?
        self.norm2 = torch.nn.BatchNorm2d(256)
        self.norm3 = torch.nn.BatchNorm2d(128)
        self.norm4 = torch.nn.BatchNorm2d(64)
        self.convt1 = torch.nn.ConvTranspose2d(512, 256, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.convt2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.convt3 = torch.nn.ConvTranspose2d(128, 64, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.convt4 = torch.nn.ConvTranspose2d(64, 3, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        self.optimizer = torch.optim.Adam(self.parameters(), 1e-3)

    def call(self, cat, con, rand):
        noise = torch.cat([cat, con, rand], 1)
        g0 = self.dense(noise)
        #batchNorm2d checks 2nd index for num_features...doing this reshape instead of (self.batch_size, 4, 4, 512)
        g1 = torch.reshape(g0, (self.batch_size, 512, 4, 4))
        print("g1 shape: ", g1.shape)
        g1_b = self.norm1(g1)
        g2 = torch.nn.functional.leaky_relu(self.convt1(g1_b), negative_slope=0.1)
        print("g2 shape: ", g2.shape)
        g2_b = self.norm2(g2)
        g3 = torch.nn.functional.leaky_relu(self.convt2(g2_b), negative_slope=0.1)
        print("g3 shape: ", g3.shape)
        g3_b = self.norm3(g3)
        g4 = torch.nn.functional.leaky_relu(self.convt3(g3_b), negative_slope=0.1)
        print("g4 shape: ", g4.shape)
        g4_b = self.norm4(g4)
        g5 = self.convt4(g4_b)
        print("g5 shape: ", g5.shape)
        return torch.tanh(g5)

    def loss(self, D_logits): #discriminator logits?
        return torch.nn.functional.cross_entropy(logits=D_logits_, labels=torch.ones_like(D_logits), reduction='mean') #ones like?

    def accuracy(self, logits, labels): # discriminator labels again?
        predicted = torch.argmax(logits, 1)
        matches = torch.eq(predicted, labels)
        matches = matches.type(torch.FloatTensor)
        return torch.mean(matches).item()

def train(self, inputs, labels):
    logits = self(inputs)
    loss = self.loss(logits, labels)
    loss.backward()
    self.optimizer.step()
    return self.accuracy(logits, labels)


def main():
    print("running generator")
    # if torch.cuda.is_available():
    #     dev = 'cuda:0'
    #     print("Training on GPU")
    # else:
    #     dev = 'cpu'
    #     print("Training on CPU")
    # #dev = 'cuda'
    # dev = torch.device(dev)
    # to_load = True
    # PATH = "gen.pth"
    # gen = Generator().to(dev)
    gen = Generator()
    cat_dim = 2
    con_dim = 2
    rand_dim = 100 #idk what these are needed for
    noise_dim = cat_dim + con_dim + rand_dim
    batch_size = 128
    #testing shapes - Got device type cpu error??
    cat = torch.Tensor(np.zeros((batch_size, cat_dim)))
    con = torch.Tensor(np.zeros((batch_size, con_dim)))
    rand = torch.Tensor(np.zeros((batch_size, rand_dim)))
    gen.call(cat, con, rand)




if __name__ == '__main__':
    main()
