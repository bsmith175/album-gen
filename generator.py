import torch
import numpy as np
from preprocess import get_data

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.batch_size = 128
        self.stride = 2
        self.kernel_size = 5
        self.cat_dim = 5
        self.con_dim = 2
        self.rand_dim = 100
        self.noise_dim = self.cat_dim + self.con_dim + self.rand_dim
        self.pad = 2
        self.out_pad = 1

        self.dense = torch.nn.Linear(self.noise_dim, 8192)
        self.convt1 = torch.nn.ConvTranspose2d(512, 256, self.kernel_size, stride=self.stride, padding=self.pad, output_padding=self.out_pad)
        self.norm1 = torch.nn.BatchNorm2d(512)
        self.convt2 = torch.nn.ConvTranspose2d(256, 128, self.kernel_size, stride=self.stride, padding=self.pad, output_padding=self.out_pad)
        self.norm2 = torch.nn.BatchNorm2d(256)
        self.convt3 = torch.nn.ConvTranspose2d(128, 64, self.kernel_size, stride=self.stride, padding=self.pad, output_padding=self.out_pad)
        self.norm3 = torch.nn.BatchNorm2d(128)
        self.convt4 = torch.nn.ConvTranspose2d(64, 3, self.kernel_size, stride=self.stride, padding=self.pad, output_padding=self.out_pad)
        self.norm4 = torch.nn.BatchNorm2d(64)
        self.learning_rate = 1e-5
        self.beta1 = 0.5
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999))

    def forward(self, cat, con, rand):
        noise = torch.cat([cat, con, rand], 1)
        g0 = self.dense(noise)
        g1 = self.norm1(torch.reshape(g0, (-1, 512, 4, 4)))
        g2 = self.norm2(torch.nn.functional.relu(self.convt1(g1)))
        g3 = self.norm3(torch.nn.functional.relu(self.convt2(g2)))
        g4 = self.norm4(torch.nn.functional.relu(self.convt3(g3)))
        g5 = self.convt4(g4)
        return torch.tanh(g5)

    def loss(self, fake_logits):
        fake_labels = torch.ones((fake_logits.shape[0],)).long().to(dev)
        return torch.nn.functional.cross_entropy(fake_logits, fake_labels, reduction='mean')


def main():
    print("running generator")
    if torch.cuda.is_available():
        dev = 'cuda:0'
        print("Training on GPU")
    else:
        dev = 'cpu'
        print("Training on CPU")
    #dev = 'cuda'
    dev = torch.device(dev)
    to_load = True
    PATH = "gen.pth"
    gen = Generator().to(dev)
    # gen = Generator()
    cat_dim = 2
    con_dim = 2
    rand_dim = 100
    noise_dim = cat_dim + con_dim + rand_dim
    batch_size = 128
    #testing shapes - Got device type cpu error??
    cat = torch.Tensor(np.random.uniform(-1, 1, size=[batch_size, cat_dim]).astype(np.float32)).to(dev)
    con = torch.Tensor(np.random.uniform(-1, 1, size=[batch_size, con_dim]).astype(np.float32)).to(dev)
    rand = torch.Tensor(np.random.uniform(-1, 1, size=[batch_size, rand_dim]).astype(np.float32)).to(dev)




if __name__ == '__main__':
    main()
