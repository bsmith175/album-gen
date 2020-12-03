from discriminator import Discriminator
from generator import Generator
from preprocess import get_data
import torch
import numpy as np
from PIL import Image


def main():

    if torch.cuda.is_available():
        dev = 'cuda:0'
        print("Training on GPU")
    else:
        dev = 'cpu'
        print("Training on CPU")
    #dev = 'cuda'
    dev = torch.device(dev)
    discriminator = Discriminator().to(dev)
    ##call .eval() before testing
    ##and call .train() if you are going to train again
    generator = Generator().to(dev)
    num_epochs = 1
    batch_size = 128
    cat_dim = 5
    con_dim = 2
    rand_dim = 100

    ##call .eval() before testing
    ##and call .train() if you are going to train again

    for epoch in range(num_epochs):
        for real_images, cat_labels in get_data('data/inputs.npy', 'data/labels.npy', batch_size):
            real_images = torch.from_numpy(real_images).to(dev)
            cat_labels = torch.from_numpy(cat_labels).long().to(dev)
            real_logits, real_cat_logits, _ = discriminator(real_images)

            discriminator.zero_grad()
            d_real_loss = discriminator.loss(real_logits, torch.ones_like(cat_labels))
            d_real_cat_loss = discriminator.loss(real_cat_logits, cat_labels)
            real_d_score = d_real_loss + d_real_cat_loss * 10

            z_cat = torch.Tensor(np.random.uniform(0, 1, size=[batch_size, cat_dim]).astype(np.float32)).to(dev)
            z_con = torch.Tensor(np.random.uniform(-1, 1, size=[batch_size, con_dim]).astype(np.float32)).to(dev)
            z_rand = torch.Tensor(np.random.uniform(-1, 1, size=[batch_size, rand_dim]).astype(np.float32)).to(dev)
            fake_images = generator(z_cat, z_con, z_rand)
            fake_logits, fake_cat_logits, latent_logits = discriminator(fake_images.detach())
            fake_labels = torch.zeros((fake_logits.shape[0],)).long()
            d_fake_loss = discriminator.loss(fake_logits, fake_labels)
            cats = torch.argmax(z_cat, 1)
            d_fake_cat_loss = discriminator.loss(fake_cat_logits, cats)
            latent_loss = discriminator.latent_loss(latent_logits, z_con)
            fake_d_score = d_fake_loss + d_fake_cat_loss * 10 + latent_loss
            d_score = real_d_score + fake_d_score
            d_score.backward()
            discriminator.optimizer.step()

            generator.zero_grad()
            fake_logits, fake_cat_logits, _ = discriminator(fake_images)
            cats = torch.argmax(z_cat, 1)
            d_fake_cat_loss = discriminator.loss(fake_cat_logits, cats)
            g_loss = generator.loss(fake_logits)
            g_score = g_loss + d_fake_cat_loss * 10
            g_score.backward()
            generator.optimizer.step()

    generator.eval()
    test_size = 1
    z_cat = torch.Tensor(np.random.uniform(0, 1, size=[test_size, cat_dim]).astype(np.float32))
    z_con = torch.Tensor(np.random.uniform(-1, 1, size=[test_size, con_dim]).astype(np.float32))
    z_rand = torch.Tensor(np.random.uniform(-1, 1, size=[test_size, rand_dim]).astype(np.float32))
    img = generator(z_cat, z_con, z_rand).detach().numpy()
    img = np.rollaxis(img,1, 4)
    img = (img+1) * 127.5
    img = img.astype(np.uint8)
    out_dir = '/Users/jtsatsaros/Documents/album-gen'
    # Convert to uint8
    # Save images to disk
    for i in range(0, test_size):
        img_i = img[i]
        s = out_dir + '/' + str(i) + '.png'
        img_i = Image.fromarray(img_i)
        img_i.save(s)




if __name__ == '__main__':
   main()
