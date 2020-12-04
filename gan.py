from discriminator import Discriminator
from generator import Generator
from preprocess import get_data
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.linalg import sqrtm

def train_gan(discriminator, generator, num_epochs, gen_save_path, discrim_save_path, fidmodel=None):
    batch_size= 128
    cat_dim = 5
    con_dim = 2
    rand_dim = 100

    ##call .eval() before testing
    ##and call .train() if you are going to train again
    generator.train()
    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch))
        for real_images, cat_labels in get_data('data/inputs.npy', 'data/labels.npy', batch_size):
            real_images = torch.from_numpy(real_images)
            cat_labels = torch.from_numpy(cat_labels).long()
            real_logits, real_cat_logits, _ = discriminator(real_images)

            discriminator.zero_grad()
            d_real_loss = discriminator.loss(real_logits, torch.ones_like(cat_labels))
            d_real_cat_loss = discriminator.loss(real_cat_logits, cat_labels)
            real_d_score = d_real_loss + d_real_cat_loss * 10

            z_cat = torch.Tensor(np.random.uniform(0, 1, size=[batch_size, cat_dim]).astype(np.float32))
            z_con = torch.Tensor(np.random.uniform(-1, 1, size=[batch_size, con_dim]).astype(np.float32))
            z_rand = torch.Tensor(np.random.uniform(-1, 1, size=[batch_size, rand_dim]).astype(np.float32))
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

    print('Saving state...\n')
    torch.save(generator.state_dict(), gen_save_path)
    torch.save(discriminator.state_dict(), discrim_save_path)
    if fidmodel:
        fid = calc_fid(fidmodel, real_images, fake_images)
    

    def test(test_size=1):
        generator.eval()
        with torch.no_grad():
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

#takes in two numpy arrays of size (num_images, 3, width, height)
def calc_fid(model, real_img, fake_img):
    preprocess = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    real_batch = preprocess(real_img)
    
    # real_batch = real_img.unsqueeze(0)
    fake_batch = preprocess(fake_img)
    # fake_batch = fake_img.unsqueeze(0)
    with torch.no_grad():
        print(real_batch.shape)
        real_activation = model(real_batch.float())
        fake_activation = model(fake_batch.float())
    fid = fid_from_activations(real_activation, fake_activation)
    return fid

def test_fid():
    images1 = np.random.randint(0, 255, 10*32*32*3).astype('float64')
    images1 = images1.reshape((10,3, 32,32))
    images2 = np.random.randint(0, 255, 10*32*32*3).astype('float64')
    images2 = images2.reshape((10, 3, 32,32))
    images1 = torch.from_numpy(images1)
    images2 = torch.from_numpy(images2)
    print(images1.shape)
    model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
    model = model.float()
    model.eval()
    fid = calc_fid(model, images1, images2)
    print(fid)


# from https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
def fid_from_activations(act1, act2):
    print(act1.shape)
    act1 = act1.numpy()
    act2 = act2.numpy()
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    print(sigma1)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(np.dot(sigma1, sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def main():
    num_epochs = 2
    num_output_imgs = 1
    discrim_save_path = './discrim.pth'
    gen_save_path = './gen.pth'
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(discrim_save_path))

    generator = Generator()
    generator.load_state_dict(torch.load(gen_save_path))

    fidmodel = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
    fidmodel = fidmodel.float()
    fidmodel.eval() 
    train_gan(discriminator, generator, num_epochs, gen_save_path, discrim_save_path, fidmodel)
    test(num_output_imgs)


if __name__ == '__main__':
   main()
