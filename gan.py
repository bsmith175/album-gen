from discriminator import Discriminator
from generator import Generator
from preprocess import get_data
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.linalg import sqrtm

def train_gan(discriminator, generator, num_epochs, gen_save_path, discrim_save_path, fidmodel=None, is_omacir=False):
    batch_size= 128
    cat_dim = 7
    con_dim = 2
    rand_dim = 100

    if torch.cuda.is_available():
        dev = 'cuda:0'
        print("Training on GPU")
    else:
        dev = 'cpu'
        print("Training on CPU")

    dev = torch.device(dev)
    discriminator = Discriminator().to(dev)
    generator = Generator().to(dev)
    if fidmodel:
        fidmodel = fidmodel.to(dev)

    generator.train()
    for epoch in range(num_epochs):
        d_accuracies_real = []
        d_accuracies_fake = []
        d_cat_accuracies_real = []
        d_cat_accuracies_fake = []
        print("Epoch: " + str(epoch))
        for real_images, cat_labels in get_data('/mnt/disks/dsk1/omacir', 'data/labels.npy', batch_size, is_omacir=is_omacir):
            print(type(real_images))
            real_images = torch.from_numpy(real_images).to(dev)
            cat_labels = torch.from_numpy(cat_labels).long().to(dev)
            real_logits, real_cat_logits, _ = discriminator(real_images)

            discriminator.zero_grad()
            d_real_loss = discriminator.loss(real_logits, torch.ones_like(cat_labels))

            d_real_accuracy = discriminator.accuracy(real_logits, torch.ones_like(cat_labels))
            d_accuracies_real.append(d_real_accuracy)
            
            d_real_cat_loss = discriminator.loss(real_cat_logits, cat_labels)
            if not is_omacir:
                d_real_cat_accuracy = discriminator.accuracy(real_cat_logits, cat_labels)
                d_cat_accuracies_real.append(d_real_cat_accuracy)

            real_d_score = d_real_loss + d_real_cat_loss * 10

            z_cat_labels = torch.Tensor(np.random.randint(0, cat_dim-1, size=[batch_size])).long().to(dev)
            z_latent = torch.Tensor(np.random.uniform(-1, 1, size=[batch_size, con_dim]).astype(np.float32)).to(dev)
            z_rand_seed = torch.Tensor(np.random.uniform(-1, 1, size=[batch_size, rand_dim]).astype(np.float32)).to(dev)
            
            fake_images = generator(z_cat_labels, z_latent, z_rand_seed)
            fake_logits, fake_cat_logits, latent_logits = discriminator(fake_images.detach())
            fake_labels = torch.zeros((fake_logits.shape[0],)).long().to(dev)

            d_fake_loss = discriminator.loss(fake_logits, fake_labels)
            d_fake_cat_loss = discriminator.loss(fake_cat_logits, z_cat_labels)
            latent_loss = discriminator.latent_loss(latent_logits, z_latent)
            fake_d_score = d_fake_loss + d_fake_cat_loss * 10 + latent_loss
            d_score = real_d_score + fake_d_score

            d_fake_accuracy = discriminator.accuracy(fake_logits, fake_labels)
            d_accuracies_fake.append(d_fake_accuracy)
            if not is_omacir:
                d_fake_cat_accuracy = discriminator.accuracy(fake_cat_logits, z_cat_labels)
                d_cat_accuracies_fake.append(d_fake_cat_accuracy)

            d_score.backward()
            discriminator.optimizer.step()

            generator.zero_grad()
            fake_logits, fake_cat_logits, _ = discriminator(fake_images)
            d_fake_cat_loss = discriminator.loss(fake_cat_logits, z_cat_labels)
            g_loss = generator.loss(fake_logits, dev)
            g_score = g_loss + d_fake_cat_loss * 10
            g_score.backward()
            generator.optimizer.step()

        print('Saving state...\n')
        torch.save(generator.state_dict(), gen_save_path)
        torch.save(discriminator.state_dict(), discrim_save_path)
        print('Discriminator accuracy on real images: ' + str(sum(d_accuracies_real) / len(d_accuracies_real)))
        print('Discriminator accuracy on generated images: ' + str(sum(d_accuracies_fake) / len(d_accuracies_fake)))
        print('Discriminator category accuracy on real images: ' + str(sum(d_cat_accuracies_real) / len(d_cat_accuracies_real)))
        print('Discriminator category accuracy on fake images: ' + str(sum(d_cat_accuracies_fake) / len(d_cat_accuracies_fake)))
        if fidmodel:
            fid = calc_fid(fidmodel, real_images, fake_images)
            print('FID: ' + str(fid))
    

def test(test_size=1):
    generator.eval()
    z_cat_labels = torch.Tensor(np.random.randint(0, cat_dim - 1, size=[batch_size])).long().to(dev)
    z_latent = torch.Tensor(np.random.uniform(-1, 1, size=[test_size, con_dim]).astype(np.float32)).to(dev)
    z_rand_seed = torch.Tensor(np.random.uniform(-1, 1, size=[test_size, rand_dim]).astype(np.float32)).to(dev)
    img = generator(z_cat_labels, z_latent, z_rand_seed).detach().cpu().numpy()
    img = np.rollaxis(img,1, 4)
    img = (img+1) * 127.5
    img = img.astype(np.uint8)
    out_dir = '.'
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
    model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
    model = model.float()
    model.eval()
    fid = calc_fid(model, images1, images2)


# based on https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
def fid_from_activations(act1, act2):
    act1 = act1.detach().cpu().numpy()
    act2 = act2.detach().cpu().numpy()
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
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
    num_epochs = 20
    num_output_imgs = 1
    discrim_save_path = './discrim_omacir.pth'
    gen_save_path = './gen_omacir.pth'
    discriminator = Discriminator()
    to_load = False

    generator = Generator()
    if to_load:
        print('loading saved GAN state...')
        discriminator.load_state_dict(torch.load(discrim_save_path))
        generator.load_state_dict(torch.load(gen_save_path))

    fidmodel = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
    fidmodel = fidmodel.float()
    fidmodel.eval()
    train_gan(discriminator, generator, num_epochs, gen_save_path, discrim_save_path, fidmodel, is_omacir=True)
    # train_gan(discriminator, generator, num_epochs, gen_save_path, discrim_save_path)

    # test(num_output_imgs)


if __name__ == '__main__':
   main()
