from PIL import ImageOps, Image
import numpy as np
import pickle
import os
def save_data(genres, data_dir, image_dims, force=False):
    if not force and os.path.isfile(data_dir + 'inputs.npy'):
        return 
    input_data = []
    data = []
    labels = []
    for ix, genre in enumerate(genres):
        print('Genre: ' + genre)
        genre_dir = data_dir + genre + '/'
        for root, dirs, files in os.walk(genre_dir):
            for cur_file in files:
                img = Image.open(genre_dir + cur_file)
                try:
                    img = ImageOps.fit(img, image_dims, Image.ANTIALIAS)
                    img = img.convert('RGB')
                    img = np.array(img, dtype=np.float32)/ 127.5 - 1
                    img = np.rollaxis(img, 2)
                    data.append(img)
                    labels.append(ix)
                except Exception as e:
                    print(e)
                    continue
    np.save(data_dir + 'inputs.npy', np.asarray(data)) 
    np.save(data_dir + 'labels.npy', np.asarray(labels)) 

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# yields tuple of 2 numpy arrays of shape (batch_size, 3, 64, 64) and (batch_size,)
def get_data(input_file_path, label_file_path, batch_size, num_classes=5, image_dims=(64, 64), is_omacir=False):
    if is_omacir:
        for path, subdirs, files in os.walk(input_file_path):
            for filename in files:
                input_data = np.load(path + '/' + filename)
                for i in range(0, input_data.shape[0], batch_size):
                    input_batch = input_data[i:min(input_data.shape[0], i + batch_size)]
                    label_batch = np.random.randint(0, num_classes, (input_batch.shape[0],))
                    yield input_batch, label_batch
    else:
        input_data = np.load(input_file_path)
        labels = np.load(label_file_path)
        indices = (np.arange(input_data.shape[0]))
        np.random.shuffle(indices)
        input_data = np.take(input_data, indices, 0)
        labels = np.take(labels, indices, 0)
        print(input_data.shape)
        for i in range(0, labels.shape[0], batch_size):
            yield input_data[i:min(input_data.shape[0], i + batch_size)], labels[i: min(labels.shape[0], i + batch_size)]


def save_omacir(data_dir, image_dims, force=False):
    batch_size = 10000
    data = []
    num_imgs = 0 
    num_batches = 0
    for path, subdirs, files in os.walk(data_dir):

            for filename in files:
                try:
                    img = Image.open(path + '/' + filename)
                    img = ImageOps.fit(img, image_dims, Image.ANTIALIAS)
                    img = img.convert('RGB')
                    img = np.array(img, dtype=np.float32) / 127.5 - 1
                    img = np.rollaxis(img, 2)
                    data.append(img)
                    num_imgs += 1
                except Exception as e:
                    os.remove(path + '/' + filename)
                    print(e)
                if num_imgs == batch_size:
                    num_imgs = 0
                    np.save(data_dir + 'saved/' + str(num_batches) + '.npy', np.asarray(data))
                    print('Batch: ' + str(num_batches) + ' saved to ' + data_dir + 'saved/' + str(num_batches) + '.npy')
                    data = []
                    num_batches += 1

genres = ['rock', 'jazz', 'pop', 'rap-hip-hop', 'classical']
#save_omacir('/mnt/disks/dsk1/omacir/', (64, 64))
#save_data(genres, './data/', (64, 64), force=False)
# for batch in get_data('./data/omacir', './data/labels.npy', 200, 7, (64, 64), True):
#     inputs, labels = batch
#     print(inputs)
