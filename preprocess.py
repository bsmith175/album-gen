from PIL import ImageOps, Image
import numpy as np
import pickle
import os
def make_pickle(genres, data_dir, image_dims, force=False):
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

# yields tuple of 2 numpy arrays of shape (batch_size, num_labels, 64, 64) and (batch_size,)
def get_data(input_file_path, label_file_path, batch_size):
    input_data = np.load(input_file_path)
    labels = np.load(label_file_path)
    indices = (np.arange(input_data.shape[0]))
    np.random.shuffle(indices)
    input_data = np.take(input_data, indices, 0)
    labels = np.take(labels, indices, 0)
    for i in range(0, labels.shape[0], batch_size):
        yield input_data[i:min(input_data.shape[0], i + batch_size)], labels[i: min(labels.shape[0], i + batch_size)]



# genres = ['rock', 'reggae', 'jazz', 'pop', 'country', 'rap-hip-hop', 'classical']
# make_pickle(genres, './data/', (64, 64), force=False)
# for batch in get_data('./data/inputs.npy', './data/labels.npy', 200):
#     breakpoint()
#     inputs, labels = batch