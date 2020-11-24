import requests
import util
import csv

API_KEY = 'ZWY5YmEzNTItNjczMS00ODEzLTg2ZTMtNjYxMDkzMmJkY2Jj'
API = 'https://api.napster.com/'
IMG_PER_ALBUM = 1000
IMG_DIR = './data/'

genres = ['rock', 'reggae', 'jazz', 'pop', 'country', 'rap-hip-hop', 'classical']

def get_genre_ids(shortcuts):
    print("Getting genre ids ...")
    ids = list(map(lambda item: requests.get(API + 'v2.2/genres/' + item + '?apikey=' + API_KEY).json()['genres'][0]['id'], shortcuts))
    print(ids)
    return list(zip(ids, genres))

def get_imgs_from_album(album_ids):
    response = requests.get(API + 'v2.2/albums/' + ",".join(album_ids) + '/images?apikey=' + API_KEY).json()['images']
    urls = list(map(lambda item: item['url'], response))
    return urls

covers_for_genre = {}
for id, name in get_genre_ids(genres):
    print('Getting images for genre: ' + name + ' (' + id + ')' '...')
    covers_for_genre[name] = []
    for i in range(0, IMG_PER_ALBUM, 200):
        print('\nbatch ' + str(int(i / 200)) + "...")
        albums = requests.get(API + 'v2.2/genres/' + id + '/albums/top?limit=200&offset=' + str(i) + '&apikey=' + API_KEY).json()['albums']
        print('API called...')
        ids = list(map(lambda alb: alb['id'], albums))
        urls = get_imgs_from_album(ids)
        covers_for_genre[name].append(urls)

    with open(IMG_DIR + name + '.csv', 'w', newline='') as cur_file:
        wr = csv.writer(cur_file)
        print("writing to " + name + '.csv')
        wr.writerow(covers_for_genre[name])




# response = requests.get(API + 'v2.2/genres/g.5/albums/top?limit=1&apikey=' + API_KEY)
# response = response.json()

# util.pretty_print_json(response.json()['albums'][0]['id'])
# get_genre_ids(genres)