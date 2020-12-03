import requests
import util
import csv
import os
from PIL import Image

API_KEY = 'ZWY5YmEzNTItNjczMS00ODEzLTg2ZTMtNjYxMDkzMmJkY2Jj'
API = 'https://api.napster.com/'
IMG_PER_ALBUM = 1000
IMG_DIR = './data/'
genres = ['rock', 'jazz', 'pop', 'rap-hip-hop', 'classical']

# genres = ['rap-hip-hop']

def get_genre_ids(shortcuts):
    print("Getting genre ids ...")
    ids = list(map(lambda item: requests.get(API + 'v2.2/genres/' + item + '?apikey=' + API_KEY).json()['genres'][0]['id'], shortcuts))
    print(ids)
    return list(zip(ids, genres))

def get_imgs_from_album(album_ids):
    response = requests.get(API + 'v2.2/albums/' + ",".join(album_ids) + '/images?apikey=' + API_KEY).json()['images']
    content_ids = set()
    urls = []
    for album in response:
        if album['contentId'] not in content_ids:
            urls.append(album['url'])
            content_ids.add(album['contentId'])
    print(len(urls))
    return urls, content_ids

covers_for_genre = {}
for id, name in get_genre_ids(genres):
    downloaded = set()
    print('\n\nGetting images for genre: ' + name + ' (' + id + ')' '...')

    if not os.path.exists(IMG_DIR + name):
        print("making directory: " + IMG_DIR + name)
        os.makedirs(IMG_DIR + name)
    ranges = ['day', 'week', 'month', 'year', 'life']
    for period_count, period in enumerate(ranges): 
        for i in range(0, 1000, 200):
            print('\nbatch ' + str(int(i / 200)) + "...")
            try:
                albums = requests.get(API + 'v2.2/genres/' + id + '/albums/top?limit=200&offset=' + str(i) + '&range=' + period + '&apikey=' + API_KEY).json()['albums']
            except Exception as e:
                print(e)
                continue
            print('API called...')
            albums = list(filter(lambda x: x['id'] not in downloaded , albums))
            ids = list(map(lambda alb: alb['id'], albums))

            downloaded.update(ids)
            if len(ids) == 0:
                continue
            urls, content_ids = get_imgs_from_album(ids)
            for ix, (url, content_id)  in enumerate(zip(urls, content_ids)):
                try:
                    response = requests.get(url, stream=True)
                except Exception as e:
                    print(e)
                    print(url)
                    continue
                if response.status_code == 200:
                    img = Image.open(response.raw)
                    img.save(IMG_DIR + name + '/' + content_id + '.jpg')
                else:
                    print("Error: " + str(response.status_code))


# covers_for_genre = {}
# for id, name in get_genre_ids(genres):
#     print('\n\nGetting images for genre: ' + name + ' (' + id + ')' '...')
#     covers_for_genre[name] = []

#     if not os.path.exists(IMG_DIR + name):
#         print("making directory: " + IMG_DIR + name)
#         os.makedirs(IMG_DIR + name)

#     for i in range(0, IMG_PER_ALBUM, 200):
#         print('\nbatch ' + str(int(i / 200)) + "...")
#         albums = requests.get(API + 'v2.2/genres/' + id + '/albums/top?limit=200&offset=' + str(i) + '&apikey=' + API_KEY).json()['albums']
#         print('API called...')
#         ids = list(map(lambda alb: alb['id'], albums))
#         urls = get_imgs_from_album(ids)
#         covers_for_genre[name].append(urls)
#         for ix, url  in enumerate(urls):
#             response = requests.get(url, stream=True)
#             if response.status_code == 200:
#                 img = Image.open(response.raw)
#                 img.save(IMG_DIR + name + '/' + str(i + ix) + '.jpg')
#             else:
#                 print("Error: " + str(response.status_code))

    # with open(IMG_DIR + name + '.csv', 'w', newline='') as cur_file:
    #     wr = csv.writer(cur_file)
    #     print("\nwriting to " + name + '.csv')
    #     wr.writerow(covers_for_genre[name])




# response = requests.get(API + 'v2.2/genres/g.5/albums/top?limit=1&apikey=' + API_KEY)
# response = response.json()

# util.pretty_print_json(response.json()['albums'][0]['id'])
# get_genre_ids(genres)