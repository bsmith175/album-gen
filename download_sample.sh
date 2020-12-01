#!/usr/bin/env bash

mkdir -p ./data/omacir
cd ./data/omacir
curl -LO "https://archive.org/download/audio-covers/album_covers_x.tar"
tar -xvf "album_covers_x.tar"
rm -rf "album_covers_x.tar"
