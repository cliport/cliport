#!/bin/bash

# Download Google Scanned Objects
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1xMB0jKKfpigsHlwr72Kq9plNJRjMkl_g' -O google.zip
unzip google.zip
mv google cliport/environments/assets
rm google.zip
