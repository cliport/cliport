import os
import gdown

url = 'https://drive.google.com/uc?id=1xMB0jKKfpigsHlwr72Kq9plNJRjMkl_g'
output = 'google.zip'

gdown.download(url, output, quiet=False)
os.system(f'unzip {output}')
os.system(f'mv google cliport/environments/assets')
os.system(f'rm {output}')