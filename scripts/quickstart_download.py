import os
import gdown

url = 'https://drive.google.com/uc?id=1w8yzqrIf-bTXp6NazQ_o8V-xiJB3tlli'
output = 'cliport_quickstart.zip'

gdown.download(url, output, quiet=False)
os.system(f'unzip {output}')
os.system(f'rm {output}')