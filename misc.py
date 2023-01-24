from duckduckgo_search import ddg_images
from fastcore.all import *

def search_images(keywords, max_results):
    print(f"Searchin for '{keywords}'.")
    return L(ddg_images(keywords, max_results = max_results)).itemgot('image')
  

from fastdownload import download_url
from fastai.vision.all import *
from time import sleep


search_terms="","","","",""
path = Path('images')

for search in search_terms:
    dest=(path/search)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{search} photos', max_results = 50))
    sleep(3)
    download_images(dest, urls=search_images(f'{search} pictures', max_results = 50))
    resize_images(path/search, max_size=400, dest=path/search)
    
    

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)
