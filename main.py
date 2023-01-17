from duckduckgo_search import ddg_images
search_terms = "bike","car"
for index,search in enumerate(search_terms):
  ddg_images(search,max_results=200,download=(True,f"{search}"))

# Damaged image breaks the training 
# failed = verify_images(get_image_files(path))
# failed.map(Path.unlink)
# len(failed)
  
from fastai.vision.all import *
path= Path('bike_or_not')
dls=DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2,seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192,method='squish')]
).dataloaders(path,bs=32)

dls.show_batch(max_n=20)

learn=vision_learner(dls,resnet18,metrics=error_rate)
learn.fine_tune(3)

pred_class,pred_idx,outputs =learn.predict('bike3.jpeg')
print(pred_class)
