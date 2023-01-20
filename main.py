from duckduckgo_search import ddg_images
search_terms = "bike","car"
for index,search in enumerate(search_terms):
  ddg_images(search,max_results=200,download=(True,f"{search}"))

# make sure that there is no broken images

# failed = verify_images(get_image_files(path))
# failed.map(Path.unlink)
# len(failed)

"""
#for ssl error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass

else:
    ssl._create_default_https_context = _create_unverified_https_context
"""

from fastai.vision.all import *
path= Path('bike_or_not') # specify the parent image folder path here
dls=DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2,seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192,method='squish')]
).dataloaders(path,bs=32) # bs= 2^n 

dls.show_batch(max_n=64) # gives a overview

learn=vision_learner(dls,resnet18,metrics=error_rate)

learn.fit(5,lr=1e-3) # with learning rate 0.001

learn.validate() # return 2 values , [loss_value,accuracy_metric]

learn.show_results() # show images from validation set or test test and predict the output 

# learn.fine_tune(3) 

pred_class,pred_idx,outputs =learn.predict('bike3.jpeg')

print(pred_class)

# learn.save('model_2') # exports the .pth file, which can be loaded by load_model()
