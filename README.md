# train-a-model
This code is using the fastai library to create a data pipeline, fine-tune a pre-trained deep learning model, and make predictions on images. The data pipeline is defined using the DataBlock class, which is used to process and split the images into training and validation sets. The pre-trained model is fine-tuned on the dataset using the fine_tune method of the Learner class. Then, the fine-tuned model is used to make predictions on a single image using the predict method of the Learner class. The code also uses the duckduckgo_search library to search and download images from the internet, The Images are then resized to 192 pixels by squishing the aspect ratio. The final output will be the prediction of the class of the image. The prediction is made on the image 'bike3.jpeg' which should be present in the same directory where the code is running.
 
```python
from duckduckgo_search import ddg_images
search_terms = "searchterm1","searchterm2"
for index,search in enumerate(search_terms):
  ddg_images(search,max_results=200,download=(True,f"{search}"))
```


 
```python 
from fastai.vision.all import *
path= Path('filepath')
dls=DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2,seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192,method='squish')]
).dataloaders(path,bs=32)

dls.show_batch(max_n=20)
```

+ dls=DataBlock(: This line creates an instance of the DataBlock class, which is used to define the overall structure and behavior of the data pipeline for a machine learning project.
+ blocks=(ImageBlock,CategoryBlock): This line specifies that the data pipeline will include two types of blocks, an ImageBlock and a CategoryBlock. The ImageBlock will handle processing of image data, and the CategoryBlock will handle processing of categorical data.
+ get_items=get_image_files: This line specifies that the get_items function, which is used to retrieve the data items, should be the get_image_files function. This function will be used to find all image files in the specified path.
+ splitter=RandomSplitter(valid_pct=0.2,seed=42): This line specifies that the data should be split into training and validation sets using the RandomSplitter class. The valid_pct argument specifies that 20% of the data should be used for validation and the remaining 80% for training, and the seed argument is used to ensure reproducibility by fixing the random seed used for splitting the data.
+ get_y=parent_label: This line specifies that the get_y function, which is used to retrieve the target values for each data item, should be the parent_label function. This function will be used to extract the parent directory name of the image file as the target label.
+ item_tfms=[Resize(192,method='squish')]: This line specifies that the item_tfms list, which is used to apply transform to each data item, should include the Resize transform. The Resize transform will be used to resize the image to a size of 192 pixels by squishing the aspect ratio.
+ dataloaders(path,bs=32): This line creates the DataLoader object by calling the dataloaders method on the DataBlock instance, passing in the path of the data and the batch size (32) as arguments. This DataLoader object can be used to iterate over the data in batches during training and evaluation.
+ dls.show_batch(max_n=20) : This line is calling the show_batch method on the DataLoader object, which will display a random sample of images from the training set and corresponding labels. The max_n argument is set to 20, which specifies the maximum number of images to be displayed.
