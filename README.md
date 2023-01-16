# Train a machine learning model in 4 steps
This code is using the fastai library to create a data pipeline, fine-tune a pre-trained deep learning model, and make predictions on images. The data pipeline is defined using the DataBlock class, which is used to process and split the images into training and validation sets. The pre-trained model is fine-tuned on the dataset using the fine_tune method of the Learner class. Then, the fine-tuned model is used to make predictions on a single image using the predict method of the Learner class. The code also uses the duckduckgo_search library to search and download images from the internet, The Images are then resized to 192 pixels by squishing the aspect ratio. The final output will be the prediction of the class of the image. The prediction is made on the image 'bike3.jpeg' which should be present in the same directory where the code is running.

### 1. Download images from duckduckgo
```python
from duckduckgo_search import ddg_images
search_terms = "searchterm1","searchterm2"
for index,search in enumerate(search_terms):
  ddg_images(search,max_results=200,download=(True,f"{search}"))
```

### 2. Create data Loader and display 20 images from the training set and the corresponding labels if it exists
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
+ sample model that predict car vs bike based on the photo that It is trained (Turn off dark mode to see the image label) <img src="https://raw.githubusercontent.com/nuhmanpk/train-a-model/main/car-vs-bike.png" alt="car-vs-bike-model"/>

### 3. fine-tune a pre-trained deep learning model on the dataset defined by the DataLoader object dls
```python
learn=vision_learner(dls,resnet18,metrics=error_rate)
learn.fine_tune(3)
```

+ learn=vision_learner(dls,resnet18,metrics=error_rate): This line creates an instance of the Learner class for working with image data and a deep learning model. The vision_learner function is being used to create the Learner instance, which takes the DataLoader object dls, the pre-trained model resnet18 and the evaluation metric error_rate as arguments.
+ learn.fine_tune(3): This line calls the fine_tune method on the Learner object, which will fine-tune the pre-trained model on the dataset defined by the DataLoader object for 3 epochs. During fine-tuning, the model's parameters will be updated based on the dataset and will be used for the task of image classification.
+ This code will fine-tune a pre-trained resnet18 model on the dataset defined by the DataLoader object dls for 3 epochs and use error rate as the evaluation metric.

### 4. Make a prediction on a single image using a pre-trained deep learning model
```python
pred_class,pred_idx,outputs =learn.predict('image_name.jpeg')
print(pred_class)
```

+ pred_class,pred_idx,outputs =learn.predict('image.jpeg'): This line calls the predict method on the Learner object, which will use the fine-tuned model to make a prediction on the image 'image.jpeg'. It will return the predicted class, the index of the predicted class, and the output tensor of the model.
+ print(pred_class): This line is printing the predicted class.

###### This README.md file is fully generated with [Chat GPT](https://chat.openai.com/chat) (Except the image)
