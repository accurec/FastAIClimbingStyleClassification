# FastAIClimbingStyleClassification
### Table of contents
- [Overview](#overview)
- [Technical Details and Project Components](#technical-details-and-project-components)
- [Jupyter Notebook Contents](#jupyter-notebook-contents)
- [Lessons Learned](#lessons-learned)
- [Ways to Improve the ML Model](#ways-to-improve-the-ml-model)

### Overview
This is a "homework" project for the first lesson of the [FastAI](https://course19.fast.ai/) course featuring image classification problem. I am a big fan of rock climbing, so I decided to train my ML model to differentiate between five different styles of climbing:

| Outdoor | Indoor |
| --- | --- |
| Bouldering ![Outdoor bouldering](./Readme_files/Climbing_examples/outdoor_bouldering.jpg) | Bouldering ![Indoor bouldering](./Readme_files/Climbing_examples/indoor_bouldering.jpg) |
| Sport climbing ![Outdoor sport climbing](./Readme_files/Climbing_examples/outdoor_sport_climbing.jpg) | Sport climbing ![Indoor sport climbing](./Readme_files/Climbing_examples/indoor_sport_climbing.jpg) |
| Trad climbing ![Outdoor trad climbing](./Readme_files/Climbing_examples/outdoor_trad_climbing.jpg) | |

In short and general terms, bouldering is a type of climbing where climbers do not use any gear or hardware as they climb - they only use climbing pads on the ground to protect their falls, and most of the time climbers go as high as 10 to 20 feet above the ground. Sport and trad climbing is when climbers use rope and specialized hardware to protect them from falling to the ground as they climb as high as 100+ feet above the ground.

It is a challenging task to identify which style of climbing it is, because pictures can be taken from different angles, climbing environment and background varies (color of the rock, color of the indoor walls and holds, etc.), climber's outfits are very different as well as the movements they are performing. Telling the difference between the outdoor and indoor climbing could be generally considered an easy task, because climbing gyms have a more colourful walls and holds, and these colors are not really seen outdoors. However, differentiating between styles withing these two subgroups is still tricky. For the indoor climbing that relates to the fact that in the same shot we could see climber bouldering in the foreground, but in the background there could be ropes (or technically a person could be doing bouldering in the same area where people do roped climbing). For the outdoor climbing making a difference between bouldering and sport/trad climbing could be an easier task, but for the trad and sport climbing it could be tricky, because the gear used for climbing looks similar, and the differences lie in the nuances of how climbers climb the rock and how and where they place the gear to stay safe (rock face versus cracks/constrictions). 

So let's see what results we going to get at the end of this project using FastAI library.

### Technical details and project components
The main file of this project is [Jupyter](https://jupyter.org/) notebook called ```fastai-climbing-style-classifier.ipynb``` and is located in the root folder of the repository. The contents of that document are going to be presented below in the next section for the sake of not making the reader to go look into separate file :smiley:. The notebook was converted from Jupyter format into markdown format using the [nbconvert](https://nbconvert.readthedocs.io/en/latest/#), so in readme the contents are rather static. The ML model got trained using [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb) to run the Jupyter notebook code and [Google Drive](https://www.google.com/intl/en_zm/drive/) to store image datasets.
The training, validation and testing image datasets are located in the ```input``` folder of the project. The ML trained models and ready for production packages are located in the ```output``` folder.

### Jupyter notebook contents

---

# **Setup**

First we connect our Google Drive to write and read data to/from. When the command is executed, we are asked to provide the key for the Google Drive user account that we are going to be using.


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    

The below commands ensure that any edits to libraries we make are reloaded here automatically, and also that any charts or images displayed are shown in this notebook.


```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```

Let's import FastAI libraries.


```python
from fastai.vision import *
from fastai.metrics import error_rate
```

Let's setup the input and output folder variables and see that we have our drive and data available.


```python
inputPath = Path('./drive/MyDrive/ML Projects/FastAIClimbingStyleClassification/input') # The folder where you place your datasets for training. "Path" comes from the FastAI library
inputPath.ls()
```




    [PosixPath('drive/MyDrive/ML Projects/FastAIClimbingStyleClassification/input/test'),
     PosixPath('drive/MyDrive/ML Projects/FastAIClimbingStyleClassification/input/train'),
     PosixPath('drive/MyDrive/ML Projects/FastAIClimbingStyleClassification/input/valid')]




```python
outputPath = Path('./drive/MyDrive/ML Projects/FastAIClimbingStyleClassification/output') # The folder where outputs will go
outputPath.ls()
```




    [PosixPath('drive/MyDrive/ML Projects/FastAIClimbingStyleClassification/output/models'),
     PosixPath('drive/MyDrive/ML Projects/FastAIClimbingStyleClassification/output/resnet34_fastai_climbing_classifier_20_percent_error.pkl')]



Let's take a look at a few images that we have in our datasets.


```python
ImageList.from_folder(inputPath.joinpath('test'))[0].show(figsize=(9,9))
```


    
![png](Readme_files/Readme_12_0.png)
    



```python
ImageList.from_folder(inputPath.joinpath('train/outdoor_bouldering'))[0].show(figsize=(9,9))
```


    
![png](Readme_files/Readme_13_0.png)
    



```python
ImageList.from_folder(inputPath.joinpath('valid/outdoor_sport_climbing'))[2].show(figsize=(9,9))
```


    
![png](Readme_files/Readme_14_0.png)
    


Setup the environment to not show warnings - this has to do with the fact that FastAI is not properly updated (as of November 23, 2020) to work with the latest Pytorch library, so that warnings are being produced when we try to load or display image datasets.


```python
import warnings
warnings.filterwarnings('ignore')
```

Set up the batch size variable for model training process.


```python
batchSize = 16 # Make this smaller, if you don't have enough processing power
imgSize = 192 # Also possible to make this smaller, if you don't have enough processing power 
tfms = get_transforms()
```

Get the image data from the folders for training.


```python
data = ImageDataBunch.from_folder(path=inputPath, ds_tfms=tfms, size=imgSize, bs=batchSize).normalize(imagenet_stats)
```

Let's see what we've got for our labels.


```python
print(data.classes)
```

    ['indoor_bouldering', 'indoor_sport_climbing', 'outdoor_bouldering', 'outdoor_sport_climbing', 'outdoor_trad_climbing']
    

Training set has around 70 images for the following labels: 'indoor_bouldering', 'indoor_sport_climbing', 'outdoor_bouldering', 'outdoor_sport_climbing'. It has around 40 images for the 'outdoor_trad_climbing' label. For the validation sets we have 25 images for the 'indoor_bouldering', 'indoor_sport_climbing', 'outdoor_bouldering', 'outdoor_sport_climbing' labels and 10 images for the 'outdoor_trad_climbing' label. This discrepancy is bacause it is harder to find good quality pictures of that climbing style on the Internet :smiley:

Let's examine what we've got in our training set.


```python
data.show_batch(ds_type=DatasetType.Train, rows=4, figsize=(10,10))
```


    
![png](Readme_files/Readme_25_0.png)
    


Let's examine what we've got in our validation set.


```python
data.show_batch(ds_type=DatasetType.Valid, rows=4, figsize=(10,10))
```


    
![png](Readme_files/Readme_27_0.png)
    


# **Training**

Setup the learner based on the RESNET34 architecture and the data that we 
provide. NOTE: this command will download already generally "pre-trained" model, so that it is going to be faster and easier for us to train it further for our specific need to classify climbing styles.


```python
learner = cnn_learner(data, models.resnet34, metrics=error_rate)
```

    Downloading: "https://download.pytorch.org/models/resnet34-333f7ec4.pth" to /root/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth
    


    HBox(children=(FloatProgress(value=0.0, max=87306240.0), HTML(value='')))


    
    

Setup the learner output path to the one that we going to be using for all output that it produces.


```python
learner.path = outputPath
```

Now let's train our model a little bit and see what we've got.


```python
learner.fit_one_cycle(4)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.997981</td>
      <td>0.757340</td>
      <td>0.275229</td>
      <td>01:53</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.371841</td>
      <td>0.742990</td>
      <td>0.256881</td>
      <td>01:14</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.073501</td>
      <td>0.618392</td>
      <td>0.229358</td>
      <td>01:14</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.839416</td>
      <td>0.621123</td>
      <td>0.201835</td>
      <td>01:14</td>
    </tr>
  </tbody>
</table>



```python
learner.recorder.plot_losses()
```


    
![png](Readme_files/Readme_35_0.png)
    


In the above we've trained our model for 4 cycles and eventually we were able to get to about 80% accuracy (error rate around 20%), which is quite good considering that we haven't trained the model as much and also the amount of training data we've got. We can see that both training loss and validation loss are trending down, and training loss is slightly higher than validation loss, which is a good indication that our model is getting better at recognizing and generalizing without overfitting. Since both metrics are trending down, we should be able to train the model a little bit more and try getting a better accuracy. To do that we can try unfreezing the model, analyze learning rate parameter and then train the whole neural net using specific learning rate interval. Let's see what happens. P.S. Before we proceed let's save the model that we have for now.


```python
learner.save('climbing-classifier-stage-1-80-percent-accuracy') # Save the model
```


```python
learner.lr_find() # Analyze learning rate
learner.recorder.plot()
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.520463</td>
      <td>#na#</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.537959</td>
      <td>#na#</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.513739</td>
      <td>#na#</td>
      <td>01:02</td>
    </tr>
  </tbody>
</table><p>



    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
    


    
![png](Readme_files/Readme_38_2.png)
    


We can see that our learning rate and loss relation is trending down at first and then trending up after. Let's try to train our model two more cycles with the specific learning rate range of about 0.8e-5 to 0.2e-4 where loss is the lowest and see what we get.


```python
learner.unfreeze()
learner.fit_one_cycle(2, max_lr=slice(0.8e-5,0.2e-4))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.496120</td>
      <td>0.624267</td>
      <td>0.201835</td>
      <td>01:39</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.522279</td>
      <td>0.621444</td>
      <td>0.201835</td>
      <td>01:39</td>
    </tr>
  </tbody>
</table>


We can see now that the training loss is quite low and started to go up, while validation loss is already slightly higher than the training loss. It seems that at this point 80% accuracy is as good as we can get. Let's save this model as our final result.


```python
learner.save('climbing-classifier-stage-2-80-percent-accuracy-unfreezed') # Save the model
```

Also let's export this model so that it is ready to be used in production. The size of the exported "pkl" file is going to be much smaller than the saved model file.


```python
learner.export(file='resnet34_fastai_climbing_classifier_20_percent_error.pkl')
```

# **Results**

Let's analyze in more detail how our trained model works.


```python
interpretation = ClassificationInterpretation.from_learner(learner) # First, we get the classification interpretation from our model
```

In the image below we see which images the model confused the most while training. We're going to also examine the heatmaps for the predictions.


```python
interpretation.plot_top_losses(9, figsize=(20,11), heatmap=True)
```


    
![png](Readme_files/Readme_49_0.png)
    


In the above picture we can make an interesting observation actually. Based on the heatmaps displayed it looks like the model has learned to make the predictions based on the environment properties of the image and not really the subject (humans). This probably is related to the fact that human representation in the dataset is very diverse - the clothes, the positions and moves these people are doing are quite different in each picture. This actually creates an interesting question whether the model would change if we had a bigger training dataset, or would more data accentuate the importance and weight of the environment even more in making the predictions. 

So based on the previous paragraph and the output above that we see it is not a surprise to see that, for example, outdoor trad climbing got confused with the indoor sport climbing, or outdoor trad climbing got confused with the outdoor bouldering. None of the characteristics that would define certain style of climbing (like certain gear or body position, etc.) contributed the most to the decision being made.

It is actually a very interesting insight on what is happening here, and starting this project I definitely did not expect to see the results like this!

Let's see what we've got in the confusion matrix and try to analyze what we see having in mind the above observations about environment being the leading factor in classification.


```python
interpretation.plot_confusion_matrix(figsize=(12,12), dpi=60)
```


    
![png](Readme_files/Readme_51_0.png)
    


As we can see in the above matrix, the model is mostly having difficulties differentiating between specific styles within two big subgroups of indoor and outdoor activities, which is expected since we know that the environments of indoor and outdoor differ greatly, having a lot more diverse colours present in the indoor setting. Knowing that the environment is the thing that adds most of the weight to the final decision, it is still quite interesting to see that the model does quite well in telling apart properly different styles of climbing within outdoor and indoor categories.

As a last step, let's try to actually pretend we are in production and test our model on the images it hasn't seen before, and see what we get.


```python
testPath = Path('./drive/MyDrive/ML Projects/FastAIClimbingStyleClassification/input/test') # Set the path where we have test data
testData = ImageList.from_folder(testPath) # Read the images into the list 
productionLearner = load_learner(path=outputPath, file='resnet34_fastai_climbing_classifier_20_percent_error.pkl', test=testData) # Load the learner from our production ready exported model file and provide test data to it
```


```python
predictions, predictedClasses = productionLearner.get_preds(ds_type=DatasetType.Test) # This will result in providing prediction tensors for every class for each image
```


```python
predictions
```




    tensor([[6.4604e-02, 9.3534e-01, 1.8537e-07, 3.6070e-06, 5.4984e-05],
            [9.6253e-01, 3.4200e-02, 3.2302e-03, 1.2106e-07, 3.7778e-05],
            [4.1486e-02, 9.4321e-01, 3.6586e-04, 1.4604e-02, 3.3272e-04],
            [2.1905e-02, 9.7808e-01, 6.1749e-07, 4.3184e-06, 1.1730e-05],
            [8.8292e-01, 1.1470e-01, 1.9434e-06, 3.4998e-04, 2.0308e-03],
            [2.8822e-02, 9.9716e-04, 8.7811e-01, 1.1056e-02, 8.1016e-02],
            [3.1518e-03, 1.7791e-03, 8.5996e-01, 1.6507e-02, 1.1860e-01],
            [5.5600e-06, 2.2172e-03, 3.7476e-05, 9.8707e-01, 1.0674e-02],
            [2.7239e-06, 2.8130e-07, 2.6186e-02, 7.9518e-01, 1.7863e-01],
            [2.0163e-03, 1.2355e-03, 5.0662e-05, 9.7814e-01, 1.8561e-02],
            [2.7569e-04, 9.0536e-04, 2.7857e-03, 3.3769e-01, 6.5834e-01],
            [3.6148e-07, 2.4097e-03, 1.4596e-01, 7.9354e-01, 5.8094e-02]])



From the code above we've got predictions for every class for all images (12 rows for 12 images in the test set, 5 predictions for each of the five classes for each image). However, because the test set is unlabelled, we still need to manually infer the predicted class for each image ourselves. Otherwise the **classes** tensor has all 0s.


```python
predictedClasses = torch.argmax(predictions, dim=1) # Get resulting prediction class for each image
predictedClasses
```




    tensor([1, 0, 1, 1, 0, 2, 2, 3, 3, 3, 4, 3])



Those numbers would correspond to the original classes that we had.


```python
data.classes # Original classes
```




    ['indoor_bouldering',
     'indoor_sport_climbing',
     'outdoor_bouldering',
     'outdoor_sport_climbing',
     'outdoor_trad_climbing']



In a similar way we could also be doing predictions for each single image by using the **productionLearner.predict** function.


```python
correctPredictions = 0

# Wasn't sure if there is a better way to get both file names and the actual image data, so opted out to go with index iteration
for idx in range(testData.items.size):
  image = testData[idx]
  _, pred_idx, outputs = productionLearner.predict(item=image)
  actualClassName = '_'.join(str(testData.items[idx]).split('/')[-1].split('_')[:-1]) # Get the actual class from the file name
  predictedClassName = data.classes[pred_idx]
  image.show(figsize=(5,5), title='Actual: ' + actualClassName + ' / Predicted: ' + predictedClassName + '\nPredicted class probabilities: ' + str(outputs))

  if (actualClassName == predictedClassName):
    correctPredictions = correctPredictions + 1
```


![png](Readme_files/Pred_1.png)
![png](Readme_files/Pred_2.png)
![png](Readme_files/Pred_3.png)
![png](Readme_files/Pred_4.png)
![png](Readme_files/Pred_5.png)
![png](Readme_files/Pred_6.png)
![png](Readme_files/Pred_7.png)
![png](Readme_files/Pred_8.png)
![png](Readme_files/Pred_9.png)
![png](Readme_files/Pred_10.png)
![png](Readme_files/Pred_11.png)
![png](Readme_files/Pred_12.png)


```python
correctPredictions / testData.items.size * 100
```




    66.66666666666666



For the test set of 12 images we've got only 66.66% of images classified correctly in this case.


# Conclusion

We've used realively small amount of data to train image classification ML model to recognize five different types of climbing styles. In the process of training and analysis of the model we were able to make an interesting observation that the model actually learned to differentiate between different classes based on the environment of each picture and not the climbers themselves. This actually makes sense, because the environments is what stays relatively the same between different images within one class, while climbers are very different in every image. To address that, one proposed approach I can think of is to actually eliminate the background from the equation either by switching it to black and white color or make everything except the climbers of completely the same color. On top of that having more data in training and validation datasets could potentially help as well.
All things considered (model training time, amount of training and test data, training accuracy and test accuracy), I think the final result of 66% accuracy is not bad at all, and I am quite happy with what I was able to learn from the process of working on this assignment.

---

### Lessons learned
Working on this project allowed me to explore some capabilities of the [FastAI](https://course19.fast.ai/) library. I learned how to prepare training datasets and use them to train my own ML model, which can be used to differentiate between multiple climbing styles. As part of this project I got to write a little bit of code in [Python](https://www.python.org/), learn about Jupyter notebooks, nbconvert and different platforms that allow to train ML models in the cloud by providing access to GPU resources. The cloud based platforms for training the ML models I've explored include:
1) [Paperspace Gradient](https://gradient.paperspace.com/)
2) [Kaggle](https://www.kaggle.com/)
3) [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb)
4) [Floyd Hub](https://www.floydhub.com/)

I ended up using Google Colaboratory, because it was the most up-to-date platform among the others, it allowed me to easily upload and manage my datasets using Google Drive and it was free :laughing:!

### Ways to improve the ML model
As mentionaed in the __Conclusion__ section of the Jupyter notebook, a few ways to improve the model is to add more data to training and validation sets, as well as adjust them in a way that the CNN is learning to recognize climbers (plus their environment) and make predictions based on that instead of making predictions based on just the environment the climbers are in.