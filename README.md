# Audio Classification
This program takes .wav files containing different sounds from this [Kaggle dataset] and classifies them into 10 separate buckets by using the Random Forest and KNN classifier, a Machine Learning method. This program is implemented in Python.


## Motivation
The goal of this project was to gain a better understanding of the transformations and pre-processing required to effectively classify audio data. Additionally, through this project we aimed to understand the theory behind the two different Machine Learning classifiers, the KNN classifier and the Random Forest Classifier. 

## Project Overview
The data consisted of short audio clips that could be part of one of the following 10 buckets:
1) Bass Drum
2) Saxophone
3) Double Bass
4) Violin
5) Gunshot
6) Burp
7) Cello
8) Fart
9) Trumpet
10) Hi-hat

Some of these audio clips were labelled, which we used as training data for our models, and we then used our trained models to classify this data.

## Theory

### K-Nearest Neighbors
The basic algorithm involves, for each testing data point, finding the K nearest data points (usually via 2-norms) and then choosing the mode of their classes. There is no training involved, which potentially leads to high computation costs, as every computation needs to be rerun per point. Additionally, choosing different values of K can impact the resulting label. As shown in Figure 1 below, the green testing point will be classified different with K = 3 (resulting in a red label) and K = 5 (resulting in a blue label). More information on the algorithm can be found [here].
<img width="265" alt="Screen Shot 2020-07-21 at 6 58 55 PM" src="https://user-images.githubusercontent.com/62456147/88047469-41e6fc80-cb84-11ea-9a44-540807fdb6a4.png">


### Random Forest
The Random Forest Classifier “creates a set of decision trees from a randomly selected subset of a training set, and aggregates the votes from different decision trees in order to decide the final class of that object.

In essence, a decision tree classifier repetitively divides the working area into subparts by identifying lines in the data. The decision tree terminates when either of two criteria have been met: when it has divided all the test data into classes that are pure, or if some criteria of classifier attributes have been met.

This [Towards Data Science article] explains this concept in further detail.

### Random Forest Classifer

## Code Structure
### Preprocessing and Feature Extraction
#### Low-pass Filtering
The audio files were first [low-pass filtered] in order to remove any noisy datapoints. This vastly improved the chances of the model correctly classifying the data. 

#### Conversion to Frequency Domain
Next, the [Mel spectrogram] of the audio signal was taken using the Python `librosa` library. The Mel spectrogram is a way of representing the features of the audio signal on the frequency spectrum instead of the time spectrum, which is more suitable to differentiate various signals from one another. The Mel spectrogram in particular is highly effective because it results in features being extracted in a way that mimics the way in which human brains extract features in sounds, making extracting te Mel spectrogram ideal for identifying audio signals of sounds that humans generate or hear routinely.

#### Zero-padding
After the Mel spectrogram was taken of each audio sample, the signals to be fed into the model were now of different length. All signals were as such zero-padded to be the same length.

### Machine Learning Model Implementation
Both models used were implemented using the `sklearn` library. The library made it very easy to split the data into training and testing data for the model. 
<img width="697" alt="Screen Shot 2020-07-21 at 6 44 24 PM" src="https://user-images.githubusercontent.com/62456147/88046209-49a5a180-cb82-11ea-8601-a7d3b45bd004.png">

The `sklearn` library also made it easy to implement the models and generate predictions. Below is a code snippet of the Random Forest model being implemented. The KNN model can be implemented similarly. 

### Results
The preliminary results were examined using a confusion matrix. Below is an example. After fine-tuning of the model's parameters, its accuracy was increased.
<img width="435" alt="Screen Shot 2020-07-21 at 6 53 33 PM" src="https://user-images.githubusercontent.com/62456147/88047018-82924600-cb83-11ea-8e48-84623b99c1da.png">



## Credits
This program was written in conjunction by me, Nikhita Gangla, and Clara Selbrede. 


[Kaggle dataset]: https://www.kaggle.com/c/rice-elec-301-f19/overview
[low-pass filtered]:https://en.wikipedia.org/wiki/Low-pass_filter#:~:text=A%20low%2Dpass%20filter%20(LPF,depends%20on%20the%20filter%20design.
[Mel spectrogram]: https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0
[here]: https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
[Towards Data Science article]: https://medium.com/machine-learning-101/chapter-5-random-forest-classifier-56dc7425c3e1
