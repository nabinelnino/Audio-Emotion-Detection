# Audio-Emotion-Detection
## This is the implementation of emotion detection from audio file, which classify the eight possible emotion from the given audio file. Audio emotion detection is a challenging task where language processing play a part to generate emotion from audio. 
 ## Use Case
 <li>It can be implemented in call center to detect the emotions of the caller</li>
<li>In virtual therapy session, the overall emotion of the patient can be detected even without facial expression.
</li>
## Dataset:
<li>(RAVDESS) https://zenodo.org/record/1188976#.X4sE0tDXKUl</li>
<li> (TESS) https://tspace.library.utoronto.ca/handle/1807/24487</li>

## Flow of the project
<li>Cleaning the data</li>
<li>Extracting features from audio using Librosa library </li>
<li>Merging various features into one </li>
<li> Building LSTM model for training </li>
<li>Predicting on test data </li>
<li> Evaluating the emotion scores as the metric </li>


## Project Pipeline:
#### Digital signal processing is the hot topic in the field of Machine learning recently. We can see a lot of research on emotion detection from video or from images and we can even get a lot of pretrained model for this. However, there is still lack of research and work in the field of Speech Emotion Recognization (SER). 
#### Since the project is a classification problem, Convolution Neural Network seems the obvious choice, and we also built Random forest model, Multilayer perceptron but they underperformed with very low accuracies which could not pass the test while predicting the right emotions.
#### Finally, build a recurrent neural network, namely, LSTM and model is then able to predict emotion form audio file with accuracy of more than 80%.

## File:
### Main.py and utils.py file contains all required code for API creation. The final model is called here and it returns JSON file consisting overall probability of each emotion present in audio file.
### UI.HTML is User interface where API is called and we can see the plot containing overall emotion from the recording.
### Final_Model.ipnp is the final model where all required code and implementation of the model is implemented.
