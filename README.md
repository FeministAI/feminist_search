# Feminist Search

Before you start, please note that this was written with Python 3.7.5.  If you have the libraries installed in requirements.txt, this code should work just fine, even if they are different versions of the libraries.  If not, run the following in the working directory: 

```pip install -r requirements.txt```

## Images

For this project, you really should have 100 images to start. You can start with less if you want. Assuming two classes, which we shall call A and B, create two directories with all of your images labeled with A in the A directory, and all of the images labeled B in the B directory. 

## Model and prediction

Next, import the model_hack.py into your script, interpreter, or notebook, something like the following: 

```import model_hack as mh```

Next, create an instance of the ClusteredImages class: 

```images = mh.ClusteredImages(PATH_TO_A_IMAGES, PATH_TO_B_IMAGES, IMAGE_SUFFIX)```

To return your labels for training the model, run: 

```y = images.return_labels()```

To return your data for training the model, you will need to instatiate a clustering model (I use KMeans):

```from sklearn.cluster import KMeans```

```clusterModel = KMeans(n_clusters=5)```

```X = images.generate_features(clusterModel)```

I've included a class for finding the best parameters for an SVM model, but if you just want a quick start: 

```from sklearn.svm import SVC```

```clf = SVC(gamma='auto')```

```clf.fit(X,y)```

For generating image histograms for prediction, you can use the KMeans model and a new instance of the ClusteredImages using your new data.

# Saved Model

I created a model with some original data we had, and saved it as fs_model.pkl. If you want to use this model, do the following: 

```import pickle as pkl```

```model = pkl.load(open('fs_model.pkl', 'rb'))```

The trained model is an svm, so use the code you would use for any trained model in scikit-learn.

