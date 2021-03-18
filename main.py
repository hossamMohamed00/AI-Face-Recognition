# Start Importing the required packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Visualization
import matplotlib.pyplot as plt

#Machine Learning
from sklearn.model_selection import train_test_split # To split data to train and test
from sklearn.cluster import KMeans

#System
import time

#----------------------------------------
# Load the data-set

data=np.load("Data-Set/olivetti_faces.npy")
target=np.load("Data-Set/olivetti_faces_target.npy")
print('Loading Done!')

#----------------------------------------
# Display some information about the data-set
print("There are {} images in the dataset".format(len(data))) # 400
print("There are {} unique targets in the dataset".format(len(np.unique(target)))) # 40
print("Size of each image is {}x{}".format(data.shape[1],data.shape[2])) # 64*64
print("Pixel values were scaled to [0,1] interval. e.g:{}".format(data[0][0,:4])) # e.g:[0.30991736 0.3677686  0.41735536 0.44214877]

print("unique target number:",np.unique(target)) # [0..39]

#----------------------------------------
# Show 40 distinct people
def show_40_distinct_people(images, unique_ids):
    # Creating 4X10 subplots in  18x9 figure size
    fig, axarr=plt.subplots(nrows=4, ncols=10, figsize=(18, 9))
    # For easy iteration flattened 4X10 subplots matrix to 40 array
    axarr=axarr.flatten()
    
    #iterating over user ids
    for unique_id in unique_ids:
        image_index=unique_id*10
        axarr[unique_id].imshow(images[image_index], cmap='gray')
        axarr[unique_id].set_xticks([])
        axarr[unique_id].set_yticks([])
        axarr[unique_id].set_title("face id:{}".format(unique_id))
    plt.suptitle("There are 40 distinct people in the dataset")
    
show_40_distinct_people(data, np.unique(target))

#----------------------------------------
# show 10 faces for n subject
def show_10_faces_of_n_subject(images, subject_ids):
    cols=10# each subject has 10 distinct face images
    rows=(len(subject_ids)*10)/cols #
    rows=int(rows)
    
    fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(18,9))
    #axarr=axarr.flatten()z
    
    for i, subject_id in enumerate(subject_ids):
        for j in range(cols):
            image_index = subject_id*10 + j
            axarr[i,j].imshow(images[image_index], cmap="gray")
            axarr[i,j].set_xticks([])
            axarr[i,j].set_yticks([])
            axarr[i,j].set_title("face id:{}".format(subject_id))
            
show_10_faces_of_n_subject(images=data, subject_ids=[0, 15, 30, 39, 10])

#----------------------------------------
# Machine learning models can work on vectors.
# Since the image data is in the matrix form, it must be converted to a vector.
# We reshape images for machine learnig  model to be vector
X = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
# Why indecies 0,1,3 --> because it is a 3 dim
#  -1 makes all in one dim (flatten)
print("X shape:",X.shape)

#----------------------------------------
# Split data and target into Random train and test Subsets

# X_train, X_test ==>  features 
# y_train, y_test ==>  labels 
# stratify ==> If not None, data is split in a stratified fashion, using this as the class labels.
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.2, stratify=target, random_state=0)


# Expected 320 train , 80 test
print("X_train shape:",X_train.shape)
print("y_train shape:{}".format(y_train.shape))

#----------------------------------------
# There is 8 (10 images * 0.8 train) Images on the train set for each class [0..39]
y_frame = pd.DataFrame()
y_frame['subject ids'] = y_train
y_frame.groupby(['subject ids']).size().plot.bar(figsize=(15,9),title="Number of Samples for Each Classes")

#----------------------------------------
# Usage of K-Mean clustring

cluster_num = 40 # The number of the distint people on the project

Kmean = KMeans(n_clusters = cluster_num)
# KMeans Configuration 
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)

'''
# KMeans Configuration 
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=400,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)

'''

# Train the model 
start = time.time()
Kmean.fit(X_train)
end = time.time()

training_time = end - start
print("Training time = " + str(training_time) + " sec.")

#----------------------------------------

# Show the final results

# Reshape the data to be in shape (280, 64, 64)
X_train_reshaped = X_train.reshape(len(X_train),64,64) 
                          
# Get the labels         
labels = Kmean.labels_ # 280 labels 
# len(Kmean.labels_) # 280 label for 40 label


for i in range(40):
    index = np.nonzero(labels == i)[0]
    
    num = len(index)
    if num == 0:
        continue
    
    this_faces = X_train_reshaped[index]
    
    fig, axes = plt.subplots(1, num, figsize=(4 * num, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    fig.suptitle("Cluster " + str(i), fontsize=25)
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(this_faces[i], cmap='gray')



#----------------------------------------
# Test the Model

t1_start = time.time() 
prediction = Kmean.predict(X_test)
t1_stop = time.time() 

print("Time elapsed: ", t1_stop - t1_start)



# Show the ouput

# Reshape the data to be in shape (280, 64, 64)
X_test_reshaped = X_test.reshape(len(X_test),64,64)

for i in range(len(X_test)):

    face = X_test_reshaped[i]
    fig, axes = plt.subplots(1, 1, figsize=(4, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    
    fig.suptitle("Cluster " + str(prediction[i]), fontsize=25)
    axes.imshow(face, cmap='gray')

#---------------------------------------