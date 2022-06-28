#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[4]:


# View a training image
img_index = 0 # can be updated this value to look at other images
img = train_images[img_index]
print("Image Label: " ,train_labels[img_index])
plt.imshow(img)


# In[5]:


# Print the shape 
print(train_images.shape) # 60,000 rows of 28 x 28 pixel images
print(test_images.shape)  # 10,000 rows of 28 x 28 pixel images


# In[6]:


Categories = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[7]:


# Visualizing a random training image
plt.figure()
plt.imshow(train_images[7])
plt.colorbar()
plt.grid(True)


# In[8]:


train_images = train_images/255.0
test_images = test_images/255.0


# In[9]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.xlabel(Categories[train_labels[i]])


# In[10]:


# Create the neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # reduce dimensionality of images
    keras.layers.Dense(128, activation=tf.nn.relu), # 128 neurons # hidden layer
    keras.layers.Dense(10, activation=tf.nn.softmax) # 10 unique labels # output layer
])


# In[16]:


#compile the model
model.compile(
    optimizer = tf.optimizers.Adam(),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)


# In[24]:


#train the model
model.fit(train_images, train_labels, epochs=5,batch_size=32)


# In[18]:


#Evaluate the model
model.evaluate(test_images,test_labels)


# In[20]:


#make a prediction
predictions[0]


# In[21]:


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(Categories[predicted_label],
                                100*np.max(predictions_array),
                                Categories[true_label]),
                                color=color)


# In[25]:



for i in range(0,30):
   plt.imshow(test_images[i])
   plt.show()
   print(test_labels[i])


# In[ ]:




