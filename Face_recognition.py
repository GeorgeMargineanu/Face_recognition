#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 #used to load the images (in the future it will have other uses)
import numpy as np
import glob #used to read image names
import random #used to generate tran/test split
import ipyplot #it will be explained
from matplotlib import pyplot as plt #for all ur plotting needs
import os
import PIL


# In[2]:


#Image reading
#base_path = r"C:\Users\georgmar\OneDrive - AMDOCS\Backup Folders\Desktop\CV\ORL\s09" #this is janky af (merge)
base_path = r"C:\Users\georgmar\OneDrive - AMDOCS\Backup Folders\Desktop\ORL"
files = os.listdir(base_path)
file_list = [i for i in files if i.endswith('.bmp')]
print(file_list)


# In[3]:


files_to_read = []
base_path = base_path + '\\'

for i in range(len(file_list)):
    files_to_read.append(base_path +   str(file_list[i]))

img = cv2.imread(files_to_read[0])
    
file_list = []
file_list = files_to_read

random.seed(42)
#(0,9) = secventa din care se extrag 7 numere random
indexes = random.sample(range(0,27), 21)

train_files =  []
test_files = []

L = file_list

t =[]
for i in indexes:
    t.append(L[i])
    
train_files.append(t)
test_files.append([x for x in L if x not in t])

train_images = []
for L in train_files:
  s = []
  for l in L:
    s.append( cv2.imread(l) )
    #cv2.imshow('image',img)#DisabledFunctionError: cv2.imshow() is disabled in Colab, because it causes Jupyter sessions to crash; see https://github.com/jupyter/notebook/issues/3935.
  train_images.append(s)

test_images = []
for L in test_files:
  s = []
  for l in L:
    s.append( cv2.imread(l) )
  test_images.append(s)

ipyplot.plot_images(train_images[0], max_images=27, img_width=100)    
ipyplot.plot_images(test_images[0], max_images=12, img_width=100)
    


# In[4]:


y = [int(x.split(sep = '\\')[-1].split(sep = '.')[0]) for x in train_files[0]]
print(y)


# In[5]:


#din ce clasa face parte imaginea
for i in range(len(y)):
    if y[i] > 20:
        y[i] = 2
    elif y[i] <10:
        y[i] = 0
    else:
        y[i] = 1


# In[6]:


y


# In[7]:


z = [int(x.split(sep = '\\')[-1].split(sep = '.')[0]) for x in test_files[0]]
print(z)

for i in range(len(z)):
    if z[i] >= 20:
        z[i] = 2
    elif y[i]<10:
        z[i] = 0
    else:
        z[i] = 1


# In[8]:


z


# In[9]:


color = ('r','g','b')
img = train_images[0][0]
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()


# In[10]:


scale_percent = 100 # % of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize images
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
plt.axis("off")
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
np.shape(resized)


# In[11]:


train_images_resized = []
for S in train_images:
  s = []
  for im in S:
    s.append(cv2.resize(im, dim, interpolation = cv2.INTER_AREA))
  train_images_resized.append(s)

test_images_resized = []
for S in test_images:
  s = []
  for im in S:
    s.append(cv2.resize(im, dim, interpolation = cv2.INTER_AREA))
  test_images_resized.append(s)

print("Train shape:",np.shape(train_images_resized))
print("Test shape:",np.shape(test_images_resized))
plt.axis("off")
plt.imshow(cv2.cvtColor(train_images_resized[0][0], cv2.COLOR_BGR2RGB))


# In[12]:


train_images_resized_gray = []
for S in train_images:
  s = []
  for im in S:
    s.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
  train_images_resized_gray.append(s)

test_images_resized_gray = []
for S in test_images:
  s = []
  for im in S:
    s.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
  test_images_resized_gray.append(s)

plt.axis("off")
#plt.imshow(train_images_resized_gray[0][0],cmap='gray')
#r,c = np.shape()
plt.imshow(train_images_resized_gray[0][0],cmap='gray')


# In[13]:


training = [item for sublist in train_images_resized_gray for item in sublist]
ipyplot.plot_images(training, max_images=27, img_width=100)
print(np.shape(training))

test = [item for sublist in test_images_resized_gray for item in sublist]
print(np.shape(test))


# In[14]:


training = np.array(training)
img = training
face = []

#for i in range(len(img)):
#    face.append(np.argwhere(arr[i] == 0))

print(np.shape(training))


# In[15]:


test = np.array(test)
print(test.shape)


# In[16]:


num_elems_training, image_size = img.shape[0], img.shape[1] * img.shape[2] 

from sklearn.metrics.pairwise import rbf_kernel

result = np.reshape(training, (num_elems_training, image_size))
result.shape
print(test.shape)


num_elems_testing = test.shape[0]

elems_to_test_raw = np.reshape(test, (num_elems_testing, image_size))

elems_to_test = rbf_kernel(elems_to_test_raw, elems_to_test_raw, gamma=None)


# In[17]:


elems_to_test


# In[18]:


from sklearn.metrics.pairwise import rbf_kernel
ASD = rbf_kernel(result, result, gamma=None)


# In[19]:


ASD.shape


# In[20]:


ASD[0]


# In[21]:


from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier(n_neighbors=5, metric='precomputed')
clf = KNeighborsClassifier(n_neighbors=5, metric='minkowski')

clf.fit(ASD,y)


# In[22]:


clf.score(ASD,y) #rezultat pe lotul de antrenare


# In[23]:


#scor pe lotul de test cu RBF
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=3, gamma=0.1, C=1)
svm.fit(result, y)
#print(svm.predict(elems_to_test_raw))
svm.predict(elems_to_test_raw)
print(svm.score(elems_to_test_raw,z))


# In[24]:


# Scor pe lotul de test cu NN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(result,y)
print(neigh.predict(elems_to_test_raw))
print(neigh.score(elems_to_test_raw,z))

