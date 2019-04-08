#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
from matplotlib import pyplot as plt
import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


# In[2]:


bike_images = []
for filename in glob.glob('Bikes/*.jpg'): 
    im=cv2.imread(filename,0)
    bike_images.append(im)
print ("No of Bikes: ",len(bike_images))
train_images = bike_images[:int(0.8*len(bike_images))]
test_images = bike_images[int(0.8*len(bike_images)):]
p = len(test_images)
# Loading the horse images
horse_images = []
for filename in glob.glob('Horses/*.jpg'): 
    im=cv2.imread(filename,0)
    horse_images.append(im)
print ("No of Horses: ",len(horse_images))
train_images = train_images+horse_images[:int(0.8*len(horse_images))]
test_images = test_images+horse_images[int(0.8*len(horse_images)):]


# In[3]:


sift = cv2.xfeatures2d.SIFT_create()


# In[8]:


# Finding interest points of train images and appending them to a numpy array
l_train = []
kp,des = sift.detectAndCompute(train_images[0],None)
l_train.append(len(des))
kpp=kp
for i in range(1,len(train_images)):
    kp1,des1 = sift.detectAndCompute(train_images[i],None)
    if(i==80):
        kppp=kp1
    l_train.append(len(des1))
    kp = np.hstack((kp,kp1))
    des = np.vstack((des,des1))


# In[9]:


img_with_keypoints = cv2.drawKeypoints(train_images[0], kpp, outImage=np.array([]), color=(0, 0, 255),
                                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# In[10]:


plt.imshow(img_with_keypoints)


# In[14]:


img_with_keypoints = cv2.drawKeypoints(train_images[80], kppp, outImage=np.array([]),
                                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# In[15]:


plt.imshow(img_with_keypoints)


# In[16]:


l_test = []
kp2,des2 = sift.detectAndCompute(test_images[0],None)
l_test.append(len(des2))
for i in range(1,len(test_images)):
    kp1,des1 = sift.detectAndCompute(test_images[i],None)
    l_test.append(len(des1))
    kp2 = np.hstack((kp2,kp1))
    des2 = np.vstack((des2,des1))


# In[17]:



print (len(kp),len(des))


# In[18]:


k=30
plt.plot(des, '-')
plt.show()
kmeans = KMeans(n_clusters=k, random_state=0).fit(des)
train_labels = kmeans.labels_
print (train_labels)
test_labels = kmeans.predict(des2)


# In[19]:


train_hist = np.zeros((len(train_images),k))
j=0
count = 0
for i in range(len(train_images)):
    count = count+l_train[i]
    while(j<count):
        train_hist[i,train_labels[j]] = train_hist[i,train_labels[j]]+1
        j=j+1


# In[20]:


test_hist = np.zeros((len(test_images),k))
j=0
count = 0
for i in range(len(test_images)):
    count = count+l_test[i]
    while(j<count):
        test_hist[i,test_labels[j]] = test_hist[i,test_labels[j]]+1
        j=j+1


# In[21]:


img0_hist = np.zeros((1,k))
j=0
for i in range(l_train[0]):
        img0_hist[0,train_labels[i]] = img0_hist[0,train_labels[i]]+1


# In[22]:


plt.hist(img0_hist.ravel(),100,[0,k])
plt.show()


# In[23]:


plt.hist(train_hist.ravel(),120,[0,k])
plt.show()


# In[24]:


plt.hist(test_hist.ravel(),120,[0,k])
plt.show()


# In[25]:


stdSlr = StandardScaler().fit(train_hist)
train_hist = stdSlr.transform(train_hist)

stdSlr = StandardScaler().fit(test_hist)
test_hist = stdSlr.transform(test_hist)


# In[26]:


y_train = []
for i in range(len(train_images)):
    if(i<0.8*len(bike_images)):
        y_train.append(0)
    else:
        y_train.append(1)


# In[27]:


y_test = []
for i in range(len(test_images)):
    if(i<p):
        y_test.append(0)
    else:
        y_test.append(1)


# In[28]:


clf = LinearSVC()
clf.fit(train_hist, np.array(y_train))
print (" .....Accuracy........=  ",clf.score(test_hist,y_test))


# In[29]:


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_hist, np.array(y_train))
print (" .....Accuracy........=  ",neigh.score(test_hist,y_test))


# In[30]:


log = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(train_hist, np.array(y_train))
print (" .....Accuracy........=  ",log.score(test_hist,y_test))


# In[ ]:




