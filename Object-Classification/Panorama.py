#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


img = cv2.imread("/home/srujan/VR/Assignment3/Panorama/institute1.jpg")
# print (img_.shape)
img2 = cv2.imread("/home/srujan/VR/Assignment3/Panorama/institute1.jpg",0)
# print (img1.shape)
im = cv2.imread("/home/srujan/VR/Assignment3/Panorama/secondPic1.jpg")
im2 = cv2.imread("/home/srujan/VR/Assignment3/Panorama/secondPic1.jpg",0)


# In[3]:


# plt.imshow(img1)


# In[4]:


# plt.imshow(im1)


# In[5]:


img_ = cv2.imread("/home/srujan/VR/Assignment3/Panorama/institute2.jpg")
print (img.shape)
img1 = cv2.imread("/home/srujan/VR/Assignment3/Panorama/institute2.jpg",0)
print (img.shape)
im_ = cv2.imread("/home/srujan/VR/Assignment3/Panorama/secondPic2.jpg")
im1 = cv2.imread("/home/srujan/VR/Assignment3/Panorama/secondPic2.jpg",0)


# In[6]:


plt.imshow(img2)


# In[7]:


plt.imshow(im2)


# In[8]:


sift = cv2.xfeatures2d.SIFT_create()


# In[10]:


kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
kp3, des3 = sift.detectAndCompute(im1,None)
kp4, des4 = sift.detectAndCompute(im2,None)


# In[20]:


bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# img3 = cv2.drawMatches(img,kp1,img_,kp2,matches,None,flags=2)
# plt.imshow(img3)
# plt.show()
matches1 = bf.knnMatch(des3,des4,k=2)


# In[21]:


good = []
for m in matches:
    if m[0].distance < 0.5*m[1].distance:
        good.append(m)
matches = np.asarray(good)

good = []
for m in matches1:
    if m[0].distance < 0.5*m[1].distance:
        good.append(m)
matches1 = np.asarray(good)


# In[22]:


# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,flags=2)


# In[23]:


if len(matches[:,0]) >= 4:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print (H)
else:
    raise AssertionError("Can’t find enough keypoints.")
    
if len(matches1[:,0]) >= 4:
    src1 = np.float32([ kp3[m.queryIdx].pt for m in matches1[:,0] ]).reshape(-1,1,2)
    dst1 = np.float32([ kp4[m.trainIdx].pt for m in matches1[:,0] ]).reshape(-1,1,2)
    H1, masked1 = cv2.findHomography(src1, dst1, cv2.RANSAC, 5.0)
    print (H1)
else:
    raise AssertionError("Can’t find enough keypoints.")


# In[24]:


dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))
dst[0:img.shape[0], 0:img.shape[1]] = img

dst1 = cv2.warpPerspective(im_,H1,(im.shape[1] + im_.shape[1], im.shape[0]))
dst1[0:im.shape[0], 0:im.shape[1]] = im


# In[25]:


plt.imshow(dst)
print (dst.shape)


# In[26]:


plt.imshow(dst1)
print (dst1.shape)


# In[ ]:





# In[ ]:





# In[ ]:




