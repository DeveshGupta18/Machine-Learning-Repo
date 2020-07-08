import cv2
import numpy as np

img=cv2.imread(r'C:\Users\DELL\Desktop\Walls\157925-full_download-wallpaper-1920x1080-beach-night-sea-sky-full-hd-hdtv.jpg')
#h=img/255
#print(h)
print(img)

cv2.imshow('image', img)
cv2.waitKey(0)



img=cv2.imwrite(r"C:\Users\DELL\Desktop\Walls\dj1.png",img)
img2=cv2.imread(r"C:\Users\DELL\Desktop\Walls\dj1.png")
cv2.imshow("image",img2)
cv2.waitKey(0)


############################################################
img2=cv2.imread(r'C:\Users\DELL\Desktop\Walls\dj1.png')

n=np.ones(img2.shape, dtype="unit8")*150
new_var=cv2.add(img2, n)
cv2.imshow(new_var)
cv2.waitKey(0)



############################################
#
##############################################

import cv2

import numpy as np
img=cv2.imread(r'D:/aa.jpg')
print(img)

cv2.imshow('image',img)
cv2.waitKey(0)



img=cv2.imwrite(r"D:/aa1.png",img)
img2=cv2.imread(r"D:/aa1.png")
cv2.imshow("image",img2)
cv2.waitKey(0)


img2=cv2.imread(r"D:/aa1.png")
print(img2.shape)



##########################################
img2=cv2.imread(r"D:/aa1.png")
m=np.ones(img2.shape,dtype="uint8")*150
new_var=cv2.add(img2,m)
cv2.imshow("image",new_var)
cv2.waitKey(0)


print()