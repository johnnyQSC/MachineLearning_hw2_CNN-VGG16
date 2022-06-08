import cv2
from imutils import paths
import os 

#Train resize
image_train = list(paths.list_images('C:/Users/User/Desktop/New_Rice_Dataset/train'))

for i in range(len(image_train)):
    image = cv2.imread(image_train[i])
    path,name = os.path.split(image_train[i])
    name = name.split(image_train[i])
    image = cv2.resize(image,(504,378))
    cv2.imwrite(os.path.join(path,'{}.jpg'.format(name)),image)
    #cv2.imwrite('C:/Users/User/Desktop/New_Rice_Dataset/train',image)
print("ok")

#Test resize
image_Test = list(paths.list_images('C:/Users/User/Desktop/New_Rice_Dataset/test'))
for i in range(len(image_Test)):
    image = cv2.imread(image_Test[i])
    path,name = os.path.split(image_Test[i])
    name = name.split(image_Test[i])
    image = cv2.resize(image,(504,378))
    cv2.imwrite(os.path.join(path,'{}.jpg'.format(name)),image)
    #cv2.imwrite('C:/Users/User/Desktop/New_Rice_Dataset/train',image)
print("ok")