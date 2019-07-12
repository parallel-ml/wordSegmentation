import os
import cv2
from WordSegmentation import seg
from east import crop
from cnnPredict import predict
import numpy as np

import time
start = time.time()

from keras.models import load_model
model = load_model('cnn.h5')
def main():
        img = cv2.imread("/home/akanksha/Desktop/img/file (3).jpeg")
        print("shape",img.shape)
        print("shape1",img.shape)
        img = crop(img)

        print("shape2",img.shape)
        

##        cv2.imshow("nan.jpg",img)
        ret,img = cv2.threshold(img,110,255,cv2.THRESH_BINARY)
        cv2.imwrite("croppy.jpg",img)
        cv2.imshow("seg",img)
##        print(img)

        n=seg(img)

        for i in range(len(n)):
                print(n[i].shape)
##                ret,img = cv2.threshold(n[i],50,170,cv2.THRESH_BINARY)
##                cv2.imshow("as",img)
####                print("shape image",n[i].shape)
                print(np.mean(n[i]))
                if(n[i].shape[0]-n[i].shape[1]<5):
                        continue
                if(np.mean(n[i])>215):
                        continue
                predict(n[i],model)
                cv2.imwrite("a"+str(i)+".jpg",n[i])
                cv2.imshow("s"+str(i),n[i])
        
	

end = time.time()
print("Tensorflow time:" , (end - start))

if __name__ == '__main__':
	main()


