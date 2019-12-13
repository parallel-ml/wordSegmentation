import os
import cv2
from WordSegmentation import seg
from east import crop
from cnnPredict import predict
import numpy as np
import time
from keras.models import load_model
start = time.time()
model = load_model('keras_mnist.h5')

def find_anomalies(data):
	anomalies = []
	data_std = np.std(data)
	data_mean = np.mean(data)
	anomaly_cut_off = data_std * 3
	lower_limit  = data_mean - anomaly_cut_off 
	upper_limit = data_mean + anomaly_cut_off
	for outlier in data:
		if outlier > upper_limit:
			anomalies.append(outlier)
		if outlier < lower_limit:
			anomalies.append(outlier)
	return anomalies

def zoom():
	img1 = cv2.imread("./img/web.jpg", cv2.IMREAD_GRAYSCALE) # Whole Picture
	img2 = cv2.imread("./img/2338c.jpeg", cv2.IMREAD_GRAYSCALE) # Object
	# ORB Detector
	orb = cv2.ORB_create()
	kp1, des1 = orb.detectAndCompute(img1, None)
	kp2, des2 = orb.detectAndCompute(img2, None)
	# Brute Force Matching
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des1, des2)
	matches = sorted(matches, key = lambda x:x.distance)

	# ----------------------------------------------------------
	# Initialize lists
	list_kp1 = []
	list_kp2 = []

	print ('length of matches: ')
	print (len(matches))

	# if len(matches) > 50:
	# 	matches = matches[:50]

	# For each match...
	for mat in matches[:50]:

	    # Get the matching keypoints for each of the images
	    img1_idx = mat.queryIdx
	    img2_idx = mat.trainIdx

	    # x - columns
	    # y - rows
	    # Get the coordinates
	    (x1,y1) = kp1[img1_idx].pt
	    (x2,y2) = kp2[img2_idx].pt

	    # Append to each list
	    list_kp1.append((x1, y1))
	    list_kp2.append((x2, y2))

	# ----------------------------------------------------------
	# print (list_kp1)

	x_list = []
	y_list = []

	for x,y in list_kp1:
		x_list.append(x)

	result = find_anomalies(x_list)

	print ("length to COMPARE")
	print (len(result))
	print (len(x_list))

	for i in reversed(result):
		index = x_list.index(i)
		print (index)
		list_kp1.pop(index)

	for x,y in list_kp1:
		y_list.append(y)

	result = find_anomalies(y_list)

	for i in reversed(result):
		index = y_list.index(i)
		list_kp1.pop(index)

	x_list = []
	y_list = []
	for x,y in list_kp1:
		x_list.append(x)
		y_list.append(y)

	x = int(np.mean(x_list))
	y = int(np.mean(y_list))
	h = 200

	matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)
	# img1_1 = cv2.circle(img1,(x, y), 400, 5, thickness=5)
	print (x)
	print (y)
	img_crop = img1[y-h:y+h, x-h:x+h]
	# cv2.imshow("Img1", img1)
	# cv2.imshow("Img2", img2)
	# cv2.imshow("Matching result", matching_result)

	# cv2.imwrite('img333.jpg', img1)
	cv2.imwrite('imgzoom.jpg', img_crop)
	cv2.imwrite("matched.jpg", matching_result)


def main():
        img = cv2.imread("imgzoom.jpg")
        # img = cv2.imread("file.jpeg")
        # img = cv2.imread("./img/file (5).jpeg")
        print("shape",img.shape)
        print("shape1",img.shape)
        img = crop(img)

        print("shape2",img.shape)
        

##        cv2.imshow("nan.jpg",img)
        ret,img = cv2.threshold(img,110,255,cv2.THRESH_BINARY)
        cv2.imwrite("croppy.jpg",img)
        # cv2.imshow("seg",img)
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
                #cv2.imshow("s"+str(i),n[i])
        
	

end = time.time()
print("Tensorflow time:" , (end - start))

if __name__ == '__main__':
	zoom()
	main()


