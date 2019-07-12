import os
import cv2
from WordSegmentation import wordSegmentation, prepareImg

import numpy as np

def main():
	"""reads images from data/ and outputs the word-segmentation to out/"""

	# read input images from 'in' directory
	imgFiles = os.listdir('../data/')
	for (i,f) in enumerate(imgFiles):
		print('Segmenting words of sample %s'%f)
		
		# read image, prepare it by resizing it to fixed height and converting it to grayscale
		img = prepareImg(cv2.imread('../data/%s'%f), 50)
##		img=cvtColor(img, gray, CV_BGR2GRAY);
		# execute segmentation with given parameters
		# -kernelSize: size of filter kernel (odd integer)
		# -sigma: standard deviation of Gaussian function used for filter kernel
		# -theta: approximated width/height ratio of words, filter function is distorted by this factor
		# - minArea: ignore word candidates smaller than specified area
		res = wordSegmentation(img, kernelSize=3, sigma=5, theta=2, minArea=140)
		
		# write output to 'out/inputFileName' directory
		if not os.path.exists('../out/%s'%f):
			os.mkdir('../out/%s'%f)
		
		# iterate over all segmented words
		print('Segmented into %d words'%len(res))
		for (j, w) in enumerate(res):
			(wordBox, wordImg) = w
			(x, y, w, h) = wordBox

			wordImg=np.pad(wordImg, ((4,4),(4,4)),"constant",constant_values=(255))
##                        ret,wordImg = cv2.threshold(wordImg,100,255,cv2.THRESH_BINARY)
			ret,wordImg = cv2.threshold(wordImg,90,255,cv2.THRESH_BINARY)
##                        x=np.mean(wordImg)
			print(np.mean(wordImg))
			cv2.imwrite('../out/%s/%d.png'%(f, j), wordImg) # save word
                        
			cv2.rectangle(img,(x,y),(x+w,y+h),0,1) # draw bounding box in summary image
		
		# output summary image with bounding boxes around words
		cv2.imwrite('../out/%s/summary.png'%f, img)


if __name__ == '__main__':
	main()
