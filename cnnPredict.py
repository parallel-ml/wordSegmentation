#from scipy.misc import imread, imresize
import numpy as np
import cv2


def predict(image,model):
    x = image
    #compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
    #make it the right size
    x = cv2.resize(x , dsize=(28,28), interpolation=cv2.INTER_CUBIC)
    #x= numpy.array(x.fromarray(arr).resize(28,28)).
    #x = imresize(x,(28,28))
    #convert to a 4D tensor to feed into our model
    x = x.reshape(1,28,28,1)
    x = x.astype('float32')
    x /= 255

    #perform the prediction

    model = model
    out = model.predict(x)
    print(np.argmax(out))
