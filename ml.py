#Importing necessary libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation as inter
import tensorflow as tf
import os
import glob
import imutils


def correct_skew(image, delta=1, limit=5):
    def determine_score(arr, angle):
        data = inter.rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        return histogram, score

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 

    scores = []
    angles = np.arange(-limit, limit + delta, delta)
    for angle in angles:
        histogram, score = determine_score(thresh, angle)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]

    (h, w) = image.shape[:2]  
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
              borderMode=cv2.BORDER_REPLICATE)

    return best_angle, rotated
     
image1=cv2.imread("C:\\JS_Projects\\Final_Project_Camera\\a.jpg")

angle, rotated = correct_skew(image1)
print(angle)
cv2.imwrite('rotated.jpg', rotated)
gray = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Remove horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(rotated, [c], -1, (255,255,255), 5)


vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
cnts = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(rotated, [c], -1, (255,255,255), 5)

gray = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)

#applying median filter for Salt and pepper/impulse noise
filter1 = cv2.medianBlur(gray,5)

#applying gaussian blur to smoothen out the image edges
filter2 = cv2.GaussianBlur(filter1,(7,7),0)

#applying non-localized means for final Denoising of the image
dst = cv2.fastNlMeansDenoising(filter2,None,11,3,11)

#converting the image to binarized form using adaptive thresholding
th1 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,3)  

cv2.imwrite('ImagePreProcessingFinal2.jpg', th1)


imdir = "C:\\JS_Projects\\Final_Project_Camera\\model\\input"
outdir = "C:\\JS_Projects\\Final_Project_Camera\\model\\output"

#prepare(file) allows us to use an image of any size, since it automatically resize it to the image size we defined in the first program.
def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


#hconcat_resize_min() takes the image list as an argument and resizes the image to the maximum height in the list
#and concatenates them horizontally
def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_max = max(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_max / im.shape[0]), h_max), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

model = tf.keras.models.load_model("C:\\JS_Projects\\Final_Project_Camera\\model\\CNN.model") # Loding pre-trained data trained 

filelist = glob.glob(os.path.join(imdir, '*.jpg')) #reading all the files in image directory

j=0
for i in range(len(filelist)):
    image = prepare(filelist[i]) #Single image that you want to predict
    prediction = model.predict([image]) #predicting if the image is part of multicharacter or not
    prediction = list(prediction[0])
    if(prediction[0]>0.5) and i+1<len(filelist): #if the image belongs to MultipartClass and is not the last image append it to next image
        im1 = cv2.imread(filelist[i])
        print(filelist[i])  #printing file path of all the images classified as MultipartCharacter
        im2 = cv2.imread(filelist[i+1])
        im_h_resize = hconcat_resize_min([im1, im2])
        cv2.imwrite(outdir+"\\"+str(j)+".jpg", im_h_resize)
        j=j+1

im = cv2.imread('C:\\JS_Projects\\Final_Project_Camera\\model\\output\\0.jpg')
#cv2_imshow('Output Sample',im)
im = cv2.imread("C:\\JS_Projects\\Final_Project_Camera\\ImagePreProcessingFinal2.jpg")

cv2.waitKey(0)
RGB_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  #opencv reads image in BGR format to display in matplot lib we convert it to RGB
plt.imshow(RGB_im)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 0)


ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
print(ret)
print(thresh1)

dilate = cv2.dilate(thresh1, None, iterations=2)  

cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[1] if imutils.is_cv3() else cnts[0]

sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * im.shape[1] )

orig = im.copy()
i = 0
ROI_number = 0
for cnt in sorted_ctrs:
    # Check the area of contour, if it is very small ignore it
    if(cv2.contourArea(cnt) < 250):
        continue

    # Filtered countours are detected
    x,y,w,h = cv2.boundingRect(cnt)
    
    # Taking ROI of the cotour
    roi = im[y:y+h, x:x+w]
    cv2.imwrite('C:\\JS_Projects\\Final_Project_Camera\\contours\\roi_{}.png'.format(ROI_number), roi)
 
    ROI_number += 1
    # Mark them on the image if you want
    cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)
    # Save your contours or characters
    cv2.imwrite("Im/roi" + str(i) + ".png", roi)

    i = i + 1 

cv2.imwrite("box.jpg",orig)

cv2.imshow('',orig)
cv2.waitKey(0)

