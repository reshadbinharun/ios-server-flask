import os, csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import logsumexp
import scipy as sp
import sklearn.preprocessing
import sklearn.pipeline
import cv2
import scipy.misc
import PIL
import glob
import matplotlib.image as mpimg


def loadImages(path):
    # Put files into lists and return them as one list of size 4
    image_files = sorted([os.path.join(path, file)
         for file in os.listdir(path) if file.endswith('.JPG')])
 
    return image_files

def create_csv(folder_list, y_val, training_path, rotate_angle):
    image_list = []
    label_list = []
    y_list = y_val * np.ones(len(folder_list))
    for folder, y in zip(folder_list, y_list):
        dataset = loadImages(training_path + folder)
        num_images = len(dataset)
        print(num_images)
        for i in range(2, num_images):
            image_list.append(processing(dataset[i], i, rotate_angle).flatten())
            label_list.append(y)
    return image_list, label_list

# Display one image
def display_one(a, title1 = "Original"):
    RGB_img = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    plt.imshow(RGB_img), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()
# Display two images
def display(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()
    
def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# Preprocessing
def processing(data, count, rotate_angle):
    # loading image
    # Getting 3 images to work with 
    #img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in data[:2]]
    img = cv2.imread(data, -1)
    #print('Original size',img.shape)
    # --------------------------------
    # setting dim of the resize
    #print('why blue')
    img = rotateImage(img, rotate_angle)
    #display_one(img)
    
    (orig_height, orig_width, _) = img.shape
    new_height = int(orig_height / 3)
    new_width = int(orig_width / 3)
    dim = (new_width, new_height)
    
    #print('cropped image')
    crop_img = img[int(new_height/1.1):orig_height-int(new_height/1.1), int(new_width/0.9):orig_width-int(new_width/0.9)].copy()
    #display_one(crop_img)
    #res_img = []
    #for i in range(len(img)):
    new_height = int(new_height / 3)
    new_width = int(new_width / 3)
    dim = (90, 70)
    
    res = cv2.resize(crop_img, dim, interpolation=cv2.INTER_LINEAR)
        #res_img.append(res)

    # Checcking the size
    #try:
        #print('RESIZED', res.shape)#res_img.shape)
    #except AttributeError:
        #print("shape not found")
        
    # Visualizing one of the images in the array
    original = res #res_img
    #display_one(original)
    #cv2.imshow("original", original)

    # Remove noise
    # Using Gaussian Blur
    #no_noise = []
    #for i in range(len(res_img)):
    blur = cv2.GaussianBlur(res, (5, 5), 0)
        #no_noise.append(blur)


    image = blur #no_noise
    (row, col, _) = image.shape
    image = image[:, (col//4):(int(col//1.25))]
    image = image[(row//4):(int(row//1.25)), :]
    #print(image.shape)
    #cv2.imwrite("training_data/cropped_image.JPG", image)
    #display(original, image, 'Original', 'Blurred')
    #print('image ', count)
    return image

picture_path = 'training_data/pH/sensitivity_images/'
def save_as_csv(rotation_list=[], pic_pathway=[], y_list=[]):
    for r_angle in angles_list:
    rotate_angle = r_angle
    print(r_angle)
    pic_list = sorted(os.listdir(pic_pathway))[1:]
    for y_val, f in zip(y_list, pic_list):
        training_path = pic_pathway + f + '/'
        save_path = pic_pathway + "_csv_files"
        ph_list = []
        ph_list = sorted(os.listdir(training_path))[1:]
        print(ph_list)
        x, y = create_csv(ph_list, y_val, training_path, rotate_angle)
        with open('%sy_train_%s_lowres_%d.csv'%(save_path, f, rotate_angle), 'a') as fp:
            wr = csv.writer(fp, dialect='excel')
            for yval in y:
                wr.writerow([yval])
        
        with open('%sx_train_%s_lowres_%d.csv' %(save_path, f, rotate_angle), 'a') as fp:
            wr = csv.writer(fp, dialect='excel')
            for image in x:
                wr.writerow(image)










