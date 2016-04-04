import mahotas
import numpy as np
import matplotlib.pyplot as plt
from helper import *
from generateTrainValTestData import normalizeImage, watershed_adjusted_membranes
import glob
import clahe
import os

def normalize_all_input(img_search_string = '/media/vkaynig/NewVolume/IAE_ISBI2012/images/images/*.tif'):
    img_files = sorted( glob.glob( img_search_string ) )
    for fileName in img_files:
        img = mahotas.imread(fileName)
        clahe.clahe(img, img, 2.0)
 #       img = normalizeImage(img, saturation_level=0.05)
        mahotas.imsave(fileName, np.uint8(img))


def create_fullContour_labels():
    #img_search_string = '/media/vkaynig/NewVolume/IAE_ISBI2012/fullContours/I*_train.tif'
    img_search_string = '/media/vkaynig/NewVolume/Cmor_paper_data/labels/*.tif'
    img_files = sorted( glob.glob( img_search_string ) )
    
    for img_index in xrange(np.shape(img_files)[0]):
        print 'reading image ' + img_files[img_index] + '.'
        label = mahotas.imread(img_files[img_index])
        
        #membranes = np.logical_and(label[:,:,0]==0, label[:,:,1]==255)
        membranes = label[:,:]==0

        shrink_radius=5
        y,x = np.ogrid[-shrink_radius:shrink_radius+1, -shrink_radius:shrink_radius+1]
        disc = x*x + y*y <= (shrink_radius ** 2)
        non_membrane = 1-mahotas.dilate(membranes, disc)

        #fileName = "/media/vkaynig/NewVolume/IAE_ISBI2012/fullContours/"
        fileName = "/media/vkaynig/NewVolume/Cmor_paper_data/labels/"
        fileName_label = fileName + (img_files[img_index])[52:58] + "_fullContour.tif"
        fileName_background = fileName + (img_files[img_index])[52:58] + "_background.tif"

        print 'writing image' + fileName_label
        mahotas.imsave(fileName_label, np.uint8(membranes*255))
        mahotas.imsave(fileName_background, np.uint8(non_membrane*255))

def create_membrane_and_background_images():
    for purpose in ['train','validate','test']:
        #img_search_string = '/media/vkaynig/NewVolume/IAE_ISBI2012/ground_truth/' + purpose + '/*.tif'
        img_search_string = '/media/vkaynig/NewVolume/Cmor_paper_data/labels/' + purpose + '/*.tif'
#        img_gray_search_string = '/media/vkaynig/NewVolume/IAE_ISBI2012/images/' + purpose + '/*.tif'
#        img_gray_search_string = '/media/vkaynig/NewVolume/Cmor_paper_data/images/' + purpose + '/*.tif'

        img_files = sorted( glob.glob( img_search_string ) )
#        img_gray_files = sorted( glob.glob( img_gray_search_string ) )
        
        for img_index in xrange(np.shape(img_files)[0]):
            print 'reading image ' + img_files[img_index] + '.'
            label_img = mahotas.imread(img_files[img_index])
            
#            gray_img =  mahotas.imread(img_gray_files[img_index])
            boundaries = label_img==0
#            boundaries = label_img == -1
            boundaries[0:-1,:] = np.logical_or(boundaries[0:-1,:], np.diff(label_img, axis=0)!=0)
            boundaries[:,0:-1] = np.logical_or(boundaries[:,0:-1], np.diff(label_img, axis=1)!=0)
            boundaries = 1-boundaries
            
            shrink_radius=1
            y,x = np.ogrid[-shrink_radius:shrink_radius+1, -shrink_radius:shrink_radius+1]
            disc = x*x + y*y <= (shrink_radius ** 2)
            background = mahotas.erode(boundaries, disc) + 1
            
            membranes = 1-background
            
            img_file_name = os.path.basename(img_files[img_index])
            #outputPath = '/media/vkaynig/NewVolume/IAE_ISBI2012/labels/'
            outputPath = '/media/vkaynig/NewVolume/Cmor_paper_data/labels/'
            
            print 'writing image' + img_file_name         
            mahotas.imsave(outputPath + 'background/' + purpose + '/' + img_file_name, np.uint8(background*255))
            mahotas.imsave(outputPath + 'membranes/' + purpose + '/' + img_file_name, np.uint8(membranes*255))



if __name__ == "__main__":
    for purpose in ['train','validate','test']:
        #img_search_string = '/media/vkaynig/NewVolume/IAE_ISBI2012/labels/membranes/' + purpose + '/*.tif'
        img_search_string = '/media/vkaynig/NewVolume/Cmor_paper_data/labels/membranes/' + purpose + '/*.tif'
        img_gray_search_string = '/media/vkaynig/NewVolume/Cmor_paper_data/images/' + purpose + '/*.tif'
        
        img_files = sorted( glob.glob( img_search_string ) )
        img_gray_files = sorted( glob.glob( img_gray_search_string ) )
        
        for img_index in xrange(np.shape(img_files)[0]):
            adjustedLabelImg = watershed_adjusted_membranes(img_gray_files[img_index], img_files[img_index])        
            
            img_file_name = os.path.basename(img_files[img_index])
            #outputPath = '/media/vkaynig/NewVolume/IAE_ISBI2012/labels/adjustedMembranes/'
            outputPath = '/media/vkaynig/NewVolume/Cmor_paper_data/labels/adjustedMembranes/'            
            print 'writing image' + img_file_name         
            mahotas.imsave(outputPath + purpose + '/' + img_file_name, np.uint8(adjustedLabelImg))
            
            imgGray = mahotas.imread(img_gray_files[img_index])
            

