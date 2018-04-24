import cv2
import numpy as np


# Loop over all images
for n_image in range(3) : 

    # Read image (suppose it is in the same directory than the python program)
    img = cv2.imread('image_' + str(n_image+1) + '.png') 
    rows, cols, channels = img.shape


    # Equations from Meyer et al. publication #
    
    # Normalize each channel (The OpenCV format is BGR)
    B_star = img[:,:,0]/255.0
    G_star = img[:,:,1]/255.0
    R_star = img[:,:,2]/255.0
    
    Sum_normalized_channels = B_star + G_star + R_star

    # Compute the chromatic coordinates
    r = R_star/Sum_normalized_channels
    g = G_star/Sum_normalized_channels
    b = B_star/Sum_normalized_channels


    # Compute the vegetation indices
    ExG = 2*g - r - b # Excess green
    ExR = 1.4*r - g # Excess red
    ExGR= ExG-ExR 


    # Positive threshold to the image
    _, ExGR = cv2.threshold(ExGR,0,255,cv2.THRESH_BINARY) # above 0 => white (the plant)
    ExGR = np.reshape(ExGR.astype(np.uint8), (rows, cols,1))

    # Erode and dilate image to keep only relevant information
    ExGR = cv2.erode(ExGR, np.ones((3,3), np.uint8))    # Kernel size is 3
    ExGR = cv2.dilate(ExGR, np.ones((19,19), np.uint8)) # Kernel size is 19
    #cv2.imshow("Diff",ExGR) # To empirically set the kernel sizes


    # Use the OpenCV function to look for the contours
    _, contours, _ = cv2.findContours(ExGR, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_EXTERNAL allows to remove internal contours

    # For each contours draw the corresponding bounding box
    for c in contours :  
        rect = cv2.boundingRect(c)  # Get coordinate of rectangle
        x,y,w,h = rect
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255),2) # Draw red rectangle 


    # Show the image with bounding boxes
    cv2.imshow("Image with bounding boxes",img) 
    cv2.imwrite('image_' + str(n_image+1) + '_with_boxes.png',img) # Save the image
    cv2.waitKey(-1)




