import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

def threshold(img, thresholdValue):
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if img[x,y] > thresholdValue:
                img[x,y] = 255
            else:
                img[x,y] = 0
    return img
        
def imageHistogram(img):
    histogram = np.zeros(256)
    for x in range (img.shape[0]):
        for y in range (img.shape[1]):
            histogram[img[x, y]] += 1
    return histogram

def calculateThreshold(img):
    histogram = imageHistogram(img)
    totalPixels = img.shape[0] * img.shape[1]
    histogram = histogram / totalPixels
    #This variable will store the highest difference in MEAN intensities of both classes of the most optimal Threshold value 
    # We set it to 0 as its default value
    maximumVariance = 0
    #This variable will store our automatically determined 'optimal threshold'
    # We set it to 0 as its default value
    optimalThreshold = 0

    #We loop through all the pixels
    for x in range(256):
        #The pixels that are in the background will be added to the 'w0' class
        w0 = np.sum(histogram[:x + 1])

        #The pixels that are in the foreground will be added to the 'w1' class
        w1 = np.sum(histogram[x + 1:])

        if w0 == 0 or w1 == 0:
            continue

        μ0 = np.sum(np.arange(0, x + 1) * histogram[:x + 1]) / w0
        μ1 = np.sum(np.arange(x + 1, 256) * histogram[x + 1:]) / w1

        variance = w0 * w1 + (μ0 - μ1) ** 2

        if variance > maximumVariance:
            maximumVariance = variance
            optimalThreshold = x
    return optimalThreshold

for i in range(1, 16):
    #We use the '.imread()' function in the OpenCV library to read in all our images into memory
    # NOTE: Since our images are grayscale and we want to load them that way, set add an extra argument at the end  '0'.
    img = cv.imread('c:/users/tommy/Downloads/Orings/Oring' + str(i) + '.jpg', 0)

    #STEP 1: Image Histogram [0 = BLACK, 255 = WHITE]
    #First off, we want to determine how many pixels in each O-Ring image is part of every intensity level.
    # We use an image histogram to do this, and we will check every intensitry level from 0 to 255. 
    #  NOTE: Since we are looking at grayscale images, when we refer to intensitry, we refer to the brigthness of the image.
    #  - If the intensity of the pixel is high, that means our pixel is brighter, and closer to 255.
    #  - If the intensity of the pixel is low, that means our pixel is darker, and closer to 0. 
    pixelIntensityHistogram = imageHistogram(img)
    plt.figure(figsize = (8, 4))
    plt.scatter(range(256), pixelIntensityHistogram, color = 'grey', s = 10)
    plt.title(f"Grayscale Histogram for O-Ring {i}")
    plt.xlabel("Pixel Intensity [0 = BLACK, 255 = WHITE]")
    plt.ylabel("Number of Pixels")
    plt.show()

    #KNOWLEDGE - Why do we convert the image from Grayscale to Binary?
    # Because we are ONLY concerned with whether the O-Rings are normal or defective,
    #  making this is a BINARY problem.
    #   We do loose information by thresholding but we maintain ONLY information we need
    #    for this task

    #KNOWLEDGE - OTSU's THRESHOLDING APPROACH
    # FORMULA: w0​ *w1​(μ0 ​− μ1​)²
    # w0 - % of Class 0
    #  w1 - % of Class 1
    #   μ0 - Mean of Class 0
    #    μ1​ - Mean of Class 1
    # 1) We check every possible threshold value (E.g. threshold = 200) from 0 to 255 and for every threshold we split
    #       the pixels into TWO classes. One class stores the pixels that have an intensity HIGHER than the threshold. The
    #       other class stores the pixels with intensity LOWER than the threshold value.
    # 2a) Next, we calculate the mean intensity of each class. This means we look at the intensity value of each pixel in the LOWER class,
    #        and calculate the mean of them. We do the same for the pixels in the HIGHER than threshold class.
    #        We repeat this process for every threshold value, meaning we have (255 x 2) MEAN values.
    # 2b) Afterwards, we also calculate the probability of each class, so the proportion of pixels that are HIGHER and LOWER than the threshold.
    # 3) Next, we comparse the mean intensity's we got from Part 2a. IF, the mean values of the HIGHER class is far apart from the LOWER class, then
    #       that means we have found a good value that SEPARATES our foreground from background pixels. If the mean intensity's are close, then that
    #       means that the threshold values does not SEPARATE our classes well.
    #       NOTE: Otsu's Method also considers the PROPORTION of pixels from Part 2b. This means a threshold value that separates the mean intensity's well
    #           but has a LARGE class imbalance is not chosen.
    #           NOTE: Otsu does NOT determine a good class balance if they are EQUALLY-SIZED (E.g. 50% of pixels BELOW, 50% of pixels ABOVE), but rather any split AS LONG AS
    #               there isn't any class that is almost empty or too full.
    # 4) Otsu's determines the most optimal threshold according to the classes being meaningfully sized and mean intensities are far apart

    #STEP 1b: Detecting the Optimal Threshold (OTSU's Method
    #Now that we have our histogram, we want to convert our image from Grayscale (I.e. 0 - 255) to Binary (I.e. 0 OR 255) 
    # We want the optimal threshold point to be set automatically as the images slightly differ from one another. 
    thresholdValue = calculateThreshold(img)
    print("Optimal Threshold Chosen: ", thresholdValue)
    bw = threshold(img, thresholdValue)

    rgb = cv.cvtColor(bw, cv.COLOR_GRAY2RGB)
    #Annotating the image. We are adding the word Hello in colour blue on the image
    cv.putText(rgb, "Image: " + str(i), (40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #Annotating the image with a circle.
    cv.circle(rgb, (40, 40), 20, (0, 0, 255))
    #We can show the image using the OpenCV open function
    cv.imshow('thresholded image', rgb)
    cv.waitKey(0)
    cv.destroyAllWindows()
