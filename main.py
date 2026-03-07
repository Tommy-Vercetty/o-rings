#KNOWLEDGE - Why do we convert the image from Grayscale to Binary?
    # Because we are ONLY concerned with whether the O-Rings are normal or defective,
    #  making this is a BINARY problem.
    #   We do loose information by thresholding but we maintain ONLY information we need
    #    for this task

    #KNOWLEDGE - OTSU's THRESHOLDING APPROACH
    # FORMULA: w0 ​* w1​(μ0 ​− μ1​)²
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

    #KNOWLEDGE - Binary Morphology
    #Binary Morphology is an image processing technique that usually acts on the foreground.
    # That is why we will flip the Foreground and Background classes (I.e. Our rings will be white and our background will be black) 
    # --------------------------
    # Types of Binary Morphology
    # --------------------------
    # EROSION: Used to remove noise/ small blobs
    # Step 1) Looks at its 3x3 neighbourhood
    # Step 2) If ALL pixels in that window are WHITE, 'current pixel' = WHITE
    # Step 3) If even one pixel is BLACK, 'current pixel = BLACK'
    
    # DILATION: Used to fill in small gaps or holes
    # Step 1) Looks at its 3x3 neighbourhood
    # Step 2) If ANY pixels in that window are WHITE, 'current pixel' = WHITE
    # Step 3) If even one pixel is BLACK, 'current pixel = BLACK'
      
    # 3) OPENING = EROSION + DILATION - Used to remove ISOLATED WHITE noise/ pixels 
    # 4) CLOSING = DILATION + EROSION - Used to fill small holes inside 

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

#STEP 1    
def imageHistogram(img):
    histogram = np.zeros(256)
    for x in range (img.shape[0]):
        for y in range (img.shape[1]):
            histogram[img[x, y]] += 1
    return histogram

#STEP 2
def threshold(img, thresholdValue):
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            if img[x,y] > thresholdValue:
                img[x,y] = 255
            else:
                img[x,y] = 0
    return img

#STEP 2
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

        variance = w0 * w1 * (μ0 - μ1) ** 2

        if variance > maximumVariance:
            maximumVariance = variance
            optimalThreshold = x
    return optimalThreshold

#STEP 3
def erosion(img, k = 1):
    #Since 'img.shape' stores our Height and Width as a tuple, we declare them together on the same line
    imageHeight, imageWidth = img.shape
    erodedImage = np.zeros_like(img)

    for x in range (k, imageHeight - k):
        for y in range (k, imageWidth - k):
            window = img[x - k: x + k + 1, y - k: y + k + 1]
            if np.all(window == 255):
                erodedImage[x, y] = 255
            else:
                erodedImage[x, y] = 0
    return erodedImage

def dilation(img, k = 1):
    imageHeight, imageWidth = img.shape
    dilatedImage = np.zeros_like(img)

    for x in range (k, imageHeight - k):
        for y in range (k, imageWidth - k):
            window = img[x - k: x + k + 1, y - k: y + k + 1]
            if np.any(window == 255):
                dilatedImage[x, y] = 255
            else:
                dilatedImage[x, y] = 0
    return dilatedImage

#STEP 4
def ccl8Neighbours(img):
    imageHeight, imageWidth = img.shape
    imageLabels = np.zeros_like(img, dtype = int)
    currentLabel = 1

    #Defining our 8-Connected Neighbour Offsets
    neighbours = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    
    for x in range(imageHeight):
        for y in range(imageWidth):
            #We start off by checking if we are looking at a pixel 
            # that is foreground that has not been given a label yet
            if (img[x, y] == 255 and imageLabels[x, y] == 0):
                #If it hasn't we will assign it a new component object
                objectStack = [(x, y)]
                imageLabels[x, y] = currentLabel

                while objectStack:
                    cx, cy = objectStack.pop()
                    for dx, dy in neighbours:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < imageHeight and 0 <= ny < imageWidth:
                            if img[nx, ny] == 255 and imageLabels[nx, ny] == 0:
                                imageLabels[nx, ny] = currentLabel
                                objectStack.append((nx, ny))
                currentLabel += 1

    return imageLabels, currentLabel - 1

def areaOfRing(cleanedImage, oRingNumber):
    acceptanceTruncationPoint = 0.97

    area = np.sum(cleanedImage == 255)

    oRingPercentage = area / oRingMean

    print(f"O-Ring {oRingNumber} - Area: ", area)
    print(f"O-Ring {oRingNumber} - % Expected Area: {oRingPercentage * 100:.2f}%")

    if oRingPercentage < acceptanceTruncationPoint:
        print(f"O-Ring {oRingNumber}: FAULTY\n")
        return area, oRingPercentage, "FAULTY"
    else:
        print(f"O-Ring {oRingNumber}: PASSED\n")
        return area, oRingPercentage, "PASSED"

allORingAreas = []
cleanedImages = []

for i in range(1, 16):
    #We use the '.imread()' function in the OpenCV library to read in all our images into memory
    # NOTE: Since our images are grayscale and we want to load them that way, set add an extra argument at the end  '0'.
    img = cv.imread('c:/users/tommy/Downloads/Orings/Oring' + str(i) + '.jpg', 0)
    #Make a copy of the images, so that we can display it to compare against Thresholded version
    originalImage = img.copy()
    
    #STEP 1: Image Histogram [0 = BLACK, 255 = WHITE]
    #First off, we want to determine how many pixels in each O-Ring image is part of every intensity level.
    # We use an image histogram to do this, and we will check every intensitry level from 0 to 255. 
    #  NOTE: Since we are looking at grayscale images, when we refer to intensitry, we refer to the brigthness of the image.
    #  - If the intensity of the pixel is high, that means our pixel is brighter, and closer to 255.
    #  - If the intensity of the pixel is low, that means our pixel is darker, and closer to 0. 

    #STEP 2: Detecting the Optimal Threshold (OTSU's Method)
    #Now that we have our histogram, we want to convert our image from Grayscale (I.e. 0 - 255) to Binary (I.e. 0 OR 255) 
    # We want the optimal threshold point to be set automatically as the images slightly differ from one another. 
    
    #STEP 3: Binary Morphology
    # Now that we have all of our images in Binary thanks to our Thresholding, we need to clean the images up:
    #  1) Small amounts of noise are present in some images and are scattered throughout the image
    #  2) White noise within the O-Rings needs should not be present

    #STEP 4: Connected Component Labelling (CCL)
    #Our image now has the foreground parts as WHITE/ 255 and the background parts as BLACK/ 0
    # Let's take our O-Ring image as our example. We have the main part of the foreground which is our O-Ring.
    #  We also have other parts such as broken off parts of rubber from the O-Rings (Referred to as noise), that are NOT
    #   connected or pixel-linked to the O-Ring
    #    CCL Technique basically traverses the image and assign labels to these regions/ parts, to allow us to REMOVE them  
    #     NOTE: Once labeled, we can then also calculate properties such as:
    #     - Area, Perimeter, Bounding Box, Circularity etc.
    #      The CORE idea behind CCL is that we STRUCTURE the image into seperate OBJECTS
    #   
    # 4 OR 8-CONNECTED CL
    #To identify components (I.e. Rubber parts, the actual O-Ring etc.) we use either 4 or 8 Connection
    #-----------------------------------------------------------------------------------------
    # EXAMPLE SECTION from an image: (Pixels are going to be BACKGROUND (0) or FOREGROUND (1))
    # 0 1 0 0 1 0
    # 1 1 0 1 1 0
    # 0 0 1 0 0 1
    # 1 0 1 1 0 0
    # 0 1 0 0 1 1
    # 0 0 1 0 1 0
    #-----------------------------------------------------------------------------------------
    # - 4-CONNECTION: We look at our N, S, W, E and we find there are 8 COMPONENTS
    # - 8-CONNECTION: We look at our N, NW, NE, S, SW, SE, W, E and we find there is 1 HUGE COMPONENT
    #NOTE: Pixels that touch diagonally or horizontally become part of the SAME component!
    #NOTE 2: We will use 8-CONNECTION as our O-Rings are circular, THEREFORE, we are going to be using diagonals a lot. 
    pixelIntensityHistogram = imageHistogram(img)
    plt.figure(figsize = (8, 4))
    plt.scatter(range(256), pixelIntensityHistogram, color = 'grey', s = 10)
    plt.title(f"Grayscale Histogram for O-Ring {i}")
    plt.xlabel("Pixel Intensity {0 = BLACK, 255 = WHITE}")
    plt.ylabel("Number of Pixels")
    plt.show()

    thresholdValue = calculateThreshold(img)
    print("Optimal Threshold Chosen: ", thresholdValue)
    thresholdedImage = threshold(img, thresholdValue)
    #Switching the Foreground (O-Rings) to WHITE, and the Background to BLACK
    # We do this as Binary Morphology techniques operate on WHITE FOREGROUND objects
    thresholdedImage = 255 - thresholdedImage

    #ERODED Image
    erodedImage = erosion(thresholdedImage, k = 1)

    #DILATED Image
    dilatedImage = dilation(erodedImage, k = 2)

    #CLEANED Image
    imageLabels, numberOfLabels = ccl8Neighbours(dilatedImage)
    print("Number of Components Detected: ", numberOfLabels)

    #Here we keep only the larget component, being the O-Ring
    imageAreas = [np.sum(imageLabels == x)
                  for x in range(1, numberOfLabels + 1)
                  ]
    largestLabel = np.argmax(imageAreas) + 1 

    cleanedImage = np.zeros_like(imageLabels, dtype = np.uint8)
    cleanedImage[imageLabels == largestLabel] = 255

    #CALCULATING AREA
    area = np.sum(cleanedImage == 255)
    allORingAreas.append(area)
    cleanedImages.append(cleanedImage)
    print(f"O-Ring {i} Area: ", area)

    plt.figure(figsize = (16, 5))

    #Displaying the original image
    plt.subplot(1, 6, 1)
    plt.title(f"Original O-Ring {i}")
    plt.imshow(originalImage, cmap = 'gray')
    plt.axis("off")

    #Displaying the thresholded image
    plt.subplot(1, 6, 2)
    plt.title(f"Thresholded O-Ring {i}")
    plt.imshow(img, cmap = 'gray')
    plt.axis("off")
    
    #Displaying the binary inverted image
    plt.subplot(1, 6, 3)
    plt.title(f"Thresholded O-Ring {i}")
    plt.imshow(thresholdedImage, cmap = 'gray')
    plt.axis("off")

    #Displaying the image after its been eroded
    plt.subplot(1, 6, 4)
    plt.title(f"Eroded O-Ring {i}")
    plt.imshow(erodedImage, cmap = 'gray')
    plt.axis("off")

    #Displaying the eroded image after its been dilated
    plt.subplot(1, 6, 5)
    plt.title(f"Dilated O-Ring {i}")
    plt.imshow(dilatedImage, cmap = 'gray')
    plt.axis("off")

    #Displaying the Cleaned image after its been CCL'd
    plt.subplot(1, 6, 6)
    plt.title(f"Cleaned O-Ring {i}")
    plt.imshow(cleanedImage, cmap = 'gray')
    plt.axis("off")
    #Display the figure that contains all images
    plt.show()

    #FIXED!!! - Due to error in threshold calculation
    #Fixing/ Inverting O-Rings whose pixels have all been determined to be 255 AFTER Thresholding
    #numberOfBlackPixels = np.sum(thresholdedImage == 0)
    #print("Number of Black Pixels in Image: ", numberOfBlackPixels)
    #numberOfWhitePixels = np.sum(thresholdedImage == 255)
    #print("Number of White Pixels in Image: ", numberOfWhitePixels)
    #totalNumberOfPixels = thresholdedImage.size
    #blackPixelRatio = numberOfBlackPixels / totalNumberOfPixels
    #if blackPixelRatio < 0.02:
    #    print("O-Ring likely misclassified.")
    #    thresholdedImage = 255 - thresholdedImage

    rgb = cv.cvtColor(thresholdedImage, cv.COLOR_GRAY2RGB)
    #Annotating the image. We are adding the word Hello in colour blue on the image
    #cv.putText(rgb, "Image: " + str(i), (25, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #Annotating the image with a circle.
    #cv.circle(rgb, (40, 40), 20, (0, 0, 255))
    #We can show the image using the OpenCV open function
    cv.waitKey(0)
    cv.destroyAllWindows()

oRingMean = np.mean(allORingAreas)
print("\n Mean O-Ring Area: , ", oRingMean)
print("------")

for i in range(len(cleanedImages)):
    areaOfRing(cleanedImages[i], i + 1)