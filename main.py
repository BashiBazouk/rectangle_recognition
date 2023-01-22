# imports
import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt

# some hardcoding, change it later!

file = 'original_screenshot.jpg'
width = 490 # width of image to resize
height = 440 # height of image to resize, top left corner only from 0,0
y1 = 55
y2 = 400
x1 = 20
x2 = 490

def auto_threshold(file):
    """Auto threshold detection playground"""
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    img = cv2.imread(file)
    x1, y1, x2, y2 = roi_definition(file)
    roi = img[y1:y2, x1:x2]

    # interesting attempt, th2 or th3 looks like good choice
    img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # but let's try another one
    # global thresholding
    ret1, th1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu's thresholding after gaussian filtering
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    images = [roi, 0, th1,
              roi, 0, th2,
              blur, 0, th3]

    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]

    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()



def roi_definition(file):
    # https://pyimagesearch.com/2014/08/04/opencv-python-color-detection/
    # colour of interest: Red: 254, green: 254, blue: 190
    # RGB: [254, 254, 190]
    # BGR: [190, 254, 254]

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # ap.add_argument("-i", file, help="path to the image")
    args = vars(ap.parse_args())

    # load the image
    image = cv2.imread(file)

    # define lower an upper limit for colour
    # RGB color space (or rather, BGR, since OpenCV represents images as NumPy arrays in reverse order)
    bounduaries = ([191, 255, 255], [189, 253, 253])

    # loop thru bounduaries (do I need loop?) # edit, nope, I have only one colour
    # for (lower, upper) in bounduaries:
    colour_delta = 2
    upper = np.array([191, 255, 255], dtype='uint8')
    lower = np.array([185, 250, 250], dtype='uint8')

    # find the colour
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    # https://stackoverflow.com/questions/57282935/how-to-detect-area-of-pixels-with-the-same-color-using-opencv
    # Remove noise
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # blur ? blur is better here
    blur = cv2.blur(mask, (5, 5))

    # Find contours
    cnts = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    c = max(cnts, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    # print('x: ', x)
    # print('y: ', y)
    # print('w: ', w, 'x2: ', x + w)
    # print('h: ', h, 'y2: ', y + h)

    return x, y, x + w, y + h

    #
    # area = 0
    # contours_counter = 0
    # for c in cnts:
    #     contours_counter += 1
    #     area += cv2.contourArea(c)
    #     cv2.drawContours(image, [c], 0, (0, 255, 0), 2)
    #     leftmost = tuple(c[c[:, :, 0].argmin()][0])
    #     print('leftmost',leftmost)
    #     rightmost = tuple(c[c[:, :, 0].argmax()][0])
    #     print('rightmost',rightmost)
    #     topmost = tuple(c[c[:, :, 1].argmin()][0])
    #     print('topmost',topmost)
    #     bottommost = tuple(c[c[:, :, 1].argmax()][0])
    #     print('bottommost',bottommost)
    #
    # print(area, contours_counter)
    # # cv2.imshow('mask', mask)
    # cv2.imshow('original', image) # original image with added contour
    # # cv2.imshow('opening', blur)
    # # cv2.imshow('detected', output)
    # cv2.waitKey()


    # todo find edges of result
    # https://pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/
    # very difficult, not exactly what I need

    # https://opencvpython.blogspot.com/2012/06/contours-3-extraction.html
    # just need to find contour cnt

    # change mask to contour
    threshold_value = 141
    # mask_grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('output', output)
    # convert gray to binary
    # ret, mask_binary_img = cv2.threshold(mask_grey, threshold_value, 255, cv2.THRESH_BINARY)

    #
    # inverted_binary_img = ~ binary_img
    # cv2.imshow('mask binary', mask_binary_img)

    # show the images
    # cv2.imshow("output", output)
    # cv2.waitKey(0)


def second_approach():
    # https://thinkinfi.com/find-contours-with-opencv-in-python/
    image = cv2.imread(file)  # read file
    # todo find area of interest automatically
    roi = image[y1:y2, x1:x2]  # definition of region of interest (we will be checking for rectangles only there)

    # convert to gray scale
    gray_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 140, 280
    # todo automatic threshold recognition
    # threshold_value = gray_img[140, 280]
    threshold_value = 141 # fixed value, maybe dynamic in future?

    # print(threshold_value)

    # Convert the grayscale image to binary (image binarization opencv python)
    ret, binary_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
    # cv2.imshow('binary image', binary_img)

    # invert image

    inverted_binary_img = ~ binary_img
    cv2.imshow('iverted image', inverted_binary_img)

    # Detect contours
    # hierarchy variable contains information about the relationship between each contours
    contours_list, hierarchy = cv2.findContours(inverted_binary_img,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)  # Find contours

    first_contour = 0
    second_contour = 1
    contour_counter = 0
    for i in contours_list:

        x, y, w, h = cv2.boundingRect(i)

        # Make sure contour area is large enough
        # if (cv2.contourArea(i)) > 6000:
        if h > 20 and w > 20:
            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            contour_counter += 1
            print('shape no: ', contour_counter)
            print('x start: ', x)
            print('y start: ', y)
            print('width :', w)
            print('height :', h)
            # rozpoznawanie cyfr tutaj
            # todo digits recognizion


    print(contour_counter)
    cv2.imshow('first detected contour', roi)

    # cv2.imshow('input image', gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def first_approach():
    # https://stackoverflow.com/questions/61166180/detect-rectangles-in-opencv-4-2-0-using-python-3-7
    # results are not good enough...
    # second attempt up
    image = cv2.imread(file)  # read file
    # image = cv2.resize(image, (width, height))  # image resize
    roi = image[y1:y2, x1:x2]  # definition of region of interest (we will be checking for rectangles only there)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # convert roi to gray
    blur = cv2.GaussianBlur(gray, (9, 9), 1)  # apply blur to roi
    canny = cv2.Canny(blur, 50, 100) # apply canny to roi

    # find contours
    contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    # loop thru contours to find rectangles
    cntrRect = []
    for i in contours:
        epsilon = 0.05 * cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, epsilon, True)
        if len(approx) == 4:
            cv2.drawContours(roi, cntrRect, -1, (0, 255, 0), 2)
            cv2.imshow('roi rect only', roi)
            cntrRect.append(approx)
    print(cntrRect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # auto_threshold('2022-04-26 14_37_31-Window.jpg')
    # x1, y1, x2, y2 = roi_definition('2022-04-26 14_37_31-Window.jpg')
    # print(x1, y1, x2, y2)
    second_approach() # succesful! need to add digit recognizion

    # first_approach()