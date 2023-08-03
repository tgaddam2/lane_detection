import cv2
import numpy as np
import matplotlib.pyplot as plt
import time, copy
from PIL import Image

# https://hackthedeveloper.com/lane-detection-opencv-python/

def process_and_draw(image, visualize=False, scale_fac=500):
    # max_w = scale_fac
    # r = max_w / image.shape[1]
    # dim = (max_w, int(image.shape[0] * r))
    # # perform the actual resizing of the image
    # resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # image = resized_image
    
    height, width = image.shape[:2]
    
    # cropped_image = image[int(height * 0.60):height, 10:width]
    crop_multiplier = 0.70
    cropped_image = image[int(height * crop_multiplier):height, 10:width]
    
    # converting to greyscale
    # gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray, cmap='gray')
    # plt.show()

    # bilateral = cv2.bilateralFilter(gray, 20, 75, 75)
    bilateral = cv2.bilateralFilter(cropped_image, d=35, sigmaColor=250, sigmaSpace=50)
    # plt.imshow(bilateral, cmap='gray')
    # plt.show()
    
    # converting to grayscale
    gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray, cmap='gray')
    
    # gaussian blur
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # plt.imshow(blur, cmap='gray')
    # plt.show()

    # canny edge detection
    # edges = cv2.Canny(blur, 50, 100)
    edges = cv2.Canny(gray, 50, 100)
    # plt.imshow(edges)
    # plt.show()

    # hough transform
    lines_right = cv2.HoughLinesP(edges, rho=6, theta=np.pi/105, threshold=160, minLineLength=125, maxLineGap=10)
    lines_left = cv2.HoughLinesP(edges, rho=6, theta=np.pi/75, threshold=160, minLineLength=125, maxLineGap=10)
    
    if lines_right is None or lines_left is None:
        return [None, None]
    
    if len(lines_right) == 0 or len(lines_left) == 0:
        return [None, None]
    
    # [:, 0:int((width)//2)]
    # [:, int((width)//2):width]
    # print(f"left: {len(lines_left)}")
    # print(f"right: {len(lines_right)}")
    lines = np.concatenate((lines_left, lines_right))
    # print(lines)
    cropped_image = bilateral
    if visualize:
        # drawing lines
        line_image = np.zeros_like(cropped_image)
        # line_image = np.zeros_like(image)
        left_lines = []
        right_lines = []

        left_x_coords = []
        right_x_coords = []
        
        x_coords = []
        y_coords = []
    
        # print(lines)
        
        if lines is None:
            return [None, None]
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # height_chopped = height - cropped_image.shape[0] - 10
            # y1 += height_chopped
            # y2 += height_chopped
            
            x_coords.append(x1)
            x_coords.append(x2)
            y_coords.append(y1)
            y_coords.append(y2)
            
            left_color = (0, 255, 0)
            right_color = (0, 0, 255)
            
            if x1 < (width / 2):
                cv2.line(line_image, (x1, y1), (x2, y2), left_color, 1)
                # print(f"left: {line}")
                left_lines.append(line[0])
                left_x_coords.append(x1)
                left_x_coords.append(x2)
            else:
                cv2.line(line_image, (x1, y1), (x2, y2), right_color, 1)
                # print(f"right: {line}")
                right_lines.append(line[0])
                right_x_coords.append(x1)
                right_x_coords.append(x2)

        # x_avg = np.average(x_coords)
        # print(x_avg)
        left_avg = np.average(left_x_coords)
        right_avg = np.average(right_x_coords)        
        x_avg = (left_avg + right_avg) / 2
        # print(left_avg)
        # print(right_avg)
        # print(x_avg)
        y_avg = np.average(y_coords)

        center_x = width / 2
        center_y = height / 2
        offset = (width / 2) - x_avg
        # offset_x = center_x + int(center_x * offset)

        # cv2.circle(line_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
        # cv2.line(line_image, (int(center_x), int(center_y)), (int(offset_x), int(center_y)), (0,0,255), 3)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontColor = (0, 0, 0)
        fontSize = 1
        cv2.putText(cropped_image, 'img center: {:.4f} px'.format(center_x), (int(center_x) - 200, 200), font, fontSize, fontColor, 4)
        cv2.putText(cropped_image, 'lane center: {:.4f} px'.format(x_avg), (int(center_x) - 200, 250), font, fontSize, fontColor, 4)
        cv2.putText(cropped_image, 'offset: {:.4f} px'.format(offset), (int(center_x) - 200, 300), font, fontSize, fontColor, 4)

        # overlyaing lane lines on original image
        final_image = cv2.addWeighted(cropped_image, 0.8, line_image, 1, 0)
        
        # display image
        # plt.imshow(final_image)
        # plt.show()
        
        # cv2.circle(final_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
        # cv2.line(final_image, (int(center_x), int(center_y)), (int(offset_x), int(center_y)), (0,0,255), 3)
        
        # image = np.array(Image.open(image))
        image_copy = copy.deepcopy(image)
        image_copy[int(height * crop_multiplier):height, 10:width, :] = final_image[:, :, :]   
        
        # plt.imshow(image_copy)
        # plt.show()
    else:
        left_x_coords = []
        right_x_coords = []
        
        x_coords = []
        y_coords = []
        
        if lines is None:
            return [None]

        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            x_coords.append(x1)
            x_coords.append(x2)
            y_coords.append(y1)
            y_coords.append(y2)
            
            if x1 < (width / 2):
                left_x_coords.append(x1)
                left_x_coords.append(x2)
            else:
                right_x_coords.append(x1)
                right_x_coords.append(x2)        

        # x_avg = np.average(x_coords)
        # print(x_avg)
        left_avg = np.average(left_x_coords)
        right_avg = np.average(right_x_coords)        
        x_avg = (left_avg + right_avg) / 2
        # print(left_avg)
        # print(right_avg)
        # print(x_avg)
        y_avg = np.average(y_coords)

        center_x = width / 2
        offset = (width / 2) - x_avg
    
    if visualize:
        return [offset, image_copy]
    return [offset]

if __name__ == '__main__':
    image = cv2.imread('test_images/5.png')
    offset, image = process_and_draw(image, visualize=True)
    print(offset)
    plt.imshow(image)
    plt.show()
    
    image = cv2.imread('test_images/6.png')
    offset, image = process_and_draw(image, visualize=True)
    print(offset)
    plt.imshow(image)
    plt.show()
    
    image = cv2.imread('test_images/7.png')
    offset, image = process_and_draw(image, visualize=True)
    print(offset)
    plt.imshow(image)
    plt.show()
    
    image = cv2.imread('test_images/8.png')
    offset, image = process_and_draw(image, visualize=True)
    print(offset)
    plt.imshow(image)
    plt.show()
    
    image = cv2.imread('test_images/9.png')
    offset, image = process_and_draw(image, visualize=True)
    print(offset)
    plt.imshow(image)
    plt.show()
    
    image = cv2.imread('test_images/10.png')
    offset, image = process_and_draw(image, visualize=True)
    print(offset)
    plt.imshow(image)
    plt.show()