import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# https://hackthedeveloper.com/lane-detection-opencv-python/

def process_and_draw(image, visualize=False, scale_fac=500):
    # max_w = scale_fac
    # r = max_w / image.shape[1]
    # dim = (max_w, int(image.shape[0] * r))
    # # perform the actual resizing of the image
    # resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # image = resized_image
    
    height, width = image.shape[:2]
    
    image = image[int(height * 0.60):height, :]
    
    # converting to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray, cmap='gray')
    # plt.show()

    bilateral = cv2.bilateralFilter(gray, 20, 75, 75)
    # plt.imshow(bilateral, cmap='gray')
    # plt.show()

    # gaussian blur
    blur = cv2.GaussianBlur(bilateral, (5, 5), 0)
    # plt.imshow(blur, cmap='gray')
    # plt.show()

    # canny edge detection
    edges = cv2.Canny(blur, 50, 100)
    # plt.imshow(edges)
    # plt.show()

    # hough transform
    lines = cv2.HoughLinesP(edges, rho=6, theta=np.pi/60, threshold=160, minLineLength=40, maxLineGap=25)
    # print(lines)

    if visualize:
        # drawing lines
        line_image = np.zeros_like(image)
        left_lines = []
        right_lines = []

        x_coords = []
        y_coords = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            x_coords.append(x1)
            x_coords.append(x2)
            y_coords.append(y1)
            y_coords.append(y2)
            
            left_color = (0, 255, 0)
            right_color = (0, 0, 255)
            
            if x1 < (width / 2):
                cv2.line(line_image, (x1, y1), (x2, y2), left_color, 5)
                # print(f"left: {line}")
                left_lines.append(line[0])
            else:
                cv2.line(line_image, (x1, y1), (x2, y2), right_color, 5)
                # print(f"right: {line}")
                right_lines.append(line[0])

        x_avg = np.average(x_coords)
        y_avg = np.average(y_coords)

        img_center = width / 2
        offset = (width / 2) - x_avg

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontColor = (0, 0, 0)
        fontSize = 1.5
        cv2.putText(image, 'img center: {:.4f} px'.format(img_center), (int(img_center) - 200, 350), font, fontSize, fontColor, 4)
        cv2.putText(image, 'lane center: {:.4f} px'.format(x_avg), (int(img_center) - 200, 400), font, fontSize, fontColor, 4)
        cv2.putText(image, 'offset: {:.4f} px'.format(offset), (int(img_center) - 200, 450), font, fontSize, fontColor, 4)

        # overlyaing line on original image
        final_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
        
        # display image
        plt.imshow(final_image)
        plt.show()
    else:        
        x_coords = []
        y_coords = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            x_coords.append(x1)
            x_coords.append(x2)
            y_coords.append(y1)
            y_coords.append(y2)            

        x_avg = np.average(x_coords)
        y_avg = np.average(y_coords)

        img_center = width / 2
        offset = (width / 2) - x_avg
    
    return offset

image = cv2.imread('test_images/5.png')

start_time = time.time()
print(process_and_draw(image, visualize=True, scale_fac=500))
end_time = time.time()