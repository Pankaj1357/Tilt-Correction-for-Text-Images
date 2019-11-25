# importing some necessary libraries
import cv2
import numpy as np
import skimage
import matplotlib.pyplot as plt
from PIL import Image
############################ This script contains the function which will take tilted image as inpu
############################ and produce final tilt corrected image alog with stepwise outputs off
############################ all six stages mentioned in jupyter notebook

def stage_wise_tilt_correction(image):
	"""This function is the main key here. It take input as tilted image and returns tilt
	   corrected image along with all stage ouput of the proecess"""
	
	################################### STAGE 1 : PRE-PROCESSING ###########################################
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # getting gray scale from RGB image


	################################### STAGE 2 : GAUSSIAN BLURRING #########################################
	#making a Gaussian low pass filter mask
	gaussian_LP_filter_mask = np.zeros((2700, 2700))
	D0 = 320
	m = gaussian_LP_filter_mask.shape[0]
	n = gaussian_LP_filter_mask.shape[1]
	for i in range(m):
	    for j in range(n):
	        power = ((i -(m/2))**2 + (j -(n/2))**2) / (2 * D0**2)
	        gaussian_LP_filter_mask[i, j] = np.exp(-power)

	        
	gaussian_LP_filter_mask = gaussian_LP_filter_mask.astype(float)
	

	# getting image fft
	gray_image_fft = np.fft.fft2(gray, s=(2700, 2700), axes=(-2, -1), norm=None) 

	# doing low pass filterin in frequency domain using Gaussian filter.
	filtered_image_fft = np.fft.fftshift(gray_image_fft) * gaussian_LP_filter_mask

	# Taking IFFT and obtaining low pass filtered Image.
	filtered_image = np.fft.ifft2(filtered_image_fft, s=None, axes=(-2, -1), norm=None)
	filtered_image = rescale(filtered_image)[:image.shape[0], :image.shape[1]].astype(np.uint8)


	################################### STAGE 3 : EDGE DETECTION- CANNY #####################################
	edges = cv2.Canny(filtered_image, 100, 250)

	################################## STAGE 4 : LINE DETECTION - HOUGH ####################################
	line_detector = Line_Detection(edges)
	lines = line_detector.get_hough_lines(min_line_len=200, max_line_len=5000)

    ################################ STAGE 5 : CORNER POINT DETECTION #####################################
	corner_detector = Corner_Detection(lines)        #instantiating the Corner detection
	points = corner_detector.get_corner_points()     # getting the four detected corner points
	img = image.copy()
	plot_corner_points = corner_detector.visualize_corner_points(img, points, 26)


	################################ STAGE 6 : PERSPECTIVE TRANSFORMATION ###############################
	# image1 = image.copy()
	warping = PerspectiveTransform(points)
	final_output = warping.warp_perspective(image)






	return (image, gray, filtered_image, edges, line_detector.draw_hough_lines(lines), plot_corner_points, final_output)
















# fucntion to help with obtaining magnitude spectrum of fft and rescale the image for 
# display purposes.
def rescale(x):
    x = np.abs(x)
    x = x - np.min(x)
    x = (x * 255) / np.max(x)
    return x




class Line_Detection:
    
    def __init__(self, edges):
        self.edges = edges
        
    
    def get_hough_lines(self, min_line_len= 200, max_line_len=5000):
        lines = cv2.HoughLinesP(self.edges, 1, np.pi/180, 100,
                                minLineLength=min_line_len, maxLineGap=max_line_len)
        return lines
    
    
    def draw_hough_lines(self, lines):
        drawing = np.zeros(self.edges.shape, np.uint8)
        for line in lines: 
            a, b, a1, b1 = line[0]
            cv2.line(drawing, (a,b), (a1, b1), (255, 255,255), 2)

        return drawing



class Corner_Detection:
    
    def __init__(self, lines):
        self.lines = lines
           
    def get_corner_points(self):
        
        toplx, toply = 10000, 10000
        toprx, topry = -10000, 10000
        botlx, botly = 10000, -10000
        botrx, botry = -10000, -10000

        for line in self.lines:              # First Point
            arr = np.array(line).squeeze()
            
            x1, y1, x2, y2 = arr
            
            if -x1-y1>-toplx-toply:
                toplx, toply = x1, y1
                
            if -x2-y2>-toplx-toply:
                toplx, toply = x2, y2
                
        for line in self.lines:                         # Second Point
            arr = np.array(line).squeeze()
            
            x1, y1, x2, y2 = arr

            if -y1+x1>toprx-topry:
                toprx, topry= x1, y1

            if -y2+x2>toprx-topry:
                toprx, topry = x2, y2

        for line in self.lines:                      # Third Point
            arr = np.array(line).squeeze()
            
            x1, y1, x2, y2 = arr

            if y1-x1>-botlx+botly:
                botlx, botly = x1, y1

            if y2-x2>-botlx+botly:
                botlx, botly = x2, y2

        for line in self.lines:                       # Fourth Point
            arr = np.array(line).squeeze()
            
            x1, y1, x2, y2 = arr

            if y1+x1>botrx+botry:
                botrx, botry = x1, y1

            if y2+x2>botrx+botry:
                botrx, botry = x2, y2
         
        # return all points in a tuple
        return (toplx, toply, toprx, topry, botlx, botly, botrx, botry)
    
    
    def visualize_corner_points(self, img, detected_points, point_size):
        toplx, toply, toprx, topry, botlx, botly, botrx, botry = detected_points
        
        cv2.circle(img,(toplx,toply), point_size, (0, 0, 255), -1)
        cv2.circle(img,(toprx,topry), point_size, (255, 255, 255), -1)

        cv2.circle(img,(botlx,botly), point_size, (0, 255, 0), -1)
        cv2.circle(img,(botrx,botry), point_size, (255, 0, 255), -1)
        
        return img





class PerspectiveTransform:
    def __init__(self, detected_points):
        
        self.detected_points = detected_points
        
    def get_rectangle(self):
        toplx, toply, toprx, topry, botlx, botly, botrx, botry = self.detected_points
        pts = np.array([(toplx,toply),(toprx,topry),(botlx,botly),(botrx,botry)])
        
        rectangle = np.zeros((4, 2), dtype = "float32")
        
        sum_row = pts.sum(axis = 1)
        rectangle[0] = pts[np.argmin(sum_row)]
        rectangle[2] = pts[np.argmax(sum_row)]
        
        diff_row = np.diff(pts, axis = 1)
        rectangle[1] = pts[np.argmin(diff_row)]
        rectangle[3] = pts[np.argmax(diff_row)]
        
        return rectangle
    
    def warp_perspective(self, image):
        
        rect = self.get_rectangle()
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([[0, 0], [maxWidth-1, 0], [maxWidth-1, maxHeight-1],
            [0, maxHeight-1]], dtype = "float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
