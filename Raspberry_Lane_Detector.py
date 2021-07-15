import cv2 as cv
import numpy as np
import math
import time



def detect_edges(frame):
    # filter for Black lane lines
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #cv.imshow("hsv", hsv)
    lower_blue = np.array([6, 6, 148])
    upper_blue = np.array([19, 18, 215])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    #cv.imshow("blue mask", mask)
    
    #Applying Threshold
    result = cv.bitwise_and(hsv, hsv, mask = mask)                             
    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 60, 255, cv.THRESH_BINARY)

    # Detecting edges using Canny Edge detection
    edges = cv.Canny(thresh, 200, 400)

    return edges



def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    #focussing on region where lane lines are
    polygon = np.array([[
    (0, height * 1/2),
    (width, height * 1/2),
    (width, height),
    (0, height)]], np.int32)

    cv.fillPoly(mask, polygon, 255)
    cropped_edges = cv.bitwise_and(edges, mask)
    return cropped_edges



def detect_line_segments(cropped_edges):
    # tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # distance precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # angular precision in radian, i.e. 1 degree
    min_threshold = 10  # minimal of votes
    line_segments = cv.HoughLinesP(cropped_edges, rho, angle, min_threshold, 
                                   np.array([]), minLineLength=8, maxLineGap=4)

    return line_segments




def average_slope_intercept(frame, line_segments):

    #This function combines line segments into one or two lane lines
    #If all line slopes are < 0: then we only have detected left lane
    #If all line slopes are > 0: then we only have detected right lane
    
    lane_lines = []
    if line_segments is None:
        print("No lines Found")
        return lane_lines

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on right 2/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                print("Skipping Verticle Line")
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    print("Lane Lines: ", lane_lines)

    return lane_lines




def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1 / 2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]




def detect_lane(frame):
    
    edges = detect_edges(frame)
    cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(frame, line_segments)
    
    return lane_lines



def display_lines(frame, lines, line_color=(0, 255, 0), line_width=10):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image





def display_steering_line(frame, steering_angle, line_color=(255, 0, 0), line_width=10 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right 
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image




def compute_steering_angle(frame, lane_lines):
    """ Find the steering angle based on lane line coordinate
        We assume that camera is calibrated to point to dead center
    """
    if len(lane_lines) == 0:
        print("No lane lines detected, do nothing")
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        print("Only detected one lane line, just follow it", lane_lines[0])
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        left_x2= lane_lines[0][0][2]
        right_x2 = lane_lines[1][0][2]
        camera_mid_offset_percent = 0.02 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # find the steering angle, which is angle between navigation direction to end of center line
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
    steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

    return steering_angle



current_steering_angle = 90

video = cv.VideoCapture("test2.mp4")

# time.sleep(3)

while True:
    start_time = time.time() # start time of the loop
    _, img = video.read()
    
    edges = detect_edges(img)
    lines = detect_lane(img)
    
    lane_lines_image = display_lines(img, lines)
    
    steering_angle = compute_steering_angle(img, lines)
    
    
    final_line = display_steering_line(img, current_steering_angle)
    combined_lines = cv.addWeighted(lane_lines_image, 0.8, final_line, 0.8, 1)
    
    print("New Steering Angle is: ", steering_angle)
    cv.imshow("lane lines", lane_lines_image)
    cv.imshow("Edges", edges)
    
    key = cv.waitKey(1)
    
    if key == ord("q"):
        break
    
    print("FPS: ", 1.0 / (time.time() - start_time)) 
    
    
video.release()
cv.destroyAllWindows()
    


# img = cv.imread("C:/Users/Warrior911/Desktop/FYP/Training Data 1/Lane_Images/img_42.jpg")


# edges = detect_edges(img)
# lines = detect_lane(img)

# lane_lines_image = display_lines(img, lines)
# steering_angle = compute_steering_angle(img, lines)
# display_angle = display_steering_line(img, steering_angle)

# final_line = display_steering_line(img, steering_angle)
# combined_lines = cv.addWeighted(lane_lines_image, 0.8, final_line, 0.8, 1)

# cv.imshow("lane lines", combined_lines)

# key = cv.waitKey(0)


    
