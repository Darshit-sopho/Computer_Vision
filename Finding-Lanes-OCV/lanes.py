import cv2
import numpy as np


def make_coordinates(image, parameters):
    slope, intercept = parameters
    y1 = image.shape[0]
    y2 = int(y1*3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])s


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_average = np.average(left_fit, axis=0)
    right_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_average)
    right_line = make_coordinates(image, right_average)
    return np.array([left_line, right_line])


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    candy = cv2.Canny(blur, 50, 100)
    return candy


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 8)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(220, height), (1100, height), (570, 260)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# image = cv2.imread("test_image.jpg")
# lane_image = np.copy(image)
# edges = canny(lane_image)
# cropped_image = region_of_interest(edges)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# detected = cv2.addWeighted(lane_image, 0.6, line_image, 1, 1)
# cv2.imshow("results", detected)
# cv2.waitKey(0)

cap = cv2.VideoCapture("test2.mp4")
while cap.isOpened():
    _, frame = cap.read()
    edges = canny(frame)
    cropped_images = region_of_interest(edges)
    lines = cv2.HoughLinesP(cropped_images, 2, np.pi / 180, 20, np.array([]), minLineLength=10, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    detected = cv2.addWeighted(frame, 0.6, line_image, 1, 1)
    cv2.imshow("result", detected)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
