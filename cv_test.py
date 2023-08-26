import cv2 as cv
import numpy as np

# define seven colors
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 0)]

img1 = cv.imread('images/test/img1.png')
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

# x1 = cv.Sobel(img1, cv.CV_16S, 1, 0)
# y1 = cv.Sobel(img1, cv.CV_16S, 0, 1)
# absX1 = cv.convertScaleAbs(x1)
# absY1 = cv.convertScaleAbs(y1)
# dst1 = cv.addWeighted(absX1, 0.5, absY1, 0.5, 0)

img2 = cv.imread('images/test/img2.png')
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# x2 = cv.Sobel(img2, cv.CV_16S, 1, 0)
# y2 = cv.Sobel(img2, cv.CV_16S, 0, 1)
# absX2 = cv.convertScaleAbs(x2)
# absY2 = cv.convertScaleAbs(y2)
# dst2 = cv.addWeighted(absX2, 0.5, absY2, 0.5, 0)

# res = np.abs(dst2 - dst1)
# res = np.hstack((dst1, dst2))

diff_gray = cv.absdiff(img1_gray, img2_gray)
diff = cv.cvtColor(diff_gray, cv.COLOR_GRAY2BGR)

diff_blurred = cv.GaussianBlur(diff_gray, (5, 5), 0)
# ret, thresh = cv.threshold(img1_blurred, 127, 255, 0)
edges = cv.Canny(diff_blurred, 50, 150, apertureSize=3)
im, contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# cv.drawContours(img1, contours, -1, (0, 0, 255), 1)
print('Objects detected: ', len(contours))
for i in range(len(contours)):
    cv.drawContours(diff, contours, i, colors[i], 1)
res = diff

cv.imshow('res', res)
if cv.waitKey(0) == 32:
    cv.destroyAllWindows()
