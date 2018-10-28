import cv2 as cv
import time
import numpy as np


#cap = cv.VideoCapture(0)
cap = cv.VideoCapture('./data/video1.mp4')
count = 0

start = time.time()
if cap.isOpened() is True:
    ret, prev_frame = cap.read()
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_RGB2GRAY)
#    height = prev_gray.shape[0]
#    width = prev_gray.shape[1]
    prev_gray = cv.resize(prev_gray, (320, 180), interpolation=cv.INTER_AREA)
    while True:
        ret, curr_frame = cap.read()
        if ret is True:
            count += 1
            curr_gray = cv.cvtColor(curr_frame, cv.COLOR_RGB2GRAY)
            curr_gray = cv.resize(curr_gray, (320, 180), interpolation=cv.INTER_AREA)
            sub_gray = cv.subtract(curr_gray, prev_gray)

            for row in range(180):
                for col in range(320):
                    if sub_gray[row][col] < 50:
                        sub_gray[row][col] = 0
                    if sub_gray[row][col] > 200:
                        sub_gray[row][col] = 255
            cv.imshow("Video", sub_gray)
#            print(sub_gray)
            prev_gray = curr_gray
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    period = time.time() - start
    print(period)
    cv.destroyAllWindows()

cap.release()