import numpy as np
import cv2

# Shi-Tomasi corner detection parameter for Feature points
st_params = {   "maxCorners":100,
                "qualityLevel":0.3,
                "minDistance":7,
                "blockSize":7
            }

# Lucas-Kanade method parameters
lk_params = {   "winSize":(15,15),
                "maxLevel": 2,
                "criteria":(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            }

# create random 100 color for plotting
color = np.random.randint(0, 255, (100, 3))

# for ploting detected feature ponints
def detect_corners_plot(img):
    # convert img to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # use Shi-Tomasi corner detection
    features = cv2.goodFeaturesToTrack(gray, mask = None, **st_params)
    # plot feature points
    for i in features:
        x,y = i[0].ravel()
        cv2.circle(img,(x,y),10,255,-1)
    plt.imshow(img)
    plt.title("Detected Corners (Feature points)")
    plt.show()


def optical_flow_plot(video,output,show=True):
    # prepare with first frame 
    end_flag, frame = cap.read()
    gray_prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # get features
    feature_prev = cv2.goodFeaturesToTrack(gray_prev, mask = None, **st_params)
    flow_mask = np.zeros_like(frame)

    while(end_flag):
        # convert to gray scale
        gray_next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect optical flow
        feature_next, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_next, feature_prev, None, **lk_params)

        # chose feature points which detect optical flow（0：not detected、1：detected）
        good_prev = feature_prev[status == 1]
        good_next = feature_next[status == 1]

        # plot optical flow
        for i, (next_point, prev_point) in enumerate(zip(good_next, good_prev)):
            # get x-y coordinate for plotting
            prev_x, prev_y = prev_point.ravel()
            next_x, next_y = next_point.ravel()
            # plot optical flow lint to "flow_mask"
            flow_mask = cv2.line(flow_mask, (next_x, next_y), (prev_x, prev_y), color[i].tolist(), 2)
            # plot current point of features
            frame = cv2.circle(frame, (next_x, next_y), 5, color[i].tolist(), -1)

        # add images with points and optical flow lines
        img = cv2.add(frame, flow_mask)
        output.write(img)

        if show:
            # show
            cv2.imshow('window', img)

        # quit with ESC
        if cv2.waitKey(30) & 0xff == 27:
            break

        # prepare for next frame
        gray_prev = gray_next.copy()
        feature_prev = good_next.reshape(-1, 1, 2)
        end_flag, frame = cap.read()

if __name__ == '__main__': 
    # get video object
    cap = cv2.VideoCapture("Sample.mp4")
    width = int(cap.get(3))
    height = int(cap.get(4))
    # prepare output file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('output.mp4', apiPreference=cv2.CAP_FFMPEG ,fourcc=fourcc, fps=20, frameSize=(width,height))
    # run optical flow plotter
    optical_flow_plot(cap,output)
    # finish
    cv2.destroyAllWindows()
    cap.release()
    output.release()