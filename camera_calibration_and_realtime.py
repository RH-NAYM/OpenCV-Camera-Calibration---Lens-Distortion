import cv2
import numpy as np
import glob
import os

# CONFIGURATION

CHESSBOARD_SIZE = (7, 6)  # inner corners
SQUARE_SIZE = 1.0  # real-world size of each square (mm)
CALIB_IMAGES_DIR = "images"
CAMERA_INDEX = 0
CALIB_WIDTH = 640
CALIB_HEIGHT = 480


# CAMERA CALIBRATION (ALWAYS RUN)
def calibrate_camera():
    print("[INFO] Starting camera calibration...")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare 3D object points
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHESSBOARD_SIZE[0], 0 : CHESSBOARD_SIZE[1]].T.reshape(
        -1, 2
    )
    objp *= SQUARE_SIZE

    objpoints, imgpoints = [], []

    images = glob.glob(os.path.join(CALIB_IMAGES_DIR, "*.jpg"))
    if not images:
        raise FileNotFoundError(f"No calibration images found in {CALIB_IMAGES_DIR}")

    for fname in images:
        img = cv2.imread(filename=fname)
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(image=gray, patternSize=CHESSBOARD_SIZE)
        if found:
            corners = cv2.cornerSubPix(image=gray, corners=corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria)
            objpoints.append(objp)
            imgpoints.append(corners)

            cv2.drawChessboardCorners(image=img, patternSize=CHESSBOARD_SIZE, corners=corners, patternWasFound=found)
            cv2.imshow(winname="Calibration", mat=img)
            cv2.waitKey(delay=100)

    cv2.destroyAllWindows()

    # Compute camera matrix and distortion coefficients
    error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints, imagePoints=imgpoints, imageSize=gray.shape[::-1], cameraMatrix=None, distCoeffs=None
    )
    print("[INFO] Calibration complete")
    print(f"[INFO] Mean reprojection error: {error:.6f}")

    return mtx, dist


# REALTIME UNDISTORTION
def realtime_undistort(mtx, dist):
    print("[INFO] Starting realtime undistortion...")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read from camera")
    h, w = frame.shape[:2]

    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(cameraMatrix=mtx, distCoeffs=dist, imageSize=(w, h), alpha=1)
    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=mtx, distCoeffs=dist, newCameraMatrix=newcameramtx, size=(w, h), rtype=cv2.CV_16SC2
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistorted = cv2.remap(src=frame, map1=map1, map2=map2, interpolation=cv2.INTER_LINEAR)

        cv2.imshow("Original", frame)
        cv2.imshow("Undistorted", undistorted)

        if cv2.waitKey(delay=1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


# OPTIONAL: REALTIME POSE ESTIMATION
def realtime_pose(mtx, dist):
    print("[INFO] Starting realtime pose estimation...")

    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHESSBOARD_SIZE[0], 0 : CHESSBOARD_SIZE[1]].T.reshape(
        -1, 2
    )
    objp *= SQUARE_SIZE

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]])

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CALIB_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CALIB_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(image=gray, patternSize=CHESSBOARD_SIZE)

        if found:
            corners = cv2.cornerSubPix(image=gray, corners=corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria)
            _, rvec, tvec = cv2.solvePnP(objectPoints=objp, imagePoints=corners, cameraMatrix=mtx, distCoeffs=dist)
            imgpts, _ = cv2.projectPoints(objectPoints=axis, rvec=rvec, tvec=tvec, cameraMatrix=mtx, distCoeffs=dist)

            corner = tuple(corners[0].ravel().astype(int))
            frame = cv2.line(
                frame, corner, tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 3
            )
            frame = cv2.line(
                frame, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3
            )
            frame = cv2.line(
                frame, corner, tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 3
            )

        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# ENTRY POINT
if __name__ == "__main__":
    # Always recalibrate
    mtx, dist = calibrate_camera()

    # Default: realtime undistortion
    realtime_undistort(mtx, dist)
    # realtime_pose(mtx, dist)  # Uncomment to enable pose estimation
