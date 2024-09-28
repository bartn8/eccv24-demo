
import shutil
import cv2
import numpy as np
import glob
import tqdm
import os
import json
import argparse
from utils import transform_inv

# Create the argument parser
parser = argparse.ArgumentParser(description="MID70+OAK calibration script.")

# Add a positional argument for the save folder path
parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the chessboard folder")
parser.add_argument("--square_size", type=int, default=17, help="Length of chessboard square in mm.")
parser.add_argument("--grid_size", nargs='+', default=[9,6], help="Chessboard pattern size")
# parser.add_argument("--initial_rt", type=str, default=None, help="Initial RT guess (not used in current implementation)")
parser.add_argument("--output_folder", type=str, default="mid70_oak_calib", help="Output file for the RT matrices")
parser.add_argument("--binary_threshold", type=int, default=10, help="Binary threshold for the chessboard detection")
parser.add_argument("--median_kernel_size", type=int, default=3, help="Median kernel size for the chessboard detection")

# Parse the command-line arguments
args = parser.parse_args()

#Path to recorded chessboard
dataset_path = args.dataset_dir

square_size = args.square_size / 1000.0
chessboard_grid_size = tuple([int(x) for x in args.grid_size])


MID70_SAVE_FOLDER = "mid70"
MID70_DEPTH_FOLDER = "depth"
MID70_REFLECTIVITY_FOLDER = "reflectivity"
MID70_CALIB_FILE = "calib.json"

OAK_SAVE_FOLDER = "oak"
OAK_LEFT_FOLDER = "left"
OAK_RIGHT_FOLDER = "right"
OAK_DEPTH_FOLDER = "depth"
OAK_DISPARITY_FOLDER = "disparity"
OAK_CALIB_FILE = "calib.json"

#Assume a folder with sinchronized depth frames

with open(os.path.join(dataset_path, OAK_SAVE_FOLDER, OAK_CALIB_FILE), "r") as f:
    oak_calib_data = json.load(f)

with open(os.path.join(dataset_path, MID70_SAVE_FOLDER, MID70_CALIB_FILE), "r") as f:
    mid70_calib_data = json.load(f)

RT_CRF_MID70 = np.array(mid70_calib_data["RT_crf"])
D_mid70 = np.zeros(5)


#Do global registration on all dataset... No use CAD to get initial RT
#Format: RT_dst_src

# Suppose Z-axis rotation >> X-axis and Y-axis rotation
# if os.path.exists(args.initial_rt):
#     initial_RT_OAK_MID70 = np.loadtxt(args.initial_rt)
# else:

#     _initial_rot_vec_1 = np.array([[0.001,0.001,0.001]], dtype=np.float32) * np.random.randn(1,3)

#     initial_RT_OAK_MID70 = np.eye(4) 
#     initial_RT_OAK_MID70[0,3] =  40.0 / 1000.0    # X translation
#     initial_RT_OAK_MID70[1,3] = -40.0 / 1000.0      # Y translation
#     initial_RT_OAK_MID70[2,3] = -40.0 / 1000.0      # Z translation
#     initial_RT_OAK_MID70[:3,:3] = cv2.Rodrigues(_initial_rot_vec_1)[0]

# print(initial_RT_OAK_MID70)

#Term criteria

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
indices = np.indices(chessboard_grid_size, dtype=np.float32)
indices *= square_size
coords_3D = np.transpose(indices, [2, 1, 0])
coords_3D = coords_3D.reshape(-1, 2)
pattern_points = np.concatenate([coords_3D, np.zeros([coords_3D.shape[0], 1], dtype=np.float32)], axis=-1)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints_mid70 = [] # 2d points in image plane.
imgpoints_oak = [] # 2d points in image plane.

mid70_reflectivitys = glob.glob(os.path.join(dataset_path,MID70_SAVE_FOLDER,MID70_REFLECTIVITY_FOLDER,"*.png"))
oak_lefts = glob.glob(os.path.join(dataset_path,OAK_SAVE_FOLDER,OAK_LEFT_FOLDER,"*.png"))

for mid70_reflectivity, oak_left in tqdm.tqdm(zip(mid70_reflectivitys, oak_lefts), total=len(mid70_reflectivitys)):
    mid70_img = cv2.imread(mid70_reflectivity)
    mid70_img = cv2.cvtColor(mid70_img, cv2.COLOR_BGR2GRAY) 
    mid70_img[mid70_img>=args.binary_threshold] = 255
    mid70_img = cv2.medianBlur(mid70_img, args.median_kernel_size)
    H_mid70, W_mid70 = mid70_img.shape[:2]

    oak_img = cv2.imread(oak_left)
    oak_img = cv2.cvtColor(oak_img, cv2.COLOR_BGR2GRAY)
    H_oak, W_oak = oak_img.shape[:2]

    # Find the chess board corners
    mid70_ret, mid70_corners = cv2.findChessboardCorners(mid70_img, chessboard_grid_size, None)
    oak_ret, oak_corners = cv2.findChessboardCorners(oak_img, chessboard_grid_size, None)

    # If found, add object points, image points (after refining them)
    if mid70_ret and oak_ret:
        mid70_corners = cv2.cornerSubPix(mid70_img, mid70_corners, (11,11), (-1,-1), criteria)
        oak_corners = cv2.cornerSubPix(oak_img, oak_corners, (11,11), (-1,-1), criteria)

        mid70_img = cv2.cvtColor(mid70_img, cv2.COLOR_GRAY2BGR)
        oak_img = cv2.cvtColor(oak_img, cv2.COLOR_GRAY2BGR)
        # Draw and display the corners
        mid70_img = cv2.drawChessboardCorners(mid70_img, chessboard_grid_size, mid70_corners, mid70_ret)
        oak_img = cv2.drawChessboardCorners(oak_img, chessboard_grid_size, oak_corners, oak_ret)
        oak_img = cv2.resize(oak_img, (mid70_img.shape[1], mid70_img.shape[0]))

        print(f"MID70 image path: {mid70_reflectivity}")
        print(f"OAK image path: {oak_left}")
        print(f"Press 's' to skip this frame, any other key to accept the frame")

        cv2.imshow('Chessboards', np.hstack([mid70_img, oak_img]))
        key = cv2.waitKey(0)

        if key != ord('s'):
            objpoints.append([pattern_points])
            imgpoints_mid70.append([mid70_corners])
            imgpoints_oak.append([oak_corners])

            print("Frame accepted!")

if len(objpoints) == 0:
    print("No valid frames found. Exiting.")
    exit()

rvecs_list = []
tvecs_list = []

K_oak = np.array(oak_calib_data[f"K_left_{H_oak}x{W_oak}"])
K_mid70 = np.array(mid70_calib_data[f"K_depth_{H_mid70}x{W_mid70}"])

for i in range(len(objpoints)):
    retval_mid70, rvecs_mid70, tvecs_mid70, error_mid70 = cv2.solvePnPRansac(np.array(objpoints[i]).squeeze(), np.array(imgpoints_mid70[i]).squeeze(), K_mid70, np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
    retval_oak, rvecs_oak, tvecs_oak, error_oak = cv2.solvePnPRansac(np.array(objpoints[i]).squeeze(), np.array(imgpoints_oak[i]).squeeze(), K_oak, np.zeros(5), flags=cv2.SOLVEPNP_IPPE)

    if retval_mid70 and retval_oak:
        _RT1 = np.eye(4)
        _RT1[:3,:3] = cv2.Rodrigues(rvecs_mid70)[0]
        _RT1[:3,3] = tvecs_mid70[:,0]

        _RT2 = np.eye(4)
        _RT2[:3,:3] = cv2.Rodrigues(rvecs_oak)[0]
        _RT2[:3,3] = tvecs_oak[:,0]

        _RT1 = transform_inv(_RT1)
        _RT = np.dot(_RT2, _RT1)

        rvecs_list.append(cv2.Rodrigues(_RT[:3,:3])[0])
        tvecs_list.append(_RT[:3,3])

        # print(f"{i} - MID70 error: {error_mid70} - OAK error: {error_oak}")
        print(f"{i} - RT tvecs: {_RT[:3,3]}")

# Do the median of rvecs and tvecs
rvecs_list = np.array(rvecs_list) # Nx3
tvecs_list = np.array(tvecs_list) # Nx3

R = cv2.Rodrigues(np.median(rvecs_list, axis=0))[0]
T = np.median(tvecs_list, axis=0)
    
RT_leftr_livox = np.eye(4)
RT_leftr_livox[:3,:3] = R
RT_leftr_livox[:3, 3] = T.flatten()

RT_leftr_livox = RT_leftr_livox @ RT_CRF_MID70
os.makedirs(args.output_folder, exist_ok=True)
np.savetxt(os.path.join(args.output_folder, "RT_LEFTR_LIVOX.txt"), RT_leftr_livox)

_R_leftr = oak_calib_data[f'R_leftr']
R_leftr = np.eye(4)
R_leftr[:3,:3] = _R_leftr
RT_left_livox = transform_inv(R_leftr) @ RT_leftr_livox
np.savetxt(os.path.join(args.output_folder, "RT_LEFT_LIVOX.txt"), RT_left_livox)

RT_rgb_left = oak_calib_data['RT_rgb_left']
RT_rgb_livox = RT_rgb_left @ RT_left_livox
np.savetxt(os.path.join(args.output_folder, "RT_RGB_LIVOX.txt"), RT_rgb_livox)

RT_right_left = oak_calib_data['RT_right_left']
RT_right_livox = RT_right_left @ RT_left_livox
np.savetxt(os.path.join(args.output_folder, "RT_RIGHT_LIVOX.txt"), RT_right_livox)

_R_rightr = oak_calib_data['R_rightr']
R_rightr = np.eye(4)
R_rightr[:3,:3] = _R_rightr
RT_rightr_livox = R_rightr @ RT_right_livox
np.savetxt(os.path.join(args.output_folder, "RT_RIGHTR_LIVOX.txt"), RT_rightr_livox)

#copy calibration files
shutil.copyfile(os.path.join(dataset_path, MID70_SAVE_FOLDER, MID70_CALIB_FILE), f"{args.output_folder}/mid70_calib.json")
shutil.copyfile(os.path.join(dataset_path, OAK_SAVE_FOLDER, OAK_CALIB_FILE), f"{args.output_folder}/oak_calib.json")

