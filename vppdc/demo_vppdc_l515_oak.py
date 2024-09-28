
import numpy as np
import random
import cv2
import argparse
import time
import os
import json

import pyrealsense2 as rs
import depthai as dai  

from utils import sample_hints_cache
from vpp_standalone import vpp
from filter import occlusion_heuristic


RS2_L500_VISUAL_PRESET_DEFAULT = 1
RS2_L500_VISUAL_PRESET_NO_AMBIENT = 2
RS2_L500_VISUAL_PRESET_LOW_AMBIENT = 3
RS2_L500_VISUAL_PRESET_MAX_RANGE = 4
RS2_L500_VISUAL_PRESET_SHORT_RANGE = 5

L515_VISUAL_PRESET_DICT = {
    "default": RS2_L500_VISUAL_PRESET_DEFAULT,
    "no_ambient": RS2_L500_VISUAL_PRESET_NO_AMBIENT,
    "low_ambient": RS2_L500_VISUAL_PRESET_LOW_AMBIENT,
    "max_range": RS2_L500_VISUAL_PRESET_MAX_RANGE,
    "short_range": RS2_L500_VISUAL_PRESET_SHORT_RANGE,
}

OAK_CALIB_FILE = "oak_calib.json"
CALIB_PATH = "l515_oak_calib"

with open(os.path.join(CALIB_PATH, OAK_CALIB_FILE), "r") as f:
    oak_calib_data = json.load(f)

TIMEOUT_SGM = 0.5
OAK_BASELINE = oak_calib_data["baseline"]
OAK_SUBPIXEL = True
OAK_LRC = True
OAK_EXTENDED_DISPARITY = False
OAK_FRACTIONAL_BITS = 3
OAK_DISPARITY_DIV = 2 ** OAK_FRACTIONAL_BITS if OAK_SUBPIXEL else 1
OAK_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P

OAK_RESOLUTION_DICT = {
    dai.MonoCameraProperties.SensorResolution.THE_400_P: (400, 640),
    dai.MonoCameraProperties.SensorResolution.THE_480_P: (480, 640),
    dai.MonoCameraProperties.SensorResolution.THE_720_P: (720, 1280),
    dai.MonoCameraProperties.SensorResolution.THE_800_P: (800, 1280),
}

H = OAK_RESOLUTION_DICT[OAK_RESOLUTION][0]
W = OAK_RESOLUTION_DICT[OAK_RESOLUTION][1]

DISP_MAX = 96

#Create argparse object: 1 arg for stereo_model_path, 1 arg for the stereo model type
parser = argparse.ArgumentParser(description='L515 VPPDC Demo with OAK-D\'s SGM')
parser.add_argument('--l515_preset', type=str, default='short_range', choices=['default', 'no_ambient', 'low_ambient', 'max_range', 'short_range'], help='L515 visual preset')
args = parser.parse_args()

args.seed = 42

args.max_z = 10
args.virtual_baseline = OAK_BASELINE
args.sampling_perc = 0.05

args.leftpadding = False
args.blending = 1.0
args.maskocc = True
args.cblending = 0
args.uniform_color = False
args.filling = True
args.o_xy = 1
args.o_i = 1
args.th_adpt = 1e-3
args.interpolate = True
args.wsize = 5
args.context = False

args.use_blob_remover = True


random.seed(args.seed)
np.random.seed(args.seed)

#Initialize OAK-D pipeline as SGM processor
pipeline_OAK = dai.Pipeline()

vpp_sgm_node = pipeline_OAK.createStereoDepth()
vpp_sgm_node.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
vpp_sgm_node.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
vpp_sgm_node.setLeftRightCheck(OAK_LRC)
vpp_sgm_node.setExtendedDisparity(OAK_EXTENDED_DISPARITY)
vpp_sgm_node.setSubpixel(OAK_SUBPIXEL)
vpp_sgm_node.setSubpixelFractionalBits(OAK_FRACTIONAL_BITS)
vpp_sgm_node.setDepthAlign(dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
vpp_sgm_node.setInputResolution(W, H)
vpp_sgm_node.setRectification(False)# Images are already rectified
vpp_sgm_node.setRuntimeModeSwitch(True)

stereo_config = vpp_sgm_node.initialConfig.get()
stereo_config.postProcessing.brightnessFilter.minBrightness = 0
stereo_config.postProcessing.brightnessFilter.maxBrightness = 255
stereo_config.postProcessing.decimationFilter.decimationFactor = 1
stereo_config.postProcessing.speckleFilter.enable = True
stereo_config.postProcessing.spatialFilter.enable = True
stereo_config.postProcessing.temporalFilter.enable = False


window_size = 5
P1 = 8 * 3 * window_size ** 2
P2 = 32 * 3 * window_size ** 2

stereo_config.costAggregation.horizontalPenaltyCostP1 = P1
stereo_config.costAggregation.horizontalPenaltyCostP2 = P2
stereo_config.costAggregation.verticalPenaltyCostP1 = P1
stereo_config.costAggregation.verticalPenaltyCostP2 = P2


xin_left_vpp = pipeline_OAK.createXLinkIn()
xin_left_vpp.setStreamName('left_vpp')
xin_right_vpp = pipeline_OAK.createXLinkIn()
xin_right_vpp.setStreamName('right_vpp')
xin_stereo_config = pipeline_OAK.createXLinkIn()
xin_stereo_config.setStreamName('stereo_config')

xout_vpp_disparity = pipeline_OAK.createXLinkOut()
xout_vpp_disparity.setStreamName('vpp_disparity')

xin_left_vpp.out.link(vpp_sgm_node.left)
xin_right_vpp.out.link(vpp_sgm_node.right)
xin_stereo_config.out.link(vpp_sgm_node.inputConfig)
vpp_sgm_node.disparity.link(xout_vpp_disparity.input)



#Cached things
sample_img = None


# Initialize pipeline
pipeline = rs.pipeline()

# Configure streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream

# Start streaming
profile = pipeline.start(config)

# Align object: this will align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

# Get the intrinsics of the RGB camera
color_stream = profile.get_stream(rs.stream.color)  # Fetch color stream profile
intrinsics = color_stream.as_video_stream_profile().get_intrinsics()  # RGB camera intrinsics

# Extract focal length and other intrinsics
fx = intrinsics.fx  # Focal length in x direction
fy = intrinsics.fy  # Focal length in y direction
ppx = intrinsics.ppx  # Principal point x
ppy = intrinsics.ppy  # Principal point y
coeffs = intrinsics.coeffs  # Distortion coefficients

depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, L515_VISUAL_PRESET_DICT[args.l515_preset]) # 5 is short range, 3 is low ambient light
depth_scale = depth_sensor.get_depth_scale()

# Create a named window
cv2.namedWindow('Aligned RGB and Depth', cv2.WINDOW_NORMAL)

# Add a cursor to modify args.sampling_perc
cv2.createTrackbar('Sampling Percentage', 'Aligned RGB and Depth', int(args.sampling_perc * 100), 100, lambda x: setattr(args, 'sampling_perc', x / 1000))
cv2.createTrackbar('Virtual Baseline', 'Aligned RGB and Depth', int(args.virtual_baseline * 100), 100, lambda x: setattr(args, 'virtual_baseline', x / 100))

try:
    with dai.Device(pipeline_OAK) as device:
        
        in_queue_left_vpp = device.getInputQueue("left_vpp", maxSize=1, blocking=True)
        in_queue_right_vpp = device.getInputQueue("right_vpp", maxSize=1, blocking=True)
        in_queue_stereo_config = device.getInputQueue("stereo_config", maxSize=1, blocking=True)

        configMessage = dai.StereoDepthConfig()
        configMessage.set(stereo_config)
        in_queue_stereo_config.send(configMessage)

        #blocking=True -> Wait until vpp disparity is ready
        out_queue_vpp_disparity = device.getOutputQueue(name="vpp_disparity", maxSize=1, blocking=False) 

        while True:
            # Wait for frames from the camera
            frames = pipeline.wait_for_frames()

            # Align depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            # Convert frames to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float32) * depth_scale
            color_image = np.asanyarray(color_frame.get_data())

            # H and W cropping
            depth_image = depth_image[:H, :W]
            color_image = color_image[:H, :W]

            # rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            rgb_image = color_image

            # Remove blobs
            if args.use_blob_remover:
                _depth_image = depth_image.copy().astype(np.uint8)
                cv2.filterSpeckles(_depth_image,0,50,1)
                _depth_image = _depth_image.astype(np.float32)
                _depth_image[_depth_image!=0] = depth_image[_depth_image!=0]
                depth_image = _depth_image

            # depth_pred, hints, im2_vpp, im3_vpp = predict(rgb_image, depth_image, fx, args.virtual_baseline, args.sampling_perc)
            hints, sample_img = sample_hints_cache(depth_image, args.sampling_perc, sample_img)
            hints[hints>0] = (fx*args.virtual_baseline)/hints[hints>0] # Convert depth to disparity

            left_pad_size = 100
            left_pad_size = min(left_pad_size, 200)        
            left_pad_size = left_pad_size + (32-(left_pad_size+W) % 32)
            left_pad_size = left_pad_size if args.leftpadding else 0
            _prepad = [left_pad_size,0,0,0]
            
            # Pad using numpy or cv2
            rgb_image = np.pad(rgb_image, ((0,0), (left_pad_size, 0), (0,0)), mode='constant', constant_values=0)
            depth_image = np.pad(depth_image, ((0,0), (left_pad_size, 0)), mode='constant', constant_values=0)
            hints = np.pad(hints, ((0,0), (left_pad_size, 0)), mode='constant', constant_values=0)

            h,w,c = rgb_image.shape
            left_black = np.full((h,w,c), 128, dtype=np.uint8)
            right_black = np.full((h,w,c), 128, dtype=np.uint8)            
            extrapolated_hints = hints
            mask_occ = occlusion_heuristic(extrapolated_hints)[1] if args.maskocc else None

            im2_vpp, im3_vpp = vpp(rgb_image, left_black, right_black,
                                            extrapolated_hints, blending=args.blending, wsize=args.wsize,
                                            c_occ=args.cblending, g_occ=mask_occ, useFilling = args.filling, useContext=args.context,
                                            fillingThreshold=args.th_adpt, o_xy=args.o_xy, o_i=args.o_i,
                                            left2right=True, method='rnd', uniform_color=args.uniform_color, interpolate=args.interpolate )  


            start_time = time.time()
            left_vpp_data = dai.ImgFrame()
            left_vpp_data.setData(cv2.cvtColor(im2_vpp, cv2.COLOR_BGR2GRAY).flatten())
            # left_vpp_data.setTimestamp(left_vanilla_rectified_data.getTimestamp())
            left_vpp_data.setInstanceNum(dai.CameraBoardSocket.LEFT)
            left_vpp_data.setType(dai.ImgFrame.Type.RAW8)
            left_vpp_data.setWidth(W)
            left_vpp_data.setHeight(H)
            in_queue_left_vpp.send(left_vpp_data)

            right_vpp_data = dai.ImgFrame()
            right_vpp_data.setData(cv2.cvtColor(im3_vpp, cv2.COLOR_BGR2GRAY).flatten())
            # right_vpp_data.setTimestamp(right_vanilla_rectified_data.getTimestamp())
            right_vpp_data.setInstanceNum(dai.CameraBoardSocket.RIGHT)
            right_vpp_data.setType(dai.ImgFrame.Type.RAW8)
            right_vpp_data.setWidth(W)
            right_vpp_data.setHeight(H)
            in_queue_right_vpp.send(right_vpp_data)

            #Drop current frame if OAK does not respond
            start_time = time.time()
            drop_frame = False
            while not out_queue_vpp_disparity.has():
                time.sleep(0.001)
                if time.time() - start_time > TIMEOUT_SGM:
                    drop_frame = True
                    print("OAK VPP SGM not responding... skip frame.")
                    break    

            vpp_disparity_data = out_queue_vpp_disparity.tryGet()
            if drop_frame or vpp_disparity_data is None:
                continue

            depth_pred = vpp_disparity_data.getCvFrame() / OAK_DISPARITY_DIV

            end_time = time.time()
            print(f"SGM VPP time: {(end_time-start_time)}")        

            #Remove left border prepadding
            ht, wd = depth_pred.shape[-2:]
            c = [_prepad[2], ht-_prepad[3], _prepad[0], wd-_prepad[1]]

            rgb_image = rgb_image[c[0]:c[1], c[2]:c[3], ...]
            depth_image = depth_image[c[0]:c[1], c[2]:c[3]]
            hints = hints[c[0]:c[1], c[2]:c[3]]

            im2_vpp = im2_vpp[c[0]:c[1], c[2]:c[3], ...]
            im3_vpp = im3_vpp[c[0]:c[1], c[2]:c[3], ...]
            depth_pred = depth_pred[c[0]:c[1], c[2]:c[3]]

            depth_pred[depth_pred>0] = (fx*args.virtual_baseline)/depth_pred[depth_pred>0] # Convert disparity to depth
            depth_pred[depth_pred>args.max_z] = args.max_z

            hints[hints>0] = (fx*args.virtual_baseline)/hints[hints>0] # Convert disparity to depth
            hints[hints>args.max_z] = args.max_z

            #----------------------------------------

            # dilate hints
            # hints = cv2.dilate(hints, np.ones((3,3), np.uint8), iterations=1)

            # Apply colormap on depth image (for visualization)
            depth_colormap = cv2.applyColorMap((depth_image/args.max_z*255).astype(np.uint8), cv2.COLORMAP_JET)
            depth_pred_colormap = cv2.applyColorMap((depth_pred/args.max_z*255).astype(np.uint8), cv2.COLORMAP_JET)
            hints_colormap = cv2.applyColorMap((hints/args.max_z*255).astype(np.uint8), cv2.COLORMAP_JET)

            # Resize depth image to match color image size
            depth_colormap_resized = cv2.resize(depth_colormap, (color_image.shape[1], color_image.shape[0]))
            depth_pred_colormap_resized = cv2.resize(depth_pred_colormap, (color_image.shape[1], color_image.shape[0]))
            hints_colormap_resized = cv2.resize(hints_colormap, (color_image.shape[1], color_image.shape[0]))

            # Alpha blend color image and depth colormap
            # blended_image = cv2.addWeighted(color_image, 0.5, hints_colormap_resized, 0.5, 0)
            # blended_image[hints_colormap_resized == 0] = color_image[hints_colormap_resized == 0]
            color_image[hints > 0] = hints_colormap_resized[hints > 0]

            # Show images
            frame_up = np.hstack((color_image, depth_pred_colormap_resized))
            frame_vpp = np.hstack((im2_vpp, im3_vpp))

            # cv2.imwrite('tmp/frame.png', frame)
            cv2.imshow('Aligned RGB and Depth', np.vstack((frame_up, frame_vpp)))

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    # Stop streaming
    pipeline.stop()

# Release resources
cv2.destroyAllWindows()


