
import numpy as np
import random
import cv2
import argparse
import time
import json
import os
import threading
from datetime import timedelta

import mid70grabber
import depthai as dai  

from mycoda import MID70FrameWrapper, SlidingWindowDevice
from losses import sample_hints_np
from vpp_standalone import vpp
from filter import occlusion_heuristic
from utils import get_view



TIMEOUT_SGM = 0.5
MID70_FPS = 10
OAK_FPS = 30
SYNC_TH_MS = 100/1000.0
MAX_BUFFER_SIZE = 3

MID70_BROADCAST_CODE = ""

CALIB_PATH = "mid70_oak_calib"
RT_OAK_MID70 = np.loadtxt(f"{CALIB_PATH}/RT_RGB_LIVOX.txt")
# K_OAK = np.loadtxt(f"{CALIB_PATH}/K_OAK.txt")#TODO: extract correct K from calibration
# OAK_BASELINE = np.loadtxt(f"{CALIB_PATH}/BASELINE_OAK.txt")

MID70_CALIB_FILE = "mid70_calib.json"
OAK_CALIB_FILE = "oak_calib.json"

with open(os.path.join(CALIB_PATH, OAK_CALIB_FILE), "r") as f:
    oak_calib_data = json.load(f)

with open(os.path.join(CALIB_PATH, MID70_CALIB_FILE), "r") as f:
    mid70_calib_data = json.load(f)

OAK_BASELINE = oak_calib_data["baseline"]

OAK_RGB_RESOLUTION = dai.ColorCameraProperties.SensorResolution.THE_1080_P

OAK_RGB_RESOLUTION_DICT = {
    dai.ColorCameraProperties.SensorResolution.THE_720_P: (720, 1280),
    dai.ColorCameraProperties.SensorResolution.THE_1080_P: (1080, 1920)
}

K_OAK = np.array(oak_calib_data[f'K_rgb_{OAK_RGB_RESOLUTION_DICT[OAK_RGB_RESOLUTION][0]}x{OAK_RGB_RESOLUTION_DICT[OAK_RGB_RESOLUTION][1]}'])
D_OAK = np.array(oak_calib_data[f'D_rgb'])

OAK_SUBPIXEL = True
OAK_LRC = True
OAK_EXTENDED_DISPARITY = False
OAK_FRACTIONAL_BITS = 3
OAK_DISPARITY_DIV = 2 ** OAK_FRACTIONAL_BITS if OAK_SUBPIXEL else 1

H_RGB = OAK_RGB_RESOLUTION_DICT[OAK_RGB_RESOLUTION][0]
W_RGB = OAK_RGB_RESOLUTION_DICT[OAK_RGB_RESOLUTION][1]

H_SGM = H_RGB // 2
W_SGM = W_RGB // 2

FOCAL_LENGTH = K_OAK[0, 0] * W_SGM / W_RGB

DISP_MAX = 96

#Thread-safe sliding window to store MID70 frames
mid70_sliding_window = SlidingWindowDevice(MAX_BUFFER_SIZE)
# MID70 thread stop condition
stop_condition_slave = False

# MID70 thread that load frames asynchronously
def slave(thread_id, args): 
    global mid70_sliding_window
    global stop_condition_slave
   
    def mylog(x, debug=False):
        if not debug or args.verbose:
            print(f"SLAVE ({thread_id}): {x}")

    started = mid70grabber.start([MID70_BROADCAST_CODE], round(1000.0/MID70_FPS))
    if not started:
        print("Error starting the stream")
        exit()

    # Wait until a device is connected
    devices = mid70grabber.get_devices()

    while len(devices) == 0:
        print("Waiting for devices...")
        time.sleep(0.1)
        devices = mid70grabber.get_devices()
     
    mylog(f"Lidar ready")

    #Assume that a PTP server is running 
    
    #Fill sliding window until stop condition
    while not stop_condition_slave: 
        # Grab a frame
        frame, timestamp_start, timestamp_end = mid70grabber.get_frame(devices[0]["handle"])

        if len(frame) > 0:
            #Convert Nx4 points to an image
            #D_OAK is bugged
            depth_image, reflectivity_image = get_view(frame, RT_OAK_MID70, K_OAK, np.zeros(5), OAK_RGB_RESOLUTION_DICT[OAK_RGB_RESOLUTION][0], OAK_RGB_RESOLUTION_DICT[OAK_RGB_RESOLUTION][1])

            if np.max(depth_image) > 0:
                frame_wrapper = MID70FrameWrapper(depth_image, reflectivity_image, start_scan_timestamp=timestamp_start, end_scan_timestamp=timestamp_end)
                mid70_sliding_window.add_element(frame_wrapper)
                #mylog(f"({mid70_sliding_window.is_empty()}) Read frame at ts: {timestamp_start/1e9}")
     
    # Stop the livox stream
    mid70grabber.stop()

def main(args):
    global stop_condition_slave
    global mid70_sliding_window

    def mylog(x, debug=False):  
        if not debug or args.verbose: 
            print(f"MASTER: {x}")

    delta_t_oak_mid70_depth = 0.0

    slave_thread = threading.Thread(target=slave, args=(1,args))
    slave_thread.start()

    #Initialize OAK-D pipeline as SGM processor
    pipeline_OAK = dai.Pipeline()

    # Create camera node istance
    # Create one SGM node instances to rectified vpp stereo pair.
    rgb_node = pipeline_OAK.createColorCamera() 
    rgb_node.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    rgb_node.setResolution(OAK_RGB_RESOLUTION)
    rgb_node.setFps(OAK_FPS) 
    rgb_node.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    vpp_sgm_node = pipeline_OAK.createStereoDepth()
    vpp_sgm_node.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    vpp_sgm_node.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    vpp_sgm_node.setLeftRightCheck(OAK_LRC)
    vpp_sgm_node.setExtendedDisparity(OAK_EXTENDED_DISPARITY)
    vpp_sgm_node.setSubpixel(OAK_SUBPIXEL)
    vpp_sgm_node.setSubpixelFractionalBits(OAK_FRACTIONAL_BITS)
    vpp_sgm_node.setDepthAlign(dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
    vpp_sgm_node.setInputResolution(W_RGB, H_RGB)
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

    xout_rgb = pipeline_OAK.createXLinkOut()
    xout_rgb.setStreamName('rgb')
    xout_vpp_disparity = pipeline_OAK.createXLinkOut()
    xout_vpp_disparity.setStreamName('vpp_disparity')

    rgb_node.video.link(xout_rgb.input)
    xin_left_vpp.out.link(vpp_sgm_node.left)
    xin_right_vpp.out.link(vpp_sgm_node.right)
    xin_stereo_config.out.link(vpp_sgm_node.inputConfig)
    vpp_sgm_node.disparity.link(xout_vpp_disparity.input)

    # Create a named window
    cv2.namedWindow('Aligned RGB and Depth', cv2.WINDOW_NORMAL)

    # Add a cursor to modify args.sampling_perc
    cv2.createTrackbar('Sampling Percentage', 'Aligned RGB and Depth', int(args.sampling_perc * 1000), 100, lambda x: setattr(args, 'sampling_perc', x / 1000))
    cv2.createTrackbar('Virtual Baseline', 'Aligned RGB and Depth', int(args.virtual_baseline * 100), 100, lambda x: setattr(args, 'virtual_baseline', x / 100))

    try:
        with dai.Device(pipeline_OAK) as device:
            #Sync OAK clock with host clock
            #Simple hack here: assume diff constant
            # device.setTimesync(True)# Already the default config
            oak_clock:timedelta = dai.Clock.now()
            host_clock = time.time()
            diff = host_clock-oak_clock.total_seconds()
            
            in_queue_left_vpp = device.getInputQueue("left_vpp", maxSize=1, blocking=True)
            in_queue_right_vpp = device.getInputQueue("right_vpp", maxSize=1, blocking=True)
            in_queue_stereo_config = device.getInputQueue("stereo_config", maxSize=1, blocking=True)

            configMessage = dai.StereoDepthConfig()
            configMessage.set(stereo_config)
            in_queue_stereo_config.send(configMessage)

            #blocking=True -> Wait until vpp disparity is ready
            out_queue_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False) 
            out_queue_vpp_disparity = device.getOutputQueue(name="vpp_disparity", maxSize=1, blocking=False) 

            counter = 0
            while True:
                mylog(f"Iteration {counter} ({delta_t_oak_mid70_depth})", True)
                counter += 1

                rgb_data = out_queue_rgb.get()   
                timestamp_OAK = rgb_data.getTimestamp(dai.CameraExposureOffset.END).total_seconds()+diff
                mylog(f"Timestamp OAK: {timestamp_OAK}", True)

                #if the MID70 sliding window is too new then drop some OAK frames
                if not mid70_sliding_window.is_window_too_new("depth", timestamp_OAK, SYNC_TH_MS):                        
                    
                    #if the OAK frame is too new then wait until the window is fresh
                    while mid70_sliding_window.is_window_too_old("depth", timestamp_OAK, SYNC_TH_MS):
                        time.sleep(0.001)

                    # Get the nearest MID70 frame based on OAK frame timestamp
                    nearest_mid70_frame:MID70FrameWrapper = mid70_sliding_window.nearest_frame("depth", timestamp_OAK)

                    if nearest_mid70_frame is not None:
                        
                        #OK now we have raw depth points in the MID70 RF -> reproject -> filter
                        depth_image = nearest_mid70_frame.get_frame("depth") * nearest_mid70_frame.get_frame("depth_scale")
                        timestamp_mid70_depth = nearest_mid70_frame.get_timestamp("depth")
                        
                        #Sync delta: assuming a global clock between MID70 and OAK, observe the time difference between them
                        delta_t_oak_mid70_depth = abs(timestamp_OAK-timestamp_mid70_depth)
                        mylog(f"Timestamp MID70: {timestamp_mid70_depth}", True)
                        mylog(f"Delta Time: {(delta_t_oak_mid70_depth)}", True)

                        #Keep frames only if meet time requirements
                        if delta_t_oak_mid70_depth < SYNC_TH_MS:         
                            color_image = rgb_data.getCvFrame()

                            # H and W cropping
                            # depth_image = depth_image[:H, :W]
                            # color_image = color_image[:H, :W]

                            depth_image = cv2.resize(depth_image, (W_SGM, H_SGM), interpolation=cv2.INTER_NEAREST)
                            color_image = cv2.resize(color_image, (W_SGM, H_SGM), interpolation=cv2.INTER_LINEAR)

                            # rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                            rgb_image = color_image

                            # Remove blobs
                            if args.use_blob_remover:
                                _depth_image = depth_image.copy().astype(np.uint8)
                                cv2.filterSpeckles(_depth_image,0,50,1)
                                _depth_image = _depth_image.astype(np.float32)
                                _depth_image[_depth_image!=0] = depth_image[_depth_image!=0]
                                depth_image = _depth_image

                            # depth_pred, hints, im2_vpp, im3_vpp = predict(rgb_image, depth_image, focal_length, args.virtual_baseline, args.sampling_perc)
                            #hints = sample_hints_np(depth_image, depth_image>0, args.sampling_perc)[0]
                            hints = depth_image
                            hints[hints>0] = (FOCAL_LENGTH*args.virtual_baseline)/hints[hints>0] # Convert depth to disparity

                            left_pad_size = 100
                            left_pad_size = min(left_pad_size, 200)        
                            left_pad_size = left_pad_size + (32-(left_pad_size+W_RGB) % 32)
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
                            left_vpp_data.setWidth(W_SGM)
                            left_vpp_data.setHeight(H_SGM)
                            in_queue_left_vpp.send(left_vpp_data)

                            right_vpp_data = dai.ImgFrame()
                            right_vpp_data.setData(cv2.cvtColor(im3_vpp, cv2.COLOR_BGR2GRAY).flatten())
                            # right_vpp_data.setTimestamp(right_vanilla_rectified_data.getTimestamp())
                            right_vpp_data.setInstanceNum(dai.CameraBoardSocket.RIGHT)
                            right_vpp_data.setType(dai.ImgFrame.Type.RAW8)
                            right_vpp_data.setWidth(W_SGM)
                            right_vpp_data.setHeight(H_SGM)
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

                            depth_pred[depth_pred>0] = (FOCAL_LENGTH*args.virtual_baseline)/depth_pred[depth_pred>0] # Convert disparity to depth
                            depth_pred[depth_pred>args.max_z] = args.max_z

                            hints[hints>0] = (FOCAL_LENGTH*args.virtual_baseline)/hints[hints>0] # Convert disparity to depth
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
                                mylog("Quitting...")
                                break
    except KeyboardInterrupt:
        mylog(f"CRTL-C received")
    finally:
        # Stop streaming
        mylog(f"Releasing resources and closing.")
        cv2.destroyAllWindows()
        stop_condition_slave = True    
        slave_thread.join() 

if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description='MID70 VPPDC Demo with OAK-D\'s SGM')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--livox_broadcast_code", type=str, default=None, help="Livox broadcast code")
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

    args.use_blob_remover = False

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.livox_broadcast_code is not None:
        MID70_BROADCAST_CODE = args.livox_broadcast_code

    print(f"Broadcast code: {MID70_BROADCAST_CODE}")
    print(f"Verbose: {args.verbose}")

    main(args)
