import time
import mid70grabber
import numpy as np
import depthai as dai  
import cv2
import threading
from datetime import timedelta
from mycoda import SlidingWindowDevice, MID70FrameWrapper
import argparse
import os
import json

from vpp_standalone import vpp
from filter import occlusion_heuristic
from filter_depth import filter_heuristic_depth
from utils import add_title_description, get_view

MID70_FPS = 10
OAK_FPS = 30
SYNC_TH_MS = 50/1000.0
TIMEOUT_SGM = 0.5

MID70_BROADCAST_CODE = ""

MAX_BUFFER_SIZE = 3
CALIB_PATH = "mid70_oak_calib"
RT_OAK_MID70 = np.loadtxt(f"{CALIB_PATH}/RT_LEFTR_LIVOX.txt")
# K_OAK = np.loadtxt(f"{CALIB_PATH}/K_OAK.txt")
# OAK_BASELINE = np.loadtxt(f"{CALIB_PATH}/BASELINE_OAK.txt")

MID70_CALIB_FILE = "mid70_calib.json"
OAK_CALIB_FILE = "oak_calib.json"

with open(os.path.join(CALIB_PATH, OAK_CALIB_FILE), "r") as f:
    oak_calib_data = json.load(f)

with open(os.path.join(CALIB_PATH, MID70_CALIB_FILE), "r") as f:
    mid70_calib_data = json.load(f)

OAK_BASELINE = oak_calib_data["baseline"]
OAK_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P

OAK_RESOLUTION_DICT = {
    dai.MonoCameraProperties.SensorResolution.THE_400_P: (400, 640),
    dai.MonoCameraProperties.SensorResolution.THE_480_P: (480, 640),
    dai.MonoCameraProperties.SensorResolution.THE_720_P: (720, 1280),
    dai.MonoCameraProperties.SensorResolution.THE_800_P: (800, 1280),
}

K_OAK = np.array(oak_calib_data[f'K_left_{OAK_RESOLUTION_DICT[OAK_RESOLUTION][0]}x{OAK_RESOLUTION_DICT[OAK_RESOLUTION][1]}'])

OAK_SUBPIXEL = True
OAK_LRC = True
OAK_EXTENDED_DISPARITY = False
OAK_FRACTIONAL_BITS = 3
OAK_DISPARITY_DIV = 2 ** OAK_FRACTIONAL_BITS if OAK_SUBPIXEL else 1

GUI_WINDOW_NAME = "VPP-DEMO-SGM"
GUI_WSIZE = 3
GUI_BLENDING = 0.5
GUI_UNIFORM_COLOR = True

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
        start_time = time.time()
        frame, timestamp_start, timestamp_end = mid70grabber.get_frame(devices[0]["handle"])
        end_time = time.time()
        mylog(f"Grab time: {(end_time-start_time)} ({timestamp_end/1e9})", True)

        if len(frame) > 0:
            #Convert Nx4 points to an image
            start_time = time.time()
            depth_image, reflectivity_image = get_view(frame, RT_OAK_MID70, K_OAK, np.zeros(5), OAK_RESOLUTION_DICT[OAK_RESOLUTION][0], OAK_RESOLUTION_DICT[OAK_RESOLUTION][1])
            end_time = time.time()
            mylog(f"Conversion time: {(end_time-start_time)}", True)

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
   
    # Use OAK camera as master device
    pipeline_OAK = dai.Pipeline()

    # Create camera node istances
    # Create two SGM node instances
    # The former is attached to cameras, the latter to rectified vpp stereo pair.
    left_node = pipeline_OAK.createMonoCamera()  # monoLeft = pipeline.create(dai.node.MonoCamera)
    right_node = pipeline_OAK.createMonoCamera() # monoRight = pipeline.create(dai.node.MonoCamera)
    vanilla_sgm_node = pipeline_OAK.createStereoDepth()
    vpp_sgm_node = pipeline_OAK.createStereoDepth()

    # Configure cameras: set fps lower than MID70 to better accumulate depth frames
    left_node.setResolution(OAK_RESOLUTION)
    left_node.setFps(OAK_FPS) 
    right_node.setResolution(OAK_RESOLUTION)
    right_node.setFps(OAK_FPS)  
    left_node.setCamera("left")
    right_node.setCamera("right")

    #Configure SGM nodes: set alignment and other stuff
    vanilla_sgm_node.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    vanilla_sgm_node.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
    vanilla_sgm_node.setLeftRightCheck(OAK_LRC)
    vanilla_sgm_node.setExtendedDisparity(OAK_EXTENDED_DISPARITY)
    vanilla_sgm_node.setSubpixel(OAK_SUBPIXEL)
    vanilla_sgm_node.setSubpixelFractionalBits(OAK_FRACTIONAL_BITS)
    vanilla_sgm_node.setDepthAlign(dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
    vanilla_sgm_node.setRuntimeModeSwitch(True)

    vpp_sgm_node.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    vpp_sgm_node.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
    vpp_sgm_node.setLeftRightCheck(OAK_LRC)
    vpp_sgm_node.setExtendedDisparity(OAK_EXTENDED_DISPARITY)
    vpp_sgm_node.setSubpixel(OAK_SUBPIXEL)
    vpp_sgm_node.setSubpixelFractionalBits(OAK_FRACTIONAL_BITS)
    vpp_sgm_node.setDepthAlign(dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
    vpp_sgm_node.setInputResolution(OAK_RESOLUTION_DICT[OAK_RESOLUTION][1], OAK_RESOLUTION_DICT[OAK_RESOLUTION][0])
    vpp_sgm_node.setRectification(False)# Images are already rectified
    vpp_sgm_node.setRuntimeModeSwitch(True)

    # Create in/out channels
    # Required input channels: left_rectified_vpp, right_rectified_vpp
    # Required output channels: vanilla_disparity, vpp_disparity, left_vanilla_rectified, right_vanilla_rectified

    xin_left_vpp = pipeline_OAK.createXLinkIn()
    xin_left_vpp.setStreamName('left_vpp')
    xin_right_vpp = pipeline_OAK.createXLinkIn()
    xin_right_vpp.setStreamName('right_vpp')

    xout_vanilla_disparity = pipeline_OAK.createXLinkOut()
    xout_vanilla_disparity.setStreamName('vanilla_disparity')
    xout_vpp_disparity = pipeline_OAK.createXLinkOut()
    xout_vpp_disparity.setStreamName('vpp_disparity')
    xout_left_vanilla_rectified = pipeline_OAK.createXLinkOut()
    xout_left_vanilla_rectified.setStreamName('left_vanilla_rectified')
    xout_right_vanilla_rectified = pipeline_OAK.createXLinkOut()
    xout_right_vanilla_rectified.setStreamName('right_vanilla_rectified')

    # Link channels and nodes
    left_node.out.link(vanilla_sgm_node.left)
    right_node.out.link(vanilla_sgm_node.right)
    vanilla_sgm_node.rectifiedLeft.link(xout_left_vanilla_rectified.input)
    vanilla_sgm_node.rectifiedRight.link(xout_right_vanilla_rectified.input)
    vanilla_sgm_node.disparity.link(xout_vanilla_disparity.input) 

    xin_left_vpp.out.link(vpp_sgm_node.left)
    xin_right_vpp.out.link(vpp_sgm_node.right)
    vpp_sgm_node.disparity.link(xout_vpp_disparity.input)
    
    
    with dai.Device(pipeline_OAK) as device:
        #Sync OAK clock with host clock
        #Simple hack here: assume diff constant
        # device.setTimesync(True)# Already the default config
        oak_clock:timedelta = dai.Clock.now()
        host_clock = time.time()
        diff = host_clock-oak_clock.total_seconds()

        #Create in/out queues
        in_queue_left_vpp = device.getInputQueue("left_vpp", maxSize=1, blocking=True)
        in_queue_right_vpp = device.getInputQueue("right_vpp", maxSize=1, blocking=True)

        out_queue_left_vanilla_rectified = device.getOutputQueue(name="left_vanilla_rectified", maxSize=1, blocking=False) 
        out_queue_right_vanilla_rectified = device.getOutputQueue(name="right_vanilla_rectified", maxSize=1, blocking=False) 
        out_queue_vanilla_disparity = device.getOutputQueue(name="vanilla_disparity", maxSize=1, blocking=False) 

        #blocking=True -> Wait until vpp disparity is ready
        out_queue_vpp_disparity = device.getOutputQueue(name="vpp_disparity", maxSize=1, blocking=False) 

        mylog(f"OAK-D ready")
        
        # Function to update the display based on slider values
        def update_display(*args):
            global GUI_WSIZE
            global GUI_BLENDING
            global GUI_UNIFORM_COLOR

            GUI_WSIZE = cv2.getTrackbarPos('Patch Size', GUI_WINDOW_NAME)
            GUI_UNIFORM_COLOR = bool(cv2.getTrackbarPos('Uniform Patch', GUI_WINDOW_NAME))
            GUI_BLENDING = cv2.getTrackbarPos('Alpha Blending', GUI_WINDOW_NAME) / 100.0

        cv2.namedWindow(GUI_WINDOW_NAME)

        # Create trackbars (sliders)
        cv2.createTrackbar('Patch Size', GUI_WINDOW_NAME, 3, 7, update_display)
        cv2.createTrackbar('Uniform Patch', GUI_WINDOW_NAME, 1, 1, update_display)
        cv2.createTrackbar('Alpha Blending', GUI_WINDOW_NAME, 50, 100, update_display)
        cv2.createTrackbar('Hints Density', GUI_WINDOW_NAME, 5, 100, update_display)

        try:
            counter = 0
            #Until exception (Keyboard interrupt) or q pressed
            while True:
                mylog(f"Iteration {counter} ({delta_t_oak_mid70_depth})", True)
                counter += 1

                #Demo pipeline:
                #1) Get left/right rectified images and vanilla prediction
                #2) Search for the nearest MID70 frame then compute hints and GT
                #3) Use VPP to generate enhanced stereo pair
                #4) Put VPP pair inside the second SGM node and wait for prediction
                #5) Show to the user all qualitatives + metrics
                
                vanilla_disparity_data = out_queue_vanilla_disparity.get()       
                left_vanilla_rectified_data = out_queue_left_vanilla_rectified.get() 
                right_vanilla_rectified_data = out_queue_right_vanilla_rectified.get() 

                #timestamp_OAK = vanilla_disparity_data.getTimestamp().total_seconds()+diff
                timestamp_OAK = vanilla_disparity_data.getTimestamp(dai.CameraExposureOffset.END).total_seconds()+diff
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
                        mid70_depth = nearest_mid70_frame.get_frame("depth") * nearest_mid70_frame.get_frame("depth_scale")
                        timestamp_mid70_depth = nearest_mid70_frame.get_timestamp("depth")
                        
                        #Sync delta: assuming a global clock between MID70 and OAK, observe the time difference between them
                        delta_t_oak_mid70_depth = abs(timestamp_OAK-timestamp_mid70_depth)
                        mylog(f"Timestamp MID70: {timestamp_mid70_depth}", True)
                        mylog(f"Delta Time: {(delta_t_oak_mid70_depth)}", True)

                        #Keep frames only if meet time requirements
                        if delta_t_oak_mid70_depth < SYNC_TH_MS:                        

                            oak_left_rectified = left_vanilla_rectified_data.getCvFrame()
                            oak_left_rectified = cv2.cvtColor(oak_left_rectified, cv2.COLOR_GRAY2BGR)
                            oak_right_rectified = right_vanilla_rectified_data.getCvFrame()
                            oak_right_rectified = cv2.cvtColor(oak_right_rectified, cv2.COLOR_GRAY2BGR)
                            oak_vanilla_disparity = vanilla_disparity_data.getCvFrame() / OAK_DISPARITY_DIV
                            H_oak, W_oak = oak_vanilla_disparity.shape[:2]

                            start_time = time.time()

                            filtered_mid70_depth = filter_heuristic_depth(mid70_depth)[0]

                            #Depth to disparity map
                            raw = filtered_mid70_depth.copy()
                            raw[raw>0] = (K_OAK[0,0] * OAK_BASELINE) / raw[raw>0]
                            hints = raw

                            occ_mask = occlusion_heuristic(hints)[1].astype(np.float32)
                            end_time = time.time()

                            mylog(f"Preprocessing time: {(end_time-start_time)}", True)

                            start_time = time.time()
                            left_vpp, right_vpp = vpp(oak_left_rectified, oak_right_rectified, hints, wsize=GUI_WSIZE, blending=GUI_BLENDING, uniform_color=GUI_UNIFORM_COLOR, g_occ=occ_mask)
                            end_time = time.time()
                            mylog(f"VPP time: {(end_time-start_time)}", True)

                            start_time = time.time()
                            left_vpp_data = dai.ImgFrame()
                            left_vpp_data.setData(cv2.cvtColor(left_vpp, cv2.COLOR_BGR2GRAY).flatten())
                            left_vpp_data.setTimestamp(left_vanilla_rectified_data.getTimestamp())
                            left_vpp_data.setInstanceNum(dai.CameraBoardSocket.CAM_B)
                            left_vpp_data.setType(dai.ImgFrame.Type.RAW8)
                            left_vpp_data.setWidth(W_oak)
                            left_vpp_data.setHeight(H_oak)
                            in_queue_left_vpp.send(left_vpp_data)

                            right_vpp_data = dai.ImgFrame()
                            right_vpp_data.setData(cv2.cvtColor(right_vpp, cv2.COLOR_BGR2GRAY).flatten())
                            right_vpp_data.setTimestamp(right_vanilla_rectified_data.getTimestamp())
                            right_vpp_data.setInstanceNum(dai.CameraBoardSocket.CAM_C)
                            right_vpp_data.setType(dai.ImgFrame.Type.RAW8)
                            right_vpp_data.setWidth(W_oak)
                            right_vpp_data.setHeight(H_oak)
                            in_queue_right_vpp.send(right_vpp_data)

                            #Drop current frame if OAK does not respond
                            start_time = time.time()
                            drop_frame = False
                            while not out_queue_vpp_disparity.has():
                                time.sleep(0.001)
                                if time.time() - start_time > TIMEOUT_SGM:
                                    drop_frame = True
                                    mylog("OAK VPP SGM not responding... skip frame.")
                                    break
                            
                            vpp_disparity_data = out_queue_vpp_disparity.tryGet()
                            if drop_frame or vpp_disparity_data is None:
                                continue

                            oak_vpp_disparity = vpp_disparity_data.getCvFrame() / OAK_DISPARITY_DIV

                            end_time = time.time()
                            mylog(f"SGM VPP time: {(end_time-start_time)}", True)


                            # OAK Left   | OAK Right  | OAK Depth
                            # -----------------------------------
                            # VPP Left   | VPP Right  | VPP Depth

                            # Press 'q' to quit

                            #Create stacked frame and add info to the images

                            vanilla_left_img = oak_left_rectified
                            vanilla_right_img = oak_right_rectified
                            vanilla_disparity_img = oak_vanilla_disparity
                            
                            vpp_left_img = left_vpp
                            vpp_right_img = right_vpp
                            vpp_disparity_img = oak_vpp_disparity

                            #Apply colormaps
                            vanilla_disparity_img = cv2.applyColorMap((255.0 * np.clip(vanilla_disparity_img, 0, DISP_MAX) / DISP_MAX).astype(np.uint8), cv2.COLORMAP_MAGMA)
                            vpp_disparity_img = cv2.applyColorMap((255.0 * np.clip(vpp_disparity_img, 0, DISP_MAX) / DISP_MAX).astype(np.uint8), cv2.COLORMAP_MAGMA)

                            #Add text
                            start_time = time.time()
                            vanilla_left_img = add_title_description(vanilla_left_img, "Vanilla Left", " ")
                            vanilla_right_img = add_title_description(vanilla_right_img, "Vanilla Right", " ")
                            # vanilla_disparity_img = add_title_description(vanilla_disparity_img, "Vanilla Disparity", f"MAE: {vanilla_metrics['avgerr']}, BAD3: {vanilla_metrics['bad 3.0']}")
                            vanilla_disparity_img = add_title_description(vanilla_disparity_img, "Vanilla Disparity", " ")

                            vpp_left_img = add_title_description(vpp_left_img, "VPP Left", " ")
                            vpp_right_img = add_title_description(vpp_right_img, "VPP Right", " ")
                            # vpp_disparity_img = add_title_description(vpp_disparity_img, "VPP Disparity", f"MAE: {vpp_metrics['avgerr']}, BAD3: {vpp_metrics['bad 3.0']}")
                            vpp_disparity_img = add_title_description(vpp_disparity_img, "VPP Disparity", " ")
                            mylog(f"Text Time: {(time.time()-start_time)}", True)
                            
                            start_time = time.time()
                            top_frame = np.hstack([vanilla_left_img, vanilla_right_img, vanilla_disparity_img])
                            bottom_frame = np.hstack([vpp_left_img, vpp_right_img, vpp_disparity_img])
                            frame_img = np.vstack([top_frame, bottom_frame])
                            mylog(f"Stack time: {(time.time()-start_time)}", True)

                            # ~1920x960 -> ~960x480
                            # frame_img = cv2.resize(frame_img, (0,0), fx=0.5, fy=0.5) 
                            # cv2.imwrite(os.path.join("tmp", "frame_img.png"), frame_img)

                            start_time = time.time()
                            cv2.imshow(GUI_WINDOW_NAME, frame_img)
                            mylog(f"IMSHOW time: {(time.time()-start_time)}", True)

                            key = cv2.waitKey(1)

                            if key == ord('q'):
                                mylog("Quitting...")
                                break

        except KeyboardInterrupt:
            mylog(f"CRTL-C received")
        finally:
            mylog(f"Releasing resources and closing.")
            cv2.destroyAllWindows()
            stop_condition_slave = True    
            slave_thread.join()            
            
if __name__ == '__main__':
    # Create the argument parser
    parser = argparse.ArgumentParser(description="MID70+OAK SGM demo.")
    
    # Add a positional argument for the save folder path
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--livox_broadcast_code", type=str, default=None, help="Livox broadcast code")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    if args.livox_broadcast_code is not None:
        MID70_BROADCAST_CODE = args.livox_broadcast_code

    print(f"Broadcast code: {MID70_BROADCAST_CODE}")
    print(f"Verbose: {args.verbose}")

    main(args)
