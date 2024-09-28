import time
import numpy as np
import depthai as dai 
import mid70grabber
import cv2
import threading
import os
import argparse
import json
import traceback

from datetime import timedelta
from mycoda import SlidingWindowDevice, MID70FrameWrapper
from utils import add_title_description, compute_camera_parameters, get_view, resize_image

SYNC_TH_MS = 100/1000.0
MAX_BUFFER_SIZE = 3
PREVIEW_SIZE = (480,640)

Z_MAX = 10.0

OAK_FPS = 30
OAK_BASELINE = 75 / 1000.0
OAK_SUBPIXEL = True
OAK_LRC = True
OAK_EXTENDED_DISPARITY = False
OAK_FRACTIONAL_BITS = 3
OAK_DISPARITY_DIV = 2 ** OAK_FRACTIONAL_BITS if OAK_SUBPIXEL else 1
OAK_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P
OAK_RGB_RESOLUTION = dai.ColorCameraProperties.SensorResolution.THE_1080_P

OAK_RESOLUTION_DICT = {
    dai.MonoCameraProperties.SensorResolution.THE_400_P: (400, 640),
    dai.MonoCameraProperties.SensorResolution.THE_480_P: (480, 640),
    dai.MonoCameraProperties.SensorResolution.THE_720_P: (720, 1280),
    dai.MonoCameraProperties.SensorResolution.THE_800_P: (800, 1280),
}

OAK_RGB_RESOLUTION_DICT = {
    dai.ColorCameraProperties.SensorResolution.THE_720_P: (720, 1280),
    dai.ColorCameraProperties.SensorResolution.THE_1080_P: (1080, 1920)
}

OAK_SAVE_FOLDER = "oak"
OAK_LEFT_FOLDER = "left"
OAK_RIGHT_FOLDER = "right"
OAK_DEPTH_FOLDER = "depth"
OAK_DISPARITY_FOLDER = "disparity"
OAK_RGB_FOLDER = "rgb"
OAK_CALIB_FILE = "calib.json"
OAK_TIMESTAMP_DEPTH_FILE = "timestamp_depth.txt"
OAK_TIMESTAMP_LEFT_FILE = "timestamp_left.txt"

MID70_FPS = 0.3
MID70_BROADCAST_CODE = ""
MID70_FAKE_W = 720
MID70_FAKE_H = 720
MID70_FAKE_FOV = np.deg2rad(72)
RT_CRF_MID70 = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]]).astype(np.float32)
MID70_K = compute_camera_parameters(MID70_FAKE_W, MID70_FAKE_H, MID70_FAKE_FOV)

MID70_SAVE_FOLDER = "mid70"
MID70_DEPTH_FOLDER = "depth"
MID70_REFLECTIVITY_FOLDER = "reflectivity"
MID70_CALIB_FILE = "calib.json"
MID70_TIMESTAMP_DEPTH_FILE = "timestamp_depth.txt"
MID70_TIMESTAMP_REFLECTIVITY_FILE = "timestamp_reflectivity.txt"

#Thread-safe sliding window to store MID70 frames
mid70_sliding_window = SlidingWindowDevice(MAX_BUFFER_SIZE)
# MID70 thread stop condition
stop_condition_slave = False

# MID70 thread that load frames asynchronously
def slave(thread_id, args): 
    global mid70_sliding_window
    global stop_condition_slave

    calibration_data_retrived = False
   
    def mylog(x, debug=False):
        if args.verbose or not debug:
            print(f"SLAVE ({thread_id}): {x}")
    
    started = mid70grabber.start([MID70_BROADCAST_CODE], round(1000/MID70_FPS))
    if not started:
        print("Error starting the stream")
        exit()

    # Wait until a device is connected
    devices = mid70grabber.get_devices()

    while len(devices) == 0:
        print("Waiting for devices...")
        time.sleep(0.1)
        devices = mid70grabber.get_devices()

    try:
        mylog(f"Lidar ready")

        #Fill sliding window until stop condition
        while not stop_condition_slave: 

            # Grab a frame
            frame, timestamp_start, timestamp_end = mid70grabber.get_frame(devices[0]["handle"])

            if len(frame) > 0:
                #Convert Nx4 points to an image
                depth_image, reflectivity_image = get_view(frame, RT_CRF_MID70, MID70_K, np.zeros(5), MID70_FAKE_H, MID70_FAKE_W)
                
                if np.max(depth_image) > 0:
                    frame_wrapper = MID70FrameWrapper(depth_image, reflectivity_image, start_scan_timestamp=timestamp_start, end_scan_timestamp=timestamp_end)
                    mid70_sliding_window.add_element(frame_wrapper)
                
                #Retrive calibration info one time only (depth==reflectivity calib)
                if not calibration_data_retrived:
                    calib_dict = {
                        f"K_depth_{MID70_FAKE_H}x{MID70_FAKE_W}": MID70_K.tolist(),
                        f"K_reflectivity_{MID70_FAKE_H}x{MID70_FAKE_W}": MID70_K.tolist(),
                        "RT_reflectivity_depth": np.eye(4).tolist(),
                        "RT_depth_reflectivity": np.eye(4).tolist(),
                        "RT_crf": RT_CRF_MID70.tolist(),
                    }

                    with open(os.path.join(args.outdir, MID70_SAVE_FOLDER, MID70_CALIB_FILE), "w") as f:
                        json.dump(calib_dict, f, indent=4)

                    mylog(f"Calibration JSON file saved at: {os.path.join(args.outdir, MID70_SAVE_FOLDER, MID70_CALIB_FILE)}")
                    calibration_data_retrived = True
                
    except Exception as e:
        mylog(f"Something went wrong: {e}")
        #logging.error(traceback.format_exc())
    finally:
        # Stop the livox stream
        mylog(f"Stopping the livox stream:{stop_condition_slave}", True)
        mid70grabber.stop()

def main(args):
    global stop_condition_slave
    global mid70_sliding_window

    os.makedirs(os.path.join(args.outdir, OAK_SAVE_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, MID70_SAVE_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, MID70_SAVE_FOLDER, MID70_DEPTH_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, MID70_SAVE_FOLDER, MID70_REFLECTIVITY_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_DISPARITY_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_DEPTH_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_LEFT_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_RIGHT_FOLDER), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_RGB_FOLDER), exist_ok=True)

    fd_ts_oak_depth = open(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_TIMESTAMP_DEPTH_FILE), "w")
    fd_ts_oak_left = open(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_TIMESTAMP_LEFT_FILE), "w")

    fd_ts_mid70_depth = open(os.path.join(args.outdir, MID70_SAVE_FOLDER, MID70_TIMESTAMP_DEPTH_FILE), "w")
    fd_ts_mid70_reflectivity = open(os.path.join(args.outdir, MID70_SAVE_FOLDER, MID70_TIMESTAMP_REFLECTIVITY_FILE), "w")

    def mylog(x, debug=False):
        if args.verbose or not debug:
            print(f"MASTER: {x}")

    save_counter = 0
    calibration_data_retrived = False

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

    # Configure cameras: set fps lower than MID70 to better accumulate depth frames
    left_node.setResolution(OAK_RESOLUTION)  
    left_node.setFps(OAK_FPS) 
    right_node.setResolution(OAK_RESOLUTION)  
    right_node.setFps(OAK_FPS)  
    left_node.setCamera("left")
    right_node.setCamera("right")

    rgb_node = pipeline_OAK.createColorCamera() 
    rgb_node.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    rgb_node.setResolution(OAK_RGB_RESOLUTION)
    rgb_node.setFps(OAK_FPS) 
    rgb_node.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

    #Configure SGM nodes: set alignment and other stuff
    vanilla_sgm_node.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
    vanilla_sgm_node.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_5x5)
    vanilla_sgm_node.setLeftRightCheck(OAK_LRC)
    vanilla_sgm_node.setExtendedDisparity(OAK_EXTENDED_DISPARITY)
    vanilla_sgm_node.setSubpixel(OAK_SUBPIXEL)
    vanilla_sgm_node.setSubpixelFractionalBits(OAK_FRACTIONAL_BITS)
    vanilla_sgm_node.setDepthAlign(dai.RawStereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_LEFT)
    # vanilla_sgm_node.setDepthAlign(dai.CameraBoardSocket.CAM_B)
    vanilla_sgm_node.setRuntimeModeSwitch(True)

    # Create in/out channels
    # Required input channels: 
    # Required output channels: vanilla_disparity, left_vanilla_rectified, right_vanilla_rectified

    xout_vanilla_disparity = pipeline_OAK.createXLinkOut()
    xout_vanilla_disparity.setStreamName('vanilla_disparity')
    xout_vanilla_depth = pipeline_OAK.createXLinkOut()
    xout_vanilla_depth.setStreamName('vanilla_depth')

    xout_left_vanilla_rectified = pipeline_OAK.createXLinkOut()
    xout_left_vanilla_rectified.setStreamName('left_vanilla_rectified')
    xout_right_vanilla_rectified = pipeline_OAK.createXLinkOut()
    xout_right_vanilla_rectified.setStreamName('right_vanilla_rectified')

    xout_rgb = pipeline_OAK.createXLinkOut()
    xout_rgb.setStreamName('rgb')

    # Link channels and nodes
    left_node.out.link(vanilla_sgm_node.left)
    right_node.out.link(vanilla_sgm_node.right)
    vanilla_sgm_node.rectifiedLeft.link(xout_left_vanilla_rectified.input)      # Should be the same as the captured frame 
    vanilla_sgm_node.rectifiedRight.link(xout_right_vanilla_rectified.input)    # Should be the same as the captured frame
    vanilla_sgm_node.disparity.link(xout_vanilla_disparity.input) 
    vanilla_sgm_node.depth.link(xout_vanilla_depth.input) 

    rgb_node.video.link(xout_rgb.input)

    with dai.Device(pipeline_OAK) as device:
        #Sync OAK clock with host clock
        #Simple hack here: assume diff constant
        # device.setTimesync(True)# Already the default config
        oak_clock:timedelta = dai.Clock.now()
        host_clock = time.time()
        diff = host_clock-oak_clock.total_seconds()

        if not calibration_data_retrived:
            calibData = device.readCalibration()
            baseline = calibData.getBaselineDistance(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)

            D_left = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_B))
            D_left = {name:value for (name, value) in zip(["k1","k2","p1","p2","k3","k4","k5","k6"], D_left[:8])}
            D_right = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_C))
            D_right = {name:value for (name, value) in zip(["k1","k2","p1","p2","k3","k4","k5","k6"], D_right[:8])}
            D_rgb = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A))
            D_rgb = {name:value for (name, value) in zip(["k1","k2","p1","p2","k3","k4","k5","k6"], D_rgb[:8])}

            RT_rgb_left = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_A))
            RT_rgb_left[:3,3] = RT_rgb_left[:3,3] / 100.0
            RT_right_left = np.array(calibData.getCameraExtrinsics(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C))
            RT_right_left[:3,3] = RT_right_left[:3,3] / 100.0
            
            calib_dict = {
                "baseline": baseline / 100.0,
                "RT_rgb_left": RT_rgb_left.tolist(),
                "RT_right_left": RT_right_left.tolist(),
                "R_leftr": np.array(calibData.getStereoLeftRectificationRotation()).tolist(),
                "R_rightr": np.array(calibData.getStereoRightRectificationRotation()).tolist(),
                "D_left": D_left,
                "D_right": D_right,
                "D_rgb": D_rgb,
            }

            for mono_res in OAK_RESOLUTION_DICT.keys():
                _H, _W = OAK_RESOLUTION_DICT[mono_res]
                calib_dict[f"K_left_{_H}x{_W}"] = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, _W, _H)).tolist()
                calib_dict[f"K_right_{_H}x{_W}"] = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_C, _W, _H)).tolist()

            for rgb_res in OAK_RGB_RESOLUTION_DICT.keys():
                _H, _W = OAK_RGB_RESOLUTION_DICT[rgb_res]
                calib_dict[f"K_rgb_{_H}x{_W}"] = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, _W, _H)).tolist()

            with open(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_CALIB_FILE), "w") as f:
                json.dump(calib_dict, f, indent=4)

            mylog(f"Calibration JSON file saved at: {os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_CALIB_FILE)}")
            calibration_data_retrived = True

        mylog(f"OAK-D ready")

        #Create in/out queues
        out_queue_left_vanilla_rectified = device.getOutputQueue(name="left_vanilla_rectified", maxSize=1, blocking=False) 
        out_queue_right_vanilla_rectified = device.getOutputQueue(name="right_vanilla_rectified", maxSize=1, blocking=False) 
        out_queue_vanilla_disparity = device.getOutputQueue(name="vanilla_disparity", maxSize=1, blocking=False) 
        out_queue_vanilla_depth = device.getOutputQueue(name="vanilla_depth", maxSize=1, blocking=False) 
        out_queue_rgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

        try:
            #Until exception (Keyboard interrupt) or q pressed
            while True:

                #Recording pipeline:
                #1) Get left/right rectified images and vanilla prediction
                #2) Search for the nearest MID70 frames
                #3) Save all
                
                vanilla_depth_data = out_queue_vanilla_depth.get()   
                vanilla_disparity_data = out_queue_vanilla_disparity.get()       
                left_vanilla_rectified_data = out_queue_left_vanilla_rectified.get() 
                right_vanilla_rectified_data = out_queue_right_vanilla_rectified.get() 
                rgb_data = out_queue_rgb.get()

                timestamp_OAK_depth = vanilla_depth_data.getTimestamp(dai.CameraExposureOffset.END).total_seconds()+diff
                timestamp_OAK_left = left_vanilla_rectified_data.getTimestamp(dai.CameraExposureOffset.END).total_seconds()+diff
                timestamp_OAK_right = right_vanilla_rectified_data.getTimestamp(dai.CameraExposureOffset.END).total_seconds()+diff
                timestamp_OAK_rgb = rgb_data.getTimestamp(dai.CameraExposureOffset.END).total_seconds()+diff

                mid70_depth, mid70_reflectivity = None, None
                oak_vanilla_disparity, oak_vanilla_depth, oak_left_rectified, oak_right_rectified = None, None, None, None
               
                #if the MID70 sliding window is too new then drop some OAK frames
                if not mid70_sliding_window.is_window_too_new("reflectivity", timestamp_OAK_left, 0.001):                              
                    
                    #if the OAK frame is too new then wait until the window is fresh
                    while mid70_sliding_window.is_window_too_old("reflectivity", timestamp_OAK_left, 0.001):
                        time.sleep(0.001)

                    # Get the nearest MID70 frame based on OAK frame timestamp
                    nearest_MID70_frame:MID70FrameWrapper = mid70_sliding_window.nearest_frame("reflectivity", timestamp_OAK_left)

                    if nearest_MID70_frame is not None:
                        mid70_depth = (nearest_MID70_frame.get_frame("depth") * nearest_MID70_frame.get_frame("depth_scale"))
                        mid70_reflectivity = (nearest_MID70_frame.get_frame("reflectivity") * 255.0)

                        mid70_depth = mid70_depth.astype(np.float32)
                        mid70_reflectivity = mid70_reflectivity.astype(np.float32)

                        timestamp_mid70_depth = nearest_MID70_frame.get_timestamp("depth")
                        timestamp_mid70_reflectivity = nearest_MID70_frame.get_timestamp("reflectivity")

                        oak_left_rectified = left_vanilla_rectified_data.getCvFrame()
                        oak_right_rectified = right_vanilla_rectified_data.getCvFrame()
                        oak_rgb = rgb_data.getCvFrame()

                        oak_vanilla_disparity = vanilla_disparity_data.getFrame() / OAK_DISPARITY_DIV
                        oak_vanilla_depth = vanilla_depth_data.getFrame() / 1000.0

                #Show captured streams and ask if the user want to save current frames 
                if mid70_depth is not None and mid70_reflectivity is not None and oak_vanilla_disparity is not None and oak_vanilla_depth is not None and oak_left_rectified is not None and oak_right_rectified is not None:
                    
                    # MID70 Reflecivity | Empty     | MID70 Depth
                    # -----------------------------------------
                    # OAK Left          | OAK Right | OAK Depth

                    # Press 's' to save, 'ENTER' to skip, 'q' to quit

                    #Sync delta: assuming a global clock between MID70 and OAK, observe the time difference between them
                    delta_oak_mid70_depth = abs(timestamp_OAK_depth-timestamp_mid70_depth)

                    mylog(f"Delta OAK-MID70 Depth: {delta_oak_mid70_depth}", True)

                    #Keep frames only if meet time requirements
                    if delta_oak_mid70_depth < SYNC_TH_MS:

                        #Create stacked frame and add timestamps to the images

                        mid70_reflectivity_img = np.copy(mid70_reflectivity.astype(np.uint8))
                        mid70_reflectivity_img[mid70_reflectivity_img>10] = 255
                        mid70_reflectivity_img = cv2.medianBlur(mid70_reflectivity_img, 3)
                        mid70_reflectivity_img = cv2.cvtColor(mid70_reflectivity_img, cv2.COLOR_GRAY2BGR)
                        # mid70_black_img = np.zeros_like(mid70_reflectivity_img, dtype=np.uint8)
                        mid70_depth_img = np.copy(mid70_depth)
                        
                        oak_left_img = cv2.cvtColor(np.copy(oak_left_rectified).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                        oak_right_img = cv2.cvtColor(np.copy(oak_right_rectified), cv2.COLOR_GRAY2BGR)
                        oak_depth_img = np.copy(oak_vanilla_depth)         
                        oak_rgb_img = np.copy(oak_rgb)               

                        #Apply colormaps
                        mid70_depth_img = cv2.applyColorMap((255.0 * np.clip(mid70_depth_img, 0, Z_MAX) / Z_MAX).astype(np.uint8), cv2.COLORMAP_MAGMA)
                        oak_depth_img = cv2.applyColorMap((255.0 * np.clip(oak_depth_img, 0, Z_MAX) / Z_MAX).astype(np.uint8), cv2.COLORMAP_MAGMA)

                        #Add text
                        mid70_depth_img = add_title_description(mid70_depth_img, "MID70 Depth", f"TS: {timestamp_mid70_depth}")
                        mid70_reflectivity_img = add_title_description(mid70_reflectivity_img, "MID70 REFLECTIVITY", f"TS: {timestamp_mid70_reflectivity}")

                        oak_left_img = add_title_description(oak_left_img, "OAK Left", f"TS: {timestamp_OAK_left}")
                        oak_right_img = add_title_description(oak_right_img, "OAK Right", f"TS: {timestamp_OAK_right}")
                        oak_depth_img = add_title_description(oak_depth_img, "OAK Depth", f"TS: {timestamp_OAK_depth}")
                        oak_rgb_img = add_title_description(oak_rgb_img, "OAK RGB", f"TS: {timestamp_OAK_rgb}")

                        top_frame = np.hstack(resize_image([mid70_reflectivity_img, mid70_depth_img, oak_depth_img], *PREVIEW_SIZE))
                        bottom_frame = np.hstack(resize_image([oak_left_img, oak_rgb_img, oak_right_img], *PREVIEW_SIZE))
                        frame_img = np.vstack([top_frame, bottom_frame])
                        # ~1920x960 -> ~960x480
                        # frame_img = cv2.resize(frame_img, (0,0), fx=0.5, fy=0.5) 
                        # cv2.imwrite(os.path.join(args.outdir, "frame_img.png"), frame_img)
                        
                        mylog("Press 'q' to quit recording; 's' to save frame; any other key to skip frame acquisition")
                        cv2.imshow("Preview", frame_img)

                        key = cv2.waitKey(0)

                        if key == ord('q'):
                            mylog("Quitting...")
                            break

                        if key == ord('s'):
                            #Save all frames and timestamps
                            cv2.imwrite(os.path.join(args.outdir, MID70_SAVE_FOLDER, MID70_DEPTH_FOLDER, f"{save_counter:06}.png"), (1000.0 * mid70_depth).astype(np.uint16))
                            cv2.imwrite(os.path.join(args.outdir, MID70_SAVE_FOLDER, MID70_REFLECTIVITY_FOLDER, f"{save_counter:06}.png"), (mid70_reflectivity).astype(np.uint8))

                            cv2.imwrite(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_DISPARITY_FOLDER, f"{save_counter:06}.png"), (256.0 * oak_vanilla_disparity).astype(np.uint16))
                            cv2.imwrite(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_DEPTH_FOLDER, f"{save_counter:06}.png"), (1000.0 * oak_vanilla_depth).astype(np.uint16))
                            cv2.imwrite(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_LEFT_FOLDER, f"{save_counter:06}.png"), (oak_left_rectified).astype(np.uint8))
                            cv2.imwrite(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_RIGHT_FOLDER, f"{save_counter:06}.png"), (oak_right_rectified).astype(np.uint8))
                            cv2.imwrite(os.path.join(args.outdir, OAK_SAVE_FOLDER, OAK_RGB_FOLDER, f"{save_counter:06}.png"), (oak_rgb).astype(np.uint8))

                            fd_ts_oak_depth.write(f"{timestamp_OAK_depth}\n")
                            fd_ts_oak_left.write(f"{timestamp_OAK_left}\n")

                            fd_ts_mid70_depth.write(f"{timestamp_mid70_depth}\n")
                            fd_ts_mid70_reflectivity.write(f"{timestamp_mid70_reflectivity}\n")

                            fd_ts_oak_depth.flush()
                            fd_ts_oak_left.flush()

                            fd_ts_mid70_depth.flush()
                            fd_ts_mid70_reflectivity.flush()


                            save_counter += 1

                            mylog(f"Frame {save_counter} saved.")

        except KeyboardInterrupt:
            mylog(f"CRTL-C received")
        except Exception:
            mylog(f"Something went wrong: {traceback.format_exc()}")
        finally:
            mylog(f"Releasing resources and closing.")

            fd_ts_oak_depth.close()
            fd_ts_oak_left.close()

            fd_ts_mid70_depth.close()
            fd_ts_mid70_reflectivity.close()

            cv2.destroyAllWindows()
            stop_condition_slave = True    
            slave_thread.join()            
            
if __name__ == '__main__':

    # Create the argument parser
    parser = argparse.ArgumentParser(description="MID70+OAK recorder.")
    
    # Add a positional argument for the save folder path
    parser.add_argument("--outdir", type=str, required=True, help="Path to the save folder")
    parser.add_argument("--verbose", action="store_true", help="Print debug messages")
    parser.add_argument("--broadcast_code", type=str, default=MID70_BROADCAST_CODE, help="MID70 broadcast code")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    MID70_BROADCAST_CODE = args.broadcast_code

    main(args)
