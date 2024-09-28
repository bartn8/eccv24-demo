import mid70grabber
import numpy as np
import time

# Start the livox stream with only one device associated
started = mid70grabber.start([""], 1000)

if not started:
    print("Error starting the stream")
    exit()

# Wait until a device is connected
devices = mid70grabber.get_devices()

while len(devices) == 0:
    print("Waiting for devices...")
    time.sleep(1)
    devices = mid70grabber.get_devices()

print(devices)

# Grab a frame
frame, timestamp_start, timestamp_end = mid70grabber.get_frame(devices[0]["handle"])

# Stop the livox stream
mid70grabber.stop()

print("Timestamp Start:", timestamp_start)
print("Timestamp End:", timestamp_end)
print("Data received:", frame.shape)

# Save the data to a numpy file
np.savetxt("frame.txt", frame)

