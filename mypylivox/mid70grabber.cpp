#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <functional>
#include <livox_sdk.h>
#include <chrono>
#include <thread>

namespace py = pybind11;

// Define the struct of a single point in the point cloud
typedef struct
{
    int32_t x;
    int32_t y;
    int32_t z;
    uint8_t reflectivity;
} Point3D;

typedef enum
{
    kDeviceStateDisconnect = 0,
    kDeviceStateConnect = 1,
    kDeviceStateSampling = 2,
} DeviceState;

// Define the main state struct for each device
// For each frame, a list of points is stored
// Send using the callback function the list of points when the frame is complete
// Assume that the scene is static during the frame
// Assume only one device attached
typedef struct
{
    uint8_t handle;
    DeviceState device_state;
    DeviceInfo info;

    std::vector<Point3D> vec;
    uint64_t firstLidarTimestamp;
    uint64_t lastLidarTimestamp;
} DeviceItem;

// GLOBALS

// Device array for each connected device
DeviceItem devices[kMaxLidarCount];
// Define the frame duration in nanoseconds
uint64_t frameDuration = 0;
// Global flag to check if the system is running
bool isRunning = false;
// Global flag to check if the frame recording has started
bool startFrameRecording[kMaxLidarCount];
// Global flag to check if the frame is ready
bool frameReady[kMaxLidarCount];
// Use this flag to enable prints
bool debug = false;

/** Connect the broadcast device in list, please input the broadcast code and modify the BROADCAST_CODE_LIST_SIZE. */
#define BROADCAST_CODE_LIST_SIZE kMaxLidarCount
char broadcast_code_list[kMaxLidarCount][kBroadcastCodeSize];



//-----------------------------------------------------------------------------------------------

/** Receiving error message from Livox Lidar. */
void OnLidarErrorStatusCallback(livox_status status, uint8_t handle, ErrorMessage *message)
{
    static uint32_t error_message_count = 0;
    if (message != NULL)
    {
        ++error_message_count;

        // TODO: handle using python exceptions
        if (0 == (error_message_count % 100))
        {
            if (debug)
            {
                printf("handle: %u\n", handle);
                printf("temp_status : %u\n", message->lidar_error_code.temp_status);
                printf("volt_status : %u\n", message->lidar_error_code.volt_status);
                printf("motor_status : %u\n", message->lidar_error_code.motor_status);
                printf("dirty_warn : %u\n", message->lidar_error_code.dirty_warn);
                printf("firmware_err : %u\n", message->lidar_error_code.firmware_err);
                printf("pps_status : %u\n", message->lidar_error_code.device_status);
                printf("fan_status : %u\n", message->lidar_error_code.fan_status);
                printf("self_heating : %u\n", message->lidar_error_code.self_heating);
                printf("ptp_status : %u\n", message->lidar_error_code.ptp_status);
                printf("time_sync_status : %u\n", message->lidar_error_code.time_sync_status);
                printf("system_status : %u\n", message->lidar_error_code.system_status);
            }
        }
    }
}

void polarToCartesian(Point3D *point, uint32_t r, uint16_t theta, uint16_t phi)
{
    point->x = r * sin((double)phi) * cos((double)theta);
    point->y = r * sin((double)phi) * sin((double)theta);
    point->z = r * cos((double)phi);
}

void writePoint(Point3D point, uint8_t type, uint8_t handle, uint64_t cur_timestamp)
{

    if (point.x == 0 && point.y == 0 && point.z == 0)
    {
        return;
    }

    // Check if user requested a frame
    // If so, push points into the vector and handle timestamps
    // When the frame is caputred, flag frameReady
    if (startFrameRecording[handle])
    {
        devices[handle].vec.push_back(point);

        if (devices[handle].firstLidarTimestamp == 0)
        {
            devices[handle].firstLidarTimestamp = cur_timestamp;
        }
        else
        {
            if (!frameReady[handle])
            {
                if (cur_timestamp >= devices[handle].firstLidarTimestamp + frameDuration)
                { // frame is ready
                    devices[handle].lastLidarTimestamp = cur_timestamp;
                    frameReady[handle] = true;
                }
            }
        }
    }
    else
    {
        frameReady[handle] = false;
        devices[handle].vec.clear();
        devices[handle].firstLidarTimestamp = 0;
        devices[handle].lastLidarTimestamp = 0;
    }
}

/** Receiving point cloud data from Livox LiDAR. */
void GetLidarData(uint8_t handle, LivoxEthPacket *data, uint32_t data_num, void *client_data)
{
    uint8_t type;

    for (uint32_t index = 0; index < data_num; index++)
    {
        /** Parsing the timestamp and the point cloud data. */
        uint64_t cur_timestamp = *((uint64_t *)(data->timestamp));

        // 1 punto coord. cartesiane
        if (data->data_type == kCartesian)
        {
            type = kCartesian;
            LivoxRawPoint *p_point_data = &(((LivoxRawPoint *)data->data)[index]);
            Point3D point = {p_point_data->x, p_point_data->y, p_point_data->z, p_point_data->reflectivity};
            writePoint(point, type, handle, cur_timestamp);
        }
        else if (data->data_type == kSpherical)
        {
            type = kSpherical;
            Point3D point;
            LivoxSpherPoint *p_point_data = &(((LivoxSpherPoint *)data->data)[index]);
            polarToCartesian(&point, p_point_data->depth, p_point_data->theta, p_point_data->phi);
            writePoint(point, type, handle, cur_timestamp);
        }
        else if (data->data_type == kExtendCartesian)
        {
            type = kExtendCartesian;
            LivoxExtendRawPoint *p_point_data = &(((LivoxExtendRawPoint *)data->data)[index]);
            Point3D point = {p_point_data->x, p_point_data->y, p_point_data->z, p_point_data->reflectivity};
            writePoint(point, type, handle, cur_timestamp);
        }
        else if (data->data_type == kExtendSpherical)
        {
            type = kExtendSpherical;
            Point3D point;
            LivoxExtendSpherPoint *p_point_data = &(((LivoxExtendSpherPoint *)data->data)[index]);
            polarToCartesian(&point, p_point_data->depth, p_point_data->theta, p_point_data->phi);
            writePoint(point, type, handle, cur_timestamp);
        }
        else if (data->data_type == kDualExtendCartesian)
        {
            type = kDualExtendCartesian;
            LivoxDualExtendRawPoint *p_point_data = &(((LivoxDualExtendRawPoint *)data->data)[index]);
            Point3D point1 = {p_point_data->x1, p_point_data->y1, p_point_data->z1, p_point_data->reflectivity1};
            Point3D point2 = {p_point_data->x2, p_point_data->y2, p_point_data->z2, p_point_data->reflectivity2};
            writePoint(point1, type, handle, cur_timestamp);
            writePoint(point2, type, handle, cur_timestamp);
        }
        else if (data->data_type == kDualExtendSpherical)
        {
            type = kDualExtendSpherical;
            LivoxDualExtendSpherPoint *p_point_data = &(((LivoxDualExtendSpherPoint *)data->data)[index]);

            Point3D point1;
            polarToCartesian(&point1, p_point_data->depth1, p_point_data->theta, p_point_data->phi);
            writePoint(point1, type, handle, cur_timestamp);

            Point3D point2;
            polarToCartesian(&point2, p_point_data->depth2, p_point_data->theta, p_point_data->phi);
            writePoint(point2, type, handle, cur_timestamp);
        }
        else if (data->data_type == kTripleExtendCartesian)
        {
            type = kTripleExtendCartesian;
            LivoxTripleExtendRawPoint *p_point_data = &(((LivoxTripleExtendRawPoint *)data->data)[index]);

            Point3D point1 = {p_point_data->x1, p_point_data->y1, p_point_data->z1, p_point_data->reflectivity1};
            Point3D point2 = {p_point_data->x2, p_point_data->y2, p_point_data->z2, p_point_data->reflectivity2};
            Point3D point3 = {p_point_data->x3, p_point_data->y3, p_point_data->z3, p_point_data->reflectivity3};

            writePoint(point1, type, handle, cur_timestamp);
            writePoint(point2, type, handle, cur_timestamp);
            writePoint(point3, type, handle, cur_timestamp);
        }
        else if (data->data_type == kTripleExtendSpherical)
        {
            type = kTripleExtendSpherical;
            LivoxTripleExtendSpherPoint *p_point_data = &(((LivoxTripleExtendSpherPoint *)data->data)[index]);

            Point3D point1;
            polarToCartesian(&point1, p_point_data->depth1, p_point_data->theta, p_point_data->phi);
            writePoint(point1, type, handle, cur_timestamp);

            Point3D point2;
            polarToCartesian(&point2, p_point_data->depth2, p_point_data->theta, p_point_data->phi);
            writePoint(point2, type, handle, cur_timestamp);

            Point3D point3;
            polarToCartesian(&point3, p_point_data->depth3, p_point_data->theta, p_point_data->phi);
            writePoint(point3, type, handle, cur_timestamp);
        }
    }
}

/** Callback function of starting sampling. */
void OnSampleCallback(livox_status status, uint8_t handle, uint8_t response, void *data)
{
    if (debug)
    {
        printf("OnSampleCallback statue %d handle %d response %d \n", status, handle, response);
    }

    if (status == kStatusSuccess)
    {
        if (response != 0)
        {
            devices[handle].device_state = kDeviceStateConnect;
        }
    }
    else if (status == kStatusTimeout)
    {
        devices[handle].device_state = kDeviceStateConnect;
    }
}

/** Callback function of stopping sampling. */
void OnStopSampleCallback(livox_status status, uint8_t handle, uint8_t response, void *data)
{
}

/** Query the firmware version of Livox LiDAR. */
void OnDeviceInformation(livox_status status, uint8_t handle, DeviceInformationResponse *ack, void *data)
{
    if (status != kStatusSuccess)
    {
        if (debug)
        {
            printf("Device Query Informations Failed %d\n", status);
        }
    }
    if (ack)
    {
        if (debug)
        {
            printf("firm ver: %d.%d.%d.%d\n",
                   ack->firmware_version[0],
                   ack->firmware_version[1],
                   ack->firmware_version[2],
                   ack->firmware_version[3]);
        }
    }
}

void LidarConnect(const DeviceInfo *info)
{
    uint8_t handle = info->handle;
    QueryDeviceInformation(handle, OnDeviceInformation, NULL);
    if (devices[handle].device_state == kDeviceStateDisconnect)
    {
        devices[handle].device_state = kDeviceStateConnect;
        devices[handle].info = *info;
    }
}

void LidarDisConnect(const DeviceInfo *info)
{
    uint8_t handle = info->handle;
    devices[handle].device_state = kDeviceStateDisconnect;
}

void LidarStateChange(const DeviceInfo *info)
{
    uint8_t handle = info->handle;
    devices[handle].info = *info;
}

/** Callback function of changing of device state. */
void OnDeviceInfoChange(const DeviceInfo *info, DeviceEvent type)
{
    if (info == NULL)
    {
        return;
    }

    uint8_t handle = info->handle;
    if (handle >= kMaxLidarCount)
    {
        return;
    }
    if (type == kEventConnect)
    {
        LidarConnect(info);

        if (debug)
        {
            printf("[WARNING] Lidar sn: [%s] Connect!!!\n", info->broadcast_code);
        }
    }
    else if (type == kEventDisconnect)
    {
        LidarDisConnect(info);

        if (debug)
        {
            printf("[WARNING] Lidar sn: [%s] Disconnect!!!\n", info->broadcast_code);
        }
    }
    else if (type == kEventStateChange)
    {
        LidarStateChange(info);

        if (debug)
        {
            printf("[WARNING] Lidar sn: [%s] StateChange!!!\n", info->broadcast_code);
        }
    }

    if (devices[handle].device_state == kDeviceStateConnect)
    {
        if (debug)
        {
            printf("Device Working State %d\n", devices[handle].info.state);
        }
        if (devices[handle].info.state == kLidarStateInit)
        {
            if (debug)
            {
                printf("Device State Change Progress %u\n", devices[handle].info.status.progress);
            }
        }
        else
        {
            if (debug)
            {
                printf("Device State Error Code 0X%08x\n", devices[handle].info.status.status_code.error_code);
            }
        }

        if (debug)
        {
            printf("Device feature %d\n", devices[handle].info.feature);
        }

        SetErrorMessageCallback(handle, OnLidarErrorStatusCallback);

        if (devices[handle].info.state == kLidarStateNormal)
        {
            LidarStartSampling(handle, OnSampleCallback, NULL);
            devices[handle].device_state = kDeviceStateSampling;
        }
    }
}

/** Callback function when broadcast message received.
 * You need to add listening device broadcast code and set the point cloud data callback in this function.
 */
void OnDeviceBroadcast(const BroadcastDeviceInfo *info)
{
    if (info == NULL || info->dev_type == kDeviceTypeHub)
    {
        return;
    }

    if (debug)
    {
        printf("Receive Broadcast Code %s\n", info->broadcast_code);
    }

    if (BROADCAST_CODE_LIST_SIZE > 0)
    {
        bool found = false;
        int i = 0;
        for (i = 0; i < BROADCAST_CODE_LIST_SIZE; ++i)
        {
            if (strncmp(info->broadcast_code, broadcast_code_list[i], kBroadcastCodeSize) == 0)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            return;
        }
    }

    bool result = false;
    uint8_t handle = 0;
    result = AddLidarToConnect(info->broadcast_code, &handle);
    if (result == kStatusSuccess)
    {
        /** Set the point cloud data for a specific Livox LiDAR. */
        SetDataCallback(handle, GetLidarData, NULL);
        devices[handle].handle = handle;
        devices[handle].device_state = kDeviceStateDisconnect;
    }
}

bool start(const std::vector<std::string> &broadcast_list, int frame_duration_ms)
{
    if (isRunning)
    {
        return false;
    }

    int broadcast_code_i = 0;

    for (const auto &broadcast_code : broadcast_list)
    {
        if (broadcast_code.size() != 15)
        {
            py::print("Error: broadcast_code should be 15 characters long");
            return false;
        }

        // Mid-70 broadcast code has "1" at the end: suppose that user already added it
        // std::string full_broadcast_code = broadcast_code + "1";
        strcpy(broadcast_code_list[broadcast_code_i], broadcast_code.c_str());
        broadcast_code_i++;
    }

    for(int i = 0; i < kMaxLidarCount; i++)
    {
        startFrameRecording[i] = false;
        frameReady[i] = false;
    }

    // Set the frame duration in nanoseconds from milliseconds
    frameDuration = (uint64_t)frame_duration_ms * 1000000;

    py::print("Livox SDK initializing.");

    /** Initialize Livox-SDK. */

    if (!Init())
    {
        return false;
    }

    py::print("Livox SDK has been initialized.");

    LivoxSdkVersion _sdkversion;
    GetLivoxSdkVersion(&_sdkversion);

    // py::print("Livox SDK version ", _sdkversion.major, _sdkversion.minor, _sdkversion.patch);

    // memset(devices, 0, sizeof(devices));
    // Use assignment or value-initialization instead
    for (int i = 0; i < kMaxLidarCount; ++i)
    {
        devices[i].device_state = kDeviceStateDisconnect;
        devices[i].handle = 0;
        devices[i].firstLidarTimestamp = 0;
        devices[i].lastLidarTimestamp = 0;
    }

    /** Set the callback function receiving broadcast message from Livox LiDAR. */
    SetBroadcastCallback(OnDeviceBroadcast);

    /** Set the callback function called when device state change,
     * which means connection/disconnection and changing of LiDAR state.
     */
    SetDeviceStateUpdateCallback(OnDeviceInfoChange);

    if (!Start())
    {
        Uninit();
        return false;
    }

    isRunning = true;

    py::print("Livox has started.");

    return true;
}

py::list get_devices()
{
    py::list devices_list;

    for (int i = 0; i < kMaxLidarCount; ++i)
    {
        if (devices[i].device_state != kDeviceStateDisconnect)
        {
            py::dict device;
            device["handle"] = devices[i].handle;
            device["device_state"] = (int)devices[i].device_state;
            // device["info"] = devices[i].info;
            devices_list.append(device);
        }
    }

    return devices_list;
}

std::tuple<py::array_t<float>, uint64_t, uint64_t> get_frame(int device_handle)
{

    std::vector<float> _empty_data = {};
    py::array_t<float> empty_data = py::array_t<float>(_empty_data.size(), _empty_data.data());

    if (device_handle >= kMaxLidarCount || devices[device_handle].device_state == kDeviceStateDisconnect)
    {
        return std::make_tuple(empty_data, 0, 0);
    }

    if (!isRunning)
    {
        return std::make_tuple(empty_data, 0, 0);
    }

    // Assume that startFrameRecording is false: get_frame is the master of this flag
    if (startFrameRecording[device_handle])
    {
        return std::make_tuple(empty_data, 0, 0);
    }

    // Wait until the buffer is cleaned
    while (frameReady[device_handle])
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    startFrameRecording[device_handle] = true;

    // Wait until the frame is ready
    while (!frameReady[device_handle])
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    long unsigned int _size = devices[device_handle].vec.size();

    // Store the point cloud data in a numpy array
    std::vector<float> _data(_size * 4);
    std::vector<ssize_t> shape = {(long int)_size, 4};

    if (debug)
    {
        py::print("Size of the point cloud: ", _size);
    }

    for (long unsigned int i = 0, j = 0; i < _size && j + 3 < _size * 4; i++, j += 4)
    {
        _data[j] = devices[device_handle].vec[i].x / 1000.0;
        _data[j + 1] = devices[device_handle].vec[i].y / 1000.0;
        _data[j + 2] = devices[device_handle].vec[i].z / 1000.0;
        _data[j + 3] = devices[device_handle].vec[i].reflectivity / 255.0;
    }

    py::array_t<float> pcd = py::array_t<float>(shape, _data.data());

    // Reset the frame recording flags
    startFrameRecording[device_handle] = false;

    return std::make_tuple(pcd, devices[device_handle].firstLidarTimestamp, devices[device_handle].lastLidarTimestamp);
}

void stop()
{
    for (int i = 0; i < kMaxLidarCount; ++i)
    {
        if (devices[i].device_state == kDeviceStateSampling)
        {
            /** Stop the sampling of Livox LiDAR. */
            LidarStopSampling(devices[i].handle, OnStopSampleCallback, NULL);
        }
    }

    /** Uninitialize Livox-SDK. */
    Uninit();

    isRunning = false;

    py::print("Livox has stopped.");
}

// Wrapping the C++ API with pybind11
PYBIND11_MODULE(mid70grabber, m)
{
    m.doc() = "Python bindings for the C++ Livox MID-70 API that captures point cloud data";
    m.def("start", &start, "Start the livox pcd recording. Returns true if started successfully, false otherwise", py::arg("broadcast_list"), py::arg("frame_duration_ms"));
    m.def("get_devices", &get_devices, "Return the list of connected devices");
    m.def("get_frame", &get_frame, "Return the captured frame of device with the inserted handler as Nx4 numpy array", py::arg("device_handle"));
    m.def("stop", &stop, "Stop the livox pcd recording");
}
