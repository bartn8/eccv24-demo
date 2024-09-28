# Livox-SDK/CMakelists.txt: complie with PIC
# Add -fPIC flag to all targets
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

g++ -O3 -Wall -shared -std=c++11 -fPIC $(python -m pybind11 --includes) -c mid70grabber.cpp -o mid70grabber.o
g++ -shared -o mid70grabber$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))') mid70grabber.o -L/usr/local/lib -llivox_sdk_static

cp mid70grabber$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))') ../vpp
cp mid70grabber$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))') ../vppdc