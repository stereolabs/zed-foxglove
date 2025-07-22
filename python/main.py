import datetime
import json
import logging
import struct
import time
from math import cos, sin
import argparse

import foxglove

import pyzed.sl as sl
import cv2
import threading
import time
import signal

from foxglove.channels import CompressedImageChannel, PointCloudChannel
from foxglove.schemas import (
    PackedElementField,
    PackedElementFieldNumericType,
    PointCloud,
    Pose,
    Quaternion,
    CompressedImage,
    Vector3,
    PointCloud,
)
from foxglove.websocket import (
    Capability,
)

any_schema = {
    "type": "object",
    "additionalProperties": True,
}

plot_schema = {
    "type": "object",
    "properties": {
        "timestamp": {"type": "number"},
        "y": {"type": "number"},
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--mcap", type=str, default="", help="Output MCAP file name")
parser.add_argument("--ws", action="store_true", help="Enable WebSocket server")
args = parser.parse_args()

zed_list = []
left_list = []
depth_list = []
pt_cloud_list = []
thread_list = []
stop_signal = False

def signal_handler(signal, frame):
    global stop_signal
    stop_signal=True
    time.sleep(0.5)
    exit()

def grab_run(index):
    global stop_signal
    global zed_list
    global left_list
    global depth_list

    img_chan = CompressedImageChannel(topic=f"image_{index}")
    depth_chan = CompressedImageChannel(topic=f"depth_{index}")
    pt_cloud_chan = PointCloudChannel(topic=f"point_cloud_{index}")

    runtime = sl.RuntimeParameters()
    while not stop_signal:
        err = zed_list[index].grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            # Log the images
            zed_list[index].retrieve_image(left_list[index], sl.VIEW.LEFT)
            zed_list[index].retrieve_image(depth_list[index], sl.VIEW.DEPTH)

            # Convert images to JPEG format using OpenCV
            left_img = left_list[index].get_data()
            depth_img = depth_list[index].get_data()

            # Encode left image as JPEG
            _, left_jpeg = cv2.imencode('.jpg', left_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # For depth image, convert to 8-bit grayscale first, then encode as JPEG
            _, depth_jpeg = cv2.imencode('.jpg', depth_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Or use typed channels directly to get better type checking
            img_chan.log(
                CompressedImage(
                    data=left_jpeg.tobytes(),
                    format="jpeg",
                ),
            )
            depth_chan.log(
                CompressedImage(
                    data=depth_jpeg.tobytes(),
                    format="jpeg",
                ),
            )

            # Log point cloud information
            zed_list[index].retrieve_measure(
                pt_cloud_list[index],
                sl.MEASURE.XYZRGBA,
            )

            # Convert the point cloud to a format suitable for Foxglove
            point_cloud = make_point_cloud(index, pt_cloud_list[index])

            pt_cloud_chan.log(point_cloud)

    zed_list[index].close()

def main() -> None:
    global stop_signal
    global zed_list
    global left_list
    global depth_list
    global thread_list
    signal.signal(signal.SIGINT, signal_handler)

    foxglove.set_log_level(logging.DEBUG)

    # listener = ExampleListener()

    server = None
    if args.ws:
        server = foxglove.start_server(
            # server_listener=listener,
            capabilities=[Capability.ClientPublish],
            supported_encodings=["json"],
        )

    mcap = None
    if args.mcap:
        mcap = foxglove.open_mcap(args.mcap, allow_overwrite=True)

    print("Running...")
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.AUTO
    init.camera_fps = 30  # The framerate is lowered to avoid any USB3 bandwidth issues
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

    # List and open cameras
    name_list = []
    cameras = sl.Camera.get_device_list()
    index = 0
    for cam in cameras:
        init.set_from_serial_number(cam.serial_number)
        name_list.append("ZED {}".format(cam.serial_number))
        print("Opening {}".format(name_list[index]))
        zed_list.append(sl.Camera())
        left_list.append(sl.Mat(width=0, height=0, mat_type=sl.MAT_TYPE.U8_C4))
        depth_list.append(sl.Mat(width=0, height=0, mat_type=sl.MAT_TYPE.U8_C4))
        pt_cloud_list.append(sl.Mat(width=0, height=0, mat_type=sl.MAT_TYPE.F32_C4))
        status = zed_list[index].open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            zed_list[index].close()
        index = index + 1

    # Start camera threads
    for index in range(0, len(zed_list)):
        if zed_list[index].is_opened():
            thread_list.append(threading.Thread(target=grab_run, args=(index,)))
            thread_list[index].start()

    # Display camera images
    try:
        key = ''
        while key != 113:  # for 'q' key
            key = cv2.waitKey(10)

    except KeyboardInterrupt:
        global stop_signal
        stop_signal = True
        time.sleep(0.5)
        if server is not None:
            server.stop()
        if mcap is not None:
            mcap.close()

    #Stop the threads
    stop_signal = True
    for index in range(0, len(thread_list)):
        thread_list[index].join()


def make_point_cloud(zed_id: int, point_cloud: sl.Mat) -> PointCloud:
    """
    https://foxglove.dev/blog/visualizing-point-clouds-with-custom-colors
    """
    f32 = PackedElementFieldNumericType.Float32
    u8 = PackedElementFieldNumericType.Uint8

    return PointCloud(
        frame_id=f"zed_{zed_id}",
        pose=Pose(
            position=Vector3(x=0, y=0, z=0),
            orientation=Quaternion(x=0, y=0, z=0, w=1),
        ),
        point_stride=16,  # 4 fields * 4 bytes
        fields=[
            PackedElementField(name="x", offset=0, type=f32),
            PackedElementField(name="y", offset=4, type=f32),
            PackedElementField(name="z", offset=8, type=f32),
            PackedElementField(name="red", offset=12, type=u8),
            PackedElementField(name="green", offset=13, type=u8),
            PackedElementField(name="blue", offset=14, type=u8),
            PackedElementField(name="alpha", offset=15, type=u8),
        ],
        data=bytes(point_cloud.get_data()),
    )

if __name__ == "__main__":
    main()
