import datetime
import json
import logging
import struct
import time
from math import cos, sin

import foxglove
import numpy as np

import pyzed.sl as sl
import cv2
import threading
import time
import signal

from foxglove.channels import RawImageChannel
from foxglove.schemas import (
    PackedElementField,
    PackedElementFieldNumericType,
    PointCloud,
    Pose,
    Quaternion,
    RawImage,
    Vector3,
)
from foxglove.websocket import (
    Capability,
    ChannelView,
    Client,
    ClientChannel,
    ServerListener,
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

zed_list = []
left_list = []
depth_list = []
timestamp_list = []
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
    global timestamp_list
    global left_list
    global depth_list

    img_chan = RawImageChannel(topic=f"/image_{index}")
    depth_chan = RawImageChannel(topic=f"/depth_{index}")

    runtime = sl.RuntimeParameters()
    while not stop_signal:
        err = zed_list[index].grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            # If you want to add more computation here and to keep maximum performance with multiple cameras:
            # 1. Minimize Python operations in the loop to avoid python to block the GIL
            # 2. Pre-allocate objects and arrays to reduce memory allocations
            # 3. Rely on the pyzed library which is optimized for performance
            zed_list[index].retrieve_image(left_list[index], sl.VIEW.LEFT)
            zed_list[index].retrieve_image(depth_list[index], sl.VIEW.DEPTH)

            # Or use typed channels directly to get better type checking
            img_chan.log(
                RawImage(
                    data=left_list[index].get_data().tobytes(),
                    width=left_list[index].get_width(),
                    height=left_list[index].get_height(),
                    step=left_list[index].get_step_bytes(),
                    encoding="bgra8",
                ),
            )
            depth_chan.log(
                RawImage(
                    data=depth_list[index].get_data().tobytes(),
                    width=depth_list[index].get_width(),
                    height=depth_list[index].get_height(),
                    step=depth_list[index].get_step_bytes(),
                    encoding="bgra8",  # 32-bit float for depth
                ),
            )
            timestamp_list[index] = zed_list[index].get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns


    zed_list[index].close()


class ExampleListener(ServerListener):
    def __init__(self) -> None:
        # Map client id -> set of subscribed topics
        self.subscribers: dict[int, set[str]] = {}

    def has_subscribers(self) -> bool:
        return len(self.subscribers) > 0

    def on_subscribe(
        self,
        client: Client,
        channel: ChannelView,
    ) -> None:
        """
        Called by the server when a client subscribes to a channel.
        We'll use this and on_unsubscribe to simply track if we have any subscribers at all.
        """
        logging.info(f"Client {client} subscribed to channel {channel.topic}")
        self.subscribers.setdefault(client.id, set()).add(channel.topic)

    def on_unsubscribe(
        self,
        client: Client,
        channel: ChannelView,
    ) -> None:
        """
        Called by the server when a client unsubscribes from a channel.
        """
        logging.info(f"Client {client} unsubscribed from channel {channel.topic}")
        self.subscribers[client.id].remove(channel.topic)
        if not self.subscribers[client.id]:
            del self.subscribers[client.id]

    def on_client_advertise(
        self,
        client: Client,
        channel: ClientChannel,
    ) -> None:
        """
        Called when a client advertises a new channel.
        """
        logging.info(f"Client {client.id} advertised channel: {channel.id}")
        logging.info(f"  Topic: {channel.topic}")
        logging.info(f"  Encoding: {channel.encoding}")
        logging.info(f"  Schema name: {channel.schema_name}")
        logging.info(f"  Schema encoding: {channel.schema_encoding}")
        logging.info(f"  Schema: {channel.schema!r}")

    def on_message_data(
        self,
        client: Client,
        client_channel_id: int,
        data: bytes,
    ) -> None:
        """
        This handler demonstrates receiving messages from the client.
        You can send messages from Foxglove app in the publish panel:
        https://docs.foxglove.dev/docs/visualization/panels/publish
        """
        logging.info(f"Message from client {client.id} on channel {client_channel_id}")
        logging.info(f"Data: {data!r}")

    def on_client_unadvertise(
        self,
        client: Client,
        client_channel_id: int,
    ) -> None:
        """
        Called when a client unadvertises a new channel.
        """
        logging.info(f"Client {client.id} unadvertised channel: {client_channel_id}")

def main() -> None:
    global stop_signal
    global zed_list
    global left_list
    global depth_list
    global timestamp_list
    global thread_list
    signal.signal(signal.SIGINT, signal_handler)

    foxglove.set_log_level(logging.DEBUG)

    listener = ExampleListener()

    server = foxglove.start_server(
        server_listener=listener,
        capabilities=[Capability.ClientPublish],
        supported_encodings=["json"],
    )

    print("Running...")
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60  # The framerate is lowered to avoid any USB3 bandwidth issues

    #List and open cameras
    name_list = []
    last_ts_list = []
    cameras = sl.Camera.get_device_list()
    index = 0
    for cam in cameras:
        init.set_from_serial_number(cam.serial_number)
        name_list.append("ZED {}".format(cam.serial_number))
        print("Opening {}".format(name_list[index]))
        zed_list.append(sl.Camera())
        left_list.append(sl.Mat())
        depth_list.append(sl.Mat())
        timestamp_list.append(0)
        last_ts_list.append(0)
        status = zed_list[index].open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            zed_list[index].close()
        index = index +1

    #Start camera threads
    for index in range(0, len(zed_list)):
        if zed_list[index].is_opened():
            thread_list.append(threading.Thread(target=grab_run, args=(index,)))
            thread_list[index].start()
    
    #Display camera images
    try:
        key = ''
        while key != 113:  # for 'q' key
            time.sleep(0.05)

            while not listener.has_subscribers():
                time.sleep(1)
                continue

            key = cv2.waitKey(10)

    except KeyboardInterrupt:
        global stop_signal
        stop_signal=True
        time.sleep(0.5)
        server.stop()

    #Stop the threads
    stop_signal = True
    for index in range(0, len(thread_list)):
        thread_list[index].join()

    cv2.destroyAllWindows()


def make_point_cloud() -> PointCloud:
    """
    https://foxglove.dev/blog/visualizing-point-clouds-with-custom-colors
    """
    point_struct = struct.Struct("<fffBBBB")
    f32 = PackedElementFieldNumericType.Float32
    u32 = PackedElementFieldNumericType.Uint32

    t = time.time()
    points = [(x + cos(t + y / 5), y, 0) for x in range(20) for y in range(20)]
    buffer = bytearray(point_struct.size * len(points))
    for i, point in enumerate(points):
        x, y, z = point
        r = int(255 * (0.5 + 0.5 * x / 20))
        g = int(255 * y / 20)
        b = int(255 * (0.5 + 0.5 * sin(t)))
        a = int(255 * (0.5 + 0.5 * ((x / 20) * (y / 20))))
        point_struct.pack_into(buffer, i * point_struct.size, x, y, z, b, g, r, a)

    return PointCloud(
        frame_id="points",
        pose=Pose(
            position=Vector3(x=0, y=0, z=0),
            orientation=Quaternion(x=0, y=0, z=0, w=1),
        ),
        point_stride=16,  # 4 fields * 4 bytes
        fields=[
            PackedElementField(name="x", offset=0, type=f32),
            PackedElementField(name="y", offset=4, type=f32),
            PackedElementField(name="z", offset=8, type=f32),
            PackedElementField(name="rgba", offset=12, type=u32),
        ],
        data=bytes(buffer),
    )


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    """Convert Euler angles to a rotation quaternion

    See e.g. https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html

    :param roll: rotation around X axis (radians)
    :param pitch: rotation around Y axis (radians)
    :param yaw: rotation around Z axis (radians)
    :returns: a protobuf Quaternion
    """
    roll, pitch, yaw = roll * 0.5, pitch * 0.5, yaw * 0.5

    sin_r, cos_r = sin(roll), cos(roll)
    sin_p, cos_p = sin(pitch), cos(pitch)
    sin_y, cos_y = sin(yaw), cos(yaw)

    w = cos_r * cos_p * cos_y + sin_r * sin_p * sin_y
    x = sin_r * cos_p * cos_y - cos_r * sin_p * sin_y
    y = cos_r * sin_p * cos_y + sin_r * cos_p * sin_y
    z = cos_r * cos_p * sin_y - sin_r * sin_p * cos_y

    return Quaternion(x=x, y=y, z=z, w=w)


if __name__ == "__main__":
    main()