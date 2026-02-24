import depthai as dai
import cv2
import time

pipeline = dai.Pipeline()
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

with dai.Device(pipeline) as device:
    print("Connected:", device.getDeviceInfo())
    q = device.getOutputQueue("depth", 4, False)
    while True:
        inDepth = q.tryGet()
        if inDepth is not None:
            frame = inDepth.getFrame()
            cv2.imshow("depth", frame)
        if cv2.waitKey(1) == ord('q'):
            break