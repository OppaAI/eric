import depthai as dai

def main():
    print("Searching for all available OAK devices...\n")
    
    # Get list of available devices (bootloader or unbooted)
    device_infos = dai.Device.getAllAvailableDevices()  # or dai.DeviceBootloader.getAllAvailableDevices() in some versions
    
    if not device_infos:
        print("No OAK devices found. Check connection, lsusb, or DepthAI installation.")
        return

    print(f"Found {len(device_infos)} device(s):")
    for info in device_infos:
        # Use method instead of attribute
        mx_id = info.getMxId() if hasattr(info, 'getMxId') else info.getDeviceId()  # fallback
        state_str = str(info.state).split('.')[-1] if hasattr(info, 'state') else "Unknown"
        
        print(f"  - Name: {info.name}")
        print(f"  - MXID / Device ID: {mx_id}")
        print(f"  - State: {state_str}")
        print(f"  - Connection type: {info.connection if hasattr(info, 'connection') else 'Unknown'}")
        print("")

    # Connect to the first available device and query real USB speed
    print("\nConnecting to the first device to query USB speed...")
    try:
        with dai.Device() as device:  # auto-picks first available
            usb_speed = device.getUsbSpeed()
            
            print("\n=== USB Speed Report (real negotiated speed) ===")
            speed_map = {
                dai.UsbSpeed.UNKNOWN:    "Unknown / not connected",
                dai.UsbSpeed.LOW:        "USB 1.x Low Speed (~1.5 Mbps)",
                dai.UsbSpeed.FULL:       "USB 1.x Full Speed (~12 Mbps)",
                dai.UsbSpeed.HIGH:       "USB 2.0 High Speed (~480 Mbps) ← fallback / limited",
                dai.UsbSpeed.SUPER:      "USB 3.x SuperSpeed (~5 Gbps) ← good",
                dai.UsbSpeed.SUPER_PLUS: "USB 3.x SuperSpeed+ (~10 Gbps) ← excellent"
            }
            print(f"Device reports: {usb_speed}")
            print(f"→ Interpreted: {speed_map.get(usb_speed, 'Unexpected value')}")
            
            # Bonus info
            print("\nConnected cameras:", device.getConnectedCameras())
            print("Device name:", device.getDeviceInfo().name if hasattr(device.getDeviceInfo(), 'name') else "N/A")
    
    except Exception as e:
        print(f"Failed to connect and query speed: {e}")
        print("Try: pip install --upgrade depthai")
        print("Or check dmesg | grep usb for errors.")

if __name__ == "__main__":
    main()
