import serial
import json
import time

PORT = '/dev/ttyTHS1'
BAUD = 115200

def send_cmd(ser, data):
    cmd = (json.dumps(data) + '\n').encode('utf-8')
    for byte in cmd:
        ser.write(bytes([byte]))
        time.sleep(0.001)

def main():
    print(f"Connecting to {PORT} at {BAUD} baud...")
    ser = serial.Serial(PORT, BAUD, timeout=2, rtscts=False, xonxoff=False)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    print("Connected!\n")

    # Test 1: LED lights on (IO4=base light, IO5=head light)
    print("Test 1: LED lights ON")
    send_cmd(ser, {"T": 132, "IO4": 255, "IO5": 255})
    time.sleep(2)

    # Test 2: LED lights off
    print("Test 2: LED lights OFF")
    send_cmd(ser, {"T": 132, "IO4": 0, "IO5": 0})
    time.sleep(1)

    # Test 3: Move forward (negative = forward based on your test)
    print("Test 3: Move FORWARD for 2 seconds")
    send_cmd(ser, {"T": 1, "L": -0.5, "R": -0.5})
    time.sleep(2)

    # Stop
    print("Stopping...")
    send_cmd(ser, {"T": 1, "L": 0, "R": 0})
    time.sleep(1)

    # Test 4: Move backward
    print("Test 4: Move BACKWARD for 2 seconds")
    send_cmd(ser, {"T": 1, "L": 0.5, "R": 0.5})
    time.sleep(2)

    # Stop
    print("Stopping...")
    send_cmd(ser, {"T": 1, "L": 0, "R": 0})
    time.sleep(1)

    # Test 5: Turn left
    print("Test 5: Turn LEFT for 1 second")
    send_cmd(ser, {"T": 1, "L": 0.5, "R": -0.5})
    time.sleep(1)

    # Stop
    print("Stopping...")
    send_cmd(ser, {"T": 1, "L": 0, "R": 0})
    time.sleep(1)

    # Test 6: Turn right
    print("Test 6: Turn RIGHT for 1 second")
    send_cmd(ser, {"T": 1, "L": -0.5, "R": 0.5})
    time.sleep(1)

    # Stop
    print("Stopping...")
    send_cmd(ser, {"T": 1, "L": 0, "R": 0})
    time.sleep(1)

    # Test 7: LED flash
    print("Test 7: LED flash 3 times")
    for i in range(3):
        send_cmd(ser, {"T": 132, "IO4": 255, "IO5": 255})
        time.sleep(0.3)
        send_cmd(ser, {"T": 132, "IO4": 0, "IO5": 0})
        time.sleep(0.3)

    print("\nAll tests done!")
    ser.close()

if __name__ == "__main__":
    main()
