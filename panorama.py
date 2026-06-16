import time
import board
import busio
import os
import cv2
import sys
import subprocess
import shutil
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from picamera2 import Picamera2

OUTPUT_DIR = "panorama_output"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)

print("initializing servos...")
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

pan_servo = servo.Servo(pca.channels[0], min_pulse=600, max_pulse=2400)
tilt_servo = servo.Servo(pca.channels[1], min_pulse=600, max_pulse=2400)

cascPath = "model/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
if faceCascade.empty():
    print("Error: cannot load model (haarcascade_frontalface_default.xml)")
    sys.exit()

CALC_W, CALC_H = 180, 320

print("starting camera...")
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)})
picam2.configure(config)
picam2.start()
picam2.set_controls({
    "AwbEnable": False,            
    "ColourGains": (1.2, 1.9),
    "ExposureValue": 1.1    
})
time.sleep(2)


pan_angles = [80, 85, 90, 95, 100]  
tilt_angles = [100, 105, 110, 115]  

print("\n=== starting panorama capture ===")

try:
    for row, t_angle in enumerate(tilt_angles):
        tilt_servo.angle = t_angle
        time.sleep(0.5) 
        
        for col, p_angle in enumerate(pan_angles):
            pan_servo.angle = p_angle
            
            time.sleep(1.0)

            frame = picam2.capture_array()
            if frame is None:
                continue

            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            small_frame = cv2.resize(rotated_frame, (CALC_W, CALC_H))
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(15, 15) 
            )

            filename = f"img_p{p_angle}_t{t_angle}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)

            if len(faces) > 0:
                print(f"shooting [{row+1}/3, {col+1}/3] - {filename} ...", end=" \n")
                cv2.imwrite(filepath, rotated_frame)
            else:
                print(f"no face detected (p:{p_angle}, t:{t_angle}) - skipping. \n")
        
    print("done!")
    print("\nRsync to Pi 5 ...")
    
    rsync_cmd = [
        "rsync", "-avz", 
        f"{OUTPUT_DIR}/", 
        "pi@192.168.0.41:/home/pi/receiver/" 
    ]

    result = subprocess.run(rsync_cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("succeedfully transferred images to Pi 5!")
    else:
        print(f"error during rsync: {result.stderr}")

except KeyboardInterrupt:
    print("\interrupt received, stopping...")

finally:
    print("\nending program, resetting servos to center...")
    pan_servo.angle = 90
    tilt_servo.angle = 90
    time.sleep(1)
    
    picam2.stop()  
    pca.deinit()  
    print("program ended successfully!")