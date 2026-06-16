import cv2
import sys
import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from picamera2 import Picamera2

print("initializing ...")
i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50

pan_servo = servo.Servo(pca.channels[0], min_pulse=600, max_pulse=2400)
tilt_servo = servo.Servo(pca.channels[1], min_pulse=600, max_pulse=2400)

pan_angle = 90
tilt_angle = 90
pan_servo.angle = pan_angle
tilt_servo.angle = tilt_angle

cascPath = "model/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

if faceCascade.empty():
    print("Error: cannot load model (haarcascade_frontalface_default.xml)")
    sys.exit()

CAM_W, CAM_H = 960, 1280  

MAIN_W, MAIN_H = 1280, 960  

CALC_W, CALC_H = 320, 240   

RATIO_X = MAIN_W // CALC_W  
RATIO_Y = MAIN_H // CALC_H  

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (CAM_W, CAM_H), "format": "BGR888"})
picam2.configure(config)
picam2.start()

center_x = MAIN_W // 2
center_y = MAIN_H // 2

deadzone = 50    
Kp_pan = 0.04    
Kp_tilt = 0.04   

try:
    while True:
        frame = picam2.capture_array()
        if frame is None:
            continue

        rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        small_frame = cv2.resize(rotated_frame, (CALC_W, CALC_H))
        output_frame = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2RGB)
        
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(15, 15) 
        )

        if len(faces) > 0:
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            small_x, small_y, small_w, small_h = largest_face

            x = small_x * RATIO_X
            y = small_y * RATIO_Y
            w = small_w * RATIO_X
            h = small_h * RATIO_Y

            face_cx = x + w // 2
            face_cy = y + h // 2

            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(output_frame, (face_cx, face_cy), 5, (0, 0, 255), -1)

            error_x = face_cx - center_x
            error_y = face_cy - center_y

            if abs(error_x) > deadzone:
                pan_step = error_x * Kp_pan
                pan_angle -= pan_step 
            
            if abs(error_y) > deadzone:
                tilt_step = error_y * Kp_tilt
                tilt_angle += tilt_step

            pan_angle = max(0, min(180, pan_angle))
            tilt_angle = max(0, min(180, tilt_angle))

            pan_servo.angle = pan_angle
            tilt_servo.angle = tilt_angle

        cv2.imshow("Face Tracking", output_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\ninterrupted...")

finally:
    cv2.destroyAllWindows()
    picam2.stop()
    
    pan_servo.angle = 90
    tilt_servo.angle = 90
    time.sleep(0.5)
    
    pca.deinit()
    print("finished successfully!")