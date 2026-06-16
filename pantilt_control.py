import time
import board
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

i2c = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c)
pca.frequency = 50  

pan_servo = servo.Servo(pca.channels[0], min_pulse=600, max_pulse=2400)
tilt_servo = servo.Servo(pca.channels[1], min_pulse=600, max_pulse=2400)

pan_angle = 90
tilt_angle = 90
step = 10 

pan_servo.angle = pan_angle
tilt_servo.angle = tilt_angle
time.sleep(1)

print("\n=== control ===")
print("W: up | S: down | A: left | D: right")
print("Q: quit")
print("============================")

try:
    while True:
        cmd = input("input: ").strip().lower()
        
        if cmd == 'q':
            break
        elif cmd == 'w':
            tilt_angle -= step
        elif cmd == 's':
            tilt_angle += step
        elif cmd == 'a':
            pan_angle += step
        elif cmd == 'd':
            pan_angle -= step
        else:
            continue
            
        pan_angle = max(0, min(180, pan_angle))
        tilt_angle = max(0, min(180, tilt_angle))
        
        pan_servo.angle = pan_angle
        tilt_servo.angle = tilt_angle
        
        print(f"now -> Pan: {pan_angle} | Tilt: {tilt_angle}")

except KeyboardInterrupt:
    print("\ninterrupted...")
finally:
    pca.deinit()
    print("finished successfully!")