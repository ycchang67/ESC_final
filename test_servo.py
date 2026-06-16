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

if __name__ == '__main__':

    print("start testing servo...")
    try:
        while True:
                tilt_servo.angle = 90
                time.sleep(1)
                tilt_servo.angle = 45
                time.sleep(1)
                tilt_servo.angle = 135
                time.sleep(1)
                pan_servo.angle = 90
                time.sleep(1)
                pan_servo.angle = 45
                time.sleep(1)
                pan_servo.angle = 135
                time.sleep(1)

    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting...")
    finally:
        pca.deinit()
        print("finished successfully!")