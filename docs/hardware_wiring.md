# Hardware Wiring Notes

## Power

- Use a 12 V, 5 A main adapter.
- Feed servos through a high-current DC-DC buck converter set to the servo rated voltage, commonly 5 V or 6 V.
- Do not power pan/tilt servos from the Raspberry Pi 5 V pin.
- Feed the Raspberry Pi and sensors from a separate 5 V, 3 A or stronger regulator.
- Tie Raspberry Pi, servo supply, and sensor supply grounds together.
- Add a bulk electrolytic capacitor near the servo power rail and size it by measurement.

## GPIO Map

| Signal | GPIO | Notes |
| --- | ---: | --- |
| I2C SDA | 2 | VL53L0X ToF and PCA9685 |
| I2C SCL | 3 | VL53L0X ToF and PCA9685 |
| Infrared obstacle | 17 | Digital input, active-low by default |
| Ultrasonic Trig | 22 | Digital output |
| Ultrasonic Echo | 27 | Use divider or level shifter for 5 V echo |
| Limit switch | 25 | Optional falling-edge safety input |
| RGB red | 5 | Optional status LED output |
| RGB green | 6 | Optional status LED output |
| RGB blue | 23 | Optional status LED output |
| Pan PWM fallback | 12 | Only when PCA9685 is not used |
| Tilt PWM fallback | 19 | Only when PCA9685 is not used |
| PCA9685 pan | channel 0 | Preferred servo output |
| PCA9685 tilt | channel 1 | Preferred servo output |

## Sensor Placement

- Mount the ToF sensor on the pan-tilt bracket as close to the camera optical axis as practical.
- Treat ToF as high-confidence only near the image center because its field of view is narrower than the camera.
- Use ultrasonic distance primarily as an occlusion indicator, not as precise target distance.
- Aim the infrared obstacle sensor at the near-field blind spot where the camera/ToF pair can be occluded.

## RGB Status

| Runtime state or alert | RGB color |
| --- | --- |
| IDLE | Off |
| SCAN / CENTERING | Blue |
| DELAY_COMPENSATION | Cyan |
| TRACKING | Green |
| SAFE_HOLD / ERROR | Red |
| LIMITED_RESCAN | Yellow |
| IR obstacle detected | Blue override |
| Tracking lost or rejected by ultrasonic jump guard | Red override |

The RGB pins are configured in `[hardware]`. Defaults match the requested
wiring: red on GPIO 5, green on GPIO 6, and blue on GPIO 23.

The ultrasonic guard compares the camera bbox center with recent distance
samples. If the bbox jumps suddenly while ultrasonic distance stays stable, the
new bbox is treated as a likely false jump and is not passed to servo control.
Tune `safety.ultrasonic_stable_delta_threshold_mm` for the actual mounting and
sensor noise.

## Bring-Up Order

1. Confirm separate power rails and common ground with a multimeter.
2. Run servos with conservative PWM limits before attaching the full bracket.
3. Verify pan and tilt directions.
4. Verify limit switch behavior, if installed.
5. Read ToF, ultrasonic, and infrared samples without moving servos.
6. Run `scripts/simulate_control_loop.py`.
7. Run `iot-edge --config config/settings.toml --run --hardware-servo --hardware-sensors --hardware-status-light`.
8. Tune `Kp`, `Kd`, `deadband_deg`, `max_speed_deg_s`, and `max_accel_deg_s2`.
