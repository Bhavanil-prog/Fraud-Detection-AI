import RPi.GPIO as GPIO
import time

# --- GPIO Pin Setup ---
# Set the GPIO mode to BCM (Broadcom SOC channel numbering)
GPIO.setmode(GPIO.BCM)

# Define GPIO pins for Trigger (Trig) and Echo
# You can choose any available GPIO pins on your Raspberry Pi
GPIO_TRIGGER = 17 # e.g., GPIO 17
GPIO_ECHO = 27    # e.g., GPIO 27

# Set up the GPIO pins
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)  # Trigger pin as output
GPIO.setup(GPIO_ECHO, GPIO.IN)     # Echo pin as input

# --- Ultrasonic Sensor Functions ---
def measure_distance():
    # Set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER, True)
    # Set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)

    # Measure the time it takes for the echo to return
    pulse_start = time.time()
    pulse_end = time.time()

    # Save start time
    while GPIO.input(GPIO_ECHO) == 0:
        pulse_start = time.time()
        # Add a timeout to prevent infinite loop if sensor fails
        if time.time() - pulse_start > 0.1: # 0.1 second timeout
            return -1 # Return -1 for error

    # Save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        pulse_end = time.time()
        # Add a timeout
        if time.time() - pulse_end > 0.1: # 0.1 second timeout
            return -1 # Return -1 for error

    pulse_duration = pulse_end - pulse_start

    # Calculate distance (speed of sound is 34300 cm/s or 343 m/s)
    # Distance = (Time * Speed of Sound) / 2 (because sound travels to object and back)
    distance = pulse_duration * 17150  # 34300 / 2 = 17150
    distance = round(distance, 2) # Round to 2 decimal places

    return distance

# --- Main Program Loop ---
try:
    print("Water Level Monitoring Started...")
    # Optional: Initial delay to allow sensor to settle
    time.sleep(2)

    # Set the total height of your tank/container in cm
    # Measure this from the sensor's position (at the top) to the bottom of the tank
    total_tank_height_cm = 30 # Example: 30 cm tank height

    while True:
        dist_to_water_surface = measure_distance()

        if dist_to_water_surface != -1: # Check if measurement was successful
            # Calculate water level
            # Water Level = Total Tank Height - Distance from Sensor to Water Surface
            water_level_cm = total_tank_height_cm - dist_to_water_surface

            # Ensure water_level_cm is not negative or exceeding tank height due to errors/fluctuations
            if water_level_cm < 0:
                water_level_cm = 0
            elif water_level_cm > total_tank_height_cm:
                water_level_cm = total_tank_height_cm

            # Calculate water level percentage
            water_level_percentage = (water_level_cm / total_tank_height_cm) * 100
            water_level_percentage = round(water_level_percentage, 2)

            print(f"Distance to water surface: {dist_to_water_surface} cm")
            print(f"Water Level: {water_level_cm} cm (approx. {water_level_percentage}%)")

            # --- Logic for alerts (e.g., using LEDs or a buzzer) ---
            if water_level_percentage < 10:
                print("--- WARNING: Water level critically LOW! ---")
                # Add code here to trigger an LED, buzzer, or notification
            elif water_level_percentage > 90:
                print("--- ALERT: Water level nearly FULL! ---")
                # Add code here to trigger an LED, buzzer, or notification
            elif water_level_percentage > 50:
                print("Water level is good.")
            else:
                print("Water level is moderate.")

        else:
            print("Measurement failed, retrying...")

        time.sleep(5) # Read every 5 seconds (adjust as needed)

except KeyboardInterrupt:
    print("\nProgram terminated by user.")

finally:
    GPIO.cleanup() # Clean up GPIO settings on exit