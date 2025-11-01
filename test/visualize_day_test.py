# test/visualize_response.py
import matplotlib.pyplot as plt
import numpy as np

time_axis = np.arange(0, 600, 5)
temp = 0.45 + 0.05*np.sin(time_axis/60)
u_fan = 0.2 + 0.05*np.sin(time_axis/120)
u_heater = 0.3 + 0.1*np.cos(time_axis/90)
u_led = 0.1 + 0.03*np.sin(time_axis/45)

plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(time_axis, temp, label="Temperature (norm)")
plt.axhline(0.55, color='r', linestyle='--', label="Target")
plt.legend(); plt.grid(True); plt.ylabel("Temp (0~1)")

plt.subplot(2,1,2)
plt.plot(time_axis, u_fan, label="Fan")
plt.plot(time_axis, u_heater, label="Heater")
plt.plot(time_axis, u_led, label="LED")
plt.legend(); plt.grid(True); plt.ylabel("Control Signal")
plt.xlabel("Time [s]")
plt.tight_layout(); plt.show()
