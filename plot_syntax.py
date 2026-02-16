# Plot the results
plt.figure(figsize=(10, 5))

# Plot position vs time
plt.subplot(1, 2, 1)
plt.plot(t, y[:, 0], 'b-', linewidth=2)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position y (m)', fontsize=12)
plt.title('Falling Body - Position vs Time', fontsize=12)
plt.grid(True, alpha=0.3)

# Plot velocity vs time
plt.subplot(1, 2, 2)
plt.plot(t, y[:, 1], 'r-', linewidth=2)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Velocity v (m/s)', fontsize=12)
plt.title('Falling Body - Velocity vs Time', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()