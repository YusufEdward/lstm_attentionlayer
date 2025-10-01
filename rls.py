import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data.csv')

# Extract relevant columns
voltage = data['Voltage (V)'].values
current = data['Current (mA)'].values / 1000  # Convert mA to A
status = data['Status'].values

# Initialize RLS parameters
n = 1  # Number of parameters to estimate (just resistance)
lambda_ = 0.99  # Forgetting factor (0 < lambda <= 1)
P = np.eye(n) * 100  # Initial covariance matrix
theta = np.zeros(n)  # Initial parameter estimate (resistance)

# Storage for results
estimated_resistance = []
time_points = []

# Process the data
for i in range(1, len(data)):
    # Only process transitions from REST to DRAIN or DRAIN to REST
    if status[i] != status[i-1]:
        # Calculate voltage and current differences
        delta_V = voltage[i] - voltage[i-1]
        delta_I = current[i] - current[i-1]
        
        # Avoid division by zero and small changes
        if abs(delta_I) > 0.01:  # Threshold to avoid noise
            # RLS update
            x = np.array([delta_I]).reshape(-1, 1)
            y = delta_V
            
            # Compute gain
            K = P @ x / (lambda_ + x.T @ P @ x)
            
            # Update parameter estimate
            theta = theta + K * (y - x.T @ theta)
            
            # Update covariance matrix
            P = (P - K @ x.T @ P) / lambda_
            
            # Store results
            estimated_resistance.append(theta[0])
            time_points.append(i)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time_points, estimated_resistance, 'b-', label='Estimated Resistance')
plt.xlabel('Data Point Index')
plt.ylabel('Resistance (Ohms)')
plt.title('Recursive Least Squares Estimation of Battery Internal Resistance')
plt.grid(True)
plt.legend()
plt.show()

# Final resistance estimate
final_resistance = theta[0]
print(f"Final estimated internal resistance")
print(final_resistance)
