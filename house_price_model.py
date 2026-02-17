# ==========================================
# TASK 01
# Linear Regression from Scratch (Pure Python)
# ==========================================

# Sample Dataset
# [sqft, bedrooms, bathrooms, price]
data = [
    [1500, 3, 2, 250000],
    [2000, 4, 3, 350000],
    [1200, 2, 1, 180000],
    [1800, 3, 2, 300000],
    [2200, 4, 3, 400000],
    [1700, 3, 2, 280000],
    [2400, 5, 4, 450000],
    [1600, 3, 2, 270000]
]

# Learning rate and iterations
learning_rate = 0.00000001
epochs = 1000

# Initialize weights
b0 = 0   # Intercept
b1 = 0   # sqft weight
b2 = 0   # bedrooms weight
b3 = 0   # bathrooms weight

# Training using Gradient Descent
for _ in range(epochs):
    for row in data:
        sqft = row[0]
        bedrooms = row[1]
        bathrooms = row[2]
        actual_price = row[3]

        # Prediction
        predicted_price = b0 + b1*sqft + b2*bedrooms + b3*bathrooms

        # Error
        error = predicted_price - actual_price

        # Update weights
        b0 = b0 - learning_rate * error
        b1 = b1 - learning_rate * error * sqft
        b2 = b2 - learning_rate * error * bedrooms
        b3 = b3 - learning_rate * error * bathrooms


# Print Model
print("Model Trained Successfully!\n")
print("Intercept (b0):", b0)
print("Weight for sqft (b1):", b1)
print("Weight for bedrooms (b2):", b2)
print("Weight for bathrooms (b3):", b3)


# Predict New House
new_sqft = 2000
new_bedrooms = 3
new_bathrooms = 2

predicted = b0 + b1*new_sqft + b2*new_bedrooms + b3*new_bathrooms

print("\nPredicted Price for 2000 sqft, 3 bed, 2 bath:", predicted)
