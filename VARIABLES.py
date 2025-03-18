import os

print("🔍 Available Environment Variables:")
for key, value in os.environ.items():
    print(f"{key}: {value[:5]}*****")  # Mask values for security

