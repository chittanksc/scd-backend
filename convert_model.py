"""
Convert old Keras model to TensorFlow 2.13 compatible format
Run this script once to convert your skin_cancer_cnn.h5 model
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import tensorflow as tf
from tensorflow import keras
import h5py

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

# Paths
old_model_path = "skin_cancer_cnn.h5"
new_model_path = "skin_cancer_cnn_converted.h5"

print(f"\nConverting model: {old_model_path}")

# Method 1: Try loading with custom objects and legacy format
try:
    print("\nAttempt 1: Loading with legacy loader...")
    
    # Load weights and architecture separately
    with h5py.File(old_model_path, 'r') as f:
        # Check model structure
        if 'model_config' in f.attrs:
            import json
            config = json.loads(f.attrs['model_config'])
            print("Model config found")
            
            # Fix batch_shape to input_shape in config
            if 'config' in config and 'layers' in config['config']:
                for layer in config['config']['layers']:
                    if 'batch_shape' in layer['config']:
                        batch_shape = layer['config']['batch_shape']
                        # Convert batch_shape to input_shape
                        layer['config']['input_shape'] = batch_shape[1:]
                        del layer['config']['batch_shape']
                        print(f"Fixed layer: {layer['class_name']}")
            
            # Reconstruct model from fixed config
            model = keras.Model.from_config(config['config'])
            print("Model reconstructed from config")
            
            # Load weights
            model.load_weights(old_model_path)
            print("Weights loaded successfully")
            
            # Save in new format
            model.save(new_model_path)
            print(f"\n✅ SUCCESS! Model converted and saved to: {new_model_path}")
            print(f"\nModel summary:")
            model.summary()
            
            # Test prediction
            import numpy as np
            test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
            pred = model.predict(test_input, verbose=0)
            print(f"\nTest prediction shape: {pred.shape}")
            print(f"Test prediction: {pred}")
            
except Exception as e:
    print(f"❌ Method 1 failed: {e}")
    print("\nTrying alternative method...")
    
    # Method 2: Manual reconstruction
    try:
        print("\nAttempt 2: Manual model reconstruction...")
        print("\nPlease provide your model architecture code.")
        print("You'll need to:")
        print("1. Recreate the model architecture in a separate script")
        print("2. Load only the weights from the old model")
        print("3. Save in the new format")
        print("\nExample:")
        print("""
# Create your model architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential([
    # Add your layers here matching your original architecture
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    # ... rest of your layers
    Dense(1, activation='sigmoid')  # or Dense(2, activation='softmax')
])

# Load weights from old model
model.load_weights('skin_cancer_cnn.h5')

# Save in new format
model.save('skin_cancer_cnn_converted.h5')
""")
        
    except Exception as e2:
        print(f"❌ Method 2 info displayed")

print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print(f"1. If conversion succeeded, update config.py to use:")
print(f"   MODEL_PATH = 'skin_cancer_cnn_converted.h5'")
print(f"\n2. If conversion failed, you need to:")
print(f"   - Recreate your model architecture")
print(f"   - Load weights with model.load_weights()")
print(f"   - Save with model.save()")
print("="*60)
