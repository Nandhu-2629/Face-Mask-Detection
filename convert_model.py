
import json
import h5py
import shutil

source = "django_project/face_mask_model.h5"
backup = "django_project/face_mask_model_backup.h5"
output = "django_project/face_mask_model_fixed.h5"

# Preserve the original model
shutil.copy2(source, backup)

# Read the saved model configuration
with h5py.File(source, "r") as f:
    config = json.loads(f.attrs["model_config"])

# Convert newer Keras configuration to older format
for layer in config["config"]["layers"]:
    layer_config = layer["config"]

    if layer["class_name"] == "InputLayer":
        layer_config["batch_input_shape"] = layer_config.pop("batch_shape")

    if isinstance(layer_config.get("dtype"), dict):
        layer_config["dtype"] = layer_config["dtype"]["config"]["name"]

# Save the updated configuration and preserve all weights
with h5py.File(source, "r") as src:
    with h5py.File(output, "w") as dst:
        for key, value in src.attrs.items():
            if key == "model_config":
                dst.attrs[key] = json.dumps(config)
            else:
                dst.attrs[key] = value

        for key in src.keys():
            src.copy(key, dst)

print("Model configuration converted.")
print("Original model preserved.")
print("New model:", output)
