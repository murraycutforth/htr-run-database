import os
import json
import shutil

# Define the original directory and JSON file
original_dir = "/p/gpfs1/cutforth1/PSAAP/debug_5M"
json_file = "GG-combustor-default-lassen-5M.json"

# Define the base name for the new directories
new_dir_base = "/p/gpfs1/cutforth1/Lassen5MTilesComparison_"

# Define 10 different configurations for tiles and tilesPerRank
configurations = [
    {"tiles": [1, 1, 1], "tilesPerRank": [1, 1, 1]},
    {"tiles": [2, 1, 1], "tilesPerRank": [2, 1, 1]},
    {"tiles": [4, 1, 1], "tilesPerRank": [4, 1, 1]},
    {"tiles": [8, 1, 1], "tilesPerRank": [8, 1, 1]},
    {"tiles": [16, 1, 1], "tilesPerRank": [16, 1, 1]},
    {"tiles": [32, 1, 1], "tilesPerRank": [32, 1, 1]},
    {"tiles": [1, 2, 2], "tilesPerRank": [1, 2, 2]},
    {"tiles": [2, 2, 2], "tilesPerRank": [2, 2, 2]},
    {"tiles": [4, 2, 2], "tilesPerRank": [4, 2, 2]},
    {"tiles": [8, 2, 2], "tilesPerRank": [8, 2, 2]},
    {"tiles": [16, 2, 2], "tilesPerRank": [16, 2, 2]},
    {"tiles": [32, 2, 2], "tilesPerRank": [32, 2, 2]},
]

# Load the original JSON file
with open(os.path.join(original_dir, json_file), 'r') as f:
    data = json.load(f)

# Create 10 new directories with modified JSON files
for i, config in enumerate(configurations):
    # Create a new directory name
    new_dir = f"{new_dir_base}{i+1}"

    # Copy the original directory to the new directory
    shutil.copytree(original_dir, new_dir)

    # Modify the JSON data with the new configuration
    data["Mapping"]["tiles"] = config["tiles"]
    data["Mapping"]["tilesPerRank"] = config["tilesPerRank"]

    # Save the modified JSON data to the new directory
    with open(os.path.join(new_dir, json_file), 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Created {new_dir} with tiles={config['tiles']} and tilesPerRank={config['tilesPerRank']}")

print("All directories created successfully.")
