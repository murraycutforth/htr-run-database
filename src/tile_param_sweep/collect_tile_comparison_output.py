import csv
import os


# Define the base directory and output file
base_dir = "/p/gpfs1/cutforth1"
output_file = "wall_times_per_iteration.csv"

# Define the columns for the output CSV
csv_columns = ["Directory", "Tiles", "TilesPerRank", "Iteration", "WallTime"]

# Open the output CSV file for writing
with open(output_file, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()

    # Iterate over each TilesComparison directory
    for dir_name in sorted(os.listdir(base_dir)):
        if dir_name.startswith("Lassen5MTilesComparison"):
            print(dir_name)

            # Extract the configuration number from the directory name
            config_number = dir_name.split("_")[-1]

            # Construct the path to the console.txt file
            console_file_path = os.path.join(base_dir, dir_name, "sample0", "console.txt")

            # Read the JSON file to extract tiles and tilesPerRank
            json_file_path = os.path.join(base_dir, dir_name, "GG-combustor-default-lassen-5M.json")
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as json_file:
                    import json
                    config_data = json.load(json_file)
                    tiles = config_data["Mapping"]["tiles"]
                    tiles_per_rank = config_data["Mapping"]["tilesPerRank"]
            else:
                tiles = "N/A"
                tiles_per_rank = "N/A"

            # Parse the console.txt file to extract wall times
            if os.path.exists(console_file_path):
                with open(console_file_path, 'r') as console_file:
                    lines = console_file.readlines()
                    for line in lines[1:]:  # Skip the header line
                        parts = line.split()
                        if len(parts) >= 3:  # Ensure the line has enough columns
                            iteration = int(parts[0])
                            wall_time = float(parts[2])
                            # Write the data to the CSV
                            writer.writerow({
                                "Directory": dir_name,
                                "Tiles": str(tiles),
                                "TilesPerRank": str(tiles_per_rank),
                                "Iteration": iteration,
                                "WallTime": wall_time
                            })

print(f"Wall times per iteration have been saved to {output_file}.")
