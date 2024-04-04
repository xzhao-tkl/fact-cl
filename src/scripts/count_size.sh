directory="."

# List all folders in the specified directory
folders=($(find "$directory" -type d -maxdepth 1))

# Loop through the folders and do something with each folder
for folder in "${folders[@]}"; do
    echo "$folder"
    find $folder -type f -name "combined_output.txt" -exec du -c {} + | awk 'END {print $1}'
    # Perform operations or use the folder variable as needed
done