#!/bin/bash

# Define a list of integer numbers
numbers=(42 727 1337 33 14 177013 39 56709 1312 241543903)

# Create outputs folder if it doesn't exist
mkdir -p outputs
mkdir -p outputs/OG

# Loop through the numbers list
for number in "${numbers[@]}"
do
    # Run the Python command with the current number as argument and redirect output to a text file
    python -m Code.main "$number" > outputs/OG/run_${number}_$i.txt
    
    # Optionally, add a message to indicate completion of each run
    echo "Run $i with $number completed."
done