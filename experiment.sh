#!/bin/bash

# Create outputs folder if it doesn't exist
mkdir -p outputs
mkdir -p outputs/OG_XofNPosOnly_Minoc4_Under

# Loop 10 times
for ((i=1; i<=10; i++))
do
    # Run the Python command and redirect output to a text file
    python -m Code.ML_Procedures.ML_performances > outputs/OG_XofNPosOnly_Minoc4_Under/run_$i.txt
    
    # Optionally, add a message to indicate completion of each run
    echo "Run $i completed."
done