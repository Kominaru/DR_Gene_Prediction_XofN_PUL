#!/bin/bash

# Define a list of integer numbers
numbers=(14 33 39 42 727 1312 1337 56709 177013 241543903)

# Define the list of datasets
datasets=("PathDIP")

# Define the list of classifiers
classifiers=("CAT")

# Get the current date and time
datetime=$(date +"%Y%m%d_%H%M%S")

# Loop through the datasets
for dataset in "${datasets[@]}"
do
    # Loop through the classifiers
    for classifier in "${classifiers[@]}"
    do
        # Loop through the numbers list
        experiment_name="PUL_FirstSearch_${dataset}_${classifier}_${datetime}"

        mkdir -p outputs
        mkdir -p outputs/$experiment_name

        for number in "${numbers[@]}"
        do            

            # Run the Python command with the current dataset, classifier, and seed as arguments
            python -m Code.main --dataset "$dataset" --classifier "$classifier" --random_state "$number" > outputs/$experiment_name/run_$number.txt

            # Optionally, add a message to indicate completion of each run
            echo "Experiment $experiment_name with seed $number completed"
        done
    done
done