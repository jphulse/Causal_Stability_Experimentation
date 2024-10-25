#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_processes>"
    exit 1
fi

NUM_PROCESSES=$1

for((i=0; i<NUM_PROCESSES; i++)); do
    python main.py &
done
echo "$NUM_PROCESSES instances of main.py have been started in the background, they may take a while"