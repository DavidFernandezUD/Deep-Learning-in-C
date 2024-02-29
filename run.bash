#!/bin/bash

# Compile both C files and link them
gcc main.c nn.c -o ./out/main

# Check if compilation was successful
if [ $? -eq 0 ]; then
    # Run the executable
    ./out/main
else
    echo "Compilation failed."
fi

