#!/bin/bash

LOGFILE=/app/startup.log

# Log the Python version
{
    echo "Logging Python version..."
    python3.9 --version
} &>> $LOGFILE


# Start SSH service and log output
{
    echo "Starting SSH service..."
    service ssh start
} &>> $LOGFILE &

# Start the Python application and log output
{
    echo "Starting Python application..."
    python3 /app/app.py
} &>> $LOGFILE