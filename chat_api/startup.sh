#!/bin/bash

#exec python3 /app/.py &
exec service ssh start &
exec python3 /app/app.py