#!/bin/bash

# This script sends two different commands to the /SaganCommands topic
# with a delay in between.

# --- Command 1: Move Forward ---
ros2 topic pub --once -r 10 /SaganCommands sagan_interfaces/msg/SaganCmd '{
  wheel_cmd: [
    {angular_velocity: 6},
    {angular_velocity: 6},
    {angular_velocity: 6},
    {angular_velocity: 6}
  ]
}'

# --- Wait for 5 seconds ---
echo "Waiting for 5 seconds..."
sleep 10

# --- Command 2: Turn in Place ---
# (Assuming wheels on the right side are indices 1 and 3, 
# and left side are 0 and 2)
ros2 topic pub --once -r 10 /SaganCommands sagan_interfaces/msg/SaganCmd '{
  wheel_cmd: [
    {angular_velocity: 7},
    {angular_velocity: 5},
    {angular_velocity: 7},
    {angular_velocity: 5}
  ]
}'

# --- Wait for 5 seconds ---
echo "Waiting for 5 seconds..."
sleep 15

ros2 topic pub --once -r 10 /SaganCommands sagan_interfaces/msg/SaganCmd '{
  wheel_cmd: [
    {angular_velocity: 6},
    {angular_velocity: 6},
    {angular_velocity: 6},
    {angular_velocity: 6}
  ]
}'

# --- Wait for 5 seconds ---
echo "Waiting for 5 seconds..."
sleep 10

ros2 topic pub --once -r 10 /SaganCommands sagan_interfaces/msg/SaganCmd '{
  wheel_cmd: [
    {angular_velocity: 7},
    {angular_velocity: 8},
    {angular_velocity: 7},
    {angular_velocity: 8}
  ]
}'

# --- Wait for 5 seconds ---
echo "Waiting for 5 seconds..."
sleep 20

ros2 topic pub --once /SaganCommands sagan_interfaces/msg/SaganCmd '{
  wheel_cmd: [
    {angular_velocity: 0},
    {angular_velocity: 0},
    {angular_velocity: 0},
    {angular_velocity: 0}
  ]
}'

echo "Script finished."

    
