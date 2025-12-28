# Multi-Object Tracking with LiDAR-Camera Fusion (KITTI Dataset)

## Overview
This project detects cars and pedestrians in the KITTI Tracking dataset, tracks them across frames, and uses LiDAR data to estimate the real-world distance to each object. The final output is a video showing bounding boxes, track IDs, labels, and distance in meters.

This work is part of an AI Engineer assignment based on autonomous-driving perception.

------------------------------------------------------------

## Features
- Object detection using YOLOv8
- Tracks objects across frames using a simple IoU-based tracker
- Projects LiDAR points into the camera frame
- Computes distance using LiDAR depth inside each bounding box
- Outputs an annotated video

Only:
- Cars
- Pedestrians (people)

are detected and tracked.

------------------------------------------------------------

## Dataset
This project uses the KITTI Tracking Dataset.

Each sequence contains:
- image_02/  -> camera images
- velodyne/  -> LiDAR scans
- calib.txt  -> calibration file

Example folder:

project/data/KITTI/0000/
 ├ image_02/
 ├ velodyne/
 └ calib.txt

You can change the sequence ID in the code:

SEQ = "0000"

------------------------------------------------------------

## How to Run

1. Install dependencies
   pip install -r requirements.txt

2. Place KITTI data in:
   project/data/KITTI/<sequence_id>/

3. Run the script
   python main.py

4. Output
   A video named output.mp4 will be created.

------------------------------------------------------------

## Output Example

ID 3 | Car | 14.2 m
ID 1 | Pedestrian | 7.8 m

------------------------------------------------------------

## Tracking Method
A simple IoU-based tracker keeps the same ID for the same object across frames. New objects get new IDs. Lost objects are removed.

------------------------------------------------------------

## Distance Estimation
LiDAR points are projected into the camera image. Points inside each bounding box are used to estimate distance. The median value is used for stability.

------------------------------------------------------------

## Limitations
- Some false detections may appear
- IDs may switch during occlusion
- Distance is less accurate far away
- Tracker is lightweight and simple

These are expected for a baseline system.

------------------------------------------------------------

## Future Work
- Stronger trackers like ByteTrack or DeepSORT
- Temporal smoothing
- KITTI model fine-tuning
- 3D bounding boxes

------------------------------------------------------------
