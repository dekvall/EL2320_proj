# Applied estimation project

## Intorduction

### Proposal
We propose a 2D mapping of a corridor environment with the help of hough transform by using the vertical components of the image as possible landmarks. The video used will be filmed at a constant speed and with no lateral movement so that the position of the observer will be known for each frame.

### Answer
Ok sounds a bit tricky but if you think it out well it can be done.  Do not try to be perfect but just get it to work well enough.  You need to be very careful taking the data(video) and try to get as good an ida of ground truth as possible.

## SLAM with hough transfrom

Runs on videos with 5 fps
Uses mainly the corridor video

Download script is in scripts

### SLAM Approach

Landmarks determined by houghspace coordinates. 

If we move forwards in a straight line no change will be made in the hough space for the 

The density in a houghspace bucket determines how long the line is. 

COMPLICATED AF.

Leaning towards the mapping approach


### Mapping approach??

1. Expect uniform motion

2. Make a 2(or 3) dimensional map of the 3d space

Lines will be laid out on a 3d grid. 





