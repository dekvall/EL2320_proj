# Applied estimation project

## Introduction
Track points in hough space and create a 2-dimensional map.

### Proposal
We propose a 2D mapping of a corridor environment with the help of hough transform by using the vertical components of the image as possible landmarks. The video used will be filmed at a constant speed and with no lateral movement so that the position of the observer will be known for each frame.

> Ok sounds a bit tricky but if you think it out well it can be done.  Do not try to be perfect but just get it to work well enough.  You need to be very careful taking the data(video) and try to get as good an ida of ground truth as possible.
>
> -- John

## Mapping a corridor with hough transform

 - Runs on videos with 5 fps
 - Mainly uses videos of corridor environments

Download script can be found in is in [scripts](./scripts/).

### Mapping approach

  - Assume uniform velocity.
  - Assume fixed camera height.
  - Create a 2 dimensional map of the 3D space.

 A particle/kalman filter is used to track the houghspace maxima. These should track the track the landmarks.

 Moving straight forward is indicated by:

  - The wall/floor transition lines are static points in houghspace.
  - The vertical lines on the screen get longer (higher density of the houghspace bucket) and move outwards.

  Track the static points in houghspace with a filter and create a 2D map of the space.







