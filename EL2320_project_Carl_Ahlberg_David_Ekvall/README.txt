The simulation, JPDA and transition model parameters can be altered in parameters.py

To run MC-JPDA on simulated balls: scripts/JPDA_simulation.py
- The noise parameters R and P are specifified per test and not in the parameters.py file

To run MC-JPDA on real balls with the included raw_data_4_balls.mp4 file: scripts/extract_balls_jpda.py
- Set real_plot parameter in main to True to see the results on the image frame

The main libraries used in this project are:
- OpenCV
- NumPy
- Matplotlib