This repository is dedicated to showcasing the highlights from the computing project I completed for the 3rd year of my MPhys Physics and Astronomy Degree at Durham University. In this project I simulated collisions between two star clusters each containing 1000 stars. The simulations were ran for 225 million years to see how the systems evolved, with the positions and velocities being tracked every 150 thousand years. Each parameter was decided on using a number of tests which tracked the numerical errors in the simulation, with the timestep size undergoing tests which caused the program to simulate the clusters over 11 billion years. I also used matplotlib to created animated graphs of the simulations and their energy profiles throughout. The videos of these can be found on youtube. First I simulated a head on collision: https://youtu.be/fw0quLn1-B0?si=KHbdo-VM-Nzpp8ke, and then a grazing collision where the clusters were initially projected to skim the edges of each other: https://youtu.be/LiAtVnNg360?si=TPNL7w-yuy10lQW6.

My code for these can be found in the following files:

A single cluster collapsing under its own gravity - single_cluster.py

Head-on collision of two clusters - collision_animation.py - and - cluster_collision.ipynb
                                  
Grazing collision of two clusters - grazing_collision.py

Other useful files:

Report detailing the findings of my investigation - Mark_Ryan_Comp_proj_Report.pdf

Poster - Summative_Poster_Mark_Ryan.pdf

As well as this I also developed a 3 body simulation so that I could get to grips with the types of calculations that would be needed to make the n-body simulation for my clusters. The code for this can be found in the file titled milestone.ipynb.
