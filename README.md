# TSP-Ant-Colony-Optimization
OSU CS325 Final Project

To run the program, call python3 tspACO.py [input-file].

The shortest path, and its distance will be printed, along with the time to execute. 

Because ACO becomes more accurate with more iterations, we are going to have to find ways to cut down on execution time to hit the 5 minute threshold.
As it stands right now, with T=100 and m=25, we can solve test-case-1 for full points in 6 seconds.

I think we should implement multiprocessing instead of multithreading (which is currently implemented) so that the ants don't run in sequence, but in parallel.
Hopefully that speed things up enough to where we can increase the number of iterations and ants without breaking that 5 minute limit.
