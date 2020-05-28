from PIL import Image
import numpy as np
from algorithms_clean_geometric import Algorithm, SA
import time
import os
import csv
from multiprocessing import Process, current_process

# genome size settings
polygons = 250
vertices = 1000

def experiment(name, algorithm, paintings, repetitions, polys, iterations, savepoints):
    # get date/time
    now = time.strftime("%c")
    new_name = "Experiments/" + name
    # create experiment directory with log .txt file
    if not os.path.exists(new_name):
        os.makedirs(new_name)

    total_runs = len(polys) * len(paintings) * repetitions

    # logging a lot of metadata
    logfile = new_name+"/"+name+"-LOG.txt"
    with open(logfile, 'a') as f:
        f.write("EXPERIMENT " + name + " LOG\n")
        f.write("DATE " + now + "\n\n")
        f.write("STOP CONDITION " +str(iterations)+ " iterations\n\n")
        f.write("LIST OF PAINTINGS (" + str(len(paintings)) +")\n")
        for painting in paintings:
            f.write(painting + "\n")
        f.write("\n")
        f.write("POLYS " + str(len(polys)) + " " + str(polys) + "\n\n")
        f.write("REPETITIONS " +str(repetitions) + "\n\n")
        f.write("RESULTING IN A TOTAL OF " + str(total_runs) + " RUNS\n\n")
        f.write("STARTING EXPERIMENT NOW!\n")

    # initializing the main datafile
    datafile = new_name+"/"+name + "-DATA.csv"
    header = ["Painting", "Vertices", " Replication", "MSE"]
    with open(datafile, 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # main experiment, looping through repetitions, poly numbers, and paintings:
    exp = 1
    for painting in paintings:
        painting_name = painting.split("/")[1].split("-")[0]
        for poly in polys:
            for repetition in range(repetitions):
                tic = time.time()
                # make a directory for this run, containing the per iteration data and a selection of images
                outdir = new_name + "/" + str(exp) + "-" + str(repetition) + "-" + str(poly) + "-" + painting_name
                os.makedirs(outdir)

                # Set image in np values
                im_goal = Image.open(painting)
                goal = np.array(im_goal)
                h, w = np.shape(goal)[0], np.shape(goal)[1]

                # Run the simulated annealing
                solver = SA(goal, w, h, poly, poly * 4, "MSE", savepoints, outdir, iterations)
                solver.run()
                solver.write_data()
                bestMSE = solver.best.fitness

                # save best value in maindata sheet
                datarow = [painting_name, str(poly * 4), str(repetition), bestMSE]

                with open(datafile, 'a', newline = '') as f:
                    writer = csv.writer(f)
                    writer.writerow(datarow)

                toc = time.time()
                now = time.strftime("%c")
                with open(logfile, 'a') as f:
                    f.write(now + " finished run " + str(exp) + "/" + str(total_runs) + " n: " + str(repetition) + " poly: " + str(poly) + " painting: " + painting_name + " in " + str((toc - tic)/60) + " minutes\n")

                exp += 1

# paintings = ["paintings/monalisa-240-180.png", "paintings/bach-240-180.png", "paintings/dali-240-180.png", "paintings/mondriaan-180-240.png", "paintings/pollock-240-180.png", "paintings/starrynight-240-180.png", "paintings/kiss-180-240.png", "paintings/vrouw-met-de-hermelein-240-180.png", "paintings/salvator-mundi-240-180.png"]
paintings = [["paintings/monalisa-240-180.png"], ["paintings/bach-240-180.png"], ["paintings/dali-240-180.png"], ["paintings/mondriaan-180-240.png"], ["paintings/pollock-240-180.png"], ["paintings/starrynight-240-180.png"], ["paintings/kiss-180-240.png"], ["paintings/vrouw-met-de-hermelein-240-180.png"], ["paintings/salvator-mundi-240-180.png"]]
# Define a list of savepoints, more in the first part of the run, and less later.
# savepoints = list(range(0, 250000, 1000)) + list(range(250000, 1000000, 10000))
savepoints = list(range(0, 1000000, 1000))
repetitions = 5
polys = [250]
iterations = 1000000

# Experiment name.
names = ["monalisa_lisa_test", "bach_lisa_test", "dali_lisa_test", "mondriaan_lisa_test", "pollock_lisa_test", "starrynight_lisa_test", "kiss_lisa_test", "hermelein_lisa_test", "salvator_lisa_test"]

if __name__ == '__main__':
    worker_count = 9
    for i in range(worker_count):
        args = (names[i], "SA", paintings[i], repetitions, polys, iterations, savepoints)
        p = Process(target=experiment, args=args)
        p.start()
