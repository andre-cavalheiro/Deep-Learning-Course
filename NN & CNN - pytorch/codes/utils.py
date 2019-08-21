from datetime import datetime
import numpy as np

def LOG_INFO(msg):
    now = datetime.now()
    display_now = str(now).split(' ')[1][:-3]
    print(display_now + ' ' + msg)

def printResults(pipeline, keys, acc, time, type):
    fname = type + '_'
    for k in keys:
        fname += k + '=' + str(pipeline[k]) + '_'

    f = open(fname + '.txt', "w")
    """
    for a in acc:
        f.write(str(a) + ' ')
    """

    f.write('Mean:')
    f.write(str(np.mean(acc)))

    f.write('Running Time:')
    f.write(str(time))
    f.close()
