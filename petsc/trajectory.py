#!/usr/bin/env python3

help =\
'''
Plot trajectory from running LOOSE or ROD.  Reads output from
   -ts_monitor binary:TDATA -ts_monitor_solution binary:UDATA
Requires copies or sym-links to $PETSC_DIR/lib/petsc/bin/PetscBinaryIO.py and
$PETSC_DIR/lib/petsc/bin/petsc_conf.py.  (Do:  make pyScripts)
Note there is a particular variable order but only the first four variables
are relevant:
  (x1, y1, x2, y2, ...)
'''

import PetscBinaryIO

from sys import exit, stdout
from time import sleep
from argparse import ArgumentParser, RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt

parser = ArgumentParser(description=help,
                        formatter_class=RawTextHelpFormatter)
parser.add_argument('tfile',metavar='TDATA',
                    help='from -ts_monitor binary:TDATA')
parser.add_argument('ufile',metavar='UDATA',
                    help='from -ts_monitor_solution binary:UDATA')
parser.add_argument('-o',metavar='FILE',dest='filename',
                    help='image file FILE for trajectory')
args = parser.parse_args()

io = PetscBinaryIO.PetscBinaryIO()
t = np.array(io.readBinaryFile(args.tfile)).flatten()
U = np.array(io.readBinaryFile(args.ufile)).transpose()
dims = np.shape(U)

if len(t) != dims[1]:
    print('time dimension mismatch: %d != %d' % (len(t),dims[1]))
    exit(1)

print('time t has length=%d, solution Y is shape=(%d,%d)' % \
          (len(t),dims[0],dims[1]))

#for k in range(dims[0]):
#    plt.plot(t,U[k],label='y[%d]' % k)

plt.plot(U[0,U[1]>=0],U[1,U[1]>=0],label='particle 1')
plt.plot(U[2,U[3]>=0],U[3,U[3]>=0],label='particle 2')
plt.xlabel('x'),  plt.ylabel('y')
plt.legend()

if args.filename:
    print('writing file %s' % args.filename)
    plt.savefig(args.filename)
else:
    plt.show()
