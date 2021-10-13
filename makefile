include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

all: pyScripts twoballs

twoballs: twoballs.o
	-${CLINKER} -o twoballs twoballs.o  ${PETSC_LIB}
	${RM} twoballs.o

# use this target to create symbolic links to the scripts for
# PETSc binary files and for plotTS.py from Bueler's book
P4PDES_DIR = ~/repos/p4pdes/
pyScripts:
	ln -sf ${PETSC_DIR}/lib/petsc/bin/PetscBinaryIO.py
	ln -sf ${PETSC_DIR}/lib/petsc/bin/petsc_conf.py
	ln -sf ${P4PDES_DIR}/c/ch5/plotTS.py

# testing
runtwoballs_1:
	-@./testit.sh twoballs "-ts_monitor binary:t.dat -ts_monitor_solution binary:u.dat -ts_adapt_dt_max 0.01" 1 1
# view .dat files by:
#    $ ./plotTS.py -o figure.png t.dat u.dat -dof 8 -c 1
#    $ eog figure.png

test_twoballs: runtwoballs_1
test: test_twoballs

.PHONY: distclean

distclean:
	@rm -f *~ twoballs *tmp
	@rm -f *.pyc *.dat *.dat.info *.png
	@rm -f PetscBinaryIO.py petsc_conf.py plotTS.py
	@rm -rf __pycache__/
