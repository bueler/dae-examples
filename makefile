include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

all: pyScripts twoballs

twoballs: twoballs.o
	-${CLINKER} -o twoballs twoballs.o  ${PETSC_LIB}
	${RM} twoballs.o

# use this target to create symbolic links to the scripts for
# PETSc binary files
pyScripts:
	ln -sf ${PETSC_DIR}/lib/petsc/bin/PetscBinaryIO.py
	ln -sf ${PETSC_DIR}/lib/petsc/bin/petsc_conf.py

# testing
runtwoballs_1:
	-@./testit.sh twoballs "-ts_monitor -ts_view" 1 1

DATFILES:
	@./twoballs -ts_max_time 10.0 -ts_monitor binary:t.dat -ts_monitor_solution binary:u.dat -ts_adapt_dt_max 0.01 > /dev/null

runtrajectory_1: DATFILES
	-@./testit.sh trajectory.py "-o figure.png t.dat u.dat" 1 1

test_twoballs: runtwoballs_1
test_trajectory: runtrajectory_1
test: test_twoballs test_trajectory

.PHONY: distclean DATFILES

distclean:
	@rm -f *~ twoballs *tmp
	@rm -f *.pyc *.dat *.dat.info *.png
	@rm -f PetscBinaryIO.py petsc_conf.py
	@rm -rf __pycache__/
