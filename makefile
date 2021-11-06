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
runtwoballs_1: twoballs
	-@./testit.sh twoballs "-ts_monitor -ts_view" 1 1

runtwoballs_2: twoballs
	-@./testit.sh twoballs "-ts_monitor -tb_connect nspring -tb_k 0.1 -tb_l 1" 1 2

runtwoballs_3: twoballs
	-@./testit.sh twoballs "-tb_connect nspring -tb_k 5.0 -ts_max_time 10.0 -ts_monitor binary:t.dat -ts_monitor_solution binary:u.dat -ts_adapt_dt_max 0.01" 1 3

runtrajectory_1: runtwoballs_3 pyScripts
	-@./testit.sh trajectory.py "-o figure.png t.dat u.dat" 1 1

test_twoballs: runtwoballs_1 runtwoballs_2 runtwoballs_3
test_trajectory: runtrajectory_1
test: test_twoballs test_trajectory

.PHONY: distclean

distclean:
	@rm -f *~ twoballs *tmp
	@rm -f *.pyc *.dat *.dat.info *.png
	@rm -f PetscBinaryIO.py petsc_conf.py
	@rm -rf __pycache__/
