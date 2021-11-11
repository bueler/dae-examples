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
	-@./testit.sh twoballs "-tb_newtonian -tb_connect free -ts_monitor" 1 1

runtwoballs_2: twoballs
	-@./testit.sh twoballs "-tb_newtonian -tb_connect spring -ts_monitor -tb_k 0.1 -tb_l 1" 1 2

# result here depends on (fixed) time step ... it is index 3 after all!
runtwoballs_3: twoballs
	-@./testit.sh twoballs "-tb_connect rod -ts_type beuler -ksp_type preonly -pc_type svd -ts_monitor binary:t.dat -ts_monitor_solution binary:u.dat -ts_dt 0.01 -ts_max_time 1.0" 1 3

runtwoballs_4: twoballs
	-@./testit.sh twoballs "-tb_connect free -ts_type bdf -ts_bdf_order 3 -ts_monitor -ts_rtol 1.0e-6 -ts_atol 1.0e-6" 1 4

runtrajectory_1: runtwoballs_3 pyScripts
	-@./testit.sh trajectory.py "-o figure.png t.dat u.dat" 1 1

test_twoballs: runtwoballs_1 runtwoballs_2 runtwoballs_3 runtwoballs_4
test_trajectory: runtrajectory_1
test: test_twoballs test_trajectory

.PHONY: distclean

distclean:
	@rm -f *~ twoballs *tmp
	@rm -f *.pyc *.dat *.dat.info *.png
	@rm -f PetscBinaryIO.py petsc_conf.py
	@rm -rf __pycache__/
