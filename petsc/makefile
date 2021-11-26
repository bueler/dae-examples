include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

all: pyScripts loose rod

loose: loose.o
	-${CLINKER} -o loose loose.o  ${PETSC_LIB}
	${RM} loose.o

rod: rod.o
	-${CLINKER} -o rod rod.o  ${PETSC_LIB}
	${RM} rod.o

# use this target to create symbolic links to the scripts for
# PETSc binary files
pyScripts:
	ln -sf ${PETSC_DIR}/lib/petsc/bin/PetscBinaryIO.py
	ln -sf ${PETSC_DIR}/lib/petsc/bin/petsc_conf.py

# testing
runloose_1: loose
	-@./testit.sh loose "-ts_monitor" 1 1

runloose_2: loose
	-@./testit.sh loose "-lse_spring -ts_monitor -lse_k 0.1 -lse_l 1" 1 2

runrod_1: rod
	-@./testit.sh rod "-ts_type beuler -ksp_type preonly -pc_type svd -ts_monitor binary:t.dat -ts_monitor_solution binary:u.dat -ts_dt 0.01 -ts_max_time 1.0" 1 1

runrod_2: rod
	-@./testit.sh rod "-rod_free -ts_type bdf -ts_bdf_order 3 -ksp_type preonly -pc_type svd -ts_monitor -ts_rtol 1.0e-6 -ts_atol 1.0e-6" 1 2

runtrajectory_1: runrod_2 pyScripts
	-@./testit.sh trajectory.py "-o figure.png t.dat u.dat" 1 1

test_loose: runloose_1 runloose_2
test_rod: runrod_1 runrod_2
test_trajectory: runtrajectory_1
test: test_loose test_rod test_trajectory

.PHONY: distclean

distclean:
	@rm -f *~ loose rod *tmp
	@rm -f *.pyc *.dat *.dat.info *.png
	@rm -f PetscBinaryIO.py petsc_conf.py
	@rm -rf __pycache__/
