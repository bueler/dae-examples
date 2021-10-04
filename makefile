include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99

odejac: odejac.o
	-${CLINKER} -o odejac odejac.o  ${PETSC_LIB}
	${RM} odejac.o

# use this target to create symbolic links to PETSc binary files scripts
petscPyScripts:
	ln -sf ${PETSC_DIR}/lib/petsc/bin/PetscBinaryIO.py
	ln -sf ${PETSC_DIR}/lib/petsc/bin/petsc_conf.py

# testing
#runodejac_1:
#	-@../testit.sh odejac "-ts_max_time 1.0" 1 1
#test_odejac: runodejac_1 runodejac_2
#test: test_odejac

.PHONY: distclean

distclean:
	@rm -f *~ odejac *tmp
	@rm -f *.pyc *.dat *.dat.info *.png PetscBinaryIO.py petsc_conf.py
	@rm -rf __pycache__/
