static char help[] = "ODE/DAE system solver example using TS.\n"
"Models two equal-mass balls (particles) moving in the plane with\n"
"forces between.  Written out in cartesian coordinates (x,y)\n"
"and velocities (v=dx/dt,w=dy/dt) the system has dimension 8:\n"
"FIXME: here is free motion case\n"
"    dot x1 = v1\n"
"    dot y1 = w1\n"
"  m dot v1 = 0\n"
"  m dot w1 = - m g\n"
"    dot x2 = v2\n"
"    dot y2 = w2\n"
"  m dot v2 = 0\n"
"  m dot w2 = - m g\n"
"\n\n";

#include <petsc.h>

extern PetscErrorCode SetInitial(Vec);
extern PetscErrorCode FormRHSFunction(TS, PetscReal, Vec, Vec, void*);

int main(int argc,char **argv) {
  PetscErrorCode ierr;
  PetscInt   steps;
  PetscReal  t0 = 0.0, tf = 2.0, dt = 0.1;
  Vec        u;
  TS         ts;

  ierr = PetscInitialize(&argc,&argv,NULL,help); if (ierr) return ierr;

  ierr = VecCreate(PETSC_COMM_WORLD,&u); CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,8); CHKERRQ(ierr);
  ierr = VecSetFromOptions(u); CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK); CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,NULL); CHKERRQ(ierr);

  // set time axis
  ierr = TSSetTime(ts,t0); CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,tf); CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt); CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts); CHKERRQ(ierr);

  // set initial values and solve
  ierr = SetInitial(u); CHKERRQ(ierr);
  ierr = TSSolve(ts,u); CHKERRQ(ierr);

  // compute error and report
  ierr = TSGetTime(ts,&tf); CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "solved to tf = %.3f with %d steps\n",
                     tf,steps); CHKERRQ(ierr);

  VecDestroy(&u);  TSDestroy(&ts);
  return PetscFinalize();
}

PetscErrorCode SetInitial(Vec u) {
    PetscReal *au;
    VecGetArray(u,&au);
    au[0] = 0.0;
    au[1] = 1.0;
    au[2] = 10.0;
    au[3] = 10.0;
    au[4] = 0.0;
    au[5] = 2.0;
    au[6] = 20.0;
    au[7] = 20.0;
    VecRestoreArray(u,&au);
    return 0;
}

PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec u, Vec G,
                               void *ptr) {
    const PetscReal m = 58.0e-3,  // 58 g is weight of a tennis ball
                    g = 9.81;     // m s-2; acceleration of gravity
    const PetscReal *au;
    PetscReal       *aG;
    VecGetArrayRead(u,&au);
    VecGetArray(G,&aG);
    aG[0] = au[2];
    aG[1] = au[3];
    aG[2] = 0.0 / m;
    aG[3] = (- m * g) / m;
    aG[4] = au[6];
    aG[5] = au[7];
    aG[6] = 0.0 / m;
    aG[7] = (- m * g) / m;
    VecRestoreArrayRead(u,&au);
    VecRestoreArray(G,&aG);
    return 0;
}
