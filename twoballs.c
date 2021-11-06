static char help[] = "ODE/DAE system solver example using TS.\n"
"Models two equal-mass balls (particles) moving in the plane with\n"
"forces or constraints between.  They are either unconnected (free),\n"
"connected by a spring, or rigidly connected by a rod.  We use\n"
"cartesian coordinates (x,y) and velocities (v=dx/dt,w=dy/dt).\n"
"The system has dimension 8 (-tb_connect [free|nspring|lspring])\n"
"or 9 (-tb_connect [rod]).\n\n";

#include <petsc.h>

typedef enum {FREE, NSPRING, LSPRING, ROD} ConnectType;
static const char* ConnectTypes[] = {"free","nspring","lspring","rod",
                                     "ProblemType", "", NULL};

typedef struct {
    ConnectType  connect;
    PetscReal    g,     // m s-2;  acceleration of gravity
                 m,     // kg;     ball mass
                 l,     // m;      spring or rod length
                 k;     // N m-1;  spring constant
} TBCtx;

extern PetscErrorCode SetInitial(Vec);
extern PetscErrorCode NewtonRHS(TS, PetscReal, Vec, Vec, void*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    PetscInt       steps;
    PetscReal      t0 = 0.0, tf = 2.0, dt = 0.1;
    Vec            u;
    TS             ts;
    TBCtx          user;

    ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;

    user.connect = FREE;
    user.g = 9.81;
    user.m = 58.0e-3; // 58 g for a tennis ball
    user.l = 0.2;
    user.k = 20.0;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "tb_", "options for twoballs", "");
           CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-connect", "connect balls: free,nspring,lspring,rod",
                            "twoballs.c", ConnectTypes, (PetscEnum)user.connect,
                            (PetscEnum*)&user.connect, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-g", "acceleration of gravity (m s-2)", "twoballs.c",
                            user.g, &user.g, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-k", "spring constant (N m-1)", "twoballs.c",
                            user.k, &user.k, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-l", "spring length (m)", "twoballs.c",
                            user.l, &user.l, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-m", "mass of each ball (kg)", "twoballs.c",
                            user.m, &user.m, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    // FIXME there will be two RHS functions
    if (user.connect == LSPRING || user.connect == ROD) {
        SETERRQ(PETSC_COMM_SELF,3,"LagrangeRHS() not implemented\n");
    }

    ierr = VecCreate(PETSC_COMM_WORLD,&u); CHKERRQ(ierr);
    ierr = VecSetSizes(u,PETSC_DECIDE,8); CHKERRQ(ierr);  // FIXME depends
    ierr = VecSetFromOptions(u); CHKERRQ(ierr);

    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetType(ts,TSRK); CHKERRQ(ierr); // FIXME
    ierr = TSSetApplicationContext(ts,&user); CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts,NULL,NewtonRHS,&user); CHKERRQ(ierr);  // FIXME depends

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
    // FIXME N or L type
    ierr = PetscPrintf(PETSC_COMM_WORLD,
                       "Newton forces problem (connect=%s) solved to tf = %.3f with %d steps\n",
                       ConnectTypes[user.connect], tf, steps); CHKERRQ(ierr);

    VecDestroy(&u);  TSDestroy(&ts);
    return PetscFinalize();
}

PetscErrorCode SetInitial(Vec u) {
    PetscErrorCode ierr;
    PetscReal *au;
    ierr = VecGetArray(u,&au); CHKERRQ(ierr);
    au[0] = 0.0;
    au[1] = 1.0;
    au[2] = 15.0;
    au[3] = 15.0;
    au[4] = 0.0;
    au[5] = 2.0;
    au[6] = 10.0;
    au[7] = 10.0;
    ierr = VecRestoreArray(u,&au); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode NewtonRHS(TS ts, PetscReal t, Vec u, Vec G, void *ctx) {
    PetscErrorCode   ierr;
    TBCtx            *user = (TBCtx*)ctx;
    const PetscReal  *au;
    PetscReal        dspring = 0.0, cspring = 0.0,
                     Fx1, Fy1, Fx2, Fy2, *aG;
    if (user->connect == LSPRING || user->connect == ROD) {
        SETERRQ(PETSC_COMM_SELF,2,"LagrangeRHS() not implemented\n");
    }
    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecGetArray(G,&aG); CHKERRQ(ierr);
    Fx1 = 0.0;
    Fy1 = - user->m * user->g;
    Fx2 = 0.0;
    Fy2 = - user->m * user->g;
    if (user->connect == NSPRING) {
        dspring = PetscSqrtReal(  (au[0] - au[4]) * (au[0] - au[4])
                                + (au[1] - au[5]) * (au[1] - au[5]) );
        if (dspring == 0.0) {
            SETERRQ(PETSC_COMM_SELF,1,"exact ball collision (unlikely?)\n");
        }
        cspring = user->k * (1.0 - user->l / dspring);
        Fx1 -= cspring * (au[0] - au[4]);
        Fy1 -= cspring * (au[1] - au[5]);
        Fx2 -= cspring * (au[4] - au[0]);
        Fy2 -= cspring * (au[5] - au[1]);
    }
    aG[0] = au[2];
    aG[1] = au[3];
    aG[2] = Fx1 / user->m;
    aG[3] = Fy1 / user->m;
    aG[4] = au[6];
    aG[5] = au[7];
    aG[6] = Fx2 / user->m;
    aG[7] = Fy2 / user->m;
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecRestoreArray(G,&aG); CHKERRQ(ierr);
    return 0;
}
