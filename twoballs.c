static char help[] = "ODE/DAE system solver example using TS.\n"
"Models two equal-mass balls (particles) moving in the plane with\n"
"forces or constraints between.  They are either unconnected (free),\n"
"connected by a spring, or rigidly connected by a rod.  We use\n"
"cartesian coordinates (x,y) and velocities (v=dx/dt,w=dy/dt).\n"
"The system has dimension 8 (-tb_connect [free|nspring|lspring])\n"
"or 9 (-tb_connect [rod]).\n\n";

// DEBUG Jacobian for -tb_connect rod using one backward-Euler step:
// ./twoballs -ts_type beuler -tb_connect rod -ts_max_time 0.01 -ts_dt 0.01
//    -ksp_type preonly -pc_type svd -snes_monitor -ksp_view_mat

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
extern PetscErrorCode NewtonRHSFcn(TS, PetscReal, Vec, Vec, void*);
extern PetscErrorCode LagrangeRHSFcn(TS, PetscReal, Vec, Vec, void*);
extern PetscErrorCode LagrangeIFcn(TS, PetscReal, Vec, Vec, Vec, void*);
extern PetscErrorCode LagrangeIJac(TS, PetscReal, Vec, Vec, PetscReal,
                                   Mat, Mat, void*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    PetscInt       steps;
    PetscReal      t0 = 0.0, tf = 2.0, dt = 0.1;
    Vec            u;
    Mat            A; // not used with free,nspring
    TS             ts;
    TBCtx          user;

    ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;

    user.connect = FREE;
    user.g = 9.81;
    user.m = 58.0e-3; // 58 g for a tennis ball
    user.l = 0.2;
    user.k = 20.0;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "tb_", "options for twoballs",
                             ""); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-connect",
                            "connect balls: free,nspring,lspring,rod",
                            "twoballs.c", ConnectTypes,
                            (PetscEnum)user.connect,
                            (PetscEnum*)&user.connect, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-g", "acceleration of gravity (m s-2)",
                            "twoballs.c", user.g, &user.g, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-k", "spring constant (N m-1)", "twoballs.c",
                            user.k, &user.k, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-l", "spring length (m)", "twoballs.c",
                            user.l, &user.l, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-m", "mass of each ball (kg)", "twoballs.c",
                            user.m, &user.m, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&u); CHKERRQ(ierr);
    // matrix A is not allocated, and ignored, for free,nspring
    ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetApplicationContext(ts,&user); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);

    // set time axis
    ierr = TSSetTime(ts,t0); CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,tf); CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dt); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);

    // set up dimension, equation type, and solver type
    switch (user.connect) {
        case FREE:
        case NSPRING:
            ierr = VecSetSizes(u,PETSC_DECIDE,8); CHKERRQ(ierr);
            ierr = TSSetEquationType(ts, TS_EQ_ODE_EXPLICIT); CHKERRQ(ierr);
            ierr = TSSetRHSFunction(ts,NULL,NewtonRHSFcn,&user); CHKERRQ(ierr);
            ierr = TSSetType(ts,TSRK); CHKERRQ(ierr);
            break;
        case LSPRING:
            // FIXME dim=8 here
            SETERRQ(PETSC_COMM_SELF,5,"lspring not yet implemented\n");
        case ROD:
            ierr = VecSetSizes(u,PETSC_DECIDE,9); CHKERRQ(ierr);
            ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,9,9); CHKERRQ(ierr);
            ierr = MatSetFromOptions(A); CHKERRQ(ierr);
            ierr = MatSetUp(A); CHKERRQ(ierr);
            ierr = TSSetEquationType(ts, TS_EQ_DAE_IMPLICIT_INDEX3); CHKERRQ(ierr);
            ierr = TSSetIFunction(ts,NULL,LagrangeIFcn,&user); CHKERRQ(ierr);
            ierr = TSSetIJacobian(ts,A,A,LagrangeIJac,&user);CHKERRQ(ierr);
            ierr = TSSetType(ts,TSBEULER); CHKERRQ(ierr); // FIXME BDF?
            break;
        default:
            SETERRQ(PETSC_COMM_SELF,3,"should not get here\n");
    }

    // set-up of u, ts is complete
    ierr = VecSetFromOptions(u); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts); CHKERRQ(ierr);

    // set initial values and solve
    ierr = SetInitial(u); CHKERRQ(ierr);
    ierr = TSSolve(ts,u); CHKERRQ(ierr);

    // report
    ierr = TSGetTime(ts,&tf); CHKERRQ(ierr);
    ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "connect=%s problem solved to tf = %.3f with %d steps\n",
               ConnectTypes[user.connect], tf, steps); CHKERRQ(ierr);

    VecDestroy(&u);  MatDestroy(&A);  TSDestroy(&ts);
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

PetscErrorCode NewtonRHSFcn(TS ts, PetscReal t, Vec u, Vec G, void *ctx) {
    PetscErrorCode   ierr;
    TBCtx            *user = (TBCtx*)ctx;
    const PetscReal  *au;
    PetscReal        *aG;
    PetscReal        dspring = 0.0, cspring = 0.0, Fx1, Fy1, Fx2, Fy2;

    PetscFunctionBeginUser;
    if (user->connect == LSPRING || user->connect == ROD) {
        SETERRQ(PETSC_COMM_SELF,2,"NewtonRHSFcn() does not do lspring,rod\n");
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
    PetscFunctionReturn(0);
}

PetscErrorCode LagrangeIFcn(TS ts, PetscReal t, Vec u, Vec udot, Vec F,
                            void *ctx) {
    PetscErrorCode   ierr;
    TBCtx            *user = (TBCtx*)ctx;
    const PetscReal  *au, *audot,
                     mg = user->m * user->g;
    PetscReal        *aF,
                     dx, dy;

    PetscFunctionBeginUser;
    if (user->connect == FREE || user->connect == NSPRING) {
        SETERRQ(PETSC_COMM_SELF,4,"LagrangeIFcn() does not do free,nspring\n");
    }
    if (user->connect == LSPRING) {
        SETERRQ(PETSC_COMM_SELF,5,"lspring not yet implemented\n");
    }
    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecGetArrayRead(udot,&audot); CHKERRQ(ierr);
    ierr = VecGetArray(F,&aF); CHKERRQ(ierr);
    dx = au[0] - au[4];
    dy = au[1] - au[5];
    aF[0] = audot[0] - au[2];
    aF[1] = audot[1] - au[3];
    aF[2] = user->m * audot[2] + 2.0 * au[8] * dx;
    aF[3] = user->m * audot[3] + mg + 2.0 * au[8] * dy;
    aF[4] = audot[4] - au[6];
    aF[5] = audot[5] - au[7];
    aF[6] = user->m * audot[6] - 2.0 * au[8] * dx;
    aF[7] = user->m * audot[7] + mg - 2.0 * au[8] * dy;
    aF[8] = dx * dx + dy * dy - user->l * user->l;
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(udot,&audot); CHKERRQ(ierr);
    ierr = VecRestoreArray(F,&aF); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode LagrangeIJac(TS ts, PetscReal t, Vec u, Vec udot, PetscReal a,
                            Mat J, Mat Jpre, void *ctx) {
    PetscErrorCode   ierr;
    TBCtx            *user = (TBCtx*)ctx;
    const PetscReal  *au;
    PetscInt         row, col[4], n;    // max nonzeros in a row of J is 4
    PetscReal        val[4];

    PetscFunctionBeginUser;
    if (user->connect == FREE || user->connect == NSPRING) {
        SETERRQ(PETSC_COMM_SELF,4,"LagrangeIJac() does not do free,nspring\n");
    }
    if (user->connect == LSPRING) {
        SETERRQ(PETSC_COMM_SELF,5,"lspring not yet implemented\n");
    }

    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    // construct Jacobian by rows, inserting nonzeros
    row = 0;  n = 2;   // row 0 has 2 nonzeros ...
    col[0] = 0;   col[1] = 2;
    val[0] = a;   val[1] = -1.0;
    ierr = MatSetValues(J,1,&row,n,col,val,INSERT_VALUES); CHKERRQ(ierr);
    row = 1;  n = 2;
    col[0] = 1;   col[1] = 3;  // same values as row 0
    ierr = MatSetValues(J,1,&row,n,col,val,INSERT_VALUES); CHKERRQ(ierr);
    row = 2;  n = 4;
    col[0] = 0;             col[1] = 2;
    val[0] = 2.0 * au[8];   val[1] = a * user->m;
    col[2] = 4;             col[3] = 8;
    val[2] = -2.0 * au[8];  val[3] = 2.0 * (au[0] - au[4]);
    ierr = MatSetValues(J,1,&row,n,col,val,INSERT_VALUES); CHKERRQ(ierr);
    row = 3;  n = 4;
    col[0] = 1;             col[1] = 3;
    col[2] = 5;             col[3] = 8;
    val[3] = 2.0 * (au[1] - au[5]);  // other values same as row 2
    ierr = MatSetValues(J,1,&row,n,col,val,INSERT_VALUES); CHKERRQ(ierr);
    row = 4;  n = 2;
    col[0] = 4;   col[1] = 6;
    val[0] = a;   val[1] = -1.0;
    ierr = MatSetValues(J,1,&row,n,col,val,INSERT_VALUES); CHKERRQ(ierr);
    row = 5;  n = 2;
    col[0] = 5;   col[1] = 7;  // same values as row 4
    ierr = MatSetValues(J,1,&row,n,col,val,INSERT_VALUES); CHKERRQ(ierr);
    row = 6;  n = 4;
    col[0] = 0;              col[1] = 4;
    val[0] = -2.0 * au[8];   val[1] = 2.0 * au[8];
    col[2] = 6;              col[3] = 8;
    val[2] = a * user->m;    val[3] = -2.0 * (au[0] - au[4]);
    ierr = MatSetValues(J,1,&row,n,col,val,INSERT_VALUES); CHKERRQ(ierr);
    row = 7;  n = 4;
    col[0] = 1;              col[1] = 5;
    col[2] = 7;              col[3] = 8;
    val[3] = -2.0 * (au[1] - au[5]);  // other values same as row 6
    ierr = MatSetValues(J,1,&row,n,col,val,INSERT_VALUES); CHKERRQ(ierr);
    row = 8;  n = 4;
    col[0] = 0;                       col[1] = 1;
    val[0] = 2.0 * (au[0] - au[4]);   val[1] = 2.0 * (au[1] - au[5]);
    col[2] = 4;                       col[3] = 5;
    val[2] = -2.0 * (au[0] - au[4]);  val[3] = -2.0 * (au[1] - au[5]);
    ierr = MatSetValues(J,1,&row,n,col,val,INSERT_VALUES); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (J != Jpre) {
        ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}
