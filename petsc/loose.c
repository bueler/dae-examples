static char help[] = "ODE system solver example using TS.  Models two\n"
"equal-mass balls (particles) moving in the plane with either no\n"
"connection (free) or connected by a spring.  Solved via a Newtonian\n"
"formulation in cartesian coordinates (x,y) and velocities\n"
"(v=dx/dt,w=dy/dt). The system has dimension 8.\n\n";

// DEBUG check BDF2 convergence in free problem using Lagrangian formulation:
//for T in 2 3 4 5 7 8 9 10 11 12; do ./loose -ts_type bdf -ts_rtol 1.0e-$T -ts_atol 1.0e-$T; done

#include <petsc.h>

typedef struct {
    PetscBool    spring;
    PetscReal    g,     // m s-2;  acceleration of gravity
                 m,     // kg;     ball mass
                 l,     // m;      spring or rod length
                 k;     // N m-1;  spring constant
} LooseCtx;

extern PetscErrorCode SetInitial(Vec, LooseCtx*);
extern PetscErrorCode FreeExact(Vec, PetscReal, Vec, LooseCtx*);
extern PetscErrorCode NewtonRHSFcn(TS, PetscReal, Vec, Vec, void*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    PetscInt       steps;
    PetscReal      t0 = 0.0, tf = 2.0, dt = 0.1, errnorm;
    Vec            u, u0, uexact;
    TS             ts;
    LooseCtx       user;
    char           fstr[20] = "free";

    ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;

    user.spring = PETSC_FALSE;
    user.g = 9.81;
    user.m = 58.0e-3; // 58 g for a tennis ball
    user.l = 0.5;     // spring length
    user.k = 20.0;    // spring constant (ignored by free)
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "lse_", "options for loose",
                             ""); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-g", "acceleration of gravity (m s-2)",
                            "loose.c", user.g, &user.g, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-k", "spring constant (N m-1)", "loose.c",
                            user.k, &user.k, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-l", "spring length (m)", "loose.c",
                            user.l, &user.l, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsReal("-m", "mass of each ball (kg)", "loose.c",
                            user.m, &user.m, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsBool("-spring","spring case (not free)",
                            "loose.c", user.spring, &user.spring,
                            NULL); CHKERRQ(ierr);
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&u); CHKERRQ(ierr);
    ierr = TSCreate(PETSC_COMM_WORLD,&ts); CHKERRQ(ierr);
    ierr = TSSetApplicationContext(ts,&user); CHKERRQ(ierr);
    ierr = TSSetProblemType(ts,TS_NONLINEAR); CHKERRQ(ierr);

    // set time axis
    ierr = TSSetTime(ts,t0); CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,tf); CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dt); CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);

    // set up vecs and solver type
    ierr = VecSetSizes(u,PETSC_DECIDE,8); CHKERRQ(ierr);
    ierr = TSSetEquationType(ts, TS_EQ_ODE_EXPLICIT); CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts,NULL,NewtonRHSFcn,&user); CHKERRQ(ierr);
    ierr = TSSetType(ts,TSRK); CHKERRQ(ierr);

    // set-up of u, ts is complete
    ierr = VecSetFromOptions(u); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts); CHKERRQ(ierr);

    // set initial values and solve
    ierr = SetInitial(u, &user); CHKERRQ(ierr);
    ierr = TSSolve(ts, u); CHKERRQ(ierr);
    ierr = TSGetTime(ts, &tf); CHKERRQ(ierr);

    // numerical error in free case
    if (!user.spring) {
        ierr = VecDuplicate(u, &uexact); CHKERRQ(ierr);
        // get initial condition for evaluating exact solution
        ierr = VecDuplicate(u, &u0); CHKERRQ(ierr);
        ierr = SetInitial(u0, &user); CHKERRQ(ierr);
        ierr = FreeExact(u0, tf, uexact, &user); CHKERRQ(ierr);
        ierr = VecAXPY(u, -1.0, uexact); CHKERRQ(ierr); // u <- u - uexact
        ierr = VecNorm(u, NORM_INFINITY, &errnorm); CHKERRQ(ierr);
        VecDestroy(&u0);  VecDestroy(&uexact);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
               "numerical error at tf in free problem: |u-uexact|_inf = %.5e\n",
               errnorm); CHKERRQ(ierr);
    }

    // report
    ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
    if (user.spring)
        strcpy(fstr, "spring");
    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "%s problem solved to tf = %.3f (%d steps)\n",
               fstr, tf, steps); CHKERRQ(ierr);

    VecDestroy(&u);  TSDestroy(&ts);
    return PetscFinalize();
}

/* Variable order used in the following functions:
u[0] = q_1 (= x_1)
u[1] = q_2 (= y_1)
u[2] = q_3 (= x_2)
u[3] = q_4 (= y_2)
u[4] = v_1 (= dx_1/dt)
u[5] = v_2 (= dy_1/dt)
u[6] = v_3 (= dx_2/dt)
u[7] = v_4 (= dy_2/dt)
*/

PetscErrorCode SetInitial(Vec u, LooseCtx *user) {
    /* Set initial conditions. */
    PetscErrorCode   ierr;
    PetscReal        *au;
    ierr = VecGetArray(u,&au); CHKERRQ(ierr);
    au[0] = 0.0;              // q1 = x1
    au[1] = 1.0;              // q2 = y1
    au[2] = 0.0;              // q3 = x2
    au[3] = au[1] + user->l;  // q4 = y2;  satisfies (1)
    au[4] = 10.0;             // v1
    au[5] = 10.0;             // v2
    au[6] = 15.0;             // v3
    au[7] = au[5];            // v4;  satisfies (2)
    ierr = VecRestoreArray(u,&au); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FreeExact(Vec u0, PetscReal tf, Vec uexact, LooseCtx *user) {
    /* Exact solution based on parabolic motion.  Problem
        m x'' = 0,       x(0) = x0,  x'(0) = v0
        m y'' = - m g,   y(0) = y0,  y'(0) = w0
    has solution
        x(t) = x0 + v0 t,              x'(t) = v0,
        y(t) = y0 + w0 t - (g/2) t^2,  y'(t) = w0 - g t    */
    PetscErrorCode ierr;
    PetscReal *au0, *auex;
    if (user->spring) {
        SETERRQ(PETSC_COMM_SELF,7,"exact solution only implemented for free\n");
    }
    ierr = VecGetArray(u0, &au0); CHKERRQ(ierr);
    ierr = VecGetArray(uexact, &auex); CHKERRQ(ierr);
    auex[0] = au0[0] + au0[4] * tf;
    auex[1] = au0[1] + au0[5] * tf - 0.5 * user->g * tf * tf;
    auex[2] = au0[2] + au0[6] * tf;
    auex[3] = au0[3] + au0[7] * tf - 0.5 * user->g * tf * tf;
    auex[4] = au0[4];
    auex[5] = au0[5] - user->g * tf;
    auex[6] = au0[6];
    auex[7] = au0[7] - user->g * tf;
    ierr = VecRestoreArray(u0, &au0); CHKERRQ(ierr);
    ierr = VecRestoreArray(uexact, &auex); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode NewtonRHSFcn(TS ts, PetscReal t, Vec u, Vec G, void *ctx) {
    PetscErrorCode   ierr;
    LooseCtx         *user = (LooseCtx*)ctx;
    const PetscReal  *au;
    PetscReal        *aG;
    PetscReal        dspring = 0.0, cspring = 0.0, Fx1, Fy1, Fx2, Fy2;

    PetscFunctionBeginUser;
    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecGetArray(G,&aG); CHKERRQ(ierr);
    Fx1 = 0.0;
    Fy1 = - user->m * user->g;
    Fx2 = 0.0;
    Fy2 = - user->m * user->g;
    if (user->spring) {
        dspring = PetscSqrtReal(  (au[0] - au[2]) * (au[0] - au[2])
                                + (au[1] - au[3]) * (au[1] - au[3]) );
        if (dspring == 0.0) {
            SETERRQ(PETSC_COMM_SELF,1,"exact ball collision (unlikely?)\n");
        }
        cspring = user->k * (1.0 - user->l / dspring);
        Fx1 -= cspring * (au[0] - au[2]);
        Fy1 -= cspring * (au[1] - au[3]);
        Fx2 += cspring * (au[0] - au[2]);
        Fy2 += cspring * (au[1] - au[3]);
    }
    aG[0] = au[4];
    aG[1] = au[5];
    aG[2] = au[6];
    aG[3] = au[7];
    aG[4] = Fx1 / user->m;
    aG[5] = Fy1 / user->m;
    aG[6] = Fx2 / user->m;
    aG[7] = Fy2 / user->m;
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecRestoreArray(G,&aG); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
