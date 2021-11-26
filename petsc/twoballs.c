static char help[] = "DAE system solver example using TS.  Models two\n"
"equal-mass balls (particles) moving in the plane with a rigid connecting\n"
"rod between.  Alternatively they can be unconnected (free) or connected\n"
"by a spring.  A constrained Lagrangian dynamics formulation, converted\n"
"to stabilized index-2 form is used.  Alternatively, the free/spring\n"
"cases can be solved via a Newtonian ODE system.  We use cartesian\n"
"coordinates (x,y) and velocities (v=dx/dt,w=dy/dt). The system has\n"
"dimension 10 in the main case (rod, stabilized index-2).\n\n";

// DEBUG check Jacobian for rod problem using one backward-Euler step:
// ./twoballs -ts_type beuler -tb_connect rod -ts_max_time 0.1 -ts_dt 0.1 -ksp_type preonly -pc_type svd -snes_monitor -ksp_view_mat

// DEBUG check BDF2 convergence in free problem using Lagrangian formulation:
//for T in 2 3 4 5 7 8 9 10 11 12; do ./twoballs -tb_connect free -ts_type bdf -ts_rtol 1.0e-$T -ts_atol 1.0e-$T; done

#include <petsc.h>

typedef enum {FREE, SPRING, ROD} ConnectType;
static const char* ConnectTypes[] = {"free","spring","rod",
                                     "ProblemType", "", NULL};

typedef struct {
    ConnectType  connect;
    PetscBool    newtonian;
    PetscReal    g,     // m s-2;  acceleration of gravity
                 m,     // kg;     ball mass
                 l,     // m;      spring or rod length
                 k;     // N m-1;  spring constant
} TBCtx;

extern PetscErrorCode SetInitial(Vec, TBCtx*);
extern PetscErrorCode FreeExact(Vec, PetscReal, Vec, TBCtx*);
extern PetscErrorCode NewtonRHSFcn(TS, PetscReal, Vec, Vec, void*);
extern PetscErrorCode LagrangeIFcn(TS, PetscReal, Vec, Vec, Vec, void*);
extern PetscErrorCode LagrangeIJac(TS, PetscReal, Vec, Vec, PetscReal,
                                   Mat, Mat, void*);

int main(int argc,char **argv) {
    PetscErrorCode ierr;
    PetscInt       steps;
    PetscReal      t0 = 0.0, tf = 2.0, dt = 0.1, errnorm;
    Vec            u, u0, uexact;
    Mat            A; // not used with free,nspring
    TS             ts;
    TBCtx          user;
    char           fstr[20] = "";

    ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;

    user.connect = FREE;
    user.newtonian = PETSC_FALSE;
    user.g = 9.81;
    user.m = 58.0e-3; // 58 g for a tennis ball
    user.l = 0.5;     // spring/rod length, and determines initial condition
    user.k = 20.0;    // spring constant (ignored by free,spring)
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "tb_", "options for twoballs",
                             ""); CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-connect","connect balls: free,spring,rod",
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
    ierr = PetscOptionsBool("-newtonian",
                            "use Newtonian formulation (free or spring only)",
                            "twoballs.c", user.newtonian, &user.newtonian,
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

    // set up dimension, equation type, and solver type
    if (user.newtonian) {
        if (user.connect == ROD) {
            SETERRQ(PETSC_COMM_SELF,1,
                    "rod is not implemented in Newtonian force formulation\n");
        }
        ierr = VecSetSizes(u,PETSC_DECIDE,8); CHKERRQ(ierr);
        ierr = TSSetEquationType(ts, TS_EQ_ODE_EXPLICIT); CHKERRQ(ierr);
        ierr = TSSetRHSFunction(ts,NULL,NewtonRHSFcn,&user); CHKERRQ(ierr);
        ierr = TSSetType(ts,TSRK); CHKERRQ(ierr);
    } else {
        if (user.connect == SPRING) {
            SETERRQ(PETSC_COMM_SELF,2,
                "spring is not YET implemented in Lagrangian formulation\n");
        }
        ierr = VecSetSizes(u,PETSC_DECIDE,10); CHKERRQ(ierr);
        ierr = MatCreate(PETSC_COMM_WORLD,&A); CHKERRQ(ierr);
        ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,10,10); CHKERRQ(ierr);
        ierr = MatSetFromOptions(A); CHKERRQ(ierr);
        ierr = MatSetUp(A); CHKERRQ(ierr);
        if (user.connect == ROD) {
            ierr = TSSetEquationType(ts, TS_EQ_DAE_IMPLICIT_INDEX2);
                CHKERRQ(ierr);
        } else {
            ierr = TSSetEquationType(ts, TS_EQ_DAE_IMPLICIT_INDEX1);
                CHKERRQ(ierr);
        }
        ierr = TSSetIFunction(ts,NULL,LagrangeIFcn,&user); CHKERRQ(ierr);
        ierr = TSSetIJacobian(ts,A,A,LagrangeIJac,&user);CHKERRQ(ierr);
        ierr = TSSetType(ts,TSBEULER); CHKERRQ(ierr); // FIXME BDF3 as default?
    }

    // set-up of u, ts is complete
    ierr = VecSetFromOptions(u); CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts); CHKERRQ(ierr);

    // set initial values and solve
    ierr = SetInitial(u, &user); CHKERRQ(ierr);
    ierr = TSSolve(ts, u); CHKERRQ(ierr);
    ierr = TSGetTime(ts, &tf); CHKERRQ(ierr);

    // numerical error in free case
    if (user.connect == FREE) {
        ierr = VecDuplicate(u, &uexact); CHKERRQ(ierr);
        // get initial condition for evaluating exact solution
        ierr = VecDuplicate(u, &u0); CHKERRQ(ierr);
        ierr = SetInitial(u0, &user); CHKERRQ(ierr);
        ierr = FreeExact(u0, tf, uexact, &user); CHKERRQ(ierr);
        //ierr = VecView(uexact, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
        ierr = VecAXPY(u, -1.0, uexact); CHKERRQ(ierr); // u <- u - uexact
        ierr = VecNorm(u, NORM_INFINITY, &errnorm); CHKERRQ(ierr);
        VecDestroy(&u0);  VecDestroy(&uexact);
        ierr = PetscPrintf(PETSC_COMM_WORLD,
               "numerical error at tf in free problem: |u-uexact|_inf = %.5e\n",
               errnorm); CHKERRQ(ierr);
    }

    // report
    ierr = TSGetStepNumber(ts,&steps); CHKERRQ(ierr);
    if (user.newtonian)
        strcpy(fstr, "Newtonian ");
    ierr = PetscPrintf(PETSC_COMM_WORLD,
               "%s%s problem solved to tf = %.3f (%d steps)\n",
               fstr, ConnectTypes[user.connect], tf, steps); CHKERRQ(ierr);

    if (!user.newtonian)
        MatDestroy(&A);
    VecDestroy(&u);  TSDestroy(&ts);
    return PetscFinalize();
}

/* Regarding the following functions:
In the rod case the solution is a 10-dimensional vector u.
Here is the correspondence with the notes (doc/twoballs.pdf):
u[0] = q_1 (= x_1)
u[1] = q_2 (= y_1)
u[2] = q_3 (= x_2)
u[3] = q_4 (= y_2)
u[4] = v_1 (= dx_1/dt)
u[5] = v_2 (= dy_1/dt)
u[6] = v_3 (= dx_2/dt)
u[7] = v_4 (= dy_2/dt)
u[8] = mu
u[9] = lambda
*/

PetscErrorCode SetInitial(Vec u, TBCtx *user) {
    /* Set initial conditions compatible with the rod constraint
        0 = g(q) = (1/2) ( (q1 - q3)^2 + (q2 - q4)^2 - l^2 )    (1)
    and the velocity constraint
        0 = G(q) v = (q1 - q3)(v1 - v3) + (q2 - q4)(v2 - v4)    (2)
    The initial conditions are based on the location of the first mass
    being at cartesian location (q1,q2)=(0,1) and the second at
    (q3,q4)=(0,1+l).  The initial velocity of the first mass is
    (v1,v2)=(10,10) and the second is (v3,v4)=(15,10).  In fact we 
    adjust q4 and v4 from the other values so as to satisfy constraints
    (1) and (2).  The initial value mu=0 is always set.  We set lambda
    from the other values according to the acceleration constraint
        lambda = (m / (2 l^2)) ((v1-v3)^2 + (v2-v4)^2)          (3)
    Initial values of mu and lambda are ignored in BDF solutions. */
    PetscErrorCode   ierr;
    const PetscReal  c = user->m / (2.0 * user->l * user->l);
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
    if (!user->newtonian) {
        au[8] = 0.0;          // mu
        if (user->connect == ROD) { // set lambda to satisfy (3)
            au[9] = c * ( (au[4] - au[6]) * (au[4] - au[6])
                         + (au[5] - au[7]) * (au[5] - au[7]));
        } else  // unconstrained FREE and SPRING cases
            au[9] = 0.0;
    }
    ierr = VecRestoreArray(u,&au); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode FreeExact(Vec u0, PetscReal tf, Vec uexact, TBCtx *user) {
    /* Exact solution based on parabolic motion.  Problem
        m x'' = 0,       x(0) = x0,  x'(0) = v0
        m y'' = - m g,   y(0) = y0,  y'(0) = w0
    has solution
        x(t) = x0 + v0 t,              x'(t) = v0,
        y(t) = y0 + w0 t - (g/2) t^2,  y'(t) = w0 - g t    */
    PetscErrorCode ierr;
    PetscReal *au0, *auex;
    if (user->connect != FREE) {
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
    if (!user->newtonian) {
        auex[8] = 0.0;  // mu
        auex[9] = 0.0;  // lambda
    }
    ierr = VecRestoreArray(u0, &au0); CHKERRQ(ierr);
    ierr = VecRestoreArray(uexact, &auex); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode NewtonRHSFcn(TS ts, PetscReal t, Vec u, Vec G, void *ctx) {
    PetscErrorCode   ierr;
    TBCtx            *user = (TBCtx*)ctx;
    const PetscReal  *au;
    PetscReal        *aG;
    PetscReal        dspring = 0.0, cspring = 0.0, Fx1, Fy1, Fx2, Fy2;

    PetscFunctionBeginUser;
    if (!user->newtonian) {
        SETERRQ(PETSC_COMM_SELF,7,"NewtonRHSFcn() is for -tb_newtonian\n");
    }
    if (user->connect == ROD) {
        SETERRQ(PETSC_COMM_SELF,2,"NewtonRHSFcn() does not do rod\n");
    }
    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecGetArray(G,&aG); CHKERRQ(ierr);
    Fx1 = 0.0;
    Fy1 = - user->m * user->g;
    Fx2 = 0.0;
    Fy2 = - user->m * user->g;
    if (user->connect == SPRING) {
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

PetscErrorCode LagrangeIFcn(TS ts, PetscReal t, Vec u, Vec udot, Vec F,
                            void *ctx) {
    PetscErrorCode   ierr;
    TBCtx            *user = (TBCtx*)ctx;
    const PetscReal  *au, *audot, mg = user->m * user->g;
    PetscReal        *aF, dx, dy, dvx, dvy;

    PetscFunctionBeginUser;
    if (user->newtonian) {
        SETERRQ(PETSC_COMM_SELF,8,"should not get here with -tb_newtonian\n");
    }
    if (user->connect == SPRING) {
        SETERRQ(PETSC_COMM_SELF,5,"spring not yet implemented\n");
    }
    ierr = VecGetArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecGetArrayRead(udot,&audot); CHKERRQ(ierr);
    ierr = VecGetArray(F,&aF); CHKERRQ(ierr);
    dx = au[0] - au[2];
    dy = au[1] - au[3];
    dvx = au[4] - au[6];
    dvy = au[5] - au[7];
    aF[0] = audot[0] - au[4] + au[8] * dx;
    aF[1] = audot[1] - au[5] + au[8] * dy;
    aF[2] = audot[2] - au[6] - au[8] * dx;
    aF[3] = audot[3] - au[7] - au[8] * dy;
    aF[4] = user->m * audot[4] + au[9] * dx;
    aF[5] = user->m * audot[5] + mg + au[9] * dy;
    aF[6] = user->m * audot[6] - au[9] * dx;
    aF[7] = user->m * audot[7] + mg - au[9] * dy;
    if (user->connect == ROD) {
        aF[8] = 0.5 * ( dx * dx + dy * dy - user->l * user->l );  // constraint
        aF[9] = dx * dvx + dy * dvy;  // velocity constraint
    } else { // FREE is trivial index 1 DAE
        aF[8] = au[8];             // equation:  mu = 0
        aF[9] = au[9];             // equation:  lambda = 0
    }
    ierr = VecRestoreArrayRead(u,&au); CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(udot,&audot); CHKERRQ(ierr);
    ierr = VecRestoreArray(F,&aF); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

FIXME
PetscErrorCode LagrangeIJac(TS ts, PetscReal t, Vec u, Vec udot, PetscReal a,
                            Mat J, Mat Jpre, void *ctx) {
    PetscErrorCode   ierr;
    TBCtx            *user = (TBCtx*)ctx;
    const PetscReal  *au;
    PetscInt         row, col[4], n;    // max nonzeros in a row of J is 4
    PetscReal        val[4];

    PetscFunctionBeginUser;
    if (user->connect == SPRING) {
        SETERRQ(PETSC_COMM_SELF,5,"spring not yet implemented\n");
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
    row = 8;
    if (user->connect == ROD) {
        n = 4;
        col[0] = 0;                       col[1] = 1;
        val[0] = 2.0 * (au[0] - au[4]);   val[1] = 2.0 * (au[1] - au[5]);
        col[2] = 4;                       col[3] = 5;
        val[2] = -2.0 * (au[0] - au[4]);  val[3] = -2.0 * (au[1] - au[5]);
    } else { // for equation  lambda = 0
        n = 1;
        col[0] = 8;
        val[0] = 1.0;
    }
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
