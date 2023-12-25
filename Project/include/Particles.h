#ifndef PARTICLES_H
#define PARTICLES_H

#include <math.h>

#include "Alloc.h"
#include "Parameters.h"
#include "PrecisionTypes.h"
#include "Grid.h"
#include "EMfield.h"
#include "InterpDensSpecies.h"

struct particles {
    
    /** species ID: 0, 1, 2 , ... */
    int species_ID;
    
    /** maximum number of particles of this species on this domain. used for memory allocation */
    long npmax;
    /** number of particles of this species on this domain */
    long nop;
    
    /** Electron and ions have different number of iterations: ions moves slower than ions */
    int NiterMover;
    /** number of particle of subcycles in the mover */
    int n_sub_cycles;
    
    
    /** number of particles per cell */
    int npcel;
    /** number of particles per cell - X direction */
    int npcelx;
    /** number of particles per cell - Y direction */
    int npcely;
    /** number of particles per cell - Z direction */
    int npcelz;
    
    
    /** charge over mass ratio */
    FPpart qom;
    
    /* drift and thermal velocities for this species */
    FPpart u0, v0, w0;
    FPpart uth, vth, wth;
    
    /** particle arrays: 1D arrays[npmax] */
    FPpart* x; FPpart*  y; FPpart* z; FPpart* u; FPpart* v; FPpart* w;
    /** q must have precision of interpolated quantities: typically double. Not used in mover */
    FPinterp* q;
    
    
    
};

/** allocate particle arrays */
void particle_allocate(struct parameters*, struct particles*, int);

/** deallocate */
void particle_deallocate(struct particles*);

/** particle mover */
int mover_PC(struct particles*, struct EMfield*, struct grid*, struct parameters*);
int mover_PC_cpu(struct particles*, struct EMfield*, struct grid*, struct parameters*);

int mover_PC_cpu_2(FPpart* Px, FPpart* Py, FPpart* Pz,
                   FPpart* Pu, FPpart* Pv, FPpart* Pw,
                   FPfield* Ex, FPfield* Ey, FPfield* Ez,
                   FPfield* Bx, FPfield* By, FPfield* Bz,
                   FPfield* XN, FPfield* YN, FPfield* ZN,
                   FPpart qom, double dt, double c,
                   FPfield invdx, FPfield invdy, FPfield invdz, FPfield invVOL,
                   double xStart, double yStart, double zStart,
                   double Lx, double Ly, double Lz,
                   bool PERIODICX, bool PERIODICY, bool PERIODICZ,
                   int nxn, int nyn, int nzn,
                   int nop, int nsc, int nim);

int mover_PC_gpu(struct particles*, struct EMfield*, struct grid*, struct parameters*);

/** particle mover kernel */
__global__ void kernel_mover_PC(FPpart* Px, FPpart* Py, FPpart* Pz,
                                FPpart* Pu, FPpart* Pv, FPpart* Pw,
                                FPfield* Ex, FPfield* Ey, FPfield* Ez,
                                FPfield* Bx, FPfield* By, FPfield* Bz,
                                FPfield* XN, FPfield* YN, FPfield* ZN,
                                FPpart qom, double dt, double c,
                                FPfield invdx, FPfield invdy, FPfield invdz, FPfield invVOL,
                                double xStart, double yStart, double zStart,
                                double Lx, double Ly, double Lz,
                                bool PERIODICX, bool PERIODICY, bool PERIODICZ,
                                int nxn, int nyn, int nzn,
                                int nop, int nsc, int nim);

__global__ void kernel_mover_PC_2(FPpart* Px, FPpart* Py, FPpart* Pz,
                                  FPpart* Pu, FPpart* Pv, FPpart* Pw,
                                  FPfield* Ex, FPfield* Ey, FPfield* Ez,
                                  FPfield* Bx, FPfield* By, FPfield* Bz,
                                  FPfield* XN, FPfield* YN, FPfield* ZN,
                                  FPpart qom, double dt, double c,
                                  FPfield invdx, FPfield invdy, FPfield invdz, FPfield invVOL,
                                  double xStart, double yStart, double zStart,
                                  double Lx, double Ly, double Lz,
                                  bool PERIODICX, bool PERIODICY, bool PERIODICZ,
                                  int nxn, int nyn, int nzn,
                                  int nop, int nsc, int nim);

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles*, struct interpDensSpecies*, struct grid*);

#endif
