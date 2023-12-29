#ifndef EMFIELD_H
#define EMFIELD_H

#include "Alloc.h"
#include "Grid.h"

// structure with field information
struct EMfield
{
    // field arrays: 4D arrays
    
    // electric field defined on nodes: last index is component
    FPfield*** Ex;
    FPfield* Ex_flat;
    FPfield*** Ey;
    FPfield* Ey_flat;
    FPfield*** Ez;
    FPfield* Ez_flat;

    // electric field flat array for gpu
    FPfield* Ex_gpu;
    FPfield* Ey_gpu;
    FPfield* Ez_gpu;

    // magnetic field defined on nodes: last index is component
    FPfield*** Bxn;
    FPfield* Bxn_flat;
    FPfield*** Byn;
    FPfield* Byn_flat;
    FPfield*** Bzn;
    FPfield* Bzn_flat;

    // magnetic field flat array for gpu
    FPfield* Bxn_gpu;
    FPfield* Byn_gpu;
    FPfield* Bzn_gpu;
};

// allocate electric and magnetic field
void field_allocate(struct grid*, struct EMfield*);

// deallocate electric and magnetic field
void field_deallocate(struct grid*, struct EMfield*);

#endif
