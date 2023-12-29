#include "EMfield.h"

// allocate electric and magnetic field
void field_allocate(struct grid* grd, struct EMfield* field)
{
    // allocate electric field on cpu with pinned memory
    field->Ex = newPinnedArr3<FPfield>(&field->Ex_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Ey = newPinnedArr3<FPfield>(&field->Ey_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Ez = newPinnedArr3<FPfield>(&field->Ez_flat, grd->nxn, grd->nyn, grd->nzn);

    // allocate electric field on gpu
    cudaMalloc(&field->Ex_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMalloc(&field->Ey_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMalloc(&field->Ez_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    // allocate magnetic field on cpu with pinned memory
    field->Bxn = newPinnedArr3<FPfield>(&field->Bxn_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Byn = newPinnedArr3<FPfield>(&field->Byn_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Bzn = newPinnedArr3<FPfield>(&field->Bzn_flat, grd->nxn, grd->nyn, grd->nzn);

    // allocate magnetic field on gpu
    cudaMalloc(&field->Bxn_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMalloc(&field->Byn_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
    cudaMalloc(&field->Bzn_gpu, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
}

// deallocate electric and magnetic field
void field_deallocate(struct grid* grd, struct EMfield* field)
{
    // deallocate electric field on cpu
    delPinnedArr3(field->Ex);
    delPinnedArr3(field->Ey);
    delPinnedArr3(field->Ez);

    // deallocate electric field on gpu
    cudaFree(field->Ex_gpu);
    cudaFree(field->Ey_gpu);
    cudaFree(field->Ez_gpu);

    // deallocate magnetic field on cpu
    delPinnedArr3(field->Bxn);
    delPinnedArr3(field->Byn);
    delPinnedArr3(field->Bzn);

    // deallocate magnetic field on gpu
    cudaFree(field->Bxn_gpu);
    cudaFree(field->Byn_gpu);
    cudaFree(field->Bzn_gpu);
}
