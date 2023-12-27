#include "EMfield.h"

/** allocate electric and magnetic field */
void field_allocate(struct grid* grd, struct EMfield* field)
{
    // E on nodes
    field->Ex = newPinnedArr3<FPfield>(&field->Ex_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Ey = newPinnedArr3<FPfield>(&field->Ey_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Ez = newPinnedArr3<FPfield>(&field->Ez_flat, grd->nxn, grd->nyn, grd->nzn);

    // B on nodes
    field->Bxn = newPinnedArr3<FPfield>(&field->Bxn_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Byn = newPinnedArr3<FPfield>(&field->Byn_flat, grd->nxn, grd->nyn, grd->nzn);
    field->Bzn = newPinnedArr3<FPfield>(&field->Bzn_flat, grd->nxn, grd->nyn, grd->nzn);
}

/** deallocate electric and magnetic field */
void field_deallocate(struct grid* grd, struct EMfield* field)
{
    // E deallocate 3D arrays
    delPinnedArr3(field->Ex);
    delPinnedArr3(field->Ey);
    delPinnedArr3(field->Ez);

    // B deallocate 3D arrays
    delPinnedArr3(field->Bxn);
    delPinnedArr3(field->Byn);
    delPinnedArr3(field->Bzn);
}
