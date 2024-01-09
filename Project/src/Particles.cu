#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "Alloc.h"
#include "Particles.h"

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];

    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0)
    {
        // electrons
        part->NiterMover   = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    }

    else
    {
        // ions: only one iteration
        part->NiterMover   = 1;
        part->n_sub_cycles = 1;
    }

    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel  = part->npcelx*part->npcely*part->npcelz;

    // cast it to required precision
    part->qom = (FPpart) param->qom[is];

    // allocate drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];

    // allocate thermal velocities
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];

    long npmax = part->npmax;

    // allocate particle position arrays on cpu
    cudaHostAlloc(&part->x, sizeof(FPpart) * npmax, cudaHostAllocDefault);
    cudaHostAlloc(&part->y, sizeof(FPpart) * npmax, cudaHostAllocDefault);
    cudaHostAlloc(&part->z, sizeof(FPpart) * npmax, cudaHostAllocDefault);

    // allocate particle velocity arrays on cpu
    cudaHostAlloc(&part->u, sizeof(FPpart) * npmax, cudaHostAllocDefault);
    cudaHostAlloc(&part->v, sizeof(FPpart) * npmax, cudaHostAllocDefault);
    cudaHostAlloc(&part->w, sizeof(FPpart) * npmax, cudaHostAllocDefault);

    // allocate particle position arrays on gpu
    cudaMalloc(&part->x_gpu, sizeof(FPpart) * npmax);
    cudaMalloc(&part->y_gpu, sizeof(FPpart) * npmax);
    cudaMalloc(&part->z_gpu, sizeof(FPpart) * npmax);

    // allocate particle velocity arrays on gpu
    cudaMalloc(&part->u_gpu, sizeof(FPpart) * npmax);
    cudaMalloc(&part->v_gpu, sizeof(FPpart) * npmax);
    cudaMalloc(&part->w_gpu, sizeof(FPpart) * npmax);

    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
}

/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle position arrays on cpu
    cudaFree(part->x);
    cudaFree(part->y);
    cudaFree(part->z);

    // deallocate particle velocity arrays on cpu
    cudaFree(part->u);
    cudaFree(part->v);
    cudaFree(part->w);

    // deallocate particle position arrays on gpu
    cudaFree(part->x_gpu);
    cudaFree(part->y_gpu);
    cudaFree(part->z_gpu);

    // deallocate particle velocity arrays on gpu
    cudaFree(part->u_gpu);
    cudaFree(part->v_gpu);
    cudaFree(part->w_gpu);

    // deallocate charge
    delete[] part->q;
}

/** particle mover on cpu */
int mover_PC_cpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "*** CPU MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++)
    {
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++)
        {
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];

            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++)
            {
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);

                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];

                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;

                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++)
                        {
                            Exl += weight[ii][jj][kk] * field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk] * field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk] * field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk] * field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk] * field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk] * field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }

                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);

                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;

                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;

                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
            }

            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;

            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx)
            {
                // PERIODIC
                if (param->PERIODICX == true)
                {
                    part->x[i] = part->x[i] - grd->Lx;
                }

                // REFLECTING
                else
                {
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }

            if (part->x[i] < 0)
            {
                // PERIODIC
                if (param->PERIODICX==true)
                {
                   part->x[i] = part->x[i] + grd->Lx;
                }

                // REFLECTING
                else
                {
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }

            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly)
            {
                // PERIODIC
                if (param->PERIODICY==true)
                {
                    part->y[i] = part->y[i] - grd->Ly;
                }

                // REFLECTING
                else
                {
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }

            if (part->y[i] < 0)
            {
                // PERIODIC
                if (param->PERIODICY==true)
                {
                    part->y[i] = part->y[i] + grd->Ly;
                }

                // REFLECTING
                else
                {
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }

            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz)
            {
                // PERIODIC
                if (param->PERIODICZ==true)
                {
                    part->z[i] = part->z[i] - grd->Lz;
                }

                // REFLECTING
                else
                {
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }

            if (part->z[i] < 0)
            {
                // PERIODIC
                if (param->PERIODICZ==true)
                {
                    part->z[i] = part->z[i] + grd->Lz;
                }

                // REFLECTING
                else
                {
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
        }
    }

    return 0;
}

/** particle mover on gpu */
int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param, cudaStream_t* stream)
{
    // print species and subcycling
    std::cout << "*** GPU MOVER with SUBCYCLYING " << param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    int stride  = part->nop / 16; // the number of elements/threads in one partition, the full array is split into 16 equally sized partitions
    int threads = 64; // one warp is 32 threads, but 64 threads are needed for 100% theoretical occupancy
    int blocks  = (stride + threads - 1) / threads; // number of thread blocks to cover the entire partition rounded upwards

    // partition the particle array into 16 equally sized partitions, which will be distributed equally amongst the 16 streams
    for (int i = 0; i != 16; ++i)
    {
        // we want to interleave kernel dispatches and memcpy dispatches so that the copy engine
        // can start working as soon as the first kernel invocation is done and not be delayed
        // because the cpu has not been quick enough to record the dispatch yet

        // dispatch gpu computation of particle movement for one partition
        kernel_mover_PC<<<blocks, threads, 0, stream[i]>>>(part->x_gpu, part->y_gpu, part->z_gpu,
                                                           part->u_gpu, part->v_gpu, part->w_gpu,
                                                           field->Ex_gpu, field->Ey_gpu, field->Ez_gpu,
                                                           field->Bxn_gpu, field->Byn_gpu, field->Bzn_gpu,
                                                           grd->XN_gpu, grd->YN_gpu, grd->ZN_gpu,
                                                           part->qom, param->dt, param->c,
                                                           grd->invdx, grd->invdy, grd->invdz, grd->invVOL,
                                                           grd->xStart, grd->yStart, grd->zStart,
                                                           grd->Lx, grd->Ly, grd->Lz,
                                                           param->PERIODICX, param->PERIODICY, param->PERIODICZ,
                                                           grd->nxn, grd->nyn, grd->nzn,
                                                           part->nop, part->n_sub_cycles, part->NiterMover,
                                                           i * stride); // this is the thread offset

        // the following memory copies are async, meaning that the cpu will not wait for them to complete
        // only the partition that was computed with the kernel invocation above is copied back, not the entire array

        // other computations on cpu read from particle position, so copy from gpu to cpu to get most recent values
        cudaMemcpyAsync(&part->x[i * stride], &part->x_gpu[i * stride], sizeof(FPpart) * stride, cudaMemcpyDeviceToHost, stream[i]);
        cudaMemcpyAsync(&part->y[i * stride], &part->y_gpu[i * stride], sizeof(FPpart) * stride, cudaMemcpyDeviceToHost, stream[i]);
        cudaMemcpyAsync(&part->z[i * stride], &part->z_gpu[i * stride], sizeof(FPpart) * stride, cudaMemcpyDeviceToHost, stream[i]);

        // other computations on cpu read from particle velocity, so copy from gpu to cpu to get most recent values
        cudaMemcpyAsync(&part->u[i * stride], &part->u_gpu[i * stride], sizeof(FPpart) * stride, cudaMemcpyDeviceToHost, stream[i]);
        cudaMemcpyAsync(&part->v[i * stride], &part->v_gpu[i * stride], sizeof(FPpart) * stride, cudaMemcpyDeviceToHost, stream[i]);
        cudaMemcpyAsync(&part->w[i * stride], &part->w_gpu[i * stride], sizeof(FPpart) * stride, cudaMemcpyDeviceToHost, stream[i]);
    }

    return(0);
}

/** particle mover kernel */
__global__ void kernel_mover_PC(FPpart* Px, FPpart* Py, FPpart* Pz,
                                FPpart* Pu, FPpart* Pv, FPpart* Pw,
                                FPfield* Ex, FPfield* Ey, FPfield* Ez,
                                FPfield* Bx, FPfield* By, FPfield* Bz,
                                FPfield* Nx, FPfield* Ny, FPfield* Nz,
                                FPpart qom, FPpart dt, FPpart c,
                                FPfield invdx, FPfield invdy, FPfield invdz, FPfield invVOL,
                                FPpart xStart, FPpart yStart, FPpart zStart,
                                FPpart Lx, FPpart Ly, FPpart Lz,
                                bool PERIODICX, bool PERIODICY, bool PERIODICZ,
                                int nxn, int nyn, int nzn,
                                int nop, int nsc, int nim,
                                int offset) // indicates the partition in the entire array that this kernel invocation should work on
{
    // global thread index in the particle arrays
    int tid = threadIdx.x + blockIdx.x * blockDim.x + offset;

    // protect against potential indices that are out-of-bounds
    if (tid < nop)
    {
        // load particle position from slow global memory into fast local registers, then use this local register
        // as a cached version of the array value and only write it back a single time at the end of the kernel
        FPpart Pxl = Px[tid];
        FPpart Pyl = Py[tid];
        FPpart Pzl = Pz[tid];

        // auxiliary variables
        FPpart dt_sub_cycling = dt / (FPpart)nsc;
        FPpart dto2 = (FPpart)0.5 * dt_sub_cycling; // make sure that constant value is cast to the chosen precision to reduce fp64 operations
        FPpart qomdt2 = qom * dto2 / c;

        // intermediate particle position and velocity
        FPpart xptilde, yptilde, zptilde;
        FPpart uptilde, vptilde, wptilde;

        // start subcycling
        for (int i = 0; i != nsc; ++i)
        {
            // use cached version of particle position instead of reading from slow global memory
            xptilde = Pxl;
            yptilde = Pyl;
            zptilde = Pzl;

            // calculate the average velocity iteratively
            for (int j = 0; j != nim; ++j)
            {
                // interpolation G-->P
                int ix = 2 + int((Pxl - xStart) * invdx);
                int iy = 2 + int((Pyl - yStart) * invdy);
                int iz = 2 + int((Pzl - zStart) * invdz);

                FPfield weight[8]; // flattened version of the previous [2][2][2] array
                FPfield xi[2], eta[2], zeta[2];

                // calculate densities
                // the cached version of particle position can be used in calculations also to reduce slow global memory accesses
                // the get_idx function converts a 3D index into a flattened 1D index
                xi[0]   = Pxl - Nx[get_idx(ix - 1, iy, iz, nyn, nzn)];
                eta[0]  = Pyl - Ny[get_idx(ix, iy - 1, iz, nyn, nzn)];
                zeta[0] = Pzl - Nz[get_idx(ix, iy, iz - 1, nyn, nzn)];
                xi[1]   = Nx[get_idx(ix, iy, iz, nyn, nzn)] - Pxl;
                eta[1]  = Ny[get_idx(ix, iy, iz, nyn, nzn)] - Pyl;
                zeta[1] = Nz[get_idx(ix, iy, iz, nyn, nzn)] - Pzl;

                // calculate weights
                for (int u = 0; u != 2; ++u)
                    for (int v = 0; v != 2; ++v)
                        for (int w = 0; w != 2; ++w)
                            weight[get_idx(u, v, w, 2, 2)] = xi[u] * eta[v] * zeta[w] * invVOL;

                // initialize local electric and magnetic field
                FPfield Exl = 0.0, Eyl = 0.0, Ezl = 0.0;
                FPfield Bxl = 0.0, Byl = 0.0, Bzl = 0.0;

                // calculate local electric and magnetic field
                for (int u = 0; u != 2; ++u)
                {
                    for (int v = 0; v != 2; ++v)
                    {
                        for (int w = 0; w != 2; ++w)
                        {
                            Exl += weight[get_idx(u, v, w, 2, 2)] * Ex[get_idx(ix - u, iy - v, iz - w, nyn, nzn)];
                            Eyl += weight[get_idx(u, v, w, 2, 2)] * Ey[get_idx(ix - u, iy - v, iz - w, nyn, nzn)];
                            Ezl += weight[get_idx(u, v, w, 2, 2)] * Ez[get_idx(ix - u, iy - v, iz - w, nyn, nzn)];
                            Bxl += weight[get_idx(u, v, w, 2, 2)] * Bx[get_idx(ix - u, iy - v, iz - w, nyn, nzn)];
                            Byl += weight[get_idx(u, v, w, 2, 2)] * By[get_idx(ix - u, iy - v, iz - w, nyn, nzn)];
                            Bzl += weight[get_idx(u, v, w, 2, 2)] * Bz[get_idx(ix - u, iy - v, iz - w, nyn, nzn)];
                        }
                    }
                }

                // end interpolation
                FPpart omdtsq = qomdt2 * qomdt2 * (Bxl * Bxl + Byl * Byl + Bzl * Bzl);
                FPpart denom  = (FPpart)1.0 / ((FPpart)1.0 + omdtsq);

                // solve the position equation
                FPpart ut = Pu[tid] + qomdt2 * Exl;
                FPpart vt = Pv[tid] + qomdt2 * Eyl;
                FPpart wt = Pw[tid] + qomdt2 * Ezl;
                FPpart udotb = ut * Bxl + vt * Byl + wt * Bzl;

                // solve the velocity equation
                uptilde = (ut + qomdt2 * (vt * Bzl - wt * Byl + qomdt2 * udotb * Bxl)) * denom;
                vptilde = (vt + qomdt2 * (wt * Bxl - ut * Bzl + qomdt2 * udotb * Byl)) * denom;
                wptilde = (wt + qomdt2 * (ut * Byl - vt * Bxl + qomdt2 * udotb * Bzl)) * denom;

                // update position
                // write the intermediate results into the cached version of particle position to reduce slow global memory accesses
                Pxl = xptilde + uptilde * dto2;
                Pyl = yptilde + vptilde * dto2;
                Pzl = zptilde + wptilde * dto2;
            }

            // update final velocity
            // make sure that constant value is cast to the chosen precision to reduce fp64 operations
            Pu[tid] = (FPpart)2.0 * uptilde - Pu[tid];
            Pv[tid] = (FPpart)2.0 * vptilde - Pv[tid];
            Pw[tid] = (FPpart)2.0 * wptilde - Pw[tid];

            // update final position
            Pxl = xptilde + uptilde * dt_sub_cycling;
            Pyl = yptilde + vptilde * dt_sub_cycling;
            Pzl = zptilde + wptilde * dt_sub_cycling;

            // in the following conditional statements, the cached version of particle position is also used
            // both in comparisons and in computations, which reduces the amount of slow global memory acceses

            // X-DIRECTION: BC particles
            if (Pxl > Lx)
            {
                // PERIODIC
                if (PERIODICX == true)
                {
                    Pxl = Pxl - Lx;
                }

                // REFLECTING
                else
                {
                    Pu[tid] = -Pu[tid];
                    Pxl = (FPpart)2.0 * Lx - Pxl;
                }
            }

            if (Pxl < (FPpart)0.0)
            {
                // PERIODIC
                if (PERIODICX == true)
                {
                    Pxl = Pxl + Lx;
                }

                // REFLECTING
                else
                {
                    Pu[tid] = -Pu[tid];
                    Pxl = -Pxl;
                }
            }

            // Y-DIRECTION: BC particles
            if (Pyl > Ly)
            {
                // PERIODIC
                if (PERIODICY == true)
                {
                    Pyl = Pyl - Ly;
                }

                // REFLECTING
                else
                {
                    Pv[tid] = -Pv[tid];
                    Pyl = (FPpart)2.0 * Ly - Pyl;
                }
            }

            if (Pyl < (FPpart)0.0)
            {
                // PERIODIC
                if (PERIODICY == true)
                {
                    Pyl = Pyl + Ly;
                }

                // REFLECTING
                else
                {
                    Pv[tid] = -Pv[tid];
                    Pyl = -Pyl;
                }
            }

            // Z-DIRECTION: BC particles
            if (Pzl > Lz)
            {
                // PERIODIC
                if (PERIODICZ == true)
                {
                    Pzl = Pzl - Lz;
                }

                // REFLECTING
                else
                {
                    Pw[tid] = -Pw[tid];
                    Pzl = (FPpart)2.0 * Lz - Pzl;
                }
            }

            if (Pzl < (FPpart)0.0)
            {
                // PERIODIC
                if (PERIODICZ == true)
                {
                    Pzl = Pzl + Lz;
                }

                // REFLECTING
                else
                {
                    Pw[tid] = -Pw[tid];
                    Pzl = -Pzl;
                }
            }
        }

        // this is the end of the kernel, and now the cached version of the particle position must be written back to global memory
        Px[tid] = Pxl;
        Py[tid] = Pyl;
        Pz[tid] = Pzl;
    }
}

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];

    // index of the cell
    int ix, iy, iz;

    for (register long long i = 0; i < part->nop; i++)
    {
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));

        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];

        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;


        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;


        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;


        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;

        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    }
}
