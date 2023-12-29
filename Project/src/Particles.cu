#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

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

    // allocate position on cpu
    cudaHostAlloc(&part->x, sizeof(FPpart) * npmax, cudaHostAllocDefault);
    cudaHostAlloc(&part->y, sizeof(FPpart) * npmax, cudaHostAllocDefault);
    cudaHostAlloc(&part->z, sizeof(FPpart) * npmax, cudaHostAllocDefault);

    // allocate velocity on cpu
    cudaHostAlloc(&part->u, sizeof(FPpart) * npmax, cudaHostAllocDefault);
    cudaHostAlloc(&part->v, sizeof(FPpart) * npmax, cudaHostAllocDefault);
    cudaHostAlloc(&part->w, sizeof(FPpart) * npmax, cudaHostAllocDefault);

    // allocate position on gpu
    cudaMalloc(&part->x_gpu, sizeof(FPpart) * npmax);
    cudaMalloc(&part->y_gpu, sizeof(FPpart) * npmax);
    cudaMalloc(&part->z_gpu, sizeof(FPpart) * npmax);

    // allocate velocity on gpu
    cudaMalloc(&part->u_gpu, sizeof(FPpart) * npmax);
    cudaMalloc(&part->v_gpu, sizeof(FPpart) * npmax);
    cudaMalloc(&part->w_gpu, sizeof(FPpart) * npmax);

    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
}

/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate position on cpu
    cudaFree(part->x);
    cudaFree(part->y);
    cudaFree(part->z);

    // deallocate velocity on cpu
    cudaFree(part->u);
    cudaFree(part->v);
    cudaFree(part->w);

    // deallocate position on gpu
    cudaFree(part->x_gpu);
    cudaFree(part->y_gpu);
    cudaFree(part->z_gpu);

    // deallocate velocity on gpu
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
    
    int stride  = part->nop / 4;
	int threads = 64;
	int blocks  = (stride + threads - 1) / threads;

    // partition the particle array into 4 equally sized partitions, which will be distributed equally amongst the available streams
    for (int i = 0; i != 4; ++i)
    {
        // dispatch gpu computation of particle movement simulation
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
                                                           i * stride);

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
	                            FPpart qom, double dt, double c,
	                            FPfield invdx, FPfield invdy, FPfield invdz, FPfield invVOL,
	                            double xStart, double yStart, double zStart,
	                            double Lx, double Ly, double Lz,
	                            bool PERIODICX, bool PERIODICY, bool PERIODICZ,
	                            int nxn, int nyn, int nzn,
	                            int nop, int nsc, int nim,
                                int offset)
{
    // global thread index for particle arrays
	int tid = threadIdx.x + blockIdx.x * blockDim.x + offset;

	if (tid < nop)
	{
        // auxiliary variables
		FPpart dt_sub_cycling = dt / (double)nsc;
		FPpart dto2 = 0.5 * dt_sub_cycling;
		FPpart qomdt2 = qom * dto2 / c;

        // intermediate particle position and velocity
		FPpart xptilde, yptilde, zptilde;
		FPpart uptilde, vptilde, wptilde;

        // start subcycling
		for (int i = 0; i != nsc; ++i)
		{
			xptilde = Px[tid];
			yptilde = Py[tid];
			zptilde = Pz[tid];

            // calculate the average velocity iteratively
			for (int j = 0; j != nim; ++j)
			{
                // interpolation G-->P
				int ix = 2 + int((Px[tid] - xStart) * invdx);
				int iy = 2 + int((Py[tid] - yStart) * invdy);
				int iz = 2 + int((Pz[tid] - zStart) * invdz);

				FPfield weight[8];
				FPfield xi[2], eta[2], zeta[2];

                // calculate densities
				xi[0]   = Px[tid] - Nx[get_idx(ix - 1, iy, iz, nyn, nzn)];
				eta[0]  = Py[tid] - Ny[get_idx(ix, iy - 1, iz, nyn, nzn)];
				zeta[0] = Pz[tid] - Nz[get_idx(ix, iy, iz - 1, nyn, nzn)];
				xi[1]   = Nx[get_idx(ix, iy, iz, nyn, nzn)] - Px[tid];
				eta[1]  = Ny[get_idx(ix, iy, iz, nyn, nzn)] - Py[tid];
				zeta[1] = Nz[get_idx(ix, iy, iz, nyn, nzn)] - Pz[tid];

                // calculate weights
				for (int u = 0; u != 2; ++u)
				{
					for (int v = 0; v != 2; ++v)
					{
						for (int w = 0; w != 2; ++w)
						{
							weight[get_idx(u, v, w, 2, 2)] = xi[u] * eta[v] * zeta[w] * invVOL;
						}
					}
				}

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
				FPpart denom  = 1.0 / (1.0 + omdtsq);

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
				Px[tid] = xptilde + uptilde * dto2;
				Py[tid] = yptilde + vptilde * dto2;
				Pz[tid] = zptilde + wptilde * dto2;
			}

			// update final velocity
			Pu[tid] = 2.0 * uptilde - Pu[tid];
			Pv[tid] = 2.0 * vptilde - Pv[tid];
			Pw[tid] = 2.0 * wptilde - Pw[tid];

			// update final position
			Px[tid] = xptilde + uptilde * dt_sub_cycling;
			Py[tid] = yptilde + vptilde * dt_sub_cycling;
			Pz[tid] = zptilde + wptilde * dt_sub_cycling;

            // X-DIRECTION: BC particles
			if (Px[tid] > Lx)
			{
                // PERIODIC
				if (PERIODICX == true)
				{
					Px[tid] = Px[tid] - Lx;
				}

                // REFLECTING
				else
				{
					Pu[tid] = -Pu[tid];
					Px[tid] = 2 * Lx - Px[tid];
				}
			}

			if (Px[tid] < 0)
			{
                // PERIODIC
				if (PERIODICX == true)
				{
					Px[tid] = Px[tid] + Lx;
				}

                // REFLECTING
				else
				{
					Pu[tid] = -Pu[tid];
					Px[tid] = -Px[tid];
				}
			}

            // Y-DIRECTION: BC particles
			if (Py[tid] > Ly)
			{
                // PERIODIC
				if (PERIODICY == true)
				{
					Py[tid] = Py[tid] - Ly;
				}

                // REFLECTING
				else
				{
					Pv[tid] = -Pv[tid];
					Py[tid] = 2 * Ly - Py[tid];
				}
			}

			if (Py[tid] < 0)
			{
                // PERIODIC
				if (PERIODICY == true)
				{
					Py[tid] = Py[tid] + Ly;
				}

                // REFLECTING
				else
				{
					Pv[tid] = -Pv[tid];
					Py[tid] = -Py[tid];
				}
			}

            // Z-DIRECTION: BC particles
			if (Pz[tid] > Lz)
			{
                // PERIODIC
				if (PERIODICZ == true)
				{
					Pz[tid] = Pz[tid] - Lz;
				}

                // REFLECTING
				else
				{
					Pw[tid] = -Pw[tid];
					Pz[tid] = 2 * Lz - Pz[tid];
				}
			}

			if (Pz[tid] < 0)
			{
                // PERIODIC
				if (PERIODICZ == true)
				{
					Pz[tid] = Pz[tid] + Lz;
				}

                // REFLECTING
				else
				{
					Pw[tid] = -Pw[tid];
					Pz[tid] = -Pz[tid];
				}
			}
		}
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
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
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
