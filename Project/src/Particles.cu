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

    // allocate position
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];

    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];

    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
}

/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    return mover_PC_gpu(part, field, grd, param);
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
int mover_PC_gpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
	std::cout << "*** GPU MOVER with SUBCYCLYING " << param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

	FPpart  *Px, *Py, *Pz;
	FPpart  *Pu, *Pv, *Pw;
	FPfield *Bx, *By, *Bz;
	FPfield *Ex, *Ey, *Ez;
	FPfield *Nx, *Ny, *Nz;

    // allocate resources on the gpu
	cudaMalloc(&Px, sizeof(FPpart) * part->npmax);
	cudaMalloc(&Py, sizeof(FPpart) * part->npmax);
	cudaMalloc(&Pz, sizeof(FPpart) * part->npmax);
	cudaMalloc(&Pu, sizeof(FPpart) * part->npmax);
	cudaMalloc(&Pv, sizeof(FPpart) * part->npmax);
	cudaMalloc(&Pw, sizeof(FPpart) * part->npmax);
	cudaMalloc(&Bx, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
	cudaMalloc(&By, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
	cudaMalloc(&Bz, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
	cudaMalloc(&Ex, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
	cudaMalloc(&Ey, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
	cudaMalloc(&Ez, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
	cudaMalloc(&Nx, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
	cudaMalloc(&Ny, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);
	cudaMalloc(&Nz, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn);

    // copy data from host to device
	cudaMemcpy(Px, part->x, sizeof(FPpart) * part->npmax, cudaMemcpyHostToDevice);
	cudaMemcpy(Py, part->y, sizeof(FPpart) * part->npmax, cudaMemcpyHostToDevice);
	cudaMemcpy(Pz, part->z, sizeof(FPpart) * part->npmax, cudaMemcpyHostToDevice);
	cudaMemcpy(Pu, part->u, sizeof(FPpart) * part->npmax, cudaMemcpyHostToDevice);
	cudaMemcpy(Pv, part->v, sizeof(FPpart) * part->npmax, cudaMemcpyHostToDevice);
	cudaMemcpy(Pw, part->w, sizeof(FPpart) * part->npmax, cudaMemcpyHostToDevice);
	cudaMemcpy(Bx, field->Bxn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
	cudaMemcpy(By, field->Byn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
	cudaMemcpy(Bz, field->Bzn_flat, sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
	cudaMemcpy(Ex, field->Ex_flat,  sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
	cudaMemcpy(Ey, field->Ey_flat,  sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
	cudaMemcpy(Ez, field->Ez_flat,  sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
	cudaMemcpy(Nx, grd->XN_flat,    sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
	cudaMemcpy(Ny, grd->YN_flat,    sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);
	cudaMemcpy(Nz, grd->ZN_flat,    sizeof(FPfield) * grd->nxn * grd->nyn * grd->nzn, cudaMemcpyHostToDevice);

    // execute particle movement simulation on device
	int threads = 64;
	int blocks = (part->nop + threads - 1) / threads;
	kernel_mover_PC<<<blocks, threads>>>(Px, Py, Pz,
		                                 Pu, Pv, Pw,
		                                 Ex, Ey, Ez,
		                                 Bx, By, Bz,
		                                 Nx, Ny, Nz,
		                                 part->qom, param->dt, param->c,
		                                 grd->invdx, grd->invdy, grd->invdz, grd->invVOL,
		                                 grd->xStart, grd->yStart, grd->zStart,
		                                 grd->Lx, grd->Ly, grd->Lz,
		                                 param->PERIODICX, param->PERIODICY, param->PERIODICZ,
		                                 grd->nxn, grd->nyn, grd->nzn,
		                                 part->nop, part->n_sub_cycles, part->NiterMover);

    // wait for device to complete execution
	cudaDeviceSynchronize();

    // copy data from device to host
	cudaMemcpy(part->x, Px, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
	cudaMemcpy(part->y, Py, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
	cudaMemcpy(part->z, Pz, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
	cudaMemcpy(part->u, Pu, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
	cudaMemcpy(part->v, Pv, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);
	cudaMemcpy(part->w, Pw, sizeof(FPpart) * part->npmax, cudaMemcpyDeviceToHost);

    // deallocate resources on the gpu
	cudaFree(Px);
	cudaFree(Py);
	cudaFree(Pz);
	cudaFree(Pu);
	cudaFree(Pv);
	cudaFree(Pw);
	cudaFree(Bx);
	cudaFree(By);
	cudaFree(Bz);
	cudaFree(Ex);
	cudaFree(Ey);
	cudaFree(Ez);
	cudaFree(Nx);
	cudaFree(Ny);
	cudaFree(Nz);

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
	                            int nop, int nsc, int nim)
{
    // global thread index for particle arrays
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

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
