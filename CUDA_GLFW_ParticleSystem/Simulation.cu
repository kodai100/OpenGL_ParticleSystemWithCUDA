#ifndef SIMULATION_CU
#define SIMULATION_CU

#include "Common.h"
#include "Parameters.h"
#include "Particle.h"

#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %f in %s at %s:%f\n", err, #x, __FILE__, __LINE__); assert(0); } }

using namespace std;

static dim3 particleDims;
static const int blockSize = 128;

__constant__ solverParams sp;

__global__ void updatePositions(Particle* particles) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	particles[index].pos += sp.deltaT * particles[index].velocity;
}

__global__ void getPos(float* positionsPtr, Particle* particles) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	positionsPtr[3 * index + 0] = particles[index].pos.x;
	positionsPtr[3 * index + 1] = particles[index].pos.y;
	positionsPtr[3 * index + 2] = particles[index].pos.z;
}


void update(Particle* particles) {
	updatePositions<<<particleDims, blockSize>>>(particles);
}

__host__ void setParams(solverParams *params) {
	particleDims = int(ceil(params->numParticles / blockSize + 0.5f));
	cudaCheck(cudaMemcpyToSymbol(sp, params, sizeof(solverParams)));	// Ç±ÇÃÉGÉâÅ[ÇÕñ≥éãÇµÇƒOK
}

__host__ void getPositions(float* positionsPtr, Particle* particles) {
	getPos<<<particleDims, blockSize>>>(positionsPtr, particles);
}

#endif