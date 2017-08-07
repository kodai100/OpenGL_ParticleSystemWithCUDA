#ifndef SIMULATION_CU
#define SIMULATION_CU

#include "Common.h"
#include "Parameters.h"
#include "Particle.h"

#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %f in %s at %s:%f\n", err, #x, __FILE__, __LINE__); assert(0); } }

using namespace std;

// ホスト側の変数
static dim3 particleDims;
static const int blockSize = 128;

// コンスタントメモリ。全スレッドからアクセス可能だがデバイスから変更はできない。
__constant__ solverParams sp;

__global__ void updatePositions(Particle* particles) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	// particles[index].velocity += sp.deltaT * sp.gravity;
	particles[index].pos += sp.deltaT * particles[index].velocity;
}

__global__ void getPos(float3* positionsPtr, Particle* particles) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= sp.numParticles) return;

	positionsPtr[index] = particles[index].pos;
}


void update(Particle* particles) {
	updatePositions<<<particleDims, blockSize>>>(particles);
}

__host__ void setParams(solverParams *params) {
	particleDims = int(ceil(params->numParticles / blockSize + 0.5f));

	// デバイス側のグローバル変数(コンスタントメモリ)にコピー
	cudaCheck(cudaMemcpyToSymbol(sp, params, sizeof(solverParams)));	// このエラーは無視してOK
}

__host__ void getPositions(float3* positionsPtr, Particle* particles) {
	getPos<<<particleDims, blockSize>>>(positionsPtr, particles);
}

#endif