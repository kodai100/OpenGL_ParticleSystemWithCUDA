#ifndef SIMULATION_CU
#define SIMULATION_CU

#include "Common.h"
#include "Parameters.h"
#include "Particle.h"

#define cudaCheck(x) { cudaError_t err = x; if (err != cudaSuccess) { printf("Cuda error: %f in %s at %s:%f\n", err, #x, __FILE__, __LINE__); assert(0); } }

using namespace std;

// �z�X�g���̕ϐ�
static dim3 particleDims;
static const int blockSize = 128;

// �R���X�^���g�������B�S�X���b�h����A�N�Z�X�\�����f�o�C�X����ύX�͂ł��Ȃ��B
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

	// �f�o�C�X���̃O���[�o���ϐ�(�R���X�^���g������)�ɃR�s�[
	cudaCheck(cudaMemcpyToSymbol(sp, params, sizeof(solverParams)));	// ���̃G���[�͖�������OK
}

__host__ void getPositions(float3* positionsPtr, Particle* particles) {
	getPos<<<particleDims, blockSize>>>(positionsPtr, particles);
}

#endif