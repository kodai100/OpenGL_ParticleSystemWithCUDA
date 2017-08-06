#include "ParticleSystem.h"
#include "Simulation.cuh"

using namespace std;

ParticleSystem::ParticleSystem(vector<Particle>& particles, solverParams& params) {

	solverParams sp = params;

	cout << "Solver parameters:" << endl;
	cout << "dt: " << sp.deltaT << endl;
	cout << "gravity: " << "(" << sp.gravity.x << ", " << sp.gravity.y << ", " << sp.gravity.z << ")" << endl;
	cout << "numParticles: " << sp.numParticles << endl;
	cout << "radius: " << sp.radius << endl;

	// デバイスメモリの確保
	cudaCheck(cudaMalloc((void**)&this->particles, params.numParticles * sizeof(Particle)));

	// デバイスメモリへコピー
	cudaCheck(cudaMemcpy(this->particles, &particles[0], params.numParticles * sizeof(Particle), cudaMemcpyHostToDevice));
}

ParticleSystem::~ParticleSystem() {
	cudaCheck(cudaFree(particles));
}

void ParticleSystem::updateWrapper(solverParams& params) {
	// cuファイルのグローバルスタティック変数に値をセット
	setParams(&params);	// CUDA
	update(particles);	// CUDA
}

void ParticleSystem::getPositionsWrapper(float* positionsPtr) {
	// cudaのメモリから取得してマッピングする
	getPositions(positionsPtr, particles);	// CUDA
}