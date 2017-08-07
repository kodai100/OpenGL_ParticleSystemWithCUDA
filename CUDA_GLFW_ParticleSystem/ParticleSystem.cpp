#include "ParticleSystem.h"
#include "Simulation.cuh"

using namespace std;

ParticleSystem::ParticleSystem(vector<Particle>& p, solverParams& params) {

	solverParams sp = params;

	cout << "Solver parameters:" << endl;
	cout << "dt: " << sp.deltaT << endl;
	cout << "gravity: " << "(" << sp.gravity.x << ", " << sp.gravity.y << ", " << sp.gravity.z << ")" << endl;
	cout << "numParticles: " << sp.numParticles << endl;
	cout << "radius: " << sp.radius << endl;

	// デバイスメモリの確保
	// cudaMalloc(void ** devPtr, size_t size);
	cudaCheck(cudaMalloc((void**)&this->particles, params.numParticles * sizeof(Particle)));

	// 初期化したパーティクルデータをデバイスメモリへコピー
	// cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
	// cudaMemcpy(d_hello, hello, 32 , cudaMemcpyHostToDevice);
	cudaCheck(cudaMemcpy(this->particles, &p[0], params.numParticles * sizeof(Particle), cudaMemcpyHostToDevice));
}

ParticleSystem::~ParticleSystem() {
	cudaCheck(cudaFree(particles));	// 必ず解放
}

void ParticleSystem::updateWrapper(solverParams& params) {
	// cuファイルのグローバルスタティック変数に値をセット
	setParams(&params);	// CUDA側の関数
	update(particles);	// CUDA側の関数	CUDA側で確保したパーティクルポインタ
}

void ParticleSystem::getPositionsWrapper(float3* positionsPtr) {
	// cudaのメモリから取得してマッピングする
	getPositions(positionsPtr, particles);	// CUDA側の関数
}