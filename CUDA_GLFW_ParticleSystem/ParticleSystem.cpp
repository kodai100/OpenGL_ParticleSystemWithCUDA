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

	// �f�o�C�X�������̊m��
	cudaCheck(cudaMalloc((void**)&this->particles, params.numParticles * sizeof(Particle)));

	// �f�o�C�X�������փR�s�[
	cudaCheck(cudaMemcpy(this->particles, &particles[0], params.numParticles * sizeof(Particle), cudaMemcpyHostToDevice));
}

ParticleSystem::~ParticleSystem() {
	cudaCheck(cudaFree(particles));
}

void ParticleSystem::updateWrapper(solverParams& params) {
	// cu�t�@�C���̃O���[�o���X�^�e�B�b�N�ϐ��ɒl���Z�b�g
	setParams(&params);	// CUDA
	update(particles);	// CUDA
}

void ParticleSystem::getPositionsWrapper(float* positionsPtr) {
	// cuda�̃���������擾���ă}�b�s���O����
	getPositions(positionsPtr, particles);	// CUDA
}