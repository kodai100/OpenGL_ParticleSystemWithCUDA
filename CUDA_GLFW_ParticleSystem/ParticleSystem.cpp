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

	// �f�o�C�X�������̊m��
	// cudaMalloc(void ** devPtr, size_t size);
	cudaCheck(cudaMalloc((void**)&this->particles, params.numParticles * sizeof(Particle)));

	// �����������p�[�e�B�N���f�[�^���f�o�C�X�������փR�s�[
	// cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
	// cudaMemcpy(d_hello, hello, 32 , cudaMemcpyHostToDevice);
	cudaCheck(cudaMemcpy(this->particles, &p[0], params.numParticles * sizeof(Particle), cudaMemcpyHostToDevice));
}

ParticleSystem::~ParticleSystem() {
	cudaCheck(cudaFree(particles));	// �K�����
}

void ParticleSystem::updateWrapper(solverParams& params) {
	// cu�t�@�C���̃O���[�o���X�^�e�B�b�N�ϐ��ɒl���Z�b�g
	setParams(&params);	// CUDA���̊֐�
	update(particles);	// CUDA���̊֐�
}

void ParticleSystem::getPositionsWrapper(float* positionsPtr) {
	// cuda�̃���������擾���ă}�b�s���O����
	getPositions(positionsPtr, particles);	// CUDA���̊֐�
}