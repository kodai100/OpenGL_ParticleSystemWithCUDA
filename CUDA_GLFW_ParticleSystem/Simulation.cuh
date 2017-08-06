#ifndef SIMULATION_CUH
#define SIMULATION_CUH

// CUDA�w�b�_�t�@�C���ɋL�q�����֐���C++�̂悤�ɌĂяo�����Ƃ��ł���

#include "Parameters.h"
#include "Particle.h"

void update(Particle* particles);
void getPositions(float* positionsPtr, Particle* particles);
void setParams(solverParams *tempParams);

#endif