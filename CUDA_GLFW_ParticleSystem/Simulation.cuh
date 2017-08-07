#ifndef SIMULATION_CUH
#define SIMULATION_CUH

// CUDAヘッダファイルに記述した関数はC++のように呼び出すことができる

#include "Parameters.h"
#include "Particle.h"

void update(Particle* particles);
void getPositions(float3* positionsPtr, Particle* particles);
void setParams(solverParams *tempParams);

#endif