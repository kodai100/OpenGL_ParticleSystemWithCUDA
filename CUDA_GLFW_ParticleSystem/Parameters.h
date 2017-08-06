#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "Common.h"

struct solverParams {
	float deltaT;
	float radius;
	int numParticles;
	float3 gravity;
};

#endif