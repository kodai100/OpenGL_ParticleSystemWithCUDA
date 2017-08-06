#ifndef PARTICLE_H
#define PARTICLE_H

#include "Common.h"

struct Particle {
	float3 pos;
	float3 velocity;

	Particle(float3 pos, float3 velocity) :
		pos(pos), velocity(velocity)
	{}
};

#endif