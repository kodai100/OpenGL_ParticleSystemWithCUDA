#ifndef SETUP_FUNCTIONS_H
#define SETUP_FUNCTIONS_H

#include "Common.h"
#include "Parameters.h"
#include "Particle.h"
#include <time.h>

float3 insideUnitCube() {
	float3 pos = make_float3(10 * (rand() / static_cast <float>(RAND_MAX) * 2 - 1), 10 * (rand() / static_cast <float>(RAND_MAX) * 2 - 1), 10 * (rand() / static_cast <float>(RAND_MAX) * 2 - 1));
	return pos;
}

//Some method for creating a snowball
inline void createParticleBox(std::vector<Particle>& particles) {

	// パーティクルを生成
	for (int i = 0; i < 1000000; i++) {
		particles.push_back(Particle(insideUnitCube() * 0.1, insideUnitCube() * 10));
	}


}


#endif