#ifndef SCENE_H
#define SCENE_H

#include "Common.h"
#include "Parameters.h"
#include "SetupFunctions.hpp"
#include "Particle.h"

class Scene {
public:
	Scene(std::string name) : name(name) {}
	virtual void init(std::vector<Particle>& particles, solverParams* sp) {
		sp->deltaT = 5e-5f;
		sp->radius = 0.017f;
		sp->gravity = make_float3(0, -9.8f, 0);
	}

private:
	std::string name;
};

class CurlNoise : public Scene {

public:
	CurlNoise(std::string name) : Scene(name) {}

	virtual void init(std::vector<Particle>& particles, solverParams* sp) {
		Scene::init(particles, sp);
		const float restDistance = sp->radius * 1.0f;

		int3 snowDims = make_int3(40);

		// パーティクル数はこの中(SetupFunctions.hpp)で決定
		createParticleBox(particles);

		sp->numParticles = int(particles.size());
	}
};

#endif