#include "Renderer.h"

using namespace std;

static const float radius = 0.008f;

Renderer::Renderer(int width, int height, solverParams* sp) :
width(width),
height(height),
particleShader(Shader("snow.vert", "snow.frag"))
{
	this->sp = sp;
	aspectRatio = float(width) / float(height);
}

Renderer::~Renderer() {
	if (particleBuffers.vao != 0) {
		glDeleteVertexArrays(1, &particleBuffers.vao);
		glDeleteBuffers(1, &particleBuffers.positions);
	}
}

void Renderer::setProjection(glm::mat4 projection) {
	this->projection = projection;
}

void Renderer::initParticleBuffers(int numParticles) {
	glGenVertexArrays(1, &particleBuffers.vao);

	glGenBuffers(1, &particleBuffers.positions);
	glBindBuffer(GL_ARRAY_BUFFER, particleBuffers.positions);
	glBufferData(GL_ARRAY_BUFFER, numParticles * 3 * sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// glGenBuffers()�Ő�������VBO��ID��o�^�B
	// http://shouyu.hatenablog.com/entry/2011/12/05/192410
	cudaGraphicsGLRegisterBuffer(&resource, particleBuffers.positions, cudaGraphicsRegisterFlagsWriteDiscard);

	particleBuffers.numParticles = numParticles;
}

void Renderer::render(Camera& cam) {
	//Set model view matrix
	mView = cam.getMView();

	//Clear buffer
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Snow
	renderSnow(cam);
}

void Renderer::renderSnow(Camera& cam) {
	glUseProgram(particleShader.program);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	particleShader.setUniformmat4("mView", mView);
	particleShader.setUniformmat4("projection", projection);
	particleShader.setUniformf("pointRadius", radius);
	particleShader.setUniformf("pointScale", width / aspectRatio * (1.0f / tanf(cam.zoom * 0.5f)));

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_CULL_FACE);

	//Draw snow
	glBindVertexArray(particleBuffers.vao);
	glBindBuffer(GL_ARRAY_BUFFER, particleBuffers.positions);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	glDrawArrays(GL_POINTS, 0, GLsizei(particleBuffers.numParticles));
}