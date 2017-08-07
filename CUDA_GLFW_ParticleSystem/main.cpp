#define GLEW_DYNAMIC
#include <GL/glew.h>
#include "Common.h"
#include <GLFW/glfw3.h>
#include "Camera.hpp"
#include "ParticleSystem.h"
#include "Renderer.h"
#include "Scene.hpp"
#include <cuda_profiler_api.h>

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

using namespace std;

static const int width = 1280;
static const int height = 720;
static const GLfloat lastX = (width / 2);
static const GLfloat lastY = (height / 2);
static float deltaTime = 0.0f;
static float lastFrame = 0.0f;
static int frameCounter = -1;
static int w = 0;
static bool paused = true;
static bool spacePressed = false;

GLFWwindow* makeGLFWwindow();
void simulationRoutine(GLFWwindow* window);
void handleInput(GLFWwindow* window, ParticleSystem &system, Camera &cam);
void mainUpdate(ParticleSystem& system, Renderer& renderer, Camera& cam, solverParams& params);

int main() {

	// crtdbg : ���������[�N���Ȃ����ǂ����̊m�F
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	// �f�o�C�X�̎w��B�ǂ�CUDA�R�}���h������ɌĂԕK�v����
	// http://shouyu.hatenablog.com/entry/2011/12/05/192410
	cudaGLSetGLDevice(0);
	
	GLFWwindow* window = makeGLFWwindow();

	simulationRoutine(window);

	glfwTerminate();

	return 0;
}

GLFWwindow* makeGLFWwindow() {
	if (!glfwInit()) exit(EXIT_FAILURE);

	// This function sets hints for the next call to glfwCreateWindow. 
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	GLFWwindow* window = glfwCreateWindow(width, height, "Snow Simulation", nullptr, nullptr);
	if (!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);	//Set callbacks for keyboard and mouse

	glewExperimental = GL_TRUE;
	glewInit();

	glGetError();
	glViewport(0, 0, width, height);	//Define the viewport dimensions

	return window;
}

void simulationRoutine(GLFWwindow* window) {

	vector<Particle> particles;	// �p�[�e�B�N���f�[�^�̏�����

	CurlNoise scene("CurlNoise");
	solverParams sp;
	scene.init(particles, &sp);	// �\���o�[�̃p�����[�^�����߂� & �p�[�e�B�N���f�[�^�̐���
	
	ParticleSystem system = ParticleSystem(particles, sp);	// �p�[�e�B�N���̃��������m�ہA�p�[�e�B�N�������f�o�C�X�ɃR�s�[

	Camera cam = Camera();
	cam.eye = glm::vec3(10, 0, 0);	// �J�����ʒu

	Renderer renderer = Renderer(width, height, &sp);
	renderer.setProjection(glm::infinitePerspective(cam.zoom, float(width) / float(height), 0.1f));
	// �p�[�e�B�N���`��̂��߂̃o�b�t�@��p�� + CUDA�f�o�C�X��VBO�f�[�^�������N�H
	renderer.initSnowBuffers(sp.numParticles);

	system.updateWrapper(sp); //Take 1 step for initialization

	while (!glfwWindowShouldClose(window)) {
		//Set frame times
		float currentFrame = float(glfwGetTime());
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		//Check and call events
		glfwPollEvents();
		handleInput(window, system, cam);

		//Step physics and render
		mainUpdate(system, renderer, cam, sp);

		//Swap back buffers
		glfwSwapBuffers(window);

		if (!paused) {
			if (frameCounter % (int)(1 / (sp.deltaT * 30 * 3)) == 0) {
				cout << lastFrame << endl;
			}
		}

		//glfwSetCursorPos(window, lastX, lastY);
	}
}

void handleInput(GLFWwindow* window, ParticleSystem &system, Camera &cam) {
	// Esc : �I��
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);

	// W : �O�i
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cam.wasdMovement(FORWARD, deltaTime);

	// S : ��i
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cam.wasdMovement(BACKWARD, deltaTime);

	// D : �E�i
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cam.wasdMovement(RIGHT, deltaTime);
	
	// A : ���i
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cam.wasdMovement(LEFT, deltaTime);

	// Shift : �㏸
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		cam.wasdMovement(UP, deltaTime);

	// Ctrl : ���~
	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
		cam.wasdMovement(DOWN, deltaTime);

	// Space : �V�~�����[�V�����J�n or �ꎞ��~
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		spacePressed = true;

	// �V�~�����[�V�����̈ꎞ��~����
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE) {
		if (spacePressed) {
			spacePressed = false;
			if (paused) {
				cout << "Running Simulation..." << endl;
			} else {
				cout << "Pausing Simulation..." << endl;
			}
			paused = !paused;
		}
	}

	// �}�E�X�ł��J��������ł��� : �������Ȃ�?
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
		cam.mouseMovement((float(xpos) - lastX), (lastY - float(ypos)), deltaTime);
}

void mainUpdate(ParticleSystem& system, Renderer& renderer, Camera& cam, solverParams& params) {

	//Step physics
	if (!paused) {
		system.updateWrapper(params);
		frameCounter++;
	}

	// Update the VBO
	// GL��CUDA�̘A�g
	// cudaGraphicsMapResources()�AcudaGraphicsResourceGetMappedPointer()�ŃJ�[�l���֐����瑀��ł���|�C���^�𓾂�B
	// ������cudaGraphicsUnmapResources()�Ń}�b�s���O�������B
	void* positionsPtr;	//�^�͕�����Ȃ����
	cudaCheck(cudaGraphicsMapResources(1, &renderer.resource));
	size_t size;
	cudaGraphicsResourceGetMappedPointer(&positionsPtr, &size, renderer.resource);	// OpenGL�Ő��������o�b�t�@�̃|�C���^���擾

	// 
	system.getPositionsWrapper((float*)positionsPtr);	// float[]�ɃL���X�g���Acuda�̌v�Z���ʂ��擾(numParticles * 3 (x,y,z))
	cudaGraphicsUnmapResources(1, &renderer.resource, 0);

	//Render
	renderer.render(cam);
}