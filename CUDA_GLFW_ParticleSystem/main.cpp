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

	// crtdbg : メモリリークがないかどうかの確認
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

	// デバイスの指定。どのCUDAコマンドよりも先に呼ぶ必要あり
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

	vector<Particle> particles;	// パーティクルデータの初期化

	CurlNoise scene("CurlNoise");
	solverParams sp;
	scene.init(particles, &sp);	// ソルバーのパラメータを決める & パーティクルデータの生成
	
	ParticleSystem system = ParticleSystem(particles, sp);	// パーティクルのメモリを確保、パーティクル情報をデバイスにコピー

	Camera cam = Camera();
	cam.eye = glm::vec3(10, 0, 0);	// カメラ位置

	Renderer renderer = Renderer(width, height, &sp);
	renderer.setProjection(glm::infinitePerspective(cam.zoom, float(width) / float(height), 0.1f));	// プロジェクション変換行列の作成
	// パーティクル描画のためのバッファを用意 + CUDAデバイスにVBOデータをリンク？
	renderer.initParticleBuffers(sp.numParticles);

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
	// Esc : 終了
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);

	// W : 前進
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cam.wasdMovement(FORWARD, deltaTime);

	// S : 後進
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cam.wasdMovement(BACKWARD, deltaTime);

	// D : 右進
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cam.wasdMovement(RIGHT, deltaTime);
	
	// A : 左進
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cam.wasdMovement(LEFT, deltaTime);

	// Shift : 上昇
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		cam.wasdMovement(UP, deltaTime);

	// Ctrl : 下降
	if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
		cam.wasdMovement(DOWN, deltaTime);

	// Space : シミュレーション開始 or 一時停止
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		spacePressed = true;

	// シミュレーションの一時停止制御
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

	// マウスでもカメラ制御できる : 反応しない?
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
	// GLとCUDAの連携
	// cudaGraphicsMapResources()、cudaGraphicsResourceGetMappedPointer()でカーネル関数から操作できるポインタを得る。
	// 操作後はcudaGraphicsUnmapResources()でマッピングを解除。
	void* positionsPtr;	//型は分からない状態
	cudaCheck(cudaGraphicsMapResources(1, &renderer.resource));
	size_t size;
	cudaGraphicsResourceGetMappedPointer(&positionsPtr, &size, renderer.resource);	// OpenGLで生成したバッファのポインタを取得

	// 
	system.getPositionsWrapper((float*)positionsPtr);	// float[]にキャストし、cudaの計算結果を取得(numParticles * 3 (x,y,z))
	cudaGraphicsUnmapResources(1, &renderer.resource, 0);

	//Render
	renderer.render(cam);
}