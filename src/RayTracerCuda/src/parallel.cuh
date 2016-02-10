#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "Types.h"
	float RayFlatnessIntersection(SRay ray);
	float RaySphereIntersection(SRay ray, glm::vec3 spos, float t);
	void MakeRayCPU(SCamera &m_camera);
