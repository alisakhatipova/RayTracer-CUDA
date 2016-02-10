#pragma once

#include "glm/glm.hpp"
#include "Types.h"
#include "Scene.h"

#include "string"
#include "atlimage.h"

class CTracer
{
public:
	void RenderImage(int xRes, int yRes);
	void SaveImageToFile(std::string fileName);

public:
	SCamera m_camera;
	CScene* m_pScene;
};

void MakeRay(uint xpos, uint ypos, SCamera &m_camera, CImage *stars, CImage *disk);  // Create ray for specified pixel

	float RayFlatnessIntersection(SRay ray);
	float RaySphereIntersection(SRay ray, glm::vec3 spos, float t);
	CImage* LoadImageFromFile(std::string fileName);