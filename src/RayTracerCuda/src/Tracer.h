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
//	CImage *stars;
//	CImage *disk;
//	CImage *earth;

public:
	SCamera m_camera;
	CScene* m_pScene;
};

	CImage* LoadImageFromFile(std::string fileName);