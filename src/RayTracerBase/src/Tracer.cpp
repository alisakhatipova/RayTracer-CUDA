#include "Tracer.h"
#include "Timer.h"

using namespace glm;

void MakeRay(uint pixelPosx, uint pixelPosy, SCamera &m_camera, CImage *stars, CImage *disk)
{
        float PI = 3.141592;
        float d_t = 28;
        float G = 6.674e-11;
        //vec3 color_disk(0, 0, 0);
        vec3 color(0, 0, 0);
        //float alpha = 0;
        float c = 3e+8;
        float hole_mass = 8.57e+36;
        float hole_rad =  2 * G * hole_mass/ (c * c);
        int disk_coeff = 5;

        SRay ray;
        ray.m_start = m_camera.m_pos;
        float angle = m_camera.m_viewAngle * (PI / 180.0f);
        float angle_tan = tan(angle);
        float h_half = m_camera.m_resolution.y / 2, w_half = m_camera.m_resolution.x / 2;
        float h = m_camera.m_resolution.y, w =  m_camera.m_resolution.x;
        float dist = h_half / angle_tan;
        vec3 ViewDir = dist * m_camera.m_forward;
        vec3 Up = m_camera.m_up * h;
        vec3 Right = m_camera.m_right * w;
        float carry1 = (pixelPosx + 0.5) * 1.0f/w - 0.5, carry2 = (pixelPosy + 0.5) * 1.0f/h - 0.5;
        ray.m_dir = ViewDir + carry1 * Right + carry2 * Up;
        vec3 norm = normalize(ray.m_dir);
        ray.m_dir = norm * c;
        while (1) {
            vec3 r = -ray.m_start;
            float r_len = length(r);
            float carry1 = G * hole_mass;
            carry1 /= r_len; carry1 /= r_len; carry1 /= r_len;
            vec3 a = r * carry1;
            vec3 oldstart = ray.m_start;
            vec3 olddir = ray.m_dir;
            ray.m_dir += a * d_t;
            vec3 carry = normalize(ray.m_dir) * c;
            ray.m_dir = carry;
            ray.m_start += olddir * d_t + a *( d_t * d_t / 2);
            float cos_beetw = dot(normalize(olddir), normalize(ray.m_dir));
            SRay ray1;
            ray1.m_dir = normalize(ray.m_start - oldstart);
            ray1.m_start = oldstart;
            float t1 = RaySphereIntersection(ray1, vec3(0, 0, 0), hole_rad);

            float t = RayFlatnessIntersection(ray1);
            if ((t > 0) &&(length(ray1.m_dir * t) < length(ray.m_start - oldstart)) && (length(ray1.m_start + ray1.m_dir * t) < hole_rad * disk_coeff)&& (length(ray1.m_start + ray1.m_dir * t) >= hole_rad )) {
                int w = disk->GetWidth(), h = disk->GetHeight();
                vec3 pos = ray1.m_start + ray1.m_dir * t;
                float xx = pos.x / (hole_rad * disk_coeff);
                xx *= w/2;
                xx += w/2;
                float yy = pos.y / (hole_rad * disk_coeff);
                yy *= h/2;
                yy += h/2;
                unsigned char* pix = static_cast<unsigned char*>(disk->GetPixelAddress(xx,yy));
                if (pix[3] == 0)
                    continue;
                //color_disk = vec3(pix[2], pix[1], pix[0])/255.0f;
                //alpha = pix[3] * 1.0f / 255;
                //break;
                color= glm::vec3(pix[2], pix[1], pix[0])/255.0f;
                break;

            }
            if ((t1 > 0) &&(length(ray1.m_dir * t1) < length(ray.m_start - oldstart)))
                break;

            if ((abs(cos_beetw) > 0.99999) && (length(oldstart) < length(ray.m_start)) ){
                ray.m_dir = normalize(ray.m_dir);
                float phi = (atan2(ray.m_dir.x, ray.m_dir.y)+PI) / (2 * PI);
                float teta = (asin(ray.m_dir.z) + PI/2) / PI;
                int w = stars->GetWidth(), h = stars->GetHeight();
                int ii = w * phi, jj = h * teta;
				if (ii >= w) ii = 0;
				if (jj >= h) jj = 0;
                unsigned char* pix = static_cast<unsigned char*>(stars->GetPixelAddress(ii, jj));
                color = vec3(pix[2], pix[1], pix[0])/255.0f;
                break; 
            }
        }
        //color = alpha * color_disk + (1 - alpha) * color;
        m_camera.m_pixels[pixelPosy * w + pixelPosx] = color;
}


float RayFlatnessIntersection(SRay ray){
    float t = -1;
	if (length(ray.m_dir) < 0.0000001) {
		return -1;
	}
    vec3 norm = normalize(ray.m_dir);
    ray.m_dir = norm;
    if (abs(ray.m_dir.z) > 1e-9)
        t = -ray.m_start.z / ray.m_dir.z;
    return t;
}


float RaySphereIntersection(SRay ray, vec3 spos, float r)
{
  float t = -1;
	if (length(ray.m_dir) < 0.0000001) {
		return -1;
	}
  vec3 norm = normalize(ray.m_dir);
  ray.m_dir = norm;
  vec3 k = ray.m_start - spos;
  float b = dot(k,ray.m_dir);
  float c = dot(k,k) - r*r;
  float d = b*b - c;

  if(d >=0)
  {
    float sqrtfd = sqrtf(d);
    // t, a == 1
    float t1 = -b + sqrtfd;
    float t2 = -b - sqrtfd;

    float min_t  = min(t1,t2);
    float max_t = max(t1,t2);

    t = (min_t >= 0) ? min_t : max_t;
  }
  return t;
}



void CTracer::RenderImage(int xRes, int yRes)
{
  // Reading input texture sample	
	CImage *stars;
	CImage *disk;
 disk = LoadImageFromFile("data/disk_32.png");
  stars = LoadImageFromFile("data/stars.png");
 // earth = LoadImageFromFile("data/fire.png"); */
  // Rendering
	Timer t;
	t.start();
  m_camera.m_resolution = uvec2(xRes, yRes);
  m_camera.m_pixels.resize(xRes * yRes);

  for(int i = 0; i < yRes; i++)
    for(int j = 0; j < xRes; j++)
    {
                        MakeRay(j, i, m_camera, stars, disk);
    }
	t.check("Not a CUDA result");
}

void CTracer::SaveImageToFile(std::string fileName)
{
  CImage image;

  int width = m_camera.m_resolution.x;
  int height = m_camera.m_resolution.y;

  image.Create(width, height, 24);

	int pitch = image.GetPitch();
	unsigned char* imageBuffer = (unsigned char*)image.GetBits();

	if (pitch < 0)
	{
		imageBuffer += pitch * (height - 1);
		pitch =- pitch;
	}

	int i, j;
	int imageDisplacement = 0;
	int textureDisplacement = 0;

	for (i = 0; i < height; i++)
	{
    for (j = 0; j < width; j++)
    {
      vec3 color = m_camera.m_pixels[textureDisplacement + j];

      imageBuffer[imageDisplacement + j * 3] = clamp(color.b, 0.0f, 1.0f) * 255.0f;
      imageBuffer[imageDisplacement + j * 3 + 1] = clamp(color.g, 0.0f, 1.0f) * 255.0f;
      imageBuffer[imageDisplacement + j * 3 + 2] = clamp(color.r, 0.0f, 1.0f) * 255.0f;
    }

		imageDisplacement += pitch;
		textureDisplacement += width;
	}

  image.Save(fileName.c_str());
	image.Destroy();
}

CImage* LoadImageFromFile(std::string fileName)
{
  CImage* pImage = new CImage;

  if(SUCCEEDED(pImage->Load(fileName.c_str())))
    return pImage;
  else
  {
    delete pImage;
    return NULL;
  }
}
