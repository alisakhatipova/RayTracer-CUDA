#define GLM_FORCE_CUDA
#include <cuda.h>
#include "parallel.cuh"
#include "glm/glm.hpp"
#include "Tracer.h"
#include <iostream>
#include "Scene.h"
#include "Tracer.h"
#include <cstdlib>
#include "Types.h"
#include "Timer.h"
#include "atlimage.h"
__global__ void MakeRay(SCamera *m_camera, float *d_pixels0, float *d_pixels1, float*d_pixels2, int m, int n, SRay *rays)
{ 

	    int idx = blockIdx.x * blockDim.x + threadIdx.x;
		SRay ray, ray1;
		if (idx<m*n) {
			int pixelPosx = idx % m;
			int pixelPosy = idx / m;
			float PI = 3.141592;
			float d_t = 28;
			float G = 6.674e-11;
			//vec3 color_disk(0, 0, 0);
			glm::vec3 color(0, 0, 0);
			//float alpha = 0;
			float c = 3e+8;
			float hole_mass = 8.57e+36;
			float hole_rad =  2 * G * hole_mass/ (c * c);
			int disk_coeff = 5;

			//SRay ray;
			ray.m_start = m_camera->m_pos;
			float angle = m_camera->m_viewAngle * (PI / 180.0f);
			float angle_tan = tan(angle);
			float h_half = m_camera->m_resolution.y / 2, w_half = m_camera->m_resolution.x / 2;
			float h = m_camera->m_resolution.y, w =  m_camera->m_resolution.x;
			float dist = h_half / angle_tan;
			glm::vec3 ViewDir = dist * m_camera->m_forward;
			glm::vec3 Up = m_camera->m_up * h;
			glm::vec3 Right = m_camera->m_right * w;
			float carry1 = (pixelPosx + 0.5) * 1.0f/w - 0.5, carry2 = (pixelPosy + 0.5) * 1.0f/h - 0.5;
			ray.m_dir = ViewDir + carry1 * Right + carry2 * Up;
			if (length(ray.m_dir) < 0.000000001) return;
			glm::vec3 norm = ray.m_dir / length(ray.m_dir);
			ray.m_dir = norm * c;
			while(1) {
				glm::vec3 r = -ray.m_start;
				float r_len = length(r);
				float carry1 = G * hole_mass;
				carry1 /= r_len; carry1 /= r_len; carry1 /= r_len;
				glm::vec3 a = r * carry1;
				glm::vec3 oldstart = ray.m_start;
				glm::vec3 olddir = ray.m_dir;
				ray.m_dir += a * d_t;
				glm::vec3 carry = (ray.m_dir / length(ray.m_dir)) * c;
				ray.m_dir = carry;
				ray.m_start += olddir * d_t + a *( d_t * d_t / 2);
				float cos_beetw = dot(olddir/length(olddir), ray.m_dir/length(ray.m_dir));
				cos_beetw = cos_beetw * cos_beetw;
				//SRay ray1;
				ray1.m_dir = (ray.m_start - oldstart) / length(ray.m_start - oldstart);
				ray1.m_start = oldstart;
				float t3 =  RaySphereIntersection(ray1, vec3(0, 0, 0), hole_rad);
				float t = RayFlatnessIntersection(ray1);
				if ((t > 0))
					if((length(ray1.m_dir * t) < length(ray.m_start - oldstart)) && (length(ray1.m_start + ray1.m_dir * t) < hole_rad * disk_coeff)&& (length(ray1.m_start + ray1.m_dir * t) >= hole_rad )) {
				   SRay ray2; ray2.m_start = ray1.m_start; ray2.m_dir = ray1.m_dir * t;
					rays[pixelPosy * m + pixelPosx] = ray2;
						/* int w = disk->GetWidth(), h = disk->GetHeight();
					vec3 pos = ray1.m_start + ray1.m_dir * t;
					float xx = pos.x / (m_pScene->hole_rad * m_pScene->disk_coeff);
					xx *= w/2;
					xx += w/2;
					float yy = pos.y / (m_pScene->hole_rad * m_pScene->disk_coeff);
					yy *= h/2;
					yy += h/2;
					unsigned char* pix = static_cast<unsigned char*>(disk->GetPixelAddress(xx,yy));
					if (pix[3] == 0)
						continue;
					color_disk = vec3(pix[2], pix[1], pix[0])/255.0f;
					alpha = pix[3] * 1.0f / 255; */
					//break; 
					color= glm::vec3(-1, 0, 0);
					//color= glm::vec3(0.5, 0.5, 0.5);
					break;

				}
				if ((t3 > 0)) 
					if ((length(ray1.m_dir * t3) < length(ray.m_start - oldstart)))
						break;
						
				if ((abs(cos_beetw) > 0.99999) && (length(oldstart) < length(ray.m_start)) ){
					rays[pixelPosy * m + pixelPosx] = ray;
					/* ray.m_dir = normalize(ray.m_dir);
					float phi = (atan2(ray.m_dir.x, ray.m_dir.y)+PI) / (2 * PI);
					float teta = (asin(ray.m_dir.z) + PI/2) / PI;
					int w = stars->GetWidth(), h = stars->GetHeight();
					int ii = w * phi, jj = h * teta;
					if (ii >= w) ii = 0;
					if (jj >= h) jj = 0;
					unsigned char* pix = static_cast<unsigned char*>(stars->GetPixelAddress(ii, jj));
					color = vec3(pix[2], pix[1], pix[0])/255.0f;
					break; */
					color = glm::vec3(-2, 0, 0);
					//color = glm::vec3(0.8, 0.8, 0.8);
					break;
				}
			} 
			//color = alpha * color_disk + (1 - alpha) * color;
			//m_camera.m_pixels[pixelPosy * w + pixelPosx] = color;
			d_pixels0[pixelPosy * m + pixelPosx] = color.x;
			d_pixels1[pixelPosy * m + pixelPosx] = color.y;
			d_pixels2[pixelPosy * m + pixelPosx] = color.z; 
		} 
} 

__device__ float RayFlatnessIntersection(SRay ray){
    float t = -1;
	if (length(ray.m_dir) < 0.0000001) {
		return -1;
	}
    vec3 norm = ray.m_dir / length(ray.m_dir);
    ray.m_dir = norm;
    if (abs(ray.m_dir.z) > 1e-9)
        t = -ray.m_start.z / ray.m_dir.z;
    return t;
}


__device__ float RaySphereIntersection(SRay ray, vec3 spos, float r)
{ 
  float t = -1;
 if (length(ray.m_dir) < 0.0000001) {
		return -1;
	}
  vec3 norm = ray.m_dir / length(ray.m_dir);
  ray.m_dir = norm;
  vec3 k = ray.m_start - spos;
  float b = dot(k,ray.m_dir);
  float c = dot(k,k) - r*r;
  float d = b*b - c;

  if(d >=0)
  {
    float sqrtfd = sqrt(d);
    // t, a == 1
    float t1 = -b + sqrtfd;
    float t2 = -b - sqrtfd;

    float min_t  = t1; //min(t1,t2);
    float max_t = t2; //max(t1,t2);

    t = (min_t >= 0) ? min_t : max_t;
  } 
  return t;
}

void MakeRayCPU(SCamera &m_camera){
	
			float G = 6.674e-11;
			float PI = 3.141592;
			float c = 3e+8;
			float hole_mass = 8.57e+36;
	float hole_rad =  2 * G * hole_mass/ (c * c);
	int disk_coeff = 5;
 	CImage *stars;
	CImage *disk;
  disk = LoadImageFromFile("data/disk_32.png");
  stars = LoadImageFromFile("data/stars.png");
  Timer t;
  t.start();
  float h_pixels0[200 * 200], h_pixels1[200 * 200], h_pixels2[200 * 200];
  float *d_pixels0, *d_pixels1, *d_pixels2;
   SRay *rays, *hrays;
   hrays = new SRay [200*200];
  SCamera *d_camera;
  cudaMalloc( (void**)&rays, 200 * 200 * sizeof(SRay));
  cudaMalloc( (void**)&d_pixels0, 200 * 200 * sizeof(float));
  cudaMalloc( (void**)&d_pixels0, 200 * 200 * sizeof(float));
  cudaMalloc( (void**)&d_pixels1, 200 * 200 * sizeof(float));
  cudaMalloc( (void**)&d_pixels2, 200 * 200 * sizeof(float));
  cudaMalloc( (void**)&d_camera,  sizeof(SCamera));
  cudaMemcpy(d_camera, &m_camera,  sizeof(SCamera), cudaMemcpyHostToDevice);
  int COLS = 200, ROWS = 200;  
  int threadsPerBlock = 512;
  int blocksPerGrid =((ROWS * COLS + threadsPerBlock - 1) / threadsPerBlock);
 // MakeRay<<<blocksPerGrid, threadsPerBlock>>>(d_camera, d_pixels0, d_pixels1, d_pixels2, ROWS, COLS, d_ray, d_ray1);
  //MakeRay<<<COLS * ROWS, 1>>>(d_camera, d_pixels0, d_pixels1, d_pixels2, ROWS, COLS);
  MakeRay<<<blocksPerGrid, threadsPerBlock>>>(d_camera, d_pixels0, d_pixels1, d_pixels2, ROWS, COLS, rays);
  cudaMemcpy(h_pixels0, d_pixels0, 200 * 200 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pixels1, d_pixels1, 200 * 200 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pixels2, d_pixels2, 200 * 200 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(hrays, rays, 200 * 200 * sizeof(SRay), cudaMemcpyDeviceToHost);
  
  for (int i =0; i < 200*200; ++i){
	   if (h_pixels0[i] == -1){
		   			int w = disk->GetWidth(), h = disk->GetHeight();
					SRay ray1 = hrays[i];
					glm::vec3 pos = ray1.m_start + ray1.m_dir;
					float xx = pos.x / (hole_rad * disk_coeff);
					xx *= w/2;
					xx += w/2;
					float yy = pos.y / (hole_rad * disk_coeff);
					yy *= h/2;
					yy += h/2;
					unsigned char* pix = static_cast<unsigned char*>(disk->GetPixelAddress(xx,yy));
					if (pix[3] == 0)
						h_pixels0[i] = -2;
					m_camera.m_pixels[i] = vec3(pix[2], pix[1], pix[0])/255.0f;
	   }
	    if (h_pixels0[i] == -2){
					SRay ray = hrays[i];
					ray.m_dir = ray.m_dir/length(ray.m_dir);
					float phi = (atan2(ray.m_dir.x, ray.m_dir.y)+PI) / (2 * PI);
					float teta = (asin(ray.m_dir.z) + PI/2) / PI;
					int w = stars->GetWidth(), h = stars->GetHeight();
					int ii = w * phi, jj = h * teta;
					if (ii >= w) ii = 0;
					if (jj >= h) jj = 0;
					unsigned char* pix = static_cast<unsigned char*>(stars->GetPixelAddress(ii, jj));
					m_camera.m_pixels[i] = vec3(pix[2], pix[1], pix[0])/255.0f;
	   } 
	   if(h_pixels0[i] >=0) 
			m_camera.m_pixels[i] = vec3(h_pixels0[i], h_pixels1[i], h_pixels2[i]);
  }
  
  t.check("CUDA result");
}
