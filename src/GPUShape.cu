#include "GPUShape.h"
#include <math.h>
#include "GPUraytri.h"

HD GPUHit intersectionSphere(const GPUShape *s, const GPURay &ray, const mat4 &modelMat, const mat4 &modelMatInv, GPUHit &hit) {
	// Rays are real, local orig and dir are real
	// Help from ChatGPT making this model to object space transformation
	//printf("rayOrigin = (%f, %f, %f)\n", ray.rayOrigin.x, ray.rayOrigin.y, ray.rayOrigin.z);
	//printf("rayDir    = (%f, %f, %f)\n", ray.rayDirection.x, ray.rayDirection.y, ray.rayDirection.z);

	// Transform ray origin into local space (use homogeneous coordinate 1)
	vec3 localOrigin = vec3(modelMatInv * vec4(ray.rayOrigin, 1.0f));


	// Transform ray direction into local space (use homogeneous coordinate 0)
	vec4 tmp = modelMatInv * vec4(ray.rayDirection, 0.0f);
	vec3 localDirection = GPUnormalize(vec3(tmp.x, tmp.y, tmp.z));
	//vec3 localDirection = GPUnormalize(vec3(modelMatInv * vec4(ray.rayDirection, 0.0f)));  <<--- this old version div by 0 therefore garbage

	// Compute quadratic equation
	float a, b, c, discriminant;
	vec3 oc = localOrigin;
	a = GPUdot(localDirection, localDirection);
	b = 2.0f * GPUdot(oc, localDirection);
	c = GPUdot(oc, oc) - s->data.SPHERE.radius * s->data.SPHERE.radius;
	discriminant = b * b - 4 * a * c;

	//printf("Intersect: a=%f, b=%f, c=%f, disc=%f\n", a, b, c, discriminant);

	// If discriminant is negative, no collision with shape
	if (discriminant < 0) {
		return hit;
	}

	// Else, colllision so solve for t (intersections)
	float e = 1e-3;
	float sqrtDiscriminant = GPUsqrtf(discriminant);
	float t = (-b - sqrtDiscriminant) / (2.0f * a);

	// Behind camera, count as miss look at other t val
	if (t < e) {
		t = (-b + sqrtDiscriminant) / (2.0f * a);
	}
	// If both neg, skip
	if (t < e) {
		return hit;
	}

	// Object space ray hit
	vec3 localHitPoint = localOrigin + t * localDirection;
	vec3 localNormal = GPUnormalize(localHitPoint);

	// World space ray hit (dividie by w to enter eye space)
	vec4 worldHitPoint4 = modelMat * vec4(localHitPoint, 1.0f);
	vec3 worldHitPoint = vec3(worldHitPoint4) / worldHitPoint4.w;

	// Transform normal to world space
	mat4 invTransModel = GPUtranspose(modelMatInv);
	vec4 worldNormal4 = invTransModel * vec4(localNormal, 0.0f);
	// Same div by 0 thing again
	vec3 worldNormal = vec3(worldNormal4.x, worldNormal4.y, worldNormal4.z);
	worldNormal = GPUnormalize(worldNormal);

	// Calc t in world space
	float t_world = GPUlength(worldHitPoint - ray.rayOrigin);
	if (GPUdot(ray.rayDirection, worldHitPoint - ray.rayOrigin) < 0.0f) {
		t_world = -t_world;
	}

	// Ray hit
	hit.t = t_world;
	vec3 P = worldHitPoint; // Intersection point
	vec3 N = worldNormal;   // Normal at intersection point
	hit.x = P;
	hit.n = N;
	hit.collision = true;
	return hit;

}

HD GPUHit intersectionEllipse(const GPUShape *s, const GPURay &ray, const mat4 &modelMat, const mat4 &modelMatInv, GPUHit &hit) {
	// Help from ChatGPT making this model to object space transformation

	// Transform ray origin into local space (use homogeneous coordinate 1)
	vec3 localOrigin = vec3(modelMatInv * vec4(ray.rayOrigin, 1.0f));

	// Transform ray direction into local space (use homogeneous coordinate 0)
	vec4 tmp = modelMatInv * vec4(ray.rayDirection, 0.0f);
	vec3 localDirection = GPUnormalize(vec3(tmp.x, tmp.y, tmp.z));
	//vec3 localDirection = GPUnormalize(vec3(modelMatInv * vec4(ray.rayDirection, 0.0f)));  <<--- this old version div by 0 therefore garbage

	// Compute quadratic equation
	float a, b, c, discriminant;
	vec3 oc = localOrigin;
	a = GPUdot(localDirection, localDirection);
	b = 2.0f * GPUdot(oc, localDirection);
	c = GPUdot(oc, oc) - s->data.ELLIPSOID.radius * s->data.ELLIPSOID.radius;
	discriminant = b * b - 4 * a * c;

	// If discriminant is negative, no collision with shape
	if (discriminant < 0) {
		return hit;
	}

	// Else, colllision so solve for t (intersections)
	float sqrtDiscriminant = sqrt(discriminant);
	float t = (-b - sqrtDiscriminant) / (2.0f * a);

	// Behind camera, count as miss look at other t val
	if (t < 0) {
		t = (-b + sqrtDiscriminant) / (2.0f * a);
	}
	// If both neg, skip
	if (t < 0) {
		return hit;
	}

	// Object space ray hit
	vec3 localHitPoint = localOrigin + t * localDirection;
	vec3 localNormal = GPUnormalize(localHitPoint); // For sphere

	// World space ray hit (dividie by w to enter eye space)
	vec4 worldHitPoint4 = modelMat * vec4(localHitPoint, 1.0f);
	vec3 worldHitPoint = vec3(worldHitPoint4) / worldHitPoint4.w;

	// Transform normal to world space
	mat4 invTransModel = GPUtranspose(modelMatInv);
	vec4 worldNormal4 = invTransModel * vec4(localNormal, 0.0f);
	// Same div by 0 thing again
	vec3 worldNormal = vec3(worldNormal4.x, worldNormal4.y, worldNormal4.z);
	worldNormal = GPUnormalize(worldNormal);

	// Calc t in world space
	float t_world = GPUlength(worldHitPoint - ray.rayOrigin);
	if (GPUdot(ray.rayDirection, worldHitPoint - ray.rayOrigin) < 0.0f) {
		t_world = -t_world;
	}

	// Ray hit
	hit.t = t_world;
	vec3 P = worldHitPoint; // Intersection point
	vec3 N = worldNormal;   // Normal at intersection point
	hit.x = P;
	hit.n = N;

	// Add all lights contribution to the color
	// hit->color = calcLightContribution(lights, P, N, shape, ray);
	hit.collision = true;

	return hit;
}

HD GPUHit intersectionPlane(const GPUShape *s, const GPURay &ray, const mat4 &modelMat, const mat4 &modelMatInv, GPUHit &hit) {
  // For an infinite plane, we assume the plane passes through shape->getPosition()
  vec3 planePos = s->position;
  vec3 worldNormal = GPUnormalize(mat3(GPUtoMat3AndTranspose(modelMatInv)) * s->data.PLANE.normal);

  //printf("World normal: %f, %f, %f\n", worldNormal.x, worldNormal.y, worldNormal.z);
  float denom = GPUdot(worldNormal, ray.rayDirection);
  // Avoid div by zero
  if (fabs(denom) > 1e-6) {
    // Compute t: distance along ray to intersection
    float t = GPUdot(planePos - ray.rayOrigin, worldNormal) / denom;
    if (t >= 0.0f) { // only count intersections in front of the ray
      vec3 P = ray.rayOrigin + t * ray.rayDirection;

      hit.t = t;
      hit.x = P;
      hit.n = worldNormal; // constant normal for an infinite plane
      hit.collision = true;
    }
  }
  return hit;

}

HD GPUHit intersectionMesh(const GPUShape *s, const GPURay &ray, const mat4 &modelMat, const mat4 &modelMatInv, GPUHit &hit) {
  // Help from ChatGPT making this model to object space transformation
  // Transform ray origin into local space (use homogeneous coordinate 1)
  vec3 rayLocalOrigin = vec3(modelMatInv * vec4(ray.rayOrigin, 1.0f));

  // Transform ray direction into local space (use homogeneous coordinate 0)
  vec4 tmp = modelMatInv * vec4(ray.rayDirection, 0.0f);
  vec3 rayLocalDirection = GPUnormalize(vec3(tmp.x, tmp.y, tmp.z));
  //vec3 localDirection = GPUnormalize(vec3(modelMatInv * vec4(ray.rayDirection, 0.0f)));  <<--- this old version div by 0 therefore garbage

  GPURay localRay(rayLocalOrigin, rayLocalDirection);

  // Ray intersection test
  float t_bounding;
  if (!s->data.MESH.bSphere->GPUBoundingSphereIntersect(localRay, 0.001f, std::numeric_limits<float>::max(), t_bounding)) {
    return hit; // Misses the bunny
  }

  // Help from ChatGPT
  // Loop over the triangles
  float T = std::numeric_limits<float>::infinity();
  float U = 0, V = 0;
  size_t bestTri = size_t(-1);
  for (size_t i = 0; i < s->data.MESH.posBufSize; i += 9) {
    double orig[3] = {double(localRay.rayOrigin.x), double(localRay.rayOrigin.y), double(localRay.rayOrigin.z)};
    double dir[3] = {double(localRay.rayDirection.x), double(localRay.rayDirection.y), double(localRay.rayDirection.z)};
    double v0[3] = {s->data.MESH.GPUposBuf[i + 0], s->data.MESH.GPUposBuf[i + 1], s->data.MESH.GPUposBuf[i + 2]};
    double v1[3] = {s->data.MESH.GPUposBuf[i + 3], s->data.MESH.GPUposBuf[i + 4], s->data.MESH.GPUposBuf[i + 5]};
    double v2[3] = {s->data.MESH.GPUposBuf[i + 6], s->data.MESH.GPUposBuf[i + 7], s->data.MESH.GPUposBuf[i + 8]};
    double t, u, v;
    // Use the optimizer codes for triange ray intersections
    if (GPUintersect_triangle(orig, dir, v0, v1, v2, &t, &u, &v) && t > 0 && t < T) {
      T = float(t);
      U = float(u);
      V = float(v);
      bestTri = i / 9;
    }
  }

  // If after looping no triangle was hit
  if (bestTri == size_t(-1)) {
    return hit;
  }

  // Hit found
  vec3 P = localRay.rayOrigin + localRay.rayDirection * T; // local
  size_t idx = bestTri * 9;
  // Interpolate normal (local)
  vec3 n0(s->data.MESH.GPUnorBuf[idx + 0], s->data.MESH.GPUnorBuf[idx + 1], s->data.MESH.GPUnorBuf[idx + 2]), n1(s->data.MESH.GPUnorBuf[idx + 3], s->data.MESH.GPUnorBuf[idx + 4], s->data.MESH.GPUnorBuf[idx + 5]),
      n2(s->data.MESH.GPUnorBuf[idx + 6], s->data.MESH.GPUnorBuf[idx + 7], s->data.MESH.GPUnorBuf[idx + 8]);
  vec3 N = GPUnormalize(n0 * (1 - U - V) + n1 * U + n2 * V); // local
  // now in world space
  vec4 P_w = modelMat * vec4(P, 1.0f);
  P = vec3(P_w / P_w.w);                                     // world
  mat3 invTrans3 = GPUtoMat3AndTranspose(modelMatInv);
  vec3 worldNormal = GPUnormalize(invTrans3 * N);
  N = worldNormal; // world

  hit.collision = true;
  hit.t = GPUlength(P - ray.rayOrigin);
  hit.x = P;
  hit.n = N;
  return hit;

};

HD GPUHit computeIntersection(const GPUShape *s, const GPURay &ray, const mat4 &modelMat, const mat4 &modelMatInv) {
	GPUHit hit;
	hit.collision = false;
	switch (s->type) {
	case GPUShapeType::SPHERE: {
		intersectionSphere(s, ray, modelMat, modelMatInv, hit);
		break;
	}
	case GPUShapeType::ELLIPSOID: {
		intersectionEllipse(s, ray, modelMat, modelMatInv, hit);
		break;
	}
	case GPUShapeType::PLANE: {
		intersectionPlane(s, ray, modelMat, modelMatInv, hit);
		break;
	}
	case GPUShapeType::MESH: {
		intersectionMesh(s, ray, modelMat, modelMatInv, hit);
		break;
	}
	}
	return hit;
};