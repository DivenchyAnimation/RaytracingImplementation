#include "GPUShape.h"
#include <math.h>

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
	float sqrtDiscriminant = GPUsqrtf(discriminant);
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
	}

	return hit;
};