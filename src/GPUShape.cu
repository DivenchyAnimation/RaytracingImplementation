#include "GPUShape.h"
#include <math.h>


HD GPUHit computeIntersection(const GPUShape *s, const GPURay &ray, const mat4 modelMat, const mat4 modelMatInv) {
	GPUHit hit;
	hit.collision = false;
	switch (s->type) {
		case GPUShapeType::SPHERE: {

			// Help from ChatGPT making this model to object space transformation

			// Transform ray origin into local space (use homogeneous coordinate 1)
			vec3 localOrigin = vec3(modelMatInv * vec4(ray.rayOrigin, 1.0f));

			// Transform ray direction into local space (use homogeneous coordinate 0)
			vec3 localDirection = GPUnormalize(vec3(modelMatInv * vec4(ray.rayDirection, 0.0f)));

			// Compute quadratic equation
			float a, b, c, discriminant;
			vec3 oc = localOrigin;
			a = GPUdot(localDirection, localDirection);
			b = 2.0f * GPUdot(oc, localDirection);
			c = GPUdot(oc, oc) - s->data.SPHERE.radius * s->data.SPHERE.radius;
			discriminant = b * b - 4 * a * c;

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
			vec3 worldNormal = GPUnormalize(vec3(worldNormal4));

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
			hit.collision = true;
		}
	}
	return hit;
};