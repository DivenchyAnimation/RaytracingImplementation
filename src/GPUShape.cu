#include "GPUShape.h"


__device__ GPUHit* computeIntersection(const GPUShape *s, const GPURay &ray, const mat4 modelMat, const mat4 modelMatInv,
	const GPULight *lights) {
	GPUHit *hit = NULL; // assume collision is false
	switch (s->type) {
		case GPUShapeType::SPHERE: {

			// Help from ChatGPT making this model to object space transformation

			// Transform ray origin into local space (use homogeneous coordinate 1)
			//glm::vec3 localOrigin = glm::vec3(modelMatInv * glm::vec4(ray.rayOrigin, 1.0f));

			// Transform ray direction into local space (use homogeneous coordinate 0)
			//glm::vec3 localDirection = glm::normalize(glm::vec3(modelMatInv * glm::vec4(ray.rayDirection, 0.0f)));

			// Compute quadratic equation
			//float a, b, c, discriminant;
			// vec3 oc = localOrigin;  //uncomment
			//a = glm::dot(localDirection, localDirection);  // uncomment
			//b = 2.0f * glm::dot(oc, localDirection); //uncomment
			//c = glm::dot(oc, oc) -  s->data.SPHERE.radius * s->data.SPHERE.radius; //uncooment
			//discriminant = b * b - 4 * a * c; //unc

			// If discriminant is negative, no collision with shape
			//discriminant = 0;
			//if (discriminant < 0) {
			//  return hit;
			//}

			// Else, colllision so solve for t (intersections)
			//float sqrtDiscriminant = sqrt(discriminant);
			//float t = (-b - sqrtDiscriminant) / (2.0f * a);

			// Behind camera, count as miss look at other t val  //unc section
			//if (t < 0) {
			//  t = (-b + sqrtDiscriminant) / (2.0f * a);
			//}
			//// If both neg, skip
			//if (t < 0) {
			//  return hit;
			//}

			// Object space ray hit
			//vec3 localHitPoint = localOrigin + t * localDirection; // unc
			//vec3 localNormal = glm::normalize(localHitPoint); // For sphere   //unc

			// World space ray hit (dividie by w to enter eye space)
			//vec4 worldHitPoint4 = modelMat * glm::vec4(localHitPoint, 1.0f); //unc
			//vec3 worldHitPoint = glm::vec3(worldHitPoint4) / worldHitPoint4.w; //unc

			// Transform normal to world space
			//glm::mat4 invTransModel = glm::transpose(modelMatInv);		// unc
			//glm::vec4 worldNormal4 = invTransModel * glm::vec4(localNormal, 0.0f); //unc
			//glm::vec3 worldNormal = glm::normalize(glm::vec3(worldNormal4));  //unc

			// Calc t in world space
			//float t_world = glm::length(worldHitPoint - ray.rayOrigin); //unc
			//if (glm::dot(ray.rayDirection, worldHitPoint - ray.rayOrigin) < 0.0f) {  //unc
			//  t_world = -t_world;
			//}

			// Ray hit  //unc section
			//hit->t = t_world;
			//vec3 P = worldHitPoint; // Intersection point
			//vec3 N = worldNormal;   // Normal at intersection point
			//hit->x = P;
			//hit->n = N;

			// Add all lights contribution to the color
			// hit->color = calcLightContribution(lights, P, N, shape, ray);
			//hit->collision = true;

		}
	}
	return hit;
};