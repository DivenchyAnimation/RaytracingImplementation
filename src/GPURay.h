#pragma once

struct GPURay {
  vec3 rayOrigin;
  vec3 rayDirection;
  HD GPURay(vec3 rayOrigin, vec3 rayDirection) : rayOrigin(rayOrigin), rayDirection(rayDirection) {}
};
