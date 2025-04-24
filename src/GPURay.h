#pragma once

struct GPURay {
  vec3 rayOrigin;
  vec3 rayDirection;
  GPURay(vec3 rayOrigin, vec3 rayDirection) : rayOrigin(rayOrigin), rayDirection(rayDirection) {}
};
