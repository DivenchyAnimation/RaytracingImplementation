// src/main.cpp
#include "Routines.h"
#include "pch.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace std;

shared_ptr<Camera> camera = make_shared<Camera>();
vector<shared_ptr<Material>> materials;
vector<shared_ptr<Hit>> hits;
vector<shared_ptr<Shape>> shapes;
vector<glm::vec3> rays;

int SCENE;
enum class SHAPE { CUBE, SPHERE, ELLIPSOID };

void init() {}

int main(int argc, char **argv) {
  if (argc < 4) {
    cout << "Usage: ./A6 <SCENE> <IMAGE SIZE> <IMAGE FILENAME>" << endl;
    return 0;
  }

  SCENE = atoi(argv[1]);

  int width = atoi(argv[2]);
  int height = width;
  char *FILENAME = argv[3];

  // Init materials
  initMaterials(materials);

  // Load geometry
  vector<float> posBuf; // list of vertex positions
  vector<float> zBuf;   // list of z-vals for image
  vector<float> norBuf; // list of vertex normals
  vector<float> texBuf; // list of vertex texture coords
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapesTINY;
  std::vector<tinyobj::material_t> materialsTINY;
  string warnStr, errStr;
  string meshName = "../resources/bunny.obj";

  //Load the OBJ file
  bool rc = tinyobj::LoadObj(&attrib, &shapesTINY, &materialsTINY, &warnStr, &errStr, meshName.c_str());
  if (!rc) {
    cerr << errStr << endl;
  } else {
    // Some OBJ files have different indices for vertex positions, normals,
    // and texture coordinates. For example, a cube corner vertex may have
    // three different normals. Here, we are going to duplicate all such
    // vertices.
    // Loop over shapes
    for (size_t s = 0; s < shapesTINY.size(); s++) {
      // Loop over faces (polygons)
      size_t index_offset = 0;
      for (size_t f = 0; f < shapesTINY[s].mesh.num_face_vertices.size(); f++) {
        size_t fv = shapesTINY[s].mesh.num_face_vertices[f];
        // Loop over vertices in the face.
        for (size_t v = 0; v < fv; v++) {
          // access to vertex
          tinyobj::index_t idx = shapesTINY[s].mesh.indices[index_offset + v];
          posBuf.push_back(attrib.vertices[3 * idx.vertex_index + 0]);
          posBuf.push_back(attrib.vertices[3 * idx.vertex_index + 1]);
          posBuf.push_back(attrib.vertices[3 * idx.vertex_index + 2]);
          if (!attrib.normals.empty()) {
            norBuf.push_back(attrib.normals[3 * idx.normal_index + 0]);
            norBuf.push_back(attrib.normals[3 * idx.normal_index + 1]);
            norBuf.push_back(attrib.normals[3 * idx.normal_index + 2]);
          }
          if (!attrib.texcoords.empty()) {
            texBuf.push_back(attrib.texcoords[2 * idx.texcoord_index + 0]);
            texBuf.push_back(attrib.texcoords[2 * idx.texcoord_index + 1]);
          }
        }
        index_offset += fv;
        // per-face material (IGNORE)
        shapesTINY[s].mesh.material_ids[f];
      }
    }
  }
  cout << "Number of vertices: " << posBuf.size() / 3 << endl;

  // Choose operation based on task
  switch (SCENE) {
  case 1:
    sceneOne(width, height, materials, shapes, hits, camera, FILENAME);
    break;
  case 2:
    sceneOne(width, height, materials, shapes, hits, camera, FILENAME);
    break;
  case 3:
    sceneThree(width, height, materials, shapes, hits, camera, FILENAME);
    break;
  case 4:
    sceneReflections(width, height, materials, shapes, hits, camera, FILENAME, SCENE);
    break;
  case 5:
    sceneReflections(width, height, materials, shapes, hits, camera, FILENAME, SCENE);
    break;
  case 6:
    sceneMesh(width, height, materials, shapes, hits, camera, FILENAME, posBuf, zBuf, norBuf, texBuf);
    break;
  case 7:
    sceneMeshTransform(width, height, materials, shapes, hits, camera, FILENAME, posBuf, zBuf, norBuf, texBuf);
    break;
  case 8:
    sceneCameraTransform(width, height, materials, shapes, hits, camera, FILENAME);
    break;
  };

  return 0;
}
