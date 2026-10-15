#pragma once
#include <string>
#include <vector>

namespace rw {
namespace mtl {

// A loose GLSL uniform that the translator moved into the stage's uniform block.
struct TransUniform {
	std::string name;
	int stage;	// 0 vertex, 1 fragment
	int offset;	// bytes inside the stage's block
	int size;	// bytes (array stride included)
};

struct TransSampler {
	std::string name;
	int stage;
	int slot;	// texture and sampler index in the MSL
};

// Result of translating one GLSL vertex/fragment pair to Metal Shading Language.
struct TransProgram {
	std::string vsMsl, fsMsl;
	std::string vsEntry, fsEntry;
	int vsBlockSize, fsBlockSize;	// bytes of the uniform blocks (0: none)
	std::vector<TransUniform> uniforms;
	std::vector<TransSampler> samplers;
	std::string error;
};

// Fixed buffer slots used by the generated MSL.
enum { TRANS_VS_UNIFORM_BUFFER = 15, TRANS_FS_UNIFORM_BUFFER = 0 };

// vsrc/fsrc: the same null terminated string arrays the OpenGL backend hands to glShaderSource.
bool translateProgram(const char **vsrc, const char **fsrc, TransProgram &out);

}
}
