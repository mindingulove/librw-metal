// Translates the engine's GLSL (as written for the OpenGL 3 backend) to MSL at run time:
// glslang preprocesses and compiles to SPIR-V, SPIRV-Cross writes the MSL.
// Loose uniforms become one std140 block per stage, varyings get matching locations by name.
#include "mtlshadertrans.h"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <SPIRV/GlslangToSpv.h>
#include <spirv_cross/spirv_msl.hpp>

#include <map>
#include <regex>
#include <sstream>
#include <mutex>

namespace rw {
namespace mtl {

static std::string
concat(const char **src)
{
	std::string s;
	for(; src && *src; src++) s += *src;
	return s;
}

static void
initGlslang(void)
{
	static std::once_flag once;
	std::call_once(once, []{ glslang::InitializeProcess(); });
}

static bool
preprocess(EShLanguage stage, const std::string &src, std::string &out, std::string &err)
{
	glslang::TShader sh(stage);
	const char *s = src.c_str();
	sh.setStrings(&s, 1);
	sh.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
	sh.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
	sh.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_3);
	glslang::TShader::ForbidIncluder inc;
	if(!sh.preprocess(GetDefaultResources(), 450, ENoProfile, false, false, EShMsgDefault, &out, inc)){
		err = std::string("preprocess: ") + sh.getInfoLog();
		return false;
	}
	return true;
}

struct StageText {
	std::string head;		// #version / #extension lines
	std::string body;		// the rest, transformed
	std::vector<std::pair<std::string, std::string>> uniforms;	// type, "name" or "name[N]"
	std::vector<std::string> samplers;	// names
	std::vector<std::string> samplerTypes;
};

// Moves loose uniforms out of the text, gives varyings locations. varyLoc maps varying names to locations.
static void
transformStage(const std::string &pre, bool isVertex, std::map<std::string, int> &varyLoc, StageText &st)
{
	static const std::regex uniformRe(R"(^\s*uniform\s+(?:highp\s+|mediump\s+|lowp\s+)?(\w+)\s+(\w+)\s*(\[\s*\d+\s*\])?\s*;\s*$)");
	static const std::regex varyRe(R"(^\s*(?:layout\s*\(\s*location\s*=\s*(\d+)\s*\)\s*)?(flat\s+|smooth\s+|noperspective\s+)?(in|out)\s+(?:highp\s+|mediump\s+|lowp\s+)?(\w+)\s+(\w+)\s*(\[\s*\d+\s*\])?\s*;\s*$)");
	std::istringstream in(pre);
	std::string line;
	std::string body;
	bool headDone = false;
	while(std::getline(in, line)){
		if(!headDone){
			if(line.compare(0, 8, "#version") == 0 || line.compare(0, 10, "#extension") == 0){
				st.head += line + "\n";
				continue;
			}
			if(line.find_first_not_of(" \t\r") != std::string::npos)
				headDone = true;
			else
				continue;
		}
		std::smatch m;
		if(std::regex_match(line, m, uniformRe)){
			std::string type = m[1], name = m[2], arr = m[3];
			if(type.compare(0, 7, "sampler") == 0){
				st.samplers.push_back(name);
				st.samplerTypes.push_back(type);
			}else
				st.uniforms.push_back({ type, name + arr.c_str() });
			continue;
		}
		if(std::regex_match(line, m, varyRe)){
			bool out = m[3] == "out";
			std::string name = m[5];
			bool explicitLoc = m[1].matched;
			bool isFragOut = !isVertex && out;
			bool isVertIn = isVertex && !out;
			std::string interp = m[2].matched ? std::string(m[2]) : std::string();
			std::string type = m[4], arr = m[6];
			int loc;
			if(explicitLoc)
				loc = atoi(std::string(m[1]).c_str());
			else if(isFragOut)
				loc = 0;
			else if(isVertIn)
				loc = 0;
			else{
				// vertex outputs and fragment inputs share locations by name
				auto it = varyLoc.find(name);
				if(it == varyLoc.end()){
					int n = (int)varyLoc.size();
					it = varyLoc.insert({ name, n }).first;
				}
				loc = it->second;
			}
			body += "layout(location = " + std::to_string(loc) + ") " + interp + (out ? "out " : "in ") + type + " " + name + arr + ";\n";
			continue;
		}
		body += line + "\n";
	}
	st.body = body;
}

static int
mslSlotSort(const void*, const void*) { return 0; }

static bool
buildStage(EShLanguage stage, StageText &st, std::vector<unsigned> &spirv, std::string &err)
{
	std::string text = st.head;
	int binding = 0;
	if(!st.uniforms.empty()){
		text += "layout(std140, set = 0, binding = 0) uniform StageBlock {\n";
		for(auto &u : st.uniforms)
			text += "\t" + u.first + " " + u.second + ";\n";
		text += "};\n";
	}
	for(size_t i = 0; i < st.samplers.size(); i++)
		text += "layout(set = 1, binding = " + std::to_string(i) + ") uniform " + st.samplerTypes[i] + " " + st.samplers[i] + ";\n";
	(void)binding;
	text += st.body;

	glslang::TShader sh(stage);
	const char *s = text.c_str();
	sh.setStrings(&s, 1);
	sh.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
	sh.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_1);
	sh.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_3);
	if(!sh.parse(GetDefaultResources(), 450, false, EShMsgDefault)){
		err = std::string("parse: ") + sh.getInfoLog() + "\n" + text;
		return false;
	}
	glslang::TProgram prog;
	prog.addShader(&sh);
	if(!prog.link(EShMsgDefault)){
		err = std::string("link: ") + prog.getInfoLog();
		return false;
	}
	glslang::GlslangToSpv(*prog.getIntermediate(stage), spirv);
	(void)mslSlotSort;
	return true;
}

static bool
toMsl(std::vector<unsigned> &spirv, bool isVertex, StageText &st, int stageIdx, std::string &msl, std::string &entry,
      int &blockSize, TransProgram &prog)
{
	spirv_cross::CompilerMSL c(spirv);
	spirv_cross::CompilerMSL::Options o;
	o.set_msl_version(2, 3);
	o.platform = spirv_cross::CompilerMSL::Options::macOS;
	o.enable_decoration_binding = true;
	c.set_msl_options(o);
	auto common = c.get_common_options();
	common.vertex.fixup_clipspace = isVertex;	// GL clip space z -1..1 -> Metal 0..1
	c.set_common_options(common);
	auto model = isVertex ? spv::ExecutionModelVertex : spv::ExecutionModelFragment;

	if(!st.uniforms.empty()){
		spirv_cross::MSLResourceBinding b;
		b.stage = model; b.desc_set = 0; b.binding = 0;
		b.msl_buffer = isVertex ? TRANS_VS_UNIFORM_BUFFER : TRANS_FS_UNIFORM_BUFFER;
		c.add_msl_resource_binding(b);
	}
	for(size_t i = 0; i < st.samplers.size(); i++){
		spirv_cross::MSLResourceBinding b;
		b.stage = model; b.desc_set = 1; b.binding = (uint32_t)i;
		b.msl_texture = (uint32_t)i; b.msl_sampler = (uint32_t)i;
		c.add_msl_resource_binding(b);
	}
	try{
		msl = c.compile();
	}catch(std::exception &e){
		prog.error = std::string("spirv-cross: ") + e.what();
		return false;
	}
	if(isVertex){
		// run-time y flip of the clip space position (render to texture keeps OpenGL's texture orientation)
		size_t u = msl.find("using namespace metal;\n");
		if(u != std::string::npos) msl.insert(u + 23, "\nconstant bool kFlipY [[function_constant(0)]];\n");
		size_t r = msl.rfind("    return out;\n}");
		if(r != std::string::npos) msl.insert(r, "    if(kFlipY) out.gl_Position.y = -out.gl_Position.y;\n");
		// attribute locations the shader reads
		size_t pos = 0;
		while((pos = msl.find("[[attribute(", pos)) != std::string::npos){
			pos += 12;
			prog.vsAttribs.push_back(atoi(msl.c_str() + pos));
		}
	}
	entry = c.get_entry_points_and_stages()[0].name;
	// the entry point of a Metal library is renamed "main0" by SPIRV-Cross when it is called main
	if(entry == "main") entry = "main0";

	blockSize = 0;
	auto res = c.get_shader_resources();
	for(auto &ub : res.uniform_buffers){
		auto &type = c.get_type(ub.base_type_id);
		blockSize = (int)c.get_declared_struct_size(type);
		for(uint32_t i = 0; i < type.member_types.size(); i++){
			TransUniform u;
			u.name = c.get_member_name(ub.base_type_id, i);
			u.stage = stageIdx;
			u.offset = (int)c.type_struct_member_offset(type, i);
			u.size = (int)c.get_declared_struct_member_size(type, i);
			prog.uniforms.push_back(u);
		}
	}
	for(size_t i = 0; i < st.samplers.size(); i++)
		prog.samplers.push_back({ st.samplers[i], stageIdx, (int)i });
	return true;
}

bool
translateProgram(const char **vsrc, const char **fsrc, TransProgram &out)
{
	initGlslang();
	std::string vs = concat(vsrc), fs = concat(fsrc);
	// the OpenGL backend declares #version 330 (or 150) first; glslang's Vulkan environment wants 450
	auto fixVersion = [](std::string &s){
		size_t p = s.find("#version");
		if(p != std::string::npos){
			size_t e = s.find('\n', p);
			s.replace(p, e - p, "#version 450");
		}
	};
	fixVersion(vs); fixVersion(fs);

	std::string vpre, fpre;
	if(!preprocess(EShLangVertex, vs, vpre, out.error)) return false;
	if(!preprocess(EShLangFragment, fs, fpre, out.error)) return false;

	std::map<std::string, int> varyLoc;
	StageText vst, fst;
	transformStage(vpre, true, varyLoc, vst);
	transformStage(fpre, false, varyLoc, fst);

	std::vector<unsigned> vspv, fspv;
	if(!buildStage(EShLangVertex, vst, vspv, out.error)){ out.error = "vertex " + out.error; return false; }
	if(!buildStage(EShLangFragment, fst, fspv, out.error)){ out.error = "fragment " + out.error; return false; }
	if(!toMsl(vspv, true, vst, 0, out.vsMsl, out.vsEntry, out.vsBlockSize, out)) return false;
	if(!toMsl(fspv, false, fst, 1, out.fsMsl, out.fsEntry, out.fsBlockSize, out)) return false;
	return true;
}

}
}
