#ifndef RW_SKINPOSITIONS_H
#define RW_SKINPOSITIONS_H
#include <stdint.h>
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <atomic>
#endif
namespace rw { namespace skinutil {
// Match the vertex shader's four weighted bone transforms; matrices are column-major, positions float3.
inline bool positions(float *out, const float *in, int count, const float *weights,
                      const uint8_t *indices, const float *bones, int boneCount)
{
 for(int v = 0; v < count; v++){
  float x = in[v*3], y = in[v*3+1], z = in[v*3+2];
  float p[3] = {};
  for(int k = 0; k < 4; k++){
   float w = weights[v*4+k];
   if(w == 0.0f) continue;
   unsigned b = indices[v*4+k];
   if(b >= (unsigned)boneCount) return false;
   const float *m = bones + b*16;
   for(int c = 0; c < 3; c++) p[c] += (m[c]*x + m[4+c]*y + m[8+c]*z + m[12+c])*w;
  }
  for(int c = 0; c < 3; c++) out[v*3+c] = p[c];
 }
 return true;
}
// Only independent vertex arithmetic runs on workers. Engine/Metal state stays on its owner thread.
inline bool positionsParallel(float *out, const float *in, int count, const float *weights,
                              const uint8_t *indices, const float *bones, int boneCount)
{
#ifdef __APPLE__
 if(count >= 16384){
  struct Work {
   float *out; const float *in, *weights; const uint8_t *indices; const float *bones;
   int count, boneCount; std::atomic<bool> valid{true};
  } work;
  work.out=out;work.in=in;work.weights=weights;work.indices=indices;work.bones=bones;
  work.count=count;work.boneCount=boneCount;
  dispatch_apply_f(4,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),&work,[](void *context,size_t job){
   auto &w=*static_cast<Work*>(context);
   int begin=int(job)*w.count/4,end=int(job+1)*w.count/4;
   if(!positions(w.out+begin*3,w.in+begin*3,end-begin,w.weights+begin*4,w.indices+begin*4,w.bones,w.boneCount))
    w.valid.store(false,std::memory_order_relaxed);
  });
  return work.valid.load(std::memory_order_relaxed);
 }
#endif
 return positions(out,in,count,weights,indices,bones,boneCount);
}
} }
#endif
