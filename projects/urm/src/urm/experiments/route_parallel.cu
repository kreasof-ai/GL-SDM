// Isolated ordered SparseStateMixer experiment. No production dispatch.
// One CTA owns (partition, D=4 slice), one thread owns (route, dimension).
// Unique within-token routes are a REQUIRED certified-input precondition.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

constexpr int THREADS = 256;
constexpr int TILE_D = 4;

// Eight routes per warp; eight warp partials. All threads participate.
// This deliberately changes FP32 reduction order, not state cast boundaries.
__device__ float route_sum(float x, float* scratch) {
    #pragma unroll
    for (int stride = 4; stride <= 16; stride *= 2)
        x += __shfl_xor_sync(0xffffffff, x, stride);
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, d = lane % 4;
    if (lane < 4) scratch[warp * 4 + d] = x;
    __syncthreads();
    float total = 0;
    #pragma unroll
    for (int w = 0; w < 8; ++w) total += scratch[w * 4 + d];
    __syncthreads();  // protect scratch before its next use
    return total;
}

__device__ float dimension_sum(float x) {
    x += __shfl_xor_sync(0xffffffff, x, 1);
    x += __shfl_xor_sync(0xffffffff, x, 2);
    return x;
}

template<typename T, bool RESIDENT>
__device__ int address(int partition, int slot, int dim, int slots, int D) {
    return RESIDENT ? slot * TILE_D + dim % TILE_D
                    : (partition * slots + slot) * D + dim;
}

template<typename T, typename I, bool RESIDENT>
__device__ void read_forward(
    T* state, const I* ri, const T* rw, T* y, T* history,
    int p, int t, int S, int D, int Tlen, float* scratch) {
    int r = threadIdx.x / 4, d = threadIdx.x % 4 + blockIdx.y * 4;
    int route = (p * Tlen + t) * 64 + r;
    float selected = float(state[address<T, RESIDENT>(p, int(ri[route]), d, S, D)]);
    history[route * D + d] = T(selected);
    float result = route_sum(float(rw[route]) * selected, scratch);
    if (r == 0) y[(p * Tlen + t) * D + d] = T(result);
}

template<typename T, typename I, bool RESIDENT, bool BEFORE>
__global__ void ordered_forward(
    const T* initial, T* final, const I* wi, const T* ww, const T* values,
    const T* beta, const T* decay_log, const I* ri, const T* rw,
    T* y, T* wh, T* rh, int Tlen, int S, int D) {
    extern __shared__ __align__(16) unsigned char storage[];
    T* state = RESIDENT ? reinterpret_cast<T*>(storage) : final;
    __shared__ float scratch[32];
    int p = blockIdx.x, r = threadIdx.x / 4, local_d = threadIdx.x % 4;
    int d = blockIdx.y * 4 + local_d;
    if (RESIDENT) {
        for (int i = threadIdx.x; i < S * 4; i += THREADS)
            state[i] = initial[(p * S + i / 4) * D + blockIdx.y * 4 + i % 4];
        __syncthreads();
    }
    for (int t = 0; t < Tlen; ++t) {
        if (BEFORE) read_forward<T, I, RESIDENT>(state, ri, rw, y, rh, p, t, S, D, Tlen, scratch);
        int tok = p * Tlen + t, route = tok * 64 + r;
        int idx = address<T, RESIDENT>(p, int(wi[route]), d, S, D);
        float old = float(state[idx]), w = float(ww[route]);
        float decay = exp2f(float(decay_log[tok]) * 1.4426950408889634f);
        float decayed = decay * old;
        wh[route * D + d] = T(old);
        float retrieved = route_sum(w * decayed, scratch);
        float delta = float(beta[tok]) * (float(values[tok * D + d]) - retrieved);
        state[idx] = T(decayed + w * delta); // required per-token BF16 store
        __syncthreads();
        if (!BEFORE) read_forward<T, I, RESIDENT>(state, ri, rw, y, rh, p, t, S, D, Tlen, scratch);
    }
    if (RESIDENT) {
        __syncthreads();
        for (int i = threadIdx.x; i < S * 4; i += THREADS)
            final[(p * S + i / 4) * D + blockIdx.y * 4 + i % 4] = state[i];
    }
}

template<typename T, typename I, bool RESIDENT>
__device__ void read_backward(
    T* state, const I* ri, const T* rw, const T* rh, float dy, float* dq,
    int p, int t, int S, int D, int Tlen) {
    int r = threadIdx.x / 4, d = threadIdx.x % 4 + blockIdx.y * 4;
    int route = (p * Tlen + t) * 64 + r;
    float dq_part = dimension_sum(float(rh[route * D + d]) * dy);
    if (threadIdx.x % 4 == 0) atomicAdd(dq + route, dq_part);
    int idx = address<T, RESIDENT>(p, int(ri[route]), d, S, D);
    state[idx] = T(float(state[idx]) + float(rw[route]) * dy);
    // Store/cast MUST precede the write-update VJP, even for overlapping routes.
    __syncthreads();
}

template<typename T, typename I, bool RESIDENT, bool BEFORE>
__global__ void ordered_backward(
    const T* final_ct, T* state_out, const T* wh, const T* rh,
    const I* wi, const T* ww, const T* values, const T* beta,
    const T* decay_log, const I* ri, const T* rw, const T* dy,
    float* dw, float* dv, float* db, float* dg, float* dq,
    int Tlen, int S, int D) {
    extern __shared__ __align__(16) unsigned char storage[];
    T* state = RESIDENT ? reinterpret_cast<T*>(storage) : state_out;
    __shared__ float scratch[32];
    int p = blockIdx.x, r = threadIdx.x / 4, local_d = threadIdx.x % 4;
    int d = blockIdx.y * 4 + local_d;
    if (RESIDENT) {
        for (int i = threadIdx.x; i < S * 4; i += THREADS)
            state[i] = final_ct[(p * S + i / 4) * D + blockIdx.y * 4 + i % 4];
        __syncthreads();
    }
    for (int t = Tlen - 1; t >= 0; --t) {
        int tok = p * Tlen + t, route = tok * 64 + r;
        float dyval = float(dy[tok * D + d]);
        if (!BEFORE) read_backward<T, I, RESIDENT>(state, ri, rw, rh, dyval, dq, p, t, S, D, Tlen);
        int idx = address<T, RESIDENT>(p, int(wi[route]), d, S, D);
        float old = float(wh[route * D + d]), w = float(ww[route]);
        float decay = exp2f(float(decay_log[tok]) * 1.4426950408889634f);
        float grad_updated = float(state[idx]);
        // Native backward groups weight*decay before multiplying old.
        float retrieved = route_sum((w * decay) * old, scratch);
        float grad_delta = route_sum(w * grad_updated, scratch);
        float gate = float(beta[tok]), value = float(values[tok * D + d]);
        float delta = gate * (value - retrieved), grad_retrieved = -gate * grad_delta;
        float decayed = decay * old;
        float dw_part = dimension_sum(grad_updated * delta + grad_retrieved * decayed);
        if (local_d == 0) atomicAdd(dw + route, dw_part);
        float grad_decayed = grad_updated + w * grad_retrieved;
        state[idx] = T(decay * grad_decayed); // required reverse-token cast
        float dg_part = dimension_sum(route_sum(grad_decayed * decayed, scratch));
        if (threadIdx.x == 0) atomicAdd(dg + tok, dg_part);
        if (r == 0) dv[tok * D + d] = gate * grad_delta;
        float db_part = dimension_sum(grad_delta * (value - retrieved));
        if (threadIdx.x == 0) atomicAdd(db + tok, db_part);
        __syncthreads();
        if (BEFORE) read_backward<T, I, RESIDENT>(state, ri, rw, rh, dyval, dq, p, t, S, D, Tlen);
    }
    if (RESIDENT) {
        __syncthreads();
        for (int i = threadIdx.x; i < S * 4; i += THREADS)
            state_out[(p * S + i / 4) * D + blockIdx.y * 4 + i % 4] = state[i];
    }
}

// Audit is opt-in and outside timing. Values come from the same launch site.
static bool audit_enabled = false;
static int last_launch[8] = {};
extern "C" void set_audit(int enabled) { audit_enabled = enabled; }
extern "C" void get_launch(int* output) {
    for (int i = 0; i < 8; ++i) output[i] = last_launch[i];
}

template<typename T, typename I, bool R, bool B>
int dispatch(int backward, void** a, int P, int Tlen, int S, int D, cudaStream_t stream, int* resources) {
    auto fwd = ordered_forward<T, I, R, B>;
    auto bwd = ordered_backward<T, I, R, B>;
    int shared = R ? S * TILE_D * sizeof(T) : 0;
    const void* fn = backward ? (const void*)bwd : (const void*)fwd;
    cudaError_t error = cudaSuccess;
    if (shared >= 48 * 1024) {
        error = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
        if (error != cudaSuccess) return int(error);
    }
    if (resources) {
        cudaFuncAttributes attr;
        error = cudaFuncGetAttributes(&attr, fn);
        if (error != cudaSuccess) return int(error);
        int blocks;
        error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, THREADS, shared);
        if (error != cudaSuccess) return int(error);
        resources[0] = attr.numRegs;
        resources[1] = attr.localSizeBytes;
        resources[2] = attr.sharedSizeBytes;
        resources[3] = shared;
        resources[4] = blocks;
        resources[5] = attr.binaryVersion;
        return 0;
    }
    dim3 grid(P, D / TILE_D);
    if (audit_enabled) {
        int v[] = {backward, int(grid.x), int(grid.y), THREADS, shared, Tlen, S, D};
        for (int i = 0; i < 8; ++i) last_launch[i] = v[i];
    }
    if (!backward) fwd<<<grid, THREADS, shared, stream>>>(
        (T*)a[0], (T*)a[1], (I*)a[2], (T*)a[3], (T*)a[4], (T*)a[5],
        (T*)a[6], (I*)a[7], (T*)a[8], (T*)a[9], (T*)a[10], (T*)a[11], Tlen, S, D);
    else bwd<<<grid, THREADS, shared, stream>>>(
        (T*)a[0], (T*)a[1], (T*)a[2], (T*)a[3], (I*)a[4], (T*)a[5],
        (T*)a[6], (T*)a[7], (T*)a[8], (I*)a[9], (T*)a[10], (T*)a[11],
        (float*)a[12], (float*)a[13], (float*)a[14], (float*)a[15], (float*)a[16], Tlen, S, D);
    return int(cudaGetLastError());
}

template<typename T, typename I>
int select(int backward, int resident, int before, void** a, int P, int Tlen, int S, int D, cudaStream_t stream, int* resources) {
    if (resident && before) return dispatch<T, I, true, true>(backward,a,P,Tlen,S,D,stream,resources);
    if (resident) return dispatch<T, I, true, false>(backward,a,P,Tlen,S,D,stream,resources);
    if (before) return dispatch<T, I, false, true>(backward,a,P,Tlen,S,D,stream,resources);
    return dispatch<T, I, false, false>(backward,a,P,Tlen,S,D,stream,resources);
}

extern "C" int launch(int backward, int resident, int before, int bf16, int int64,
    void** a, int P, int Tlen, int S, int D, void* stream, int* resources) {
    if (bf16 && int64) return select<__nv_bfloat16, int64_t>(backward,resident,before,a,P,Tlen,S,D,(cudaStream_t)stream,resources);
    if (bf16) return select<__nv_bfloat16, int32_t>(backward,resident,before,a,P,Tlen,S,D,(cudaStream_t)stream,resources);
    if (int64) return select<float, int64_t>(backward,resident,before,a,P,Tlen,S,D,(cudaStream_t)stream,resources);
    return select<float, int32_t>(backward,resident,before,a,P,Tlen,S,D,(cudaStream_t)stream,resources);
}
