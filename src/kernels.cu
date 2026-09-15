// Preserve QRAD's transfer order and separately rounded float operations.
// Each thread computes one receiving patch/channel; no unordered reductions.
extern "C" __global__ void gather_light(const unsigned int *offsets, const unsigned int *transfers,
                                        const float *emission, float *result, unsigned int patches) {
    unsigned int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= patches * 3)
        return;
    unsigned int patch = channel / 3;
    unsigned int c = channel % 3;
    float sum = 0;
    for (unsigned int i = offsets[patch]; i < offsets[patch + 1]; ++i) {
        unsigned int transfer = transfers[i];
        unsigned int source = transfer & 65535u;
        float weight = float(transfer >> 16);
        sum = __fadd_rn(sum, __fmul_rn(emission[source * 3 + c], weight));
    }
    result[channel] = sum;
}

// Return only certified float-rounding decisions. Boundary cases go back to
// QRAD's x87 reference tracer, rather than accepting a different shadow edge.
__device__ float stable_float(double value, double error, bool &certain) {
    float lo = __double2float_rn(__dadd_rn(value, -error));
    float hi = __double2float_rn(__dadd_rn(value, error));
    if (lo != hi)
        certain = false;
    return __double2float_rn(value);
}

__device__ int trace_visibility(const LMTraceNode *nodes, unsigned node_count, const float *start,
                                const float *end, int root = 0) {
    struct Segment {
        int node;
        float a[3], b[3];
    } stack[64];
    int pending = 1;
    stack[0].node = root;
    for (int c = 0; c < 3; ++c) {
        stack[0].a[c] = start[c];
        stack[0].b[c] = end[c];
    }
    unsigned visits = 0;
    while (pending) {
        Segment s = stack[--pending];
        for (;;) {
            if (++visits > node_count * 4 + 64)
                return 2;
            if (s.node == -2 || s.node == -6)
                return 0; // SOLID or SKY
            if (s.node < 0)
                break;
            if ((unsigned)s.node >= node_count)
                return 2;
            const LMTraceNode &node = nodes[s.node];
            float front, back;
            bool certain = true;
            if (node.type >= 0 && node.type < 3) {
                front = __double2float_rn(__dadd_rn((double)s.a[node.type], -double(node.dist)));
                back = __double2float_rn(__dadd_rn((double)s.b[node.type], -double(node.dist)));
            } else {
#ifdef LM_WINDOWS_ARITHMETIC
                // Match the individual float operations in MSVC's x64 C engine.
                front = __fsub_rn(__fadd_rn(__fadd_rn(__fmul_rn(s.a[0], node.normal[0]),
                    __fmul_rn(s.a[1], node.normal[1])), __fmul_rn(s.a[2], node.normal[2])), node.dist);
                back = __fsub_rn(__fadd_rn(__fadd_rn(__fmul_rn(s.b[0], node.normal[0]),
                    __fmul_rn(s.b[1], node.normal[1])), __fmul_rn(s.b[2], node.normal[2])), node.dist);
#else
                double f = -double(node.dist), b = f, magnitude = fabs(f);
                for (int c = 0; c < 3; ++c) {
                    f = __dadd_rn(f, __dmul_rn((double)s.a[c], node.normal[c]));
                    b = __dadd_rn(b, __dmul_rn((double)s.b[c], node.normal[c]));
                    magnitude +=
                        fabs((double)s.a[c] * node.normal[c]) + fabs((double)s.b[c] * node.normal[c]);
                }
                front = stable_float(f, magnitude * 2e-15 + 1e-30, certain);
                back = stable_float(b, magnitude * 2e-15 + 1e-30, certain);
#endif
            }
            if (!certain)
                return 2;
            if (front >= -0.01 && back >= -0.01) {
                s.node = node.children[0];
                continue;
            }
            if (front < 0.01 && back < 0.01) {
                s.node = node.children[1];
                continue;
            }
            int side = front < 0;
#ifdef LM_WINDOWS_ARITHMETIC
            float frac = __fdiv_rn(front, __fsub_rn(front, back));
#else
            double fraction = __ddiv_rn((double)front, __dadd_rn((double)front, -double(back)));
            float frac = stable_float(fraction, fabs(fraction) * 2e-15, certain);
#endif
            float mid[3];
            for (int c = 0; c < 3; ++c) {
#ifdef LM_WINDOWS_ARITHMETIC
                mid[c] = __fadd_rn(s.a[c], __fmul_rn(__fsub_rn(s.b[c], s.a[c]), frac));
#else
                if (s.a[c] == s.b[c])
                    mid[c] = s.a[c];
                else {
                    double product = __dmul_rn(__dadd_rn((double)s.b[c], -double(s.a[c])), frac);
                    double value = __dadd_rn((double)s.a[c], product);
                    mid[c] = stable_float(value, (fabs(product) + fabs((double)s.a[c])) * 2e-15, certain);
                }
#endif
            }
            if (!certain || pending >= 64)
                return 2;
            Segment &far = stack[pending++];
            far.node = node.children[!side];
            for (int c = 0; c < 3; ++c) {
                far.a[c] = mid[c];
                far.b[c] = s.b[c];
                s.b[c] = mid[c];
            }
            s.node = node.children[side];
        }
    }
    return 1;
}

extern "C" __global__ void trace_rays(const LMTraceNode *nodes, unsigned node_count, const LMTraceRay *rays,
                                      unsigned count, unsigned char *result) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        result[i] = trace_visibility(nodes, node_count, rays[i].start, rays[i].stop, rays[i].node);
}

// One thread per final lightmap sample. Geometry decisions come from QRAD's
// freshly built triangulation; rounding-ambiguous arithmetic returns to the CPU.
extern "C" __global__ void interpolate_light(const LMIndirectSample *plans, unsigned count,
                                            const float *light, LMIndirectResult *result) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const LMIndirectSample &p = plans[i];
    LMIndirectResult out{};
    bool certain = true;
    if (p.kind)
        for (int c = 0; c < 3; ++c) {
            float base = light[p.patch[0] * 3 + c], value = base;
            if (p.kind == 3) {
#ifdef LM_WINDOWS_ARITHMETIC
                value = __fadd_rn(base, __fmul_rn(p.x, __fsub_rn(light[p.patch[1] * 3 + c], base)));
#else
                double delta = __dadd_rn((double)light[p.patch[1] * 3 + c], -double(base));
                double term = __dmul_rn(p.x, delta);
                value = stable_float(__dadd_rn(base, term), (fabs((double)base) + fabs(term)) * 4e-15,
                                     certain);
#endif
            } else if (p.kind == 2) {
                float d1 = __fsub_rn(light[p.patch[1] * 3 + c], base);
                float d2 = __fsub_rn(light[p.patch[2] * 3 + c], base);
                if (fabs((double)p.x2) >= 0.01) {
#ifdef LM_WINDOWS_ARITHMETIC
                    value = __fadd_rn(value, __fdiv_rn(__fmul_rn(p.x, d2), p.x2));
#else
                    double term = __ddiv_rn(__dmul_rn(p.x, d2), p.x2);
                    value = stable_float(__dadd_rn(value, term),
                                         (fabs((double)value) + fabs(term)) * 4e-15, certain);
#endif
                }
                if (fabs((double)p.y1) >= 0.01) {
#ifdef LM_WINDOWS_ARITHMETIC
                    value = __fadd_rn(value, __fdiv_rn(__fmul_rn(p.y, d1), p.y1));
#else
                    double term = __ddiv_rn(__dmul_rn(p.y, d1), p.y1);
                    value = stable_float(__dadd_rn(value, term),
                                         (fabs((double)value) + fabs(term)) * 4e-15, certain);
#endif
                }
            }
            if (!isfinite(value))
                certain = false;
            out.light[c] = value;
        }
    out.uncertain = !certain;
    result[i] = out;
}
