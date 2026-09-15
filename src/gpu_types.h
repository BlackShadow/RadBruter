#ifndef LM_GPU_TYPES_H
#define LM_GPU_TYPES_H
typedef struct LMTraceNode {
    int type;
    float normal[3], dist;
    int children[2], pad;
} LMTraceNode;
typedef struct LMTraceRay {
    float start[3], stop[3];
    int node;
    unsigned int bit;
} LMTraceRay;
// Geometry-only interpolation recipe. Triangle x/y divisions are retained
// separately to preserve QRAD's rounding and ordered additions.
typedef struct LMIndirectSample {
    unsigned int kind, patch[3]; // 0: empty, 1: copy, 2: triangle, 3: edge
    float x, y, x2, y1;
} LMIndirectSample;
typedef struct LMIndirectResult {
    float light[3];
    unsigned int uncertain;
} LMIndirectResult;
#endif
