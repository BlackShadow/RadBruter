#pragma once
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#include <stdlib.h>

/* The runner gives every worker an independent input handle. */
static inline FILE *lm_fopen(const char *path, const char *mode) {
    if (!strcmp(path, "@radbruter-input") && !strcmp(mode, "rb")) {
        int descriptor = _dup(120);
        FILE *file;
        if (descriptor < 0) return NULL;
        file = _fdopen(descriptor, "rb");
        if (!file) { _close(descriptor); return NULL; }
        if (_fseeki64(file, 0, SEEK_SET)) { fclose(file); return NULL; }
        return file;
    }
    return fopen(path, mode);
}
#endif
