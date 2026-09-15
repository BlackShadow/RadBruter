#pragma once
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <direct.h>
#include <stdint.h>
#include <stddef.h>
#include <fcntl.h>
#include <stdlib.h>
#include <math.h>
#include "platform_io.h"
#define fopen lm_fopen
#else
#include <unistd.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <fcntl.h>
#include <stdlib.h>
#include <math.h>
#define _int64 int64_t
#define MAX_PATH 4096
#define _MAX_PATH 4096
#define HANDLE void *
#define GMEM_FIXED 0
#define GMEM_ZEROINIT 0
#define GlobalAlloc(flags,n) calloc(1,n)
#define GlobalLock(h) (h)
#define GlobalHandle(h) (h)
#define GlobalUnlock(h) ((void)0)
#define GlobalFree(h) free(h)
#define _stat stat
#define _open open
#define _read read
#define _write write
#define _close close
#define _lseek lseek
#define _access access
#define _O_RDONLY O_RDONLY
#define _O_WRONLY O_WRONLY
#define _O_RDWR O_RDWR
#define _O_BINARY 0
#define _O_CREAT O_CREAT
#define _O_TRUNC O_TRUNC
#define _S_IREAD S_IRUSR
#define _S_IWRITE S_IWUSR
#ifndef min
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#endif

#define FALSE 0
static inline uint32_t _rotl(uint32_t v, unsigned s) { return (v << s) | (v >> (32-s)); }
#endif
#ifndef __cplusplus
#ifndef min
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#endif
#endif
