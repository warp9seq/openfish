/* @file error.h
**
** error checking macros/functions and error messages

MIT License

Copyright (c) 2018  Hasindu Gamaarachchi (hasindu@unsw.edu.au)
Copyright (c) 2018  Thomas Daniell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


******************************************************************************/

#ifndef ERROR_H
#define ERROR_H

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include <openfish/openfish_error.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MALLOC_CHK(ret) { \
    if ((ret) == NULL) { \
        MALLOC_ERROR() \
        exit(EXIT_FAILURE); \
    } \
}

#define MALLOC_ERROR() OPENFISH_ERROR("Failed to allocate memory: %s", strerror(errno))

#define F_CHK(ret, file) { \
    if ((ret) == NULL) { \
        OPENFISH_ERROR("Could not to open file %s: %s", file, strerror(errno)) \
        exit(EXIT_FAILURE); \
    } \
}

#define NULL_CHK(ret) { \
    if ((ret) == NULL) { \
        OPENFISH_ERROR("NULL returned: %s.", strerror(errno)) \
        exit(EXIT_FAILURE); \
    } \
}

#define NEG_CHK(ret) { \
    if ((ret) < 0) { \
        OPENFISH_ERROR("Negative value returned: %s.", strerror(errno)) \
        exit(EXIT_FAILURE); \
    } \
}

#define ASSERT(ret) { \
    if ((ret) == 0){ \
        fprintf(stderr, OPENFISH_ERROR_PREFIX "Assertion failed." OPENFISH_NO_COLOUR \
                " At %s:%d\nExiting.\n", \
                __func__ , __FILE__, __LINE__ - 1); \
        exit(EXIT_FAILURE); \
    } \
}

#ifdef __cplusplus
}
#endif

#endif
