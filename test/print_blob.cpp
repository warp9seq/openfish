#include <stdio.h>
#include <math.h>
#include <cstdint>
#include <float.h>

// compile me for tests:
// g++ -o print_blob test/print_blob.cpp

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "%s\n", "usage: print_blob <tens_0.blob> <tens_1.blob>");
    }
    FILE *fp;
    size_t result;

    fprintf(stderr, "comapring files %s and %s\n", argv[1], argv[2]);

    size_t tens_len = strtol(argv[3], NULL, 10);
    float *tens0 = (float *)calloc(tens_len, sizeof(float));
    float *tens1 = (float *)calloc(tens_len, sizeof(float));

    fp = fopen(argv[1], "rb");
    result = fread(tens0, sizeof(float), tens_len, fp);
    if (result != tens_len) {
        fprintf(stderr, "%s\n", "error reading file 0");
        exit(1);
    }
    fclose(fp);

    fp = fopen(argv[2], "rb");
    result = fread(tens1, sizeof(float), tens_len, fp);
    if (result != tens_len) {
        fprintf(stderr, "%s\n", "error reading file 1");
        exit(1);
    }
    fclose(fp);

    int n = 2048;
    for (int i = 0; i < n; ++i) {
        float tens0_i = *(tens0 + i);
        float tens1_i = *(tens1 + i);

        fprintf(stdout, "%.1f\n", tens0_i);
    }
    return 0;
}