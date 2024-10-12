#include <stdio.h>
#include <math.h>
#include <cstdint>

// g++ -o compare_blob compare_blob.cpp 

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "%s\n", "usage: compare_blob <tens_0.blob> <tens_1.blob>");
    }
    FILE *fp;
    size_t result;

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

    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    uint64_t n_diff = 0;
    for (int i = 0; i < tens_len; ++i) {
        float diff = fabs(tens0[i] - tens1[i]);
        if (diff != 0.0f) {
            if (diff > max_diff) {
                max_diff = diff;
            }
            avg_diff += diff;
            n_diff += 1;
        }
    }
    avg_diff /= tens_len;
    fprintf(stderr, "tensor max elem diff by %.32f, avg diff: %f, tens_len: %zu, n_diffs: %zu\n", max_diff, avg_diff, tens_len, n_diff);

    return 0;
}