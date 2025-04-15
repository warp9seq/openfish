#include <stdio.h>
#include <math.h>
#include <cstdint>
#include <float.h>

#include <vector>

// g++ -o kth test/kth.cpp

void swapf(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

int partitionf(float *nums, int left, int right) {
	int pivot = left;
    swapf(&nums[pivot], &nums[right]);

    for (int i = left; i < right; ++i) {
        if (nums[i] > nums[right]) {
            swapf(&nums[pivot], &nums[i]);
            ++pivot;
        }
    }
    
    swapf(&nums[pivot], &nums[right]);
    return pivot;
}

float kth_largestf(float *nums, int k, int n) {
	int left = 0;
    int right = n-1;
	while (left <= right) {
        int pivot = partitionf(nums, left, right);
        if (pivot < k) {
            left = pivot+1;
        } else if (pivot > k) {
            right = pivot-1;
        } else {
            return nums[pivot];
        }
	}
	return -FLT_MAX;
}

int main(int argc, char* argv[]) {
    // auto arr = std::vector<float>({0.0, 0.0, 0.0, 0.0, 0.0});
    auto arr = std::vector<float>({5.0, 4.0, 3.0, 2.0, 1.0});
    auto n = arr.size();
    auto k = 1;

    fprintf(stderr, "starting arr:\n");
    for (auto num: arr) {
        fprintf(stderr, "%f\n", num);
    }
    fprintf(stderr, "\n");
    auto res = kth_largestf(arr.data(), k, n);
    
    fprintf(stderr, "ans: %f\n", res);
    return 0;
}