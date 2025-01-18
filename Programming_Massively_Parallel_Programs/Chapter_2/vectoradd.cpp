#include <iostream>
#include <math.h>
#include <chrono>

//Running on my Laptop (12th Gen Intel(R) Core(TM) i7-12700H   2.30 GHz) Function Execution Time: 0.00204605 seconds 

//Compute vector sum C_h = A_h + B_h
void vecAdd(float* A_h, float* B_h, float* C_h, int n) {

	for (int i = 0; i < n; i++) {
		C_h[i] = A_h[i] + B_h[i];
		//printf("Current value of C[%d]: %f", i, C_h[i]);
	}
}
int main(void) {

	int N = 1 << 20;

	float* A = new float[N];
	float* B = new float[N];
	float* C = new float[N];

	for (int i = 0; i < N; i++) {
		A[i] = 1.0f;
		B[i] = 2.0f;
		C[i] = 3.0f;
	}

	//Clock Start
	auto start = std::chrono::high_resolution_clock::now();
	
	//Function Call
	vecAdd(A, B, C, N);
	
	//Clock End
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Function execution time: " << elapsed.count() << " seconds\n";


	delete[] A; 
	delete[] B;
	delete[] C;

	return 0;

}