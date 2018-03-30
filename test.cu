#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>

using namespace std;

// Training image file name
const string training_image_fn = "train-images.idx3-ubyte";

// Training label file name
const string training_label_fn = "train-labels.idx1-ubyte";

int classes = 10;

__global__
void saxpy(float n, float a, float *x, float *w)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	//printf("%d", index);
	int classes = 10;
	for (int i = index; i < n; i += stride)
		for(int k = 0; k < classes; k++)
			w[i + k * (int)n] = w[i + k * (int)n]*x[i] + a;
}

__global__
void sum_cuda(float n, float *w, float *sum)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	//printf("Index --- %d", index);
	int classes = 10;
	for (int i = index; i < classes; i += stride) {
		//printf("i = %d %f\n", i, sum[i]);
		for(int k = 0; k < n; k++) {
			//printf("i = %d %f\n",i, sum[i]);
			//sum[i] += w[i + k * (int)n];
			sum[i] += w[i*(int)n + k];
			//printf("%f\n",sum[i]);
		}
		//printf("cuda --- %f\n",sum[i]);
	}
}

__global__
void softMax(float n, float a, float *x, float *w)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		w[i] = w[i]*x[i] + a;
}

// Software: Training Artificial Neural Network for MNIST database
// Author: Hy Truong Son
// Major: BSc. Computer Science
// Class: 2013 - 2016
// Institution: Eotvos Lorand University
// Email: sonpascal93@gmail.com
// Website: http://people.inf.elte.hu/hytruongson/
// Copyright 2015 (c). All rights reserved.

// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;
ofstream report;

// Number of training samples
const int nTraining = 1;

// Image size in MNIST database
const int width = 28;
const int height = 28;

// Image. In MNIST: 28x28 gray scale images.
int d[width][height];

char inputNum;




void input() {
	// Reading image
	for(int i = 0; i < 10; i++ ) {

		for (int j = 0; j < height; ++j) {
			for (int i = 0; i < width; ++i) {
				image.read(&inputNum, sizeof(char));
				if (inputNum == 0) {
					d[i][j] = 0;
				} else {
					d[i][j] = 1;
				}
			}
		}
		label.read(&inputNum, sizeof(char));
		cout << "Label:" << (int)inputNum << endl;
	}
}

int main(void)
{
	float *x, *d_x, *d_w, *w, *sum, *d_sum;

	int N = width * height;

	cout << "Starting code....... 124"  << endl;

	x = (float *)malloc( N * sizeof(float));
	w = (float *)malloc( N * classes * sizeof(float));
	sum = (float *)malloc( classes * sizeof(float));

	cudaMalloc(&d_x, N * sizeof(float));
	cudaMalloc(&d_w, N * classes * sizeof(float));
	cudaMalloc(&d_sum, classes * sizeof(float));

	image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
	label.open(training_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

	// Reading file headers
	char number;
	for (int i = 1; i <= 16; ++i) {
		image.read(&number, sizeof(char));
	}
	for (int i = 1; i <= 8; ++i) {
		label.read(&number, sizeof(char));
	}

	for (int sample = 1; sample <= nTraining; ++sample) {
		cout << "Sample ---------- **************" << sample << endl;

		input();
	}


	report.close();
	image.close();
	label.close();


	for (int i = 0; i < N; i++) {
		x[i] = (float)d[i % width][i / width];
		for(int j = 0; j < classes; j++)
			w[i + j * N] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}

	for(int j = 0; j < classes; j++)
		sum[j] = 0;

	cout << "Image:" << endl;
	for (int j = 0; j < height; ++j) {
		for (int i = 0; i < width; ++i) {
			cout << x[ (j ) * height + (i )];
		}
		cout << endl;
	}
	cout << "Label:" << (int)inputNum << endl;

	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w, w, N * classes * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sum, sum, classes * sizeof(float), cudaMemcpyHostToDevice);

	// Perform SAXPY on 1M elements
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;

	saxpy<<<numBlocks, blockSize>>>(N, 0.0f, d_x, d_w);

	blockSize = 256;
	numBlocks = (classes + blockSize - 1) / blockSize;

	sum_cuda<<<numBlocks, blockSize>>>(N, d_w, d_sum);
	//sum_cuda<<<1, 10>>>(N, d_w, d_sum);

	cudaMemcpy(w, d_w, N*classes*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(sum, d_sum, classes*sizeof(float), cudaMemcpyDeviceToHost);

	for(int j = 0; j < classes; j++)
		cout << sum[j] << endl;

}
