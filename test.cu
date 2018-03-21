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


__global__
void saxpy(float n, float a, float *x, float *w)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i += stride)
		w[i] = w[i]*x[i] + a;
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
int classes = 1;



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
	float *x, *d_x;
	float **d_w;
	float **w;

	int N = width * height;

	cout << "Starting code......."  << endl;

	x = (float *)malloc( N *sizeof(float));
	w = (float **)malloc( classes *sizeof(float*));
	for(int i = 0; i < classes; i++) {
		w[i] = (float *)malloc( N *sizeof(float));
	}

	cudaMalloc(&d_x, N *sizeof(float));

	for(int i = 0; i < classes; i++) {
		cudaMalloc(&d_w[i], N *sizeof(float));
	}

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

	// Neural Network Initialization
	//init_array();

	for (int sample = 1; sample <= nTraining; ++sample) {
		cout << "Sample ---------- **************" << sample << endl;

		// Getting (image, label)
		input();
	}


	report.close();
	image.close();
	label.close();


	for (int i = 0; i < width * height; i++) {
		x[i] = (float)d[i % width][i / width];
		for(int j = 0; j < 10; j++)
			w[j][i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}

	cout << "Image:" << endl;
	for (int j = 0; j < height; ++j) {
		for (int i = 0; i < width; ++i) {
			cout << x[ (j ) * height + (i )];
		}
		cout << endl;
	}
	cout << "Label:" << (int)inputNum << endl;

	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

	for(int i = 0; i < classes; i++)
		cudaMemcpy(d_w[i], w[i], N * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_w, w, N * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_w, w, N * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_w, w, N * sizeof(float), cudaMemcpyHostToDevice);
//	cudaMemcpy(d_w, w, N * sizeof(float), cudaMemcpyHostToDevice);

	// Perform SAXPY on 1M elements
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;
	saxpy<<<numBlocks, blockSize>>>(N, 2.0f, d_x, d_w[0]);

	cudaMemcpy(w[0], d_w[0], N*sizeof(float), cudaMemcpyDeviceToHost);

	for (int j = 0; j < height; ++j) {
		for (int i = 0; i < width; ++i) {
			cout << (float)w[i][(j) * height + (i)] << " ";
		}
		cout << endl;
	}
	cout << "Label:" << (int)inputNum << endl;

	cudaFree(d_x);
	for(int i = 0; i < classes; i++)
		cudaFree(d_w[i]);
	free(x);
	for(int i = 0; i < classes; i++)
		free(w[i]);
}
