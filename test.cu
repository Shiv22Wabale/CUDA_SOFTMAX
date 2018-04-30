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
#include <time.h>

using namespace std;

// Training image file name
const string training_image_fn = "train-images.idx3-ubyte";

// Training label file name
const string training_label_fn = "train-labels.idx1-ubyte";

int classes = 10;
int iteration = 5000;

__global__
void saxpy(float n, float a, float *x, float *w, float *sum)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	//printf("%d", index);
	int classes = 10;
	for (int i = index; i < n; i += stride)
		for(int k = 0; k < classes; k++) {
			sum[i + k * (int)n] = w[i + k * (int)n]*x[i] + a;
			//sum[i + k * (int)n] = intermediateW[i + k * (int)n];
		}
}

__global__
void sum_cuda(float n, float *sum, float *total, int run)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	//printf("Index --- %d", index);
	//int classes = 1;
	for (int idx = index; idx < n; idx += stride) {
		//for(int k = 0; k < classes; k++) {
		register int i = atomicAdd(&total[0], sum[idx + run * (int)n]);
		sum[i + run * (int)n] = idx;
		//}
	}
	//for (int idx = index; idx < classes; idx += stride) {
	//printf("i = %d %f\n", i, sum[i]);
	//		for(int k = 0; k < n; k++) {
	//			//printf("i = %d %f\n",i, sum[i]);
	//			//sum[i] += w[i + k * (int)n];
	//			sum[i] += w[i*(int)n + k];
	//			//printf("%f\n",sum[i]);
	//		}
	//	register int i = atomicAdd(total, sum[idx]);
	//	sum[i] = idx;
	//printf("cuda --- %f\n",sum[i]);
	//}
}

__global__
void updateWeights(float n, float *err, float *w, float *x)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int classes = 10;
	//float a;
	for (int i = index; i < n; i += stride)
		for(int k = 0; k < classes; k++) {
			//printf(" %f  ", w[i + k * (int)n] );
			//a = w[i + k * (int)n];
			w[i + k * (int)n] -= 0.001 * ( ( -1 * err[k] * x[i] ) + w[i + k * (int)n] );
			//printf(" %f  after %f changes required %f\n", a, w[i + k * (int)n], err[k] );
		}
	//printf(" after changes required %f\n", err[0] );
	//theta[m2][n] += (alpha * (labelTrain[j][m2] - prob[m2]) * dataTrain[j][n]);
}

void softMax(float *sum)
{
	float total = 0.0f;
	for (int i = 0; i < classes; i += 1)
		total += exp(sum[i]);
	for (int i = 0; i < classes; i += 1)
		sum[i] = exp(sum[i]) / total;
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
	for(int i = 0; i < 1; i++ ) {

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
//		cout << "Label:" << (int)inputNum << endl;
	}
}

void check(float *sum, int N){
	float total = 0.0f;
	for(int j = 0; j < N; j++)
		total += sum[j];

//	cout<<total<< endl;
}

int main( int argc, char *argv[] )
{
	float *x, *d_x, *d_w, *w, *sum, *d_sum;
	//float total = 0, *d_total = 0;
	float *d_index = 0;
	float *h_index = 0;
	float err[10], *d_err;

	int N = width * height;

//	cout << "Starting code....... 124"  << endl;

	x = (float *)malloc( N * sizeof(float));
	w = (float *)malloc( N * classes * sizeof(float));
	sum = (float *)malloc( N * classes * sizeof(float));
	h_index = (float *)malloc( classes * sizeof(float));
	//total = (float *)malloc( classes * sizeof(float) );

	for(int i = 0; i < classes; i++)
		h_index[i] = 0;

	for(int conf = 1; conf <= 200; conf++) {
		//iteration = conf * 100;

		/*********** initializing wights *******************/
		for (int i = 0; i < N; i++) {
			for(int j = 0; j < classes; j++)
				w[i + j * N] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		}

		/*************Transfer Data from host to device *********************/
		cudaMalloc(&d_x, N * sizeof(float));
		cudaMalloc(&d_w, N * classes * sizeof(float));
		cudaMalloc(&d_sum, N * classes * sizeof(float));
		//cudaMalloc(&d_total, classes * sizeof(float));

		cudaMalloc(&d_err, classes * sizeof(float));
		cudaMalloc( (void**) &d_index, classes * sizeof(float) );
		cudaMemcpy(d_w, w, N * classes * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_sum, sum, N * classes * sizeof(float), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_total, total, sizeof(float), cudaMemcpyHostToDevice);
		/*************Transfer Data from host to device *********************/

		/*********** initializing wights *******************/

		/***************** *****************/
		//cudaMemcpy(w, d_w, N * classes * sizeof(float), cudaMemcpyDeviceToHost);
//		for(int k = 9; k < 10; k++) {
//			for (int j = 0; j < height; ++j) {
//				for (int i = 0; i < width; ++i) {
//					cout <<  " " << w[ (j ) * height + (i ) + k * N];
//				}
//				cout << endl;
//			}
//		}
//		cout << endl;
//		cout << endl;
//		cout << endl;

		/***************** *****************/

		/************************** *************************************************
		 *******************************
		 *******************************
		 *******************************  LOAD AND UPDATE
		 *******************************
		 *******************************
		 *  ********************* **************************************************/
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

//		int blockSize = conf * 4;
//		int numBlocks = (N + blockSize - 1) / blockSize;
//		int numBlocks = conf * 4;

//		cout << "blockSize"	<< blockSize <<	"numBlocks" << numBlocks << endl;
//		cout << " calculations for reduction  : " << conf << endl;

		double difference = 0, diff_cpu = 0;
		for(int l = 0; l < iteration; l++) {
//			blockSize = conf * 4;
//			numBlocks = (N + blockSize - 1) / blockSize;

			/***************** Image Loading **********************/



			for (int sample = 1; sample <= nTraining; ++sample) {
	//			cout << "Sample ---------- **************" << sample << endl;

				input();
			}





			for (int i = 0; i < N; i++) {
				x[i] = (float)d[i % width][i / width];
			}

			//			cout << "Image:" << endl;
			//			for (int j = 0; j < height; ++j) {
			//				for (int i = 0; i < width; ++i) {
			//					cout << x[ (j ) * height + (i )];
			//				}
			//				cout << endl;
			//			}
			int hostNum[10];
			for(int j = 0; j < classes; j++)
				hostNum[j] = 0;
			hostNum[(int)inputNum] = 1;

			//	cout << "Label: ";
			//	for(int j = 0; j < classes; j++)
			//		cout << " " << hostNum[j];
			//	cout << endl;

			/***************** Image Loading **********************/


			/********* Multiplying ******************/
			clock_t t, t1, t2;
//			t = clock();
//			t1 = clock();
			cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

			// Perform SAXPY on 1M elements
			//int blockSize = 27;
			int blockSize = N;
			int numBlocks = (N + blockSize - 1) / blockSize;

	//		saxpy<<<1, 1>>>(N, 0.0f, d_x, d_w, d_sum);
			saxpy<<<numBlocks, blockSize>>>(N, 0.0f, d_x, d_w, d_sum);
			/********* Multiplying ******************/

			/***************** *****************/
			//		cudaMemcpy(w, d_sum, N * classes * sizeof(float), cudaMemcpyDeviceToHost);
			//		for(int k = 2; k < 3; k++) {
			//			for (int j = 0; j < height; ++j) {
			//				for (int i = 0; i < width; ++i) {
			//					cout << " " << w[ (j ) * height + (i ) + k * N];
			//				}
			//				cout << endl;
			//			}
			//		}
			//		cout << endl;
			//		cout << endl;
			//		cout << endl;
			/***************** *****************/


			/*********** Finding Softmax ************************/
			//cudaMemcpy(sum, d_sum, N*classes*sizeof(float), cudaMemcpyDeviceToHost);

			blockSize = 27 * 27;
			numBlocks = (classes + blockSize - 1) / blockSize;
//			blockSize = conf;
//			numBlocks = (classes + blockSize - 1) / blockSize;
//			numBlocks = 1;

			int max_index = 0;
			float total[10], summation = 0;

			double time_taken = 0;// = ((double)t)/CLOCKS_PER_SEC; // in seconds
			for(int k = 0; k < classes; ++k) {
				h_index[0] = 0;
				cudaMemcpy(d_index, h_index , classes * sizeof(float), cudaMemcpyHostToDevice);

				t = clock();
				sum_cuda<<<numBlocks, blockSize>>>(N, d_sum, d_index, k);
				t = clock() - t;
				time_taken += ((double)t)/CLOCKS_PER_SEC;

				cudaMemcpy(h_index , d_index, classes * sizeof(int), cudaMemcpyDeviceToHost);

				total[k] = h_index[0];
	//						cout << total[k] << endl;
				//			check( sum + k * N, N);
				//summation += total[k];
				max_index = total[k] > total[max_index] ? k : max_index;
			}
			time_taken /= classes;

			t1 = clock();
			cudaMemcpy(sum, d_sum, N*classes*sizeof(float), cudaMemcpyDeviceToHost);
			check(sum,N);
			t2 = clock();
			double time_taken_by_cpu = ((double) (t2 - t1) )/CLOCKS_PER_SEC; // in seconds

			for(int k = 0; k < classes; ++k) {
				//total[k] = total[k] / summation;
				//				cout << total[k] << endl;
				total[k] = total[k] / total[max_index];
				summation += total[k];
				//max_index = total[k] > total[max_index] ? k : max_index;
			}
			for(int k = 0; k < classes; ++k) {
				total[k] = total[k] / summation;
	//			cout << total[k] << endl;
				max_index = total[k] > total[max_index] ? k : max_index;
				//			total[k] = total[k] - total[max_index];
				//			summation += total[k];
				//max_index = total[k] > total[max_index] ? k : max_index;
			}

			/*********** Finding Softmax ************************/

			/***************** Checking the softmax **********/
			float temp = 0;
			for(int k = 0; k < classes; ++k) {
				temp += total[k];
			}
	//		cout << temp << " ---- " << max_index << endl;
			/***************** Checking the softmax **********/


			/*********** Finding Error ************************/
			//	cout << " Error : ";

			for(int k = 0; k < classes; k++) {
				err[k] = hostNum[k] - total[k];
	//								cout << " e: " << err[k];
			}
	//			cout << endl;

			cudaMemcpy(d_err, err, classes * sizeof(float), cudaMemcpyHostToDevice);
			/*********** Finding Error ************************/



			/************* Updating the weights *******************/
	//		blockSize = 27 * 27;
	//		numBlocks = (classes + blockSize - 1) / blockSize;

			updateWeights<<<numBlocks, blockSize>>>(N, d_err, d_w, d_x); // updateWeights(float n, float *err, float *w, float *x)


			/************* Updating the weights *******************/

//			t = clock() - t;
//			t2 = clock();
//			double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
	//		cout << " saxpy took " << time_taken << "seconds to execute on CUDA " << CLOCKS_PER_SEC << " the t1 = " << t1 << " and t2 = " << t2 << endl;
			//cout << endl;

			difference += time_taken;
			diff_cpu += time_taken_by_cpu;

		}

		cout << iteration << " total time  : " << (difference / iteration ) << "  " << ( diff_cpu / iteration ) << endl;
		//cout << iteration << " total time  : " << (difference / iteration ) << endl;



		report.close();
		image.close();
		label.close();

	}

	/************************** *************************************************
	 *******************************
	 *******************************
	 *******************************  LOAD AND UPDATE
	 *******************************
	 *******************************
	 *  ********************* **************************************************/

	/***************** *****************/
//	cudaMemcpy(w, d_w, N * classes * sizeof(float), cudaMemcpyDeviceToHost);
//	for(int k = 0; k < 10; k++) {
//		for (int j = 0; j < height; ++j) {
//			for (int i = 0; i < width; ++i) {
//				cout << " " << (int) ( w[ (j ) * height + (i ) + k * N] * 200);
//			}
//			cout << endl;
//		}
//		cout << endl;
//		cout << endl;
//	}
	/***************** *****************/

	//	/***************** Checking the softmax **********/
	//	float temp = 0;
	//	for(int k = 0; k < classes; ++k) {
	//		temp += total[k];
	//	}
	//	cout << temp << "  " << max_index << endl;
	/***************** Checking the softmax **********/

}
