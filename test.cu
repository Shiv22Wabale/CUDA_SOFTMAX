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
int batch_size;
int iteration;

int classes = 10;

__global__
void saxpy(float n, float a, float *x, float *w, float *sum, size_t pitch, int batch_size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;
	int stride = blockDim.x * gridDim.x;
	//printf("%d\n", index);
	int classes = 10;
	for (int i = index; i < n; i += stride)
		for(int k = 0; k < classes; k++) {
			if ((i < n) && (tidy < batch_size)) {
				float *row_x = (float *)((char*)x + tidy * pitch);
				float *row_sum = (float *)((char*)sum + tidy * pitch);
				row_sum[i + k * (int)n] = w[i + k * (int)n]*row_x[i] + a;
				//printf("%f\n", row_sum[i + k * (int)n]);
			}
			//sum[i + k * (int)n] = w[i + k * (int)n]*x[i] + a;
			//sum[i + k * (int)n] = intermediateW[i + k * (int)n];

		}
}

__global__
void sum_cuda(float n, float *sum, float *total, int run, int batch, size_t pitch)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;
	//printf("Index --- %d", index);
	//int classes = 1;
	for (int idx = index; idx < n; idx += stride) {
		//for(int k = 0; k < classes; k++) {
		float *row_sum = (float *)((char*)sum + tidy * pitch);
		register int i = atomicAdd(&total[0], row_sum[idx + run * (int)n]);
		row_sum[i + run * (int)n] = idx;
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
void updateWeights(float n, float *err, float *w, float *x, int batch_size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int classes = 10;
	//float a;
	for (int i = index; i < n; i += stride)
		for(int k = 0; k < classes; k++) {
			//printf(" %f  ", w[i + k * (int)n] );
			//a = w[i + k * (int)n];
			w[i + k * (int)n] -= 0.001 * ( ( ( -1 * err[k] * x[i] ) / batch_size ) + w[i + k * (int)n] );
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
		//cout << "Label:" << (int)inputNum << endl;
	}
}

void check(float *sum, int N){
	float total = 0.0f;
	for(int j = 0; j < N; j++)
		total += sum[j];

	cout<<total<< endl;
}

void printUsage() {
	// show memory usage of GPU

	        size_t free_byte ;

	        size_t total_byte ;

//	        cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
	        cudaMemGetInfo( &free_byte, &total_byte ) ;

//	        if ( cudaSuccess != cuda_status ){
//
//	            printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
//
//	            exit(1);
//
//	        }



	        double free_db = (double)free_byte ;

	        double total_db = (double)total_byte ;

	        double used_db = total_db - free_db ;

	        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",

	            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}

int main(int argc, char *argv[])
{



//	batch_size = 20000;
//	iteration = 3;

	//String b = argv[0], iter = argv[1];
	//batch_size = atoi(b);
	//iteration = atoi(iter);
	batch_size = atoi(argv[1]);
	iteration = atoi(argv[2]) / batch_size;
	cout << "----  The arguments received are " << argv[1] << " " << batch_size << " with iteratiosn " << argv[2] << " " <<iteration << endl;
	//float **x, *d_x, *d_w, *w, *sum, *d_sum;
	float **x, *d_x, *d_w, *w, **sum,*d_sum;
	//float total = 0, *d_total = 0;
	float *d_index = 0;
	float *h_index = 0;
	float err[10], *d_err;

	int N = ( width ) * ( height );

	cout << "Starting code....... " << N << "the float size is : " << sizeof(float) << endl;

	x = (float **)malloc( batch_size * sizeof(float *));
	for(int i = 0; i < batch_size; i++)
		x[i] = (float *)malloc( N * sizeof(float));
	sum = (float **)malloc( batch_size * sizeof(float *));
		for(int i = 0; i < batch_size; i++)
			sum[i] = (float *)malloc( N * classes * sizeof(float));
	//x = (float *)malloc( N * sizeof(float));
	w = (float *)malloc( N * classes * sizeof(float));
	//sum = (float *)malloc( N * classes * sizeof(float));
	h_index = (float *)malloc( classes * sizeof(float));
	//total = (float *)malloc( classes * sizeof(float) );

	for(int i = 0; i < classes; i++)
		h_index[i] = 0;


	/*********** initializing wights *******************/
	for (int i = 0; i < N; i++) {
		for(int j = 0; j < classes; j++)
			w[i + j * N] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}

	/*************Transfer Data from host to device *********************/
	size_t pitch_x, pitch_sum;

	cudaMallocPitch(&d_x, &pitch_x, N * sizeof(float), batch_size);
	cudaMalloc(&d_w, N * classes * sizeof(float));
	cudaMallocPitch(&d_sum, &pitch_sum, N * classes * sizeof(float), batch_size);
	//cudaMalloc(&d_sum, N * classes * sizeof(float));
	//cudaMalloc(&d_total, classes * sizeof(float));

	cudaMalloc(&d_err, classes * sizeof(float));
	cudaMalloc( (void**) &d_index, classes * sizeof(float) );
	cudaMemcpy(d_w, w, N * classes * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_sum, sum, N * classes * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_total, total, sizeof(float), cudaMemcpyHostToDevice);
	/*************Transfer Data from host to device *********************/

//	printUsage();
//	getchar();

	//getchar();
	/*********** i	nitializing wights *******************/

	/***************** *****************/
//	//cudaMemcpy(w, d_w, N * classes * sizeof(float), cudaMemcpyDeviceToHost);
//	for(int k = 9; k < 10; k++) {
//		for (int j = 0; j < height; ++j) {
//			for (int i = 0; i < width; ++i) {
//				cout <<  " " << w[ (j ) * height + (i ) + k * N];
//			}
//			cout << endl;
//		}
//	}
//	cout << endl;
//	cout << endl;
//	cout << endl;

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


	double diff = 0;
	/********************** ITERATION STARTS **************************/
	for(int l = 0; l < iteration; l++) {
//		cout << "Iteration no. : " << l << endl;

		/***************** Image Loading **********************/

		int hostNum[batch_size][10];
//		cout << "HostNum oneHot vector  : " << endl;

		for(int batch = 0; batch < batch_size; batch++) {
			for (int sample = 1; sample <= nTraining; ++sample) {
				//cout << "Sample ---------- **************" << sample << endl;

				input();
			}


			for (int i = 0; i < N; i++) {
				x[batch][i] = (float)d[i % width][i / width];
			}

			//			cout << "Image:" << endl;
			//			for (int j = 0; j < height; ++j) {
			//				for (int i = 0; i < width; ++i) {
			//					cout << x[ (j ) * height + (i )];
			//				}
			//				cout << endl;
			//			}
			for(int j = 0; j < classes; j++)
				hostNum[batch][j] = 0;
			hostNum[batch][(int)inputNum] = 1;
		}
//		printUsage();
//		getchar();

		//getchar();
		//	cout << "Label: ";
		//	for(int j = 0; j < classes; j++)
		//		cout << " " << hostNum[j];
		//	cout << endl;

		/***************** Image Loading **********************/


		/********* Multiplying ******************/
		clock_t t, t1, t2;
		t = clock();
		t1 = clock();

//		cout << "Before memory transfer----  : pitch x  is : " << pitch_x << " pitch sum is : " << pitch_sum << endl;
//		getchar();
		//cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy2D(d_x, pitch_x, x, N*sizeof(float), N*sizeof(float), batch_size, cudaMemcpyHostToDevice);

//		printUsage();
//		getchar();
		// Perform SAXPY on 1M elements
		//int blockSize = 27;
		int blockSize = N * batch_size;
		int numBlocks = ( ( N * batch_size ) + blockSize - 1) / blockSize;


		//saxpy<<<numBlocks, blockSize>>>(N, 0.0f, d_x, d_w, d_sum, pitch_x);
		saxpy<<<batch_size, N>>>(N, 0.0f, d_x, d_w, d_sum, pitch_x, batch_size);

//		printUsage();
//		getchar();

		/********* Multiplying ******************/

		/***************** *****************/
//		cout << " The sum is : " << endl;
//				cudaMemcpy2D(sum, sizeof(float) * N * classes, d_sum, pitch_sum, N * classes * sizeof(float), batch_size, cudaMemcpyDeviceToHost);
//				for(int k = 2; k < 3; k++) {
//					for (int j = 0; j < height; ++j) {
//						for (int i = 0; i < width; ++i) {
//							cout << " " << sum[0][ (j ) * height + (i ) + k * N];
//						}
//						cout << endl;
//					}
//				}
//				cout << endl;
//				cout << endl;
//				cout << endl;
		/***************** *****************/


		/*********** Finding Softmax ************************/
		//cudaMemcpy(sum, d_sum, N*classes*sizeof(float), cudaMemcpyDeviceToHost);

		blockSize = 27 * 27;
		numBlocks = (classes + blockSize - 1) / blockSize;

		for(int k = 0; k < classes; ++k) {
			err[k] = 0;
		}

		int max_index = 0;
		float total[batch_size][10], summation = 0;
		for( int batch = 0; batch < batch_size; batch++) {

			for(int k = 0; k < classes; ++k) {
				h_index[0] = 0;
				cudaMemcpy(d_index, h_index , classes * sizeof(float), cudaMemcpyHostToDevice);

				sum_cuda<<<numBlocks, blockSize>>>(N, d_sum, d_index, k, batch, pitch_sum);
				cudaMemcpy(h_index , d_index, classes * sizeof(int), cudaMemcpyDeviceToHost);

				total[batch][k] = h_index[0];
				max_index = total[batch][k] > total[batch][max_index] ? k : max_index;
			}
			for(int k = 0; k < classes; ++k) {
				total[batch][k] = total[batch][k] / total[batch][max_index];
				summation += total[batch][k];
			}
			for(int k = 0; k < classes; ++k) {
				total[batch][k] = total[batch][k] / summation;
				max_index = total[batch][k] > total[batch][max_index] ? k : max_index;
			}

			/*********** Finding Softmax ************************/

			/***************** Checking the softmax **********/
			float temp = 0;
			for(int k = 0; k < classes; ++k) {
				temp += total[batch][k];
			}
			//cout << temp << " ---- " << max_index << endl;
			/***************** Checking the softmax **********/


			/*********** Finding Error ************************/
			//	cout << " Error : ";

			for(int k = 0; k < classes; k++) {
				err[k] += hostNum[batch][k] - total[batch][k];
	//								cout << " e: " << err[k];
			}
			//cout << endl;
		}


		cudaMemcpy(d_err, err, classes * sizeof(float), cudaMemcpyHostToDevice);
		/*********** Finding Error ************************/



		/************* Updating the weights *******************/
		blockSize = 27 * 27;
		numBlocks = (classes + blockSize - 1) / blockSize;
		updateWeights<<<numBlocks, blockSize>>>(N, d_err, d_w, d_x, batch_size); // updateWeights(float n, float *err, float *w, float *x)


		/************* Updating the weights *******************/

		// Wait for GPU to finish before accessing on host
		cudaDeviceSynchronize();

		t = clock() - t;
		t2 = clock();
		double time_taken = ((double)t)/CLOCKS_PER_SEC; // in seconds
//		cout << " saxpy took " << time_taken << "seconds to execute on CUDA " << CLOCKS_PER_SEC << " the t1 = " << t1 << " and t2 = " << t2 << endl;
		//cout << endl;


		diff += time_taken;
	}
	/********************** ITERATION ENDS **************************/

//	cout << "Total avera" <<  diff << "  / " << iteration << " for N : " << iteration << endl;
	//cout << "Total average time it took : " << ( diff / ( iteration * batch_size ) ) << " for iteration : " << iteration << endl;
	cout << "run " << batch_size << " " << ( diff / ( iteration * batch_size ) ) << endl;

	report.close();
	image.close();
	label.close();

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
