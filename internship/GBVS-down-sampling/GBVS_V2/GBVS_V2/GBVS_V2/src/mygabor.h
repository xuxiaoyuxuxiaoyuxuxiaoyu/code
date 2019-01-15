#ifndef MYGABOR_H
#define MYGABOR_H

#include <opencv2\opencv.hpp>
using namespace cv;
#include "iostream"
using namespace std;
#include "math.h"
#include "algo.h"

#define PI       3.14159265358979323846
void GetOFeatureMap(Mat *pyrmap, Mat *ochannel, int* angs, int length);

#endif