#ifndef RGB2DKL_H
#define RGB2DKL_H
#include <opencv2\opencv.hpp>
using namespace cv;
#include "iostream"
using namespace std;
#include "math.h"

#define DEBUG
#include "assert.h"
void rgb2dkl(Mat&inmap, Mat* outmap);

#endif