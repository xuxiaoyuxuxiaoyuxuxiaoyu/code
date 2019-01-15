#include <opencv2\opencv.hpp>
using namespace cv;
#include "iostream"
using namespace std;
#include <highgui.h>
#include "Algo.h"
#pragma comment(lib,"libmat.lib")
#pragma comment(lib,"libmx.lib")

/*
副本修改记录：
1："rgb2dkl.cpp"中im数组由length行3列改为3行length列。
2："Algo.h"--vector<vector<double> >frame_lx;
			 vector<vector<double> > frame_dis;
			 改为double**frame_lx;double**frame_lx;
			 对应的初始化也要改变："Algo.cpp->GraphSailInit(param* p)"
			 对应的应用函数也要改变："graphsalapply.cpp->GraphSalApply(Mat& A, param* p, double sigma_frac, int num_iters, int algtype, double tol)"
3："graphsalapply.cpp"中vector<vector<double> >lx;
						vector<vector<double> >dw;
						vector<double>al;
						vector<vector<double> >MM;
						改为：
						double** lx;
						double** dw;
						double* al;
						double** MM;
*/
const int alphaslider_max = 100;
int alphaslider;
double alpha;
double beta;

Mat gmap;
Mat dmap;
Mat dst;
Mat picture2;

double savg;
double smax;
double smin;
void on_trackbar(int, void*)
{
	alpha = (double)alphaslider / alphaslider_max;
	beta = (1.0 - alpha);

	addWeighted(gmap, alpha, dmap, beta, 0.0, dst);
	savg = mean(dst)[0];
	minMaxIdx(dst,&smin,&smax);
	qnode *q = new qnode();
	Mat b = q->QTree(dst,picture2);
	imshow("salmap", dst); waitKey(30);
	imshow("samplemap", b); waitKey(30);
	//imwrite("final_sawtooth.png", b); waitKey(30);
	Mat x = picture2 - b;
	for (int i = 0; i < 512; i++)
	{
		for (int j = 0; j < 512; j++)
		{
			if (x.at<uchar>(i, j) != 0)
			{
				cout << i << "  "<< j<<endl;
			}
		}
	}
}
int main()
{
	//Mat picture = imread("view1.png");
	picture2 = imread("disp3.png"); cvtColor(picture2, picture2, CV_RGB2GRAY);

	gmap = imread("gsalmap_sawtooth.png"); cvtColor(gmap, gmap, CV_RGB2GRAY);
	dmap = imread("dsalmap_sawtooth.png"); cvtColor(dmap, dmap, CV_RGB2GRAY);
	param p;
	makesalmap m(p);
	//gmap = m.YieldSalMap(picture, p); 
	//dmap = m.YieldDOGSalMap(picture2); 
	//imwrite("gsalmap.png", gmap); waitKey(30);
	//imwrite("dsalmap.png", dmap); waitKey(30);
	//创建滑动条窗口
	alphaslider = 50;
	namedWindow("salmap",1);
	namedWindow("samplemap",1);
	createTrackbar("trackbar", "salmap", &alphaslider, alphaslider_max, on_trackbar);
	on_trackbar(alphaslider, 0);

	//
	//qnode *q = new qnode();
	//q->QTree(picture2);
	waitKey(0);
}
