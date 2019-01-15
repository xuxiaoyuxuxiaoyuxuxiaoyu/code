#ifndef ALGO_H
#define ALGO_H
#include <opencv2\opencv.hpp>
using namespace cv;
#include "iostream"
#include "fstream"
#include "iomanip"
using namespace std;
#include "rgb2dkl.h"
#include "mygabor.h"
#include "mat.h"
#include "matrix.h"
#include "vector"
#include <opencv2/imgproc/types_c.h>

class param
{
public:
	int salmapmaxsize;
	int maxcomputelevel;
	int levels[3];
	int gaborangs[4];
	int multilevels[2];
	int salmapsize[2];
	int inter_type;
	int intra_type;
	int cyclic_type;
	//vector<vector<double> >frame_lx;
	double** frame_lx;
	//vector<vector<double> > frame_dis;
	double** frame_dis;
public:
	param();
};

class makesalmap
{
public:
	Mat* pyrmap;	 //原始图像金字塔
	Mat* ifeaturemap;//强度特征图
	Mat* ifeaturemap_sal;
	Mat* cfeaturemap;//颜色特征图
	Mat* cfeaturemap_sal;
	Mat* ofeaturemap;//方向特征图
	Mat* ofeaturemap_sal;

	Mat* actfeaturemap_sal;
	Mat* normfeaturemap_sal;
	Mat* smap;

public:
	makesalmap();
	makesalmap(param p);
	Mat YieldSalMap(Mat& inmap, param p);
	Mat YieldDOGSalMap(Mat& inmap);
	void GetFeatureMaps(Mat& inmap, param p);
	void YieldPyrMap(Mat& inmap, param p, Mat* outmap);
	void GraphSailInit(param* p);
	Mat GraphSalApply(Mat& A, param* p, double sigma_frac, int num_iters, int algtype, double tol);
private:
	void GetImap(Mat* inmap, param p, Mat* outmap);
	void GetDklCmap(Mat* inmap, param p, Mat* outmap);
	void GetOmap(Mat *inmap, Mat *outmap, int* angs, param p);
	double* GetMatData(char* filename,char* dname);
};

//void YieldSalMap(Mat& inmap, param p);

class qnode
{
public:
	double std;
	Mat pnode;
	qnode* snode[4];
public:
	Mat QTree(Mat& gmap, Mat& dmap);
private:
	void CreatBranch(qnode& gmap,qnode& dmap);
	Mat CombBranch(qnode& innode);
	void Sample(qnode& innode1, qnode& innode2);
	double GetStd(Mat& inmap);
};

#endif