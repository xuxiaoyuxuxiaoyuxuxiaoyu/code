#include"Algo.h"
/***********
*参数初始化函数
************/
param::param()
{
	salmapmaxsize = 32;
	maxcomputelevel = 4;
	for (int i = 0; i < 3; i++)
	{
		levels[i] = i + 2;
	}
	for (int i = 0; i < 4; i++)
	{
		gaborangs[i] = i * 45;
	}
	inter_type = 2;
	intra_type = 2;
	cyclic_type = 2;
}
/***********
*图像初始化函数
************/
makesalmap::makesalmap(param p)
{
	const int length = p.maxcomputelevel;
	pyrmap = new Mat[length];
	ifeaturemap = new Mat[length];
	ifeaturemap_sal = new Mat[length];
	cfeaturemap = new Mat[3 * length];
	cfeaturemap_sal = new Mat[3 * length];
	ofeaturemap = new Mat[4 * length];
	ofeaturemap_sal = new Mat[4 * length];

	actfeaturemap_sal = new Mat[3 * length + length + 4*length];//CIO-2，3，4层
	normfeaturemap_sal = new Mat[3 * length + length + 4 * length];//CIO-2，3，4层
	smap = new Mat[3];
}
makesalmap::makesalmap()
{
	pyrmap = NULL;
	ifeaturemap = NULL;
	cfeaturemap = NULL;
	ofeaturemap = NULL;
}

/***********
*生成图像金字塔
*输入参数：原始图像inmap，参数类p
*输出参数：outmap数组
************/
void makesalmap::YieldPyrMap(Mat& inmap, param p, Mat* outmap)
{
	*outmap = inmap;
	for (int i = 0; i < (p.maxcomputelevel - 1); i++)
	{
		outmap++;
		pyrDown(*(outmap - 1), *(outmap), Size((*(outmap - 1)).cols / 2, (*(outmap - 1)).rows / 2));
	}
}
/***********
*获取强度通道特征图
*输入参数：原始图像inmap
*输出参数：outmap数组
************/
void makesalmap::GetImap(Mat* inmap, param p, Mat* outmap)
{
	for (int i = 0; i < p.maxcomputelevel; i++)
	{

		for (int j = 0; j < inmap[i].rows; j++)
		{
			Vec3b* inptr = inmap[i].ptr<Vec3b>(j);
			uchar* outptr = (*(outmap + i)).ptr<uchar>(j);
			for (int k = 0; k < inmap[i].cols; k++)
			{
				outptr[k] = (inptr[k][0] + inptr[k][1] + inptr[k][2]) / 3;
			}
		}
	}
}
/***********
*获取dkl颜色通道特征图
*输入参数：原始图像inmap
*输出参数：outmap数组
************/
void makesalmap::GetDklCmap(Mat* inmap, param p, Mat* outmap)
{
	for (int i = 0; i < p.maxcomputelevel; i++)
	{
		rgb2dkl(inmap[i], (outmap + i * 3));
	}
}

/***********
*获取方向通道特征图
*输入参数：强度通道特征图，角度
*输出参数：ofeaturemap
************/
void makesalmap::GetOmap(Mat* inmap, Mat* outmap, int* angs, param p)
{
	GetOFeatureMap(ifeaturemap, ofeaturemap, p.gaborangs, p.maxcomputelevel);
}
void makesalmap::GetFeatureMaps(Mat& inmap, param p)
{
	YieldPyrMap(inmap, p, pyrmap);
	cout << endl << "图像金字塔生成完成，存储于pyrmap中" << endl;

	for (int i = 0; i < p.maxcomputelevel; i++)
	{
		ifeaturemap[i].create(pyrmap[i].rows, pyrmap[i].cols, CV_8UC1);
		ifeaturemap_sal[i].create(p.salmapsize[0],p.salmapsize[1],CV_8UC1);
	}
	GetImap(pyrmap, p, ifeaturemap);
	for (int i = 0; i < p.maxcomputelevel; i++)
	{
		resize(ifeaturemap[i],ifeaturemap_sal[i],Size(p.salmapsize[1],p.salmapsize[0]),0,0,INTER_CUBIC);
		ifeaturemap_sal[i].convertTo(ifeaturemap_sal[i], CV_64F);
		normalize(ifeaturemap_sal[i], ifeaturemap_sal[i], 0.0, 1.0, NORM_MINMAX);
	}
	cout << endl << "强度特征图金字塔生成完成，存储于ifeaturemap中" << endl;

	for (int i = 0; i < p.maxcomputelevel; i++)
	{
		cfeaturemap[3 * i].create(pyrmap[i].rows, pyrmap[i].cols, CV_64FC1);
		cfeaturemap_sal[3 * i].create(p.salmapsize[0], p.salmapsize[1], CV_64FC1);
		cfeaturemap[3 * i + 1].create(pyrmap[i].rows, pyrmap[i].cols, CV_64FC1);
		cfeaturemap_sal[3 * i+1].create(p.salmapsize[0], p.salmapsize[1], CV_64FC1);
		cfeaturemap[3 * i + 2].create(pyrmap[i].rows, pyrmap[i].cols, CV_64FC1);
		cfeaturemap_sal[3 * i+2].create(p.salmapsize[0], p.salmapsize[1], CV_64FC1);
	}
	GetDklCmap(pyrmap, p, cfeaturemap);
	for (int i = 0; i < 3*p.maxcomputelevel; i++)
	{
		resize(cfeaturemap[i], cfeaturemap_sal[i], Size(p.salmapsize[1], p.salmapsize[0]), 0, 0, INTER_CUBIC);
		//imshow("1", cfeaturemap_sal[i]); waitKey(30);
	}
	cout << endl << "颜色特征图金字塔生成完成，存储于cfeaturemap中" << endl;
	for (int i = 0; i < 4 * p.maxcomputelevel; i++)
	{
		ofeaturemap_sal[i].create(p.salmapsize[0],p.salmapsize[1],CV_8UC1);
	}
	GetOmap(ifeaturemap, ofeaturemap, p.gaborangs, p);
	for (int i = 0; i < 4 * p.maxcomputelevel; i++)
	{
		resize(ofeaturemap[i], ofeaturemap_sal[i], Size(p.salmapsize[1], p.salmapsize[0]), 0, 0, INTER_CUBIC);
		ofeaturemap_sal[i].convertTo(ofeaturemap_sal[i],CV_64F);
		normalize(ofeaturemap_sal[i], ofeaturemap_sal[i], 0.0, 1.0, NORM_MINMAX);
	}
	cout << endl << "方向通道计算完成，存储于ofeaturemap中" << endl;
}
/************
*显著图初始化函数
*读取locationmap和distancemap
************/
double* makesalmap::GetMatData(char* filename,char* dname)
{
	MATFile* pmatfile = NULL;
	mxArray* preadarray = NULL;
	const char* file;
	file = filename;
	double* data;

	pmatfile = matOpen(file, "r");
	preadarray = matGetVariable(pmatfile, dname);
	data = (double*)mxGetData(preadarray);
	size_t M = mxGetM(preadarray);
	size_t N = mxGetN(preadarray);
	/*for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < N; j++)
		{
			cout << data[j*M + i] << "  ";
		}
		cout << endl;
	}*/
	matClose(pmatfile);
	return data;
}
void makesalmap::GraphSailInit(param* p)
{
	double* lx = new double; double* dis = new double;
	const int length = p->salmapsize[0] * p->salmapsize[1];
	lx = GetMatData("E:\\keil\\C语言\\VS2013\\GBVS_V2 - 副本 - 副本 - 副本\\GBVS_V2\\GBVS_V2\\grframe_location.mat","grframe_lx");
	dis = GetMatData("E:\\keil\\C语言\\VS2013\\GBVS_V2 - 副本 - 副本 - 副本\\GBVS_V2\\GBVS_V2\\grframe_distance.mat", "grframe_d");
	/////////////////////////////////////
	p->frame_dis = new double*[length];
	p->frame_lx = new double*[length];
	for (int i = 0; i < length; i++)
	{
		p->frame_lx[i] = new double[3];
		p->frame_dis[i] = new double[length];
	}
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < length; j++)
		{
			p->frame_dis[i][j] = dis[j*length+i];
		}
	}
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			p->frame_lx[i][j] = lx[j*length + i];
		}
	}
}
Mat makesalmap::YieldSalMap(Mat& inmap, param p)
{
	int mapsize[2] = { inmap.rows, inmap.cols };
	int w = mapsize[1]; int h = mapsize[0];
	double scale = (double)(p.salmapmaxsize) / max(w, h);
	p.salmapsize[0] = (int)(h*scale);
	p.salmapsize[1] = (int)(w*scale);

	GetFeatureMaps(inmap, p);
	GraphSailInit(&p);
	//activiation
	double tol = 0.003;
	for (int i = 0; i < 3*p.maxcomputelevel; i++)
	{
		actfeaturemap_sal[i] = GraphSalApply(cfeaturemap_sal[i], &p, 0.15, 1, 2, tol);
	}
	for (int i = 0; i < p.maxcomputelevel; i++)
	{
		actfeaturemap_sal[3 * p.maxcomputelevel + i] = GraphSalApply(ifeaturemap_sal[i], &p, 0.15, 1, 2, tol);
	}
	for (int i = 0; i < 4 * p.maxcomputelevel; i++)
	{
		actfeaturemap_sal[4 * p.maxcomputelevel + i] = GraphSalApply(ofeaturemap_sal[i], &p, 0.15, 1, 2, tol);
	}
	cout << endl << "activiation结束" << endl;
	//normilization	
	for (int i = 0; i < 8 * p.maxcomputelevel; i++)
	{
		normfeaturemap_sal[i] = GraphSalApply(actfeaturemap_sal[i], &p, 0.06, 1, 2, tol);
	}
	delete[]actfeaturemap_sal;
	cout << endl << "normilization结束" << endl;
	//average
	for (int i = 0; i < 3; i++)
	{
		//smap[i].create(p.salmapsize[0],p.salmapsize[1],CV_64FC1);
		smap[i] = Mat::zeros(p.salmapsize[0], p.salmapsize[1], CV_64FC1);
	}
	for (int i = 0; i < 3 * (p.maxcomputelevel - 1); i++)
	{
		addWeighted(smap[0],1.0,normfeaturemap_sal[i+3],1.0,0.0,smap[0]);
	}
	for (int i = 0; i < p.maxcomputelevel - 1; i++)
	{
		addWeighted(smap[1], 1.0, normfeaturemap_sal[i + 13], 1.0, 0.0, smap[1]);
	}
	for (int i = 0; i < 4 * (p.maxcomputelevel - 1); i++)
	{
		addWeighted(smap[2], 1.0, normfeaturemap_sal[i + 20], 1.0, 0.0, smap[2]);
	}
	//sum across
	Mat mastermap(p.salmapsize[0],p.salmapsize[1],CV_64FC1,Scalar(0));
	for (int i = 0; i < 3; i++)
	{
		addWeighted(mastermap, 1.0, smap[i], 1.0, 0.0, mastermap);
	}
	GaussianBlur(mastermap, mastermap, Size(3, 3), 0.64, 0.64);
	normalize(mastermap, mastermap, 0, 255, NORM_MINMAX);
	mastermap.convertTo(mastermap,CV_8U);
	resize(mastermap, mastermap, Size(inmap.cols, inmap.rows), 0, 0, INTER_CUBIC);
	return mastermap;
}
Mat makesalmap::YieldDOGSalMap(Mat& inmap)
{
	Mat outmap(inmap.rows,inmap.cols,CV_8UC1);
	Mat outmap2(inmap.rows, inmap.cols, CV_8UC1);;
	for (int j = 0; j < inmap.rows; j++)
	{
		Vec3b* inptr = inmap.ptr<Vec3b>(j);
		uchar* outptr = outmap.ptr<uchar>(j);
		for (int k = 0; k < inmap.cols; k++)
		{
			outptr[k] = (inptr[k][0] + inptr[k][1] + inptr[k][2]) / 3;
		}
	}
	GaussianBlur(outmap, outmap2, Size(9, 9), 1, 1);
	GaussianBlur(outmap2, outmap2, Size(9, 9), 1, 1);
	Mat map = outmap - outmap2;
	//GaussianBlur(map, map, Size(25, 25), 2, 2);
	normalize(map,map,0,255,NORM_MINMAX);
	return map;
}