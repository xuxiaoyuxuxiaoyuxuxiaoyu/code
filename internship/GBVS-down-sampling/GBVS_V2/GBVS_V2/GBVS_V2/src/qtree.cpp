#include "Algo.h"
extern double savg;
extern double smax;
Mat qnode::QTree(Mat& gmap,Mat& dmap)
{
	int chan = gmap.channels();
	if (chan == 1)
	{
		//cout << "输入图像为单通道图" << endl;
		Mat gray;
		//cvtColor(gmap, gray, CV_RGB2GRAY);
		qnode* g = new qnode();
		qnode* d = new qnode();
		g->std = 0; d->std = 0;
		g->pnode = gmap; d->pnode = dmap;
		for (int i = 0; i < 4; i++)
		{
			g->snode[i] = NULL;
			d->snode[i] = NULL;
		}
		CreatBranch(*g,*d);
		//下采样
		Sample(*g,*d);
		//还原
		//Mat a = CombBranch(*g);
		Mat b = CombBranch(*d);
		return b;
	}
	else
		return dmap;
}
double qnode::GetStd(Mat& inmap)
{
	if (inmap.channels() != 1)
	{
		cout << "输入图像通道不正确" << endl;
		return 0;
	}
	double std, mean;
	Mat tmp_std, tmp_m;
	meanStdDev(inmap, tmp_m, tmp_std);
	std = tmp_std.at<double>(0, 0);
	return std;
}
void qnode::CreatBranch(qnode& gmap, qnode& dmap)
{
	double std = GetStd(dmap.pnode);
	if ((std >= 2) && (dmap.pnode.rows>5))
	{
		Mat ul = dmap.pnode(Range(0, dmap.pnode.rows / 2), Range(0, dmap.pnode.cols / 2));
		Mat ur = dmap.pnode(Range(0, dmap.pnode.rows / 2), Range(dmap.pnode.rows / 2, dmap.pnode.rows));
		Mat ll = dmap.pnode(Range(dmap.pnode.rows / 2, dmap.pnode.rows), Range(0, dmap.pnode.cols / 2));
		Mat lr = dmap.pnode(Range(dmap.pnode.rows / 2, dmap.pnode.rows), Range(dmap.pnode.rows / 2, dmap.pnode.rows));
		//
		Mat gul = gmap.pnode(Range(0, gmap.pnode.rows / 2), Range(0, gmap.pnode.cols / 2));
		Mat gur = gmap.pnode(Range(0, gmap.pnode.rows / 2), Range(gmap.pnode.rows / 2, gmap.pnode.rows));
		Mat gll = gmap.pnode(Range(gmap.pnode.rows / 2, gmap.pnode.rows), Range(0, gmap.pnode.cols / 2));
		Mat glr = gmap.pnode(Range(gmap.pnode.rows / 2, gmap.pnode.rows), Range(gmap.pnode.rows / 2, gmap.pnode.rows));
		qnode* g = new qnode[4];
		g[0].pnode = gul;
		g[1].pnode = gur;
		g[2].pnode = gll;
		g[3].pnode = glr;
		//
		qnode* n = new qnode[4];
		n[0].pnode = ul;
		n[1].pnode = ur;
		n[2].pnode = ll;
		n[3].pnode = lr;
		for (int i = 0; i < 4; i++)
		{
			dmap.snode[i] = n + i;
			gmap.snode[i] = g + i;
		}
		for (int i = 0; i < 4; i++)
		{
			CreatBranch(g[i],n[i]);
		}

	}
	else
	{
		for (int i = 0; i < 4; i++)
		{
			dmap.snode[i] = NULL;
			gmap.snode[i] = NULL;
		}
		/*for (int i = 0; i < dmap.pnode.rows; i++)
		{
			if ((i == 0) || (i == dmap.pnode.rows - 1))
		{
		for (int j = 0; j < dmap.pnode.cols; j++)
		{
			dmap.pnode.ptr<uchar>(i)[j] = 255;
		}
		}
		else
		{
			dmap.pnode.ptr<uchar>(i)[0] = 255;
			dmap.pnode.ptr<uchar>(i)[dmap.pnode.cols - 1] = 255;
		}
		}*/
	}
}
Mat qnode::CombBranch(qnode& innode)
{
	Mat outmap(innode.pnode.rows, innode.pnode.cols, CV_8UC1,Scalar(0,0,0));
	if (innode.snode[0] != NULL)
	{
		Mat dstroiul = outmap(Rect(0, 0, innode.pnode.rows / 2, innode.pnode.cols / 2));
		Mat dstroiur = outmap(Rect(innode.pnode.cols / 2, 0, innode.pnode.rows / 2, innode.pnode.cols / 2));
		Mat dstroill = outmap(Rect(0, innode.pnode.rows / 2, innode.pnode.rows / 2, innode.pnode.cols / 2));
		Mat dstroilr = outmap(Rect(innode.pnode.cols / 2, innode.pnode.rows / 2, innode.pnode.rows / 2, innode.pnode.cols / 2));

		CombBranch(*innode.snode[0]).convertTo(dstroiul, dstroiul.type(), 1, 0);
		CombBranch(*innode.snode[1]).convertTo(dstroiur, dstroiur.type(), 1, 0);
		CombBranch(*innode.snode[2]).convertTo(dstroill, dstroill.type(), 1, 0);
		CombBranch(*innode.snode[3]).convertTo(dstroilr, dstroilr.type(), 1, 0);
	}
	else
	{
		//cout << "最后一个啦" << endl;
		innode.pnode.convertTo(outmap, CV_8UC1);
	}
	return outmap;
}
void qnode::Sample(qnode& innode1, qnode& innode2)
{
	
	if (innode2.snode[0] != NULL)
	{
		for (int i = 0; i < 4; i++)
		{
			Sample(*innode1.snode[i],*innode2.snode[i]);
		}
	}
	else
	{
		//执行采样
		if (innode2.pnode.rows > 4)
		{
			if (innode2.pnode.rows > 60)
			{
				Mat temp(innode2.pnode.rows/8, innode2.pnode.cols/8, CV_8UC1, Scalar(0, 0, 0));
				for (int i = 0; i < (innode2.pnode.rows / 8); i++)
				{
					for (int j = 0; j < (innode2.pnode.cols / 8); j++)
					{
						temp.ptr<uchar>(i)[j] = innode2.pnode.ptr<uchar>(8 * i)[8 * j];
					}
				}
				resize(temp, innode2.pnode, Size(innode2.pnode.cols, innode2.pnode.rows), 0, 0, INTER_CUBIC);
			}
			else
			{
				//8-32，采样率
				double siavg = mean(innode1.pnode)[0];
				int sr = (int)(4 + 3*((siavg - 2 * savg) / (smax - 2 * savg)));
				//cout << sr << endl;
				if (sr>1&&sr<4)
				{
					Mat temp(innode2.pnode.rows/4, innode2.pnode.cols/4, CV_8UC1, Scalar(0, 0, 0));
					for (int i = 0; i < (innode2.pnode.rows/4); i++)
					{
						for (int j = 0; j < (innode2.pnode.cols/4); j++)
						{
								temp.ptr<uchar>(i)[j] = innode2.pnode.ptr<uchar>(4*i)[4*j];
						}
					}
					resize(temp, innode2.pnode, Size(innode2.pnode.cols, innode2.pnode.rows), 0, 0, INTER_CUBIC);
				}

				else
				if (sr>3&&sr<=6)
				{
					Mat temp(innode2.pnode.rows/2, innode2.pnode.cols/2, CV_8UC1, Scalar(0, 0, 0));
					for (int i = 0; i < (innode2.pnode.rows / 2); i++)
					{
						for (int j = 0; j < (innode2.pnode.cols / 2); j++)
						{
							temp.ptr<uchar>(i)[j] = innode2.pnode.ptr<uchar>(2 * i)[2 * j];
						}
					}
					resize(temp, innode2.pnode, Size(innode2.pnode.cols, innode2.pnode.rows), 0, 0, INTER_CUBIC);
				}
				else
					if (sr>=7&&sr<=8)
					{

					}
				else
				{
					Mat temp(innode2.pnode.rows/8, innode2.pnode.cols/8, CV_8UC1, Scalar(0, 0, 0));
					for (int i = 0; i < (innode2.pnode.rows / 8); i++)
					{
						for (int j = 0; j < (innode2.pnode.cols / 8); j++)
						{
							temp.ptr<uchar>(i)[j] = innode2.pnode.ptr<uchar>(8 * i)[8 * j];
						}
					}
					resize(temp, innode2.pnode, Size(innode2.pnode.cols, innode2.pnode.rows), 0, 0, INTER_CUBIC);
				}
			}
			/*for (int i = 0; i < (innode.pnode.rows / 2); i++)
			{
				for (int j = 0; j < (innode.pnode.cols / 2); j++)
				{
					temp.ptr<uchar>(2 * i)[2 * j] = innode.pnode.ptr<uchar>(2 * i)[2 * j];
				}
			}
			temp.convertTo(innode.pnode, innode.pnode.type(), 1, 0);*/
		}

	}
}