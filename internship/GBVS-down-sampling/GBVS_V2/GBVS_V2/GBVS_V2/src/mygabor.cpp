#include "mygabor.h"
static const uint omega = 2;
static const uint major_stddev = 2;						//分母的参数之一
static const uint minor_stddev = 4;						//分母的参数之一
static const uint max_stddev = 4;						//用来确定滤波器长度

/*****************
*Gabor滤波器生成：
*长=宽=27；
*****************/
Mat YeildGabor(int ang, int phase)
{
	const int h_length = (int)(max_stddev*sqrt(10) + 1);
	const int length = 2 * h_length + 1;
	double psi = PI / 180 * (double)phase;		//初相
	double rtDeg = PI / 180 * (double)ang;		//滤波器角度
	Mat filter_kernel(length, length, CV_64FC1);
	//生成滤波器矩阵
	double **filtermatrix = new double*[length];
	for (int i = 0; i < length; i++)
	{
		filtermatrix[i] = new double[length];
	}

	double **major = new double*[length];//存放x*cos+y*sin
	double **minor = new double*[length];//存放-x*sin+y*cos
	for (int i = 0; i < length; i++)
	{
		major[i] = new double[length];
		minor[i] = new double[length];
	}

	double *temp1 = new double[length];
	double *temp2 = new double[length];

	for (int i = 0; i < length; i++)
	{
		temp1[i] = (i - h_length)*cos(rtDeg);//
		temp2[i] = (-(i - h_length)*sin(rtDeg));//
	}
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < length; j++)
		{
			major[i][j] = temp1[i] + temp2[j];
			minor[i][j] = temp2[i] - temp1[j];
		}
	}
	delete[]temp1;
	delete[]temp2;

	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < length; j++)
		{
			filtermatrix[i][j] = cos(omega*major[i][j] + psi)*exp(-pow(major[i][j], 2) / (2 * pow(major_stddev, 2)) - (pow(minor[i][j], 2) / (2 * pow(minor_stddev, 2))));
		}
	}

	for (int i = 0; i < length; i++)
	{
		delete[]major[i];
		delete[]minor[i];
	}
	delete[]major;
	delete[]minor;
	//归一化
	double mean = 0, cov = 0;
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < length; j++)
		{
			mean += filtermatrix[i][j];
		}
	}
	mean /= (length*length);
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < length; j++)
		{
			cov += pow((filtermatrix[i][j] - mean), 2);
		}
	}
	cov = sqrt(cov);
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < length; j++)
		{
			filtermatrix[i][j] /= cov;
		}
	}
	//存放到一个Mat类以便下面运算
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < length; j++)
		{
			filter_kernel.ptr<double>(i)[j] = filtermatrix[i][j];
		}
	}
	//释放数组
	for (int i = 0; i < length; i++)
	{
		delete[]filtermatrix[i];
	}
	delete[]filtermatrix;

	return filter_kernel;
}

/*****************
*获取方向通道
*角度值[0,45,90,180]
*输入参数：原始单通道图像，角度，输出图像
*输出参数：void
*********************/
void GetOFeatureMap(Mat *pyrmap, Mat *ochannel, int* angs, int length)
{
	Mat kernel_1;
	Mat kernel_2;
	Mat temp1, temp2, temp_pyrmap;
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			kernel_1 = YeildGabor(angs[j], 0);
			kernel_2 = YeildGabor(angs[j], 90);
			temp1.create(pyrmap[i].rows, pyrmap[i].cols, CV_64FC1);
			temp2.create(pyrmap[i].rows, pyrmap[i].cols, CV_64FC1);
			pyrmap[i].convertTo(temp_pyrmap, CV_64F);
			//			cout << temp_pyrmap.channels();
			filter2D(temp_pyrmap, temp1, -1, kernel_1, Point(-1, -1), 0, BORDER_DEFAULT);
			filter2D(temp_pyrmap, temp2, -1, kernel_2, Point(-1, -1), 0, BORDER_DEFAULT);
			addWeighted(temp1, 1, temp2, 1, 0.0, ochannel[i * 4 + j]);
			ochannel[i * 4 + j].convertTo(ochannel[i * 4 + j], CV_8U);
			for (int k = 0; k < ochannel[i * 4 + j].rows; k++)
			{
				for (int l = 0; l < ochannel[i * 4 + j].cols; l++)
				{
					ochannel[i * 4 + j].ptr<uchar>(k)[l] = abs(ochannel[i * 4 + j].ptr<uchar>(k)[l]);
				}
			}
			normalize(ochannel[i * 4 + j], ochannel[i * 4 + j], 1, 255, NORM_MINMAX);
			//imshow("omap", ochannel[i * 4 + j]); waitKey(30);
		}

	}
	temp1.release(); temp2.release(); temp_pyrmap.release(); kernel_1.release(); kernel_2.release();
}