#include "Algo.h"
int counter=0;
Mat makesalmap::GraphSalApply(Mat& A, param* p, double sigma_frac, int num_iters, int algtype, double tol)
{
	counter++;
	if (counter >= 33)
		counter = 1;
	cout << endl << "operating NO."<<counter<<"graph  rows="<<A.rows<<"  cols="<<A.cols << endl;
	int iters;
//	A.convertTo(A,CV_64F);
	Mat Anorm(A.rows,A.cols,CV_64FC1);
	if (algtype == 4)
	{
		pow(A,1.5,Anorm);
		iters = 1;
		return A;
	}
	double** lx;
	lx = p->frame_lx;
	double sig = sigma_frac*((A.rows+A.cols)/2);

	const int length = p->salmapsize[0] * p->salmapsize[1];
	int len = length;
	if (len != A.rows*A.cols)
	{
		cout << "图像和locationmap长宽不符合" << endl;
		return A;
	}

	double** dw;
	dw = new double*[length];
	double** MM = new double*[len];
	for (int i = 0; i < length; i++)
	{
		dw[i] = new double[length];
		MM[i] = new double[len];
	}
	double* al = new double[A.rows*A.cols];
	//将A排成一列元素，顺序为：第一列-第二列-第三列...

	for (int i = 0; i < A.cols; i++)
	{
		for (int j = 0; j < A.rows; j++)
		{
			al[i*A.rows + j] = A.ptr<double>(j)[i];//A.at<double>(j,i);
		}
	}


	iters = 0;
	double s = 0;
	for (int i = 0; i < num_iters; i++)
	{
		//mexassignweights
		for (int _col = 0; _col < len; _col++)
		{
			s = 0;
			for (int _row = 0; _row < len; _row++)
			{
				dw[_row][_col] = exp(-1 * (p->frame_dis[_row][_col] / (2 * pow(sig, 2))));
				if (algtype == 1)
					MM[_row][_col] = dw[_row][_col] * al[_row];
				else
					if (algtype == 2)
						MM[_row][_col] = dw[_row][_col] * abs(al[_row]-al[_col]);
					else
						if (algtype == 3)
							MM[_row][_col] = dw[_row][_col] * abs(log(al[_row]/al[_col]));
						else
							if (algtype == 4)
								MM[_row][_col] = dw[_row][_col] * 1/(abs(al[_row] - al[_col])+1e-12);
				s += MM[_row][_col];
			}
			for (int _row = 0; _row < len; _row++)
			{
				MM[_row][_col] /= s;
			}
		}

		//////////////////////////
		//principalEigenvectorRaw
		const int vlen = (int)len;
		double df = 1;
		double* v = new double[vlen];
		double* ov = new double[vlen];
		double* oov = new double[vlen];
		for (int i = 0; i < vlen; i++)
		{
			v[i] = 1 / (double)vlen;
			ov[i] = v[i];
			oov[i] = v[i];
		}
		int iter = 0;
		double s2;
		while (df > tol)
		{			
			for (int i = 0; i < vlen; i++)
			{
				ov[i] = v[i];
				oov[i] = ov[i];
			}
			if (vlen != len)
			{
				cout << "相乘矩阵行列不等";
				return A;
			}
			else
			{
				df = 0;
				s2 = 0;
				for (int i = 0; i < len; i++)
				{
					s = 0;
					for (int j = 0; j < len; j++)
					{
						s += MM[i][j] * ov[j];
					}
					v[i] = s;
					df += pow((v[i] - ov[i]), 2);
					s2 += v[i];
				}
				df = sqrt(df);
				iter++;
				if (s2 >= 0)continue;
				else
				{
					for (int i = 0; i < len; i++)
					{
						v[i] = oov[i];	
					}
					break;
				}
			}
		}

		for (int i = 0; i < len; i++)
		{
			v[i] /= s2;
			al[i] = v[i];
		}
		delete[]v; delete[]ov; delete[]oov;
		////////////////////////////
		iters += iter;
	}
	for (int i = 0; i < Anorm.cols;i++)
	{ 
		for (int j = 0; j < Anorm.rows; j++)
		{
			//Anorm.at<double>(j,i) = al[i*Anorm.rows+j];
			Anorm.ptr<double>(j)[i] = al[i*Anorm.rows + j];
		}
	}
	for (int i = 0; i < length; i++)
	{
		 delete[]dw[i]; delete[]MM[i];
	}
	 delete[]dw; delete[]MM; delete[]al;
	return Anorm;
}