#include "rgb2dkl.h"
/*****************
*rgb��ɫ�ռ�ת����dkl��ɫ�ռ�
*�������Ϊ3ͨ��rgbͼ��ע��Mat������Ϊbgr
******************/
static double lut_rgb[][3] = {
	{ 0.024935, 0.0076954, 0.042291 },
	{ 0.024974, 0.0077395, 0.042346 },
	{ 0.025013, 0.0077836, 0.042401 },
	{ 0.025052, 0.0078277, 0.042456 },
	{ 0.025091, 0.0078717, 0.042511 },
	{ 0.02513, 0.0079158, 0.042566 },
	{ 0.025234, 0.007992, 0.042621 },
	{ 0.025338, 0.0080681, 0.042676 },
	{ 0.025442, 0.0081443, 0.042731 },
	{ 0.025545, 0.0082204, 0.042786 },
	{ 0.025649, 0.0082966, 0.042841 },
	{ 0.025747, 0.0084168, 0.042952 },
	{ 0.025844, 0.0085371, 0.043062 },
	{ 0.025942, 0.0086573, 0.043172 },
	{ 0.026039, 0.0087776, 0.043282 },
	{ 0.026136, 0.0088978, 0.043392 },
	{ 0.026234, 0.0090581, 0.043502 },
	{ 0.026331, 0.0092184, 0.043612 },
	{ 0.026429, 0.0093788, 0.043722 },
	{ 0.026526, 0.0095391, 0.043833 },
	{ 0.026623, 0.0096994, 0.043943 },
	{ 0.026818, 0.0099198, 0.044141 },
	{ 0.027013, 0.01014, 0.044339 },
	{ 0.027208, 0.010361, 0.044537 },
	{ 0.027403, 0.010581, 0.044736 },
	{ 0.027597, 0.010802, 0.044934 },
	{ 0.027857, 0.010994, 0.04522 },
	{ 0.028117, 0.011186, 0.045507 },
	{ 0.028377, 0.011379, 0.045793 },
	{ 0.028636, 0.011571, 0.046079 },
	{ 0.028896, 0.011764, 0.046366 },
	{ 0.029104, 0.012068, 0.046652 },
	{ 0.029312, 0.012373, 0.046938 },
	{ 0.029519, 0.012677, 0.047225 },
	{ 0.029727, 0.012982, 0.047511 },
	{ 0.029935, 0.013287, 0.047797 },
	{ 0.030273, 0.013663, 0.048326 },
	{ 0.03061, 0.01404, 0.048855 },
	{ 0.030948, 0.014417, 0.049383 },
	{ 0.031286, 0.014794, 0.049912 },
	{ 0.031623, 0.01517, 0.050441 },
	{ 0.032156, 0.015707, 0.051035 },
	{ 0.032688, 0.016244, 0.05163 },
	{ 0.033221, 0.016782, 0.052225 },
	{ 0.033753, 0.017319, 0.052819 },
	{ 0.034286, 0.017856, 0.053414 },
	{ 0.034961, 0.018693, 0.054493 },
	{ 0.035636, 0.019531, 0.055573 },
	{ 0.036312, 0.020369, 0.056652 },
	{ 0.036987, 0.021206, 0.057731 },
	{ 0.037662, 0.022044, 0.058811 },
	{ 0.038623, 0.023246, 0.060044 },
	{ 0.039584, 0.024449, 0.061278 },
	{ 0.040545, 0.025651, 0.062511 },
	{ 0.041506, 0.026854, 0.063744 },
	{ 0.042468, 0.028056, 0.064978 },
	{ 0.043857, 0.029659, 0.066806 },
	{ 0.045247, 0.031263, 0.068634 },
	{ 0.046636, 0.032866, 0.070463 },
	{ 0.048026, 0.034469, 0.072291 },
	{ 0.049416, 0.036072, 0.074119 },
	{ 0.051221, 0.038156, 0.076476 },
	{ 0.053026, 0.04024, 0.078833 },
	{ 0.054831, 0.042325, 0.081189 },
	{ 0.056636, 0.044409, 0.083546 },
	{ 0.058442, 0.046493, 0.085903 },
	{ 0.06039, 0.048737, 0.087996 },
	{ 0.062338, 0.050982, 0.090088 },
	{ 0.064286, 0.053226, 0.092181 },
	{ 0.066234, 0.055471, 0.094273 },
	{ 0.068182, 0.057715, 0.096366 },
	{ 0.070519, 0.06012, 0.098921 },
	{ 0.072857, 0.062525, 0.10148 },
	{ 0.075195, 0.06493, 0.10403 },
	{ 0.077532, 0.067335, 0.10659 },
	{ 0.07987, 0.069739, 0.10914 },
	{ 0.082208, 0.072345, 0.11176 },
	{ 0.084545, 0.07495, 0.11438 },
	{ 0.086883, 0.077555, 0.117 },
	{ 0.089221, 0.08016, 0.11963 },
	{ 0.091558, 0.082766, 0.12225 },
	{ 0.094026, 0.085611, 0.12533 },
	{ 0.096494, 0.088457, 0.12841 },
	{ 0.098961, 0.091303, 0.1315 },
	{ 0.10143, 0.094148, 0.13458 },
	{ 0.1039, 0.096994, 0.13767 },
	{ 0.10688, 0.10028, 0.14119 },
	{ 0.10987, 0.10357, 0.14471 },
	{ 0.11286, 0.10685, 0.14824 },
	{ 0.11584, 0.11014, 0.15176 },
	{ 0.11883, 0.11343, 0.15529 },
	{ 0.12208, 0.11695, 0.15903 },
	{ 0.12532, 0.12048, 0.16278 },
	{ 0.12857, 0.12401, 0.16652 },
	{ 0.13182, 0.12754, 0.17026 },
	{ 0.13506, 0.13106, 0.17401 },
	{ 0.1387, 0.13499, 0.17819 },
	{ 0.14234, 0.13892, 0.18238 },
	{ 0.14597, 0.14285, 0.18656 },
	{ 0.14961, 0.14677, 0.19075 },
	{ 0.15325, 0.1507, 0.19493 },
	{ 0.15727, 0.15519, 0.19956 },
	{ 0.1613, 0.15968, 0.20419 },
	{ 0.16532, 0.16417, 0.20881 },
	{ 0.16935, 0.16866, 0.21344 },
	{ 0.17338, 0.17315, 0.21806 },
	{ 0.17805, 0.17796, 0.22291 },
	{ 0.18273, 0.18277, 0.22775 },
	{ 0.1874, 0.18758, 0.2326 },
	{ 0.19208, 0.19238, 0.23744 },
	{ 0.19675, 0.19719, 0.24229 },
	{ 0.20156, 0.20224, 0.24758 },
	{ 0.20636, 0.20729, 0.25286 },
	{ 0.21117, 0.21234, 0.25815 },
	{ 0.21597, 0.21739, 0.26344 },
	{ 0.22078, 0.22244, 0.26872 },
	{ 0.2261, 0.22806, 0.27423 },
	{ 0.23143, 0.23367, 0.27974 },
	{ 0.23675, 0.23928, 0.28524 },
	{ 0.24208, 0.24489, 0.29075 },
	{ 0.2474, 0.2505, 0.29626 },
	{ 0.25299, 0.25651, 0.3022 },
	{ 0.25857, 0.26253, 0.30815 },
	{ 0.26416, 0.26854, 0.3141 },
	{ 0.26974, 0.27455, 0.32004 },
	{ 0.27532, 0.28056, 0.32599 },
	{ 0.28156, 0.28697, 0.33238 },
	{ 0.28779, 0.29339, 0.33877 },
	{ 0.29403, 0.2998, 0.34515 },
	{ 0.30026, 0.30621, 0.35154 },
	{ 0.30649, 0.31263, 0.35793 },
	{ 0.3126, 0.31904, 0.36388 },
	{ 0.3187, 0.32545, 0.36982 },
	{ 0.32481, 0.33186, 0.37577 },
	{ 0.33091, 0.33828, 0.38172 },
	{ 0.33701, 0.34469, 0.38767 },
	{ 0.34325, 0.3511, 0.39361 },
	{ 0.34948, 0.35752, 0.39956 },
	{ 0.35571, 0.36393, 0.40551 },
	{ 0.36195, 0.37034, 0.41145 },
	{ 0.36818, 0.37675, 0.4174 },
	{ 0.37429, 0.38317, 0.42313 },
	{ 0.38039, 0.38958, 0.42885 },
	{ 0.38649, 0.39599, 0.43458 },
	{ 0.3926, 0.4024, 0.44031 },
	{ 0.3987, 0.40882, 0.44604 },
	{ 0.40494, 0.41523, 0.45198 },
	{ 0.41117, 0.42164, 0.45793 },
	{ 0.4174, 0.42806, 0.46388 },
	{ 0.42364, 0.43447, 0.46982 },
	{ 0.42987, 0.44088, 0.47577 },
	{ 0.43623, 0.44689, 0.48128 },
	{ 0.4426, 0.45291, 0.48678 },
	{ 0.44896, 0.45892, 0.49229 },
	{ 0.45532, 0.46493, 0.4978 },
	{ 0.46169, 0.47094, 0.5033 },
	{ 0.46792, 0.47695, 0.50837 },
	{ 0.47416, 0.48297, 0.51344 },
	{ 0.48039, 0.48898, 0.5185 },
	{ 0.48662, 0.49499, 0.52357 },
	{ 0.49286, 0.501, 0.52863 },
	{ 0.49805, 0.50701, 0.53392 },
	{ 0.50325, 0.51303, 0.53921 },
	{ 0.50844, 0.51904, 0.54449 },
	{ 0.51364, 0.52505, 0.54978 },
	{ 0.51883, 0.53106, 0.55507 },
	{ 0.52442, 0.53667, 0.55969 },
	{ 0.53, 0.54228, 0.56432 },
	{ 0.53558, 0.5479, 0.56894 },
	{ 0.54117, 0.55351, 0.57357 },
	{ 0.54675, 0.55912, 0.57819 },
	{ 0.55182, 0.56433, 0.58304 },
	{ 0.55688, 0.56954, 0.58789 },
	{ 0.56195, 0.57475, 0.59273 },
	{ 0.56701, 0.57996, 0.59758 },
	{ 0.57208, 0.58517, 0.60242 },
	{ 0.57675, 0.58998, 0.60639 },
	{ 0.58143, 0.59479, 0.61035 },
	{ 0.5861, 0.5996, 0.61432 },
	{ 0.59078, 0.60441, 0.61828 },
	{ 0.59545, 0.60922, 0.62225 },
	{ 0.60065, 0.61403, 0.62709 },
	{ 0.60584, 0.61884, 0.63194 },
	{ 0.61104, 0.62365, 0.63678 },
	{ 0.61623, 0.62846, 0.64163 },
	{ 0.62143, 0.63327, 0.64648 },
	{ 0.62584, 0.63808, 0.65088 },
	{ 0.63026, 0.64289, 0.65529 },
	{ 0.63468, 0.6477, 0.65969 },
	{ 0.63909, 0.65251, 0.6641 },
	{ 0.64351, 0.65731, 0.6685 },
	{ 0.64857, 0.66132, 0.67269 },
	{ 0.65364, 0.66533, 0.67687 },
	{ 0.6587, 0.66934, 0.68106 },
	{ 0.66377, 0.67335, 0.68524 },
	{ 0.66883, 0.67735, 0.68943 },
	{ 0.67273, 0.68136, 0.69361 },
	{ 0.67662, 0.68537, 0.6978 },
	{ 0.68052, 0.68938, 0.70198 },
	{ 0.68442, 0.69339, 0.70617 },
	{ 0.68831, 0.69739, 0.71035 },
	{ 0.69221, 0.7022, 0.7141 },
	{ 0.6961, 0.70701, 0.71784 },
	{ 0.7, 0.71182, 0.72159 },
	{ 0.7039, 0.71663, 0.72533 },
	{ 0.70779, 0.72144, 0.72907 },
	{ 0.71169, 0.72505, 0.73348 },
	{ 0.71558, 0.72866, 0.73789 },
	{ 0.71948, 0.73226, 0.74229 },
	{ 0.72338, 0.73587, 0.7467 },
	{ 0.72727, 0.73948, 0.7511 },
	{ 0.73247, 0.74349, 0.75507 },
	{ 0.73766, 0.74749, 0.75903 },
	{ 0.74286, 0.7515, 0.763 },
	{ 0.74805, 0.75551, 0.76696 },
	{ 0.75325, 0.75952, 0.77093 },
	{ 0.75714, 0.76393, 0.77599 },
	{ 0.76104, 0.76834, 0.78106 },
	{ 0.76494, 0.77275, 0.78612 },
	{ 0.76883, 0.77715, 0.79119 },
	{ 0.77273, 0.78156, 0.79626 },
	{ 0.77792, 0.78677, 0.80132 },
	{ 0.78312, 0.79198, 0.80639 },
	{ 0.78831, 0.79719, 0.81145 },
	{ 0.79351, 0.8024, 0.81652 },
	{ 0.7987, 0.80762, 0.82159 },
	{ 0.80519, 0.81283, 0.82687 },
	{ 0.81169, 0.81804, 0.83216 },
	{ 0.81818, 0.82325, 0.83744 },
	{ 0.82468, 0.82846, 0.84273 },
	{ 0.83117, 0.83367, 0.84802 },
	{ 0.83636, 0.83888, 0.85286 },
	{ 0.84156, 0.84409, 0.85771 },
	{ 0.84675, 0.8493, 0.86256 },
	{ 0.85195, 0.85451, 0.8674 },
	{ 0.85714, 0.85972, 0.87225 },
	{ 0.86364, 0.86613, 0.87819 },
	{ 0.87013, 0.87255, 0.88414 },
	{ 0.87662, 0.87896, 0.89009 },
	{ 0.88312, 0.88537, 0.89604 },
	{ 0.88961, 0.89178, 0.90198 },
	{ 0.8961, 0.8986, 0.90947 },
	{ 0.9026, 0.90541, 0.91696 },
	{ 0.90909, 0.91222, 0.92445 },
	{ 0.91558, 0.91904, 0.93194 },
	{ 0.92208, 0.92585, 0.93943 },
	{ 0.92857, 0.93307, 0.94493 },
	{ 0.93506, 0.94028, 0.95044 },
	{ 0.94156, 0.94749, 0.95595 },
	{ 0.94805, 0.95471, 0.96145 },
	{ 0.95455, 0.96192, 0.96696 },
	{ 0.96364, 0.96954, 0.97357 },
	{ 0.97273, 0.97715, 0.98018 },
	{ 0.98182, 0.98477, 0.98678 },
	{ 0.99091, 0.99238, 0.99339 },
	{ 1, 1, 1 }
};
static double lms0[3] = { 34.918538957799996, 19.314796676499999, 0.585610818500000 };
static double m[9] = { 18.32535, 44.60077, 7.46216, 4.09544, 28.20135, 6.66066, 0.02114, 0.10325, 1.05258 };
void rgb2dkl(Mat&inmap, Mat* outmap)
{
	//����3��length�����飬�������rgb����ͨ����ֵ
	const int length = inmap.rows*inmap.cols;
	int** im = new int*[3];
	for (int i = 0; i < 3; i++)
	{
		im[i] = new int[length];
	}
	for (int i = 0; i < inmap.cols; i++)
	{
		for (int j = 0; j < inmap.rows; j++)
		{
			im[0][i*inmap.rows + j] = inmap.ptr<Vec3b>(j)[i][2];
			im[1][i*inmap.rows + j] = inmap.ptr<Vec3b>(j)[i][1];
			im[2][i*inmap.rows + j] = inmap.ptr<Vec3b>(j)[i][0];
		}
	}
	double fac = 1 / (lms0[0] + lms0[1]);
	double mm[] = { sqrt(3.0)*fac, sqrt(3.0)*fac, 0.0, sqrt(lms0[0] * lms0[0] + lms0[1] * lms0[1]) / lms0[0] * fac, -sqrt(lms0[0] * lms0[0] + lms0[1] * lms0[1]) / lms0[1] * fac, 0.0, -fac, -fac, (lms0[0] + lms0[1]) / lms0[2] * fac };

	double* aa1 = new double[length];
	double* aa2 = new double[length];
	double* aa3 = new double[length];

	double* lms1 = new double[length];
	double* lms2 = new double[length];
	double* lms3 = new double[length];

	double* dkl1 = new double[length];
	double* dkl2 = new double[length];
	double* dkl3 = new double[length];
	for (int i = 0; i < length; i++)
	{
		aa1[i] = lut_rgb[im[0][i]][0];
		aa2[i] = lut_rgb[im[1][i]][1];
		aa3[i] = lut_rgb[im[2][i]][2];
	}
	for (int i = 0; i < length; i++)
	{
		lms1[i] = m[0] * aa1[i] + m[1] * aa2[i] + m[2] * aa3[i] - lms0[0];
		lms2[i] = m[3] * aa1[i] + m[4] * aa2[i] + m[5] * aa3[i] - lms0[1];
		lms3[i] = m[6] * aa1[i] + m[7] * aa2[i] + m[8] * aa3[i] - lms0[2];
	}
	for (int i = 0; i < length; i++)
	{
		dkl1[i] = mm[0] * lms1[i] + mm[1] * lms2[i] + mm[2] * lms3[i];
		dkl2[i] = mm[3] * lms1[i] + mm[4] * lms2[i] + mm[5] * lms3[i];
		dkl3[i] = mm[6] * lms1[i] + mm[7] * lms2[i] + mm[8] * lms3[i];
	}
	for (int i = 0; i < 3; i++)
	{
		for (int k = 0; k < (*(outmap + i)).cols; k++)
		{
			for (int j = 0; j < (*(outmap + i)).rows; j++)
			{
				if (i == 0)
				{
					(*(outmap + i)).ptr<double>(j)[k] = dkl1[k*(*(outmap + i)).rows + j]*0.5774;
				}
				else
					if (i == 1)
					{
						(*(outmap + i)).ptr<double>(j)[k] = dkl2[k*(*(outmap + i)).rows + j]*2.7525;
					}
					else
						if (i == 2)
						{
							(*(outmap + i)).ptr<double>(j)[k] = dkl3[k*(*(outmap + i)).rows + j]*0.4526;
						}
			}
		}

		//normalize((*(outmap + i)), (*(outmap + i)), 1, 255, NORM_MINMAX);
		//(*(outmap + i)).convertTo((*(outmap + i)), CV_8U);
	}
	delete[]aa1; delete[]aa2; delete[]aa3; delete[]lms1; delete[]lms2; delete[]lms3; delete[]dkl1; delete[]dkl2; delete[]dkl3;
	for (int i = 0; i < 3; i++)
	{
		delete[]im[i];
	}
	delete[]im;
}