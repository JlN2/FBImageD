#ifndef PYRAMID_H
#define PYRAMID_H

#include "S2_PyramidLayer.h"

class Pyramid{
	vector<PyramidLayer> pyramid;			// Ĭ����private
public:
	Pyramid();
	Pyramid(Mat & Image);					// ���캯������Image����һ��Image Pyramid
	void calFeaturePyramid();				// ����ÿһ��������㣨��coarse level��������scale�������㣩
	void distributeFeaturePtsByLayer();		// ��ÿһ���������ֵ�ÿһ��ImageNode
	void calHomographyPyramid();			// �������ͼ���������homography������
	void calHomographyFlowPyramid();		// �������ͼ���������homography flow����ʵ���൱��ÿ����������ڲο�֡��ƫ������
	vector<Mat> getImagePyramid();
	PyramidLayer* getPyramidLayer(int layer);


	void calFeaturePyramid1();
	void distributeFeaturePtsByLayer1(int rows, int cols);
};

#endif

