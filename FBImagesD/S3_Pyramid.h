#ifndef PYRAMID_H
#define PYRAMID_H

#include "S2_PyramidLayer.h"

class Pyramid{
	vector<PyramidLayer> pyramid;			// 默认是private
public:
	Pyramid();
	Pyramid(Mat & Image);					// 构造函数，对Image构造一个Image Pyramid
	void calFeaturePyramid();				// 计算每一层的特征点（将coarse level的特征点scale到其他层）
	void distributeFeaturePtsByLayer();		// 将每一层的特征点分到每一个ImageNode
	void calHomographyPyramid();			// 计算这个图像金字塔的homography金字塔
	void calHomographyFlowPyramid();		// 计算这个图像金字塔的homography flow（其实就相当于每个像素相对于参考帧的偏移量）
	vector<Mat> getImagePyramid();
	PyramidLayer* getPyramidLayer(int layer);


	void calFeaturePyramid1();
	void distributeFeaturePtsByLayer1(int rows, int cols);
};

#endif

