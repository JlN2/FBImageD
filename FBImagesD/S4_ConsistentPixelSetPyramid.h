#ifndef CONSISTENT_PIXEL_SET_PYRAMID
#define CONSISTENT_PIXEL_SET_PYRAMID

#include "S0_ConstDef.h"

class ConnectedComponent{
	vector<Point> pts;
	int idx;

public:
	ConnectedComponent();
	ConnectedComponent(int _idx);
	void addPoint(Point p);
	int getIdx();
	vector<Point> & getCCPts();
};

class ConsistentPixelSetPyramid{
	vector<Mat> consistentPixels;			// 这个相当于每层一个
	vector<Mat> channels;					// 用于形态学过滤时，分离各帧图像

public:
	ConsistentPixelSetPyramid();
	ConsistentPixelSetPyramid(int layerNum);
	//uchar find(uchar x, uchar parent[]);					// 并查集操作
	//void join(uchar s, uchar l, uchar parent[]);

	// 使用BFS实现洪水灌溉算法
	ConnectedComponent floodFill(int idx, int x0, int y0, Mat & undecidedPixels);
	void findConnectedComponents(vector<ConnectedComponent> & connComps, Mat & undecidedPixels);
	int checkMajority(const Mat& src, const int row, const int col);
	void morphMajority(const Mat & src, Mat & dst);

	// 计算layer层的consistent pixels set
	void calConsistentPixelSet(int layer, vector<Mat> & integralImageSet, Mat & integralMedianImg, const int thre);

	// 由上下采样得到所有层的consistent pixels
	void calConsistentPixelsAllLayer(vector<Mat> & refPyramid);
	vector<Mat> & getConsistentPixelPyramid();
};

#endif
