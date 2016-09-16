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
	vector<Mat> consistentPixels;			// ����൱��ÿ��һ��
	vector<Mat> channels;					// ������̬ѧ����ʱ�������֡ͼ��

public:
	ConsistentPixelSetPyramid();
	ConsistentPixelSetPyramid(int layerNum);
	//uchar find(uchar x, uchar parent[]);					// ���鼯����
	//void join(uchar s, uchar l, uchar parent[]);

	// ʹ��BFSʵ�ֺ�ˮ����㷨
	ConnectedComponent floodFill(int idx, int x0, int y0, Mat & undecidedPixels);
	void findConnectedComponents(vector<ConnectedComponent> & connComps, Mat & undecidedPixels);
	int checkMajority(const Mat& src, const int row, const int col);
	void morphMajority(const Mat & src, Mat & dst);

	// ����layer���consistent pixels set
	void calConsistentPixelSet(int layer, vector<Mat> & integralImageSet, Mat & integralMedianImg, const int thre);

	// �����²����õ����в��consistent pixels
	void calConsistentPixelsAllLayer(vector<Mat> & refPyramid);
	vector<Mat> & getConsistentPixelPyramid();
};

#endif


