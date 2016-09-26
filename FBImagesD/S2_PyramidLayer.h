#ifndef PYRAMIDLAYER_H
#define PYRAMIDLAYER_H

#include "S1_ImageNode.h"

class PyramidLayer{
	Mat image;								// ��һ���ͼƬ
	vector<KeyPoint> keypoints;				// ��ͼƬ��������
	Mat descriptors;						// ��ͼƬ����������  
	vector<Point2f> inlierPts;				// ��refImageƥ��ɹ���������
	vector<Point2f> refInlierPts;
	vector<DMatch> inlierMatches;			// ƥ�䣬queryIdx, trainIdx��Ȼ��Ӧ���ʼ�����keypoints��refKpoint������
	int layer;								// �ڼ��㣨0����coarsest��
	vector<ImageNode> nodes;
	Mat homoFlow;							// homography flow ����
	Mat consistImage;

	/* ---------- */
	vector<DMatch> matches;					// ƥ����
	vector<DMatch> goodMatches;
	vector<uchar> RANSACStatus;				// ����������ڴ洢RANSAC��ÿ�����״̬,ֵΪ0������ƥ��,Ұ�㣩,1 
	vector<Point2f> srcPts, dstPts;			// src��refImage��dst�ǵ�ǰͼƬ

public:
	PyramidLayer();
	PyramidLayer(Mat & newImage, int _layer);
	Mat getImage();
	void calKeyPoints();					// �����һ���ͼƬ��������(ʹ��SURF���ӣ�

	// ʹ��BRIEF������ȡ����������������������, descriptor; BRIEF: ��ʡ�ռ䣬��
	void calImageDescriptors();

	// ����ƥ���������
	void calMatchPtsWithRef(Mat & refDescriptor, vector<KeyPoint> & refKpoint, Ptr<DescriptorMatcher> matcher);

	void distributeFeaturePts();				// ����ͼ��matched keypoints���䵽����ImageNode

	void addMatchedPts(Point2f & point, Point2f & refPoint);

	void showMatchedImg(Mat & refImg, vector<KeyPoint> & refKpt);		// ��ʾƥ��ͼ��

	Mat getImageDescriptors();

	void calNodeHomography(int row, int col, Mat parentHomography);		// ��������ڵ��Homography

	void optimizeHomography();                                          // �Ż���һ���homography

	void calHomographyFlow();											// ������һ���Homography Flow

	void calHomographyFlowByScale(Mat & finestHomoFlow, int scale);		// ����homography flow

	void calConsistImage();

	Mat getConsistImage();
	Mat & getHomoFlow();
	Mat getNodesHomography(int row, int col);
	vector<KeyPoint> & getKeypoints();
	vector<Point2f> & getRefMatchPts();
	vector<Point2f> & getCurMatchPts();


	void distributeFeaturePts1(int rows, int cols);
};

#endif

