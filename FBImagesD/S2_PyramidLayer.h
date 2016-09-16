#ifndef PYRAMIDLAYER_H
#define PYRAMIDLAYER_H

#include "S1_ImageNode.h"

class PyramidLayer{
	Mat image;								// 这一层的图片
	vector<KeyPoint> keypoints;				// 该图片的特征点
	Mat descriptors;						// 该图片的特征向量  
	vector<Point2f> inlierPts;				// 和refImage匹配成功的特征点
	vector<Point2f> refInlierPts;
	vector<DMatch> inlierMatches;			// 匹配，queryIdx, trainIdx依然对应着最开始算出的keypoints和refKpoint的索引
	int layer;								// 第几层（0层是coarsest）
	vector<ImageNode> nodes;
	Mat homoFlow;							// homography flow 矩阵
	Mat consistImage;

	/* ---------- */
	vector<DMatch> matches;					// 匹配结果
	vector<DMatch> goodMatches;
	vector<uchar> RANSACStatus;				// 这个变量用于存储RANSAC后每个点的状态,值为0（错误匹配,野点）,1 
	vector<Point2f> srcPts, dstPts;			// src是refImage，dst是当前图片

public:
	PyramidLayer();
	PyramidLayer(Mat & newImage, int _layer);
	Mat getImage();
	void calKeyPoints();					// 检测这一层的图片的特征点(使用SURF算子）

	// 使用BRIEF算子提取特征（计算特征向量）即, descriptor; BRIEF: 节省空间，快
	void calImageDescriptors();

	// 计算匹配的特征点
	void calMatchPtsWithRef(Mat & refDescriptor, vector<KeyPoint> & refKpoint, Ptr<DescriptorMatcher> matcher);

	void distributeFeaturePts();				// 将该图形matched keypoints分配到各个ImageNode

	void addMatchedPts(Point2f & point, Point2f & refPoint);

	void showMatchedImg(Mat & refImg, vector<KeyPoint> & refKpt);		// 显示匹配图形

	Mat getImageDescriptors();

	void calNodeHomography(int row, int col, Mat parentHomography);		// 计算给定节点的Homography

	void calHomographyFlow();											// 计算这一层的Homography Flow

	void calHomographyFlowByScale(Mat & finestHomoFlow, int scale);		// 缩放homography flow

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

