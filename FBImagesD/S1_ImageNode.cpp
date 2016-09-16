#include "S1_ImageNode.h"

void ImageNode::addMatchedPts(int idx){
	matchedPts.push_back(idx);
}

void ImageNode::calHomography(vector<Point2f> & pts, vector<Point2f> & refPts){
	vector<Point2f> points, refPoints;
	for (unsigned int i = 0; i < matchedPts.size(); i++){
		points.push_back(pts[matchedPts[i]]);
		refPoints.push_back(refPts[matchedPts[i]]);
	}
	H = findHomography(refPoints, points, CV_RANSAC); // 从参考帧->当前帧
}

void ImageNode::passParentHomography(Mat parentH){
	H = parentH;
}

Mat ImageNode::getHomography(){
	return H;
}

unsigned int ImageNode::getMatchedPtsSize(){
	return matchedPts.size();
}