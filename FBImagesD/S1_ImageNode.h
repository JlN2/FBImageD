#ifndef IMAGENODE_H
#define IMAGENODE_H

#include "S0_ConstDef.h"

class ImageNode{
	vector<int> matchedPts;
	Mat H;										// ���Ӧ�Ĳο�֡��ImageNode��homography����3*3��

public:
	void addMatchedPts(int idx);
	void calHomography(vector<Point2f> & pts, vector<Point2f> & refPts);
	void passParentHomography(Mat parentH);

	Mat getHomography();
	unsigned int getMatchedPtsSize();
};


#endif

