#include "S2_PyramidLayer.h"

PyramidLayer::PyramidLayer(){}

PyramidLayer::PyramidLayer(Mat & newImage, int _layer){
	image = newImage;
	layer = _layer;
}

Mat PyramidLayer::getImage(){
	return image;
}

// �����һ���ͼƬ��������(ʹ��SURF���ӣ�
void PyramidLayer::calKeyPoints(){
	int hessianThreshold = 400;								// Hessian ��������ʽ��Ӧֵ����ֵ, hҪ���ڸ�ֵ
	SurfFeatureDetector surfDetector(hessianThreshold);		// ����һ��SurfFeatureDetector��SURF, SurfDescriptorExtractor�� ������������
	surfDetector.detect(image, keypoints);
	cout << "Keypoints Num: " << keypoints.size() << endl;
	/* Mat imgKeypoints;									// ���������ͼ,���¼�����ʾ��ͼ
	drawKeypoints(image, keypoints, imgKeypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("surfImage", imgKeypoints);
	waitKey(0); */
}

// ʹ��BRIEF������ȡ����������������������, descriptor; BRIEF: ��ʡ�ռ䣬��
void PyramidLayer::calImageDescriptors(){
	calKeyPoints();
	BriefDescriptorExtractor briefExtractor;
	briefExtractor.compute(image, keypoints, descriptors);
	cout << "Descriptors Size: " << descriptors.size() << endl;
}

// ����ƥ���������
void PyramidLayer::calMatchPtsWithRef(Mat & refDescriptor, vector<KeyPoint> & refKpoint, Ptr<DescriptorMatcher> matcher){
	calImageDescriptors();									// ���㵱֡������������������

	// ������������ƥ��
	matches.clear();
	matcher->match(descriptors, refDescriptor, matches);	// queryDescriptor, trainDescriptor
	cout << "Matches Num: " << matches.size() << endl;		// �����size��queryDescriptor������һ��,Ϊquery��ÿһ������������һ��ƥ������

	// ���ݾ��룬ѡ�����еĽ��ŵ�ƥ���
	double maxDist = 0;
	double minDist = 100;
	for (unsigned int i = 0; i < matches.size(); i++){
		double dist = matches[i].distance;
		if (dist < minDist) minDist = dist;
		if (dist > maxDist) maxDist = dist;
	}
	cout << "Max Distance: " << maxDist << endl;
	cout << "Min Distance: " << minDist << endl;

	goodMatches.clear();
	for (unsigned int i = 0; i < matches.size(); i++){
		if (matches[i].distance <= minDist + GOOD_MATCH_THRE * (maxDist - minDist)){
			goodMatches.push_back(matches[i]);
		}
	}
	cout << "Good Matches Num: " << goodMatches.size() << endl;

	// �ֱ�ȡ������ͼ����ƥ���������
	int matchedNum = (int)goodMatches.size();
	vector<Point2f> refMatchPts, curMatchPts;
	for (int i = 0; i < matchedNum; i++){
		refMatchPts.push_back(refKpoint[goodMatches[i].trainIdx].pt);
		curMatchPts.push_back(keypoints[goodMatches[i].queryIdx].pt);
	}

	// �����������F(��RANSAC����)����ʾ����ĳ������򳡾��������ڲ�ͬ��������Ƭ��Ӧ������ͼ������Ĺ�ϵ��x'��ת�ó���F���ٳ���x�Ľ��Ϊ0
	// RANSACΪRANdom Sample Consensus����д�����Ǹ���һ������쳣���ݵ��������ݼ�����������ݵ���ѧģ�Ͳ������õ���Ч�������ݵ��㷨
	Mat fundMat;
	RANSACStatus.clear(); // ����������ڴ洢RANSAC��ÿ�����״̬,ֵΪ0������ƥ��,Ұ�㣩,1 
	findFundamentalMat(curMatchPts, refMatchPts, RANSACStatus, FM_RANSAC);

	// ʹ��RANSAC������������������Եõ�һ��status����������ɾ�������ƥ�䣨֮ǰ�Ѿ�ɸ��һ���ˣ����Բ��Ǻ���Ч����
	for (int i = 0; i < matchedNum; i++){
		if (RANSACStatus[i] != 0){
			refInlierPts.push_back(refMatchPts[i]);
			inlierPts.push_back(curMatchPts[i]);
			inlierMatches.push_back(goodMatches[i]);		// ���inlierMatches��queryIdx, trainIdx��Ȼ��Ӧ���ʼ�����keypoints��refKpoint������
		}
	}
	cout << "Matches Num After RANSAC: " << inlierMatches.size() << endl;
	cout << endl;
}

// ����ͼ��matched keypoints���䵽����ImageNode
void PyramidLayer::distributeFeaturePts(){
	int nodeNumPerEdge = 1 << layer;
	int nodeLength = image.cols / nodeNumPerEdge;
	int nodeWidth = image.rows / nodeNumPerEdge;
	//cout << "node length: " << nodeLength << " node width: " << nodeWidth << endl;

	int nodeNum = nodeNumPerEdge * nodeNumPerEdge;
	nodes.resize(nodeNum);

	for (unsigned int i = 0; i < refInlierPts.size(); i++){
		int col = (int)floor(refInlierPts[i].x / nodeLength);
		int row = (int)floor(refInlierPts[i].y / nodeWidth);

		//cout << inlierPts[i].x << "," << inlierPts[i].y << endl;
		//cout << col << "," << row << endl;
		nodes[row * nodeNumPerEdge + col].addMatchedPts(i);
	}
}

void PyramidLayer::distributeFeaturePts1(int rows, int cols){  // rows, cols��Ϊԭͼ�ߴ�
	int nodeNumPerEdge = 1 << layer;
	int nodeLength = cols / nodeNumPerEdge;
	int nodeWidth = rows / nodeNumPerEdge;
	//cout << "node length: " << nodeLength << " node width: " << nodeWidth << endl;

	int nodeNum = nodeNumPerEdge * nodeNumPerEdge;
	nodes.resize(nodeNum);

	for (unsigned int i = 0; i < refInlierPts.size(); i++){
		int col = (int)floor(refInlierPts[i].x / nodeLength);
		int row = (int)floor(refInlierPts[i].y / nodeWidth);

		//cout << inlierPts[i].x << "," << inlierPts[i].y << endl;
		//cout << col << "," << row << endl;
		nodes[row * nodeNumPerEdge + col].addMatchedPts(i);
	}
}

void PyramidLayer::addMatchedPts(Point2f & point, Point2f & refPoint){
	inlierPts.push_back(point);
	refInlierPts.push_back(refPoint);
}

// ��ʾƥ��ͼ��
void PyramidLayer::showMatchedImg(Mat & refImg, vector<KeyPoint> & refKpt){
	Mat matchedImg;
	drawMatches(image, keypoints, refImg, refKpt, inlierMatches, matchedImg,
		Scalar::all(-1), CV_RGB(0, 255, 0), Mat(), 2);
	imshow("Matched Result", matchedImg);
	waitKey(0);
}

Mat PyramidLayer::getImageDescriptors(){
	calImageDescriptors();
	return descriptors;
}

// ��������ڵ��Homography
void PyramidLayer::calNodeHomography(int row, int col, Mat parentHomography){
	int featurePtSize = nodes[row * (1 << layer) + col].getMatchedPtsSize();
	//cout << "Feature Points Num��" << featurePtSize << endl;
	// �����Node����������̫�٣���ȡ��һ���Homography
	if (featurePtSize < 8){
		nodes[row * (1 << layer) + col].passParentHomography(parentHomography);
	}
	else{
		nodes[row * (1 << layer) + col].calHomography(inlierPts, refInlierPts);
	}
}

//������һ���Homography Flow
void PyramidLayer::calHomographyFlow(){
	int edgeLen = 1 << layer;
	int nodeLength = image.cols / edgeLen;
	int nodeWidth = image.rows / edgeLen;
	homoFlow = Mat::zeros(image.rows, image.cols, CV_32FC2);
	// ����ÿһ��node��homography flow
	for (int r = 0; r < edgeLen; r++){
		for (int c = 0; c < edgeLen; c++){
			Mat H = nodes[r * edgeLen + c].getHomography();
			// �����node��Χ
			int rowStart = r * nodeWidth;
			int rowEnd = (r + 1) * nodeWidth - 1;
			if (r == edgeLen - 1) rowEnd = image.rows - 1;
			int colStart = c * nodeLength;
			int colEnd = (c + 1) * nodeLength - 1;
			if (c == edgeLen - 1) colEnd = image.cols - 1;

			srcPts.clear();  // src��refImage��dst�ǵ�ǰͼƬ
			dstPts.clear();
			for (int row = rowStart; row <= rowEnd; row++)   // y
			for (int col = colStart; col <= colEnd; col++)  // x
				srcPts.push_back(Point2f((float)col, (float)row));

			perspectiveTransform(srcPts, dstPts, H);  // [x y 1]*H = [x' y' w'], src(x, y) -> dst(x'/w', y'/w')

			int idx = 0;
			for (int row = rowStart; row <= rowEnd; row++){   // y
				for (int col = colStart; col <= colEnd; col++){  // x
					float deltaRow = dstPts[idx].y - srcPts[idx].y;    //  (�ɲο�ͼ��任��)��ǰͼ���row - refImage��row
					float deltaCol = dstPts[idx].x - srcPts[idx].x;    //  (�ɲο�ͼ��任��)��ǰͼ���col - refImage��col

					Vec2f& elem = homoFlow.at<Vec2f>(row, col);// or homoFlow.at<Vec2f>( Point(col,row) );
					elem[0] = deltaRow;
					elem[1] = deltaCol;

					idx++;
				}
			}

		}
	}
}

// ����homography flow
void PyramidLayer::calHomographyFlowByScale(Mat & finestHomoFlow, int scale){
	resize(finestHomoFlow, homoFlow, image.size(), 0, 0, CV_INTER_AREA);
	homoFlow = homoFlow / (float)scale;
}

void PyramidLayer::calConsistImage(){
	consistImage = Mat::zeros(image.size(), CV_8UC3);
	for (int r = 0; r < image.rows; r++){     // consistentͼ��(r,c)
		for (int c = 0; c < image.cols; c++){
			Vec2f& elem = homoFlow.at<Vec2f>(r, c);
			int oriRow = (int)(r + elem[0] + 0.5);   // ��������
			int oriCol = (int)(c + elem[1] + 0.5);
			oriRow = min(max(oriRow, 0), image.rows - 1);
			oriCol = min(max(oriCol, 0), image.cols - 1);
			consistImage.at<Vec3b>(r, c) = image.at<Vec3b>(oriRow, oriCol);
		}
	}

	//imshow("consist", consistImage);
	//waitKey(0);
}

Mat PyramidLayer::getConsistImage(){
	return consistImage;
}

Mat & PyramidLayer::getHomoFlow(){
	return homoFlow;
}

Mat PyramidLayer::getNodesHomography(int row, int col){
	return nodes[row * (1 << layer) + col].getHomography();
}

vector<KeyPoint> & PyramidLayer::getKeypoints(){
	return keypoints;
}

vector<Point2f> & PyramidLayer::getRefMatchPts(){
	return refInlierPts;
}

vector<Point2f> & PyramidLayer::getCurMatchPts(){
	return inlierPts;
}


