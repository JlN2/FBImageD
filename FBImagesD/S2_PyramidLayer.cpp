#include "S2_PyramidLayer.h"

PyramidLayer::PyramidLayer(){}

PyramidLayer::PyramidLayer(Mat & newImage, int _layer){
	image = newImage;
	layer = _layer;
}

Mat PyramidLayer::getImage(){
	return image;
}

// 检测这一层的图片的特征点(使用SURF算子）
void PyramidLayer::calKeyPoints(){
	int hessianThreshold = 400;								// Hessian 矩阵行列式响应值的阈值, h要大于该值
	SurfFeatureDetector surfDetector(hessianThreshold);		// 定义一个SurfFeatureDetector（SURF, SurfDescriptorExtractor） 特征检测类对象
	surfDetector.detect(image, keypoints);
	cout << "Keypoints Num: " << keypoints.size() << endl;
	/* Mat imgKeypoints;									// 带特征点的图,以下几步显示该图
	drawKeypoints(image, keypoints, imgKeypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("surfImage", imgKeypoints);
	waitKey(0); */
}

// 使用BRIEF算子提取特征（计算特征向量）即, descriptor; BRIEF: 节省空间，快
void PyramidLayer::calImageDescriptors(){
	calKeyPoints();
	BriefDescriptorExtractor briefExtractor;
	briefExtractor.compute(image, keypoints, descriptors);
	cout << "Descriptors Size: " << descriptors.size() << endl;
}

// 计算匹配的特征点
void PyramidLayer::calMatchPtsWithRef(Mat & refDescriptor, vector<KeyPoint> & refKpoint, Ptr<DescriptorMatcher> matcher){
	calImageDescriptors();									// 计算当帧的特征向量和特征点

	// 进行特征向量匹配
	matches.clear();
	matcher->match(descriptors, refDescriptor, matches);	// queryDescriptor, trainDescriptor
	cout << "Matches Num: " << matches.size() << endl;		// 这里的size和queryDescriptor的行数一样,为query的每一个向量都找了一个匹配向量

	// 根据距离，选出其中的较优的匹配点
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

	// 分别取出两个图像中匹配的特征点
	int matchedNum = (int)goodMatches.size();
	vector<Point2f> refMatchPts, curMatchPts;
	for (int i = 0; i < matchedNum; i++){
		refMatchPts.push_back(refKpoint[goodMatches[i].trainIdx].pt);
		curMatchPts.push_back(keypoints[goodMatches[i].queryIdx].pt);
	}

	// 计算基础矩阵F(用RANSAC方法)：表示的是某个物体或场景各特征在不同的两张照片对应特征点图像坐标的关系，x'的转置乘以F，再乘以x的结果为0
	// RANSAC为RANdom Sample Consensus的缩写，它是根据一组包含异常数据的样本数据集，计算出数据的数学模型参数，得到有效样本数据的算法
	Mat fundMat;
	RANSACStatus.clear(); // 这个变量用于存储RANSAC后每个点的状态,值为0（错误匹配,野点）,1 
	findFundamentalMat(curMatchPts, refMatchPts, RANSACStatus, FM_RANSAC);

	// 使用RANSAC方法计算基础矩阵后可以得到一个status向量，用来删除错误的匹配（之前已经筛过一遍了，所以不是很有效果）
	for (int i = 0; i < matchedNum; i++){
		if (RANSACStatus[i] != 0){
			refInlierPts.push_back(refMatchPts[i]);
			inlierPts.push_back(curMatchPts[i]);
			inlierMatches.push_back(goodMatches[i]);		// 这个inlierMatches的queryIdx, trainIdx依然对应着最开始算出的keypoints和refKpoint的索引
		}
	}
	cout << "Matches Num After RANSAC: " << inlierMatches.size() << endl;
	cout << endl;
}

// 将该图形matched keypoints分配到各个ImageNode
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

void PyramidLayer::distributeFeaturePts1(int rows, int cols){  // rows, cols均为原图尺寸
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

// 显示匹配图形
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

// 计算给定节点的Homography
void PyramidLayer::calNodeHomography(int row, int col, Mat parentHomography){
	int featurePtSize = nodes[row * (1 << layer) + col].getMatchedPtsSize();
	//cout << "Feature Points Num：" << featurePtSize << endl;
	// 如果该Node特征点数量太少，则取上一层的Homography
	if (featurePtSize < 8){
		nodes[row * (1 << layer) + col].passParentHomography(parentHomography);
	}
	else{
		nodes[row * (1 << layer) + col].calHomography(inlierPts, refInlierPts);
	}
}

// 优化这一层的homography
void PyramidLayer::optimizeHomography(){
	int nodeNumPerEdge = 1 << layer;
	int totNode = nodeNumPerEdge * nodeNumPerEdge;
	int dr[] = {-1, 1, 0, 0};   // 上，下，左，右
	int dc[] = {0, 0, -1, 1};

	Mat R(3, 3 * totNode, CV_64F, Scalar::all(0));              // 这个矩阵装所有的原来的homography
	Mat A(3 * totNode, 3 * totNode, CV_64F, Scalar::all(0));    // 这个矩阵是系数矩阵，来代表H的运算
	Mat H(3, 3 * totNode, CV_64F, Scalar::all(0));              // 未知待求的矩阵，优化后的homography
	
	for(int r = 0; r < nodeNumPerEdge; r++)
		for(int c = 0; c < nodeNumPerEdge; c++){
			Mat homo = getNodesHomography(r, c);
			
			// 填R矩阵
			int curIdx = r * nodeNumPerEdge + c;        
			for(int i = 0; i < 3; i++)
				for(int j = 0; j < 3; j++){
					R.at<double>(i, j + curIdx * 3) = homo.at<double>(i, j); 
				}
			
			// 填A矩阵
			int cnt = 0;                                // 算homo有几个邻接的homo
			for(int i = 0; i < 4; i++){                    
				int neighborR = r + dr[i];
				int neighborC = c + dc[i];
				if(neighborR < 0 || neighborR >= nodeNumPerEdge ||  neighborC < 0 || neighborC >= nodeNumPerEdge)
					continue;

				cnt++;

				int neighborIdx = neighborR * nodeNumPerEdge + neighborC;
				
				int startR = neighborIdx * 3;
				int startC = curIdx * 3;
				for(int k = 0; k < 3; k++) A.at<double>(startR + k, startC + k) = -1;
			}

			for(int k = 0; k < 3; k++) A.at<double>(curIdx * 3 + k, curIdx * 3 + k) = cnt;

		}
	
	Mat P = Mat::eye(3 * totNode, 3 * totNode, CV_64F);
	P = P + LAMBDA * 2 * A;
	H = R * P.inv();          // H的尺寸为3*(3*totNode)

	// 将算出的H填回到各个node的homography矩阵中
	for(int r = 0; r < nodeNumPerEdge; r++)
		for(int c = 0; c < nodeNumPerEdge; c++){
			Mat tempHomo(3, 3, CV_64F, Scalar::all(0));
			int curIdx = r * nodeNumPerEdge + c;
			for(int i = 0; i < 3; i++)
				for(int j = 0; j < 3; j++){
					tempHomo.at<double>(i, j) = H.at<double>(i, j + curIdx * 3); 
				}
			nodes[r * nodeNumPerEdge + c].updateOptimizedHomo(tempHomo);
		}
}

//计算这一层的Homography Flow
void PyramidLayer::calHomographyFlow(){
	int edgeLen = 1 << layer;
	int nodeLength = image.cols / edgeLen;
	int nodeWidth = image.rows / edgeLen;
	homoFlow = Mat::zeros(image.rows, image.cols, CV_32FC2);
	// 计算每一个node的homography flow
	for (int r = 0; r < edgeLen; r++){
		for (int c = 0; c < edgeLen; c++){
			Mat H = nodes[r * edgeLen + c].getHomography();
			// 算出该node范围
			int rowStart = r * nodeWidth;
			int rowEnd = (r + 1) * nodeWidth - 1;
			if (r == edgeLen - 1) rowEnd = image.rows - 1;
			int colStart = c * nodeLength;
			int colEnd = (c + 1) * nodeLength - 1;
			if (c == edgeLen - 1) colEnd = image.cols - 1;

			srcPts.clear();  // src是refImage，dst是当前图片
			dstPts.clear();
			for (int row = rowStart; row <= rowEnd; row++)   // y
			for (int col = colStart; col <= colEnd; col++)  // x
				srcPts.push_back(Point2f((float)col, (float)row));

			perspectiveTransform(srcPts, dstPts, H);  // [x y 1]*H = [x' y' w'], src(x, y) -> dst(x'/w', y'/w')

			int idx = 0;
			for (int row = rowStart; row <= rowEnd; row++){   // y
				for (int col = colStart; col <= colEnd; col++){  // x
					float deltaRow = dstPts[idx].y - srcPts[idx].y;    //  (由参考图像变换的)当前图像的row - refImage的row
					float deltaCol = dstPts[idx].x - srcPts[idx].x;    //  (由参考图像变换的)当前图像的col - refImage的col

					Vec2f& elem = homoFlow.at<Vec2f>(row, col);// or homoFlow.at<Vec2f>( Point(col,row) );
					elem[0] = deltaRow;
					elem[1] = deltaCol;

					idx++;
				}
			}

		}
	}
}

// 缩放homography flow
void PyramidLayer::calHomographyFlowByScale(Mat & finestHomoFlow, int scale){
	resize(finestHomoFlow, homoFlow, image.size(), 0, 0, CV_INTER_AREA);
	homoFlow = homoFlow / (float)scale;
}

void PyramidLayer::calConsistImage(){
	consistImage = Mat::zeros(image.size(), CV_8UC3);
	for (int r = 0; r < image.rows; r++){     // consistent图像(r,c)
		for (int c = 0; c < image.cols; c++){
			Vec2f& elem = homoFlow.at<Vec2f>(r, c);
			int oriRow = (int)(r + elem[0] + 0.5);   // 四舍五入
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


