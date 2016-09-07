#include <iostream>
#include <vector>
#include <stdio.h>
#include <opencv2\core\core.hpp>     // 有Mat数据类型
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <math.h>

using namespace std;
using namespace cv;

const string fileDir  = "SrcData\\Data\\Bookshelf_2\\";
const string resultDir = "Result\\";
const string imageFormat = ".jpg";
const int FRAME_NUM = 10;
const int REF = FRAME_NUM / 2;
const int FEATURE_LAYER = 0;   // 应该从哪一层开始算特征向量：starting from a global homography at the coarsest level
const int CONSIST_LAYER = 0;

int max(int a, int b){
	if(a < b) return b;
	return a;
}

int min(int a, int b){
	if(a < b) return a;
	return b;
}

class ImageNode{
	vector<int> matchedPts;
	Mat H; // 与对应的参考帧的ImageNode的homography矩阵（3*3）

public:

	void addMatchedPts(int idx){
		matchedPts.push_back(idx);
	}

	void calHomography(vector<Point2f> & pts, vector<Point2f> & refPts){
		vector<Point2f> points, refPoints;
		for(unsigned int i = 0; i < matchedPts.size(); i++){
			points.push_back(pts[matchedPts[i]]);
			refPoints.push_back(refPts[matchedPts[i]]);
		}
		H = findHomography(refPoints, points, CV_RANSAC); // 从参考帧->当前帧
	}

	void passParentHomography(Mat parentH){
		H = parentH;
	}

	Mat getHomography(){
		return H;
	}

	unsigned int getMatchedPtsSize(){
		return matchedPts.size();
	}
	
};

class PyramidLayer{
	Mat image; // 这一层的图片
	vector<KeyPoint> keypoints;  // 该图片的特征点
	Mat descriptors; // 该图片的特征向量  
	vector<Point2f> inlierPts; // 和refImage匹配成功的特征点
	vector<Point2f> refInlierPts; 
	vector<DMatch> inlierMatches; // 匹配，queryIdx, trainIdx依然对应着最开始算出的keypoints和refKpoint的索引
	int layer;  // 第几层（0层是coarsest）
	vector<ImageNode> nodes;
	Mat homoFlow;  // homography flow 矩阵
	Mat consistImage;

public:
	PyramidLayer(){}
	PyramidLayer(Mat & newImage, int _layer){
		image = newImage;
		layer = _layer;
	}

	Mat getImage(){
		return image;
	}

	// 检测这一层的图片的特征点(使用SURF算子）
	void calKeyPoints(){
		int hessianThreshold = 400; // Hessian 矩阵行列式响应值的阈值, h要大于该值
		SurfFeatureDetector surfDetector(hessianThreshold); // 定义一个SurfFeatureDetector（SURF, SurfDescriptorExtractor） 特征检测类对象
		surfDetector.detect(image, keypoints);
		cout << "Keypoints Num: " << keypoints.size() << endl;
		/* Mat imgKeypoints;   // 带特征点的图,以下几步显示该图
		drawKeypoints(image, keypoints, imgKeypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		imshow("surfImage", imgKeypoints);
		waitKey(0); */
	}

	// 使用BRIEF算子提取特征（计算特征向量）即, descriptor; BRIEF: 节省空间，快
	void calImageDescriptors(){
		calKeyPoints();
		BriefDescriptorExtractor briefExtractor;
		briefExtractor.compute(image, keypoints, descriptors);
		cout << "Descriptors Size: " << descriptors.size() << endl;
	}

	// 计算匹配的特征点
	void calMatchPtsWithRef(Mat & refDescriptor, vector<KeyPoint> & refKpoint, Ptr<DescriptorMatcher> matcher){
		calImageDescriptors();  // 计算当帧的特征向量和特征点

		// 进行特征向量匹配
		vector<DMatch> matches; // 匹配结果
		matcher->match(descriptors, refDescriptor, matches); // queryDescriptor, trainDescriptor
		cout << "Matches Num: " << matches.size() << endl; // 这里的size和queryDescriptor的行数一样,为query的每一个向量都找了一个匹配向量
		
		// 根据距离，选出其中的较优的匹配点
		double maxDist = 0;
		double minDist = 100;
		for(unsigned int i = 0; i < matches.size(); i++){
			double dist = matches[i].distance;
			if(dist < minDist) minDist = dist;
			if(dist > maxDist) maxDist = dist;
		}
		cout << "Max Distance: " << maxDist << endl;
		cout << "Min Distance: " << minDist << endl;

		vector<DMatch> goodMatches;
		for(unsigned int i = 0; i < matches.size(); i++){
			if(matches[i].distance < minDist + 0.38 * (maxDist - minDist)){
				goodMatches.push_back(matches[i]);
			}
		}
		cout << "Good Matches Num: " << goodMatches.size() << endl;

		// 分别取出两个图像中匹配的特征点
		int matchedNum = (int)goodMatches.size();
		vector<Point2f> refMatchPts, curMatchPts;
		for(int i = 0; i < matchedNum; i++){
			refMatchPts.push_back(refKpoint[goodMatches[i].trainIdx].pt);
			curMatchPts.push_back(keypoints[goodMatches[i].queryIdx].pt);
		}

		// 计算基础矩阵F(用RANSAC方法)：表示的是某个物体或场景各特征在不同的两张照片对应特征点图像坐标的关系，x'的转置乘以F，再乘以x的结果为0
		// RANSAC为RANdom Sample Consensus的缩写，它是根据一组包含异常数据的样本数据集，计算出数据的数学模型参数，得到有效样本数据的算法
		Mat fundMat;
		vector<uchar> RANSACStatus;   // 这个变量用于存储RANSAC后每个点的状态,值为0（错误匹配,野点）,1 
		findFundamentalMat(curMatchPts, refMatchPts, RANSACStatus, FM_RANSAC);
			
		// 使用RANSAC方法计算基础矩阵后可以得到一个status向量，用来删除错误的匹配（之前已经筛过一遍了，所以不是很有效果）
		for(int i = 0; i < matchedNum; i++){
			if(RANSACStatus[i] != 0){ 
				refInlierPts.push_back(refMatchPts[i]);
				inlierPts.push_back(curMatchPts[i]);
				inlierMatches.push_back(goodMatches[i]);  // 这个inlierMatches的queryIdx, trainIdx依然对应着最开始算出的keypoints和refKpoint的索引
			}
		}
		cout << "Matches Num After RANSAC: " << inlierMatches.size() << endl;
		cout << endl;
	}

	// 将该图形matched keypoints分配到各个ImageNode
	void distributeFeaturePts(){
		int nodeNumPerEdge = 1 << layer;  
		int nodeLength = image.cols / nodeNumPerEdge;
		int nodeWidth = image.rows / nodeNumPerEdge;
		//cout << "node length: " << nodeLength << " node width: " << nodeWidth << endl;

		int nodeNum = nodeNumPerEdge * nodeNumPerEdge;
		nodes.resize(nodeNum);

		for(unsigned int i = 0; i < refInlierPts.size(); i++){
			int col = (int)floor(refInlierPts[i].x / nodeLength);
			int row = (int)floor(refInlierPts[i].y / nodeWidth);

			//cout << inlierPts[i].x << "," << inlierPts[i].y << endl;
			//cout << col << "," << row << endl;
			nodes[row * nodeNumPerEdge + col].addMatchedPts(i);
		}
	}

	void addMatchedPts(Point2f & point, Point2f & refPoint){
		inlierPts.push_back(point);
		refInlierPts.push_back(refPoint);
	}

	// 显示匹配图形
	void showMatchedImg(Mat & refImg, vector<KeyPoint> & refKpt){
		Mat matchedImg;
		drawMatches(image, keypoints, refImg, refKpt, inlierMatches, matchedImg, 
				Scalar::all(-1), CV_RGB(0,255,0), Mat(), 2);
		imshow("Matched Result", matchedImg);
		waitKey(0);
	}

	Mat getImageDescriptors(){
		calImageDescriptors();
		return descriptors;
	}

	// 计算给定节点的Homography
	void calNodeHomography(int row, int col, Mat parentHomography){
		int featurePtSize = nodes[row * (1 << layer) + col].getMatchedPtsSize();
		//cout << "Feature Points Num：" << featurePtSize << endl;
		// 如果该Node特征点数量太少，则取上一层的Homography
		if(featurePtSize < 8){    
			nodes[row * (1 << layer) + col].passParentHomography(parentHomography);
		}
		else{
			nodes[row * (1 << layer) + col].calHomography(inlierPts, refInlierPts); 
		}
	}

	//计算这一层的Homography Flow
	void calHomographyFlow(){
		int edgeLen = 1 << layer;
		int nodeLength = image.cols / edgeLen;
		int nodeWidth = image.rows / edgeLen;
		homoFlow = Mat::zeros(image.rows, image.cols, CV_32FC2);
		// 计算每一个node的homography flow
		for(int r = 0; r < edgeLen; r++){
			for(int c = 0; c < edgeLen; c++){
				Mat H =	nodes[r * edgeLen + c].getHomography();
				// 算出该node范围
				int rowStart = r * nodeWidth;
				int rowEnd = (r + 1) * nodeWidth - 1;
				if(r == edgeLen - 1) rowEnd = image.rows - 1;
				int colStart = c * nodeLength;
				int colEnd = (c + 1) * nodeLength - 1;
				if(c == edgeLen - 1) colEnd = image.cols - 1;

				vector<Point2f> srcPts, dstPts; // src是refImage，dst是当前图片
				srcPts.clear();
				dstPts.clear();
				for(int row = rowStart; row <= rowEnd; row++)   // y
					for(int col = colStart; col <= colEnd; col++)  // x
						srcPts.push_back(Point2f((float)col, (float)row));
				
				perspectiveTransform(srcPts, dstPts, H);  // [x y 1]*H = [x' y' w'], src(x, y) -> dst(x'/w', y'/w')

				int idx = 0;
				for(int row = rowStart; row <= rowEnd; row++){   // y
					for(int col = colStart; col <= colEnd; col++){  // x
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
	void calHomographyFlowByScale(Mat & finestHomoFlow, int scale){
		resize(finestHomoFlow, homoFlow, image.size(), 0, 0, CV_INTER_AREA);
		homoFlow = homoFlow / (float)scale;
	}

	void calConsistImage(){
		consistImage = Mat::zeros(image.size(), CV_8UC3);
		for(int r = 0; r < image.rows; r++){     // consistent图像(r,c)
			for(int c = 0; c < image.cols; c++){
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

	Mat & getConsistImage(){
		return consistImage;
	}

	Mat & getHomoFlow(){
		return homoFlow;
	}

	Mat getNodesHomography(int row, int col){
		return nodes[row * (1 << layer) + col].getHomography();
	}

	vector<KeyPoint> & getKeypoints(){
		return keypoints;
	}

	vector<Point2f> & getRefMatchPts(){
		return refInlierPts; 
	}

	vector<Point2f> & getCurMatchPts(){
		return inlierPts;
	}
};

class Pyramid{
	vector<PyramidLayer> pyramid;  // 默认是private

public:
	Pyramid(){}    
	Pyramid(Mat & Image){     // 构造函数，对Image构造一个Image Pyramid
		int cols = Image.cols ;
		int rows = Image.rows;
		int layerNum = 1;
		while(cols > 400 || rows > 400){    // 计算这个Pyramid有几层，高斯金字塔最粗糙层image的长边不比400pixels大(文中说）
			cols = cols >> 1;
			rows = rows >> 1;
			//printf("Col Num: %d, Row Num: %d\n", cols, rows);
			layerNum++;
		}
		printf("Layer Num: %d\n", layerNum);
		
		// 计算Pyramid, 最高层是原图， pyramid[0]是最粗糙的
		pyramid.resize(layerNum);
		PyramidLayer oriLayer(Image, layerNum - 1);
		pyramid[layerNum - 1] = oriLayer;
		for(int i = layerNum - 2; i >= 0; i--){
			Mat srcImage = pyramid[i+1].getImage();
			Mat dstImage;
			// 用pyrDown进行下采样操作，执行了高斯金字塔建造的向下采样的步骤; 改变图像尺寸还可以用resize()
			pyrDown(srcImage, dstImage, Size(srcImage.cols >> 1, srcImage.rows >> 1));
			PyramidLayer newLayer(dstImage, i);
			pyramid[i] = newLayer;
		}
	}

	// 计算每一层的特征点（将coarse level的特征点scale到其他层）
	void calFeaturePyramid(){
		vector<Point2f> & featMatchPts = pyramid[FEATURE_LAYER].getCurMatchPts();
		vector<Point2f> & featRefMatchPts = pyramid[FEATURE_LAYER].getRefMatchPts();
		for(unsigned int layer = 0; layer < pyramid.size(); layer++){
			if(layer == FEATURE_LAYER) continue;

			int featureRow = 1 << FEATURE_LAYER;
			int row = 1 << layer;
			float ratio = (float)row / (float)featureRow;
			
			// 将点从FEATURE_LAYER（0层）scale到其他层
			for(unsigned int i = 0; i < featMatchPts.size(); i++){
				Point2f tempPts(featMatchPts[i].x * ratio, featMatchPts[i].y * ratio);
				Point2f tempRefPts(featRefMatchPts[i].x * ratio, featRefMatchPts[i].y * ratio);
				//cout << featMatchPts[i].x * ratio << "," << featMatchPts[i].y * ratio << endl;
				pyramid[layer].addMatchedPts(tempPts, tempRefPts);
			}
		}
	}

	// 将每一层的特征点分到每一个ImageNode
	void distributeFeaturePtsByLayer(){
		for(unsigned int layer = 0; layer < pyramid.size(); layer++){
			pyramid[layer].distributeFeaturePts();
		}
	}

	// 计算这个图像金字塔的homography金字塔
	void calHomographyPyramid(){
		for(unsigned int layer = 0; layer < pyramid.size(); layer++){
			int nodeNumPerEdge = 1 << layer;
			// 一层一层算（从最粗糙层开始）
			for(int row = 0; row < nodeNumPerEdge; row++){   // 对每一层，算每个node的homography
				for(int col = 0; col < nodeNumPerEdge; col++){
					Mat parentHomography(3, 3, CV_64F, Scalar::all(0));
					
					if(layer != 0){
						int parentRow = row >> 1;
						int parentCol = col >> 1;
						parentHomography = pyramid[layer-1].getNodesHomography(parentRow, parentCol);
					}
					pyramid[layer].calNodeHomography(row, col, parentHomography); 
					//cout << row << "," << col << endl << pyramid[layer].getNodesHomography(row, col) << endl;
				}
			}
			//cout << endl;
		}
	}

	// 计算这个图像金字塔的homography flow（其实就相当于每个像素相对于参考帧的偏移量）
	void calHomographyFlowPyramid(){
		// 计算finest level 的homography flow
		int fineLevel = pyramid.size() - 1;
		pyramid[fineLevel].calHomographyFlow();  
		Mat & homoFlow = pyramid[fineLevel].getHomoFlow();
		cout << "homoFlow size: " << homoFlow.size() << endl;
		/*Mat m(4, 4, CV_32FC2);
		for(int i = 0; i < 4; i++)
			for(int j = 0; j < 4; j++)
				m.at<Vec2f>(i,j) = homoFlow.at<Vec2f>(i,j);
		cout << m << endl;*/
		
		// 将最底层的homographyFLow等比例缩小，得到其他层的homographyFlow
		for(int i = fineLevel - 1; i >= 0; i--){
			int scale = 1 << (fineLevel - i);
			pyramid[i].calHomographyFlowByScale(homoFlow, scale);
			Mat & homof = pyramid[i].getHomoFlow();
			cout << "homof size: " << homof.size() << endl;
		}
	}

	vector<Mat> getImagePyramid(){
		vector<Mat> imagePyramid;
		for(unsigned int i = 0; i < pyramid.size(); i++){
			imagePyramid.push_back(pyramid[i].getImage());
		}
		return imagePyramid;
	}

	PyramidLayer* getPyramidLayer(int layer){
		return &(pyramid[layer]);
	}
};

class FastBurstImagesDenoising{
public:
	vector<Mat> oriImageSet;      // 存储原来的每帧图片
	vector<Pyramid*> imagePyramidSet;  // 图片金字塔（高斯金字塔）
	Pyramid* refPyramid;   // 参考图片的金字塔
	vector<Mat> integralImageSet;   // 所有Consistent Image的积分图
	vector<Mat> consistGrayImageSet; // 所有Consistent Image的灰度图

	void readBurstImages(const string fileDir){
		Mat img;
		char index[10];
		for(int i = 0; i < FRAME_NUM; i++){
			sprintf_s(index, "%d", i);
			img = imread(fileDir + string(index) + imageFormat);
			oriImageSet.push_back(img);
		}
	}

	// 计算图像金字塔
	void calPyramidSet(){
		int frameNum = oriImageSet.size();
		imagePyramidSet.resize(frameNum);
		for(int i = 0; i < frameNum; i++){
			printf("Frame %d: ", i);
			imagePyramidSet[i] = new Pyramid(oriImageSet[i]);
		}
		refPyramid = imagePyramidSet[REF];
		//showImages(refPyramid->getImagePyramid());
	}

	// 第一步： 计算每帧图片的homography flow 金字塔
	void calHomographyFlowPyramidSet(){

		// 计算refImage的特征层的特征向量和特征点
		PyramidLayer* refpLayer = refPyramid->getPyramidLayer(FEATURE_LAYER);
		Mat refDescriptor = refpLayer->getImageDescriptors();  
		vector<KeyPoint> & refKpoint = refpLayer->getKeypoints();

		// BruteForce和FlannBased是opencv二维特征点匹配常用的两种办法，BF找最佳，比较暴力，Flann快，找近似，但是uchar类型描述子（BRIEF）只能用BF
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce" ); 
		
		
		for(int frame = 0; frame < FRAME_NUM; frame++){

			if(frame == REF) continue;
			// 1. 计算每一帧->参考帧的Homography（3*3矩阵） 金字塔
			// 计算当前帧（最粗糙层）与参考帧的匹配特征点
			Pyramid* curPyramid = imagePyramidSet[frame]; // 当前图片金字塔
			PyramidLayer* curFrame = curPyramid->getPyramidLayer(FEATURE_LAYER);
			curFrame->calMatchPtsWithRef(refDescriptor, refKpoint, matcher);
	
			// 画出匹配结果
			//curFrame->showMatchedImg(curFrame->getImage(), refKpoint);

			// 计算每一层的特征点（将coarse level的特征点scale到其他层）
			curPyramid->calFeaturePyramid();

			// 将每一层的特征点分配到每个ImageNode
			curPyramid->distributeFeaturePtsByLayer();

			// 计算每一帧（除参考帧）的homography金字塔
			curPyramid->calHomographyPyramid();

			// 计算每一帧（除参考帧）的homography flow金字塔
			curPyramid->calHomographyFlowPyramid();

			cout << endl;
		}
	}

	// 第二步：选择consistent pixel 
	void consistentPixelSelection(){
		
		PyramidLayer* refpLayer = refPyramid->getPyramidLayer(CONSIST_LAYER);
		Mat refImage = refpLayer->getImage();
		Mat medianImg = refImage.clone();
		medianImg = medianImg / FRAME_NUM;
		integralImageSet.resize(FRAME_NUM);   // 所有Consistent Image的积分图
		consistGrayImageSet.resize(FRAME_NUM); // 所有Consistent Image的灰度图

		for(int frame = 0; frame < FRAME_NUM; frame++){
			if(frame == REF){
				cvtColor(refImage, consistGrayImageSet[frame], CV_RGB2GRAY);
				integral(consistGrayImageSet[frame], integralImageSet[frame], CV_32S);
				// integral(refImage, integralImageSet[frame], CV_32SC3);  如果不转灰度就算积分图，用这个
				continue;
			}

			// 将图片用homography flow调整成一个consistent image(即和参考帧一致)(CONSIST_LAYER)
			Pyramid* curPyramid = imagePyramidSet[frame]; // 当前图片金字塔
			PyramidLayer* curFrame = curPyramid->getPyramidLayer(CONSIST_LAYER);
			curFrame->calConsistImage();

			// 求median图像
			Mat & consistImg = curFrame->getConsistImage();
			medianImg = medianImg + consistImg / FRAME_NUM;
			
			// 转灰度图
			cvtColor(consistImg, consistGrayImageSet[frame], CV_RGB2GRAY);

			// 求所有consistent 灰度图的积分图(原图行列各加1，第一行第一列均为0）
			integral(consistGrayImageSet[frame], integralImageSet[frame], CV_32S);
			//integral(consistImage, integralImageSet[frame], CV_32SC3);  // 如果不转灰度就算积分图，用这个
		}

		// 求median图的灰度图和积分图
		Mat grayMedianImg, integralMedianImg;
		cvtColor(medianImg, grayMedianImg, CV_RGB2GRAY);
		integral(grayMedianImg, integralMedianImg, CV_32S);


		

	}

	void showImages(vector<Mat> Images){
		int imageNum = Images.size();
		cout << "Image Num: " << imageNum << endl;
		cout << "Image Size(cols(x) * rows(y)): " << Images[0].size() << endl;  
		char index[10];
		for(int i = 0; i < imageNum; i++){
			sprintf_s(index, "%d", i);
			imshow(index, Images[i]);
			waitKey(0);  // waitKey(1000),等待1000 ms后窗口自动关闭; waitKey(0)等待按键
		}
	}

};

FastBurstImagesDenoising FBID;

int main(){
	FBID.readBurstImages(fileDir);
	//FBID.showImages(FBID.oriImageSet); 
	FBID.calPyramidSet();
	FBID.calHomographyFlowPyramidSet();
	FBID.consistentPixelSelection();

	//Mat m(Size(3,3), CV_32FC2 , Scalar::all(0));
	//Vec2f& elem = m.at<Vec2f>( 1 , 2 );// or m.at<Vec2f>( Point(col,row) );
	//elem[0]=1212.0f;
	//elem[1]=326.0f;
	//cout << m << endl;

	system("pause");

	return 0;
}