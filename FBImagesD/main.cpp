#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
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
const int MAXIDX = 500; // 找连通分量时，并查集最大下标值

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

	vector<DMatch> matches; // 匹配结果
	vector<DMatch> goodMatches;
	vector<uchar> RANSACStatus;   // 这个变量用于存储RANSAC后每个点的状态,值为0（错误匹配,野点）,1 
	vector<Point2f> srcPts, dstPts; // src是refImage，dst是当前图片
 

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
		matches.clear();
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

		goodMatches.clear();
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
		RANSACStatus.clear(); // 这个变量用于存储RANSAC后每个点的状态,值为0（错误匹配,野点）,1 
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

				srcPts.clear();  // src是refImage，dst是当前图片
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

	Mat getConsistImage(){
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

class ConnectedComponent{
	vector<Point> pts;
	int idx;

public:
	ConnectedComponent(){}
	ConnectedComponent(int _idx){
		idx = _idx;
	}

	void addPoint(Point p){
		pts.push_back(p);
	}

	int getIdx(){
		return idx;
	}

	vector<Point> & getCCPts(){
		return pts;
	}
};

class ConsistentPixelSetPyramid{
	vector<Mat> consistentPixels;  // 这个相当于每层一个

	
	vector<Mat> channels;			// 用于形态学过滤时，分离各帧图像
	/*// 并查集操作
	uchar find(uchar x, uchar parent[]){
		uchar r = x;
		while(parent[r] != r) r = parent[r];
		return r;
	}

	void join(uchar s, uchar l, uchar parent[]){
		uchar x = find(s, parent);
		uchar y = find(l, parent);
		if(x < y) parent[y] = x;
		else parent[x] = y;
	}*/

public:
	ConsistentPixelSetPyramid(){}
	ConsistentPixelSetPyramid(int layerNum){
		consistentPixels.resize(layerNum);
	}

	// 形态学过滤器，周围3*3内值为 1 的数量 <5 时返回0
	int checkMajority(const Mat& src, const int row, const int col) {
		int cnt = 0;
		int startR = row - 1, startC = col - 1, endR = row + 1, endC = col + 1;
		for (int r = startR; r <= endR; r++) {
			for (int c = startC; c <= endC; c++) {
				if (r == row && c == col) continue;
				if (r < 0 || r >= src.rows || c < 0 || c >= src.cols) continue;
				if (src.at<uchar>(r, c) == 1) cnt++;
			}
		}
		if(cnt < 5) return 0;					
		return 1;
	}

	// 形态学filter
	void morphMajority(const Mat & src, Mat & dst){
		Mat temp = src.clone();
		for (int row = 0; row < dst.rows; row++) {
			for (int col = 0; col < dest.cols; col++) {
				dst.at<uchar>(row, col) = checkMajority(temp, row, col);
			}
		}
	}

	// 使用BFS实现Flood-Fill算法
	ConnectedComponent floodFill(int idx, int x0, int y0, Mat & undecidedPixels){
 		int dx[] = { -1, 0, 1, 0 };
 		int dy[] = { 0, 1, 0, -1 };
 		int x = x0, y = y0;
 		queue<Point> Q;										// 用于BFS, floodFill的实现
 		Point cur;
 		ConnectedComponent ans(idx);
		ans.addPoint(Point(x0, y0));

		while(!Q.empty()) Q.pop();
		Q.push(Point(x0, y0));

		// BFS,以当前点为中心，向四周搜索可以扩展的点（值为 0 即为可扩展）
		while(!Q.empty()){
			cur = Q.front(); Q.pop();
			for (int i = 0; i < 4; i++){
				x = cur.x + dx[i]; 
				y = cur.y + dy[i];
				// 判断越界和是否可扩展
				if (x < 0 || x >= undecidedPixels.cols || y < 0 || y >= undecidedPixels.rows)
					continue;								
				if (undecidedPixels.at<uchar>(y, x) != 255) continue;

				undecidedPixels.at<uchar>(y, x) = 127;		// 将此点标记为已确定
				ans.addPoint(Point(x, y));					// 加入到联通块中
				Q.push(Point(x, y));						// 加入到队列中，以便从此点开始扩展
			}
		}
		return ans;											// 返回当前连通分量
	}

	// 寻找连通分量(Flood-Fill算法）	
	void findConnectedComponents(vector<ConnectedComponent> & connComps, Mat & undecidedPixels){
		int idx = 0;
		for (int y = 0; y < undecidedPixels.rows; y++)
			for (int x = 0; x < undecidedPixels.cols; x++)
			if (undecidedPixels.at<uchar>(y, x) == 255){
				connComps.push_back(floodFill(++idx, x, y, undecidedPixels));
			}
	}

	// 寻找连通分量(Two-Pass算法）			
	/*void findConnectedComponents(vector<ConnectedComponent> & connComps, Mat & undecidedPixels){
		int idx = 1;
		Mat labeled(undecidedPixels.size(), CV_8U, Scalar::all(0)); 

		uchar parent[MAXIDX] = {0};   // 并查集操作
		for(int i = 0; i < MAXIDX; i++) parent[i] = i;
		bool isInConnComps[MAXIDX] = {false};

		// first pass
		for(int y = 0; y < undecidedPixels.rows; y++){
			for(int x = 0; x < undecidedPixels.cols; x++){
				if(undecidedPixels.at<uchar>(y, x) == 1){
					uchar left = x - 1 < 0 ? 0 : labeled.at<uchar>(y, x - 1);
					uchar up = y - 1 < 0 ? 0 : labeled.at<uchar>(y - 1, x);
					if(left == 0 && up == 0) labeled.at<uchar>(y, x) = ++idx;  
					else{
						if(left != 0 && up != 0){
							labeled.at<uchar>(y, x) = min(left, up);
							join(min(left, up), max(left, up), parent);
						}
						else labeled.at<uchar>(y, x) = max(left, up);
					}
				}
			}
		}

		// second pass
		for(int y = 0; y < undecidedPixels.rows; y++){
			for(int x = 0; x < undecidedPixels.cols; x++){
				if(labeled.at<uchar>(y, x) != 0){
					uchar idx = find(labeled.at<uchar>(y, x), parent);
					labeled.at<uchar>(y, x) = idx;
					if(isInConnComps[idx] == false){
						ConnectedComponent cc(idx);
						cc.addPoint(Point(x, y));
						isInConnComps[idx] = true;
						connComps.push_back(cc);
					}
					else{
						for(unsigned int i = 0; i < connComps.size(); i++){
							if(connComps[i].getIdx() == idx){
								connComps[idx].addPoint(Point(x, y));
								break;
							}
						}
					}
				}
			}
		}

		// 两遍之后就得到了所有的Connected Component
		cout << connComps.size() << endl;
	}*/
	
	// 计算layer层的consistent pixels set
	void calConsistentPixelSet(int layer, vector<Mat> & integralImageSet, Mat & integralMedianImg, const int thre){
		Mat refConsistPixelSet, medConsistPixelSet, consistPixelSet;   // 记录consistent pixel
		Mat reliablePixelSet; // 记录reliable pixels

		// 初始化consistent pixel set
		refConsistPixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8UC(FRAME_NUM));
		medConsistPixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8UC(FRAME_NUM));
		consistPixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8UC(FRAME_NUM));
		reliablePixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8U);
		Mat undecidedPixels(consistPixelSet.rows, consistPixelSet.cols, CV_8U, Scalar::all(0));
		
		// 计算reference-based 和 median-based consistent pixels
		for(int r = 0; r < integralMedianImg.rows - 1; r++){   // 积分图比普通图行列各多一列
			for(int c = 0; c < integralMedianImg.cols - 1; c++){

				// 算参数
				int half = 2;
				int startR = max(0, r - half);
				int endR = min(integralMedianImg.rows - 2, r + half);
				int startC = max(0, c - half);
				int endC = min(integralMedianImg.cols - 2, c + half);
				int pixelNum = (endR - startR + 1) * (endC - startC + 1);

				/* -----计算reference-based consistent pixels----- */
				// 先计算ref image的5*5块像素
				int pixelRefSum = integralImageSet[REF].at<int>(endR+1, endC+1) - integralImageSet[REF].at<int>(startR, endC+1)
					- integralImageSet[REF].at<int>(endR+1, startC) + integralImageSet[REF].at<int>(startR, startC);
				int aveRefPixel = pixelRefSum / pixelNum;

				// 先把ref Image的该点标记为consistent pixel
				Vec<uchar, FRAME_NUM> & elem = refConsistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c);
				elem[REF] = 1;
				
				// 从ref开始往0计算每一帧
				for(int i = REF - 1; i >= 0; i--){
					int pixelSum = integralImageSet[i].at<int>(endR+1, endC+1) - integralImageSet[i].at<int>(startR, endC+1)
						- integralImageSet[i].at<int>(endR+1, startC) + integralImageSet[i].at<int>(startR, startC);
					int avePixel = pixelSum / pixelNum;
					if(abs(aveRefPixel - avePixel) < thre)
						elem[i] = 1;
					else
						break;
				}

				// 从ref开始往右计算每一帧
				for(int i = REF + 1; i < FRAME_NUM; i++){
					int pixelSum = integralImageSet[i].at<int>(endR+1, endC+1) - integralImageSet[i].at<int>(startR, endC+1)
						- integralImageSet[i].at<int>(endR+1, startC) + integralImageSet[i].at<int>(startR, startC);
					int avePixel = pixelSum / pixelNum;
					if(abs(aveRefPixel - avePixel) < thre)
						elem[i] = 1;
					else
						break;
				}

				/* -----计算median-based consistent pixels----- */
				// 计算median image的5*5块像素
				int pixelMedSum =integralMedianImg.at<int>(endR+1, endC+1) - integralMedianImg.at<int>(startR, endC+1)
					- integralMedianImg.at<int>(endR+1, startC) + integralMedianImg.at<int>(startR, startC);
				int aveMedPixel = pixelMedSum / pixelNum;

				Vec<uchar, FRAME_NUM> & medElem = medConsistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c);
				int cnt = 0; // 记录有多少个median-based consistent pixels
				for(int i = 0; i < FRAME_NUM; i++){
					int pixelSum = integralImageSet[i].at<int>(endR+1, endC+1) - integralImageSet[i].at<int>(startR, endC+1)
						- integralImageSet[i].at<int>(endR+1, startC) + integralImageSet[i].at<int>(startR, startC);
					int avePixel = pixelSum / pixelNum;
					if(abs(aveMedPixel - avePixel) < thre){
						medElem[i] = 1;
						cnt++;
					}	
				}

				/* -----结合reference-based 和 median-based 的结果----- */
				// 如果ref frame属于median-based consistent pixels, 那么取M和R的并集				
				Vec<uchar, FRAME_NUM> & finalElem = consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c);
				if(medElem[REF] == 1){
					for(int i = 0; i < FRAME_NUM; i++){
						finalElem[i] = elem[i] | medElem[i];
					}
				}
				else undecidedPixels.at<uchar>(r, c) = 255;		

				// 统计reliable pixels
				if(cnt > REF) reliablePixelSet.at<uchar>(r, c) = 1;
			}
		}

		// 显示undecidedPixels
		/*namedWindow("src", WINDOW_NORMAL);
 		imshow("src", undecidedPixels);
 		waitKey(0);*/

		// 否则找出那些undecided pixels的联通分量
		vector<ConnectedComponent> connComps;
		findConnectedComponents(connComps, undecidedPixels);

		/*// 此时，原来255的部分都被置为127
		namedWindow("AfterProcess", WINDOW_NORMAL);
		imshow("AfterProcess", undecidedPixels);
		waitKey(0);

		// 把找到的联通块画到图上
		for (int i = 0; i < connComps.size(); i++){
			uchar color = (i % 2 == 0) ? 100 : 200;						// 随机给一个颜色
			vector<Point> & now = connComps[i].getCCPts();
			for (int k = 0; k < now.size(); k++){
				undecidedPixels.at<uchar>(now[k]) = color;
			}
		}
		namedWindow("Component", WINDOW_NORMAL);
 		imshow("Component", undecidedPixels);
		waitKey(0);*/

		// 统计每一个连通分量是reliable pixels多还是unreliable多（majority voting 多数同意），来做出不同的combine策略
		for(unsigned int i = 0; i < connComps.size(); i++){
			vector<Point> & CCpts = connComps[i].getCCPts();
			int cnt = 0; // 统计连通分量中有多少个reliable pixel
			
			for(unsigned int j = 0; j < CCpts.size(); j++){
				if(reliablePixelSet.at<uchar>(CCpts[j]) == 1) cnt++;
			}

			// 如果reliable pixel多，则整个连通分量都当作reliable处理，取M的结果
			if(cnt >= CCpts.size() - cnt){ 
				for(unsigned int j = 0; j < CCpts.size(); j++){
					Vec<uchar, FRAME_NUM> & finalElem = consistPixelSet.at<Vec<uchar, FRAME_NUM> >(CCpts[j]);
					Vec<uchar, FRAME_NUM> & medElem = medConsistPixelSet.at<Vec<uchar, FRAME_NUM> >(CCpts[j]);
					finalElem = medElem;
				}
			}
			// 否则整个CC都当作unreliable处理，取R的结果
			else{
				for(unsigned int j = 0; j < CCpts.size(); j++){
					Vec<uchar, FRAME_NUM> & finalElem = consistPixelSet.at<Vec<uchar, FRAME_NUM> >(CCpts[j]);
					Vec<uchar, FRAME_NUM> & refElem = refConsistPixelSet.at<Vec<uchar, FRAME_NUM> >(CCpts[j]);
					finalElem = refElem;
				}
			}
		}

		// 形态学过滤，morphological(majority) filter
		for (int k = 0; k < FRAME_NUM; k++){
			int cnt = 0;
			for (int i = 0; i < consistPixelSet.rows; i++)
				for (int j = 0; j < consistPixelSet.cols; j++)
					cnt += consistPixelSet.at<Vec<uchar, FRAME_NUM> >(Point(j, i))[k];
			printf("Before: Frame: %d    cnt = %d\n", k, cnt);
		}

		channels.clear();
		split(consistPixelSet, channels);

		for (int i = 0; i < channels.size(); i++){
			morphMajority(channels[i], channels[i]);
		}
		merge(channels, consistPixelSet);

		for (int k = 0; k < FRAME_NUM; k++){
			int cnt = 0;
			for (int i = 0; i < consistPixelSet.rows; i++)
			for (int j = 0; j < consistPixelSet.cols; j++)
				cnt += consistPixelSet.at<Vec<uchar, FRAME_NUM> >(Point(j, i))[k];
			printf("After: Frame: %d    cnt = %d\n", k, cnt);
		}


		consistentPixels[layer] = consistPixelSet;
	}

	// 由上下采样得到所有层的consistent pixels
	void calConsistentPixelsAllLayer(vector<Mat> & refPyramid){
		for(unsigned int layer = 0; layer < consistentPixels.size(); layer++){
			if(layer == CONSIST_LAYER) continue;
			resize(consistentPixels[CONSIST_LAYER], consistentPixels[layer], refPyramid[layer].size(), 0, 0, CV_INTER_LINEAR);  // CV_NEAREST
		}
	}

	vector<Mat> & getConsistentPixelPyramid(){
		return consistentPixels;
	}
};

class FastBurstImagesDenoising{
public: 
	vector<Mat> oriImageSet;                       // 存储原来的每帧图片
	vector<Pyramid*> imagePyramidSet;              // 图片金字塔（高斯金字塔）
	Pyramid* refPyramid;                           // 参考图片的金字塔
	ConsistentPixelSetPyramid consistPixelPyramid; // 存储consistent pixel
	Mat grayMedianImg;                             // CONSIST_LAYER的中位图（灰度图） 
	vector<Mat> temporalResult;                    // temporal fusion的结果图

	Mat refConsistPixelSet, medConsistPixelSet, consistPixelSet;   // 记录consistent pixel

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
		vector<Mat> integralImageSet;    // 所有Consistent Image的积分图
		vector<Mat> consistGrayImageSet; // 所有Consistent Image的灰度图
		const int threshold = 10;        // 阈值

		// 取出refImage
		PyramidLayer* refpLayer = refPyramid->getPyramidLayer(CONSIST_LAYER);
		Mat refImage = refpLayer->getImage();

		integralImageSet.resize(FRAME_NUM);    // 所有Consistent Image的积分图
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
			
			// 转灰度图
			Mat consistImg = curFrame->getConsistImage();
			cvtColor(consistImg, consistGrayImageSet[frame], CV_RGB2GRAY);

			// 求所有consistent 灰度图的积分图(原图行列各加1，第一行第一列均为0）
			integral(consistGrayImageSet[frame], integralImageSet[frame], CV_32S);
			//integral(consistImage, integralImageSet[frame], CV_32SC3);  // 如果不转灰度就算积分图，用这个
		}

		// 求median图(灰度图)和积分图
		grayMedianImg = Mat::zeros(refImage.rows, refImage.cols, CV_8U);
		for(int r = 0; r < grayMedianImg.rows; r++)
			for(int c = 0; c < grayMedianImg.cols; c++){
				vector<uchar> pixelValues;
				for(int f = 0; f < FRAME_NUM; f++){
					pixelValues.push_back(consistGrayImageSet[f].at<uchar>(r, c));
				}
				nth_element(pixelValues.begin(), pixelValues.begin() + REF, pixelValues.end());
				grayMedianImg.at<uchar>(r, c) = pixelValues[REF];
			}
		//imshow("median", grayMedianImg);
		//waitKey(0);
		Mat integralMedianImg;
		integral(grayMedianImg, integralMedianImg, CV_32S);

		// 初始化Consistent pixel pyramid
		consistPixelPyramid = ConsistentPixelSetPyramid(refPyramid->getImagePyramid().size());

		// 算Consistent Pixels(CONSIST_LAYER的)
		consistPixelPyramid.calConsistentPixelSet(CONSIST_LAYER, integralImageSet, integralMedianImg, threshold);

		// reuse the indices of computed consistent pixels by upsampling and downsampling，把refPyramid传进去是为了知道每层的尺寸
		consistPixelPyramid.calConsistentPixelsAllLayer(refPyramid->getImagePyramid());
	}

	

	// 第三步：融合得到最后的去噪图像
	void pixelsFusion(){

		vector<Mat> refImagePyramid = refPyramid->getImagePyramid();
		int layersNum = refImagePyramid.size();

		/* -----计算噪声方差----- */
		// 取出ref image并转成灰度图像
		Mat refGrayImg;
		cvtColor(refImagePyramid[layersNum - 1], refGrayImg, CV_RGB2GRAY);

		// 取得中位图（灰度）
		Mat medGrayImg;  
		resize(grayMedianImg, medGrayImg, refGrayImg.size(), 0, 0, CV_INTER_LINEAR);  // grayMedianImg是CONSIST_LAYER的
		
		// 边缘提取
		Mat edgeImg;
		Canny(grayMedianImg, edgeImg, 50, 125, 3);   // canny边缘检测采用双阈值值法，高阈值用来检测图像中重要的、显著的线条、轮廓等，而低阈值用来保证不丢失细节部分
		resize(edgeImg, edgeImg, refGrayImg.size(), 0, 0, CV_INTER_NN);
		//imshow("edge", edgeImg);
		//waitKey(0);

		// 求平坦区域和平坦区域的像素数
		double cnt = 0;
		for(int r = 0; r < refGrayImg.rows; r++)
			for(int c = 0; c < refGrayImg.cols; c++){
				edgeImg.at<uchar>(r, c) = (edgeImg.at<uchar>(r, c) == 0);
				if(edgeImg.at<uchar>(r, c) == 1) cnt++;
			}
		refGrayImg = refGrayImg.mul(edgeImg);
		refGrayImg.convertTo(refGrayImg, CV_64F);
		medGrayImg = medGrayImg.mul(edgeImg);
		medGrayImg.convertTo(medGrayImg, CV_64F);

		// 求med和ref之差的均值和方差（平坦区域）
		Mat diffImg = medGrayImg - refGrayImg;
		double aveNum = sum(diffImg)[0]/ cnt;
		for(int r = 0; r < refGrayImg.rows; r++)
			for(int c = 0; c < refGrayImg.cols; c++){
				if(edgeImg.at<uchar>(r, c) == 1) diffImg.at<double>(r, c) -= aveNum;
				
			}
		diffImg = diffImg.mul(diffImg);
		double noiseVar = sum(diffImg)[0] / cnt;  // sigma2
		

		/* -----进行temporal fusion----- */
		vector<Mat> & consistentPixelPyramid = consistPixelPyramid.getConsistentPixelPyramid();
		for(int layer = 0; layer < layersNum; layer++){
			
			// 取出这一层的consistent pixels集
			consistPixelSet = consistentPixelPyramid[layer];

			vector<Mat> consistImgSet;
			Mat consistGrayImgSet(consistPixelSet.size(), CV_8UC(FRAME_NUM), Scalar::all(0));
			Mat meanImg(consistPixelSet.size(), CV_8U, Scalar::all(0));
			Mat meanRGBImg(consistPixelSet.size(), CV_8UC3, Scalar::all(0));
			Mat tVar(consistPixelSet.size(), CV_64F, Scalar::all(0));
			Mat var(consistPixelSet.size(), CV_64F, Scalar::all(0));

			// 得到consist gray image set (col * row * 10) 和 consistImgSet(vector, col * row * 3 * 10) 彩色
			if(layer != CONSIST_LAYER)
				for(int frame = 0; frame < FRAME_NUM; frame++){
					PyramidLayer* curFrame = imagePyramidSet[frame]->getPyramidLayer(layer);
					Mat consistImg;
					if(frame == REF){
						consistImg = curFrame->getImage();
					}
					else{
						curFrame->calConsistImage();
						consistImg = curFrame->getConsistImage();
					}
					consistImgSet.push_back(consistImg);
					Mat consistGrayImg;
					cvtColor(consistImg, consistGrayImg, CV_RGB2GRAY);
					for(int r = 0; r < consistImg.rows; r++)
						for(int c = 0; c < consistImg.cols; c++){
							consistGrayImgSet.at<Vec<uchar, FRAME_NUM>>(r, c)[frame] = consistGrayImg.at<uchar>(r, c);
						}
				}
			
			// 点乘，将不是consistent pixel的地方置零
			consistGrayImgSet = consistGrayImgSet.mul(consistPixelSet);

			
			for(int r = 0; r < consistPixelSet.rows; r++)
				for(int c = 0; c < consistPixelSet.cols; c++){
					// 计算平均图像（灰度）
					meanImg.at<uchar>(r, c) = sum(consistGrayImgSet.at<Vec<uchar, FRAME_NUM>>(r, c))[0] 
						/ sum(consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c))[0];

					// 计算sigmat方，即每个像素点consistent pixels的方差
					Vec<uchar, FRAME_NUM> elem = consistGrayImgSet.at<Vec<uchar, FRAME_NUM>>(r, c);
					Vec<uchar, FRAME_NUM> & consistVector = consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c);
					
					for(int i = 0; i < FRAME_NUM; i++){
						if(consistVector[i] > 0)  elem[i] = abs(elem[i] - meanImg.at<uchar>(r, c));
					}
					elem = elem.mul(elem);
					tVar.at<double>(r, c) = (double)sum(elem)[0] / (double)sum(consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c))[0]; 

					// 计算sigmac方
					tVar.at<double>(r, c) = max((double)0, tVar.at<double>(r, c) - noiseVar);
					
					// 计算sigmac2/(sigmac2 + sigma2)
					var.at<double>(r, c) = tVar.at<double>(r, c) / (tVar.at<double>(r, c) + noiseVar);
				}

			// 得到row*col*3 * 10 的consistPixelsChannels（将每帧的consist map分开，并变成3通道（rgb））
			vector<Mat> tempConsistPixelsChannels;             // row*col*1 * 10
			vector<Mat> consistPixelsChannels;                 // row*col*3 * 10
			Mat sumImg(consistPixelSet.size(), CV_32SC3, Scalar::all(0));
			Mat sumConsistImg(consistPixelSet.size(), CV_8UC3, Scalar::all(0));
			split(consistPixelSet, tempConsistPixelsChannels); // 将consistPixelSet 10个通道分开
			
			for(int frame = 0; frame < FRAME_NUM; frame++){
				vector<Mat> tempChannels;                      // row*col*1 * 3
				for(int i = 0; i < 3; i++) tempChannels.push_back(tempConsistPixelsChannels[frame]);
				merge(tempChannels, consistPixelsChannels[frame]);

				// 求sumImg
				consistImgSet[frame] = consistImgSet[frame].mul(consistPixelsChannels[frame]);
				sumImg += consistImgSet[frame];
				sumConsistImg += consistPixelsChannels[frame];
			}

			// 求meanRGBImg
			meanRGBImg = sumImg / sumConsistImg;   // 点除

			// 求融合后的结果
			Mat tempResult(meanRGBImg.size(), CV_32FC3);
			Mat refTempImage = refPyramid->getPyramidLayer(layer)->getImage();
			refTempImage.convertTo(refTempImage, CV_32FC3);
			meanRGBImg.convertTo(meanRGBImg, CV_32FC3);
			for(int r = 0; r < meanRGBImg.rows; r++)
				for(int c = 0; c < meanRGBImg.cols; c++){
					tempResult.at<Vec3f>(r, c) = meanRGBImg.at<Vec3f>(r, c) - 
						var.at<double>(r, c) * (refTempImage.at<Vec3f>(r, c) - meanRGBImg.at<Vec3f>(r, c));
				}
			tempResult.convertTo(tempResult, CV_8UC3);
			temporalResult.push_back(tempResult);
		}

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
	FBID.pixelsFusion();
	FBID.showImages(FBID.temporalResult);

	//Mat m(Size(3,3), CV_32FC2 , Scalar::all(0));
	//Vec2f& elem = m.at<Vec2f>( 1 , 2 );// or m.at<Vec2f>( Point(col,row) );
	//elem[0]=1212.0f;
	//elem[1]=326.0f;
	//cout << m << endl;
	/*int sz[] = {3, 3};
	Mat m(2, sz, CV_8UC3, Scalar::all(2));
	m.convertTo(m, CV_32FC3);
	Mat p(2, sz, CV_8UC3, Scalar::all(6));
	p.convertTo(p, CV_32FC3);
	p.at<Vec3f>(0,0)[0] = 1.534;
	cout << p << endl;
	p.convertTo(p, CV_8UC3);
	cout << p << endl;*/
	//cout << 2*(m.at<Vec3f>(1,1)-p.at<Vec3f>(1,1)) << endl; 
	//cout << m << endl;
	//cout << m.at<>(0, 0) << endl;
	//cout << m.mul(p) << endl;
	/*int idx = 1;
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			m.at<Vec2f>(i, j)[0] = idx++;
			m.at<Vec2f>(i, j)[1] = idx++;
		}
	}
	Mat a(Size(3,3), CV_32FC2 , Scalar::all(0));
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			a.at<Vec2f>(i, j)[0] = idx++;
			a.at<Vec2f>(i, j)[1] = idx++;
		}
	}
	Vec2f elem = a.at<Vec2f>(2, 2);
	elem[0] = 1;

	cout << a << endl << m << endl << 	elem << endl;*/
	/*Mat tmp_m, tmp_sd;
	double a, sd;
	meanStdDev(m, tmp_m, tmp_sd);
	a = tmp_m.at<double>(0,0);  
    sd = tmp_sd.at<double>(0,0);  
	cout << "Mean: " << tmp_m << " , StdDev: " << sd << endl;  
	Mat b(Size(3,3), CV_8U , Scalar::all(0));
	
	for(int i = 0; i < 3; i++){
		for(int j = 0; j < 3; j++){
			b.at<uchar>(i, j) = idx;
			idx++;
		}
	}
	cout << m << endl;
	cout << b << endl;
	Mat c(Size(3,3), CV_64F, Scalar::all(0));
	m.convertTo(m,CV_64F);
	b.convertTo(b, CV_64F);
	c =  m.mul(m)  ;
	cout << c << endl;*/
	

	system("pause");

	return 0;
}