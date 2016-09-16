#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <stdio.h>
#include <opencv2\core\core.hpp>     // ��Mat��������
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <math.h>
#define M_PI 3.141592653589793238
using namespace std;
using namespace cv;

const string fileDir  = "SrcData\\Data\\Bookshelf_2\\";
const string resultDir = "Result\\";
const string imageFormat = ".jpg";
const int FRAME_NUM = 10;
const int REF = FRAME_NUM / 2;
const int FEATURE_LAYER = 0;   // Ӧ�ô���һ�㿪ʼ������������starting from a global homography at the coarsest level
const int CONSIST_LAYER = 0;
//const int MAXIDX = 500; // ����ͨ����ʱ�����鼯����±�ֵ

class ImageNode{
	vector<int> matchedPts;
	Mat H; // ���Ӧ�Ĳο�֡��ImageNode��homography����3*3��

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
		H = findHomography(refPoints, points, CV_RANSAC); // �Ӳο�֡->��ǰ֡
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
	Mat image; // ��һ���ͼƬ
	vector<KeyPoint> keypoints;  // ��ͼƬ��������
	Mat descriptors; // ��ͼƬ����������  
	vector<Point2f> inlierPts; // ��refImageƥ��ɹ���������
	vector<Point2f> refInlierPts; 
	vector<DMatch> inlierMatches; // ƥ�䣬queryIdx, trainIdx��Ȼ��Ӧ���ʼ�����keypoints��refKpoint������
	int layer;  // �ڼ��㣨0����coarsest��
	vector<ImageNode> nodes;
	Mat homoFlow;  // homography flow ����
	Mat consistImage;

	vector<DMatch> matches; // ƥ����
	vector<DMatch> goodMatches;
	vector<uchar> RANSACStatus;   // ����������ڴ洢RANSAC��ÿ�����״̬,ֵΪ0������ƥ��,Ұ�㣩,1 
	vector<Point2f> srcPts, dstPts; // src��refImage��dst�ǵ�ǰͼƬ
 

public:
	PyramidLayer(){}
	PyramidLayer(Mat & newImage, int _layer){
		image = newImage;
		layer = _layer;
	}

	Mat getImage(){
		return image;
	}

	// �����һ���ͼƬ��������(ʹ��SURF���ӣ�
	void calKeyPoints(){
		int hessianThreshold = 400; // Hessian ��������ʽ��Ӧֵ����ֵ, hҪ���ڸ�ֵ
		SurfFeatureDetector surfDetector(hessianThreshold); // ����һ��SurfFeatureDetector��SURF, SurfDescriptorExtractor�� ������������
		surfDetector.detect(image, keypoints);
		cout << "Keypoints Num: " << keypoints.size() << endl;
		/* Mat imgKeypoints;   // ���������ͼ,���¼�����ʾ��ͼ
		drawKeypoints(image, keypoints, imgKeypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
		imshow("surfImage", imgKeypoints);
		waitKey(0); */
	}

	// ʹ��BRIEF������ȡ����������������������, descriptor; BRIEF: ��ʡ�ռ䣬��
	void calImageDescriptors(){
		calKeyPoints();
		BriefDescriptorExtractor briefExtractor;
		briefExtractor.compute(image, keypoints, descriptors);
		cout << "Descriptors Size: " << descriptors.size() << endl;
	}

	// ����ƥ���������
	void calMatchPtsWithRef(Mat & refDescriptor, vector<KeyPoint> & refKpoint, Ptr<DescriptorMatcher> matcher){
		calImageDescriptors();  // ���㵱֡������������������

		// ������������ƥ��
		matches.clear();
		matcher->match(descriptors, refDescriptor, matches); // queryDescriptor, trainDescriptor
		cout << "Matches Num: " << matches.size() << endl; // �����size��queryDescriptor������һ��,Ϊquery��ÿһ������������һ��ƥ������
		
		// ���ݾ��룬ѡ�����еĽ��ŵ�ƥ���
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

		// �ֱ�ȡ������ͼ����ƥ���������
		int matchedNum = (int)goodMatches.size();
		vector<Point2f> refMatchPts, curMatchPts;
		for(int i = 0; i < matchedNum; i++){
			refMatchPts.push_back(refKpoint[goodMatches[i].trainIdx].pt);
			curMatchPts.push_back(keypoints[goodMatches[i].queryIdx].pt);
		}

		// �����������F(��RANSAC����)����ʾ����ĳ������򳡾��������ڲ�ͬ��������Ƭ��Ӧ������ͼ������Ĺ�ϵ��x'��ת�ó���F���ٳ���x�Ľ��Ϊ0
		// RANSACΪRANdom Sample Consensus����д�����Ǹ���һ������쳣���ݵ��������ݼ�����������ݵ���ѧģ�Ͳ������õ���Ч�������ݵ��㷨
		Mat fundMat;
		RANSACStatus.clear(); // ����������ڴ洢RANSAC��ÿ�����״̬,ֵΪ0������ƥ��,Ұ�㣩,1 
		findFundamentalMat(curMatchPts, refMatchPts, RANSACStatus, FM_RANSAC);
			
		// ʹ��RANSAC������������������Եõ�һ��status����������ɾ�������ƥ�䣨֮ǰ�Ѿ�ɸ��һ���ˣ����Բ��Ǻ���Ч����
		for(int i = 0; i < matchedNum; i++){
			if(RANSACStatus[i] != 0){ 
				refInlierPts.push_back(refMatchPts[i]);
				inlierPts.push_back(curMatchPts[i]);
				inlierMatches.push_back(goodMatches[i]);  // ���inlierMatches��queryIdx, trainIdx��Ȼ��Ӧ���ʼ�����keypoints��refKpoint������
			}
		}
		cout << "Matches Num After RANSAC: " << inlierMatches.size() << endl;
		cout << endl;
	}

	// ����ͼ��matched keypoints���䵽����ImageNode
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

	// ��ʾƥ��ͼ��
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

	// ��������ڵ��Homography
	void calNodeHomography(int row, int col, Mat parentHomography){
		int featurePtSize = nodes[row * (1 << layer) + col].getMatchedPtsSize();
		//cout << "Feature Points Num��" << featurePtSize << endl;
		// �����Node����������̫�٣���ȡ��һ���Homography
		if(featurePtSize < 8){    
			nodes[row * (1 << layer) + col].passParentHomography(parentHomography);
		}
		else{
			nodes[row * (1 << layer) + col].calHomography(inlierPts, refInlierPts); 
		}
	}

	//������һ���Homography Flow
	void calHomographyFlow(){
		int edgeLen = 1 << layer;
		int nodeLength = image.cols / edgeLen;
		int nodeWidth = image.rows / edgeLen;
		homoFlow = Mat::zeros(image.rows, image.cols, CV_32FC2);
		// ����ÿһ��node��homography flow
		for(int r = 0; r < edgeLen; r++){
			for(int c = 0; c < edgeLen; c++){
				Mat H =	nodes[r * edgeLen + c].getHomography();
				// �����node��Χ
				int rowStart = r * nodeWidth;
				int rowEnd = (r + 1) * nodeWidth - 1;
				if(r == edgeLen - 1) rowEnd = image.rows - 1;
				int colStart = c * nodeLength;
				int colEnd = (c + 1) * nodeLength - 1;
				if(c == edgeLen - 1) colEnd = image.cols - 1;

				srcPts.clear();  // src��refImage��dst�ǵ�ǰͼƬ
				dstPts.clear();
				for(int row = rowStart; row <= rowEnd; row++)   // y
					for(int col = colStart; col <= colEnd; col++)  // x
						srcPts.push_back(Point2f((float)col, (float)row));
				
				perspectiveTransform(srcPts, dstPts, H);  // [x y 1]*H = [x' y' w'], src(x, y) -> dst(x'/w', y'/w')

				int idx = 0;
				for(int row = rowStart; row <= rowEnd; row++){   // y
					for(int col = colStart; col <= colEnd; col++){  // x
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
	void calHomographyFlowByScale(Mat & finestHomoFlow, int scale){
		resize(finestHomoFlow, homoFlow, image.size(), 0, 0, CV_INTER_AREA);
		homoFlow = homoFlow / (float)scale;
	}

	void calConsistImage(){
		consistImage = Mat::zeros(image.size(), CV_8UC3);
		for(int r = 0; r < image.rows; r++){     // consistentͼ��(r,c)
			for(int c = 0; c < image.cols; c++){
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
	vector<PyramidLayer> pyramid;  // Ĭ����private

public:
	Pyramid(){}    
	Pyramid(Mat & Image){     // ���캯������Image����һ��Image Pyramid
		int cols = Image.cols ;
		int rows = Image.rows;
		int layerNum = 1;
		while(cols > 400 && rows > 400){    // �������Pyramid�м��㣬��˹��������ֲڲ�image�ĳ��߲���400pixels��(����˵��
			cols = cols >> 1;
			rows = rows >> 1;
			//printf("Col Num: %d, Row Num: %d\n", cols, rows);
			layerNum++;
		}
		printf("Layer Num: %d\n", layerNum);
		
		// ����Pyramid, ��߲���ԭͼ�� pyramid[0]����ֲڵ�
		pyramid.resize(layerNum);
		PyramidLayer oriLayer(Image, layerNum - 1);
		pyramid[layerNum - 1] = oriLayer;
		for(int i = layerNum - 2; i >= 0; i--){
			Mat srcImage = pyramid[i+1].getImage();
			Mat dstImage;
			// ��pyrDown�����²���������ִ���˸�˹��������������²����Ĳ���; �ı�ͼ��ߴ绹������resize()
			pyrDown(srcImage, dstImage, Size(srcImage.cols >> 1, srcImage.rows >> 1));
			PyramidLayer newLayer(dstImage, i);
			pyramid[i] = newLayer;
		}
	}

	// ����ÿһ��������㣨��coarse level��������scale�������㣩
	void calFeaturePyramid(){
		vector<Point2f> & featMatchPts = pyramid[FEATURE_LAYER].getCurMatchPts();
		vector<Point2f> & featRefMatchPts = pyramid[FEATURE_LAYER].getRefMatchPts();
		for(unsigned int layer = 0; layer < pyramid.size(); layer++){
			if(layer == FEATURE_LAYER) continue;

			int featureRow = 1 << FEATURE_LAYER;
			int row = 1 << layer;
			float ratio = (float)row / (float)featureRow;
			
			// �����FEATURE_LAYER��0�㣩scale��������
			for(unsigned int i = 0; i < featMatchPts.size(); i++){
				Point2f tempPts(featMatchPts[i].x * ratio, featMatchPts[i].y * ratio);
				Point2f tempRefPts(featRefMatchPts[i].x * ratio, featRefMatchPts[i].y * ratio);
				//cout << featMatchPts[i].x * ratio << "," << featMatchPts[i].y * ratio << endl;
				pyramid[layer].addMatchedPts(tempPts, tempRefPts);
			}
		}
	}

	// ��ÿһ���������ֵ�ÿһ��ImageNode
	void distributeFeaturePtsByLayer(){
		for(unsigned int layer = 0; layer < pyramid.size(); layer++){
			pyramid[layer].distributeFeaturePts();
		}
	}

	// �������ͼ���������homography������
	void calHomographyPyramid(){
		for(unsigned int layer = 0; layer < pyramid.size(); layer++){
			int nodeNumPerEdge = 1 << layer;
			// һ��һ���㣨����ֲڲ㿪ʼ��
			for(int row = 0; row < nodeNumPerEdge; row++){   // ��ÿһ�㣬��ÿ��node��homography
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

	// �������ͼ���������homography flow����ʵ���൱��ÿ����������ڲο�֡��ƫ������
	void calHomographyFlowPyramid(){
		// ����finest level ��homography flow
		int fineLevel = pyramid.size() - 1;
		pyramid[fineLevel].calHomographyFlow();  
		Mat & homoFlow = pyramid[fineLevel].getHomoFlow();
		cout << "homoFlow size: " << homoFlow.size() << endl;
		/*Mat m(4, 4, CV_32FC2);
		for(int i = 0; i < 4; i++)
			for(int j = 0; j < 4; j++)
				m.at<Vec2f>(i,j) = homoFlow.at<Vec2f>(i,j);
		cout << m << endl;*/
		
		// ����ײ��homographyFLow�ȱ�����С���õ��������homographyFlow
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
	vector<Mat> consistentPixels;  // ����൱��ÿ��һ��

	
	vector<Mat> channels;			// ������̬ѧ����ʱ�������֡ͼ��
	/*// ���鼯����
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

	// ��̬ѧ����������Χ3*3��ֵΪ 1 ������ <5 ʱ����0
	int checkMajority(const Mat& src, const int row, const int col) {
		int thre = 5;
		if(row == 0 || row == src.rows || col == 0 || col == src.cols) thre = 3;
		if((row == 0 && col == 0) || (row == 0 && col == src.cols) || (row == src.rows && col == 0) || (row == src.rows && col == src.cols))
			thre = 2;
		int cnt = 0;
		int startR = row - 1, startC = col - 1, endR = row + 1, endC = col + 1;
		for (int r = startR; r <= endR; r++) {
			for (int c = startC; c <= endC; c++) {
				if (r == row && c == col) continue;
				if (r < 0 || r >= src.rows || c < 0 || c >= src.cols) continue;
				if (src.at<uchar>(r, c) == 1) cnt++;
			}
		}
		if(cnt < thre) return 0;					
		return 1;
	}

	// ��̬ѧfilter
	void morphMajority(const Mat & src, Mat & dst){
		Mat temp = src.clone();
		for (int row = 0; row < dst.rows; row++) {
			for (int col = 0; col < dst.cols; col++) {
				dst.at<uchar>(row, col) = checkMajority(temp, row, col);
			}
		}
	}

	// ʹ��BFSʵ��Flood-Fill�㷨
	ConnectedComponent floodFill(int idx, int x0, int y0, Mat & undecidedPixels){
 		int dx[] = { -1, 0, 1, 0 };
 		int dy[] = { 0, 1, 0, -1 };
 		int x = x0, y = y0;
 		queue<Point> Q;										// ����BFS, floodFill��ʵ��
 		Point cur;
 		ConnectedComponent ans(idx);
		ans.addPoint(Point(x0, y0));

		while(!Q.empty()) Q.pop();
		Q.push(Point(x0, y0));

		// BFS,�Ե�ǰ��Ϊ���ģ�����������������չ�ĵ㣨ֵΪ 0 ��Ϊ����չ��
		while(!Q.empty()){
			cur = Q.front(); Q.pop();
			for (int i = 0; i < 4; i++){
				x = cur.x + dx[i]; 
				y = cur.y + dy[i];
				// �ж�Խ����Ƿ����չ
				if (x < 0 || x >= undecidedPixels.cols || y < 0 || y >= undecidedPixels.rows)
					continue;								
				if (undecidedPixels.at<uchar>(y, x) != 255) continue;

				undecidedPixels.at<uchar>(y, x) = 127;		// ���˵���Ϊ��ȷ��
				ans.addPoint(Point(x, y));					// ���뵽��ͨ����
				Q.push(Point(x, y));						// ���뵽�����У��Ա�Ӵ˵㿪ʼ��չ
			}
		}
		return ans;											// ���ص�ǰ��ͨ����
	}

	// Ѱ����ͨ����(Flood-Fill�㷨��	
	void findConnectedComponents(vector<ConnectedComponent> & connComps, Mat & undecidedPixels){
		int idx = 0;
		for (int y = 0; y < undecidedPixels.rows; y++)
			for (int x = 0; x < undecidedPixels.cols; x++)
			if (undecidedPixels.at<uchar>(y, x) == 255){
				connComps.push_back(floodFill(++idx, x, y, undecidedPixels));
			}
	}

	// Ѱ����ͨ����(Two-Pass�㷨��			
	/*void findConnectedComponents(vector<ConnectedComponent> & connComps, Mat & undecidedPixels){
		int idx = 1;
		Mat labeled(undecidedPixels.size(), CV_8U, Scalar::all(0)); 

		uchar parent[MAXIDX] = {0};   // ���鼯����
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

		// ����֮��͵õ������е�Connected Component
		cout << connComps.size() << endl;
	}*/
	
	// ����layer���consistent pixels set
	void calConsistentPixelSet(int layer, vector<Mat> & integralImageSet, Mat & integralMedianImg, const int thre){
		Mat refConsistPixelSet, medConsistPixelSet, consistPixelSet;   // ��¼consistent pixel
		Mat reliablePixelSet; // ��¼reliable pixels

		// ��ʼ��consistent pixel set
		refConsistPixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8UC(FRAME_NUM));
		medConsistPixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8UC(FRAME_NUM));
		consistPixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8UC(FRAME_NUM));
		reliablePixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8U);
		Mat undecidedPixels(consistPixelSet.rows, consistPixelSet.cols, CV_8U, Scalar::all(0));
		
		// ����reference-based �� median-based consistent pixels
		for(int r = 0; r < integralMedianImg.rows - 1; r++){   // ����ͼ����ͨͼ���и���һ��
			for(int c = 0; c < integralMedianImg.cols - 1; c++){

				// �����
				int half = 2;
				int startR = max(0, r - half);
				int endR = min(integralMedianImg.rows - 2, r + half);
				int startC = max(0, c - half);
				int endC = min(integralMedianImg.cols - 2, c + half);
				int pixelNum = (endR - startR + 1) * (endC - startC + 1);

				/* -----����reference-based consistent pixels----- */
				// �ȼ���ref image��5*5������
				int pixelRefSum = integralImageSet[REF].at<int>(endR+1, endC+1) - integralImageSet[REF].at<int>(startR, endC+1)
					- integralImageSet[REF].at<int>(endR+1, startC) + integralImageSet[REF].at<int>(startR, startC);
				int aveRefPixel = pixelRefSum / pixelNum;

				// �Ȱ�ref Image�ĸõ���Ϊconsistent pixel
				Vec<uchar, FRAME_NUM> & elem = refConsistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c);
				elem[REF] = 1;
				
				// ��ref��ʼ��0����ÿһ֡
				for(int i = REF - 1; i >= 0; i--){
					int pixelSum = integralImageSet[i].at<int>(endR+1, endC+1) - integralImageSet[i].at<int>(startR, endC+1)
						- integralImageSet[i].at<int>(endR+1, startC) + integralImageSet[i].at<int>(startR, startC);
					int avePixel = pixelSum / pixelNum;
					if(abs(aveRefPixel - avePixel) < thre)
						elem[i] = 1;
					else
						break;
				}

				// ��ref��ʼ���Ҽ���ÿһ֡
				for(int i = REF + 1; i < FRAME_NUM; i++){
					int pixelSum = integralImageSet[i].at<int>(endR+1, endC+1) - integralImageSet[i].at<int>(startR, endC+1)
						- integralImageSet[i].at<int>(endR+1, startC) + integralImageSet[i].at<int>(startR, startC);
					int avePixel = pixelSum / pixelNum;
					if(abs(aveRefPixel - avePixel) < thre)
						elem[i] = 1;
					else
						break;
				}

				/* -----����median-based consistent pixels----- */
				// ����median image��5*5������
				int pixelMedSum =integralMedianImg.at<int>(endR+1, endC+1) - integralMedianImg.at<int>(startR, endC+1)
					- integralMedianImg.at<int>(endR+1, startC) + integralMedianImg.at<int>(startR, startC);
				int aveMedPixel = pixelMedSum / pixelNum;

				Vec<uchar, FRAME_NUM> & medElem = medConsistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c);
				int cnt = 0; // ��¼�ж��ٸ�median-based consistent pixels
				for(int i = 0; i < FRAME_NUM; i++){
					int pixelSum = integralImageSet[i].at<int>(endR+1, endC+1) - integralImageSet[i].at<int>(startR, endC+1)
						- integralImageSet[i].at<int>(endR+1, startC) + integralImageSet[i].at<int>(startR, startC);
					int avePixel = pixelSum / pixelNum;
					if(abs(aveMedPixel - avePixel) < thre){
						medElem[i] = 1;
						cnt++;
					}	
				}

				/* -----���reference-based �� median-based �Ľ��----- */
				// ���ref frame����median-based consistent pixels, ��ôȡM��R�Ĳ���				
				Vec<uchar, FRAME_NUM> & finalElem = consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c);
				if(medElem[REF] == 1){
					for(int i = 0; i < FRAME_NUM; i++){
						finalElem[i] = elem[i] | medElem[i];
					}
				}
				else undecidedPixels.at<uchar>(r, c) = 255;		

				// ͳ��reliable pixels
				if(cnt > REF) reliablePixelSet.at<uchar>(r, c) = 1;
			}
		}

		/*// ��ʾundecidedPixels
		namedWindow("src", WINDOW_NORMAL);
 		imshow("src", undecidedPixels);
 		waitKey(0);*/

		// �����ҳ���Щundecided pixels����ͨ����
		vector<ConnectedComponent> connComps;
		findConnectedComponents(connComps, undecidedPixels);

		/*// ��ʱ��ԭ��255�Ĳ��ֶ�����Ϊ127
		namedWindow("AfterProcess", WINDOW_NORMAL);
		imshow("AfterProcess", undecidedPixels);
		waitKey(0);

		// ���ҵ�����ͨ�黭��ͼ��
		for (int i = 0; i < connComps.size(); i++){
			uchar color = (i % 2 == 0) ? 100 : 200;						// �����һ����ɫ
			vector<Point> & now = connComps[i].getCCPts();
			for (int k = 0; k < now.size(); k++){
				undecidedPixels.at<uchar>(now[k]) = color;
			}
		}
		namedWindow("Component", WINDOW_NORMAL);
 		imshow("Component", undecidedPixels);
		waitKey(0);*/

		// ͳ��ÿһ����ͨ������reliable pixels�໹��unreliable�ࣨmajority voting ����ͬ�⣩����������ͬ��combine����
		for(unsigned int i = 0; i < connComps.size(); i++){
			vector<Point> & CCpts = connComps[i].getCCPts();
			unsigned int cnt = 0; // ͳ����ͨ�������ж��ٸ�reliable pixel
			
			for(unsigned int j = 0; j < CCpts.size(); j++){
				if(reliablePixelSet.at<uchar>(CCpts[j]) == 1) cnt++;
			}

			// ���reliable pixel�࣬��������ͨ����������reliable����ȡM�Ľ��
			if(cnt >= CCpts.size() - cnt){ 
				for(unsigned int j = 0; j < CCpts.size(); j++){
					Vec<uchar, FRAME_NUM> & finalElem = consistPixelSet.at<Vec<uchar, FRAME_NUM> >(CCpts[j]);
					Vec<uchar, FRAME_NUM> & medElem = medConsistPixelSet.at<Vec<uchar, FRAME_NUM> >(CCpts[j]);
					finalElem = medElem;
				}
			}
			// ��������CC������unreliable����ȡR�Ľ��
			else{
				for(unsigned int j = 0; j < CCpts.size(); j++){
					Vec<uchar, FRAME_NUM> & finalElem = consistPixelSet.at<Vec<uchar, FRAME_NUM> >(CCpts[j]);
					Vec<uchar, FRAME_NUM> & refElem = refConsistPixelSet.at<Vec<uchar, FRAME_NUM> >(CCpts[j]);
					finalElem = refElem;
				}
			}
		}

		// ��̬ѧ���ˣ�morphological(majority) filter
		for (int k = 0; k < FRAME_NUM; k++){
			int cnt = 0;
			for (int i = 0; i < consistPixelSet.rows; i++)
				for (int j = 0; j < consistPixelSet.cols; j++)
					cnt += consistPixelSet.at<Vec<uchar, FRAME_NUM> >(Point(j, i))[k];
			printf("Before: Frame: %d    cnt = %d\n", k, cnt);
		}

		channels.clear();
		split(consistPixelSet, channels);

		for (unsigned int i = 0; i < channels.size(); i++){
			morphMajority(channels[i], channels[i]);
		}
		merge(channels, consistPixelSet);

		// ��ĳ��ȫΪ0��consistPixelSet��ref��һ֡��1����ֹ��̬ѧ���˺�ĳ����consistent pixels�����
		for(int i = 0; i < consistPixelSet.rows; i++)
			for(int j = 0; j < consistPixelSet.cols; j++){
				int isAllZero = 1;
				for(int f = 0; f < FRAME_NUM; f++){
					if(consistPixelSet.at<Vec<uchar, FRAME_NUM> >(i, j)[f] != 0) isAllZero = 0;
				}
				if(isAllZero){
					consistPixelSet.at<Vec<uchar, FRAME_NUM> >(i, j)[REF] = 1; 
				}
			}
		
		for (int k = 0; k < FRAME_NUM; k++){
			int cnt = 0;
			for (int i = 0; i < consistPixelSet.rows; i++)
			for (int j = 0; j < consistPixelSet.cols; j++)
				cnt += consistPixelSet.at<Vec<uchar, FRAME_NUM> >(Point(j, i))[k];
			printf("After: Frame: %d    cnt = %d\n", k, cnt);
		}
		
		consistentPixels[layer] = consistPixelSet;
	}

	// �����²����õ����в��consistent pixels
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
	vector<Mat> oriImageSet;                       // �洢ԭ����ÿ֡ͼƬ
	vector<Pyramid*> imagePyramidSet;              // ͼƬ����������˹��������
	Pyramid* refPyramid;                           // �ο�ͼƬ�Ľ�����
	ConsistentPixelSetPyramid consistPixelPyramid; // �洢consistent pixel
	Mat grayMedianImg;                             // CONSIST_LAYER����λͼ���Ҷ�ͼ�� 
	vector<Mat> temporalResult;                    // temporal fusion�Ľ��ͼ

	Mat refConsistPixelSet, medConsistPixelSet, consistPixelSet;   // ��¼consistent pixel

	double noiseVar;                               // ��������


	/* �ڵ������õ����õ�row*col*3 * 10 ��consistPixelsChannels����ÿ֡��consist map�ֿ��������3ͨ����rgb���� */
	vector<Mat> tempConsistPixelsChannels;             // row*col*1 * 10
	vector<Mat> consistPixelsChannels;                 // row*col*3 * 10
	vector<Mat> tempChannels;                          // row*col*1 * 3

	void readBurstImages(const string fileDir){
		Mat img;
		char index[10];
		for(int i = 0; i < FRAME_NUM; i++){
			sprintf_s(index, "%d", i);
			img = imread(fileDir + string(index) + imageFormat);
			if(img.data == false){
				printf("Cannot Find Img.\n");
				return;
			}
			oriImageSet.push_back(img);
		}
	}

	// ����ͼ�������
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

	// ��һ���� ����ÿ֡ͼƬ��homography flow ������
	void calHomographyFlowPyramidSet(){

		// ����refImage�������������������������
		PyramidLayer* refpLayer = refPyramid->getPyramidLayer(FEATURE_LAYER);
		Mat refDescriptor = refpLayer->getImageDescriptors();  
		vector<KeyPoint> & refKpoint = refpLayer->getKeypoints();

		// BruteForce��FlannBased��opencv��ά������ƥ�䳣�õ����ְ취��BF����ѣ��Ƚϱ�����Flann�죬�ҽ��ƣ�����uchar���������ӣ�BRIEF��ֻ����BF
		//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce" ); 
		Ptr<DescriptorMatcher> matcher = new BruteForceMatcher<L2<float>>;
		
		for(int frame = 0; frame < FRAME_NUM; frame++){

			if(frame == REF) continue;
			// 1. ����ÿһ֡->�ο�֡��Homography��3*3���� ������
			// ���㵱ǰ֡����ֲڲ㣩��ο�֡��ƥ��������
			Pyramid* curPyramid = imagePyramidSet[frame]; // ��ǰͼƬ������
			PyramidLayer* curFrame = curPyramid->getPyramidLayer(FEATURE_LAYER);
			curFrame->calMatchPtsWithRef(refDescriptor, refKpoint, matcher);
	
			// ����ƥ����
			//curFrame->showMatchedImg(curFrame->getImage(), refKpoint);

			// ����ÿһ��������㣨��coarse level��������scale�������㣩
			curPyramid->calFeaturePyramid();

			// ��ÿһ�����������䵽ÿ��ImageNode
			curPyramid->distributeFeaturePtsByLayer();

			// ����ÿһ֡�����ο�֡����homography������
			curPyramid->calHomographyPyramid();

			// ����ÿһ֡�����ο�֡����homography flow������
			curPyramid->calHomographyFlowPyramid();

			cout << endl;
		}
	}

	// �ڶ�����ѡ��consistent pixel 
	void consistentPixelSelection(){
		vector<Mat> integralImageSet;    // ����Consistent Image�Ļ���ͼ
		vector<Mat> consistGrayImageSet; // ����Consistent Image�ĻҶ�ͼ
		const int threshold = 10;        // ��ֵ

		// ȡ��refImage
		PyramidLayer* refpLayer = refPyramid->getPyramidLayer(CONSIST_LAYER);
		Mat refImage = refpLayer->getImage();

		integralImageSet.resize(FRAME_NUM);    // ����Consistent Image�Ļ���ͼ
		consistGrayImageSet.resize(FRAME_NUM); // ����Consistent Image�ĻҶ�ͼ

		for(int frame = 0; frame < FRAME_NUM; frame++){
			if(frame == REF){
				cvtColor(refImage, consistGrayImageSet[frame], CV_RGB2GRAY);
				integral(consistGrayImageSet[frame], integralImageSet[frame], CV_32S);
				// integral(refImage, integralImageSet[frame], CV_32SC3);  �����ת�ҶȾ������ͼ�������
				continue;
			}

			// ��ͼƬ��homography flow������һ��consistent image(���Ͳο�֡һ��)(CONSIST_LAYER)
			Pyramid* curPyramid = imagePyramidSet[frame]; // ��ǰͼƬ������
			PyramidLayer* curFrame = curPyramid->getPyramidLayer(CONSIST_LAYER);
			curFrame->calConsistImage();
			
			// ת�Ҷ�ͼ
			Mat consistImg = curFrame->getConsistImage();
			cvtColor(consistImg, consistGrayImageSet[frame], CV_RGB2GRAY);

			// ������consistent �Ҷ�ͼ�Ļ���ͼ(ԭͼ���и���1����һ�е�һ�о�Ϊ0��
			integral(consistGrayImageSet[frame], integralImageSet[frame], CV_32S);
			//integral(consistImage, integralImageSet[frame], CV_32SC3);  // �����ת�ҶȾ������ͼ�������
		}

		// ��medianͼ(�Ҷ�ͼ)�ͻ���ͼ
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

		// ��ʼ��Consistent pixel pyramid
		consistPixelPyramid = ConsistentPixelSetPyramid(refPyramid->getImagePyramid().size());

		// ��Consistent Pixels(CONSIST_LAYER��)
		consistPixelPyramid.calConsistentPixelSet(CONSIST_LAYER, integralImageSet, integralMedianImg, threshold);

		// reuse the indices of computed consistent pixels by upsampling and downsampling����refPyramid����ȥ��Ϊ��֪��ÿ��ĳߴ�
		consistPixelPyramid.calConsistentPixelsAllLayer(refPyramid->getImagePyramid());
	}

	

	// ���������ںϵõ�����ȥ��ͼ��
	void pixelsFusion(){

		vector<Mat> refImagePyramid = refPyramid->getImagePyramid();
		int layersNum = refImagePyramid.size();

		vector<Mat> numOfInliersAllLayer;    // multi-scale fusionʱҪ�õ�
		numOfInliersAllLayer.resize(layersNum);

		/* -----������������----- */
		// ȡ��ref image��ת�ɻҶ�ͼ��
		Mat refGrayImg;
		cvtColor(refImagePyramid[layersNum - 1], refGrayImg, CV_RGB2GRAY);

		// ȡ����λͼ���Ҷȣ�
		Mat medGrayImg;  
		resize(grayMedianImg, medGrayImg, refGrayImg.size(), 0, 0, CV_INTER_LINEAR);  // grayMedianImg��CONSIST_LAYER��
		
		// ��Ե��ȡ
		Mat edgeImg;
		Canny(grayMedianImg, edgeImg, 50, 125, 3);   // canny��Ե������˫��ֵֵ��������ֵ�������ͼ������Ҫ�ġ������������������ȣ�������ֵ������֤����ʧϸ�ڲ���
		resize(edgeImg, edgeImg, refGrayImg.size(), 0, 0, CV_INTER_NN);
		//imshow("edge", edgeImg);
		//waitKey(0);

		// ��ƽ̹�����ƽ̹�����������
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

		// ��med��ref֮��ľ�ֵ�ͷ��ƽ̹����
		Mat diffImg = medGrayImg - refGrayImg;
		double aveNum = sum(diffImg)[0]/ cnt;
		for(int r = 0; r < refGrayImg.rows; r++)
			for(int c = 0; c < refGrayImg.cols; c++){
				if(edgeImg.at<uchar>(r, c) == 1) diffImg.at<double>(r, c) -= aveNum;
				
			}
		diffImg = diffImg.mul(diffImg);
		noiseVar = sum(diffImg)[0] / cnt;  // sigma2
		

		/* -----����temporal fusion----- */
		vector<Mat> & consistentPixelPyramid = consistPixelPyramid.getConsistentPixelPyramid();
		for(int layer = 0; layer < layersNum; layer++){
			
			// ȡ����һ���consistent pixels��
			consistPixelSet = consistentPixelPyramid[layer];

			vector<Mat> consistImgSet;
			Mat consistGrayImgSet(consistPixelSet.size(), CV_8UC(FRAME_NUM), Scalar::all(0));
			Mat meanImg(consistPixelSet.size(), CV_8U, Scalar::all(0));
			Mat meanRGBImg(consistPixelSet.size(), CV_8UC3, Scalar::all(0));
			Mat tVar(consistPixelSet.size(), CV_64F, Scalar::all(0));
			Mat var(consistPixelSet.size(), CV_64F, Scalar::all(0));

			// �õ�consist gray image set (col * row * 10) �� consistImgSet(vector, col * row * 3 * 10) ��ɫ
			for(int frame = 0; frame < FRAME_NUM; frame++){
				PyramidLayer* curFrame = imagePyramidSet[frame]->getPyramidLayer(layer);
				Mat consistImg;
				if(frame == REF){
					consistImg = curFrame->getImage();
				}
				else{
					if(layer != CONSIST_LAYER)	curFrame->calConsistImage();
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
			
			// ��ˣ�������consistent pixel�ĵط�����
			consistGrayImgSet = consistGrayImgSet.mul(consistPixelSet);

			/*vector<Mat> chan;
			split(consistGrayImgSet, chan);
			for(int i = 0; i < chan.size(); i++){
				imshow("g", chan[i]);
				waitKey(0);
			}*/

			for(int r = 0; r < consistPixelSet.rows; r++)
				for(int c = 0; c < consistPixelSet.cols; c++){
					// ����ƽ��ͼ�񣨻Ҷȣ�
					meanImg.at<uchar>(r, c) = sum(consistGrayImgSet.at<Vec<uchar, FRAME_NUM> >(r, c))[0] 
						/ sum(consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c))[0];
					//cout << "Num of consistent pixels: " << sum(consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c))[0] << endl;

					// ����sigmat������ÿ�����ص�consistent pixels�ķ���
					Vec<int, FRAME_NUM> elem = consistGrayImgSet.at<Vec<uchar, FRAME_NUM>>(r, c);
					//cout << "Consistent Pixels Mark: " << elem << endl;
					Vec<uchar, FRAME_NUM> & consistVector = consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c);
					//cout << "Consistent Image: " << consistVector << endl;
					
					for(int i = 0; i < FRAME_NUM; i++){
						if(consistVector[i] > 0)  elem[i] = abs(elem[i] - meanImg.at<uchar>(r, c));
					}
					elem = elem.mul(elem);
					
					tVar.at<double>(r, c) = (double)sum(elem)[0] / (double)sum(consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c))[0]; 
					//cout << "( " << r << ", " << c << " )  sigmat2: " << tVar.at<double>(r, c) << endl;
					
					// ����sigmac��
					var.at<double>(r, c) = max((double)0, tVar.at<double>(r, c) - noiseVar);

					// ����sigmac2/(sigmac2 + sigma2)
					var.at<double>(r, c) = var.at<double>(r, c) / (var.at<double>(r, c) + noiseVar);
				}
				
			// �õ�row*col*3 * 10 ��consistPixelsChannels����ÿ֡��consist map�ֿ��������3ͨ����rgb����
			tempConsistPixelsChannels.clear();             // row*col*1 * 10
			consistPixelsChannels.clear();
			consistPixelsChannels.resize(FRAME_NUM);                 // row*col*3 * 10
			Mat sumImg(consistPixelSet.size(), CV_32SC3, Scalar::all(0));
			Mat sumConsistImg(consistPixelSet.size(), CV_8UC3, Scalar::all(0));
			split(consistPixelSet, tempConsistPixelsChannels); // ��consistPixelSet 10��ͨ���ֿ�
			
			for(int frame = 0; frame < FRAME_NUM; frame++){
				
				tempChannels.clear();
				for(int i = 0; i < 3; i++) tempChannels.push_back(tempConsistPixelsChannels[frame]);
				merge(tempChannels, consistPixelsChannels[frame]);

				// ��sumImg
				consistImgSet[frame] = consistImgSet[frame].mul(consistPixelsChannels[frame]);
				add(sumImg, consistImgSet[frame], sumImg, Mat(), CV_32SC3); // sumImg(CV_32SC3) += consistImgSet[frame](CV_8UC3); ���Ͳ�ͬ���Ըĳ����������
				sumConsistImg += consistPixelsChannels[frame];
			}

			// ��meanRGBImg
			divide(sumImg, sumConsistImg, meanRGBImg, 1, CV_8UC3);  //meanRGBImg = sumImg / sumConsistImg;  ���
			//imshow("mean", meanRGBImg);
			//waitKey(0);

			// ���ںϺ�Ľ��
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
			//imshow("result", tempResult);
			//waitKey(0);

			// ����number of inliers ��multi-scale fusionʱҪ�õ���
			if(layer != 0){
				calNumOfInlierPerPixel(numOfInliersAllLayer[layer], tVar, consistGrayImgSet, layer);
			}
			
		}

		imwrite(resultDir + "temporal" + imageFormat, temporalResult[layersNum - 1]);

		/* -----����multi-scale fusion----- */
		for(int layer = 1; layer < layersNum; layer++){
			// ͨ��bilinear upscale��scale��һ���ͼ��
			Mat formerLayerImg;
			resize(temporalResult[layer - 1], formerLayerImg, temporalResult[layer].size(), 0, 0, CV_INTER_LINEAR);
			formerLayerImg.convertTo(formerLayerImg, CV_32FC3);

			// ����textureness probalitity ptex
			Mat pTex;
			calPtex(pTex, layer);

			// ����omega w = sqrt(m/FRAME_NUM)
			Mat omega;
			calOmega(omega, numOfInliersAllLayer[layer], layer);

			// �����Щptex > 0.01 �ĵ�
			Mat isTex(pTex.size(), CV_8U);
			for(int r = 0; r < pTex.rows; r++)
				for(int c = 0; c < pTex.cols; c++){
					isTex.at<uchar>(r, c) = (pTex.at<float>(r, c) > 0.01) * 255;
				}
			//imshow("texture", isTex);
			//waitKey(0);
			
			// ����f(xs) directional spatial pixel fusion ��ֻ��ptex>0.01�ĵ���У�
			Mat spatialFusion = temporalResult[layer].clone();
			calSpatialFusion(spatialFusion, isTex);

			// �滻temporal fusion resultΪptex * f(xs) + (1 - ptex) * formerLayer
			// ���滻Ϊ w * xs + (1 - w) * formerLayer���õ�ͼ���ںϵĽ��
			Mat tempResult(spatialFusion.size(), CV_32FC3, Scalar::all(0));
			for(int r = 0; r < pTex.rows; r++)
				for(int c = 0; c < pTex.cols; c++){
					float p = pTex.at<float>(r, c);
					float w = omega.at<float>(r, c);
					tempResult.at<Vec3f>(r, c) = p * spatialFusion.at<Vec3f>(r, c) + (1 - p) * formerLayerImg.at<Vec3f>(r, c);
					tempResult.at<Vec3f>(r, c) = w * tempResult.at<Vec3f>(r, c) + (1 - w) * formerLayerImg.at<Vec3f>(r, c);
				} 
			tempResult.convertTo(tempResult, CV_8UC3);
			temporalResult[layer] = tempResult.clone();
		}

		imwrite(resultDir + "temporalandmultiscale" + imageFormat, temporalResult[layersNum - 1]);
		
	}

	void getSpatialInfo(Vec<int, 5> & spatialPts, Vec3f & sum, int deltar[], int deltac[], int r, int c, Mat & spatialFusion, Mat & graySpatialFusion){
		for(int i = 0; i < 5; i++){
			sum += spatialFusion.at<Vec3f>(r+deltar[i], c+deltac[i]);
			spatialPts[i] = (int)graySpatialFusion.at<uchar>(r+deltar[i], c+deltac[i]);
		}
	}
	
	// ����Spatial Fusion
	void calSpatialFusion(Mat & spatialFusion, Mat & isTex){
		Mat graySpatialFusion;
		cvtColor(spatialFusion, graySpatialFusion, CV_RGB2GRAY);
		spatialFusion.convertTo(spatialFusion, CV_32FC3);

		// �����ݶȣ��ҳ�most probable edge
		Mat grad_x, grad_y;
		Sobel(graySpatialFusion, grad_x, spatialFusion.depth(), 1, 0);
		Sobel(graySpatialFusion, grad_y, spatialFusion.depth(), 0, 1);
		
		for(int r = 2; r < spatialFusion.rows - 2; r++)
			for(int c = 2; c < spatialFusion.cols - 2; c++){
				float angle = atan(grad_y.at<float>(r, c) / grad_x.at<float>(r, c));
				if(isTex.at<uchar>(r, c) == 255){
					Vec3f sum;
					for(int i = 0; i < 3; i++) sum[i] = 0;
					Vec<int, 5> spatialPts;

					if(angle < M_PI / 8 && angle >= - M_PI / 8){  
						int deltar[] = { -2, -1, 0, 1, 2 };
						int deltac[] = { 0, 0, 0, 0, 0 };
						getSpatialInfo(spatialPts, sum, deltar, deltac, r, c, spatialFusion, graySpatialFusion);	
					}
					else if(angle >= M_PI / 8 && angle < M_PI * 3 / 8){
						int deltar[] = { -2, -1, 0, 1, 2 };
						int deltac[] = { -2, -1, 0, 1, 2 };
						getSpatialInfo(spatialPts, sum, deltar, deltac, r, c, spatialFusion, graySpatialFusion);
					}
					else if(angle >= -M_PI * 3 / 8 && angle < -M_PI / 8){
						int deltar[] = { -2, -1, 0, 1, 2 };
						int deltac[] = { 2, 1, 0, -1, -2 };
						getSpatialInfo(spatialPts, sum, deltar, deltac, r, c, spatialFusion, graySpatialFusion);
					}
					else{
						int deltar[] = { 0, 0, 0, 0, 0 };
						int deltac[] = { -2, -1, 0, 1, 2 };
						getSpatialInfo(spatialPts, sum, deltar, deltac, r, c, spatialFusion, graySpatialFusion);	
					}

					Scalar m, sd; // ��ֵ�ͱ�׼��
					meanStdDev(spatialPts, m, sd);
					double var = max((double)0, sd[0] * sd[0] - noiseVar);
					var = var / (var + noiseVar);

					spatialFusion.at<Vec3f>(r, c) = sum / 5 + var * (spatialFusion.at<Vec3f>(r, c) - sum / 5);

				}
			}
	}

	// ����layer���temporal fusion�����ÿһ�������ڵ�ĸ���   |xt - x^| < 3sigmat
	void calNumOfInlierPerPixel(Mat & numOfInlier, Mat & tVar, Mat & consistGrayImgSet, int layer){
		Mat temporalImg = temporalResult[layer];
		Mat temporalGrayImg;
		cvtColor(temporalImg, temporalGrayImg, CV_RGB2GRAY);

		numOfInlier = Mat::zeros(temporalImg.size(), CV_8U);
		
		for(int r = 0; r < numOfInlier.rows; r++)
			for(int c = 0; c < numOfInlier.cols; c++){
				for(int f = 0; f < FRAME_NUM; f++){
					double diff = (double)abs(consistGrayImgSet.at<Vec<uchar, FRAME_NUM>>(r, c)[f] - temporalGrayImg.at<uchar>(r, c));
					if(diff < 3 * tVar.at<double>(r, c))
						numOfInlier.at<uchar>(r, c)++;
				}
			}
	}

	// ����layer���temporal fusion�����omega w = sqrt(m/FRAME_NUM)
	void calOmega(Mat & omega, Mat & numOfInlier, int layer){
		omega = Mat::zeros(temporalResult[layer].size(), CV_32F);

		for(int r = 0; r < omega.rows; r++)
			for(int c = 0; c < omega.cols; c++){
				omega.at<float>(r, c) = sqrt((float)numOfInlier.at<uchar>(r, c) / (float)FRAME_NUM);
			}
	}

	// ����layer���temporal fusion�����textureness probalitity Ptex
	void calPtex(Mat & pTex, int layer){
		
		Mat temporalImg = temporalResult[layer];
		pTex = Mat::zeros(temporalImg.size(), CV_32F);

		Mat temporalGrayImg;
		cvtColor(temporalImg, temporalGrayImg, CV_RGB2GRAY);
		int dx[] = { -1, 0, 1, 0 };
 		int dy[] = { 0, 1, 0, -1 };
		for(int y = 0; y < temporalImg.rows; y++)
			for(int x = 0; x < temporalImg.cols; x++){
				// ��max absolute difference
				float maxAbsDiff = 0;
				for(int i = 0; i < 4; i++){
					int x_ = x + dx[i];
					int y_ = y + dy[i];
					if(x_ < 0 || y_ < 0 || x_ >= temporalImg.cols || y_ >= temporalImg.rows) continue;
					float absDiff = abs((float)temporalGrayImg.at<uchar>(y_, x_) - (float)temporalGrayImg.at<uchar>(y, x));
					maxAbsDiff = max(absDiff, maxAbsDiff);
				}
				float p = 1 / (1 + exp(-5 * (maxAbsDiff / sqrt(noiseVar) - 3)));
				pTex.at<float>(y, x) = p;
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
			waitKey(0);  // waitKey(1000),�ȴ�1000 ms�󴰿��Զ��ر�; waitKey(0)�ȴ�����
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
	

	system("pause");

	return 0;
}