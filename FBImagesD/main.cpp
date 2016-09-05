#include <iostream>
#include <vector>
#include <stdio.h>
#include <opencv2\core\core.hpp>     // ��Mat��������
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\features2d.hpp>

using namespace std;
using namespace cv;

const string fileDir  = "SrcData\\Data\\Bookshelf_2\\";
const string resultDir = "Result\\";
const string imageFormat = ".jpg";
const int FRAME_NUM = 10;
const int REF = FRAME_NUM / 2;
const int FEATURE_LAYER = 0;   // Ӧ�ô���һ�㿪ʼ������������starting from a global homography at the coarsest level

class PyramidLayer{
	Mat image; // ��һ���ͼƬ
	vector<KeyPoint> keypoints;  // ��ͼƬ��������
	Mat descriptors; // ��ͼƬ����������  
	vector<Point2f> inlierPts; // ��refImageƥ��ɹ���������
	vector<Point2f> refInlierPts; 
	vector<DMatch> inlierMatches; // ƥ�䣬queryIdx, trainIdx��Ȼ��Ӧ���ʼ�����keypoints��refKpoint������

public:
	PyramidLayer(){}
	PyramidLayer(Mat & newImage){
		image = newImage;
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

	// 
	void calMatchPtsWithRef(Mat & refDescriptor, vector<KeyPoint> & refKpoint, Ptr<DescriptorMatcher> matcher){
		calImageDescriptors();  // ���㵱֡������������������

		// ������������ƥ��
		vector<DMatch> matches; // ƥ����
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

		vector<DMatch> goodMatches;
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
		vector<uchar> RANSACStatus;   // ����������ڴ洢RANSAC��ÿ�����״̬,ֵΪ0������ƥ��,Ұ�㣩,1 
		findFundamentalMat(curMatchPts, refMatchPts, RANSACStatus, FM_RANSAC);
			
		// ʹ��RANSAC������������������Եõ�һ��status����������ɾ�������ƥ��
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
		while(cols > 400 || rows > 400){    // �������Pyramid�м��㣬��˹��������ֲڲ�image�ĳ��߲���400pixels��(����˵��
			cols = cols >> 1;
			rows = rows >> 1;
			//printf("Col Num: %d, Row Num: %d\n", cols, rows);
			layerNum++;
		}
		printf("Layer Num: %d\n", layerNum);
		
		// ����Pyramid, ��߲���ԭͼ�� pyramid[0]����ֲڵ�
		pyramid.resize(layerNum);
		pyramid[layerNum - 1] = Image;
		for(int i = layerNum - 2; i >= 0; i--){
			Mat srcImage = pyramid[i+1].getImage();
			Mat dstImage;
			// ��pyrDown�����²���������ִ���˸�˹��������������²����Ĳ���; �ı�ͼ��ߴ绹������resize()
			pyrDown(srcImage, dstImage, Size(srcImage.cols >> 1, srcImage.rows >> 1));
			PyramidLayer newLayer(dstImage);
			pyramid[i] = newLayer;
		}
	}

	vector<Mat> getImagePyramid(){
		vector<Mat> imagePyramid;
		for(int i = 0; i < pyramid.size(); i++){
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
	vector<Mat> oriImageSet;      // �洢ԭ����ÿ֡ͼƬ
	vector<Pyramid*> imagePyramidSet;  // ͼƬ����������˹��������
	Pyramid* refPyramid;   // �ο�ͼƬ�Ľ�����

	void readBurstImages(const string fileDir){
		Mat img;
		char index[10];
		for(int i = 0; i < FRAME_NUM; i++){
			sprintf_s(index, "%d", i);
			img = imread(fileDir + string(index) + imageFormat);
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
		Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce" ); 
		
		// 1. ����ÿһ֡�Ͳο�֡��Homography��3*3���� ������
		for(int frame = 0; frame < FRAME_NUM; frame++){

			if(frame == REF) continue;

			// ���㵱ǰ֡����ֲڲ㣩��ο�֡��ƥ��������
			Pyramid* curPyramid = imagePyramidSet[frame]; // ��ǰͼƬ������
			PyramidLayer* curFrame = curPyramid->getPyramidLayer(FEATURE_LAYER);
			curFrame->calMatchPtsWithRef(refDescriptor, refKpoint, matcher);
	
			// ����ƥ����
			curFrame->showMatchedImg(curFrame->getImage(), refKpoint);

			// ����homography
			//Mat homography = findHomography(curFrame->getCurMatchPts(), refInlierPt, CV_FM_RANSAC);  // srcPt, dstPt
			//cout << homography << endl;

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

	system("pause");

	return 0;
}