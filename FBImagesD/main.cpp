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
const string imageFormat = ".jpg";
const int FRAME_NUM = 10;
const int REF = FRAME_NUM / 2;
const int FEATURE_LAYER = 0;   //------------------Ӧ�ô���һ�㿪ʼ�������㣿

class PyramidLayer{
	Mat image; // ��һ���ͼƬ
	vector<KeyPoint> keypoints;  // ��ͼƬ��������
	Mat descriptors; // ��ͼƬ����������  

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

	Mat getImageDescriptors(){
		calImageDescriptors();
		return descriptors;
	}

	vector<KeyPoints> & getKeypoints(){
		return keypoints;
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

		// 1. ����ÿһ֡�Ͳο�֡��Homography��3*3���� ������
		for(int frame = 0; frame < FRAME_NUM; frame++){
			// ���㵱ǰ֡������������������
			curPyramid = imagePyramidSet[frame]; // ��ǰͼƬ������


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

	system("pause");

	return 0;
}