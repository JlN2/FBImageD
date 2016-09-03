#include <iostream>
#include <vector>
#include <stdio.h>
#include <opencv2\core\core.hpp>     // 有Mat数据类型
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
const int FEATURE_LAYER = 0;   //------------------应该从哪一层开始算特征点？

class PyramidLayer{
	Mat image; // 这一层的图片
	vector<KeyPoint> keypoints;  // 该图片的特征点
	Mat descriptors; // 该图片的特征向量  

public:
	PyramidLayer(){}
	PyramidLayer(Mat & newImage){
		image = newImage;
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
		BriefDescriptorExtractor briefExtractor;
		briefExtractor.compute(image, keypoints, descriptors);
		cout << "Descriptors Size: " << descriptors.size() << endl;
	}

	Mat getImageDescriptors(){
		return descriptors;
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
		pyramid[layerNum - 1] = Image;
		for(int i = layerNum - 2; i >= 0; i--){
			Mat srcImage = pyramid[i+1].getImage();
			Mat dstImage;
			// 用pyrDown进行下采样操作，执行了高斯金字塔建造的向下采样的步骤; 改变图像尺寸还可以用resize()
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
	vector<Mat> oriImageSet;      // 存储原来的每帧图片
	vector<Pyramid*> imagePyramidSet;  // 图片金字塔（高斯金字塔）
	Pyramid* refPyramid;   // 参考图片的金字塔

	void readBurstImages(const string fileDir){
		Mat img;
		char index[10];
		for(int i = 0; i < FRAME_NUM; i++){
			sprintf_s(index, "%d", i);
			img = imread(fileDir + string(index) + imageFormat);
			oriImageSet.push_back(img);
		}
	}

	void calPyramidSet(){
		int frameNum = oriImageSet.size();
		imagePyramidSet.resize(frameNum);
		for(int i = 0; i < frameNum; i++){
			printf("Frame %d: ", i);
			imagePyramidSet[i] = new Pyramid(oriImageSet[i]);
		}
		refPyramid = imagePyramidSet[REF];
		//showImages(refPyramid->getImagePyramid());
		PyramidLayer* pLayer = refPyramid->getPyramidLayer(FEATURE_LAYER);
		pLayer->calKeyPoints();
		pLayer->calImageDescriptors();
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

	system("pause");

	return 0;
}