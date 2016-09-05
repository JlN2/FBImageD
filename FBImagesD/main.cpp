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
const int FEATURE_LAYER = 0;   // 应该从哪一层开始算特征向量：starting from a global homography at the coarsest level

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
		calKeyPoints();
		BriefDescriptorExtractor briefExtractor;
		briefExtractor.compute(image, keypoints, descriptors);
		cout << "Descriptors Size: " << descriptors.size() << endl;
	}

	Mat getImageDescriptors(){
		calImageDescriptors();
		return descriptors;
	}

	vector<KeyPoint> & getKeypoints(){
		return keypoints;
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
		
		// 1. 计算每一帧和参考帧的Homography（3*3矩阵） 金字塔
		for(int frame = 0; frame < FRAME_NUM; frame++){

			if(frame == REF) continue;

			// 计算当前帧的特征向量和特征点
			Pyramid* curPyramid = imagePyramidSet[frame]; // 当前图片金字塔
			PyramidLayer* curFrame = curPyramid->getPyramidLayer(FEATURE_LAYER);
			Mat curDescriptor = curFrame->getImageDescriptors();
			vector<KeyPoint> & curKpoint = curFrame->getKeypoints();

			// 进行特征向量匹配
			vector<DMatch> matches; // 匹配结果
			matcher->match(curDescriptor, refDescriptor, matches); // queryDescriptor, trainDescriptor
			cout << "Matches Num: " << matches.size() << endl; // 这里的size和queryDescriptor的行数一样,为query的每一个向量都找了一个匹配向量

			// 根据距离，选出其中的较优的匹配点
			double maxDist = 0;
			double minDist = 100;
			for(int i = 0; i < matches.size(); i++){
				double dist = matches[i].distance;
				if(dist < minDist) minDist = dist;
				if(dist > maxDist) maxDist = dist;
			}
			cout << "Max Distance: " << maxDist << endl;
			cout << "Min Distance: " << minDist << endl;

			vector<DMatch> goodMatches;
			for(unsigned int i = 0; i < matches.size(); i++){
				if(matches[i].distance < minDist + 0.4 * (maxDist - minDist)){
					goodMatches.push_back(matches[i]);
				}
			}
			cout << "Good Matches Num: " << goodMatches.size() << endl;

			// 分别取出两个图像中匹配的特征点
			int matchedNum = (int)goodMatches.size();
			vector<Point2f> refMatchPts, curMatchPts;
			for(int i = 0; i < matchedNum; i++){
				refMatchPts.push_back(refKpoint[goodMatches[i].trainIdx].pt);
				curMatchPts.push_back(curKpoint[goodMatches[i].queryIdx].pt);
			}
			
			// 计算基础矩阵F(用RANSAC方法)：表示的是某个物体或场景各特征在不同的两张照片对应特征点图像坐标的关系，x'的转置乘以F，再乘以x的结果为0
			// RANSAC为RANdom Sample Consensus的缩写，它是根据一组包含异常数据的样本数据集，计算出数据的数学模型参数，得到有效样本数据的算法
			Mat fundMat;
			vector<uchar> RANSACStatus;   // 这个变量用于存储RANSAC后每个点的状态,值为0（错误匹配,野点）,1 
			findFundamentalMat(curMatchPts, refMatchPts, RANSACStatus, FM_RANSAC);
			
			// 使用RANSAC方法计算基础矩阵后可以得到一个status向量，用来删除错误的匹配
			vector<Point2f> refInlierPt, curInlierPt; 
			vector<DMatch> inlierMatches;
			for(int i = 0; i < matchedNum; i++){
				if(RANSACStatus[i] != 0){
					refInlierPt.push_back(refMatchPts[i]);
					curInlierPt.push_back(curMatchPts[i]);
					inlierMatches.push_back(goodMatches[i]);
				}
			}
			cout << "Matches Num After RANSAC: " << inlierMatches.size() << endl;

			// 画出匹配结果
			/*Mat matchedImg;
			drawMatches(curFrame->getImage(), curKpoint, refpLayer->getImage(), refKpoint, inlierMatches, matchedImg, 
				Scalar::all(-1), CV_RGB(0,255,0), Mat(), 2);
			imshow("Matched Result", matchedImg);
			waitKey(0);*/

					
			

			
			cout << endl;
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

	system("pause");

	return 0;
}