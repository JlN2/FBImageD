#include <iostream>
#include <vector>
#include <stdio.h>
#include <opencv2\core\core.hpp>     // 有Mat数据类型
#include <opencv2\highgui\highgui.hpp>

using namespace std;
using namespace cv;

const string fileDir  = "SrcData\\Data\\Bookshelf_2\\";
const string imageFormat = ".jpg";
const int FRAME_NUM = 10;
const int REF = FRAME_NUM / 2;

class FastBurstImagesDenoising{
public:
	vector<Mat> oriImageSet;      // 存储原来的每帧图片

	void readBurstImages(const string fileDir){
		Mat img;
		char index[10];
		for(int i = 0; i < FRAME_NUM; i++){
			sprintf_s(index, "%d", i);
			img = imread(fileDir + string(index) + imageFormat);
			oriImageSet.push_back(img);
		}
	}

	void showImages(vector<Mat> Images){
		cout << "Image Size: " << Images.size() << endl;
		char index[10];
		for(int i = 0; i < FRAME_NUM; i++){
			sprintf_s(index, "%d", i);
			imshow(index, Images[i]);
		}
		waitKey(1000);
	}


};

FastBurstImagesDenoising FBID;

int main(){
	FBID.readBurstImages(fileDir);
	FBID.showImages(FBID.oriImageSet);

	system("pause");

	return 0;
}