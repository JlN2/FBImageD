
#include "S3_Pyramid.h"

Pyramid::Pyramid(){}
Pyramid::Pyramid(Mat & Image){					// 构造函数，对Image构造一个Image Pyramid
	int cols = Image.cols;
	int rows = Image.rows;
	int layerNum = 1;
	while (cols > 600 && rows > 600){			// 计算这个Pyramid有几层，高斯金字塔最粗糙层image的长边不比400pixels大(文中说）
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
	for (int i = layerNum - 2; i >= 0; i--){
		Mat srcImage = pyramid[i + 1].getImage();
		Mat dstImage;
		// 用pyrDown进行下采样操作，执行了高斯金字塔建造的向下采样的步骤; 改变图像尺寸还可以用resize()
		pyrDown(srcImage, dstImage, Size(srcImage.cols >> 1, srcImage.rows >> 1));
		PyramidLayer newLayer(dstImage, i);
		pyramid[i] = newLayer;
	}
}

// 计算每一层的特征点（将coarse level的特征点scale到其他层）
void Pyramid::calFeaturePyramid(){
	vector<Point2f> & featMatchPts = pyramid[FEATURE_LAYER].getCurMatchPts();
	vector<Point2f> & featRefMatchPts = pyramid[FEATURE_LAYER].getRefMatchPts();
	for (unsigned int layer = 0; layer < pyramid.size(); layer++){
		if (layer == FEATURE_LAYER) continue;

		int featureRow = 1 << FEATURE_LAYER;
		int row = 1 << layer;
		float ratio = (float)row / (float)featureRow;

		// 将点从FEATURE_LAYER（0层）scale到其他层
		for (unsigned int i = 0; i < featMatchPts.size(); i++){
			Point2f tempPts(featMatchPts[i].x * ratio, featMatchPts[i].y * ratio);
			Point2f tempRefPts(featRefMatchPts[i].x * ratio, featRefMatchPts[i].y * ratio);
			//cout << featMatchPts[i].x * ratio << "," << featMatchPts[i].y * ratio << endl;
			pyramid[layer].addMatchedPts(tempPts, tempRefPts);
		}
	}
}

void Pyramid::calFeaturePyramid1(){
	vector<Point2f> & featMatchPts = pyramid[FEATURE_LAYER].getCurMatchPts();
	vector<Point2f> & featRefMatchPts = pyramid[FEATURE_LAYER].getRefMatchPts();
	for (unsigned int layer = 0; layer < pyramid.size(); layer++){
		if (layer == FEATURE_LAYER) continue;

		int featureRow = 1 << FEATURE_LAYER;
		int row = 1 << layer;
		float ratio = 1;      // ---------------- ratio改成1

		// 将点从FEATURE_LAYER（0层）scale到其他层
		for (unsigned int i = 0; i < featMatchPts.size(); i++){
			Point2f tempPts(featMatchPts[i].x * ratio, featMatchPts[i].y * ratio);
			Point2f tempRefPts(featRefMatchPts[i].x * ratio, featRefMatchPts[i].y * ratio);
			//cout << featMatchPts[i].x * ratio << "," << featMatchPts[i].y * ratio << endl;
			pyramid[layer].addMatchedPts(tempPts, tempRefPts);
		}
	}
}

// 将每一层的特征点分到每一个ImageNode
void Pyramid::distributeFeaturePtsByLayer(){
	for (unsigned int layer = 0; layer < pyramid.size(); layer++){
		pyramid[layer].distributeFeaturePts();
	}
}

void Pyramid::distributeFeaturePtsByLayer1(int rows, int cols){
	for (unsigned int layer = 0; layer < pyramid.size(); layer++){
		pyramid[layer].distributeFeaturePts1(rows, cols);
	}
}

// 计算这个图像金字塔的homography金字塔
void Pyramid::calHomographyPyramid(){
	for (unsigned int layer = 0; layer < pyramid.size(); layer++){
		int nodeNumPerEdge = 1 << layer;
		// 一层一层算（从最粗糙层开始）
		for (int row = 0; row < nodeNumPerEdge; row++){   // 对每一层，算每个node的homography
			for (int col = 0; col < nodeNumPerEdge; col++){
				Mat parentHomography(3, 3, CV_64F, Scalar::all(0));

				if (layer != 0){
					int parentRow = row >> 1;
					int parentCol = col >> 1;
					parentHomography = pyramid[layer - 1].getNodesHomography(parentRow, parentCol);
				}
				pyramid[layer].calNodeHomography(row, col, parentHomography);
				//cout << row << "," << col << endl << pyramid[layer].getNodesHomography(row, col) << endl;
			}
		}
		//cout << endl;
	}
}

// 计算这个图像金字塔的homography flow（其实就相当于每个像素相对于参考帧的偏移量）
void Pyramid::calHomographyFlowPyramid(){
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
	for (int i = fineLevel - 1; i >= 0; i--){
		int scale = 1 << (fineLevel - i);
		pyramid[i].calHomographyFlowByScale(homoFlow, scale);
		Mat & homof = pyramid[i].getHomoFlow();
		cout << "homof size: " << homof.size() << endl;
	}
}

vector<Mat> Pyramid::getImagePyramid(){
	vector<Mat> imagePyramid;
	for (unsigned int i = 0; i < pyramid.size(); i++){
		imagePyramid.push_back(pyramid[i].getImage());
	}
	return imagePyramid;
}

PyramidLayer* Pyramid::getPyramidLayer(int layer){
	return &(pyramid[layer]);
}


