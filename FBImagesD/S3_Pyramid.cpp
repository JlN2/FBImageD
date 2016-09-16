
#include "S3_Pyramid.h"

Pyramid::Pyramid(){}
Pyramid::Pyramid(Mat & Image){					// ���캯������Image����һ��Image Pyramid
	int cols = Image.cols;
	int rows = Image.rows;
	int layerNum = 1;
	while (cols > 600 && rows > 600){			// �������Pyramid�м��㣬��˹��������ֲڲ�image�ĳ��߲���400pixels��(����˵��
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
	for (int i = layerNum - 2; i >= 0; i--){
		Mat srcImage = pyramid[i + 1].getImage();
		Mat dstImage;
		// ��pyrDown�����²���������ִ���˸�˹��������������²����Ĳ���; �ı�ͼ��ߴ绹������resize()
		pyrDown(srcImage, dstImage, Size(srcImage.cols >> 1, srcImage.rows >> 1));
		PyramidLayer newLayer(dstImage, i);
		pyramid[i] = newLayer;
	}
}

// ����ÿһ��������㣨��coarse level��������scale�������㣩
void Pyramid::calFeaturePyramid(){
	vector<Point2f> & featMatchPts = pyramid[FEATURE_LAYER].getCurMatchPts();
	vector<Point2f> & featRefMatchPts = pyramid[FEATURE_LAYER].getRefMatchPts();
	for (unsigned int layer = 0; layer < pyramid.size(); layer++){
		if (layer == FEATURE_LAYER) continue;

		int featureRow = 1 << FEATURE_LAYER;
		int row = 1 << layer;
		float ratio = (float)row / (float)featureRow;

		// �����FEATURE_LAYER��0�㣩scale��������
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
		float ratio = 1;      // ---------------- ratio�ĳ�1

		// �����FEATURE_LAYER��0�㣩scale��������
		for (unsigned int i = 0; i < featMatchPts.size(); i++){
			Point2f tempPts(featMatchPts[i].x * ratio, featMatchPts[i].y * ratio);
			Point2f tempRefPts(featRefMatchPts[i].x * ratio, featRefMatchPts[i].y * ratio);
			//cout << featMatchPts[i].x * ratio << "," << featMatchPts[i].y * ratio << endl;
			pyramid[layer].addMatchedPts(tempPts, tempRefPts);
		}
	}
}

// ��ÿһ���������ֵ�ÿһ��ImageNode
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

// �������ͼ���������homography������
void Pyramid::calHomographyPyramid(){
	for (unsigned int layer = 0; layer < pyramid.size(); layer++){
		int nodeNumPerEdge = 1 << layer;
		// һ��һ���㣨����ֲڲ㿪ʼ��
		for (int row = 0; row < nodeNumPerEdge; row++){   // ��ÿһ�㣬��ÿ��node��homography
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

// �������ͼ���������homography flow����ʵ���൱��ÿ����������ڲο�֡��ƫ������
void Pyramid::calHomographyFlowPyramid(){
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


