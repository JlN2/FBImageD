
#include "S4_ConsistentPixelSetPyramid.h"

ConnectedComponent::ConnectedComponent(){}
ConnectedComponent::ConnectedComponent(int _idx){
	idx = _idx;
}
void ConnectedComponent::addPoint(Point p){
	pts.push_back(p);
}
int ConnectedComponent::getIdx(){
	return idx;
}
vector<Point> & ConnectedComponent::getCCPts(){
	return pts;
}


ConsistentPixelSetPyramid::ConsistentPixelSetPyramid(){}
ConsistentPixelSetPyramid::ConsistentPixelSetPyramid(int layerNum){
	consistentPixels.resize(layerNum);
}

int ConsistentPixelSetPyramid::checkMajority(const Mat& src, const int row, const int col) {
	int thre = 5;
	if (row == 0 || row == src.rows || col == 0 || col == src.cols) thre = 3;
	if ((row == 0 && col == 0) || (row == 0 && col == src.cols) || (row == src.rows && col == 0) || (row == src.rows && col == src.cols))
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
	if (cnt < thre) return 0;
	return 1;
}
// ��̬ѧfilter
void ConsistentPixelSetPyramid::morphMajority(const Mat & src, Mat & dst){
	Mat temp = src.clone();
	for (int row = 0; row < dst.rows; row++) {
		for (int col = 0; col < dst.cols; col++) {
			dst.at<uchar>(row, col) = checkMajority(temp, row, col);
		}
	}
}

// ʹ��BFSʵ�ֺ�ˮ����㷨
ConnectedComponent ConsistentPixelSetPyramid::floodFill(int idx, int x0, int y0, Mat & undecidedPixels){
	int dx[] = { -1, 0, 1, 0 };
	int dy[] = { 0, 1, 0, -1 };
	int x = x0, y = y0;
	queue<Point> Q;										// ����BFS, floodFill��ʵ��
	Point cur;
	ConnectedComponent ans(idx);
	ans.addPoint(Point(x0, y0));

	while (!Q.empty()) Q.pop();
	Q.push(Point(x0, y0));

	// BFS,�Ե�ǰ��Ϊ���ģ�����������������չ�ĵ㣨ֵΪ 0 ��Ϊ����չ��
	while (!Q.empty()){
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

void ConsistentPixelSetPyramid::findConnectedComponents(vector<ConnectedComponent> & connComps, Mat & undecidedPixels){
	int idx = 0;
	for (int y = 0; y < undecidedPixels.rows; y++)
	for (int x = 0; x < undecidedPixels.cols; x++)
	if (undecidedPixels.at<uchar>(y, x) == 255){
		connComps.push_back(floodFill(++idx, x, y, undecidedPixels));
	}
}

// ����layer���consistent pixels set
void ConsistentPixelSetPyramid::calConsistentPixelSet(int layer, vector<Mat> & integralImageSet, Mat & integralMedianImg, const int thre){
	Mat refConsistPixelSet, medConsistPixelSet, consistPixelSet;   // ��¼consistent pixel
	Mat reliablePixelSet; // ��¼reliable pixels

	// ��ʼ��consistent pixel set
	refConsistPixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8UC(FRAME_NUM));
	medConsistPixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8UC(FRAME_NUM));
	consistPixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8UC(FRAME_NUM));
	reliablePixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8U);
	Mat undecidedPixels(consistPixelSet.rows, consistPixelSet.cols, CV_8U, Scalar::all(0));

	// ����reference-based �� median-based consistent pixels
	for (int r = 0; r < integralMedianImg.rows - 1; r++){   // ����ͼ����ͨͼ���и���һ��
		for (int c = 0; c < integralMedianImg.cols - 1; c++){

			// �����
			int half = 2;
			int startR = max(0, r - half);
			int endR = min(integralMedianImg.rows - 2, r + half);
			int startC = max(0, c - half);
			int endC = min(integralMedianImg.cols - 2, c + half);
			int pixelNum = (endR - startR + 1) * (endC - startC + 1);

			/* -----����reference-based consistent pixels----- */
			// �ȼ���ref image��5*5������
			int pixelRefSum = integralImageSet[REF].at<int>(endR + 1, endC + 1) - integralImageSet[REF].at<int>(startR, endC + 1)
				- integralImageSet[REF].at<int>(endR + 1, startC) + integralImageSet[REF].at<int>(startR, startC);
			int aveRefPixel = pixelRefSum / pixelNum;

			// �Ȱ�ref Image�ĸõ���Ϊconsistent pixel
			Vec<uchar, FRAME_NUM> & elem = refConsistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c);
			elem[REF] = 1;

			// ��ref��ʼ��0����ÿһ֡
			for (int i = REF - 1; i >= 0; i--){
				int pixelSum = integralImageSet[i].at<int>(endR + 1, endC + 1) - integralImageSet[i].at<int>(startR, endC + 1)
					- integralImageSet[i].at<int>(endR + 1, startC) + integralImageSet[i].at<int>(startR, startC);
				int avePixel = pixelSum / pixelNum;
				if (abs(aveRefPixel - avePixel) < thre)
					elem[i] = 1;
				else
					break;
			}

			// ��ref��ʼ���Ҽ���ÿһ֡
			for (int i = REF + 1; i < FRAME_NUM; i++){
				int pixelSum = integralImageSet[i].at<int>(endR + 1, endC + 1) - integralImageSet[i].at<int>(startR, endC + 1)
					- integralImageSet[i].at<int>(endR + 1, startC) + integralImageSet[i].at<int>(startR, startC);
				int avePixel = pixelSum / pixelNum;
				if (abs(aveRefPixel - avePixel) < thre)
					elem[i] = 1;
				else
					break;
			}

			/* -----����median-based consistent pixels----- */
			// ����median image��5*5������
			int pixelMedSum = integralMedianImg.at<int>(endR + 1, endC + 1) - integralMedianImg.at<int>(startR, endC + 1)
				- integralMedianImg.at<int>(endR + 1, startC) + integralMedianImg.at<int>(startR, startC);
			int aveMedPixel = pixelMedSum / pixelNum;

			Vec<uchar, FRAME_NUM> & medElem = medConsistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c);
			int cnt = 0; // ��¼�ж��ٸ�median-based consistent pixels
			for (int i = 0; i < FRAME_NUM; i++){
				int pixelSum = integralImageSet[i].at<int>(endR + 1, endC + 1) - integralImageSet[i].at<int>(startR, endC + 1)
					- integralImageSet[i].at<int>(endR + 1, startC) + integralImageSet[i].at<int>(startR, startC);
				int avePixel = pixelSum / pixelNum;
				if (abs(aveMedPixel - avePixel) < thre){
					medElem[i] = 1;
					cnt++;
				}
			}

			/* -----���reference-based �� median-based �Ľ��----- */
			// ���ref frame����median-based consistent pixels, ��ôȡM��R�Ĳ���				
			Vec<uchar, FRAME_NUM> & finalElem = consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c);
			if (medElem[REF] == 1){
				for (int i = 0; i < FRAME_NUM; i++){
					finalElem[i] = elem[i] | medElem[i];
				}
			}
			else undecidedPixels.at<uchar>(r, c) = 255;

			// ͳ��reliable pixels
			if (cnt > REF) reliablePixelSet.at<uchar>(r, c) = 1;
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
	for (unsigned int i = 0; i < connComps.size(); i++){
		vector<Point> & CCpts = connComps[i].getCCPts();
		unsigned int cnt = 0; // ͳ����ͨ�������ж��ٸ�reliable pixel

		for (unsigned int j = 0; j < CCpts.size(); j++){
			if (reliablePixelSet.at<uchar>(CCpts[j]) == 1) cnt++;
		}

		// ���reliable pixel�࣬��������ͨ����������reliable����ȡM�Ľ��
		if (cnt >= CCpts.size() - cnt){
			for (unsigned int j = 0; j < CCpts.size(); j++){
				Vec<uchar, FRAME_NUM> & finalElem = consistPixelSet.at<Vec<uchar, FRAME_NUM> >(CCpts[j]);
				Vec<uchar, FRAME_NUM> & medElem = medConsistPixelSet.at<Vec<uchar, FRAME_NUM> >(CCpts[j]);
				finalElem = medElem;
			}
		}
		// ��������CC������unreliable����ȡR�Ľ��
		else{
			for (unsigned int j = 0; j < CCpts.size(); j++){
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
	for (int i = 0; i < consistPixelSet.rows; i++)
	for (int j = 0; j < consistPixelSet.cols; j++){
		int isAllZero = 1;
		for (int f = 0; f < FRAME_NUM; f++){
			if (consistPixelSet.at<Vec<uchar, FRAME_NUM> >(i, j)[f] != 0) isAllZero = 0;
		}
		if (isAllZero){
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
void ConsistentPixelSetPyramid::calConsistentPixelsAllLayer(vector<Mat> & refPyramid){
	for (unsigned int layer = 0; layer < consistentPixels.size(); layer++){
		if (layer == CONSIST_LAYER) continue;
		resize(consistentPixels[CONSIST_LAYER], consistentPixels[layer], refPyramid[layer].size(), 0, 0, CV_INTER_LINEAR);  // CV_NEAREST
	}
}
vector<Mat> & ConsistentPixelSetPyramid::getConsistentPixelPyramid(){
	return consistentPixels;
}