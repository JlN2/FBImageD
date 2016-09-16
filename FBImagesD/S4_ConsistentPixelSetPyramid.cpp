
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
// 形态学filter
void ConsistentPixelSetPyramid::morphMajority(const Mat & src, Mat & dst){
	Mat temp = src.clone();
	for (int row = 0; row < dst.rows; row++) {
		for (int col = 0; col < dst.cols; col++) {
			dst.at<uchar>(row, col) = checkMajority(temp, row, col);
		}
	}
}

// 使用BFS实现洪水灌溉算法
ConnectedComponent ConsistentPixelSetPyramid::floodFill(int idx, int x0, int y0, Mat & undecidedPixels){
	int dx[] = { -1, 0, 1, 0 };
	int dy[] = { 0, 1, 0, -1 };
	int x = x0, y = y0;
	queue<Point> Q;										// 用于BFS, floodFill的实现
	Point cur;
	ConnectedComponent ans(idx);
	ans.addPoint(Point(x0, y0));

	while (!Q.empty()) Q.pop();
	Q.push(Point(x0, y0));

	// BFS,以当前点为中心，向四周搜索可以扩展的点（值为 0 即为可扩展）
	while (!Q.empty()){
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

void ConsistentPixelSetPyramid::findConnectedComponents(vector<ConnectedComponent> & connComps, Mat & undecidedPixels){
	int idx = 0;
	for (int y = 0; y < undecidedPixels.rows; y++)
	for (int x = 0; x < undecidedPixels.cols; x++)
	if (undecidedPixels.at<uchar>(y, x) == 255){
		connComps.push_back(floodFill(++idx, x, y, undecidedPixels));
	}
}

// 计算layer层的consistent pixels set
void ConsistentPixelSetPyramid::calConsistentPixelSet(int layer, vector<Mat> & integralImageSet, Mat & integralMedianImg, const int thre){
	Mat refConsistPixelSet, medConsistPixelSet, consistPixelSet;   // 记录consistent pixel
	Mat reliablePixelSet; // 记录reliable pixels

	// 初始化consistent pixel set
	refConsistPixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8UC(FRAME_NUM));
	medConsistPixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8UC(FRAME_NUM));
	consistPixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8UC(FRAME_NUM));
	reliablePixelSet = Mat::zeros(integralMedianImg.rows - 1, integralMedianImg.cols - 1, CV_8U);
	Mat undecidedPixels(consistPixelSet.rows, consistPixelSet.cols, CV_8U, Scalar::all(0));

	// 计算reference-based 和 median-based consistent pixels
	for (int r = 0; r < integralMedianImg.rows - 1; r++){   // 积分图比普通图行列各多一列
		for (int c = 0; c < integralMedianImg.cols - 1; c++){

			// 算参数
			int half = 2;
			int startR = max(0, r - half);
			int endR = min(integralMedianImg.rows - 2, r + half);
			int startC = max(0, c - half);
			int endC = min(integralMedianImg.cols - 2, c + half);
			int pixelNum = (endR - startR + 1) * (endC - startC + 1);

			/* -----计算reference-based consistent pixels----- */
			// 先计算ref image的5*5块像素
			int pixelRefSum = integralImageSet[REF].at<int>(endR + 1, endC + 1) - integralImageSet[REF].at<int>(startR, endC + 1)
				- integralImageSet[REF].at<int>(endR + 1, startC) + integralImageSet[REF].at<int>(startR, startC);
			int aveRefPixel = pixelRefSum / pixelNum;

			// 先把ref Image的该点标记为consistent pixel
			Vec<uchar, FRAME_NUM> & elem = refConsistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c);
			elem[REF] = 1;

			// 从ref开始往0计算每一帧
			for (int i = REF - 1; i >= 0; i--){
				int pixelSum = integralImageSet[i].at<int>(endR + 1, endC + 1) - integralImageSet[i].at<int>(startR, endC + 1)
					- integralImageSet[i].at<int>(endR + 1, startC) + integralImageSet[i].at<int>(startR, startC);
				int avePixel = pixelSum / pixelNum;
				if (abs(aveRefPixel - avePixel) < thre)
					elem[i] = 1;
				else
					break;
			}

			// 从ref开始往右计算每一帧
			for (int i = REF + 1; i < FRAME_NUM; i++){
				int pixelSum = integralImageSet[i].at<int>(endR + 1, endC + 1) - integralImageSet[i].at<int>(startR, endC + 1)
					- integralImageSet[i].at<int>(endR + 1, startC) + integralImageSet[i].at<int>(startR, startC);
				int avePixel = pixelSum / pixelNum;
				if (abs(aveRefPixel - avePixel) < thre)
					elem[i] = 1;
				else
					break;
			}

			/* -----计算median-based consistent pixels----- */
			// 计算median image的5*5块像素
			int pixelMedSum = integralMedianImg.at<int>(endR + 1, endC + 1) - integralMedianImg.at<int>(startR, endC + 1)
				- integralMedianImg.at<int>(endR + 1, startC) + integralMedianImg.at<int>(startR, startC);
			int aveMedPixel = pixelMedSum / pixelNum;

			Vec<uchar, FRAME_NUM> & medElem = medConsistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c);
			int cnt = 0; // 记录有多少个median-based consistent pixels
			for (int i = 0; i < FRAME_NUM; i++){
				int pixelSum = integralImageSet[i].at<int>(endR + 1, endC + 1) - integralImageSet[i].at<int>(startR, endC + 1)
					- integralImageSet[i].at<int>(endR + 1, startC) + integralImageSet[i].at<int>(startR, startC);
				int avePixel = pixelSum / pixelNum;
				if (abs(aveMedPixel - avePixel) < thre){
					medElem[i] = 1;
					cnt++;
				}
			}

			/* -----结合reference-based 和 median-based 的结果----- */
			// 如果ref frame属于median-based consistent pixels, 那么取M和R的并集				
			Vec<uchar, FRAME_NUM> & finalElem = consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c);
			if (medElem[REF] == 1){
				for (int i = 0; i < FRAME_NUM; i++){
					finalElem[i] = elem[i] | medElem[i];
				}
			}
			else undecidedPixels.at<uchar>(r, c) = 255;

			// 统计reliable pixels
			if (cnt > REF) reliablePixelSet.at<uchar>(r, c) = 1;
		}
	}

	/*// 显示undecidedPixels
	namedWindow("src", WINDOW_NORMAL);
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
	for (unsigned int i = 0; i < connComps.size(); i++){
		vector<Point> & CCpts = connComps[i].getCCPts();
		unsigned int cnt = 0; // 统计连通分量中有多少个reliable pixel

		for (unsigned int j = 0; j < CCpts.size(); j++){
			if (reliablePixelSet.at<uchar>(CCpts[j]) == 1) cnt++;
		}

		// 如果reliable pixel多，则整个连通分量都当作reliable处理，取M的结果
		if (cnt >= CCpts.size() - cnt){
			for (unsigned int j = 0; j < CCpts.size(); j++){
				Vec<uchar, FRAME_NUM> & finalElem = consistPixelSet.at<Vec<uchar, FRAME_NUM> >(CCpts[j]);
				Vec<uchar, FRAME_NUM> & medElem = medConsistPixelSet.at<Vec<uchar, FRAME_NUM> >(CCpts[j]);
				finalElem = medElem;
			}
		}
		// 否则整个CC都当作unreliable处理，取R的结果
		else{
			for (unsigned int j = 0; j < CCpts.size(); j++){
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

	for (unsigned int i = 0; i < channels.size(); i++){
		morphMajority(channels[i], channels[i]);
	}
	merge(channels, consistPixelSet);

	// 将某点全为0的consistPixelSet的ref那一帧置1，防止形态学过滤后，某点无consistent pixels的情况
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

// 由上下采样得到所有层的consistent pixels
void ConsistentPixelSetPyramid::calConsistentPixelsAllLayer(vector<Mat> & refPyramid){
	for (unsigned int layer = 0; layer < consistentPixels.size(); layer++){
		if (layer == CONSIST_LAYER) continue;
		resize(consistentPixels[CONSIST_LAYER], consistentPixels[layer], refPyramid[layer].size(), 0, 0, CV_INTER_LINEAR);  // CV_NEAREST
	}
}
vector<Mat> & ConsistentPixelSetPyramid::getConsistentPixelPyramid(){
	return consistentPixels;
}