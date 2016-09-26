#include "S1_ImageNode.h"
#include "S2_PyramidLayer.h"
#include "S3_Pyramid.h"
#include "S4_ConsistentPixelSetPyramid.h"

class FastBurstImagesDenoising{
public: 
	vector<Mat> oriImageSet;                       // 存储原来的每帧图片
	vector<Pyramid*> imagePyramidSet;              // 图片金字塔（高斯金字塔）
	Pyramid* refPyramid;                           // 参考图片的金字塔
	ConsistentPixelSetPyramid consistPixelPyramid; // 存储consistent pixel
	Mat grayMedianImg;                             // CONSIST_LAYER的中位图（灰度图） 
	vector<Mat> temporalResult;                    // temporal fusion的结果图

	Mat refConsistPixelSet, medConsistPixelSet, consistPixelSet;   // 记录consistent pixel

	double noiseVar;                               // 噪声方差


	/* 在第三步用到：得到row*col*3 * 10 的consistPixelsChannels（将每帧的consist map分开，并变成3通道（rgb）） */
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
		/*-------*/PyramidLayer* refpLayer = refPyramid->getPyramidLayer(FEATURE_LAYER);/*-------*/
		/*
		int layersNum = refPyramid->getImagePyramid().size();
		PyramidLayer* refpLayer = refPyramid->getPyramidLayer(layersNum - 1);
		*/
		Mat refDescriptor = refpLayer->getImageDescriptors();  
		vector<KeyPoint> & refKpoint = refpLayer->getKeypoints();

		// BruteForce和FlannBased是opencv二维特征点匹配常用的两种办法，BF找最佳，比较暴力，Flann快，找近似，但是uchar类型描述子（BRIEF）只能用BF
		//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce" ); 
		Ptr<DescriptorMatcher> matcher = new BruteForceMatcher<L2<float>>;
		
		for(int frame = 0; frame < FRAME_NUM; frame++){

			if(frame == REF) continue;
			// 1. 计算每一帧->参考帧的Homography（3*3矩阵） 金字塔
			/*-------*/
			// 计算当前帧（最粗糙层）与参考帧的匹配特征点
			Pyramid* curPyramid = imagePyramidSet[frame]; // 当前图片金字塔
			PyramidLayer* curFrame = curPyramid->getPyramidLayer(FEATURE_LAYER);
			curFrame->calMatchPtsWithRef(refDescriptor, refKpoint, matcher);
	
			// 画出匹配结果
			//curFrame->showMatchedImg(curFrame->getImage(), refKpoint);

			// 计算每一层的特征点（将coarse level的特征点scale到其他层）
			curPyramid->calFeaturePyramid();

			// 将每一层的特征点分配到每个ImageNode
			curPyramid->distributeFeaturePtsByLayer();
			/*-------*/

			/*
			// 计算当前帧（原图）与参考帧的匹配特征点
			Pyramid* curPyramid = imagePyramidSet[frame]; // 当前图片金字塔
			PyramidLayer* curFrame = curPyramid->getPyramidLayer(layersNum - 1);
			curFrame->calMatchPtsWithRef(refDescriptor, refKpoint, matcher);

			// 将原图这一层的特征点传给其它层
			curPyramid->calFeaturePyramid1();

			// 将特征点分配到ImageNode
			curPyramid->distributeFeaturePtsByLayer1(curPyramid->getImagePyramid()[layersNum - 1].rows, curPyramid->getImagePyramid()[layersNum - 1].cols);
			*/

			// 计算每一帧（除参考帧）的homography金字塔
			curPyramid->calHomographyPyramid(); 


			// 2. 计算每一帧（除参考帧）的homography flow金字塔
			curPyramid->calHomographyFlowPyramid();


			cout << endl;
		}
	}

	// 第二步：选择consistent pixel 
	void consistentPixelSelection(){
		cout << "Consistent Pixel Selection Start. " << endl;
		vector<Mat> integralImageSet;    // 所有Consistent Image的积分图
		vector<Mat> consistGrayImageSet; // 所有Consistent Image的灰度图
		

		// 取出refImage
		PyramidLayer* refpLayer = refPyramid->getPyramidLayer(CONSIST_LAYER);
		Mat refImage = refpLayer->getImage();

		integralImageSet.resize(FRAME_NUM);    // 所有Consistent Image的积分图
		consistGrayImageSet.resize(FRAME_NUM); // 所有Consistent Image的灰度图

		for(int frame = 0; frame < FRAME_NUM; frame++){
			if(frame == REF){
				cvtColor(refImage, consistGrayImageSet[frame], CV_RGB2GRAY);
				integral(consistGrayImageSet[frame], integralImageSet[frame], CV_32S);
				// integral(refImage, integralImageSet[frame], CV_32SC3);  如果不转灰度就算积分图，用这个
				continue;
			}

			// 将图片用homography flow调整成一个consistent image(即和参考帧一致)(CONSIST_LAYER)
			Pyramid* curPyramid = imagePyramidSet[frame]; // 当前图片金字塔
			PyramidLayer* curFrame = curPyramid->getPyramidLayer(CONSIST_LAYER);
			curFrame->calConsistImage();
			
			// 转灰度图
			Mat consistImg = curFrame->getConsistImage();
			char index[10];
			sprintf_s(index, "%d", frame);
			imwrite(resultDir + "Consistent Img " + (string)index + imageFormat, consistImg); 
			cvtColor(consistImg, consistGrayImageSet[frame], CV_RGB2GRAY);

			// 求所有consistent 灰度图的积分图(原图行列各加1，第一行第一列均为0）
			integral(consistGrayImageSet[frame], integralImageSet[frame], CV_32S);
			//integral(consistImage, integralImageSet[frame], CV_32SC3);  // 如果不转灰度就算积分图，用这个
		}

		// 求median图(灰度图)和积分图
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

		// 初始化Consistent pixel pyramid
		consistPixelPyramid = ConsistentPixelSetPyramid(refPyramid->getImagePyramid().size());

		// 算Consistent Pixels(CONSIST_LAYER的)
		consistPixelPyramid.calConsistentPixelSet(CONSIST_LAYER, integralImageSet, integralMedianImg, CONSIST_THRE);

		// reuse the indices of computed consistent pixels by upsampling and downsampling，把refPyramid传进去是为了知道每层的尺寸
		consistPixelPyramid.calConsistentPixelsAllLayer(refPyramid->getImagePyramid());

		cout << "Consistent Pixel Selection End. " << endl;
	}

	

	// 第三步：融合得到最后的去噪图像
	void pixelsFusion(){

		vector<Mat> refImagePyramid = refPyramid->getImagePyramid();
		int layersNum = refImagePyramid.size();

		vector<Mat> numOfInliersAllLayer;    // multi-scale fusion时要用到
		numOfInliersAllLayer.resize(layersNum);

		/* -----计算噪声方差----- */
		// 取出ref image并转成灰度图像
		Mat refGrayImg;
		cvtColor(refImagePyramid[layersNum - 1], refGrayImg, CV_RGB2GRAY);

		// 取得中位图（灰度）
		Mat medGrayImg;  
		resize(grayMedianImg, medGrayImg, refGrayImg.size(), 0, 0, CV_INTER_LINEAR);  // grayMedianImg是CONSIST_LAYER的
		
		// 边缘提取
		Mat edgeImg;
		Canny(grayMedianImg, edgeImg, 50, 125, 3);   // canny边缘检测采用双阈值值法，高阈值用来检测图像中重要的、显著的线条、轮廓等，而低阈值用来保证不丢失细节部分
		resize(edgeImg, edgeImg, refGrayImg.size(), 0, 0, CV_INTER_NN);
		//imshow("edge", edgeImg);
		//waitKey(0);

		// 求平坦区域和平坦区域的像素数
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

		// 求med和ref之差的均值和方差（平坦区域）
		Mat diffImg = medGrayImg - refGrayImg;
		double aveNum = sum(diffImg)[0]/ cnt;
		for(int r = 0; r < refGrayImg.rows; r++)
			for(int c = 0; c < refGrayImg.cols; c++){
				if(edgeImg.at<uchar>(r, c) == 1) diffImg.at<double>(r, c) -= aveNum;
				
			}
		diffImg = diffImg.mul(diffImg);
		noiseVar = sum(diffImg)[0] / cnt;  // sigma2
		

		/* -----进行temporal fusion----- */
		cout << "Temporal Fusion Start. " << endl;
		vector<Mat> & consistentPixelPyramid = consistPixelPyramid.getConsistentPixelPyramid();
		for(int layer = 0; layer < layersNum; layer++){
			
			// 取出这一层的consistent pixels集
			consistPixelSet = consistentPixelPyramid[layer];

			vector<Mat> consistImgSet;
			Mat consistGrayImgSet(consistPixelSet.size(), CV_8UC(FRAME_NUM), Scalar::all(0));
			Mat meanImg(consistPixelSet.size(), CV_8U, Scalar::all(0));
			Mat meanRGBImg(consistPixelSet.size(), CV_8UC3, Scalar::all(0));
			Mat tVar(consistPixelSet.size(), CV_64F, Scalar::all(0));
			Mat var(consistPixelSet.size(), CV_64F, Scalar::all(0));

			// 得到consist gray image set (col * row * 10) 和 consistImgSet(vector, col * row * 3 * 10) 彩色
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
			
			// 点乘，将不是consistent pixel的地方置零
			consistGrayImgSet = consistGrayImgSet.mul(consistPixelSet);

			/*vector<Mat> chan;
			split(consistGrayImgSet, chan);
			for(int i = 0; i < chan.size(); i++){
				imshow("g", chan[i]);
				waitKey(0);
			}*/

			for(int r = 0; r < consistPixelSet.rows; r++)
				for(int c = 0; c < consistPixelSet.cols; c++){
					// 计算平均图像（灰度）
					meanImg.at<uchar>(r, c) = sum(consistGrayImgSet.at<Vec<uchar, FRAME_NUM> >(r, c))[0] 
						/ sum(consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c))[0];
					//cout << "Num of consistent pixels: " << sum(consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c))[0] << endl;

					// 计算sigmat方，即每个像素点consistent pixels的方差
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
					
					// 计算sigmac方
					var.at<double>(r, c) = max((double)0, tVar.at<double>(r, c) - noiseVar);

					// 计算sigmac2/(sigmac2 + sigma2)
					var.at<double>(r, c) = var.at<double>(r, c) / (var.at<double>(r, c) + noiseVar);
				}
				
			// 得到row*col*3 * 10 的consistPixelsChannels（将每帧的consist map分开，并变成3通道（rgb））
			tempConsistPixelsChannels.clear();             // row*col*1 * 10
			consistPixelsChannels.clear();
			consistPixelsChannels.resize(FRAME_NUM);                 // row*col*3 * 10
			Mat sumImg(consistPixelSet.size(), CV_32SC3, Scalar::all(0));
			Mat sumConsistImg(consistPixelSet.size(), CV_8UC3, Scalar::all(0));
			split(consistPixelSet, tempConsistPixelsChannels); // 将consistPixelSet 10个通道分开
			
			for(int frame = 0; frame < FRAME_NUM; frame++){
				
				tempChannels.clear();
				for(int i = 0; i < 3; i++) tempChannels.push_back(tempConsistPixelsChannels[frame]);
				merge(tempChannels, consistPixelsChannels[frame]);

				// 求sumImg
				consistImgSet[frame] = consistImgSet[frame].mul(consistPixelsChannels[frame]);
				add(sumImg, consistImgSet[frame], sumImg, Mat(), CV_32SC3); // sumImg(CV_32SC3) += consistImgSet[frame](CV_8UC3); 类型不同所以改成了左边那种
				sumConsistImg += consistPixelsChannels[frame];
			}

			// 求meanRGBImg
			divide(sumImg, sumConsistImg, meanRGBImg, 1, CV_8UC3);  //meanRGBImg = sumImg / sumConsistImg;  点除
			//imshow("mean", meanRGBImg);
			//waitKey(0);

			// 求融合后的结果
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

			// 计算number of inliers （multi-scale fusion时要用到）
			if(layer != 0){
				calNumOfInlierPerPixel(numOfInliersAllLayer[layer], tVar, consistGrayImgSet, layer);
			}
			
		}
		cout << "Temporal Fusion End. " << endl;
		imwrite(resultDir + "temporal" + imageFormat, temporalResult[layersNum - 1]);

		/* -----进行multi-scale fusion----- */
		cout << "Multi-scale Fusion Start. " << endl;
		for(int layer = 1; layer < layersNum; layer++){
			// 通过bilinear upscale来scale上一层的图像
			Mat formerLayerImg;
			resize(temporalResult[layer - 1], formerLayerImg, temporalResult[layer].size(), 0, 0, CV_INTER_LINEAR);
			formerLayerImg.convertTo(formerLayerImg, CV_32FC3);

			// 计算textureness probalitity ptex
			Mat pTex;
			calPtex(pTex, layer);

			// 计算omega w = sqrt(m/FRAME_NUM)
			Mat omega;
			calOmega(omega, numOfInliersAllLayer[layer], layer);

			// 标记那些ptex > 0.01 的点
			Mat isTex(pTex.size(), CV_8U);
			for(int r = 0; r < pTex.rows; r++)
				for(int c = 0; c < pTex.cols; c++){
					isTex.at<uchar>(r, c) = (pTex.at<float>(r, c) > 0.01) * 255;
				}
			//imshow("texture", isTex);
			//waitKey(0);
			
			// 计算f(xs) directional spatial pixel fusion （只在ptex>0.01的点进行）
			Mat spatialFusion = temporalResult[layer].clone();
			calSpatialFusion(spatialFusion, isTex);

			// 替换temporal fusion result为ptex * f(xs) + (1 - ptex) * formerLayer
			// 再替换为 w * xs + (1 - w) * formerLayer，得到图像融合的结果
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
		cout << "Multi-scale Fusion End. " << endl;
		//imwrite(resultDir + "temporalandmultiscale1" + imageFormat, temporalResult[0]);
		imwrite(resultDir + "temporalandmultiscale" + imageFormat, temporalResult[layersNum - 1]);
		
	}

	void getSpatialInfo(Vec<int, 5> & spatialPts, Vec3f & sum, int deltar[], int deltac[], int r, int c, Mat & spatialFusion, Mat & graySpatialFusion){
		for(int i = 0; i < 5; i++){
			sum += spatialFusion.at<Vec3f>(r+deltar[i], c+deltac[i]);
			spatialPts[i] = (int)graySpatialFusion.at<uchar>(r+deltar[i], c+deltac[i]);
		}
	}
	
	// 进行Spatial Fusion
	void calSpatialFusion(Mat & spatialFusion, Mat & isTex){
		Mat graySpatialFusion;
		cvtColor(spatialFusion, graySpatialFusion, CV_RGB2GRAY);
		spatialFusion.convertTo(spatialFusion, CV_32FC3);

		// 计算梯度，找出most probable edge
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

					Scalar m, sd; // 均值和标准差
					meanStdDev(spatialPts, m, sd);
					double var = max((double)0, sd[0] * sd[0] - noiseVar);
					var = var / (var + noiseVar);

					spatialFusion.at<Vec3f>(r, c) = sum / 5 + var * (spatialFusion.at<Vec3f>(r, c) - sum / 5);

				}
			}
	}

	// 计算layer层的temporal fusion结果的每一个像素内点的个数   |xt - x^| < 3sigmat
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

	// 计算layer层的temporal fusion结果的omega w = sqrt(m/FRAME_NUM)
	void calOmega(Mat & omega, Mat & numOfInlier, int layer){
		omega = Mat::zeros(temporalResult[layer].size(), CV_32F);

		for(int r = 0; r < omega.rows; r++)
			for(int c = 0; c < omega.cols; c++){
				omega.at<float>(r, c) = sqrt((float)numOfInlier.at<uchar>(r, c) / (float)FRAME_NUM);
			}
	}

	// 计算layer层的temporal fusion结果的textureness probalitity Ptex
	void calPtex(Mat & pTex, int layer){
		
		Mat temporalImg = temporalResult[layer];
		pTex = Mat::zeros(temporalImg.size(), CV_32F);

		Mat temporalGrayImg;
		cvtColor(temporalImg, temporalGrayImg, CV_RGB2GRAY);
		int dx[] = { -1, 0, 1, 0 };
 		int dy[] = { 0, 1, 0, -1 };
		for(int y = 0; y < temporalImg.rows; y++)
			for(int x = 0; x < temporalImg.cols; x++){
				// 求max absolute difference
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
	FBID.consistentPixelSelection();
	FBID.pixelsFusion();
	FBID.showImages(FBID.temporalResult);

	system("pause");

	return 0;
}