#include "S1_ImageNode.h"
#include "S2_PyramidLayer.h"
#include "S3_Pyramid.h"
#include "S4_ConsistentPixelSetPyramid.h"

class FastBurstImagesDenoising{
public: 
	vector<Mat> oriImageSet;                       // �洢ԭ����ÿ֡ͼƬ
	vector<Pyramid*> imagePyramidSet;              // ͼƬ����������˹��������
	Pyramid* refPyramid;                           // �ο�ͼƬ�Ľ�����
	ConsistentPixelSetPyramid consistPixelPyramid; // �洢consistent pixel
	Mat grayMedianImg;                             // CONSIST_LAYER����λͼ���Ҷ�ͼ�� 
	vector<Mat> temporalResult;                    // temporal fusion�Ľ��ͼ

	Mat refConsistPixelSet, medConsistPixelSet, consistPixelSet;   // ��¼consistent pixel

	double noiseVar;                               // ��������


	/* �ڵ������õ����õ�row*col*3 * 10 ��consistPixelsChannels����ÿ֡��consist map�ֿ��������3ͨ����rgb���� */
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
		/*PyramidLayer* refpLayer = refPyramid->getPyramidLayer(FEATURE_LAYER);*/
		/*-------*/
		int layersNum = refPyramid->getImagePyramid().size();
		PyramidLayer* refpLayer = refPyramid->getPyramidLayer(layersNum - 1);
		/*-------*/
		Mat refDescriptor = refpLayer->getImageDescriptors();  
		vector<KeyPoint> & refKpoint = refpLayer->getKeypoints();

		// BruteForce��FlannBased��opencv��ά������ƥ�䳣�õ����ְ취��BF����ѣ��Ƚϱ�����Flann�죬�ҽ��ƣ�����uchar���������ӣ�BRIEF��ֻ����BF
		//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "BruteForce" ); 
		Ptr<DescriptorMatcher> matcher = new BruteForceMatcher<L2<float>>;
		
		for(int frame = 0; frame < FRAME_NUM; frame++){

			if(frame == REF) continue;
			// 1. ����ÿһ֡->�ο�֡��Homography��3*3���� ������
			/* // ���㵱ǰ֡����ֲڲ㣩��ο�֡��ƥ��������
			Pyramid* curPyramid = imagePyramidSet[frame]; // ��ǰͼƬ������
			PyramidLayer* curFrame = curPyramid->getPyramidLayer(FEATURE_LAYER);
			curFrame->calMatchPtsWithRef(refDescriptor, refKpoint, matcher);
	
			// ����ƥ����
			//curFrame->showMatchedImg(curFrame->getImage(), refKpoint);

			// ����ÿһ��������㣨��coarse level��������scale�������㣩
			curPyramid->calFeaturePyramid();

			// ��ÿһ�����������䵽ÿ��ImageNode
			curPyramid->distributeFeaturePtsByLayer();*/

			/*-------*/
			// ���㵱ǰ֡��ԭͼ����ο�֡��ƥ��������
			Pyramid* curPyramid = imagePyramidSet[frame]; // ��ǰͼƬ������
			PyramidLayer* curFrame = curPyramid->getPyramidLayer(layersNum - 1);
			curFrame->calMatchPtsWithRef(refDescriptor, refKpoint, matcher);

			// ��ԭͼ��һ��������㴫��������
			curPyramid->calFeaturePyramid1();

			// ����������䵽ImageNode
			curPyramid->distributeFeaturePtsByLayer1(curPyramid->getImagePyramid()[layersNum - 1].rows, curPyramid->getImagePyramid()[layersNum - 1].cols);
			/*-------*/

			// ����ÿһ֡�����ο�֡����homography������
			curPyramid->calHomographyPyramid(); 


			// 2. ����ÿһ֡�����ο�֡����homography flow������
			curPyramid->calHomographyFlowPyramid();

			cout << endl;
		}
	}

	// �ڶ�����ѡ��consistent pixel 
	void consistentPixelSelection(){
		vector<Mat> integralImageSet;    // ����Consistent Image�Ļ���ͼ
		vector<Mat> consistGrayImageSet; // ����Consistent Image�ĻҶ�ͼ
		const int threshold = 10;        // ��ֵ

		// ȡ��refImage
		PyramidLayer* refpLayer = refPyramid->getPyramidLayer(CONSIST_LAYER);
		Mat refImage = refpLayer->getImage();

		integralImageSet.resize(FRAME_NUM);    // ����Consistent Image�Ļ���ͼ
		consistGrayImageSet.resize(FRAME_NUM); // ����Consistent Image�ĻҶ�ͼ

		for(int frame = 0; frame < FRAME_NUM; frame++){
			if(frame == REF){
				cvtColor(refImage, consistGrayImageSet[frame], CV_RGB2GRAY);
				integral(consistGrayImageSet[frame], integralImageSet[frame], CV_32S);
				// integral(refImage, integralImageSet[frame], CV_32SC3);  �����ת�ҶȾ������ͼ�������
				continue;
			}

			// ��ͼƬ��homography flow������һ��consistent image(���Ͳο�֡һ��)(CONSIST_LAYER)
			Pyramid* curPyramid = imagePyramidSet[frame]; // ��ǰͼƬ������
			PyramidLayer* curFrame = curPyramid->getPyramidLayer(CONSIST_LAYER);
			curFrame->calConsistImage();
			
			// ת�Ҷ�ͼ
			Mat consistImg = curFrame->getConsistImage();
			cvtColor(consistImg, consistGrayImageSet[frame], CV_RGB2GRAY);

			// ������consistent �Ҷ�ͼ�Ļ���ͼ(ԭͼ���и���1����һ�е�һ�о�Ϊ0��
			integral(consistGrayImageSet[frame], integralImageSet[frame], CV_32S);
			//integral(consistImage, integralImageSet[frame], CV_32SC3);  // �����ת�ҶȾ������ͼ�������
		}

		// ��medianͼ(�Ҷ�ͼ)�ͻ���ͼ
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

		// ��ʼ��Consistent pixel pyramid
		consistPixelPyramid = ConsistentPixelSetPyramid(refPyramid->getImagePyramid().size());

		// ��Consistent Pixels(CONSIST_LAYER��)
		consistPixelPyramid.calConsistentPixelSet(CONSIST_LAYER, integralImageSet, integralMedianImg, threshold);

		// reuse the indices of computed consistent pixels by upsampling and downsampling����refPyramid����ȥ��Ϊ��֪��ÿ��ĳߴ�
		consistPixelPyramid.calConsistentPixelsAllLayer(refPyramid->getImagePyramid());
	}

	

	// ���������ںϵõ�����ȥ��ͼ��
	void pixelsFusion(){

		vector<Mat> refImagePyramid = refPyramid->getImagePyramid();
		int layersNum = refImagePyramid.size();

		vector<Mat> numOfInliersAllLayer;    // multi-scale fusionʱҪ�õ�
		numOfInliersAllLayer.resize(layersNum);

		/* -----������������----- */
		// ȡ��ref image��ת�ɻҶ�ͼ��
		Mat refGrayImg;
		cvtColor(refImagePyramid[layersNum - 1], refGrayImg, CV_RGB2GRAY);

		// ȡ����λͼ���Ҷȣ�
		Mat medGrayImg;  
		resize(grayMedianImg, medGrayImg, refGrayImg.size(), 0, 0, CV_INTER_LINEAR);  // grayMedianImg��CONSIST_LAYER��
		
		// ��Ե��ȡ
		Mat edgeImg;
		Canny(grayMedianImg, edgeImg, 50, 125, 3);   // canny��Ե������˫��ֵֵ��������ֵ�������ͼ������Ҫ�ġ������������������ȣ�������ֵ������֤����ʧϸ�ڲ���
		resize(edgeImg, edgeImg, refGrayImg.size(), 0, 0, CV_INTER_NN);
		//imshow("edge", edgeImg);
		//waitKey(0);

		// ��ƽ̹�����ƽ̹�����������
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

		// ��med��ref֮��ľ�ֵ�ͷ��ƽ̹����
		Mat diffImg = medGrayImg - refGrayImg;
		double aveNum = sum(diffImg)[0]/ cnt;
		for(int r = 0; r < refGrayImg.rows; r++)
			for(int c = 0; c < refGrayImg.cols; c++){
				if(edgeImg.at<uchar>(r, c) == 1) diffImg.at<double>(r, c) -= aveNum;
				
			}
		diffImg = diffImg.mul(diffImg);
		noiseVar = sum(diffImg)[0] / cnt;  // sigma2
		

		/* -----����temporal fusion----- */
		vector<Mat> & consistentPixelPyramid = consistPixelPyramid.getConsistentPixelPyramid();
		for(int layer = 0; layer < layersNum; layer++){
			
			// ȡ����һ���consistent pixels��
			consistPixelSet = consistentPixelPyramid[layer];

			vector<Mat> consistImgSet;
			Mat consistGrayImgSet(consistPixelSet.size(), CV_8UC(FRAME_NUM), Scalar::all(0));
			Mat meanImg(consistPixelSet.size(), CV_8U, Scalar::all(0));
			Mat meanRGBImg(consistPixelSet.size(), CV_8UC3, Scalar::all(0));
			Mat tVar(consistPixelSet.size(), CV_64F, Scalar::all(0));
			Mat var(consistPixelSet.size(), CV_64F, Scalar::all(0));

			// �õ�consist gray image set (col * row * 10) �� consistImgSet(vector, col * row * 3 * 10) ��ɫ
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
			
			// ��ˣ�������consistent pixel�ĵط�����
			consistGrayImgSet = consistGrayImgSet.mul(consistPixelSet);

			/*vector<Mat> chan;
			split(consistGrayImgSet, chan);
			for(int i = 0; i < chan.size(); i++){
				imshow("g", chan[i]);
				waitKey(0);
			}*/

			for(int r = 0; r < consistPixelSet.rows; r++)
				for(int c = 0; c < consistPixelSet.cols; c++){
					// ����ƽ��ͼ�񣨻Ҷȣ�
					meanImg.at<uchar>(r, c) = sum(consistGrayImgSet.at<Vec<uchar, FRAME_NUM> >(r, c))[0] 
						/ sum(consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c))[0];
					//cout << "Num of consistent pixels: " << sum(consistPixelSet.at<Vec<uchar, FRAME_NUM>>(r, c))[0] << endl;

					// ����sigmat������ÿ�����ص�consistent pixels�ķ���
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
					
					// ����sigmac��
					var.at<double>(r, c) = max((double)0, tVar.at<double>(r, c) - noiseVar);

					// ����sigmac2/(sigmac2 + sigma2)
					var.at<double>(r, c) = var.at<double>(r, c) / (var.at<double>(r, c) + noiseVar);
				}
				
			// �õ�row*col*3 * 10 ��consistPixelsChannels����ÿ֡��consist map�ֿ��������3ͨ����rgb����
			tempConsistPixelsChannels.clear();             // row*col*1 * 10
			consistPixelsChannels.clear();
			consistPixelsChannels.resize(FRAME_NUM);                 // row*col*3 * 10
			Mat sumImg(consistPixelSet.size(), CV_32SC3, Scalar::all(0));
			Mat sumConsistImg(consistPixelSet.size(), CV_8UC3, Scalar::all(0));
			split(consistPixelSet, tempConsistPixelsChannels); // ��consistPixelSet 10��ͨ���ֿ�
			
			for(int frame = 0; frame < FRAME_NUM; frame++){
				
				tempChannels.clear();
				for(int i = 0; i < 3; i++) tempChannels.push_back(tempConsistPixelsChannels[frame]);
				merge(tempChannels, consistPixelsChannels[frame]);

				// ��sumImg
				consistImgSet[frame] = consistImgSet[frame].mul(consistPixelsChannels[frame]);
				add(sumImg, consistImgSet[frame], sumImg, Mat(), CV_32SC3); // sumImg(CV_32SC3) += consistImgSet[frame](CV_8UC3); ���Ͳ�ͬ���Ըĳ����������
				sumConsistImg += consistPixelsChannels[frame];
			}

			// ��meanRGBImg
			divide(sumImg, sumConsistImg, meanRGBImg, 1, CV_8UC3);  //meanRGBImg = sumImg / sumConsistImg;  ���
			//imshow("mean", meanRGBImg);
			//waitKey(0);

			// ���ںϺ�Ľ��
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

			// ����number of inliers ��multi-scale fusionʱҪ�õ���
			if(layer != 0){
				calNumOfInlierPerPixel(numOfInliersAllLayer[layer], tVar, consistGrayImgSet, layer);
			}
			
		}

		imwrite(resultDir + "temporal" + imageFormat, temporalResult[layersNum - 1]);

		/* -----����multi-scale fusion----- */
		for(int layer = 1; layer < layersNum; layer++){
			// ͨ��bilinear upscale��scale��һ���ͼ��
			Mat formerLayerImg;
			resize(temporalResult[layer - 1], formerLayerImg, temporalResult[layer].size(), 0, 0, CV_INTER_LINEAR);
			formerLayerImg.convertTo(formerLayerImg, CV_32FC3);

			// ����textureness probalitity ptex
			Mat pTex;
			calPtex(pTex, layer);

			// ����omega w = sqrt(m/FRAME_NUM)
			Mat omega;
			calOmega(omega, numOfInliersAllLayer[layer], layer);

			// �����Щptex > 0.01 �ĵ�
			Mat isTex(pTex.size(), CV_8U);
			for(int r = 0; r < pTex.rows; r++)
				for(int c = 0; c < pTex.cols; c++){
					isTex.at<uchar>(r, c) = (pTex.at<float>(r, c) > 0.01) * 255;
				}
			//imshow("texture", isTex);
			//waitKey(0);
			
			// ����f(xs) directional spatial pixel fusion ��ֻ��ptex>0.01�ĵ���У�
			Mat spatialFusion = temporalResult[layer].clone();
			calSpatialFusion(spatialFusion, isTex);

			// �滻temporal fusion resultΪptex * f(xs) + (1 - ptex) * formerLayer
			// ���滻Ϊ w * xs + (1 - w) * formerLayer���õ�ͼ���ںϵĽ��
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

		imwrite(resultDir + "temporalandmultiscale" + imageFormat, temporalResult[layersNum - 1]);
		
	}

	void getSpatialInfo(Vec<int, 5> & spatialPts, Vec3f & sum, int deltar[], int deltac[], int r, int c, Mat & spatialFusion, Mat & graySpatialFusion){
		for(int i = 0; i < 5; i++){
			sum += spatialFusion.at<Vec3f>(r+deltar[i], c+deltac[i]);
			spatialPts[i] = (int)graySpatialFusion.at<uchar>(r+deltar[i], c+deltac[i]);
		}
	}
	
	// ����Spatial Fusion
	void calSpatialFusion(Mat & spatialFusion, Mat & isTex){
		Mat graySpatialFusion;
		cvtColor(spatialFusion, graySpatialFusion, CV_RGB2GRAY);
		spatialFusion.convertTo(spatialFusion, CV_32FC3);

		// �����ݶȣ��ҳ�most probable edge
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

					Scalar m, sd; // ��ֵ�ͱ�׼��
					meanStdDev(spatialPts, m, sd);
					double var = max((double)0, sd[0] * sd[0] - noiseVar);
					var = var / (var + noiseVar);

					spatialFusion.at<Vec3f>(r, c) = sum / 5 + var * (spatialFusion.at<Vec3f>(r, c) - sum / 5);

				}
			}
	}

	// ����layer���temporal fusion�����ÿһ�������ڵ�ĸ���   |xt - x^| < 3sigmat
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

	// ����layer���temporal fusion�����omega w = sqrt(m/FRAME_NUM)
	void calOmega(Mat & omega, Mat & numOfInlier, int layer){
		omega = Mat::zeros(temporalResult[layer].size(), CV_32F);

		for(int r = 0; r < omega.rows; r++)
			for(int c = 0; c < omega.cols; c++){
				omega.at<float>(r, c) = sqrt((float)numOfInlier.at<uchar>(r, c) / (float)FRAME_NUM);
			}
	}

	// ����layer���temporal fusion�����textureness probalitity Ptex
	void calPtex(Mat & pTex, int layer){
		
		Mat temporalImg = temporalResult[layer];
		pTex = Mat::zeros(temporalImg.size(), CV_32F);

		Mat temporalGrayImg;
		cvtColor(temporalImg, temporalGrayImg, CV_RGB2GRAY);
		int dx[] = { -1, 0, 1, 0 };
 		int dy[] = { 0, 1, 0, -1 };
		for(int y = 0; y < temporalImg.rows; y++)
			for(int x = 0; x < temporalImg.cols; x++){
				// ��max absolute difference
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
	FBID.consistentPixelSelection();
	FBID.pixelsFusion();
	FBID.showImages(FBID.temporalResult);
	

	system("pause");

	return 0;
}