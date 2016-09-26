#ifndef CONSTDEF
#define CONSTDEF

#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <stdio.h>
#include <opencv2\core\core.hpp>     // 有Mat数据类型
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <math.h>
#define M_PI 3.141592653589793238

using namespace std;
using namespace cv;

const string fileDir  = "SrcData\\Data\\Bookshelf_2\\";
const string resultDir = "Result\\";
const string imageFormat = ".jpg";
const int FRAME_NUM = 10;
const int REF = FRAME_NUM / 2;
const int FEATURE_LAYER = 1;   // 应该从哪一层开始算特征向量：starting from a global homography at the coarsest level
const int CONSIST_LAYER = 0;
const double GOOD_MATCH_THRE = 0.38;
const int CONSIST_THRE = 15;        // 阈值
const double LAMBDA = 0.1;

#endif