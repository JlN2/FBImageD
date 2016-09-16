#ifndef CONSTDEF
#define CONSTDEF

#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <stdio.h>
#include <opencv2\core\core.hpp>     // ��Mat��������
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
const int FEATURE_LAYER = 0;   // Ӧ�ô���һ�㿪ʼ������������starting from a global homography at the coarsest level
const int CONSIST_LAYER = 0;

#endif