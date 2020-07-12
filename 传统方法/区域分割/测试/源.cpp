#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <iterator>
#include <fstream>
#include <algorithm>
using namespace std;
using namespace cv;
void onmouse(int event, int x, int y, int flags, void* param);
int EntropySeg(Mat src);
Mat RegionGrow(Mat src, Point2i pt, int th);

vector<Point2i> mypoint;
int main(int argc, char* argv[])
{
	Mat srcImg = imread("9.jpg");
	namedWindow("img", WINDOW_NORMAL);
	cvResizeWindow("img", 800, 800); //创建一个800*800大小的窗口

	/*去除下面字体*/
	Mat dstImg;
	dstImg = srcImg.clone();
	for (int row = srcImg.rows - 100; row < srcImg.rows; row++)
	{
		for (int col = 0; col < srcImg.cols; col++)
		{
			dstImg.at<Vec3b>(row, col)[0] = 0;
			dstImg.at<Vec3b>(row, col)[1] = 0;
			dstImg.at<Vec3b>(row, col)[2] = 0;
		}
	}
	Mat src_gray;
	cvtColor(dstImg, src_gray, CV_RGB2GRAY);//灰度转换
	Mat HistImg;
	equalizeHist(src_gray, HistImg);
	Mat GaussianImg;
	GaussianBlur(HistImg, GaussianImg, Size(5, 5), 1);
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(GaussianImg, GaussianImg, MORPH_OPEN, element);  //开运算
	//namedWindow("GaussianImg", WINDOW_NORMAL);
	//imshow("GaussianImg", GaussianImg);
	//cvResizeWindow("GaussianImg", 500, 500); //创建一个500*500大小的窗口
	//Sobel算子
	Mat SobelImg;
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Sobel(GaussianImg, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(GaussianImg, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, SobelImg);
	GaussianImg = GaussianImg + SobelImg;
	//imwrite("GaussianImg.jpg", GaussianImg);
	//namedWindow("Sobel", WINDOW_NORMAL);
	//imshow("Sobel", GaussianImg);
	//cvResizeWindow("Sobel", 500, 500); //创建一个500*500大小的窗口
	int th;
	th = EntropySeg(GaussianImg);
	Mat Growing;
	Mat Growing2(Size(srcImg.cols, srcImg.rows),CV_8UC1,Scalar(0,0,0));
	Point2i pt;
	namedWindow("Growing2", WINDOW_NORMAL);
	cvResizeWindow("Growing2", 500, 500); //创建一个500*500大小的窗口
	setMouseCallback("img", onmouse, &GaussianImg);//注册鼠标事件到“img”窗口，即使在该窗口下出现鼠标事件就执行onmouse函数的内容
//开始取点循环
	while (1)
	{
		imshow("img", GaussianImg);
		imwrite("img.jpg", GaussianImg);
		while (!mypoint.empty())
		{
			pt = mypoint.back();   //取出一个生长点进行生长
			mypoint.pop_back();	//弹出该生长点
			Growing = RegionGrow(GaussianImg, pt, th-40);
			morphologyEx(Growing, Growing, MORPH_OPEN, element);  //开运算
			morphologyEx(Growing, Growing, MORPH_CLOSE, element);  //闭运算
			bitwise_or(Growing, Growing2, Growing2);
		}
		imshow("Growing2", Growing2);
		imwrite("Growing2.jpg", Growing2);
		if (waitKey(100) == 27)  //按下Esc跳出
			break;
	}
	// draw result
	vector<vector<Point>> contours;
	findContours(Growing2, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<double> area; //用于计算细胞面积
	vector<vector<Point> > contours_poly(contours.size()); //近似后的轮廓点集
	vector<Rect> boundRect(contours.size()); //包围点集的最小矩形vector
	vector<RotatedRect> box(contours.size()); //定义最小外接矩形集合
	Point2f vertices[4];

	for (size_t t = 0; t < contours.size(); t++) {
		approxPolyDP(Mat(contours[t]), contours_poly[t], 3, true);      //对多边形曲线做适当近似，contours_poly[i]是输出的近似点集
		boundRect[t] = boundingRect(Mat(contours_poly[t]));         //计算并返回包围轮廓点集的最小矩形
		box[t] = minAreaRect(Mat(contours_poly[t]));  //计算每个轮廓最小外接矩形
		box[t].points(vertices);//获取矩形的四个点
		area.push_back(contourArea(contours[t]));
	}
	ofstream outfile;//输出数据
	outfile.open("1.csv", ios::out | ios::trunc);
	outfile << "长" << "," << "宽" << ","  << "面积" << endl;
	// 画包围的矩形框
	double L, W;  //定义长宽
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(0, 0, 255);
		rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 10, 10, 0);              //画矩形，tl矩形左上角，br右上角
		//for (int i = 0; i < 4; i++)
		//	line(srcImg, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255));	//画最小矩形
		if (box[i].size.height>box[i].size.width)
		{
			L = box[i].size.height;
			W = box[i].size.width;
		}
		else
		{
			W = box[i].size.height;
			L = box[i].size.width;
		}
		outfile << L << "," << W  << "," << area[i] << endl;//输出最小矩形长宽面积
	}
	/// 显示在一个窗口
	namedWindow("Contours", WINDOW_NORMAL);
	imshow("Contours", srcImg);
	cvResizeWindow("Contours", 500, 500); //创建一个500*500大小的窗口
	imwrite("Contours.jpg", srcImg);
	waitKey(0);
	return 0;
}


void onmouse(int event, int x, int y, int flags, void* param)//鼠标事件回调函数，鼠标点击后执行的内容应在此
{
	Mat img = *(Mat*)param;
	Point cent;
	if (event == EVENT_LBUTTONDOWN)//鼠标左键按下事件
	{
		cent.x = x;
		cent.y = y;
		//circle(img, cent, 10, Scalar(255, 255, 0), 8);
		cout <<  x << " " << y << endl;
		mypoint.push_back(cent);
	}
}

int EntropySeg(Mat src)
{
	int tbHist[256] = { 0 };                                          //每个像素值个数
	int index = 0;                                                  //最大熵对应的灰度
	double Property = 0.0;                                          //像素所占概率
	double maxEntropy = -1.0;                                       //最大熵
	double frontEntropy = 0.0;                                      //前景熵
	double backEntropy = 0.0;                                       //背景熵
	//纳入计算的总像素数
	int TotalPixel = 0;
	int nCol = src.cols * src.channels();                           //每行的像素个数
	for (int i = 0; i < src.rows; i++)
	{
		uchar* pData = src.ptr<uchar>(i);
		for (int j = 0; j < nCol; ++j)
		{
			++TotalPixel;
			tbHist[pData[j]] += 1;
		}
	}

	for (int i = 0; i < 256; i++)
	{
		//计算背景像素数
		double backTotal = 0;
		for (int j = 0; j < i; j++)
		{
			backTotal += tbHist[j];
		}

		//背景熵
		for (int j = 0; j < i; j++)
		{
			if (tbHist[j] != 0)
			{
				Property = tbHist[j] / backTotal;
				backEntropy += -Property * logf((float)Property);
			}
		}
		//前景熵
		for (int k = i; k < 256; k++)
		{
			if (tbHist[k] != 0)
			{
				Property = tbHist[k] / (TotalPixel - backTotal);
				frontEntropy += -Property * logf((float)Property);
			}
		}

		if (frontEntropy + backEntropy > maxEntropy)    //得到最大熵
		{
			maxEntropy = frontEntropy + backEntropy;
			index = i;
		}
		//清空本次计算熵值
		frontEntropy = 0.0;
		backEntropy = 0.0;
	}
	Mat dst;
	//index += 20;
	return index;
}

Mat RegionGrow(Mat src, Point2i pt, int th)
{
	Point2i ptGrowing;     //待生长点位置
	int nGrowLable = 0;		//标记是否生长过
	int nSrcValue = 0;		//生长起点灰度值
	int nCurValue = 0;      //当前生长点灰度值
	Mat matDst = Mat::zeros(src.size(), CV_8UC1); //创建一个空白区域，用于标记每一点是否被生长过
	//生长方向顺序数据
	int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };
	vector<Point2i> vcGrowPt;			//初始化生长点栈
	vcGrowPt.push_back(pt);
	matDst.at<uchar>(pt.y, pt.x) = 255;    //标记初始生长点
	nSrcValue = src.at<uchar>(pt.y, pt.x);

	while (!vcGrowPt.empty())
	{
		pt = vcGrowPt.back();   //取出一个生长点进行生长
		vcGrowPt.pop_back();	//弹出该生长点
		for (int i = 0; i < 9; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];    //按生长方向顺序进行生长
			ptGrowing.y = pt.y + DIR[i][1];
			if (ptGrowing.x<0 || ptGrowing.y<0 || ptGrowing.x>(src.cols - 1) || ptGrowing.y>(src.rows - 1))  //是否到达边缘点
				continue;
			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);//当前待生长点的灰度值
			if (nGrowLable == 0)						//如果没生长过
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (abs(nSrcValue - nCurValue) < th)			//如果该生长点在阈值内
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;
					vcGrowPt.push_back(ptGrowing);			//将该点作为生长点进行下次循环
				}
			}
		}
	}
	return matDst.clone();
}
