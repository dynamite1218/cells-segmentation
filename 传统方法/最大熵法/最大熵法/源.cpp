#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <iterator>
#include <algorithm>
using namespace std;
using namespace cv;

Mat EntropySeg(Mat src);

int main(int argc, char* argv[])
{
	Mat srcImg = imread("9.jpg");
	if (!srcImg.data)
	{
		cout << "图片未找到" << endl;
		return -1;
	}
	namedWindow("img", WINDOW_NORMAL);
	imshow("img", srcImg);
	cvResizeWindow("img", 500, 500); //创建一个500*500大小的窗口
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
	//Rect m_select;
	//m_select = Rect(0, 0, 1500, 1500);
	//dstImg = dstImg(m_select);     //截取部分图像
	Mat src_gray;
	cvtColor(dstImg, src_gray, CV_RGB2GRAY);//灰度转换
	Mat HistImg;
	equalizeHist(src_gray, HistImg);
	Mat medImg;
	medianBlur(HistImg, medImg, 9);
	Mat grad_x, grad_y;						  //Sobel算子锐化
	Mat abs_grad_x, abs_grad_y;
	Mat SobelImg;
	Sobel(medImg, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(medImg, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, SobelImg);
	medImg = medImg - SobelImg;
	Mat KSWImg;
	KSWImg = EntropySeg(medImg);
	Mat openImg, closeImg;
	Mat element = getStructuringElement(MORPH_RECT, Size(6, 6));
	morphologyEx(KSWImg, openImg, MORPH_OPEN, element);  //开运算
	morphologyEx(openImg, closeImg, MORPH_CLOSE, element);  //闭运算
	// 连通区域计数
	vector<vector<Point>> contours;
	findContours(closeImg, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point> > contours_poly(contours.size()); //近似后的轮廓点集
	vector<Rect> boundRect(contours.size()); //包围点集的最小矩形vector
	vector<RotatedRect> box(contours.size()); //定义最小外接矩形集合
	//Mat markers = Mat::zeros(srcImg.size(), CV_8UC3);
	for (size_t t = 0; t < contours.size(); t++) {
		approxPolyDP(Mat(contours[t]), contours_poly[t], 3, true);      //对多边形曲线做适当近似，contours_poly[i]是输出的近似点集
		boundRect[t] = boundingRect(Mat(contours_poly[t]));         //计算并返回包围轮廓点集的最小矩形
		box[t] = minAreaRect(Mat(contours_poly[t]));  //计算每个轮廓最小外接矩形
	}
	// 画包围的矩形框
	int j = 0;
	Scalar color = Scalar(0, 0, 255);
	for (int i = 0; i< contours.size(); i++)
	{
		if (box[i].size.height / box[i].size.width < 4 && contourArea(contours[i]) < 50000 && contourArea(contours[i]) > 760 && box[i].size.height * box[i].size.width < contourArea(contours[i]) * 2)
		{
			rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 5, 10, 0);              //画矩形，tl矩形左上角，br右上角
			j++;
		}
		//if (contourArea(contours[i]) > 760)
		//{
		//	rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 5, 10, 0);              //画矩形，tl矩形左上角，br右上角
		//	j++;
		//}
	}
	cout << "细胞总数" << j << endl;
	/// 显示在一个窗口
	namedWindow("Contours", WINDOW_NORMAL);
	imshow("Contours", srcImg);
	cvResizeWindow("Contours", 800, 800); //创建一个500*500大小的窗口
	imwrite("Contours.jpg", srcImg);
	waitKey(0);
	return 0;
}

/***************************************************************************************
Function: 最大熵分割算法
Input:    Mat 待分割的原图像
Output:   分割后图像
***************************************************************************************/
Mat EntropySeg(Mat src)
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
	threshold(src, dst, index, 255, 0);             //进行阈值分割
	return dst.clone();
}