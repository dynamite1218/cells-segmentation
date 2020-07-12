#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <iterator>
#include <algorithm>
using namespace std;
using namespace cv;


//细胞识别主函数

int main(int argc, char* argv[])
{
	Mat srcImg = imread("1.jpg");
	namedWindow("src", WINDOW_NORMAL);
	imshow("src", srcImg);
	cvResizeWindow("src", 800, 800); //创建一个500*500大小的窗口
	if (!srcImg.data)
	{
		cout << "图片未找到" << endl;
		return -1;
	}
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
	Mat element = getStructuringElement(MORPH_RECT, Size(12, 12));
	//dilate(srcImg, dstImg, element, Point(-1, -1), 3);
	//erode(srcImg, dstImg, element, Point(-1, -1), 5);
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
	namedWindow("medImg", WINDOW_NORMAL);
	imshow("medImg", medImg);
	cvResizeWindow("medImg", 800, 800); //创建一个500*500大小的窗口
	imwrite("medImg.jpg", medImg);
	Mat OTSUImg;
	threshold(medImg, OTSUImg, 0, 255, THRESH_OTSU);   //OTSU法
	Mat openImg, closeImg;
	morphologyEx(OTSUImg, openImg, MORPH_OPEN, element);  //开运算
	morphologyEx(openImg, closeImg, MORPH_CLOSE, element);  //闭运算
	imwrite("closeImg.jpg", closeImg);
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
		//if (box[i].size.height / box[i].size.width < 6 && contourArea(contours[i]) < 50000 && contourArea(contours[i]) > 760 && box[i].size.height * box[i].size.width < contourArea(contours[i])*2)
		//{
		//	rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 5, 10, 0);              //画矩形，tl矩形左上角，br右上角
		//	j++;
		//}
		if (contourArea(contours[i]) > 760)
		{
			//rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 5, 10, 0);              //画矩形，tl矩形左上角，br右上角
			j++;
		}

	}
	cout << "细胞总数"<<j << endl;
	/// 显示在一个窗口
	namedWindow("Contours", WINDOW_NORMAL);
	imshow("Contours", srcImg);
	cvResizeWindow("Contours", 800, 800); //创建一个500*500大小的窗口
	imwrite("Contours.jpg", srcImg);
	waitKey(0);
	return 0;
}