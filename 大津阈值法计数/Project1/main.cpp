#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <iterator>
#include <algorithm>
using namespace std;
using namespace cv;


//计数
int main(int argc, char* argv[])
{
	Mat srcImg = imread("1_predict.png");
	namedWindow("src", WINDOW_NORMAL);
	imshow("src", srcImg);
	cvResizeWindow("src", 800, 800); //创建一个500*500大小的窗口
	Mat src_gray;
	cvtColor(srcImg, src_gray, CV_RGB2GRAY);//灰度转换
	Mat OTSUImg;
	threshold(src_gray, OTSUImg, 0, 255, THRESH_OTSU);   //OTSU法
	// 连通区域计数
	vector<vector<Point>> contours;
	findContours(OTSUImg, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point> > contours_poly(contours.size()); //近似后的轮廓点集
	vector<Rect> boundRect(contours.size()); //包围点集的最小矩形vector
	vector<RotatedRect> box(contours.size()); //定义最小外接矩形集合
	vector<int> hull;   //定义凸包
	int k = 0;
	int a, b;
	Mat dst(srcImg.cols, srcImg.rows, CV_8UC(1), Scalar::all(0));
	for (size_t t = 0; t < contours.size(); t++) {
		approxPolyDP(Mat(contours[t]), contours_poly[t], 3, true);      //对多边形曲线做适当近似，contours_poly[i]是输出的近似点集
		convexHull(contours_poly[t], hull, true);     //计算凸包
		a = contours_poly[t].size();
		b = hull.size();
		//if (b/a > 0.7 &&a > 2)
		if (b / a > 0.7&&a > 0.5)
		{
			k += 1;
			drawContours(dst, contours, t, (255, 255, 255), -1, 8);
		}
		boundRect[t] = boundingRect(Mat(contours_poly[t]));         //计算并返回包围轮廓点集的最小矩形
		box[t] = minAreaRect(Mat(contours_poly[t]));  //计算每个轮廓最小外接矩形
	}
	cout << "细胞总数" << k << endl;
	// 画包围的矩形框
	//int j = 0;
	//Scalar color = Scalar(0, 0, 255);
	//for (int i = 0; i< contours.size(); i++)
	//{
	//	if (box[i].size.height / box[i].size.width < 6)
	//	{
	//		rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 5, 10, 0);              //画矩形，tl矩形左上角，br右上角
	//		j++;
	//	}
		//if (contourArea(contours[i]) > 760)
		//{
		//	rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 5, 10, 0);              //画矩形，tl矩形左上角，br右上角
		//	j++;
		//}

		//}
	//cout << "细胞总数" << j << endl;
	/// 显示在一个窗口
	namedWindow("Contours", WINDOW_NORMAL);
	imshow("Contours", dst);
	cvResizeWindow("Contours", 800, 800); //创建一个500*500大小的窗口
	imwrite("Contours.jpg", dst);
	waitKey(0);
	return 0;
}