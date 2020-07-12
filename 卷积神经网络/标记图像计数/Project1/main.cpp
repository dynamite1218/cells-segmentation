#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <iterator>
#include <algorithm>
using namespace std;
using namespace cv;


//����
int main(int argc, char* argv[])
{
	Mat srcImg = imread("1_predict.png");
	namedWindow("src", WINDOW_NORMAL);
	imshow("src", srcImg);
	cvResizeWindow("src", 800, 800); //����һ��500*500��С�Ĵ���
	Mat src_gray;
	cvtColor(srcImg, src_gray, CV_RGB2GRAY);//�Ҷ�ת��
	Mat OTSUImg;
	threshold(src_gray, OTSUImg, 0, 255, THRESH_OTSU);   //OTSU��
	// ��ͨ�������
	vector<vector<Point>> contours;
	findContours(OTSUImg, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point> > contours_poly(contours.size()); //���ƺ�������㼯
	vector<Rect> boundRect(contours.size()); //��Χ�㼯����С����vector
	vector<RotatedRect> box(contours.size()); //������С��Ӿ��μ���
	vector<int> hull;   //����͹��
	int k = 0;
	int a, b;
	Mat dst(srcImg.cols, srcImg.rows, CV_8UC(1), Scalar::all(0));
	for (size_t t = 0; t < contours.size(); t++) {
		approxPolyDP(Mat(contours[t]), contours_poly[t], 3, true);      //�Զ�����������ʵ����ƣ�contours_poly[i]������Ľ��Ƶ㼯
		convexHull(contours_poly[t], hull, true);     //����͹��
		a = contours_poly[t].size();
		b = hull.size();
		//if (b/a > 0.7 &&a > 2)
		if (b / a > 0.7&&a > 0.5)
		{
			k += 1;
			drawContours(dst, contours, t, (255, 255, 255), -1, 8);
		}
		boundRect[t] = boundingRect(Mat(contours_poly[t]));         //���㲢���ذ�Χ�����㼯����С����
		box[t] = minAreaRect(Mat(contours_poly[t]));  //����ÿ��������С��Ӿ���
	}
	cout << "ϸ������" << k << endl;
	// ����Χ�ľ��ο�
	//int j = 0;
	//Scalar color = Scalar(0, 0, 255);
	//for (int i = 0; i< contours.size(); i++)
	//{
	//	if (box[i].size.height / box[i].size.width < 6)
	//	{
	//		rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 5, 10, 0);              //�����Σ�tl�������Ͻǣ�br���Ͻ�
	//		j++;
	//	}
		//if (contourArea(contours[i]) > 760)
		//{
		//	rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 5, 10, 0);              //�����Σ�tl�������Ͻǣ�br���Ͻ�
		//	j++;
		//}

		//}
	//cout << "ϸ������" << j << endl;
	/// ��ʾ��һ������
	namedWindow("Contours", WINDOW_NORMAL);
	imshow("Contours", dst);
	cvResizeWindow("Contours", 800, 800); //����һ��500*500��С�Ĵ���
	imwrite("Contours.jpg", dst);
	waitKey(0);
	return 0;
}