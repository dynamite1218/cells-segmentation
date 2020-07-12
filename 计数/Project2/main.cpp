#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <iterator>
#include <algorithm>
using namespace std;
using namespace cv;


//ϸ��ʶ��������

int main(int argc, char* argv[])
{
	Mat srcImg = imread("1.jpg");
	namedWindow("src", WINDOW_NORMAL);
	imshow("src", srcImg);
	cvResizeWindow("src", 800, 800); //����һ��500*500��С�Ĵ���
	if (!srcImg.data)
	{
		cout << "ͼƬδ�ҵ�" << endl;
		return -1;
	}
	/*ȥ����������*/
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
	cvtColor(dstImg, src_gray, CV_RGB2GRAY);//�Ҷ�ת��
	Mat HistImg;
	equalizeHist(src_gray, HistImg);
	Mat element = getStructuringElement(MORPH_RECT, Size(12, 12));
	//dilate(srcImg, dstImg, element, Point(-1, -1), 3);
	//erode(srcImg, dstImg, element, Point(-1, -1), 5);
	Mat medImg;
	medianBlur(HistImg, medImg, 9);
	Mat grad_x, grad_y;						  //Sobel������
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
	cvResizeWindow("medImg", 800, 800); //����һ��500*500��С�Ĵ���
	imwrite("medImg.jpg", medImg);
	Mat OTSUImg;
	threshold(medImg, OTSUImg, 0, 255, THRESH_OTSU);   //OTSU��
	Mat openImg, closeImg;
	morphologyEx(OTSUImg, openImg, MORPH_OPEN, element);  //������
	morphologyEx(openImg, closeImg, MORPH_CLOSE, element);  //������
	imwrite("closeImg.jpg", closeImg);
	// ��ͨ�������
	vector<vector<Point>> contours;
	findContours(closeImg, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point> > contours_poly(contours.size()); //���ƺ�������㼯
	vector<Rect> boundRect(contours.size()); //��Χ�㼯����С����vector
	vector<RotatedRect> box(contours.size()); //������С��Ӿ��μ���
	//Mat markers = Mat::zeros(srcImg.size(), CV_8UC3);
	for (size_t t = 0; t < contours.size(); t++) {
		approxPolyDP(Mat(contours[t]), contours_poly[t], 3, true);      //�Զ�����������ʵ����ƣ�contours_poly[i]������Ľ��Ƶ㼯
		boundRect[t] = boundingRect(Mat(contours_poly[t]));         //���㲢���ذ�Χ�����㼯����С����
		box[t] = minAreaRect(Mat(contours_poly[t]));  //����ÿ��������С��Ӿ���
	}
	// ����Χ�ľ��ο�
	int j = 0;
	Scalar color = Scalar(0, 0, 255);
	for (int i = 0; i< contours.size(); i++)
	{
		//if (box[i].size.height / box[i].size.width < 6 && contourArea(contours[i]) < 50000 && contourArea(contours[i]) > 760 && box[i].size.height * box[i].size.width < contourArea(contours[i])*2)
		//{
		//	rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 5, 10, 0);              //�����Σ�tl�������Ͻǣ�br���Ͻ�
		//	j++;
		//}
		if (contourArea(contours[i]) > 760)
		{
			//rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 5, 10, 0);              //�����Σ�tl�������Ͻǣ�br���Ͻ�
			j++;
		}

	}
	cout << "ϸ������"<<j << endl;
	/// ��ʾ��һ������
	namedWindow("Contours", WINDOW_NORMAL);
	imshow("Contours", srcImg);
	cvResizeWindow("Contours", 800, 800); //����һ��500*500��С�Ĵ���
	imwrite("Contours.jpg", srcImg);
	waitKey(0);
	return 0;
}