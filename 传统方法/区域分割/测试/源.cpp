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
	cvResizeWindow("img", 800, 800); //����һ��800*800��С�Ĵ���

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
	Mat GaussianImg;
	GaussianBlur(HistImg, GaussianImg, Size(5, 5), 1);
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(GaussianImg, GaussianImg, MORPH_OPEN, element);  //������
	//namedWindow("GaussianImg", WINDOW_NORMAL);
	//imshow("GaussianImg", GaussianImg);
	//cvResizeWindow("GaussianImg", 500, 500); //����һ��500*500��С�Ĵ���
	//Sobel����
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
	//cvResizeWindow("Sobel", 500, 500); //����һ��500*500��С�Ĵ���
	int th;
	th = EntropySeg(GaussianImg);
	Mat Growing;
	Mat Growing2(Size(srcImg.cols, srcImg.rows),CV_8UC1,Scalar(0,0,0));
	Point2i pt;
	namedWindow("Growing2", WINDOW_NORMAL);
	cvResizeWindow("Growing2", 500, 500); //����һ��500*500��С�Ĵ���
	setMouseCallback("img", onmouse, &GaussianImg);//ע������¼�����img�����ڣ���ʹ�ڸô����³�������¼���ִ��onmouse����������
//��ʼȡ��ѭ��
	while (1)
	{
		imshow("img", GaussianImg);
		imwrite("img.jpg", GaussianImg);
		while (!mypoint.empty())
		{
			pt = mypoint.back();   //ȡ��һ���������������
			mypoint.pop_back();	//������������
			Growing = RegionGrow(GaussianImg, pt, th-40);
			morphologyEx(Growing, Growing, MORPH_OPEN, element);  //������
			morphologyEx(Growing, Growing, MORPH_CLOSE, element);  //������
			bitwise_or(Growing, Growing2, Growing2);
		}
		imshow("Growing2", Growing2);
		imwrite("Growing2.jpg", Growing2);
		if (waitKey(100) == 27)  //����Esc����
			break;
	}
	// draw result
	vector<vector<Point>> contours;
	findContours(Growing2, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<double> area; //���ڼ���ϸ�����
	vector<vector<Point> > contours_poly(contours.size()); //���ƺ�������㼯
	vector<Rect> boundRect(contours.size()); //��Χ�㼯����С����vector
	vector<RotatedRect> box(contours.size()); //������С��Ӿ��μ���
	Point2f vertices[4];

	for (size_t t = 0; t < contours.size(); t++) {
		approxPolyDP(Mat(contours[t]), contours_poly[t], 3, true);      //�Զ�����������ʵ����ƣ�contours_poly[i]������Ľ��Ƶ㼯
		boundRect[t] = boundingRect(Mat(contours_poly[t]));         //���㲢���ذ�Χ�����㼯����С����
		box[t] = minAreaRect(Mat(contours_poly[t]));  //����ÿ��������С��Ӿ���
		box[t].points(vertices);//��ȡ���ε��ĸ���
		area.push_back(contourArea(contours[t]));
	}
	ofstream outfile;//�������
	outfile.open("1.csv", ios::out | ios::trunc);
	outfile << "��" << "," << "��" << ","  << "���" << endl;
	// ����Χ�ľ��ο�
	double L, W;  //���峤��
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(0, 0, 255);
		rectangle(srcImg, boundRect[i].tl(), boundRect[i].br(), color, 10, 10, 0);              //�����Σ�tl�������Ͻǣ�br���Ͻ�
		//for (int i = 0; i < 4; i++)
		//	line(srcImg, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255));	//����С����
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
		outfile << L << "," << W  << "," << area[i] << endl;//�����С���γ������
	}
	/// ��ʾ��һ������
	namedWindow("Contours", WINDOW_NORMAL);
	imshow("Contours", srcImg);
	cvResizeWindow("Contours", 500, 500); //����һ��500*500��С�Ĵ���
	imwrite("Contours.jpg", srcImg);
	waitKey(0);
	return 0;
}


void onmouse(int event, int x, int y, int flags, void* param)//����¼��ص��������������ִ�е�����Ӧ�ڴ�
{
	Mat img = *(Mat*)param;
	Point cent;
	if (event == EVENT_LBUTTONDOWN)//�����������¼�
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
	int tbHist[256] = { 0 };                                          //ÿ������ֵ����
	int index = 0;                                                  //����ض�Ӧ�ĻҶ�
	double Property = 0.0;                                          //������ռ����
	double maxEntropy = -1.0;                                       //�����
	double frontEntropy = 0.0;                                      //ǰ����
	double backEntropy = 0.0;                                       //������
	//����������������
	int TotalPixel = 0;
	int nCol = src.cols * src.channels();                           //ÿ�е����ظ���
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
		//���㱳��������
		double backTotal = 0;
		for (int j = 0; j < i; j++)
		{
			backTotal += tbHist[j];
		}

		//������
		for (int j = 0; j < i; j++)
		{
			if (tbHist[j] != 0)
			{
				Property = tbHist[j] / backTotal;
				backEntropy += -Property * logf((float)Property);
			}
		}
		//ǰ����
		for (int k = i; k < 256; k++)
		{
			if (tbHist[k] != 0)
			{
				Property = tbHist[k] / (TotalPixel - backTotal);
				frontEntropy += -Property * logf((float)Property);
			}
		}

		if (frontEntropy + backEntropy > maxEntropy)    //�õ������
		{
			maxEntropy = frontEntropy + backEntropy;
			index = i;
		}
		//��ձ��μ�����ֵ
		frontEntropy = 0.0;
		backEntropy = 0.0;
	}
	Mat dst;
	//index += 20;
	return index;
}

Mat RegionGrow(Mat src, Point2i pt, int th)
{
	Point2i ptGrowing;     //��������λ��
	int nGrowLable = 0;		//����Ƿ�������
	int nSrcValue = 0;		//�������Ҷ�ֵ
	int nCurValue = 0;      //��ǰ������Ҷ�ֵ
	Mat matDst = Mat::zeros(src.size(), CV_8UC1); //����һ���հ��������ڱ��ÿһ���Ƿ�������
	//��������˳������
	int DIR[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };
	vector<Point2i> vcGrowPt;			//��ʼ��������ջ
	vcGrowPt.push_back(pt);
	matDst.at<uchar>(pt.y, pt.x) = 255;    //��ǳ�ʼ������
	nSrcValue = src.at<uchar>(pt.y, pt.x);

	while (!vcGrowPt.empty())
	{
		pt = vcGrowPt.back();   //ȡ��һ���������������
		vcGrowPt.pop_back();	//������������
		for (int i = 0; i < 9; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];    //����������˳���������
			ptGrowing.y = pt.y + DIR[i][1];
			if (ptGrowing.x<0 || ptGrowing.y<0 || ptGrowing.x>(src.cols - 1) || ptGrowing.y>(src.rows - 1))  //�Ƿ񵽴��Ե��
				continue;
			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);//��ǰ��������ĻҶ�ֵ
			if (nGrowLable == 0)						//���û������
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				if (abs(nSrcValue - nCurValue) < th)			//���������������ֵ��
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;
					vcGrowPt.push_back(ptGrowing);			//���õ���Ϊ����������´�ѭ��
				}
			}
		}
	}
	return matDst.clone();
}
