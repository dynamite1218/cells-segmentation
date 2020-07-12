#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <iterator>
#include <algorithm>
using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	Mat srcImg = imread("6.jpg");
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
	Rect m_select;
	m_select = Rect(0, 0, 2000, 2000);
	dstImg = dstImg(m_select);     //��ȡ����ͼ��
	Mat src_gray;
	cvtColor(dstImg, src_gray, CV_RGB2GRAY);//�Ҷ�ת��
	Mat HistImg;
	equalizeHist(src_gray, HistImg);
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	//dilate(srcImg, dstImg, element, Point(-1, -1), 3);
	//erode(srcImg, dstImg, element, Point(-1, -1), 5);
	Mat GaussianImg;
	GaussianBlur(HistImg, GaussianImg, Size(5, 5), 1);   //��˹�˲�
	imshow("GaussianImg", GaussianImg);
	imwrite("GaussianImg.jpg", GaussianImg);
	//robert����
	int rows = GaussianImg.rows;
	int cols = GaussianImg.cols;
	Mat newImg = GaussianImg.clone();
	for (int i = 0; i < rows - 1; i++)
	{
		for (int j = 0; j < cols - 1; j++)
		{
			newImg.at<uchar>(i, j) = abs(GaussianImg.at<uchar>(i + 1, j + 1) - GaussianImg.at<uchar>(i, j)) + abs(GaussianImg.at<uchar>(i + 1, j) - GaussianImg.at<uchar>(i, j + 1));
		}
	}
	namedWindow("ԭͼ", WINDOW_NORMAL);
	cvResizeWindow("ԭͼ", 700, 700); //����һ��500*500��С�Ĵ���
	imshow("ԭͼ", dstImg);
	imwrite("img.jpg", dstImg);
	namedWindow("robert", WINDOW_NORMAL);
	cvResizeWindow("robert", 700, 700); //����һ��500*500��С�Ĵ���
	imshow("robert", newImg);
	imwrite("robert.jpg", newImg);

//prewitt
	Mat src, gray, Kernelx, Kernely;
	Kernelx = (Mat_<double>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
	Kernely = (Mat_<double>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y, grad;
	filter2D(GaussianImg, grad_x, CV_16S, Kernelx, Point(-1, -1));
	filter2D(GaussianImg, grad_y, CV_16S, Kernely, Point(-1, -1));
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	namedWindow("prewitt", WINDOW_NORMAL);
	cvResizeWindow("prewitt", 700, 700); //����һ��500*500��С�Ĵ���
	imshow("prewitt",grad);
	imwrite("prewitt.jpg", grad);
//Canny����
	Mat CannyImg;
	Canny(GaussianImg, CannyImg, 30, 150, 3);  
	namedWindow("Canny", WINDOW_NORMAL);
	cvResizeWindow("Canny", 700, 700); //����һ��500*500��С�Ĵ���
	imshow("Canny", CannyImg);
	imwrite("Canny.jpg", CannyImg);
//Sobel����
	Mat SobelImg;
	//Mat grad_x, grad_y;
	//Mat abs_grad_x, abs_grad_y;
	Sobel(GaussianImg, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	Sobel(GaussianImg, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, SobelImg);
	namedWindow("Sobel", WINDOW_NORMAL);
	cvResizeWindow("Sobel", 700, 700); //����һ��500*500��С�Ĵ���
	imshow("Sobel", SobelImg);
	imwrite("Sobel.jpg", SobelImg);

//LoG����
	int kernel_size = 3;
	Mat LoGImg, abs_dst;
	Laplacian(GaussianImg, LoGImg, CV_16S, kernel_size);	//ͨ��������˹��������Ե���
	convertScaleAbs(LoGImg, abs_dst);
	namedWindow("LoG", WINDOW_NORMAL);
	cvResizeWindow("LoG", 700, 700); //����һ��500*500��С�Ĵ���
	imshow("LoG", abs_dst);
	imwrite("LoG.jpg", abs_dst);
	waitKey(0);
	return 0;
}